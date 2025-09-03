#!/usr/bin/env python3
"""
athena_vtk_to_h5.py
===================

Read two Athena VTK dumps:
  - Hydra ('w')  : ms02ma20_256.mhd_w.00058.vtk  → density
  - Magnetic ('bcc'): ms02ma20_256.mhd_bcc.00058.vtk → B-vector

and write an HDF5 with datasets:
  gas_density : n_e(x,y,z)
  k_mag_field : B_z(x,y,z)
  x_coor, y_coor, z_coor : broadcasted coordinates

The reader prefers 'meshio' if installed; otherwise falls back to a minimal
legacy-ASCII VTK parser for STRUCTURED_* grids. Data are assumed POINT_DATA.
"""

from __future__ import annotations
import os
import io
import sys
import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths if needed
# ──────────────────────────────────────────────────────────────────────
VTK_W_PATH   = "../ms02ma20_256.mhd_w.00117.vtk"    # density etc.
VTK_BCC_PATH = "../ms02ma20_256.mhd_bcc.00117.vtk"  # magnetic field
OUT_H5       = "athena_ms02ma20_256_00117.h5"

# field name guesses in the VTK files (case-insensitive search)
DENSITY_KEYS = ("rho", "density", "n_e", "ne")
B_VECTOR_KEYS= ("B", "Bcc", "magnetic_field", "b")

# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def _median_spacing(vals: np.ndarray) -> float:
    dif = np.diff(np.unique(np.asarray(vals, dtype=np.float64)))
    dif = dif[dif > 0]
    return float(np.median(dif)) if dif.size else 1.0

def _build_xyz_from_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int,int,int]]:
    """Given (N,3) points on a rectilinear structured grid, recover (nx,ny,nz) and 3D broadcasted X,Y,Z arrays."""
    xvals = np.unique(points[:,0])
    yvals = np.unique(points[:,1])
    zvals = np.unique(points[:,2])
    nx, ny, nz = len(xvals), len(yvals), len(zvals)

    # broadcast to shape (nx,ny,nz) as in your HDF5 convention
    X = np.broadcast_to(xvals[:,None,None], (nx,ny,nz)).astype(np.float32)
    Y = np.broadcast_to(yvals[None,:,None], (nx,ny,nz)).astype(np.float32)
    Z = np.broadcast_to(zvals[None,None,:], (nx,ny,nz)).astype(np.float32)
    return X, Y, Z, (nx, ny, nz)

def _sort_index_zyx(points: np.ndarray) -> np.ndarray:
    """
    Lexicographic order by (z, y, x) so that after reshaping to (nz,ny,nx)
    the 'x' index runs fastest. This matches a natural C-order reshape.
    """
    # np.lexsort uses LAST key as primary
    return np.lexsort((points[:,0], points[:,1], points[:,2]))

def _reshape_point_scalar(points: np.ndarray, scal: np.ndarray, shape: Tuple[int,int,int]) -> np.ndarray:
    """Sort points by (z,y,x), reshape to (nz,ny,nx) then transpose → (nx,ny,nz)."""
    nx, ny, nz = shape
    idx = _sort_index_zyx(points)
    s = np.asarray(scal, dtype=np.float64).reshape(-1)[idx]
    s_zyx = s.reshape(nz, ny, nx)
    return np.transpose(s_zyx, (2,1,0)).astype(np.float32)  # (nx,ny,nz)

def _reshape_point_vector(points: np.ndarray, vec: np.ndarray, shape: Tuple[int,int,int]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """As above, but for a (N,3) vector; returns (Bx,By,Bz) as (nx,ny,nz)."""
    nx, ny, nz = shape
    idx = _sort_index_zyx(points)
    v = np.asarray(vec, dtype=np.float64)[idx,:]
    vx_zyx = v[:,0].reshape(nz,ny,nx)
    vy_zyx = v[:,1].reshape(nz,ny,nx)
    vz_zyx = v[:,2].reshape(nz,ny,nx)
    Bx = np.transpose(vx_zyx, (2,1,0)).astype(np.float32)
    By = np.transpose(vy_zyx, (2,1,0)).astype(np.float32)
    Bz = np.transpose(vz_zyx, (2,1,0)).astype(np.float32)
    return Bx, By, Bz

def _match_key(d: Dict[str,np.ndarray], options: Tuple[str,...]) -> Optional[str]:
    lower = {k.lower(): k for k in d.keys()}
    for want in options:
        if want.lower() in lower:
            return lower[want.lower()]
    # partial contains?
    for k in d.keys():
        if any(w in k.lower() for w in options):
            return k
    return None

# ──────────────────────────────────────────────────────────────────────
# Reader route A: meshio (preferred)
# ──────────────────────────────────────────────────────────────────────
def read_with_meshio(vtk_path: str):
    try:
        import meshio
    except Exception:
        return None
    m = meshio.read(vtk_path)

    # Prefer point_data
    if not m.point_data:
        # We could attempt cell_data → points interpolation, but avoid silent misuse
        raise RuntimeError(f"{vtk_path}: no point_data. Re-export as POINT_DATA or install a pointization step.")

    points = np.asarray(m.points, dtype=np.float64)  # (N,3)

    # Ensure structured rectilinear grid
    X, Y, Z, shape = _build_xyz_from_points(points)

    # Prepare data dict mapping lower-name -> array (keeping original names too)
    pdata = {k: np.asarray(v) for k,v in m.point_data.items()}
    return dict(points=points, shape=shape, X=X, Y=Y, Z=Z, point_data=pdata)

# ──────────────────────────────────────────────────────────────────────
# Reader route B: minimal legacy-ASCII VTK parser (STRUCTURED_* only)
# ──────────────────────────────────────────────────────────────────────
def read_legacy_ascii_structured_points(vtk_path: str):
    """
    Very small parser for VTK legacy ASCII 'STRUCTURED_POINTS' or 'STRUCTURED_GRID'
    emitting points (N,3) and a dict of point_data arrays.
    """
    with open(vtk_path, "rt", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    s = io.StringIO(text)
    header = [s.readline() for _ in range(5)]
    fmt_line = header[2].strip().upper()
    if "ASCII" not in fmt_line:
        raise RuntimeError("Fallback parser handles only ASCII legacy VTK.")
    grid_type = header[3].strip().upper()
    dims = None; origin = (0.0,0.0,0.0); spacing = (1.0,1.0,1.0)

    # Scan tokens
    points = None
    point_data_count = None
    arrays: Dict[str,np.ndarray] = {}

    for line in s:
        t = line.strip()
        if not t:
            continue
        U = t.upper()

        if U.startswith("DIMENSIONS"):
            _, nx, ny, nz = t.split()
            dims = (int(nx), int(ny), int(nz))
        elif U.startswith("ORIGIN"):
            _, x0, y0, z0 = t.split()
            origin = (float(x0), float(y0), float(z0))
        elif U.startswith("SPACING"):
            _, dx, dy, dz = t.split()
            spacing = (float(dx), float(dy), float(dz))
        elif U.startswith("POINT_DATA"):
            _, n = t.split()
            point_data_count = int(n)
        elif U.startswith("SCALARS"):
            # SCALARS name type [numComp]
            parts = t.split()
            name = parts[1]
            # Expect next line LOOKUP_TABLE
            lt = s.readline()
            vals = []
            # read until we have 'point_data_count' numbers
            while len(vals) < (point_data_count or 0):
                vals.extend(s.readline().strip().split())
            arr = np.array(vals[:point_data_count], dtype=float)
            arrays[name] = arr
        elif U.startswith("VECTORS"):
            # VECTORS name type
            parts = t.split()
            name = parts[1]
            vals = []
            need = 3 * (point_data_count or 0)
            while len(vals) < need:
                vals.extend(s.readline().strip().split())
            arr = np.array(vals[:need], dtype=float).reshape(-1, 3)
            arrays[name] = arr

    if dims is None or point_data_count is None:
        raise RuntimeError("Could not find DIMENSIONS/POINT_DATA in VTK.")

    nx, ny, nz = dims
    # Build points in i-fastest order (x-fastest), then j, then k
    x = origin[0] + spacing[0] * np.arange(nx)
    y = origin[1] + spacing[1] * np.arange(ny)
    z = origin[2] + spacing[2] * np.arange(nz)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")  # (nx,ny,nz)
    points = np.column_stack([Xg.ravel(order="F"), Yg.ravel(order="F"), Zg.ravel(order="F")])  # F to emulate i-fastest

    # Some writers order points as x fastest; our later sort-by-(z,y,x) will fix it.
    pdata = arrays
    X, Y, Z, shape = _build_xyz_from_points(points)
    return dict(points=points, shape=shape, X=X, Y=Y, Z=Z, point_data=pdata)

def read_vtk_any(vtk_path: str):
    """Try meshio; if not present, try minimal ASCII legacy parser."""
    try:
        info = read_with_meshio(vtk_path)
        if info is not None:
            return info
    except Exception as e:
        print(f"[meshio] {vtk_path}: {e}")

    # fallback
    info = read_legacy_ascii_structured_points(vtk_path)
    return info

# ──────────────────────────────────────────────────────────────────────
# Extraction of density and Bz, reshape to (nx,ny,nz)
# ──────────────────────────────────────────────────────────────────────
def extract_density(info: Dict, wanted=DENSITY_KEYS) -> np.ndarray:
    pdata = info["point_data"]
    k = _match_key(pdata, wanted)
    if k is None:
        raise RuntimeError(f"Could not find a density key among {list(pdata.keys())}")
    arr = pdata[k]
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:,0]
    if arr.ndim != 1 or arr.size != info["points"].shape[0]:
        raise RuntimeError("Density appears to be cell-centered or has unexpected shape.")
    return _reshape_point_scalar(info["points"], arr, info["shape"])

def extract_Bz(info: Dict, wanted=B_VECTOR_KEYS) -> np.ndarray:
    pdata = info["point_data"]
    k = _match_key(pdata, wanted)
    if k is None:
        # Some outputs may store components as separate scalars Bx,By,Bz
        # Try to reconstruct from scalars:
        bxk = _match_key(pdata, ("Bx","B_x","Bx1","B1"))
        byk = _match_key(pdata, ("By","B_y","Bx2","B2"))
        bzk = _match_key(pdata, ("Bz","B_z","Bx3","B3","Bz1"))
        if bzk is not None:
            arr = pdata[bzk]
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:,0]
            if arr.ndim != 1 or arr.size != info["points"].shape[0]:
                raise RuntimeError("Bz found but shape unexpected (cell-centered?).")
            return _reshape_point_scalar(info["points"], arr, info["shape"])
        raise RuntimeError(f"Could not find magnetic vector among keys {list(pdata.keys())}")
    arr = pdata[k]
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != info["points"].shape[0]:
        raise RuntimeError("Magnetic field looks cell-centered or not a 3-component point vector.")
    _, _, Bz = _reshape_point_vector(info["points"], arr, info["shape"])
    return Bz

# ──────────────────────────────────────────────────────────────────────
# HDF5 writer
# ──────────────────────────────────────────────────────────────────────
def write_h5(out_path: str, ne: np.ndarray, bz: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("gas_density", data=ne.astype(np.float32), compression="gzip")
        h5.create_dataset("k_mag_field", data=bz.astype(np.float32), compression="gzip")
        h5.create_dataset("x_coor", data=X.astype(np.float32), compression="gzip")
        h5.create_dataset("y_coor", data=Y.astype(np.float32), compression="gzip")
        h5.create_dataset("z_coor", data=Z.astype(np.float32), compression="gzip")

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    # Read hydro (density)
    info_w = read_vtk_any(VTK_W_PATH)
    print(f"[read] {VTK_W_PATH}: points={info_w['points'].shape[0]} shape={info_w['shape']}")
    ne = extract_density(info_w)
    X, Y, Z = info_w["X"], info_w["Y"], info_w["Z"]

    # Read magnetic (B)
    info_b = read_vtk_any(VTK_BCC_PATH)
    print(f"[read] {VTK_BCC_PATH}: points={info_b['points'].shape[0]} shape={info_b['shape']}")
    Bz = extract_Bz(info_b)

    # Sanity: shapes must match
    if ne.shape != Bz.shape:
        raise RuntimeError(f"Shape mismatch: density {ne.shape} vs Bz {Bz.shape}")

    # Save
    write_h5(OUT_H5, ne, Bz, X, Y, Z)
    print(f"[ok] wrote {OUT_H5}")
    # Quick stats
    ne64 = ne.astype(np.float64); bz64 = Bz.astype(np.float64)
    print(f"  gas_density: mean={ne64.mean():.6g}  min={ne64.min():.6g}  max={ne64.max():.6g}  rms={np.sqrt((ne64*ne64).mean()):.6g}")
    print(f"  k_mag_field: mean={bz64.mean():.6g}  min={bz64.min():.6g}  max={bz64.max():.6g}  rms={np.sqrt((bz64*bz64).mean()):.6g}")

if __name__ == "__main__":
    main()
