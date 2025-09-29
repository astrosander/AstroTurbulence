#!/usr/bin/env python3
"""
vtk_to_h5_mhd_no_args.py

Reads two VTK ImageData files (density & magnetic field) and writes HDF5 with
labels matching your old pipeline:
  - gas_density      (from 'dens')
  - k_mag_field      (alias of Bz, from 'bcc3')
  - k_mag_field_x/y/z (from 'bcc1', 'bcc2', 'bcc3')
  - x_coor, y_coor, z_coor (3D coordinate arrays)
"""

from pathlib import Path
import numpy as np
import h5py
import pyvista as pv

# ── EDIT THESE PATHS ───────────────────────────────────────────────────
VTK_W_PATH   = Path("../ms1ma2_256.mhd_w.00013.vtk")    # contains 'dens'
VTK_BCC_PATH = Path("../ms1ma2_256.mhd_bcc.00013.vtk")  # contains 'bcc1/2/3'
OUT_H5       = Path("../faradays_angles_stats/lp_structure_tests/mhd_fields.h5")
WRITE_3D_COORDS = True   # set False to write 1-D x/y/z instead of 3-D X/Y/Z
# ───────────────────────────────────────────────────────────────────────

def _get_structured_array(mesh: pv.ImageData, name: str) -> np.ndarray:
    """
    Return a (nx, ny, nz) array for a scalar stored on VTK ImageData.
    Works for point or cell data; prefers point_data if both exist.
    """
    nx, ny, nz = mesh.dimensions
    if name in mesh.point_data:
        arr = np.asarray(mesh.point_data[name]).reshape((nx, ny, nz), order="F")
        return arr
    if name in mesh.cell_data:
        # Cell-centered: reshape to (nx-1, ny-1, nz-1) and pad to (nx, ny, nz)
        carr = np.asarray(mesh.cell_data[name]).reshape((nx-1, ny-1, nz-1), order="F")
        return _cell_to_point_like(carr)
    raise KeyError(f"Array '{name}' not found in point_data or cell_data.")

def _cell_to_point_like(cell_arr: np.ndarray) -> np.ndarray:
    """
    Convert (nx-1, ny-1, nz-1) cell-centered values to (nx, ny, nz) point-like
    via nearest-neighbor padding on +faces.
    """
    cx, cy, cz = cell_arr.shape
    out = np.zeros((cx+1, cy+1, cz+1), dtype=cell_arr.dtype)
    out[:cx, :cy, :cz] = cell_arr
    out[cx, :cy, :cz]  = cell_arr[cx-1, :, :]
    out[:cx, cy, :cz]  = cell_arr[:, cy-1, :]
    out[:cx, :cy, cz]  = cell_arr[:, :, cz-1]
    out[cx, cy, :cz]   = cell_arr[cx-1, cy-1, :]
    out[cx, :cy, cz]   = cell_arr[cx-1, :, cz-1]
    out[:cx, cy, cz]   = cell_arr[:, cy-1, cz-1]
    out[cx, cy, cz]    = cell_arr[cx-1, cy-1, cz-1]
    return out

def _coords_from_imagedata(mesh: pv.ImageData, make_3d: bool = True):
    """
    Coordinates from ImageData origin/spacing/dimensions.
    """
    nx, ny, nz = mesh.dimensions
    dx, dy, dz = mesh.spacing
    ox, oy, oz = mesh.origin

    x = ox + np.arange(nx) * dx
    y = oy + np.arange(ny) * dy
    z = oz + np.arange(nz) * dz

    if not make_3d:
        return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)

    X = np.broadcast_to(x[:, None, None], (nx, ny, nz)).astype(np.float32)
    Y = np.broadcast_to(y[None, :, None], (nx, ny, nz)).astype(np.float32)
    Z = np.broadcast_to(z[None, None, :], (nx, ny, nz)).astype(np.float32)
    return X, Y, Z

def run(vtk_w_path: Path, vtk_bcc_path: Path, out_path: Path, write_3d_coords: bool = True):
    # Read
    w_mesh   = pv.read(str(vtk_w_path))
    bcc_mesh = pv.read(str(vtk_bcc_path))

    print("Density file arrays (tags):", w_mesh.array_names)
    print("Magnetic field file arrays (tags):", bcc_mesh.array_names)

    # Extract fields
    dens = _get_structured_array(w_mesh,   "dens").astype(np.float32)
    bx   = _get_structured_array(bcc_mesh, "bcc1").astype(np.float32)
    by   = _get_structured_array(bcc_mesh, "bcc2").astype(np.float32)
    bz   = _get_structured_array(bcc_mesh, "bcc3").astype(np.float32)
    vx = _get_structured_array(w_mesh, "velx").astype(np.float32)
    vy = _get_structured_array(w_mesh, "vely").astype(np.float32)
    vz = _get_structured_array(w_mesh, "velz").astype(np.float32)

    # Coordinates
    coords = _coords_from_imagedata(w_mesh, make_3d=write_3d_coords)

    print(f"• writing {out_path} …")
    with h5py.File(out_path, "w") as h5:
        # Same labels as your previous H5 conventions
        h5.create_dataset("gas_density",   data=dens, compression="gzip")
        h5.create_dataset("k_mag_field",   data=bz,   compression="gzip")  # alias of Bz
        h5.create_dataset("k_mag_field_x", data=bx,   compression="gzip")
        h5.create_dataset("k_mag_field_y", data=by,   compression="gzip")
        h5.create_dataset("k_mag_field_z", data=bz,   compression="gzip")
        h5.create_dataset("velx", data=vx, compression="gzip")
        h5.create_dataset("vely", data=vy, compression="gzip")
        h5.create_dataset("velz", data=vz, compression="gzip")
# optional aliases for B to keep old readers happy
        h5.create_dataset("bcc1", data=bx, compression="gzip")
        h5.create_dataset("bcc2", data=by, compression="gzip")
        h5.create_dataset("bcc3", data=bz, compression="gzip")

        if write_3d_coords:
            X, Y, Z = coords
            h5.create_dataset("x_coor", data=X, compression="gzip")
            h5.create_dataset("y_coor", data=Y, compression="gzip")
            h5.create_dataset("z_coor", data=Z, compression="gzip")
        else:
            x, y, z = coords
            h5.create_dataset("x_coor", data=x, compression="gzip")
            h5.create_dataset("y_coor", data=y, compression="gzip")
            h5.create_dataset("z_coor", data=z, compression="gzip")

        # Minimal metadata
        g = h5.create_group("metadata")
        g.attrs["source_vtk_w"]   = str(vtk_w_path)
        g.attrs["source_vtk_bcc"] = str(vtk_bcc_path)
        g.attrs["vtk_dims"]       = w_mesh.dimensions
        g.attrs["vtk_spacing"]    = w_mesh.spacing
        g.attrs["vtk_origin"]     = w_mesh.origin

    print("done.")

if __name__ == "__main__":
    run(VTK_W_PATH, VTK_BCC_PATH, OUT_H5, write_3d_coords=WRITE_3D_COORDS)
