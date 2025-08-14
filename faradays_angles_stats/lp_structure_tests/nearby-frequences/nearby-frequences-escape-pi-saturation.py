#!/usr/bin/env python3
"""
Athena HDF5 comparison: single-frequency vs. two-close-frequency (Δχ) projection

What this script does
---------------------
1) Load an Athena MHD cube (default: 'ms01ma08.mhd_w.00300.vtk.h5').
2) Read electron density (gas_density) and line-of-sight magnetic field (B_parallel).
   - By default we use the z-component as LOS, but logic below will try to find a z-component
     even if the magnetic field is stored as a vector (e.g., [3, Nx, Ny, Nz] or [Nx,Ny,Nz,3]).
3) Build an RM(x,z) map by integrating n_e * B_parallel along the y-axis (LOS).
4) Compute polarization angles at two close frequencies and evaluate spatial statistics:
   - Nearest-neighbor mean squared differences of wrapped angles (single-frequency).
   - Same for Δχ = wrap_pi(χ2 - χ1) (two-close-frequency method).
   - Show saturation behavior vs. an overall amplitude scale A.
5) Produce three plots and print diagnostics.

Notes
-----
- Units: if your cube is not in (cm^-3, μG, pc) units, the overall conversion to rad m^-2 is unknown.
  We therefore scan over an amplitude factor A to expose the saturation phenomenon independent of units.
- Angle wrapping: polarization angles are orientations modulo π → we wrap to (-π/2, π/2].
- Two spatial-difference conventions are shown:
    (i) wrap-after-diff (circular distance) → saturation π^2/12
    (ii) plain difference (no wrap after differencing) → saturation π^2/6
- If your LOS is not 'y', change LOS_AXIS = 1 below.

Author: (you)
"""

# import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration (defaults)
# -----------------------------
CUBE_PATH_DEFAULT = "../ms01ma08.mhd_w.00300.vtk.h5"
NE_KEY_DEFAULT = "gas_density"
BZ_KEY_DEFAULT = "k_mag_field"   # flexible: can be scalar Bz or a vector container; see extractor below

# Choose LOS axis: 0 → x, 1 → y, 2 → z
LOS_AXIS = 1   # integrate along y by default, RM(x,z)

# Frequencies (Hz). We'll use two nearby frequencies by default.
NU1_DEFAULT = 1.400e9
DNU_DEFAULT = 5.0e6   # ν2 = ν1 + Δν
# Amplitude scan (dimensionless factor A multiplying RM)
A_MIN, A_MAX, A_N = 1.0, 1.0e5, 25

# -----------------------------
# Utilities
# -----------------------------
def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap orientation angle(s) to (-π/2, π/2]."""
    return (x + np.pi/2) % np.pi - np.pi/2

def neighbor_msd_wrap_after_diff(phi: np.ndarray) -> float:
    """
    Mean squared *wrapped* nearest-neighbor difference for an orientation field phi (mod π).
    Saturation for random orientations is π^2/12.
    """
    dx = wrap_pi(phi[:, 1:] - phi[:, :-1])
    dy = wrap_pi(phi[1:, :] - phi[:-1, :])
    return np.mean(np.concatenate([dx.ravel(), dy.ravel()])**2)

def neighbor_msd_no_wrap_after_diff(phi: np.ndarray) -> float:
    """
    Mean squared nearest-neighbor difference without wrapping *after* differencing.
    Saturation for random orientations is π^2/6.
    """
    dx = phi[:, 1:] - phi[:, :-1]
    dy = phi[1:, :] - phi[:-1, :]
    return np.mean(np.concatenate([dx.ravel(), dy.ravel()])**2)

def axis_spacing(coord_line: np.ndarray, name: str) -> float:
    """
    Estimate mean grid spacing along a 1D coordinate line.
    Falls back to 1.0 if the array is invalid or too short.
    """
    try:
        coord_line = np.asarray(coord_line).astype(float)
        if coord_line.size >= 2:
            diffs = np.diff(coord_line)
            # Prefer median to be robust to eccentricities
            return float(np.median(diffs))
        else:
            return 1.0
    except Exception:
        print(f"[warn] Could not compute spacing for {name}; defaulting to 1.0", file=sys.stderr)
        return 1.0

def _find_dataset_recursive(h5: h5py.Group, name_substr: str):
    """
    Search for a dataset whose name contains 'name_substr' (case-insensitive).
    Returns the (group, key) pair or (None, None) if not found.
    """
    target = name_substr.lower()
    for key, obj in h5.items():
        if isinstance(obj, h5py.Dataset):
            if target in key.lower():
                return (h5, key)
        elif isinstance(obj, h5py.Group):
            g, k = _find_dataset_recursive(obj, name_substr)
            if g is not None:
                return (g, k)
    return (None, None)

def _extract_bz_from_dataset(dset: h5py.Dataset) -> np.ndarray:
    """
    Extract a z-component from an HDF5 dataset that might be:
      - already scalar Bz with shape (Nx,Ny,Nz)
      - vector with component axis first: (3, Nx, Ny, Nz)
      - vector with component axis last:  (Nx, Ny, Nz, 3)
    If it's a vector, we take index 2 as 'z'.
    """
    arr = dset[()]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # Which axis has length 3?
        axis_with_three = [i for i, s in enumerate(arr.shape) if s == 3]
        if len(axis_with_three) != 1:
            raise ValueError(f"Cannot identify component axis in shape {arr.shape}")
        comp_axis = axis_with_three[0]
        # Move comp axis to front, take z=2
        arr_comp_first = np.moveaxis(arr, comp_axis, 0)
        if arr_comp_first.shape[0] != 3:
            raise ValueError("Component axis not of size 3 after move.")
        return arr_comp_first[2]
    raise ValueError(f"Unsupported magnetic field dataset shape: {arr.shape}")

def _maybe_get_bz(f: h5py.File, bz_key_hint: str) -> np.ndarray:
    """
    Try hard to retrieve Bz (or z-component of B) from the file.
    Strategy:
      1) If 'bz_key_hint' exists → use it (with vector extraction if needed).
      2) Else look for a dataset containing 'bz' or 'mag_field_z' in its name.
      3) Else look for a dataset containing 'k_mag_field' (vector), extract z.
    """
    # 1) exact hint present?
    if bz_key_hint in f:
        return _extract_bz_from_dataset(f[bz_key_hint])

    # 2) try common z-component names
    for cand in ("bz", "Bz", "mag_field_z", "magnetic_field_z", "k_mag_field_z"):
        g, k = _find_dataset_recursive(f, cand)
        if g is not None:
            return g[k][()]

    # 3) try to find the vector container and extract z
    for cand in ("k_mag_field", "magnetic_field", "B", "bfield", "mag_field"):
        g, k = _find_dataset_recursive(f, cand)
        if g is not None:
            return _extract_bz_from_dataset(g[k])

    raise KeyError("Could not locate Bz or vector magnetic field dataset in the file.")

def _maybe_get_ne(f: h5py.File, ne_key_hint: str) -> np.ndarray:
    """Retrieve electron density (gas_density) with a couple of fallbacks."""
    if ne_key_hint in f:
        return f[ne_key_hint][()]
    for cand in ("gas_density", "ne", "electron_density", "rho"):
        g, k = _find_dataset_recursive(f, cand)
        if g is not None:
            return g[k][()]
    raise KeyError("Could not locate electron density dataset in the file.")

def _read_spacings(f: h5py.File):
    """
    Read (dx, dy, dz) from x_coor, y_coor, z_coor if available, else default 1.0.
    Expect these to be 3D grids with lines like f['x_coor'][:,0,0], etc.
    """
    dx = 1.0
    dy = 1.0
    dz = 1.0
    if "x_coor" in f:
        dx = axis_spacing(f["x_coor"][:, 0, 0], "x_coor")
    if "y_coor" in f:
        dy = axis_spacing(f["y_coor"][0, :, 0], "y_coor")
    if "z_coor" in f:
        dz = axis_spacing(f["z_coor"][0, 0, :], "z_coor")
    return dx, dy, dz

def compute_rm_map(ne: np.ndarray, bpar: np.ndarray, spacing_los: float, los_axis: int = 1) -> np.ndarray:
    """
    Compute a Rotation Measure (RM) map by integrating n_e * B_parallel along the LOS.
    RM(x,z) = sum_y [ n_e(x,y,z) * B_par(x,y,z) ] * Δl
    We keep RM in arbitrary units; an amplitude factor A will be scanned later.
    """
    if ne.shape != bpar.shape:
        raise ValueError(f"ne and B_parallel shapes differ: {ne.shape} vs {bpar.shape}")

    # Move LOS axis to index 1 temporarily for clear (x, y, z) semantics
    # but we only need to sum over LOS, not reorder spatial axes fully.
    RM = np.sum(ne * bpar, axis=los_axis) * float(spacing_los)

    # After summing over 'los_axis', RM has the remaining two dimensions.
    # If the original arrays were (Nx,Ny,Nz) and los_axis=1, RM is (Nx, Nz).
    return RM

# -----------------------------
# Main workflow
# -----------------------------
def main():
    doesPreview = True

    los_axis = {"x": 0, "y": 1, "z": 2}["y"]

    cube_path = Path(CUBE_PATH_DEFAULT)
    if not cube_path.exists():
        print(f"[error] Cube not found: {cube_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Reading: {cube_path}")
    with h5py.File(cube_path, "r") as f:
        ne = _maybe_get_ne(f, NE_KEY_DEFAULT)
        bpar = _maybe_get_bz(f, BZ_KEY_DEFAULT)   # treat as B_parallel (z by default)
        dx, dy, dz = _read_spacings(f)

    # Sanity
    if ne.shape != bpar.shape:
        raise RuntimeError(f"Shapes mismatch: ne {ne.shape} vs bpar {bpar.shape}")

    spacings = [dx, dy, dz]
    spacing_los = spacings[los_axis]
    print(f"[info] Grid spacing (dx,dy,dz) ~ ({dx:.6g}, {dy:.6g}, {dz:.6g}); LOS='y' → Δl={spacing_los:.6g}")

    # Build RM map (x,z if LOS=y; order is whatever remains after summation)
    RM = compute_rm_map(ne, bpar, spacing_los, los_axis=los_axis)
    RM = RM.astype(np.float64)
    print(f"[info] RM map shape: {RM.shape}")

    # Frequencies
    c = 299_792_458.0
    nu1 = float(NU1_DEFAULT)
    nu2 = float(NU1_DEFAULT + DNU_DEFAULT)
    lam1_sq = (c / nu1) ** 2
    lam2_sq = (c / nu2) ** 2
    dlam_sq = lam2_sq - lam1_sq
    print(f"[info] ν1={nu1:.6e} Hz, ν2={nu2:.6e} Hz  →  λ1²={lam1_sq:.6e} m², λ2²={lam2_sq:.6e} m², Δλ²={dlam_sq:.6e} m²")

    # Angle fields depend only on RM (intrinsic angle cancels in Δχ)
    # Scan amplitude A to show saturation behavior regardless of units
    A_list = np.logspace(math.log10(A_MIN), math.log10(A_MAX), A_N)
    S1_wrap, S1_nowrap = [], []
    Sd_wrap, Sd_nowrap = [], []

    for A in A_list:
        chi1 = wrap_pi(A * lam1_sq * RM)
        chi2 = wrap_pi(A * lam2_sq * RM)
        dchi = wrap_pi(chi2 - chi1)

        S1_wrap.append(neighbor_msd_wrap_after_diff(chi1))
        S1_nowrap.append(neighbor_msd_no_wrap_after_diff(chi1))
        Sd_wrap.append(neighbor_msd_wrap_after_diff(dchi))
        Sd_nowrap.append(neighbor_msd_no_wrap_after_diff(dchi))

    S1_wrap = np.array(S1_wrap)
    S1_nowrap = np.array(S1_nowrap)
    Sd_wrap = np.array(Sd_wrap)
    Sd_nowrap = np.array(Sd_nowrap)

    sat_wrap = np.pi**2 / 12.0
    sat_nowrap = np.pi**2 / 6.0

    print("\n[diagnostics] Saturation constants:")
    print(f"  wrap-after-diff:   π^2/12 = {sat_wrap:.6f}")
    print(f"  no-wrap-after-diff π^2/6  = {sat_nowrap:.6f}")
    print("\n[diagnostics] Tail values at largest A:")
    print(f"  Single freq (wrap-after-diff):   {S1_wrap[-1]:.6f}")
    print(f"  Single freq (no-wrap-after-diff):{S1_nowrap[-1]:.6f}")
    print(f"  Δχ method (wrap-after-diff):     {Sd_wrap[-1]:.6f}")
    print(f"  Δχ method (no-wrap-after-diff):  {Sd_nowrap[-1]:.6f}")

    # -----------------------------
    # Plots
    # -----------------------------
    # 1) wrap-after-diff variant
    plt.figure(figsize=(7,5))
    plt.loglog(A_list, S1_wrap, label="Single freq (wrap-after-diff)")
    plt.loglog(A_list, Sd_wrap, label="Δχ method (wrap-after-diff)")
    plt.axhline(sat_wrap, linestyle="--", linewidth=1.5, label="π²/12 saturation")
    plt.xlabel("RM amplitude scale A")
    plt.ylabel("Nearest-neighbor MSD (wrap-after-diff)")
    plt.title("Saturation check on Athena cube (wrap-after-diff)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("athena_wrap_after_diff.png", dpi=160)
    if not doesPreview:
        plt.show()

    # 2) no-wrap-after-diff variant
    plt.figure(figsize=(7,5))
    plt.loglog(A_list, S1_nowrap, label="Single freq (no-wrap-after-diff)")
    plt.loglog(A_list, Sd_nowrap, label="Δχ method (no-wrap-after-diff)")
    plt.axhline(sat_nowrap, linestyle="--", linewidth=1.5, label="π²/6 saturation")
    plt.xlabel("RM amplitude scale A")
    plt.ylabel("Nearest-neighbor MSD (no-wrap-after-diff)")
    plt.title("Saturation check on Athena cube (no-wrap-after-diff)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("athena_no_wrap_after_diff.png", dpi=160)
    if not doesPreview:
        plt.show()

    # 3) Distributions at a large A to visualize wrapping vs Δχ
    A_show = A_list[-1]
    chi1_show = wrap_pi(A_show * lam1_sq * RM)
    chi2_show = wrap_pi(A_show * lam2_sq * RM)
    dchi_show = wrap_pi(chi2_show - chi1_show)

    plt.figure(figsize=(7,5))
    bins = 100
    hist1, bin_edges1 = np.histogram(chi1_show.ravel(), bins=bins, range=(-np.pi/2, np.pi/2), density=True)
    hist2, bin_edges2 = np.histogram(dchi_show.ravel(), bins=bins, range=(-np.pi/2, np.pi/2), density=True)
    bc1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
    bc2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
    plt.plot(bc1, hist1, label="χ at ν₁ (wrapped)")
    plt.plot(bc2, hist2, label="Δχ = χ(ν₂)-χ(ν₁) (wrapped)")

    print("hist1.min():\t", hist1.min(), "\thist1.max():\t", hist1.max(), "\thist1.sum():\t", hist1.sum(), "\thist1.mean():\t", hist1.mean(), "\thist1.std():\t", hist1.std())
    print("hist2.min():\t", hist2.min(), "\thist2.max():\t", hist2.max(), "\thist2.sum():\t", hist2.sum(), "\thist2.mean():\t", hist2.mean(), "\thist2.std():\t", hist2.std())

    hist2_min_idx = np.argmin(hist2)
    hist2_max_idx = np.argmax(hist2)

    hist2_min_x = bc2[hist2_min_idx]
    hist2_max_x = bc2[hist2_max_idx]

    print(f"hist2 min at X = {hist2_min_x}, max at X = {hist2_max_x}")

    # Plot points for hist2 min/max
    plt.scatter(hist2_min_x, hist2[hist2_min_idx], color='red', zorder=5, label="hist2 min")
    plt.scatter(hist2_max_x, hist2[hist2_max_idx], color='green', zorder=5, label="hist2 max")

    plt.xlabel("Angle (radians)")
    plt.ylabel("Density")
    plt.title(f"Angle distributions at large A = {A_show:.1e}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("athena_distributions.png", dpi=160)
    if not doesPreview:
        plt.show()

    print("\n[done] Saved figures:")
    print("  - athena_wrap_after_diff.png")
    print("  - athena_no_wrap_after_diff.png")
    print("  - athena_distributions.png")

if __name__ == "__main__":
    main()
