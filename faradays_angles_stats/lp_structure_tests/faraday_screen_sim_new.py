#!/usr/bin/env python3
"""
Faraday-screen angle statistics from a real 3-D MHD cube
--------------------------------------------------------

* Reads a .vtk.h5 snapshot that contains:
    gas_density, k_mag_field, (optionally i/j components),
    x_coor, y_coor, z_coor  (coordinate grids)

* Computes Φ(X) = Σ n_e B_z dz, its 2-D structure function D_Φ(R),
  and the polarization-angle structure function
      D_φ(R, λ) = ½[1 − exp(−2 λ⁴ D_Φ)]
  for a few sample wavelengths.

Author : <you>
Date   : 2025-06-23
License: MIT
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from numpy.fft import rfftn, irfftn, fftshift


# ──────────────────────────────────────────────────────────────────────
# 1. Helpers
# ──────────────────────────────────────────────────────────────────────
def _axis_spacing(coord_1d, name="axis"):
    """
    Return the median positive spacing of a 1-D coordinate array.
    Fall back to 1.0 if the array is degenerate (duplicates, constant, …).
    """
    unique = np.unique(coord_1d.ravel())
    diffs = np.diff(np.sort(unique))
    diffs = diffs[diffs > 0]

    if diffs.size:
        return float(np.median(diffs))

    print(f"[!] {name}: could not determine spacing – using dx=1")
    return 1.0


LOG_BINS = True       # log-spaced radial bins give smoother curves
NBINS    = 480         # number of radial bins to average over


def structure_function_2d(field, dx=1.0, nbins=NBINS, r_min=1e-3):
    """
    Isotropic second-order structure function of a 2-D scalar field.

    Parameters
    ----------
    field : 2-D ndarray
    dx    : pixel size (same units as R you want to plot)
    nbins : number of radial bins
    r_min : smallest radius to keep (avoid R≈0 artefacts)

    Returns
    -------
    R_centers, D_R : 1-D arrays of length `nbins` (minus possibly empty bins)
    """
    f = field - field.mean()

    # autocorrelation via Wiener–Khinchin
    power = np.abs(rfftn(f)) ** 2
    ac    = irfftn(power, s=f.shape) / f.size
    ac    = fftshift(ac)
    
    D = 2.0 * f.var() - 2.0 * ac
    D[D < 0] = 0      # numerical noise guard

    ny, nx = field.shape
    y_idx  = np.arange(ny)[:, None] - ny//2
    x_idx  = np.arange(nx)[None, :] - nx//2
    R                  = np.hypot(x_idx, y_idx) * dx

    r_max = R.max()*0.45
    if LOG_BINS:
        bins = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
    else:
        bins = np.linspace(0.0, r_max, nbins + 1)

    D_R, _, _ = binned_statistic(R.ravel(), D.ravel(),
                                 statistic="mean", bins=bins)
    R_cent = 0.5 * (bins[1:] + bins[:-1])
    mask   = ~np.isnan(D_R) & (R_cent > r_min)

    return R_cent[mask], D_R[mask]


def angle_structure_function(D_phi, lam):
    """ D_φ(R, λ) = ½ [1 − exp(−2 λ⁴ D_Φ)] """
    return 0.5 * (1.0 - np.exp(-2.0 * lam**4 * D_phi))


# ──────────────────────────────────────────────────────────────────────
# 2. Main driver
# ──────────────────────────────────────────────────────────────────────
def main(cube_path: Path,
         ne_key="gas_density",
         bz_key="k_mag_field",
         lam_list=(0.06, 0.11, 0.21)):
    cube_path = cube_path.expanduser()
    if not cube_path.exists():
        raise FileNotFoundError(cube_path)

    # ── read data ────────────────────────────────────────────────────
    with h5py.File(cube_path, "r") as f:
        ne = f[ne_key][:]   # electron number density
        bz = f[bz_key][:]   # magnetic field along the line of sight (z-axis)

        dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0    # physical pixel spacing in the x–y plane
        dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0    # physical grid spacing along the line-of-sight direction (z-axis)

    print(f"   cube shape : {ne.shape}")
    print(f"   dx, dz     : {dx} {dz}")

    # Φ(X) = Σ n_e B_z dz
    Phi = (ne * bz).sum(axis=2) * dz
    sigma_phi = Phi.std()   # standard deviation
    if sigma_phi == 0:
        raise RuntimeError("Φ has zero variance – B_z component appears constant!")

    # ── structure functions ──────────────────────────────────────────
    R, D_phi = structure_function_2d(Phi, dx=dx)

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: D_Φ
    ax[0].loglog(R, D_phi, label="simulation", lw=1.5)
    ax[0].loglog(R,
                 D_phi[1] * (R / R[1]) ** (5/3),
                 "--", lw=1.0,
                 label=r"$\propto R^{5/3}$")
    ax[0].set(xlabel="R  (same units as dx)",
              ylabel=r"$D_{\Phi}(R)$",
              title="RM structure")
    ax[0].legend(frameon=False)

    # Right: D_φ(R, λ)
    for lam in lam_list:
        ax[1].loglog(R,
                     angle_structure_function(D_phi, lam),
                     label=fr"$\lambda={lam:.2f}$ m",
                     lw=1.5)
        ax[1].loglog(R,
                     angle_structure_function(D_phi, lam)[1] * (R / R[1]) ** (5/3),
                     "--", lw=1.0,
                     label=r"$\propto R^{5/3}$")
    ax[1].set(xlabel="R  (same units as dx)",
              ylabel=r"$D_{\varphi}(R,\lambda)$",
              title="Polarization-angle structure")
    ax[1].set_ylim(top=0.6)  # Limit y-axis max to 0.5
    ax[1].legend(frameon=False)

    fig.savefig('figures/fig1_rm_angle.pdf', bbox_inches='tight')
    fig.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # main(Path("synthetic_kolmogorov.h5"),
    main(Path("synthetic_tuned.h5"),
    # main(Path("ms01ma08.mhd_w.00300.vtk.h5"),
         ne_key="gas_density",
         bz_key="k_mag_field",
         lam_list=(0.06, 0.11, 0.21))
