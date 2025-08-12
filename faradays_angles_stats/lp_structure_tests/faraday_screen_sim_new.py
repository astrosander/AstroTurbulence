#!/usr/bin/env python3
"""
Faraday-screen angle statistics: Athena (top) vs Synthetic (bottom) in a 2x2 layout.

- Top row:   Athena — [D_Φ(R), D_φ(R, λ)]
- Bottom row: Synthetic — [D_Φ(R), D_φ(R, λ)]
- Red vertical floating labels on the left mark each row.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from numpy.fft import rfftn, irfftn, fftshift


# ──────────────────────────────────────────────────────────────────────
# 1. Helpers
# ──────────────────────────────────────────────────────────────────────
def _axis_spacing(coord_1d, name="axis"):
    """Median positive spacing of a 1-D coordinate array, fallback to 1.0."""
    unique = np.unique(coord_1d.ravel())
    diffs = np.diff(np.sort(unique))
    diffs = diffs[diffs > 0]
    if diffs.size:
        return float(np.median(diffs))
    print(f"[!] {name}: could not determine spacing – using dx=1")
    return 1.0


LOG_BINS = True
NBINS    = 480


def structure_function_2d(field, dx=1.0, nbins=NBINS, r_min=1e-3):
    """Isotropic second-order structure function of a 2-D scalar field."""
    f = field - field.mean()

    # autocorrelation via Wiener–Khinchin
    power = np.abs(rfftn(f)) ** 2
    ac    = irfftn(power, s=f.shape) / f.size
    ac    = fftshift(ac)

    D = 2.0 * f.var() - 2.0 * ac
    D[D < 0] = 0  # numerical guard

    ny, nx = field.shape
    y_idx  = np.arange(ny)[:, None] - ny//2
    x_idx  = np.arange(nx)[None, :] - nx//2
    R      = np.hypot(x_idx, y_idx) * dx

    r_max = R.max() * 0.45
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
    """D_φ(R, λ) = ½ [1 − exp(−2 λ⁴ D_Φ)]"""
    return 0.5 * (1.0 - np.exp(-2.0 * lam**4 * D_phi))


def load_compute(cube_path, ne_key="gas_density", bz_key="k_mag_field"):
    """Load cube and compute Φ, D_Φ(R)."""
    cube_path = os.path.expanduser(cube_path)
    if not os.path.exists(cube_path):
        raise FileNotFoundError(cube_path)

    with h5py.File(cube_path, "r") as f:
        ne = f[ne_key][:]   # electron density
        bz = f[bz_key][:]   # LOS magnetic field

        dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0
        dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0

    print(f"[{os.path.basename(cube_path)}] shape={ne.shape}  dx={dx}  dz={dz}")

    # Φ(X) = Σ n_e B_z dz
    Phi = (ne * bz).sum(axis=2) * dz
    if Phi.std() == 0:
        raise RuntimeError(f"{cube_path}: Φ has zero variance – B_z appears constant!")
    R, D_phi = structure_function_2d(Phi, dx=dx)
    return R, D_phi


def plot_row(ax_left, ax_right, R, Dphi, lam_list, title_prefix):
    """Plot D_Φ and D_φ for one dataset into a row of axes."""
    # Left: D_Φ
    ax_left.loglog(R, Dphi, label="simulation", lw=1.5)
    ax_left.loglog(R, Dphi[1] * (R / R[1]) ** (5/3),
                   "--", lw=1.0, label=r"$\propto R^{5/3}$")
    ax_left.set(xlabel="R (same units as dx)", ylabel=r"$D_{\Phi}(R)$",
                title=f"{title_prefix}: RM structure")
    ax_left.legend(frameon=False)

    # Right: D_φ(R, λ)
    for lam in lam_list:
        y = angle_structure_function(Dphi, lam)
        ax_right.loglog(R, y, lw=1.5, label=fr"$\lambda={lam:.2f}$ m")
        ax_right.loglog(R, y[1] * (R / R[1]) ** (5/3), "--", lw=1.0,
                        label=r"$\propto R^{5/3}$")
    ax_right.set(xlabel="R (same units as dx)", ylabel=r"$D_{\varphi}(R,\lambda)$",
                 title=f"{title_prefix}: Polarization-angle structure")
    ax_right.set_ylim(top=0.6)
    ax_right.legend(frameon=False)


# ──────────────────────────────────────────────────────────────────────
# 2. Main driver
# ──────────────────────────────────────────────────────────────────────
def main():
    # Fixed inputs
    athena_path    = "ms01ma08.mhd_w.00300.vtk.h5"
    synthetic_path = "synthetic_kolmogorov.h5"
    lam_list       = (0.06, 0.11, 0.21)

    # compute both datasets
    R_A, D_A = load_compute(athena_path)
    R_S, D_S = load_compute(synthetic_path)

    # 2x2 grid: top (Athena), bottom (Synthetic)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)

    # Top row: Athena
    plot_row(axs[0, 0], axs[0, 1], R_A, D_A, lam_list, title_prefix="Athena")

    # Bottom row: Synthetic
    plot_row(axs[1, 0], axs[1, 1], R_S, D_S, lam_list, title_prefix="Synthetic")

    # Red vertical floating labels on the left of each row
    fig.text(0.02, 0.75, "ATHENA", rotation=90, color="red",
             va="center", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.25, "SYNTHETIC", rotation=90, color="red",
             va="center", ha="center", fontsize=12, fontweight="bold")

    # layout & save
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(rect=(0.05, 0.0, 1.0, 1.0))  # leave room for vertical labels
    fig.savefig('figures/fig_rm_angle_2x2.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
