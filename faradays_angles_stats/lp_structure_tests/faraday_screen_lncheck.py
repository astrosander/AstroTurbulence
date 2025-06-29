#!/usr/bin/env python3
"""
LN-collapse & local-slope test for Lazarian–Pogosyan Faraday screen
==================================================================

* derives S_sim(R,λ)=⟨cos[2λ²ΔΦ]⟩ from the cube (complex polarisation field);
* checks   −ln S_sim /(2λ⁴) ≈ D_Φ(R);
* computes the smoothed local slope d ln D_Φ / d ln R.

Author : <you>
Date   : 2025-06-23
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftn, irfftn, fftshift
from numpy.fft import fftn, ifftn
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter


# ────────────────────────────────────────────────────────────────────
# helper utilities
# ────────────────────────────────────────────────────────────────────
def _axis_spacing(coord1d, name="axis"):
    uniq = np.unique(coord1d.ravel())
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    if d.size:
        return float(np.median(d))
    print(f"[!] {name}: spacing undetermined – using 1")
    return 1.0


def radial_average(map2d, dx=1.0, nbins=64, r_min=1e-3, log_bins=True):
    ny, nx = map2d.shape
    y, x   = np.indices((ny, nx))
    y      = y - ny // 2
    x      = x - nx // 2
    R      = np.hypot(x, y) * dx
    r_max  = R.max() / 2

    if log_bins:
        bins = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
    else:
        bins = np.linspace(0.0, r_max, nbins + 1)

    prof, _, _ = binned_statistic(R.ravel(), map2d.ravel(),
                                  statistic="mean", bins=bins)
    R_cent = 0.5 * (bins[1:] + bins[:-1])
    mask   = ~np.isnan(prof) & (R_cent > r_min)
    return R_cent[mask], prof[mask]


def structure_function_2d(field2d, dx=1.0, **kwa):
    f   = field2d - field2d.mean()
    acf = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    acf = fftshift(acf)
    D   = 2.0 * f.var() - 2.0 * acf
    D[D < 0] = 0
    return radial_average(D, dx=dx, **kwa)


def autocorr_complex(field2d, dx=1.0, **kwa):
    """
    Real part of ⟨f(x) f*(x+R)⟩, *normalised* so that S(R=0)=1.
    """
    ac = ifftn(np.abs(fftn(field2d))**2) / field2d.size   # works for complex
    ac = fftshift(ac).real
    ac /= ac.max()
    return radial_average(ac, dx=dx, **kwa)


def local_log_slope(R, D, win=9, poly=2):
    """
    Smoothed derivative  d ln D / d ln R  using Savitzky-Golay.
    """
    logR  = np.log10(R)
    logD  = np.log10(D)
    # ignore bins with tiny signal before smoothing
    good  = logD > -3
    logDs = savgol_filter(logD[good], win, poly)
    slope = np.gradient(logDs, logR[good], edge_order=2)
    return logR[good], slope


# ────────────────────────────────────────────────────────────────────
def main(cube: Path,
         ne_key="gas_density",
         bz_key="k_mag_field",
         lam_list=(0.06, 0.11, 0.21),
         nbins=72):
    cube = cube.expanduser()
    with h5py.File(cube, "r") as f:
        ne = f[ne_key][:]
        bz = f[bz_key][:]

        dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0
        dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0

    Phi = (ne * bz).sum(axis=2) * dz
    if Phi.std() == 0:
        raise RuntimeError("Φ has zero variance – choose another B component.")

    # reference structure function
    R, D_phi = structure_function_2d(Phi, dx=dx, nbins=nbins, log_bins=True)

    sigma_phi = Phi.std(ddof=0)
    plateau   = 2.0 * sigma_phi**2

    # ─── figure 1 : collapse test ───────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.loglog(R, D_phi, "k", lw=1.8, label=r"$D_\Phi$  (direct)")
    ax1.axhline(plateau,
            color="red", ls=":", lw=1.2,
            label=r"$2\sigma_{\Phi}^{2}$ (saturation)")

    for lam in lam_list:
        # --- safe 2π ambiguity handling: set wrapped pixels to zero weight
        wrap = np.abs(2.0 * lam**2 * Phi) > np.pi
        weight = (~wrap).astype(float)             # 1 for good pixels, 0 else
        P = np.exp(1j * 2.0 * lam**2 * Phi) * weight
        R_S, S  = autocorr_complex(P, dx=dx, nbins=nbins, log_bins=True)

        # interpolate to the R-grid
        if len(R_S) < 4:                 # guard against empty sample set
            continue
        S_interp = np.interp(R, R_S, S)

        valid    = (S_interp > 1e-4)        # avoid −log tiny
        D_est    = np.empty_like(S_interp)
        D_est[:] = np.nan
        D_est[valid] = -np.log(S_interp[valid]) / (2.0 * lam**4)
        if lam > 0.2:
            continue
        ax1.loglog(R[valid], D_est[valid],
                   "--", lw=1.1,
                   label=fr"$-\ln S/(2\lambda^4)$,  $\lambda={lam:.2f}$ m")


        # ax1.loglog(R[valid],
        #              D_est[valid][1] * (R[valid] / R[valid][1]) ** (5/3),
        #              "--", lw=0.5,
        #              label=r"$\propto R^{5/3}$")
            
        ax1.loglog(R[valid],
           D_est[valid][1] * (R[valid] / R[valid][1]) ** (5/3),
           ":", lw=1.0, color="gray",
           label=r"$\propto R^{5/3}$")

    ax1.set(xlabel="R  (same units as dx)", ylabel=r"$D_\Phi$",
            title="Log-collapse: simulation vs theory")
    ax1.legend(fontsize=7, frameon=False)
    fig1.tight_layout()
    fig1.savefig('fig2_collapse.pdf',  bbox_inches='tight')

    # ─── figure 2 : inertial-range slope ────────────────────────────
    slope = local_log_slope(R, D_phi, win=11, poly=2)

    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    logRgood, slope = local_log_slope(R, D_phi, win=11, poly=2)
    ax2.semilogx(10**logRgood, slope, "k", lw=1.5,
                 label=r"$d\ln D_\Phi / d\ln R$")
    ax2.axhline(5/3, color="tab:red", ls=":", lw=1.1, label=r"$5/3$")
    ax2.fill_between(R, 5/3 - 0.15, 5/3 + 0.15,
                     color="tab:red", alpha=0.1)

    ax2.set(xlabel="R  (same units as dx)",
            ylabel="local slope",
            title="Inertial-range exponent")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig('fig3_slope.pdf',     bbox_inches='tight')

    plt.show()


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # main(Path("synthetic_kolmogorov.h5"),
    # main(Path("synthetic_tuned.h5"),
    main(Path("ms01ma08.mhd_w.00300.vtk.h5"),
         ne_key="gas_density",
         bz_key="k_mag_field",
         lam_list=(0.06, 0.11, 0.21))
