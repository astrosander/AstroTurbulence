#!/usr/bin/env python3
"""
3D spectrum with low/high-slope overlays and band fits.

- Loads B_z from an HDF5 cube (Athena or synthetic).
- Computes 3D spectrum: shell-averaged P3D(k) and E1D(k)=4π k^2 P3D(k).
- Overlays expected (theory) slopes and least-squares fits in two k-bands.

Notes
-----
• Units: k is in cycles per dx (FFT convention via np.fft.fftfreq with d=dx).
• Slopes are measured on log–log axes, so they’re exponents of k.
• Choose fit windows via Config.fit_low / Config.fit_high (cycles/dx).
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    h5_path: str = "../mhd_fields.h5"   # or "two_slope_2D_s4_r00.h5"
    dataset_name: str = "k_mag_field"  # 3D Bz dataset in HDF5
    outdir: str = "fig/iso3d_spectrum"

    # Grid spacing (overrides coords if present; set to 1.0 if unknown)
    dx: float = 1.0

    # What to plot: "E1D" (recommended) or "P3D"
    which: str = "P3D"#"E1D"

    # Expected (guide) slopes for E1D in the two asymptotic ranges (optional)
    # E.g. Kolmogorov: low ~ +3/2 (if that’s your model), high ~ -5/3
    alpha_low_expect: float  = +1.5
    alpha_high_expect: float = -5.0/3.0

    # k-break (cycles/dx) to position guide lines if desired (only for picking an anchor)
    k_break_cyc: float = 0.06

    # Shell-average bins and fit bands (cycles/dx)
    nbins: int = 80
    kmin: float = 1e-3
    kmax_frac: float = 1.0  # use up to the Nyquist

    # Fit windows for LS slope estimation (cycles/dx)
    fit_low: Tuple[float, float]  = (0.01, 0.04)
    fit_high: Tuple[float, float] = (0.08, 0.20)

    dpi: int = 160

C = Config()

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _axis_spacing_from_h5(arr, fallback: float) -> float:
    try:
        u = np.unique(arr.ravel())
        dif = np.diff(np.sort(u))
        dif = dif[dif > 0]
        if dif.size:
            return float(np.median(dif))
    except Exception:
        pass
    return float(fallback)

def load_bz_cube(path: str, dataset: str, dx_override: float) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as f:
        bz = f[dataset][:].astype(np.float64)
        # Try to infer dx from coords if present; otherwise use override
        if "x_coor" in f:
            dx = _axis_spacing_from_h5(f["x_coor"][:,0,0], dx_override)
        else:
            dx = dx_override

    dx=1.0
    return bz, float(dx)

def shell_average_3d(power3d: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    """
    Shell-average |B_k|^2 over |k| shells (isotropic).
    k is in cycles/dx using np.fft.fftfreq.
    Returns k_centers, P3D_shell (bin-averaged power density).
    """
    nz, ny, nx = power3d.shape
    kz = np.fft.fftfreq(nz, d=dx)
    ky = np.fft.fftfreq(ny, d=dx)
    kx = np.fft.fftfreq(nx, d=dx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K = np.sqrt(KX*KX + KY*KY + KZ*KZ)

    kmax = K.max() * float(kmax_frac)
    bins = np.logspace(np.log10(max(kmin, 1e-8)), np.log10(kmax), nbins+1)

    k_flat = K.ravel()
    p_flat = power3d.ravel()
    idx = np.digitize(k_flat, bins) - 1

    good = (idx >= 0) & (idx < nbins) & np.isfinite(p_flat)
    sums = np.bincount(idx[good], weights=p_flat[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)

    prof = np.full(nbins, np.nan, float)
    nzb  = cnts > 0
    prof[nzb] = sums[nzb] / cnts[nzb]

    kcent = 0.5*(bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def fit_loglog(x, y, xmin, xmax):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= xmin) & (x <= xmax)
    if np.count_nonzero(m) < 5:
        return np.nan, np.nan, np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    # slope, intercept with covariance
    coeff, cov = np.polyfit(X, Y, deg=1, cov=True)
    a, b = coeff[0], coeff[1]
    a_err = float(np.sqrt(cov[0,0]))
    # R^2
    yhat = a*X + b
    ss_res = np.sum((Y - yhat)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), a_err, float(r2)

# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    # 1) load cube
    bz, dx = load_bz_cube(C.h5_path, C.dataset_name, C.dx)
    nz, ny, nx = bz.shape
    print(f"[load] {C.dataset_name} shape = {bz.shape}, dx={dx}")

    # 2) 3D FFT and power
    F = np.fft.fftn(bz)
    P3 = (F * np.conj(F)).real  # |B_k|^2 (unnormalized overall constant; slopes unaffected)

    # 3) isotropic shell average
    k1d, P3_shell = shell_average_3d(P3, dx, C.nbins, C.kmin, C.kmax_frac)

    # 4) choose what to plot
    if C.which.upper() == "E1D":
        E1D = 4.0 * np.pi * (k1d**2) * P3_shell
        y = E1D
        y_lab = r"$E_{1\mathrm{D}}(k)$"
    else:
        y = P3_shell
        y_lab = r"$P_{3\mathrm{D}}(k)$"

    # 5) least-squares fits in low/high bands
    a_lo, b_lo, aerr_lo, r2_lo = fit_loglog(k1d, y, *C.fit_low)
    a_hi, b_hi, aerr_hi, r2_hi = fit_loglog(k1d, y, *C.fit_high)

    print("\n— Fit results (log–log) —")
    print(f"Low band  [{C.fit_low[0]:.4g}, {C.fit_low[1]:.4g}] : slope = {a_lo:+.4f} ± {aerr_lo:.4f}   R² = {r2_lo:.4f}")
    print(f"High band [{C.fit_high[0]:.4g}, {C.fit_high[1]:.4g}]: slope = {a_hi:+.4f} ± {aerr_hi:.4f}   R² = {r2_hi:.4f}")

    # 6) build guide lines (expected & fitted), anchored at pivots
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # data
    ax.loglog(k1d, y, lw=2.0, label="spectrum")

    # choose pivots near the middle of each fit band
    k_piv_lo = np.sqrt(C.fit_low[0]*C.fit_low[1])
    k_piv_hi = np.sqrt(C.fit_high[0]*C.fit_high[1])

    # interpolate y at pivots for anchoring the lines
    def interp_loglog(xq, x, y):
        return np.exp(np.interp(np.log(xq), np.log(x), np.log(y)))

    y_piv_lo = interp_loglog(k_piv_lo, k1d, y)
    y_piv_hi = interp_loglog(k_piv_hi, k1d, y)

    # (A) expected slope guides (optional)
    if np.isfinite(C.alpha_low_expect):
        kr = np.logspace(np.log10(C.fit_low[0]/1.2), np.log10(C.fit_low[1]*1.2), 100)
        # ax.loglog(kr, y_piv_lo*(kr/k_piv_lo)**(C.alpha_low_expect),
        #           "--", lw=1.3, alpha=0.7, label=fr"guide: $\alpha_{{\rm low}}={C.alpha_low_expect:+.3f}$")
    if np.isfinite(C.alpha_high_expect):
        kr = np.logspace(np.log10(C.fit_high[0]/1.2), np.log10(C.fit_high[1]*1.2), 100)
        # ax.loglog(kr, y_piv_hi*(kr/k_piv_hi)**(C.alpha_high_expect),
        #           "--", lw=1.3, alpha=0.7, label=fr"guide: $\alpha_{{\rm high}}={C.alpha_high_expect:+.3f}$")

    # (B) fitted slope overlays
    if np.isfinite(a_lo):
        kr = np.logspace(np.log10(C.fit_low[0]), np.log10(C.fit_low[1]), 200)
        ax.loglog(kr, np.exp(b_lo)*(kr**a_lo), "-", lw=2.0,
                  label=fr"{a_lo:+.3f}±{aerr_lo:.3f} (R$^2$={r2_lo:.3f})")
    if np.isfinite(a_hi):
        kr = np.logspace(np.log10(C.fit_high[0]), np.log10(C.fit_high[1]), 200)
        ax.loglog(kr, np.exp(b_hi)*(kr**a_hi), "-", lw=2.0,
                  label=fr"{a_hi:+.3f}±{aerr_hi:.3f} (R$^2$={r2_hi:.3f})")

    # cosmetics
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(y_lab)
    # ax.set_title(rf"3D spectrum ({C.which}), {os.path.basename(C.h5_path)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    os.makedirs(C.outdir, exist_ok=True)
    out = os.path.join(C.outdir, f"iso3d_{C.which.lower()}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=C.dpi)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"\nSaved → {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
