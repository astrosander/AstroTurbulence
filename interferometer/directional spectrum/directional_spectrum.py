#!/usr/bin/env python3
"""
Directional spectrum for S_+(R) = <cos(2(f1 − f2))>
===============================================

Goal
----
Numerically demonstrate that the *directional* correlation
    S_+(R) ≡ <cos(2(f(x)) cos(2(f(x+R))) + <sin(2f(x)) sin(2f(x+R))>
           = <cos( 2[f(x) − f(x+R)] )>
is obtained by the inverse FFT of the **directional spectrum**
    P_dir(k) = |FFT{cos(2f)}|^2 + |FFT{sin(2f)}|^2,
and that this equals the **analytic Gaussian prediction**
    S_+^ana(R) = exp( -4 [ σ_f^2 − C_f(R) ] ),
where C_f(R) = < f(x) f(x+R) > is the covariance of a zero-mean Gaussian field f.

We also compare the *spectra* themselves: the ring-averaged P_dir(k)
measured from A=cos(2f), B=sin(2f) versus the ring-averaged FFT of S_+^ana(R).

Field model
-----------
We synthesize f(x,y) as a zero-mean Gaussian random field with a two-slope,
isotropic ring spectrum:
  α_low = +3/2 at k << k_break,   α_high = −5/3 at k >> k_break,
with a smooth transition controlled by s:
  α(k) = α_low * t(k) + α_high * (1 − t(k)),
  t(k) = 1 / (1 + (k/k_break)^s).
As in earlier scripts, we shape the *2D power density* as
  P2D_f(k) ∝ (k / k_pivot)^{α(k) − 1},     (dimensionless pivot)
so that E1D_f(k) = 2π k P2D_f(k) has slope α(k).

Outputs (in fig/directional_compare/)
-------------------------------------
- Splus_vs_R_s{...}.png/.pdf          : S_+(R) numeric (IFFT of P_dir) vs analytic S_+^ana(R)
- Pdir_vs_k_s{...}.png/.pdf           : ring-averaged P_dir(k): numeric vs analytic (FFT of S_+^ana)
- residuals_R_s{...}.png/.pdf         : |S_+^num − S_+^ana| and relative error vs R
- residuals_k_s{...}.png/.pdf         : |P_dir^num − P_dir^ana| (ring-avg) vs k
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable
from numpy.fft import fft2, ifft2, fftfreq

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    nx: int = 512
    ny: int = 512
    dx: float = 1.0
    seed: int = 2025

    # two-slope target for f’s 1D ring spectrum
    alpha_low: float = +1.5
    alpha_high: float = -5.0/3.0
    k_break: float = 0.06        # cycles/dx
    s_list: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)

    # dimensionless pivot for P2D_f ∝ (k/k_pivot)^{α−1}
    k_pivot: float = None        # default: set to k_break below

    # realizations per s (averaged to reduce noise)
    n_real: int = 3

    # ring / radial binning
    nbins_k: int = 280
    nbins_R: int = 280
    kmin: float = 1e-3
    kmax_frac: float = 1.0
    Rmin: float = 1e-3
    Rmax_frac: float = 0.45

    outdir: str = "fig/directional_compare"
    dpi: int = 160

C = Config()
if C.k_pivot is None:
    C.k_pivot = C.k_break

# ──────────────────────────────────────────────────────────────────────
# Utilities (ring / radial averages, correlations)
# ──────────────────────────────────────────────────────────────────────

def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    kmax = K.max() * float(kmax_frac)
    if kmax <= kmin:
        raise ValueError("kmax must exceed kmin; check grid and kmax_frac.")
    bins = np.logspace(np.log10(max(kmin,1e-8)), np.log10(kmax), nbins+1)
    k = K.ravel(); p = P2D.ravel()
    idx = np.digitize(k, bins) - 1
    good = (idx >= 0) & (idx < nbins) & np.isfinite(p)
    sums = np.bincount(idx[good], weights=p[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)
    prof = np.full(nbins, np.nan, float)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    kcent = 0.5*(bins[1:]+bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def radial_average_map(Map2D: np.ndarray, dx: float, nbins: int, Rmin: float, Rmax_frac: float):
    ny, nx = Map2D.shape
    Y = (np.arange(ny) - ny//2)[:,None]
    X = (np.arange(nx) - nx//2)[None,:]
    R = np.hypot(Y, X) * dx
    Rmax = R.max() * float(Rmax_frac)
    bins = np.logspace(np.log10(max(Rmin,1e-8)), np.log10(Rmax), nbins+1)
    r = R.ravel(); m = Map2D.ravel()
    idx = np.digitize(r, bins) - 1
    good = (idx >= 0) & (idx < nbins) & np.isfinite(m)
    sums = np.bincount(idx[good], weights=m[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)
    prof = np.full(nbins, np.nan, float)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    rcent = 0.5*(bins[1:]+bins[:-1])
    m2 = np.isfinite(prof) & (rcent > Rmin)
    return rcent[m2], prof[m2]

def corr_map(field: np.ndarray):
    """
    Cyclic autocorrelation (no mean subtraction) normalized so C(0)=<f^2>.
    Returned in *unshifted* layout (zero-lag at [0,0]) for FFT consistency.
    """
    F = fft2(field)
    C = ifft2(F * np.conj(F)).real / (field.shape[0] * field.shape[1])
    return C

# ──────────────────────────────────────────────────────────────────────
# Two-slope P2D for f (dimensionless pivot)
# ──────────────────────────────────────────────────────────────────────

def P2D_two_slope(alpha_low, alpha_high, k_break, sharp, k_pivot):
    kb = float(k_break); bp = float(k_pivot)
    def P2D(K, eps=1e-20):
        Ksafe = np.maximum(K, eps)
        t = 1.0 / (1.0 + (Ksafe/kb)**sharp)
        alpha = alpha_low * t + alpha_high * (1.0 - t)
        return (Ksafe/bp) ** (alpha - 1.0)
    return P2D

# ──────────────────────────────────────────────────────────────────────
# Field synthesis
# ──────────────────────────────────────────────────────────────────────

def synthesize_f(nx, ny, dx, P2D_func: Callable, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(ny, nx))
    Wk = fft2(w)
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    A = np.sqrt(np.maximum(P2D_func(K), 0.0))
    Fk = Wk * A
    f = ifft2(Fk).real
    f -= f.mean()
    f /= (f.std(ddof=0) + 1e-12)   # unit variance
    return f

# ──────────────────────────────────────────────────────────────────────
# Core experiment for one s
# ──────────────────────────────────────────────────────────────────────

def run_for_s(s: float, C=C):
    rng = np.random.default_rng(C.seed + int(10*s))
    P2D_f = P2D_two_slope(C.alpha_low, C.alpha_high, C.k_break, s, C.k_pivot)

    # Accumulators across realizations (2D)
    acc_Pdir_num = None
    acc_Pdir_ana = None
    acc_Splus_num = None
    acc_Splus_ana = None
    nacc = 0

    for _ in range(C.n_real):
        f = synthesize_f(C.nx, C.ny, C.dx, P2D_f, seed=rng.integers(0, 2**31 - 1))

        # Build A,B and their spectra
        twof = 2.0 * f
        A = np.cos(twof)
        B = np.sin(twof)

        FA = fft2(A); FB = fft2(B)
        Pdir_num = (FA*np.conj(FA)).real + (FB*np.conj(FB)).real  # ≡ P_dir(k)

        # Correlation from numerics (unshifted): S_+^num(R) = IFFT(Pdir)/Npix
        Splus_num = ifft2(Pdir_num).real / (C.nx*C.ny)

        # Analytic S_+ from f’s covariance
        Cf = corr_map(f)                   # unshifted; Cf[0,0] = σ_f^2
        sigma2 = Cf[0,0]
        Splus_ana = np.exp(-4.0 * (sigma2 - Cf))

        # Spectrum from analytic S_+ (should match P_dir): Pdir_ana = FFT(Splus_ana) * Npix
        Pdir_ana = fft2(Splus_ana).real * (C.nx*C.ny)

        # accumulate
        if acc_Pdir_num is None:
            acc_Pdir_num = np.zeros_like(Pdir_num)
            acc_Pdir_ana = np.zeros_like(Pdir_ana)
            acc_Splus_num = np.zeros_like(Splus_num)
            acc_Splus_ana = np.zeros_like(Splus_ana)

        acc_Pdir_num += Pdir_num
        acc_Pdir_ana += Pdir_ana
        acc_Splus_num += Splus_num
        acc_Splus_ana += Splus_ana
        nacc += 1

    # averages over realizations
    Pdir_num_mean = acc_Pdir_num / nacc
    Pdir_ana_mean = acc_Pdir_ana / nacc
    Splus_num_mean = acc_Splus_num / nacc
    Splus_ana_mean = acc_Splus_ana / nacc

    # Radial / ring averages for plots
    k_num, Pk_num = ring_average_2d(Pdir_num_mean, C.dx, C.nbins_k, C.kmin, C.kmax_frac)
    k_ana, Pk_ana = ring_average_2d(Pdir_ana_mean, C.dx, C.nbins_k, C.kmin, C.kmax_frac)

    # center S(R) using fftshift for radial average
    S_num_c = np.fft.fftshift(Splus_num_mean)
    S_ana_c = np.fft.fftshift(Splus_ana_mean)
    R_num, S_num = radial_average_map(S_num_c, C.dx, C.nbins_R, C.Rmin, C.Rmax_frac)
    R_ana, S_ana = radial_average_map(S_ana_c, C.dx, C.nbins_R, C.Rmin, C.Rmax_frac)

    # Interpolate the analytic curves onto numeric grids (for residuals)
    Pk_ana_on_num = np.interp(k_num, k_ana, Pk_ana)
    S_ana_on_num = np.interp(R_num, R_ana, S_ana)

    return dict(
        s=s,
        k=k_num, P_num=Pk_num, P_ana=Pk_ana_on_num,
        R=R_num, S_num=S_num, S_ana=S_ana_on_num
    )

# ──────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────

def plot_S(R, Snum, Sana, s):
    plt.figure(figsize=(7.0,5.0))
    plt.loglog(R, np.maximum(Snum, 1e-30), lw=1.8, label="numerical (IFFT of $P$)")
    plt.loglog(R, np.maximum(Sana, 1e-30), ":", lw=2.2, label=r"analytic $e^{-4(\sigma^2 - C_f)}$")
    plt.xlabel(r"$R$")
    plt.ylabel(r"$S_+(R)=\langle \cos(2[f_1-f_2])\rangle$")
    plt.title(fr"Directional correlation $S_+(R)$  (s={s:g})")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

def plot_P(k, Pnum, Pana, s):
    plt.figure(figsize=(7.0,5.0))
    plt.loglog(k, np.maximum(Pnum, 1e-30), lw=1.8, label=r"numerical")
    plt.loglog(k, np.maximum(Pana, 1e-30), ":", lw=2.2, label=r"analytic") # $\mathcal{F}\{S_+\}\times N$
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$P(k)$")
    plt.title(fr"Directional spectrum  (s={s:g})")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

def plot_residuals_R(R, Snum, Sana, s):
    plt.figure(figsize=(7.0,5.0))
    abs_err = np.abs(Snum - Sana)
    rel_err = abs_err / (np.abs(Sana) + 1e-15)
    plt.loglog(R, np.maximum(abs_err, 1e-30), lw=1.7, label="abs")
    plt.loglog(R, np.maximum(rel_err, 1e-30), "--", lw=1.7, label="rel")
    plt.xlabel(r"$R$")
    plt.ylabel("error")
    plt.title(fr"Errors for $S_+(R)$  (s={s:g})")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

def plot_residuals_k(k, Pnum, Pana, s):
    plt.figure(figsize=(7.0,5.0))
    abs_err = np.abs(Pnum - Pana)
    plt.loglog(k, np.maximum(abs_err, 1e-30), lw=1.7)
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$ (cycles/dx)")
    plt.ylabel(r"$|P^{\rm num}-P^{\rm ana}|$")
    plt.title(fr"Spectrum absolute error  (s={s:g})")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)
    for s in C.s_list:
        res = run_for_s(s, C=C)

        # S(R) compare
        plot_S(res["R"], res["S_num"], res["S_ana"], s=s)
        plt.savefig(os.path.join(C.outdir, f"Splus_vs_R_s{s:g}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Splus_vs_R_s{s:g}.pdf"))
        plt.close()

        # Spectrum compare
        plot_P(res["k"], res["P_num"], res["P_ana"], s=s)
        plt.savefig(os.path.join(C.outdir, f"Pdir_vs_k_s{s:g}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Pdir_vs_k_s{s:g}.pdf"))
        plt.close()

        # Residuals
        plot_residuals_R(res["R"], res["S_num"], res["S_ana"], s=s)
        plt.savefig(os.path.join(C.outdir, f"residuals_R_s{s:g}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"residuals_R_s{s:g}.pdf"))
        plt.close()

        plot_residuals_k(res["k"], res["P_num"], res["P_ana"], s=s)
        plt.savefig(os.path.join(C.outdir, f"residuals_k_s{s:g}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"residuals_k_s{s:g}.pdf"))
        plt.close()

        # Quick console summary at the break scale
        k0 = C.k_break
        # pick the nearest numeric k-bin
        i0 = np.argmin(np.abs(res["k"] - k0))
        print(f"[s={s:>5.1f}]  at k≈k_b:  P_num/P_ana ≈ {res['P_num'][i0]/(res['P_ana'][i0]+1e-300):.3f}   "
              f"S_num/S_ana at R≈1/k_b ≈ {np.interp(1/k0, res['R'], res['S_num'])/(np.interp(1/k0, res['R'], res['S_ana'])+1e-300):.3f}")

    print("Saved figures →", os.path.abspath(C.outdir))
    print("Done.")

if __name__ == "__main__":
    main()
