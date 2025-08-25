#!/usr/bin/env python3
"""
Transition smoothness demo (patched with dimensionless pivot b)
===============================================================

We build 2D random fields whose target 1D ring spectrum has two slopes:
  α_low = +3/2 at k << k_break, and α_high = -5/3 at k >> k_break,
with a blended transition α(k) = α_low * t(k) + α_high * [1 - t(k)],
t(k) = 1 / (1 + (k/k_break)^s). The parameter 's' controls smoothness:
small s = smooth/wide transition; large s = sharp/narrow transition.

PATCH REQUESTED:
- Use a dimensionless pivot in the Fourier power, i.e. replace k^ν with (k/b)^ν.
- Here we set b = k_break by default (can be changed in Config.k_pivot).

For several s values, we:
  • synthesize fields by shaping FFT amplitudes with P2D(k) ∝ (k/b)^{α(k)-1},
  • compute E1D(k) = 2π k P2D_ring(k),
  • estimate local slope α_est(k) = d log E1D / d log k,
  • compare to the analytic α(k), and quantify a simple transition width.

Outputs (in fig/transition_smoothness/):
  - E1D_vs_k_all_s.png/.pdf : E1D(k) for all s
  - alpha_local_all_s.png/.pdf : local slope α_est(k) + analytic α(k)
  - dalpha_dlnk_all_s.png/.pdf : |dα_est/dlnk| to visualize break width
  - Prints fitted low/high slopes and a width metric for each s
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

    # target shell slopes and break
    alpha_low: float = +1.5
    alpha_high: float = -5.0/3.0
    k_break: float = 0.06   # choose inside inertial range

    # PIVOT for dimensionless power (PATCH): P2D ∝ (k / k_pivot)^{α-1}
    # Good default is k_pivot = k_break
    k_pivot: float = None

    # transition sharpness values to test
    s_list: Tuple[float, ...] = (2.0, 4.0)

    # averaging over realizations per s (reduces noise)
    n_real: int = 3

    # ring-average bins and fit bands
    nbins: int = 70
    kmin: float = 1e-3
    kmax_frac: float = 1.0
    fit_low: Tuple[float, float] = (0.01, 0.04)
    fit_high: Tuple[float, float] = (0.08, 0.20)

    # local-slope window (in bins) for regression in log–log
    slope_win: int = 11  # odd number recommended

    outdir: str = "fig/transition_smoothness"
    dpi: int = 160

C = Config()
if C.k_pivot is None:
    C.k_pivot = C.k_break  # pivot defaults to the break

# ──────────────────────────────────────────────────────────────────────
# Helpers
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
    kcent = 0.5 * (bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def fit_loglog(x, y, xmin, xmax):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0) & (x>=xmin) & (x<=xmax)
    if np.count_nonzero(m) < 5: return np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(np.exp(b))

def local_slope_loglog(x, y, win: int):
    """
    Estimate local slope α_est(k) = d ln y / d ln x by sliding
    least-squares in log–log space with window 'win' (odd).
    """
    assert win >= 3 and win % 2 == 1
    n = len(x)
    alpha = np.full(n, np.nan, float)
    lx = np.log(x); ly = np.log(y)
    half = win // 2
    for i in range(half, n-half):
        sl = slice(i-half, i+half+1)
        X = np.vstack([lx[sl], np.ones(win)]).T
        if np.all(np.isfinite(X)) and np.all(np.isfinite(ly[sl])):
            a, _ = np.linalg.lstsq(X, ly[sl], rcond=None)[0]
            alpha[i] = a
    return alpha

# ──────────────────────────────────────────────────────────────────────
# Two-slope density with dimensionless pivot (PATCHED)
# ──────────────────────────────────────────────────────────────────────

def P2D_two_slope(alpha_low, alpha_high, k_break, sharp, k_pivot) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return P2D(k) that yields ring slope α(k) with two asymptotes.
    Since E1D = 2πk P2D, set P2D ∝ (k/k_pivot)^{α(k)-1}.
    The blend t(k) still uses k_break for where the transition happens.
    """
    k_break = float(k_break)
    k_pivot = float(k_pivot)
    def P2D(K, eps=1e-20):
        Ksafe = np.maximum(K, eps)
        x = Ksafe / k_break
        t = 1.0 / (1.0 + x**sharp)           # ~1 low-k, ~0 high-k
        alpha = alpha_low * t + alpha_high * (1.0 - t)
        Y = Ksafe / k_pivot                  # PATCH: dimensionless pivot
        return Y ** (alpha - 1.0)
    return P2D

# ──────────────────────────────────────────────────────────────────────
# Field synthesis
# ──────────────────────────────────────────────────────────────────────

def synthesize_2d(nx, ny, dx, P2D_func: Callable, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(ny, nx))
    Wk = fft2(w)
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    A = np.sqrt(np.maximum(P2D_func(K), 0.0))
    Fk = Wk * A
    f = ifft2(Fk).real
    f -= f.mean(); f /= (f.std(ddof=0) + 1e-12)
    return f

# ──────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)
    rng = np.random.default_rng(C.seed)

    results = []  # store (s, k, E1D_mean, alpha_local, alpha_target, dalpha_dlnk)

    for s in C.s_list:
        P2D_func = P2D_two_slope(C.alpha_low, C.alpha_high, C.k_break, s, C.k_pivot)

        # average over realizations for cleaner curves
        acc_E = None
        acc_P = None
        acc_counts = 0

        for _ in range(C.n_real):
            seed_r = rng.integers(0, 2**31 - 1)
            f2 = synthesize_2d(C.nx, C.ny, C.dx, P2D_func, seed=seed_r)

            Fk = fft2(f2)
            P2D = (Fk * np.conj(Fk)).real

            k1d, Pk = ring_average_2d(P2D, C.dx, C.nbins, C.kmin, C.kmax_frac)
            E1D = 2*np.pi * k1d * Pk

            if acc_E is None:
                acc_E = np.zeros_like(E1D)
                acc_P = np.zeros_like(Pk)
                acc_k = k1d.copy()

            # Rebin check (should match)
            if not np.allclose(acc_k, k1d):
                raise RuntimeError("Inconsistent k-binning across realizations")

            acc_E += E1D
            acc_P += Pk
            acc_counts += 1

        # averages
        E1D_mean = acc_E / acc_counts
        Pk_mean  = acc_P / acc_counts

        # local slope on E1D
        alpha_local = local_slope_loglog(acc_k, E1D_mean, C.slope_win)

        # analytic target α(k)
        x = acc_k / C.k_break
        t = 1.0 / (1.0 + x**s)
        alpha_target = C.alpha_low * t + C.alpha_high * (1.0 - t)

        # numerical derivative magnitude |dα/dlnk|
        dalpha = np.full_like(alpha_local, np.nan)
        lnk = np.log(acc_k)
        for i in range(1, len(acc_k)-1):
            if np.isfinite(alpha_local[i-1:i+2]).all():
                dalpha[i] = (alpha_local[i+1] - alpha_local[i-1]) / (lnk[i+1] - lnk[i-1])
        results.append((s, acc_k, E1D_mean, alpha_local, alpha_target, dalpha))

        # Fit low/high slopes for a quick report
        a_lo, _ = fit_loglog(acc_k, E1D_mean, *C.fit_low)
        a_hi, _ = fit_loglog(acc_k, E1D_mean, *C.fit_high)

        # crude "transition width": between 10% and 90% of the total slope drop
        aL, aH = C.alpha_low, C.alpha_high
        a10 = aH + 0.9*(aL - aH)
        a90 = aH + 0.1*(aL - aH)

        def cross_k(level):
            ok = np.isfinite(alpha_local)
            k = acc_k[ok]; a = alpha_local[ok]
            idx = np.where((a[:-1] >= level) & (a[1:] < level))[0]
            if idx.size == 0:
                return np.nan
            i = idx[0]
            # interpolate in log k
            x0, x1 = np.log(k[i]), np.log(k[i+1])
            y0, y1 = a[i], a[i+1]
            if y1 == y0: return np.exp(x0)
            f = (level - y0) / (y1 - y0)
            return np.exp(x0 + f*(x1 - x0))

        k10 = cross_k(a10); k90 = cross_k(a90)
        width = (k90/k10) if (np.isfinite(k10) and np.isfinite(k90) and k10>0) else np.nan

        print(f"[s={s:>5.1f}]  fit α_low≈{a_lo:+.3f}  α_high≈{a_hi:+.3f}   "
              f"break width (k90/k10)≈{width:.3f}")

    # ── Plots across all s ─────────────────────────────────────────────

    # 1) E1D(k) for all s
    plt.figure(figsize=(7.2,5.2))
    for (s, k, E, alpha_loc, alpha_tar, dalpha) in results:
        plt.loglog(k, E, lw=1.7, label=fr"$s={s:g}$")
    # slope guides anchored near k_break
    for slope, xfac, txt in [(C.alpha_low, 1.8, r"$+3/2$"), (C.alpha_high, 2.5, r"$-5/3$")]:
        kp = C.k_break / xfac if slope>0 else C.k_break * xfac
        yref = np.interp(kp, results[0][1], results[0][2])
        kr = np.logspace(np.log10(kp/6), np.log10(kp*6), 200)
        plt.loglog(kr, yref*(kr/kp)**slope, "--", lw=1.0, alpha=0.7)
        plt.text(kp, yref*(1.5), txt, fontsize=10, ha='center')
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    # plt.ylabel(r"$E_{1 D}(k)$")
    plt.title(r"Effect of transition smoothness $s$ on $E_{1D}(k)$")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which='both', alpha=0.3)
    os.makedirs(C.outdir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "E1D_vs_k_all_s.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "E1D_vs_k_all_s.pdf")); plt.close()

    # 2) Local slope α_est(k) vs analytic α(k)
    plt.figure(figsize=(7.2,5.2))
    for (s, k, E, alpha_loc, alpha_tar, dalpha) in results:
        plt.semilogx(k, alpha_loc, lw=1.7, label=fr"numerical $s={s:g}$")
    # overlay analytic α(k) for two representative s (min and max)
    s_min = min(C.s_list); s_max = max(C.s_list)
    kgrid = results[0][1]
    for s in (s_min, s_max):
        x = kgrid / C.k_break
        alpha_tar = C.alpha_low/(1+x**s) + C.alpha_high*(1 - 1/(1+x**s))
        plt.semilogx(kgrid, alpha_tar, ':', lw=2.2, label=fr"analytic $s={s:g}$")
    # plt.axhline(C.alpha_low, color='k', ls='--', lw=0.8)
    # plt.axhline(C.alpha_high, color='k', ls='--', lw=0.8)
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"local slope")
    plt.title(r"Local slope: numerical vs analytic")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "alpha_local_all_s.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "alpha_local_all_s.pdf")); plt.close()

    # 3) |dα/dln k| to visualize transition width
    plt.figure(figsize=(7.2,5.2))
    for (s, k, E, alpha_loc, alpha_tar, dalpha) in results:
        plt.semilogx(k, np.abs(dalpha), lw=1.7, label=fr"$s={s:g}$")
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$|\frac{d\alpha_{\rm est}}{d\ln k}|$")
    plt.title(r"Transition sharpness (peak height/width vs $s$)")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "dalpha_dlnk_all_s.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "dalpha_dlnk_all_s.pdf")); plt.close()

    print(f"\nSaved figures → {os.path.abspath(C.outdir)}")
    print(f"PATCH active: P2D ∝ (k / b)^{{α-1}} with b = {C.k_pivot:.4g} cycles/dx")

if __name__ == "__main__":
    main()
