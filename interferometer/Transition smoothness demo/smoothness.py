#!/usr/bin/env python3
"""
Transition smoothness demo: how t(k) sharpness affects observed spectra
=======================================================================

We build 2D random fields whose target 1D ring spectrum has two slopes:
  α_low = +3/2 at k << k_break, and α_high = -5/3 at k >> k_break,
with a blended transition α(k) = α_low * t(k) + α_high * (1 - t(k)),
t(k) = 1 / (1 + (k/k_break)^s). The parameter 's' controls smoothness:
small s = smooth/wide transition; large s = sharp/narrow transition.

For several s values, we:
  • synthesize fields by shaping FFT amplitudes with P2D(k) ∝ k^{α(k)-1},
  • compute E1D(k) = 2π k P2D_ring(k),
  • estimate local slope α_est(k) from log–log derivatives,
  • compare to analytic α(k), and quantify the transition width.

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
import h5py
rng = np.random.default_rng()

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
    k_break: float = 0.06   # cycles/dx (pick inside inertial range)

    # transition sharpness values to test
    s_list: Tuple[float, ...] = (2.0, 3.0, 4.0)

    # averaging over realizations per s (reduces noise in E1D, alpha_est)
    n_real: int = 3

    # ring-average bins and fit bands
    nbins: int = 50
    kmin: float = 1e-3
    kmax_frac: float = 1.0
    fit_low: Tuple[float, float] = (0.01, 0.04)
    fit_high: Tuple[float, float] = (0.08, 0.20)

    # local-slope window (in bins) for regression in log–log
    slope_win: int = 11  # odd number recommended

    outdir: str = "fig/transition_smoothness"
    dpi: int = 160

C = Config()

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def save_field_as_h5(map2d: np.ndarray, dx: float, out_path: str, *,
                     alpha_low: float, alpha_high: float, k_break: float, s_value: float):
    """
    Save a 2D map as a (x,y,z=1) 'cube' with the same dataset names used elsewhere:
      gas_density, k_mag_field, x_coor, y_coor, z_coor
    Coordinates are written so downstream code can recover dx via x_coor[:,0,0].
    """
    ny, nx = map2d.shape
    Nx, Ny = nx, ny

    # Arrange datasets as (x, y, z) with z=1
    bz = map2d.T[:, :, None].astype(np.float32)   # (Nx, Ny, 1)
    ne = np.ones_like(bz, dtype=np.float32)       # simple positive field (optional)

    # Coordinates (match shapes and axis order used by your other scripts)
    x = (np.arange(Nx, dtype=np.float64) - Nx/2 + 0.5) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny/2 + 0.5) * dx
    z = np.array([0.0], dtype=np.float64)  # single slab

    X = np.broadcast_to(x[:, None, None], (Nx, Ny, 1)).astype(np.float32)
    Y = np.broadcast_to(y[None, :, None], (Nx, Ny, 1)).astype(np.float32)
    Z = np.broadcast_to(z[None, None, :], (Nx, Ny, 1)).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("gas_density", data=ne, compression="gzip")
        h5.create_dataset("k_mag_field", data=bz, compression="gzip")
        h5.create_dataset("x_coor",      data=X,  compression="gzip")
        h5.create_dataset("y_coor",      data=Y,  compression="gzip")
        h5.create_dataset("z_coor",      data=Z,  compression="gzip")
        # Useful metadata
        h5.attrs["alpha_low"]  = float(alpha_low)
        h5.attrs["alpha_high"] = float(alpha_high)
        h5.attrs["k_break"]    = float(k_break)
        h5.attrs["s"]          = float(s_value)
        h5.attrs["dx"]         = float(dx)
        h5.attrs["note"]       = "2D map saved as (x,y,z=1) cube for compatibility"


def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    kmax = K.max() * float(kmax_frac)
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

def P2D_two_slope(alpha_low, alpha_high, k_break, sharp) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return P2D(k) that yields ring slope α(k) with two asymptotes.
    Since E1D = 2πk P2D, set P2D ∝ k^{α(k)-1}.
    """
    def P2D(K, eps=1e-12):
        Ksafe = np.maximum(K, eps)
        x = Ksafe / float(k_break)
        t = 1.0 / (1.0 + x**sharp)          # ~1 low-k, ~0 high-k
        alpha = alpha_low * t + alpha_high * (1.0 - t)
        return ((Ksafe / float(k_break)) ** (alpha - 1.0))
    return P2D

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
    h5_dir = "h5/transition_smoothness"
    os.makedirs(h5_dir, exist_ok=True)
    results = [] 
    for s in C.s_list:
        P2D_func = P2D_two_slope(C.alpha_low, C.alpha_high, C.k_break, s)

        acc_E = acc_P = None
        acc_counts = 0
        saved_one = False  # save only the first realization per s

        for r in range(C.n_real):
            seed_r = rng.integers(0, 2**31 - 1)
            f2 = synthesize_2d(C.nx, C.ny, C.dx, P2D_func, seed=seed_r)

            # ── NEW: save this realization as .h5 (compatible with your loader)
            if not saved_one:
                out_h5 = os.path.join(h5_dir, f"two_slope_2D_s{s:g}_r{r:02d}.h5")
                save_field_as_h5(
                    f2, C.dx, out_h5,
                    alpha_low=C.alpha_low, alpha_high=C.alpha_high,
                    k_break=C.k_break, s_value=s
                )
                saved_one = True

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
        # finite-difference on alpha_local (ignore NaNs at edges)
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
        # from α_low to α_high in alpha_local
        aL, aH = C.alpha_low, C.alpha_high
        a10 = aH + 0.9*(aL - aH)
        a90 = aH + 0.1*(aL - aH)
        # find k where alpha_local crosses a10 and a90
        def cross_k(level):
            ok = np.isfinite(alpha_local)
            k = acc_k[ok]; a = alpha_local[ok]
            # find first index where a goes below 'level'
            idx = np.where((a[:-1] >= level) & (a[1:] < level))[0]
            if idx.size == 0:
                return np.nan
            i = idx[0]
            # linear interp in log space of k for better stability
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
    # add slope guides
    # x232=1.2
    for slope, xfac, txt in [(C.alpha_low, 1.8, r"$+3/2$"), (C.alpha_high, 2.5, r"$-5/3$")]:
        kp = C.k_break / xfac if slope>0 else C.k_break * xfac
        # anchor line through a point on first curve
        yref = np.interp(kp, results[0][1], results[0][2])
        kr = np.logspace(np.log10(kp/6), np.log10(kp*6), 100)
        plt.loglog(kr, yref*(kr/kp)**slope, "--", lw=1.0, alpha=0.7)
        plt.text(kp, yref*0.7, txt, fontsize=10)
        # x232-=0.5
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.title(r"$E(k)$ for different $s$")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which='both', alpha=0.3)
    os.makedirs(C.outdir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "E1D_vs_k_all_s.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "E1D_vs_k_all_s.pdf")); plt.close()

    # 2) Local slope α_est(k) vs analytic α(k)
    plt.figure(figsize=(7.2,5.2))
    for (s, k, E, alpha_loc, alpha_tar, dalpha) in results:
        plt.semilogx(k, alpha_loc, lw=1.7, label=fr"$s={s:g}$")
    # overlay analytic α(k) for two representative s (min and max)
    s_min = min(C.s_list); s_max = max(C.s_list)
    for s in (s_min, s_max):
        x = results[0][1] / C.k_break
        alpha_tar = C.alpha_low/(1+x**s) + C.alpha_high*(1 - 1/(1+x**s))
        # plt.semilogx(results[0][1], alpha_tar, ':', lw=2.0,
        #              label=fr"analytic $\alpha(k)$, $s={s:g}$")
    # plt.axhline(C.alpha_low, color='k', ls='--', lw=0.8)
    # plt.axhline(C.alpha_high, color='k', ls='--', lw=0.8)
    plt.axvline(C.k_break, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"local slope $d\log E/d\log k$")
    plt.title(r"$\alpha_{\rm est}(k)$")
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
    plt.title(r"transition sharpness")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "dalpha_dlnk_all_s.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "dalpha_dlnk_all_s.pdf")); plt.close()

    print(f"\nSaved figures → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()
