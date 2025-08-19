#!/usr/bin/env python3
"""
Two-slope spectrum → observed 2D spectra (direct 2D and 3D→2D projection)
===========================================================================

We construct synthetic, statistically isotropic random fields whose *ring-averaged*
(1D) spectrum has two power-law segments: a low-k rising slope +3/2 and a high-k
falling slope −5/3, with a smooth break at k_break.

We then:
  • (A) build the field directly in 2D from the *target 2D spectral density* P2D(k),
  • (B) build a 3D field from a *target 3D spectral density* P3D(k), project along z
        to a 2D "sky" map, and compare the observed 2D spectra.
For each case we compute:
  - isotropic ring-averaged 2D spectral density  ⟨|F|^2⟩(k) ≡ P2D_obs(k),
  - the corresponding 1D (shell) spectrum       E1D(k) = 2π k P2D_obs(k),
and fit slopes below and above the break.

Conventions
-----------
- k-units: "cycles per pixel" (from numpy.fft.fftfreq). Slopes are invariant to
  the 2π choice; we keep cycles for simplicity.
- To target a *1D shell* slope α, use P2D ∝ k^(α−1).
- For the 3D case, to target a *3D 1D-shell* slope α_3D for the underlying field,
  use P3D ∝ k^(α_3D−2). For pure power laws, projection preserves the α in the
  observed 2D shell spectrum.

No argparse; edit CONFIG below.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft2, ifft2, fftfreq, fftn, ifftn

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # grid sizes
    nx: int = 512
    ny: int = 512
    nz: int = 256          # used only for 3D synthesis
    dx: float = 1.0        # pixel size (arbitrary units)
    seed: int = 42

    # target 1D (ring-averaged) slopes
    alpha_low: float = +1.5           # +3/2 (rising)
    alpha_high: float = -5.0/3.0      # -5/3 (falling)
    k_break_cyc: float = 0.06         # in cycles/dx (choose inside inertial range)
    transition_sharpness: float = 8.0 # larger = sharper transition in log-k

    # plotting / fitting ranges (cycles/dx)
    kfit_low: tuple = (0.01, 0.04)    # fit low-k slope here
    kfit_high: tuple = (0.08, 0.20)   # fit high-k slope here

    # output
    outdir: str = "fig_two_slopes"
    dpi: int = 160

CFG = Config()

# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def kgrid_2d(nx, ny, dx=1.0):
    ky = fftfreq(ny, d=dx)
    kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    return np.hypot(KY, KX)

def kgrid_3d(nx, ny, nz, dx=1.0):
    ky = fftfreq(ny, d=dx)
    kx = fftfreq(nx, d=dx)
    kz = fftfreq(nz, d=dx)
    KY, KX, KZ = np.meshgrid(ky, kx, kz, indexing="ij")
    return np.sqrt(KY**2 + KX**2 + KZ**2)

def smooth_piecewise_power(k, k_break, alpha_low, alpha_high, sharp=8.0, eps=1e-12):
    """
    Return a *2D spectral density* P2D(k) that yields a ring-averaged (1D) slope α.
    For 2D isotropic fields: E1D(k) = 2π k P2D(k) ⇒ P2D ∝ k^(α−1).
    We blend low/high exponents smoothly in log-k.

    For the *3D* spectral density P3D when targeting a *3D* 1D-shell slope α_3D,
    use P3D ∝ k^(α_3D−2). That can also be produced by this routine by passing
    exponents already converted to the *density* slope.

    Parameters
    ----------
    k : array (>=0)
    k_break : scalar
    alpha_low, alpha_high : desired *1D shell* slopes for 2D (or preconverted)
    sharp : transition sharpness in log-k
    """
    k = np.asarray(k)
    ksafe = np.maximum(k, eps)
    # logistic in log10(k)
    t = 1.0 / (1.0 + (ksafe / k_break)**sharp)  # ~1 at k<<kb, ~0 at k>>kb
    # For 2D spectral density:
    gamma_low  = alpha_low  - 1.0
    gamma_high = alpha_high - 1.0
    # amplitude continuity at k_break:
    A_low  = (ksafe / k_break)**(gamma_low)
    A_high = (ksafe / k_break)**(gamma_high)
    P = t * A_low + (1.0 - t) * A_high
    P[k == 0.0] = 0.0  # no DC power
    return P

def smooth_piecewise_density(k, k_break, dens_low, dens_high, sharp=8.0, eps=1e-12):
    """
    Blend *density* exponents directly: return D(k) ∝ k^(dens_low) at k<<kb
    and ∝ k^(dens_high) at k>>kb (smooth in log-k).
    """
    k = np.asarray(k)
    ksafe = np.maximum(k, eps)
    t = 1.0 / (1.0 + (ksafe / k_break)**sharp)
    A_low  = (ksafe / k_break)**(dens_low)
    A_high = (ksafe / k_break)**(dens_high)
    D = t * A_low + (1.0 - t) * A_high
    D[k == 0.0] = 0.0
    return D

def isotropic_ring_spectrum(P2D, dx=1.0, nbins=240, kmin=1e-4, kmax_frac=1.0):
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
    nonz = cnts > 0
    prof[nonz] = sums[nonz] / cnts[nonz]
    kcent = 0.5*(bins[1:]+bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def fit_loglog(x, y, xmin, xmax):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0) & (x>=xmin) & (x<=xmax)
    if np.count_nonzero(m) < 5: return np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(np.exp(b))

# ──────────────────────────────────────────────────────────────────────
# Synthesis
# ──────────────────────────────────────────────────────────────────────

def synthesize_2d_from_P2D(nx, ny, dx, P2D_func, seed=0):
    """
    Create a real 2D field whose Fourier spectrum is shaped by P2D_func(K).
    We start from *real-space* white noise (real) to preserve Hermitian symmetry.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(ny, nx))
    Wk = fft2(w)  # complex with Hermitian symmetry
    K = kgrid_2d(nx, ny, dx)
    P = P2D_func(K)
    Ak = np.sqrt(np.maximum(P, 0.0))
    Fk = Wk * Ak
    f = ifft2(Fk).real
    f -= f.mean()
    f /= (f.std(ddof=0) + 1e-12)
    return f

def synthesize_3d_from_P3D(nx, ny, nz, dx, P3D_func, seed=0):
    """
    Create a real 3D field with target 3D spectral density P3D_func(K).
    Start from real-space white noise → FFT → shape → IFFT.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(ny, nx, nz))
    Wk = fftn(w)
    K = kgrid_3d(nx, ny, nz, dx)
    P = P3D_func(K)  # density in k-space
    Ak = np.sqrt(np.maximum(P, 0.0))
    Fk = Wk * Ak
    f3 = ifftn(Fk).real
    f3 -= f3.mean()
    f3 /= (f3.std(ddof=0) + 1e-12)
    return f3

def project_to_2d_along_z(field3d):
    """Simple LOS projection: sum along z (you can also average)."""
    return field3d.sum(axis=2)

# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

def analyze_map_2d(Map2D, dx, label, kfit_low, kfit_high, outdir, dpi=160):
    Fk = fft2(Map2D)
    P2D_obs = (Fk * np.conj(Fk)).real
    k1d, Pk = isotropic_ring_spectrum(P2D_obs, dx=dx, nbins=260, kmin=1e-4, kmax_frac=1.0)
    E1D = 2*np.pi * k1d * Pk

    # Fit slopes
    aP_lo, _ = fit_loglog(k1d, Pk, *kfit_low)
    aP_hi, _ = fit_loglog(k1d, Pk, *kfit_high)
    aE_lo, _ = fit_loglog(k1d, E1D, *kfit_low)
    aE_hi, _ = fit_loglog(k1d, E1D, *kfit_high)

    print(f"[{label}] fitted slopes (cycles/dx):")
    print(f"  P2D(k)    slope low  ≈ {aP_lo:+.3f},   high ≈ {aP_hi:+.3f}   (expect α−1)")
    print(f"  E1D(k)    slope low  ≈ {aE_lo:+.3f},   high ≈ {aE_hi:+.3f}   (expect α)")

    # Plots
    os.makedirs(outdir, exist_ok=True)

    # P2D
    plt.figure(figsize=(6.5,4.8))
    m = (k1d>0) & np.isfinite(Pk) & (Pk>0)
    plt.loglog(k1d[m], Pk[m], lw=1.6, label="measured")
    for rng, slope, name in [(kfit_low, aP_lo, "low-fit"), (kfit_high, aP_hi, "high-fit")]:
        if np.isfinite(slope):
            kref = np.sqrt(rng[0]*rng[1])
            pref = np.interp(kref, k1d[m], Pk[m])
            kr = np.logspace(np.log10(rng[0]), np.log10(rng[1]), 50)
            plt.loglog(kr, pref*(kr/kref)**slope, "--", lw=1.0, label=f"{name}: k^{slope:+.2f}")
    plt.xlabel(r"$k$ [cycles/dx]"); plt.ylabel(r"$P_{2D}(k)$")
    plt.title(f"{label}: ring-avg 2D spectrum")
    plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{label}_P2D.png"), dpi=dpi); plt.close()

    # E1D
    plt.figure(figsize=(6.5,4.8))
    m = (k1d>0) & np.isfinite(E1D) & (E1D>0)
    plt.loglog(k1d[m], E1D[m], lw=1.6, label="measured")
    for rng, slope, name in [(kfit_low, aE_lo, "low-fit"), (kfit_high, aE_hi, "high-fit")]:
        if np.isfinite(slope):
            kref = np.sqrt(rng[0]*rng[1])
            pref = np.interp(kref, k1d[m], E1D[m])
            kr = np.logspace(np.log10(rng[0]), np.log10(rng[1]), 50)
            plt.loglog(kr, pref*(kr/kref)**slope, "--", lw=1.0, label=f"{name}: k^{slope:+.2f}")
    plt.xlabel(r"$k$ [cycles/dx]"); plt.ylabel(r"$E_{1D}(k)=2\pi k\,P_{2D}(k)$")
    plt.title(f"{label}: 1D (ring) spectrum")
    plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{label}_E1D.png"), dpi=dpi); plt.close()

    return dict(k1d=k1d, Pk=Pk, E1D=E1D, slopes=dict(P_low=aP_lo, P_high=aP_hi, E_low=aE_lo, E_high=aE_hi))

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(C=CFG):
    np.random.seed(C.seed)

    # ---------- 2D target: build P2D to achieve 1D α = (+3/2, -5/3)
    def P2D_target(K):
        return smooth_piecewise_power(K, C.k_break_cyc, C.alpha_low, C.alpha_high, sharp=C.transition_sharpness)

    f2 = synthesize_2d_from_P2D(C.nx, C.ny, C.dx, P2D_target, seed=C.seed)
    res2 = analyze_map_2d(f2, C.dx, "2D_direct", C.kfit_low, C.kfit_high, C.outdir, dpi=C.dpi)

    # Print expectations:
    print("\nExpected (from construction):")
    print(f"  E1D low-slope  ≈ {C.alpha_low:+.3f},  high-slope ≈ {C.alpha_high:+.3f}")
    print(f"  P2D low-slope  ≈ {C.alpha_low-1:+.3f}, high-slope ≈ {C.alpha_high-1:+.3f}")

    # ---------- 3D target & projection:
    # To target 3D shell slope α_3D = α (same desired values), set P3D ∝ k^(α−2)
    dens_low_3D  = C.alpha_low  - 2.0   # for P3D low-k exponent
    dens_high_3D = C.alpha_high - 2.0   # for P3D high-k exponent

    def P3D_target(K):
        return smooth_piecewise_density(K, C.k_break_cyc, dens_low_3D, dens_high_3D, sharp=C.transition_sharpness)

    f3 = synthesize_3d_from_P3D(C.nx, C.ny, C.nz, C.dx, P3D_target, seed=C.seed+1)
    sky = project_to_2d_along_z(f3)
    sky -= sky.mean(); sky /= (sky.std(ddof=0) + 1e-12)

    res3 = analyze_map_2d(sky, C.dx, "3D_projected", C.kfit_low, C.kfit_high, C.outdir, dpi=C.dpi)

    print("\nProjection theory (pure power laws): 3D→2D preserves the 1D shell slope α.")
    print("Compare measured 2D ring slopes from '3D_projected' with the targets above.")

    print("\nSaved figures in:", os.path.abspath(C.outdir))

if __name__ == "__main__":
    main()
