#!/usr/bin/env python3
"""
Four measures on a common x–axis χ ≡ 2 σ_Φ λ² (LP16-like):
  1) PSA slope of P(k;λ)          [2D PSD slope of the polarization map]
  2) PSA slope of ∂P/∂(λ²)(k;λ)   [2D PSD slope of the derivative map]
  3) PFA variance ⟨|P(λ)|²⟩       [one-point statistic]
  4) Directional spectrum slope   [2D PSD slope of |Â|²+|B̂|² with A=cos 2χ, B=sin 2χ]

• x–axis is χ = 2 σ_Φ λ² with 0 < χ < 20 (you can set the range via flags)
• We internally calibrate φ so that σ_Φ=1, hence χ = 2 λ² in code units
• Break band (expected) is highlighted at 1 ≤ χ ≤ 3

Geometry options:
  - mixed:     emission and Faraday rotation co-exist along the LOS
  - separated: external Faraday screen (emission slab + thin screen)

Reference: Lazarian & Pogosyan 2016 (LP16). This script is aligned with their mixed/separated
setups, uses the midpoint (z+½) convention for the mixed case phase to avoid λ→0 bias,
and follows your PSA ring-binning+fits.
"""
from __future__ import annotations
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

# -----------------------------
# Keys and config
# -----------------------------
@dataclass
class FieldKeys:
    Bx: str = "i_mag_field"
    By: str = "j_mag_field"
    Bz: str = "k_mag_field"
    ne: str = "gas_density"

@dataclass
class Config:
    gamma: float = 2.0                 # LP16 emissivity index
    use_lp16_amp: bool = False         # if True: |B⊥|^{(γ-3)/2} factor (LP16); else pure quadratic
    los_axis: int = 2                  # integrate along this axis
    emit_frac: float = 0.9             # separated geometry: fraction of depth for emit slab
    screen_frac: float = 0.1           # separated geometry: screen thickness fraction
    ring_bins: int = 48                # PSA bin count
    k_min: float = 6.0                 # PSA min k for fitting/binning
    pad: int = 1                       # zero-padding factor (1=no pad)
    apodize: bool = True               # Hann window
    fit_kmin: float = 4.0              # slope-fit lower bound (k)
    fit_kmax: float = 25.0             # slope-fit upper bound (k)
    seed_roll: int = 0                 # if nonzero, can roll φ along z per-LOS (decorrelate Pi–Φ)

# -----------------------------
# I/O
# -----------------------------

def load_fields(h5_path: str, keys: FieldKeys):
    with h5py.File(h5_path, "r") as f:
        Bx = f[keys.Bx][...].astype(np.float64)
        By = f[keys.By][...].astype(np.float64)
        Bz = f[keys.Bz][...].astype(np.float64)
        ne = f[keys.ne][...].astype(np.float64)
    return Bx, By, Bz, ne

# -----------------------------
# Emissivity and Faraday ingredients
# -----------------------------

def emissivity(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0, use_lp16_amp: bool = False) -> np.ndarray:
    """Polarized emissivity.
    If use_lp16_amp:  P_i = (Bx+iBy)^2 * |B⊥|^{(γ-3)/2} (LP16)
    Else:             P_i = (Bx+iBy)^2 (pure quadratic)
    """
    Pi = (Bx + 1j*By)**2
    if not use_lp16_amp:
        return Pi
    Bp2 = Bx*Bx + By*By
    eps = np.finfo(Bp2.dtype).tiny
    amp = np.power(np.maximum(Bp2, eps), 0.5*(gamma-3.0))
    return Pi * amp


def faraday_density(ne: np.ndarray, Bpar: np.ndarray, C: float = 1.0) -> np.ndarray:
    return C * ne * Bpar


def move_los(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(arr, axis, 0)


def midphase_mixed(phi_los: np.ndarray, dz: float) -> np.ndarray:
    """Return Φ(z+½) along the LOS: cumulative to cell center.
    Φ(z+½) = (Σ_{m=0}^{z} φ[m])·dz - ½ φ[z]·dz
    """
    Phi_cum = np.cumsum(phi_los*dz, axis=0)
    return Phi_cum - 0.5*phi_los*dz

# -----------------------------
# P(λ) and dP/d(λ²) maps
# -----------------------------

def P_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam2: float, los_axis: int) -> np.ndarray:
    Pi_los = move_los(Pi, los_axis)
    phi_los = move_los(phi, los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_half = midphase_mixed(phi_los, dz)
    phase = np.exp(2j * lam2 * Phi_half)
    return (Pi_los * phase).sum(axis=0) * dz


def dP_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam2: float, los_axis: int) -> np.ndarray:
    Pi_los = move_los(Pi, los_axis)
    phi_los = move_los(phi, los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_half = midphase_mixed(phi_los, dz)
    phase = np.exp(2j * lam2 * Phi_half)
    return (2j * Pi_los * Phi_half * phase).sum(axis=0) * dz


def P_map_separated(Pi: np.ndarray, phi: np.ndarray, lam2: float, los_axis: int,
                    emit_frac: float = 0.9, screen_frac: float = 0.1) -> np.ndarray:
    Pi_los = move_los(Pi, los_axis)
    phi_los = move_los(phi, los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    eN = max(1, int(emit_frac * Nz))
    sN = max(1, int(screen_frac * Nz))
    e0, e1 = 0, eN
    s0, s1 = max(e1, Nz - sN), Nz
    P_emit = Pi_los[e0:e1, :, :].sum(axis=0) * dz
    Phi_screen = phi_los[s0:s1, :, :].sum(axis=0) * dz
    return P_emit * np.exp(2j * lam2 * Phi_screen)


def dP_map_separated(Pi: np.ndarray, phi: np.ndarray, lam2: float, los_axis: int,
                      emit_frac: float = 0.9, screen_frac: float = 0.1) -> np.ndarray:
    Pi_los = move_los(Pi, los_axis)
    phi_los = move_los(phi, los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    eN = max(1, int(emit_frac * Nz))
    sN = max(1, int(screen_frac * Nz))
    e0, e1 = 0, eN
    s0, s1 = max(e1, Nz - sN), Nz
    P_emit = Pi_los[e0:e1, :, :].sum(axis=0) * dz
    Phi_screen = phi_los[s0:s1, :, :].sum(axis=0) * dz
    P = P_emit * np.exp(2j * lam2 * Phi_screen)
    return 2j * Phi_screen * P

# -----------------------------
# PSA (ring-averaged 2D PSD) and slope fits
# -----------------------------

def psa_of_map(P_map: np.ndarray, ring_bins: int = 48, pad: int = 1, apodize: bool = True,
               k_min: float = 6.0, min_counts_per_ring: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    Y = P_map - P_map.mean()
    if apodize:
        wy = np.hanning(Y.shape[0]); wx = np.hanning(Y.shape[1])
        Y = Y * np.outer(wy, wx)
    if pad > 1:
        Yp = np.zeros((pad*Y.shape[0], pad*Y.shape[1]), dtype=complex)
        Yp[:Y.shape[0], :Y.shape[1]] = Y
        Y = Yp
    F = np.fft.fftshift(np.fft.fft2(Y))
    P2 = (F * np.conj(F)).real
    ky = np.fft.fftshift(np.fft.fftfreq(P2.shape[0])) * P2.shape[0]
    kx = np.fft.fftshift(np.fft.fftfreq(P2.shape[1])) * P2.shape[1]
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY).ravel()
    k_max = min(P2.shape) / 3.5
    bins = np.geomspace(max(1.0, k_min), k_max, ring_bins + 1)
    lab = np.digitize(KR, bins) - 1
    counts = np.array([(lab == i).sum() for i in range(ring_bins)])
    Ek = np.array([P2.ravel()[lab == i].mean() if counts[i] >= min_counts_per_ring else np.nan
                   for i in range(ring_bins)])
    kcen = np.sqrt(bins[:-1] * bins[1:])
    msk = np.isfinite(Ek) & (kcen >= k_min) & (kcen <= k_max)
    if msk.sum() < 10:
        kr = np.floor(np.hypot(KX, KY) + 0.5).astype(int)
        kmax_int = int(min(P2.shape) / 2)
        w = P2.ravel(); idx = kr.ravel()
        Ek_int = np.bincount(idx, weights=w, minlength=kmax_int + 1)
        Nk_int = np.bincount(idx, minlength=kmax_int + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Ek_int = Ek_int / np.maximum(Nk_int, 1)
        kvec = np.arange(kmax_int + 1)
        ok = (kvec >= int(k_min)) & (kvec <= int(k_max)) & (Nk_int > 25)
        return kvec[ok], Ek_int[ok]
    return kcen[msk], Ek[msk]


def fit_log_slope_with_bounds(k: np.ndarray, E: np.ndarray, kmin: float, kmax: float):
    k = np.asarray(k); E = np.asarray(E)
    sel = np.isfinite(k) & np.isfinite(E) & (k>0) & (E>0)
    sel &= (k >= kmin) & (k <= kmax)
    k = k[sel]; E = E[sel]
    if k.size < 10:
        return np.nan, np.nan
    lk, lE = np.log10(k), np.log10(E)
    m, a = np.polyfit(lk, lE, 1)
    return float(m), float(a)

# -----------------------------
# Directional spectrum of the polarization angle field
# -----------------------------

def directional_spectrum_of_map(P_map: np.ndarray, ring_bins: int = 24, pad: int = 1,
                                apodize: bool = True, k_min: float = 6.0,
                                min_counts_per_ring: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    Q = P_map.real; U = P_map.imag
    amp = np.hypot(Q, U)
    eps = np.finfo(amp.dtype).eps
    amp = np.maximum(amp, eps)
    A = Q / amp   # cos 2χ
    B = U / amp   # sin 2χ
    def _prep(M):
        Y = M - np.nanmean(M)
        if apodize:
            wy = np.hanning(Y.shape[0]); wx = np.hanning(Y.shape[1])
            Y = Y * np.outer(wy, wx)
        if pad > 1:
            Yp = np.zeros((pad*Y.shape[0], pad*Y.shape[1]), dtype=float)
            Yp[:Y.shape[0], :Y.shape[1]] = Y
            Y = Yp
        return Y
    A = _prep(A); B = _prep(B)
    FA = np.fft.fftshift(np.fft.fft2(A))
    FB = np.fft.fftshift(np.fft.fft2(B))
    P2 = (FA*np.conj(FA)).real + (FB*np.conj(FB)).real
    ky = np.fft.fftshift(np.fft.fftfreq(P2.shape[0])) * P2.shape[0]
    kx = np.fft.fftshift(np.fft.fftfreq(P2.shape[1])) * P2.shape[1]
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY).ravel()
    k_max = min(P2.shape) / 3.5
    bins = np.geomspace(max(1.0, k_min), k_max, ring_bins + 1)
    lab = np.digitize(KR, bins) - 1
    counts = np.array([(lab == i).sum() for i in range(ring_bins)])
    Ek = np.array([P2.ravel()[lab == i].mean() if counts[i] >= min_counts_per_ring else np.nan
                   for i in range(ring_bins)])
    kcen = np.sqrt(bins[:-1] * bins[1:])
    msk = np.isfinite(Ek) & (kcen >= k_min) & (kcen <= k_max)
    if msk.sum() < 10:
        kr = np.floor(np.hypot(KX, KY) + 0.5).astype(int)
        kmax_int = int(min(P2.shape) / 2)
        w = P2.ravel(); idx = kr.ravel()
        Ek_int = np.bincount(idx, weights=w, minlength=kmax_int + 1)
        Nk_int = np.bincount(idx, minlength=kmax_int + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Ek_int = Ek_int / np.maximum(Nk_int, 1)
        kvec = np.arange(kmax_int + 1)
        ok = (kvec >= int(k_min)) & (kvec <= int(k_max)) & (Nk_int > 25)
        return kvec[ok], Ek_int[ok]
    return kcen[msk], Ek[msk]

# -----------------------------
# Core routine: compute all four measures on χ-grid
# -----------------------------

def compute_sigma_Phi(phi: np.ndarray, los_axis: int) -> float:
    phi_los = move_los(phi, los_axis)
    Nz = phi_los.shape[0]
    dz = 1.0/float(Nz)
    Phi_tot = phi_los.sum(axis=0) * dz  # total Faraday depth per LOS
    return float(Phi_tot.std())


def maybe_roll_phi(phi: np.ndarray, los_axis: int, seed: int = 0) -> np.ndarray:
    if seed == 0:
        return phi
    rng = np.random.default_rng(seed)
    arr = move_los(phi, los_axis)
    out = np.empty_like(arr)
    Nz, Ny, Nx = arr.shape
    for j in range(Ny):
        for i in range(Nx):
            s = int(rng.integers(0, Nz))
            out[:, j, i] = np.roll(arr[:, j, i], s)
    return move_los(out, 0 if los_axis==0 else (1 if los_axis==1 else 2))


def four_measures(h5_path: str,
                  geometry: str = "mixed",
                  gamma: float = 2.0,
                  use_lp16_amp: bool = False,
                  los_axis: int = 2,
                  chi_min: float = 1e-3,
                  chi_max: float = 20.0,
                  n_chi: int = 60,
                  ring_bins: int = 48,
                  k_min: float = 6.0,
                  fit_kmin: float = 4.0,
                  fit_kmax: float = 25.0,
                  pad: int = 1,
                  apodize: bool = True,
                  seed_roll: int = 0):
    keys = FieldKeys()
    Bx, By, Bz, ne = load_fields(h5_path, keys)

    # Emissivity and B_parallel per LOS
    Pi = emissivity(Bx, By, gamma=gamma, use_lp16_amp=use_lp16_amp)
    if   los_axis == 0: Bpar = Bx
    elif los_axis == 1: Bpar = By
    else:               Bpar = Bz

    # Start with C=1, compute σ_Φ, then rescale phi→phi/σ_Φ so that σ_Φ=1
    phi = faraday_density(ne, Bpar, C=1.0)
    sigma_Phi0 = compute_sigma_Phi(phi, los_axis)
    C = 1.0 / max(sigma_Phi0, 1e-30)
    phi = phi * C
    sigma_Phi = compute_sigma_Phi(phi, los_axis)  # should be ≈ 1

    # (Optional) decorrelate Π–Φ by random LOS roll to sharpen -2 slopes
    phi = maybe_roll_phi(phi, los_axis, seed=seed_roll)

    # χ grid (strictly within (0, 20)) and λ² from χ = 2σ_Φ λ² (σ_Φ≈1 here)
    chi = np.geomspace(max(1e-6, chi_min), min(chi_max, 20.0), n_chi)
    lam2 = chi / (2.0 * sigma_Phi)

    # slots for measures
    pfa_var = np.empty_like(chi)
    dvar    = np.empty_like(chi)
    mP      = np.empty_like(chi)
    mD      = np.empty_like(chi)
    mDir    = np.empty_like(chi)

    for t, l2 in enumerate(lam2):
        if geometry.lower().startswith("mix"):
            Pmap  = P_map_mixed(Pi, phi, l2, los_axis)
            dPmap = dP_map_mixed(Pi, phi, l2, los_axis)
        else:
            Pmap  = P_map_separated(Pi, phi, l2, los_axis)
            dPmap = dP_map_separated(Pi, phi, l2, los_axis)

        # (3) PFA & derivative variances
        pfa_var[t] = np.mean(np.abs(Pmap)**2)
        dvar[t]    = np.mean(np.abs(dPmap)**2)

        # (1) PSA slopes (P and dP)
        kP, EP  = psa_of_map(Pmap,  ring_bins=ring_bins, pad=pad, apodize=apodize,
                              k_min=k_min, min_counts_per_ring=10)
        kD, ED  = psa_of_map(dPmap, ring_bins=ring_bins, pad=pad, apodize=apodize,
                              k_min=k_min, min_counts_per_ring=10)
        mP[t], _ = fit_log_slope_with_bounds(kP, EP, fit_kmin, fit_kmax)
        mD[t], _ = fit_log_slope_with_bounds(kD, ED, fit_kmin, fit_kmax)

        # (4) Directional spectrum slope
        kR, EDir = directional_spectrum_of_map(Pmap, ring_bins=24, pad=pad,
                                               apodize=apodize, k_min=k_min,
                                               min_counts_per_ring=8)
        mDir[t], _ = fit_log_slope_with_bounds(kR, EDir, fit_kmin, fit_kmax)

    return {
        'chi': chi,
        'lam2': lam2,
        'sigma_Phi': sigma_Phi,
        'pfa_var': pfa_var,
        'dvar': dvar,
        'mP': mP,
        'mD': mD,
        'mDir': mDir,
    }

# -----------------------------
# Plotting
# -----------------------------

def plot_four_measures(res: dict, geometry: str, out_png: Optional[str] = None,
                       shade_break: Tuple[float,float] = (1.0, 3.0),
                       normalize_pfa: bool = True):
    chi = res['chi']
    pfa = res['pfa_var'].copy()
    dvr = res['dvar'].copy()
    mP  = res['mP']
    mD  = res['mD']
    mR  = res['mDir']

    # Normalize PFA and derivative variances for readability (take log10)
    if normalize_pfa:
        pfa /= max(pfa[0], 1e-300)
        dvr /= max(dvr[0], 1e-300)
    lpfa = np.log10(pfa)
    ldvr = np.log10(dvr)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True)

    # Panel A: k-slopes
    ax = axes[0]
    ax.semilogx(chi, mP,  'o-', ms=3, label=r"PSA slope of $P$ (2D PSD)")
    ax.semilogx(chi, mD,  's-', ms=3, label=r"PSA slope of $\partial P/\partial\lambda^2$")
    ax.semilogx(chi, mR,  'd-', ms=3, label=r"Directional slope $|\hat{\cos2\chi}|^2+|\hat{\sin2\chi}|^2$")
    ax.axvspan(shade_break[0], shade_break[1], color='k', alpha=0.08, lw=0)
    ax.axhline(0, color='k', lw=0.6, alpha=0.35)
    ax.set_ylabel(r"slope $m$ in $E(k)\propto k^{\,m}$")
    ax.set_title(f"Four measures on χ-axis (geometry: {geometry},  χ=2\,σ_Φ\,λ²)")
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='best')

    # Panel B: PFA & derivative variances (log10)
    ax = axes[1]
    ax.semilogx(chi, lpfa, '-',  lw=1.4, color='gray', label=r"$\log_{10}\langle |P|^2\rangle$ (norm.)")
    ax.semilogx(chi, ldvr, '--', lw=1.2, color='tab:blue', label=r"$\log_{10}\langle |\partial P/\partial\lambda^2|^2\rangle$ (norm.)")
    ax.axvspan(shade_break[0], shade_break[1], color='k', alpha=0.08, lw=0)
    ax.set_xlabel(r"$\chi \equiv 2\,\sigma_\Phi\,\lambda^2$")
    ax.set_ylabel(r"log$_{10}$ variance (arb.)")
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='best')

    fig.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Saved figure to: {out_png}")
    else:
        plt.show()

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare four measures on χ = 2 σ_Φ λ² axis (0<χ<20)")
    ap.add_argument('--h5', required=True, help='Path to HDF5 cube')
    ap.add_argument('--geometry', choices=['mixed','separated'], default='mixed')
    ap.add_argument('--los-axis', type=int, default=2)
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--lp16-amp', action='store_true', help='Use LP16 amplitude factor |B⊥|^{(γ−3)/2}')
    ap.add_argument('--chi-min', type=float, default=1e-3)
    ap.add_argument('--chi-max', type=float, default=20.0)
    ap.add_argument('--n-chi', type=int, default=60)
    ap.add_argument('--ring-bins', type=int, default=48)
    ap.add_argument('--k-min', type=float, default=6.0)
    ap.add_argument('--fit-kmin', type=float, default=4.0)
    ap.add_argument('--fit-kmax', type=float, default=25.0)
    ap.add_argument('--pad', type=int, default=1)
    ap.add_argument('--no-apod', action='store_true', help='Disable Hann window')
    ap.add_argument('--roll', type=int, default=0, help='Random-roll φ along LOS per-LOS (0=off)')
    ap.add_argument('--out', default=None, help='Output PNG path')
    args = ap.parse_args()

    res = four_measures(
        h5_path=args.h5,
        geometry=args.geometry,
        gamma=args.gamma,
        use_lp16_amp=args.lp16_amp,
        los_axis=args.los_axis,
        chi_min=args.chi_min,
        chi_max=min(args.chi_max, 20.0),
        n_chi=args.n_chi,
        ring_bins=args.ring_bins,
        k_min=args.k_min,
        fit_kmin=args.fit_kmin,
        fit_kmax=args.fit_kmax,
        pad=args.pad,
        apodize=(not args.no_apod),
        seed_roll=args.roll,
    )

    print(f"Measured σ_Φ (after rescale) ≈ {res['sigma_Phi']:.3f} (target 1.0)")
    print("x-axis is χ = 2 σ_Φ λ² with 0<χ<20; break band shaded at 1≲χ≲3.")

    out = args.out or f"four_measures_{args.geometry}.png"
    plot_four_measures(res, geometry=args.geometry, out_png=out)

if __name__ == '__main__':
    main()
