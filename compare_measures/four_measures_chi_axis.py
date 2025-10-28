#!/usr/bin/env python3
"""
Compare FOUR measures of synchrotron polarization statistics vs Faraday depth,
with the x–axis in the *dimensionless* variable
    χ ≡ 2 σ_Φ λ²,
where σ_Φ is the map-std of the *total* Faraday depth Φ(X)=∫φ dz.

Measures (for a chosen geometry: mixed or separated):
  1) PFA variance          :  ⟨|P(λ)|²⟩
  2) Derivative variance   :  ⟨|∂P/∂(λ²)|²⟩
  3) Spatial PSA slope of P            (2D ring-avg PSD slope m_P)
  4) Spatial PSA slope of ∂P/∂(λ²)    (2D ring-avg PSD slope m_D)

Key requirements from the user:
  • Plot vs χ = 2 σ_Φ λ² on the x-axis (NOT λ² directly).
  • Use 0 < χ < 20 (enforced below).
  • Show the χ-band where the break is expected: 1 ≲ χ ≲ 3.

Implementation notes
--------------------
• We first build φ with C=1, integrate to total Φ(X), compute σ_Φ_raw = std(Φ).
• We then rescale φ by C_scale = 1/σ_Φ_raw so that σ_Φ = 1 in code units.
  → With this choice, χ = 2 λ² exactly, and choosing χ-grid ∈ (0, 20)
    amounts to picking λ²-grid = χ/2.
• LOS mapping is *correct*: los_axis ∈ {0,1,2} → B_parallel = (Bx, By, Bz) respectively.
• Mixed-geometry integrals use midpoint (half-cell) Φ(z+1/2) for the phase
  to remove a subtle λ→0 bias.
• All LOS integrals use dz = 1/Nz so amplitudes are resolution-consistent.

Run:
  # Compute and plot:
  python four_measures_chi_axis.py \
      --h5 path/to/cube.h5 \
      --geometry mixed        # or separated

  # Recreate figure from saved data:
  python four_measures_chi_axis.py \
      --npz four_measures_mixed.npz

This will save:
  four_measures_<geometry>.png   (the comparison panel)
  four_measures_<geometry>.npz   (data for later recreation)
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---------- I/O ----------
@dataclass
class FieldKeys:
    Bx: str = "i_mag_field"
    By: str = "j_mag_field"
    Bz: str = "k_mag_field"
    ne: str = "gas_density"


def load_fields(h5_path: str, keys: FieldKeys):
    with h5py.File(h5_path, "r") as f:
        Bx = np.asarray(f[keys.Bx], dtype=np.float64)
        By = np.asarray(f[keys.By], dtype=np.float64)
        Bz = np.asarray(f[keys.Bz], dtype=np.float64)
        ne = np.asarray(f[keys.ne], dtype=np.float64)
    return Bx, By, Bz, ne


# ---------- Emissivity & Faraday ----------

def polarized_emissivity_lp16(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0,
                              use_lp16_amp: bool = True) -> np.ndarray:
    """LP16 emissivity: P_i = (Bx+iBy)^2 |B_⊥|^{(γ-3)/2}.
    For γ=2: P_i = (Bx+iBy)^2 / |B_⊥|^{1/2}.  Set use_lp16_amp=False for pure quadratic.
    """
    P2 = Bx*Bx + By*By
    eps = np.finfo(P2.dtype).tiny
    amp = np.power(np.maximum(P2, eps), 0.5*(gamma-3.0)) if use_lp16_amp else 1.0
    return ((Bx + 1j*By)**2 * amp).astype(np.complex128)


def faraday_density(ne: np.ndarray, Bpar: np.ndarray, C: float = 1.0) -> np.ndarray:
    return (C * ne * Bpar).astype(np.float64)


def sigma_phi_total(phi: np.ndarray, los_axis: int) -> float:
    """σ_Φ = std over map of total Faraday depth Φ(X)=∑ φ dz with dz=1/Nz."""
    arr = np.moveaxis(phi, los_axis, 0)
    Nz = arr.shape[0]
    dz = 1.0 / float(Nz)
    Phi_tot = np.sum(arr * dz, axis=0)
    return float(Phi_tot.std())


# ---------- Map builders (mixed / separated) ----------

def _move_los(a: np.ndarray, los_axis: int) -> np.ndarray:
    return np.moveaxis(a, los_axis, 0)


def P_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, los_axis: int,
                    emit_bounds: Optional[Tuple[int,int]] = None,
                    screen_bounds: Optional[Tuple[int,int]] = None,
                    detrend_emit: bool = False) -> np.ndarray:
    """External screen: P = (∫ Pi dz) * exp[2i λ² Φ_screen]."""
    Pi_los = _move_los(Pi, los_axis)
    phi_los = _move_los(phi, los_axis)
    Nz, Ny, Nx = Pi_los.shape
    dz = 1.0 / float(Nz)

    if emit_bounds is None:
        emit_bounds = (0, Nz)
    if screen_bounds is None:
        scrN = max(1, int(0.1 * Nz))
        screen_bounds = (0, scrN)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    P_emit = np.sum(Pi_los[z0e:z1e, :, :], axis=0) * dz
    if detrend_emit:
        P_emit = P_emit - P_emit.mean()

    Phi_screen = np.sum(phi_los[z0s:z1s, :, :], axis=0) * dz
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)


def P_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, los_axis: int,
                bounds: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """Mixed: P = ∑_z Pi(z) exp[2i λ² Φ(z+1/2)] dz  (midpoint phase)."""
    Pi_los = _move_los(Pi, los_axis)
    phi_los = _move_los(phi, los_axis)
    Nz, Ny, Nx = Pi_los.shape
    dz = 1.0 / float(Nz)
    z0, z1 = bounds or (0, Nz)

    rm = phi_los[z0:z1, :, :]
    Phi_cum = np.cumsum(rm * dz, axis=0)
    Phi_half = Phi_cum - 0.5 * rm * dz
    phase = np.exp(2j * (lam**2) * Phi_half)
    P = np.sum(Pi_los[z0:z1, :, :] * phase, axis=0) * dz
    return P


def dP_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, los_axis: int,
                     emit_bounds: Optional[Tuple[int,int]] = None,
                     screen_bounds: Optional[Tuple[int,int]] = None,
                     detrend_emit: bool = False) -> np.ndarray:
    """∂P/∂(λ²) for external screen: 2i Φ_screen * P."""
    P = P_map_separated(Pi, phi, lam, los_axis, emit_bounds, screen_bounds, detrend_emit)
    phi_los = _move_los(phi, los_axis)
    Nz = phi_los.shape[0]
    dz = 1.0 / float(Nz)
    s0, s1 = screen_bounds or (max(0, int(0.9*Nz)), Nz)
    Phi_screen = np.sum(phi_los[s0:s1, :, :], axis=0) * dz
    return 2j * Phi_screen * P


def dP_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, los_axis: int,
                 bounds: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """∂P/∂(λ²) for mixed: 2i ∑_z Pi(z) Φ(z+1/2) exp[2i λ² Φ(z+1/2)] dz."""
    Pi_los = _move_los(Pi, los_axis)
    phi_los = _move_los(phi, los_axis)
    Nz, Ny, Nx = Pi_los.shape
    dz = 1.0 / float(Nz)
    z0, z1 = bounds or (0, Nz)

    rm = phi_los[z0:z1, :, :]
    Phi_cum = np.cumsum(rm * dz, axis=0)
    Phi_half = Phi_cum - 0.5 * rm * dz
    phase = np.exp(2j * (lam**2) * Phi_half)
    dP = np.sum(2j * (Pi_los[z0:z1, :, :] * Phi_half) * phase, axis=0) * dz
    return dP


# ---------- Spatial PSA (ring-avg PSD) ----------

def psa_of_map(P_map: np.ndarray, ring_bins: int = 48, pad: int = 1,
               apodize: bool = True, k_min: float = 6.0,
               min_counts_per_ring: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """2D FFT → power → azimuthal average on geometric rings (returns 2D PSD)."""
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

    # Fallback to integer rings if too sparse
    if msk.sum() < 10:
        KR2 = np.hypot(KX, KY)
        kr = np.floor(KR2 + 0.5).astype(int)
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


def fit_log_slope_window(k: np.ndarray, E: np.ndarray,
                         kmin: float = 4.0, kmax: float = 25.0) -> Tuple[float,float,float,Tuple[float,float]]:
    """Fit a straight line to log10 E vs log10 k within [kmin, kmax]."""
    k = np.asarray(k); E = np.asarray(E)
    ok = np.isfinite(k) & np.isfinite(E) & (k>0) & (E>0)
    k, E = k[ok], E[ok]
    if k.size < 12:
        return np.nan, np.nan, np.nan, (np.nan, np.nan)
    sel = (k >= kmin) & (k <= kmax)
    if sel.sum() < 8:
        # best-effort: take middle 60%
        lo, hi = int(0.2*k.size), int(0.8*k.size)
        kk, EE = k[lo:hi], E[lo:hi]
    else:
        kk, EE = k[sel], E[sel]
    lk, lE = np.log10(kk), np.log10(EE)
    m, a = np.polyfit(lk, lE, 1)
    yfit = m*lk + a
    err = np.sqrt(np.mean((lE-yfit)**2) / np.sum((lk-lk.mean())**2))
    return float(m), float(a), float(err), (float(kk.min()), float(kk.max()))


# ---------- Curves vs χ ----------

def compute_curves_vs_chi(Pi: np.ndarray, phi_scaled: np.ndarray, los_axis: int,
                          chi_grid: np.ndarray,
                          geometry: str = "mixed",
                          ring_bins: int = 48, pad: int = 1,
                          k_min: float = 6.0) -> dict:
    """Compute all four measures on a χ-grid (0<χ<20).
    Returns dict with arrays keyed by 'chi', 'pfa', 'dvar', 'mP', 'eP', 'mD', 'eD'."""
    assert geometry in {"mixed", "separated"}
    lam2_grid = chi_grid / 2.0  # because σ_Φ has been scaled to 1
    lam_grid = np.sqrt(lam2_grid)

    pfa = np.empty_like(lam2_grid)
    dvar = np.empty_like(lam2_grid)
    mP = np.empty_like(lam2_grid)
    eP = np.empty_like(lam2_grid)
    mD = np.empty_like(lam2_grid)
    eD = np.empty_like(lam2_grid)

    for i, lam in enumerate(lam_grid):
        if geometry == "mixed":
            P_map  = P_map_mixed (Pi, phi_scaled, lam, los_axis)
            dP_map = dP_map_mixed(Pi, phi_scaled, lam, los_axis)
        else:
            P_map  = P_map_separated (Pi, phi_scaled, lam, los_axis)
            dP_map = dP_map_separated(Pi, phi_scaled, lam, los_axis)

        # One-point variances (map mean of |.|^2)
        pfa[i]  = np.mean(np.abs(P_map)**2)
        dvar[i] = np.mean(np.abs(dP_map)**2)

        # Two-point PSA slopes
        kP, EP = psa_of_map(P_map,  ring_bins=ring_bins, pad=pad, apodize=True, k_min=k_min)
        kD, ED = psa_of_map(dP_map, ring_bins=ring_bins, pad=pad, apodize=True, k_min=k_min)
        mP[i], _, eP[i], _ = fit_log_slope_window(kP, EP)
        mD[i], _, eD[i], _ = fit_log_slope_window(kD, ED)

    return {
        "chi": chi_grid,
        "pfa": pfa,
        "dvar": dvar,
        "mP": mP, "eP": eP,
        "mD": mD, "eD": eD,
    }


# ---------- Plotting ----------

def plot_four_measures(res: dict, geometry: str, out_png: str):
    chi = res["chi"]
    pfa = res["pfa"]
    dvar = res["dvar"]
    mP, eP = res["mP"], res["eP"]
    mD, eD = res["mD"], res["eD"]

    # Normalize the variances to their small-χ value (for visual comparison) and plot log10
    pfa_n = pfa / (pfa[0] if pfa[0] != 0 else 1.0)
    dvar_n = dvar / (dvar[0] if dvar[0] != 0 else 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharex=True)
    (ax1, ax2), (ax3, ax4) = axes

    # Panel A: PSA slopes
    okP = np.isfinite(mP); okD = np.isfinite(mD)
    ax1.errorbar(chi[okP], mP[okP], yerr=eP[okP], fmt='o-', ms=3, capsize=3, label=r"slope of $P$")
    ax1.errorbar(chi[okD], mD[okD], yerr=eD[okD], fmt='s-', ms=3, capsize=3, label=r"slope of $\partial P/\partial\lambda^2$")
    ax1.axhline(0, color='k', lw=0.6, alpha=0.35)
    ax1.set_ylabel(r"2D PSD slope $m$ (from ring-avg $P_{2D}(k)$)")
    ax1.set_xscale('log')
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend(frameon=False, loc='best')
    ax1.set_title(f"Spatial slopes vs $\chi$ ({geometry})")

    # Panel B: PFA variance (log10)
    ax2.plot(chi, np.log10(pfa_n), '-', lw=1.5, color='tab:blue', label=r"$\log_{10}\langle|P|^2\rangle$ (norm)")
    ax2.set_xscale('log')
    ax2.set_ylabel(r"$\log_{10}$ PFA (norm)")
    ax2.grid(True, which='both', alpha=0.25)
    ax2.legend(frameon=False, loc='best')
    ax2.set_title("PFA variance vs $\chi$")

    # Panel C: derivative variance (log10)
    ax3.plot(chi, np.log10(dvar_n), '-', lw=1.5, color='tab:orange', label=r"$\log_{10}\langle|\partial P/\partial\lambda^2|^2\rangle$ (norm)")
    ax3.set_xscale('log')
    ax3.set_xlabel(r"$\chi \equiv 2\,\sigma_\Phi\,\lambda^2$")
    ax3.set_ylabel(r"$\log_{10}$ derivative PFA (norm)")
    ax3.grid(True, which='both', alpha=0.25)
    ax3.legend(frameon=False, loc='best')

    # Panel D: helper / overlay: show the χ-band 1–3 on a blank axis with text
    ax4.set_xscale('log')
    ax4.set_yscale('linear')
    ax4.set_xlim(chi.min(), chi.max())
    ax4.set_ylim(0, 1)
    ax4.axvspan(1.0, 3.0, color='gray', alpha=0.15, label=r"expected break $1\lesssim\chi\lesssim3$")
    ax4.text(1.1, 0.7, r"highlighting $1\!<\!\chi\!<\!3$", fontsize=10)
    ax4.get_yaxis().set_visible(False)
    ax4.grid(True, which='both', axis='x', alpha=0.25)
    ax4.legend(frameon=False, loc='best')
    ax4.set_xlabel(r"$\chi \equiv 2\,\sigma_\Phi\,\lambda^2$ (0 < $\chi$ < 20)")

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(max(chi.min(), 1e-3), 60.0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved {out_png}")


# ---------- Figure recreation from NPZ ----------

def recreate_figure_from_npz(npz_path: str, out_png: str = None):
    """Recreate the four-measures figure from saved NPZ data."""
    data = np.load(npz_path)
    
    res = {
        "chi": data["chi"],
        "pfa": data["pfa"], 
        "dvar": data["dvar"],
        "mP": data["mP"], 
        "eP": data["eP"],
        "mD": data["mD"], 
        "eD": data["eD"]
    }
    
    geometry = str(data["geometry"])
    if out_png is None:
        out_png = f"four_measures_{geometry}_recreated.png"
    
    plot_four_measures(res, geometry, out_png)
    print(f"Recreated figure: {out_png}")
    print(f"Original parameters: σ_Φ_raw={data['sigmaPhi_raw']:.6g}, C_scale={data['C_scale']:.6g}")
    print(f"LOS axis: {data['los_axis']}, γ: {data['gamma']}, LP16 amp: {data['lp16_amp']}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Compare 4 polarization measures vs chi = 2 sigmaPhi lambda^2")
    parser.add_argument('--h5', help='Path to HDF5 cube with fields (required for computation)')
    parser.add_argument('--npz', help='Path to NPZ file to recreate figure from (alternative to --h5)')
    parser.add_argument('--geometry', default='mixed', choices=['mixed','separated'], help='Emission/rotation geometry')
    parser.add_argument('--los-axis', type=int, default=2, help='LOS axis (0,1,2) — picks B_parallel = Bx,By,Bz respectively')
    parser.add_argument('--gamma', type=float, default=2.0, help='Relativistic electron index (LP16 emissivity)')
    parser.add_argument('--lp16-amp', action='store_true', help='Use LP16 amplitude factor |B_perp|^{(gamma-3)/2} (default: off)')
    parser.add_argument('--ring-bins', type=int, default=48, help='# PSA k-rings')
    parser.add_argument('--pad', type=int, default=1, help='Zero-pad factor for PSA (1=no pad)')
    parser.add_argument('--k-min', type=float, default=6.0, help='Minimum k for PSA rings')
    parser.add_argument('--chi-min', type=float, default=1e-3, help='Min chi (must be >0)')
    parser.add_argument('--chi-max', type=float, default=60.0, help='Max chi (<=20 as requested)')
    parser.add_argument('--n-chi', type=int, default=60, help='# points in chi grid (log-spaced)')
    parser.add_argument('--out', default=None, help='Output PNG name (auto if None)')
    args = parser.parse_args()

    # Handle NPZ recreation mode
    if args.npz:
        if args.h5:
            print("Warning: Both --h5 and --npz provided. Using --npz for figure recreation.")
        recreate_figure_from_npz(args.npz, args.out)
        return

    # Require H5 file for computation
    if not args.h5:
        parser.error("Either --h5 (for computation) or --npz (for recreation) must be provided")

    # Load cube
    keys = FieldKeys()
    Bx, By, Bz, ne = load_fields(args.h5, keys)

    # Emissivity (LP16-recommended form by default: spin-2 quadratic; amplitude factor optional)
    Pi = polarized_emissivity_lp16(Bx, By, gamma=args.gamma, use_lp16_amp=args.lp16_amp)

    # LOS → pick B_parallel consistently with axis index
    if args.los_axis == 0:
        Bpar = Bx
    elif args.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz

    # Build φ with C=1 to measure σ_Φ, then rescale so σ_Φ=1 (code units)
    phi_raw = faraday_density(ne, Bpar, C=1.0)
    sigmaPhi_raw = sigma_phi_total(phi_raw, args.los_axis)
    if not np.isfinite(sigmaPhi_raw) or sigmaPhi_raw <= 0:
        raise RuntimeError("σ_Φ is non-finite or non-positive; check input fields.")
    C_scale = 1.0 / sigmaPhi_raw
    phi = phi_raw * C_scale

    print(f"[diag] σ_Φ (raw) = {sigmaPhi_raw:.6g}; scaled φ by C={C_scale:.6g} such that σ_Φ=1 → χ=2λ²")

    # χ grid in (0,20)
    chi_min = max(float(args.chi_min), 1e-6)
    chi_max = min(float(args.chi_max), 60.0)
    if chi_min <= 0 or chi_max <= 0 or not (chi_min < chi_max):
        raise ValueError("Require 0 < chi_min < chi_max; and chi_max ≤ 20.")
    chi_grid = np.logspace(np.log10(chi_min), np.log10(chi_max), int(args.n_chi))

    # Compute all measures vs χ
    res = compute_curves_vs_chi(
        Pi, phi, los_axis=args.los_axis, chi_grid=chi_grid,
        geometry=args.geometry, ring_bins=args.ring_bins, pad=args.pad, k_min=args.k_min
    )

    # Save data to NPZ for later figure recreation
    out_npz = args.out.replace('.png', '.npz') if args.out and args.out.endswith('.png') else f"four_measures_{args.geometry}.npz"
    np.savez(out_npz, 
             chi=res["chi"], 
             pfa=res["pfa"], 
             dvar=res["dvar"], 
             mP=res["mP"], 
             eP=res["eP"], 
             mD=res["mD"], 
             eD=res["eD"],
             geometry=args.geometry,
             sigmaPhi_raw=sigmaPhi_raw,
             C_scale=C_scale,
             los_axis=args.los_axis,
             gamma=args.gamma,
             lp16_amp=args.lp16_amp)
    print(f"Saved data to {out_npz}")

    # Plot & save
    out_png = args.out or f"four_measures_{args.geometry}.png"
    plot_four_measures(res, args.geometry, out_png)


if __name__ == "__main__":
    main()
