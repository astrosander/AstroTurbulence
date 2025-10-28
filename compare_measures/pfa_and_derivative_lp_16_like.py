#!/usr/bin/env python3
"""
PFA and derivative measure (LP16 §6) on a sub-Alfvénic turbulence cube.

- Two geometries:
  1) Separated regions (external Faraday screen): synchrotron emission and Faraday rotation are spatially distinct.
  2) Mixed volume: emission and Faraday rotation co-exist along the LOS.

- Outputs (LP16-like presentation):
  • ⟨|P(λ)|^2⟩ vs λ^2  (PFA)
  • ⟨|dP/dλ^2|^2⟩ vs λ^2 (derivative measure from LP16 §6)

The formulas used:
  P(X, λ) = ∫ dz P_i(X,z) exp[ 2 i λ^2 Φ(X,z) ],  with  Φ(X,z) = ∫_0^z φ(X,z') dz',
  where φ ∝ n_e B_∥ and P_i ∝ (B_x + i B_y)^{(γ+1)/2}.  In the external-screen case,
  P(X, λ) = P_emit(X) exp[ 2 i λ^2 Φ_screen(X) ].

  Derivative measure:
  dP/dλ^2 = ∫ dz ( 2 i P_i Φ ) exp[ 2 i λ^2 Φ ]   (mixed),
  dP/dλ^2 = ( 2 i Φ_screen ) P  (separated).

These follow LP16 eqs. (158–163) for the derivative correlation and the definitions
of P; the PFA itself is the R=0 case of the two-point correlation (variance). See LP16
for regimes and scalings.

NOTE: This script focuses on getting LP16-like curves from your cube; absolute units
are arbitrary unless faraday_const is calibrated.
"""
from __future__ import annotations
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
from multiprocessing import Pool, cpu_count
import functools
from numpy.random import default_rng

# -----------------------------
# User I/O settings
# -----------------------------
H5_PATH = r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

# -----------------------------
# Config and helpers
# -----------------------------
@dataclass
class PFAConfig:
    lam_grid: Iterable[float] = tuple(np.sqrt(np.linspace(0.05**2, 10.0**2, 50)))  # Homogeneous in λ², 50 points
    lam_mark: float = 1.0
    gamma: float = 2.0            # electron index → P_i ∝ (B_⊥)^{(γ+1)/2}
    faraday_const: float = 0.81    # arbitrary units here; relative scaling only
    los_axis: int = 2              # integrate along this axis (z is LOS by default)
    voxel_depth: float = 1.0       # unused after Patch 1; retained for backward compat
    lam2_break_target: float = 1e-1  # place χ≈1 near this λ² (matches Zhang fig)
    use_auto_faraday: bool = True


class FieldKeys:
    Bx: str = BX_KEY
    By: str = BY_KEY
    Bz: str = BZ_KEY
    ne: str = NE_KEY


def load_fields(h5_path: str, keys: FieldKeys):
    with h5py.File(h5_path, "r") as f:
        Bx = np.array(f[keys.Bx])
        By = np.array(f[keys.By])
        Bz = np.array(f[keys.Bz])
        ne = np.array(f[keys.ne])
    return Bx, By, Bz, ne


def polarized_emissivity_lp16(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Compute polarized emissivity P_i = (B_x + iB_y)^2 |B_perp|^((γ-3)/2)
    
    This is the LP16 formula with γ-dependent amplitude factor.
    For γ=2: P_i = (B_x + iB_y)^2 / sqrt(|B_perp|)
    For γ=3: P_i = (B_x + iB_y)^2 (pure quadratic, no amplitude factor)
    
    Args:
        Bx, By: magnetic field components perpendicular to LOS
        gamma: spectral index of relativistic electron distribution
    
    Returns:
        Pi: complex polarized emissivity
    """
    Bp2 = Bx*Bx + By*By
    amp = np.maximum(Bp2, 1e-30)**(0.5*(gamma-3.0))
    return (Bx + 1j*By)**2 * amp


def polarized_emissivity_simple(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    LP16 emissivity:
      P_i = (Bx + i By)^2 * |B_perp|^{(gamma-3)/2}
    For gamma=2: P_i = (Bx + i By)^2 / |B_perp|
    """
    B2 = Bx**2 + By**2
    eps = np.finfo(B2.dtype).tiny
    amp = np.power(np.maximum(B2, eps), 0.5*(gamma - 3.0))
    return ((Bx + 1j*By)**2 * amp).astype(np.complex128)


def faraday_density(ne: np.ndarray, Bpar: np.ndarray, C: float = 0.81) -> np.ndarray:
    return C * ne * Bpar


def linear_tracer(Bx: np.ndarray, By: np.ndarray) -> np.ndarray:
    """
    Create a linear tracer field for comparison with quadratic emissivity.
    
    This returns Bx + i*By (linear in the magnetic field components)
    instead of (Bx + i*By)^2 (quadratic).
    
    This should show the expected -8/3 ≈ -2.67 slope for 2D PSD
    of a linear field projected from 3D Kolmogorov turbulence.
    
    Args:
        Bx, By: magnetic field components perpendicular to LOS
    
    Returns:
        Linear tracer field (complex)
    """
    return Bx + 1j*By


def sigma_phi_total(phi: np.ndarray, los_axis: int) -> float:
    """σ_Φ from integrating φ along LOS with Δz=1/Nz."""
    arr = np.moveaxis(phi, los_axis, 0)
    Nz = arr.shape[0]
    dz = 1.0 / float(Nz)
    Phi_tot = np.sum(arr * dz, axis=0)   # Ny×Nx map
    return float(Phi_tot.std())


def _move_los(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(arr, axis, 0)


def psa_of_map(P_map: np.ndarray, ring_bins: int = 48, pad: int = 1,
               apodize: bool = True, k_min: float = 6.0,
               min_counts_per_ring: int = 10,
               beam_sigma_px: float = 0.0,
               return_energy_like: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute azimuthally-averaged power spectrum (PSA) of a 2D polarization map.
    This is the spatial statistic discussed in LP16 §6.
    
    Improved version with:
    - Geometric (log) binning for better slope fitting
    - Zero-padding to reduce aliasing
    - Hann windowing to reduce edge effects
    - Careful k-range selection (avoids undersampled low-k and noisy high-k)
    
    Args:
        P_map: 2D complex polarization map
        ring_bins: number of logarithmic k-bins
        pad: zero-padding factor (1=no padding, 2=double size)
        apodize: apply 2D Hann window
        k_min: minimum k to include (avoid undersampled modes)
        min_counts_per_ring: minimum points per ring for valid bin
        beam_sigma_px: Gaussian beam smoothing in pixels (0=no beam)
        return_energy_like: if True, return E_2D(k) = 2πk P_2D(k) instead of P_2D(k)
    
    Returns:
        k_centers: wavenumber bins (geometric mean)
        E_k: ring-averaged power spectrum (PSD or energy-like depending on return_energy_like)
    """
    # Detrend (always)
    Y = P_map - P_map.mean()
    
    # Apply 2D Hann window
    if apodize:
        wy = np.hanning(Y.shape[0])
        wx = np.hanning(Y.shape[1])
        Y = Y * np.outer(wy, wx)
    
    # Zero-pad
    if pad > 1:
        Yp = np.zeros((pad * Y.shape[0], pad * Y.shape[1]), dtype=complex)
        Yp[:Y.shape[0], :Y.shape[1]] = Y
        Y = Yp
    
    # 2D FFT (optionally apply Gaussian telescope beam in Fourier space)
    F = np.fft.fft2(Y)
    if beam_sigma_px > 0:
        ky = np.fft.fftfreq(Y.shape[0]) * Y.shape[0]
        kx = np.fft.fftfreq(Y.shape[1]) * Y.shape[1]
        KX, KY = np.meshgrid(kx, ky)
        G = np.exp(-0.5 * beam_sigma_px**2 * (KX**2 + KY**2))
        F *= G
    F = np.fft.fftshift(F)
    P2 = (F * np.conj(F)).real
    
    # Wavenumber grids
    ky = np.fft.fftshift(np.fft.fftfreq(P2.shape[0])) * P2.shape[0]
    kx = np.fft.fftshift(np.fft.fftfreq(P2.shape[1])) * P2.shape[1]
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY).ravel()
    
    # Geometric binning (better for power-law fitting)
    k_max = min(P2.shape) / 3.5  # a bit lower than Nyquist/4 on 256 grids
    bins = np.geomspace(max(1.0, k_min), k_max, ring_bins + 1)
    lab = np.digitize(KR, bins) - 1
    
    # Ring-average with geometric mean for k
    # ring population counts (avoid under-sampled bins)
    counts = np.array([(lab == i).sum() for i in range(ring_bins)])
    Ek = np.array([P2.ravel()[lab == i].mean() if counts[i] >= min_counts_per_ring else np.nan
                   for i in range(ring_bins)])
    kcen = np.sqrt(bins[:-1] * bins[1:])  # geometric mean
    
    # Keep only well-sampled bins
    msk = np.isfinite(Ek) & (kcen >= k_min) & (kcen <= k_max)

    # Fallback: integer-radius rings if too few bins survived
    if msk.sum() < 10:
        ky = np.fft.fftshift(np.fft.fftfreq(P2.shape[0])) * P2.shape[0]
        kx = np.fft.fftshift(np.fft.fftfreq(P2.shape[1])) * P2.shape[1]
        KX, KY = np.meshgrid(kx, ky)
        KR = np.hypot(KX, KY)
        kr = np.floor(KR + 0.5).astype(int)
        kmax_int = int(min(P2.shape) / 2)  # up to Nyquist
        weights = P2.ravel()
        idx = kr.ravel()
        Ek_int = np.bincount(idx, weights=weights, minlength=kmax_int + 1)
        Nk_int = np.bincount(idx, minlength=kmax_int + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Ek_int = Ek_int / np.maximum(Nk_int, 1)
        kvec = np.arange(kmax_int + 1)
        msk2 = (kvec >= int(k_min)) & (kvec <= int(k_max)) & (Nk_int > 25)
        # drop very low-k and very high-k
        kvec = kvec[msk2]
        Ek_int = Ek_int[msk2]
        if return_energy_like:
            Ek_int = 2.0 * np.pi * kvec * Ek_int
        return kvec, Ek_int

    if return_energy_like:
        Ek = 2.0 * np.pi * kcen * Ek
    return kcen[msk], Ek[msk]


def compute_faraday_regime(phi: np.ndarray, verbose: bool = True) -> dict:
    """
    Diagnose the Faraday rotation regime and estimate key length scales.
    
    Returns dict with:
        - phi_mean: mean Faraday density
        - phi_std: std Faraday density  
        - regime: "random-dominated" or "mean-dominated"
        - ratio: |phi_mean| / phi_std
        - r_i: crude RM correlation length estimate (in pixels)
    """
    phi_mean = np.mean(phi)
    phi_std = np.std(phi)
    ratio = abs(phi_mean) / phi_std if phi_std > 0 else np.inf
    regime = "random-dominated" if ratio < 1 else "mean-dominated"
    
    # Crude correlation length estimate (autocorrelation scale)
    # Use central slice for speed
    mid = phi.shape[0] // 2
    phi_slice = phi[mid, :, :]
    phi_slice = phi_slice - phi_slice.mean()
    
    # 2D autocorrelation via FFT
    F = np.fft.fft2(phi_slice)
    acf = np.fft.ifft2(F * np.conj(F)).real
    acf = np.fft.fftshift(acf)
    acf = acf / acf.max()
    
    # Find where ACF drops to 1/e
    center = np.array(acf.shape) // 2
    y, x = np.ogrid[:acf.shape[0], :acf.shape[1]]
    r = np.hypot(y - center[0], x - center[1])
    
    # Radial profile
    r_bins = np.arange(0, min(acf.shape) // 2, 1)
    acf_radial = []
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            acf_radial.append(acf[mask].mean())
        else:
            acf_radial.append(0)
    
    acf_radial = np.array(acf_radial)
    # Find where it drops below 1/e
    idx = np.where(acf_radial < 1.0 / np.e)[0]
    r_i = r_bins[idx[0]] if len(idx) > 0 else 5.0  # default fallback
    
    result = {
        'phi_mean': phi_mean,
        'phi_std': phi_std,
        'ratio': ratio,
        'regime': regime,
        'r_i': float(r_i)
    }
    
    if verbose:
        print("\nFaraday regime diagnostic:")
        print(f"  Mean φ: {phi_mean:.4f}")
        print(f"  Std φ:  {phi_std:.4f}")
        print(f"  |φ̄|/σ_φ: {ratio:.4f}")
        print(f"  → {regime}")
        print(f"  RM correlation length r_i ≈ {r_i:.1f} pixels")
    
    return result


def effective_faraday_depth(lam: float, phi_info: dict) -> float:
    """
    Compute effective Faraday depth L_eff ~ 1 / [λ² max(|φ̄|, σ_φ)]
    
    This determines if we're in short-λ (L_eff >> L) or long-λ (L_eff << L) regime.
    """
    phi_scale = max(abs(phi_info['phi_mean']), phi_info['phi_std'])
    if phi_scale == 0:
        return np.inf
    return 1.0 / (lam**2 * phi_scale)


def choose_lambda_for_regime(phi_info: dict, LOS_depth: float, 
                             target_regime: str = "long") -> float:
    """
    Choose λ to target specific regime.
    
    Args:
        phi_info: output from compute_faraday_regime
        LOS_depth: depth of LOS integration in pixels
        target_regime: "short" (weak rotation) or "long" (strong rotation)
    
    Returns:
        optimal λ value
    """
    phi_scale = max(abs(phi_info['phi_mean']), phi_info['phi_std'])
    r_i = phi_info['r_i']
    
    if target_regime == "long":
        # target exactly L_eff≈r_i
        lam_opt = 1.0 / np.sqrt(phi_scale * (1.0 * r_i))
    else:  # "short"
        # Want L_eff ~ LOS_depth (weak rotation)
        # λ ~ 1/√(φ_scale * LOS_depth)
        lam_opt = 1.0 / np.sqrt(phi_scale * LOS_depth)
        # Nudge slightly smaller, but stay within regime; 0.8 is plenty
        lam_opt *= 0.8
    
    return lam_opt


def decorrelate_phi_along_z(phi: np.ndarray, los_axis: int, seed: int = 42) -> np.ndarray:
    """Break Pi–Φ correlations by rolling φ along the LOS per sightline (preserves r_i)."""
    rng = default_rng(seed)
    arr = np.moveaxis(phi, los_axis, 0)
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        for i in range(arr.shape[2]):
            shift = rng.integers(0, arr.shape[0])
            out[:, j, i] = np.roll(arr[:, j, i], shift)
    return np.moveaxis(out, 0, los_axis)

# -----------------------------
# P and dP/dλ² maps for two geometries
# -----------------------------

def P_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                    emit_bounds: Optional[Tuple[int, int]] = None,
                    screen_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """P for external screen: P = P_emit * exp(2 i λ² Φ_screen)."""
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)   
    e0 = emit_bounds or (0, max(1, int(0.9 * Nz)))
    s0 = screen_bounds or (max(0, e0[1]), Nz)
    P_emit = np.sum(Pi_los[e0[0]:e0[1], :, :], axis=0) * dz
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * dz
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)


def P_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    z0, z1 = bounds or (0, Nz)
    dz = 1.0 / float(Nz)   
    # half-cell Faraday depth: Φ(z+½) = (Σ_0^z φ Δz) + ½ φ Δz
    rm = phi_los[z0:z1, :, :]
    Phi_cum = np.cumsum(rm * dz, axis=0)
    Phi_half = Phi_cum - 0.5 * rm * dz
    phase = np.exp(2j * (lam**2) * Phi_half)
    contrib = Pi_los[z0:z1, :, :] * phase
    return np.sum(contrib, axis=0) * dz


def dP_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                      emit_bounds: Optional[Tuple[int, int]] = None,
                      screen_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Derivative dP/d(λ²) for external screen: 2i Φ_screen * P_emit * e^{2i λ² Φ_screen}."""
    P = P_map_separated(Pi, phi, lam, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = phi_los.shape[0]
    dz = 1.0 / float(Nz)
    s0 = screen_bounds or (max(0, int(0.9 * Nz)), Nz)
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * dz
    return 2j * Phi_screen * P


def dP_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                  bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Derivative dP/d(λ²) for mixed geometry: 2i ∫ Pi(z) Φ(z) e^{2i λ² Φ(z)} dz."""
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    z0, z1 = bounds or (0, Nz)
    dz = 1.0 / float(Nz)
    # half-cell Faraday depth: Φ(z+½) = (Σ_0^z φ Δz) + ½ φ Δz
    rm = phi_los[z0:z1, :, :]
    Phi_cum = np.cumsum(rm * dz, axis=0)
    Phi_half = Phi_cum - 0.5 * rm * dz
    phase = np.exp(2j * (lam**2) * Phi_half)
    contrib = 2j * (Pi_los[z0:z1, :, :] * Phi_half) * phase
    return np.sum(contrib, axis=0) * dz

# -----------------------------
# Multiprocessing helper functions
# -----------------------------

def _compute_separated_single(args):
    """Helper function for parallel computation of single lambda value"""
    lam2, P_emit, Phi_screen = args
    P_map = P_emit * np.exp(2j * lam2 * Phi_screen)
    return np.mean(np.abs(P_map)**2)


def _compute_mixed_single(args):
    """Helper for mixed case (uses half-cell Φ and Δz)."""
    lam2, Pi_los, rm, z0, z1 = args
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_cum = np.cumsum(rm[z0:z1, :, :] * dz, axis=0)
    Phi_half = Phi_cum - 0.5 * rm[z0:z1, :, :] * dz
    phase = np.exp(2j * lam2 * Phi_half)
    P_map = np.sum(Pi_los[z0:z1, :, :] * phase, axis=0) * dz
    return np.mean(np.abs(P_map)**2)


def _compute_derivative_separated_single(args):
    """Helper function for parallel computation of derivative measure for separated case"""
    lam2, P_emit, Phi_screen = args
    P_map = P_emit * np.exp(2j * lam2 * Phi_screen)
    dP_map = 2j * Phi_screen * P_map
    return np.mean(np.abs(dP_map)**2)


def _compute_derivative_mixed_single(args):
    """Helper for dP/dλ² in mixed case with half-cell Φ and Δz."""
    lam2, Pi_los, rm, z0, z1 = args
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_cum = np.cumsum(rm[z0:z1, :, :] * dz, axis=0)
    Phi_half = Phi_cum - 0.5 * rm[z0:z1, :, :] * dz
    phase = np.exp(2j * lam2 * Phi_half)
    contrib = 2j * (Pi_los[z0:z1, :, :] * Phi_half) * phase
    dP_map = np.sum(contrib, axis=0) * dz
    return np.mean(np.abs(dP_map)**2)


# -----------------------------
# PFA curves (variance over sky vs λ²)
# -----------------------------

def pfa_curve_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                        emit_bounds: Optional[Tuple[int, int]] = None,
                        screen_bounds: Optional[Tuple[int, int]] = None,
                        n_processes: int = 11):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    
    # Pre-compute P_emit and Phi_screen once
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    e0 = emit_bounds or (0, max(1, int(0.9 * Nz)))
    s0 = screen_bounds or (max(0, e0[1]), Nz)
    dz = 1.0 / float(Nz)
    P_emit = np.sum(Pi_los[e0[0]:e0[1], :, :], axis=0) * dz
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * dz
    
    # Parallel computation for all lambda values
    lam2_grid = lam_grid**2
    args_list = [(lam2, P_emit, Phi_screen) for lam2 in lam2_grid]
    
    with Pool(processes=n_processes) as pool:
        var = pool.map(_compute_separated_single, args_list)
    
    return lam2_grid, np.array(var)


def pfa_curve_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                    bounds: Optional[Tuple[int, int]] = None,
                    n_processes: int = 11):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    
    # Pre-compute arrays once for efficiency
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    z0, z1 = bounds or (0, Nz)
    
    # Parallel computation for all λ: pass raw rm=φ to worker (it builds Φ_half)
    lam2_grid = lam_grid**2
    args_list = [(lam2, Pi_los, phi_los, z0, z1) for lam2 in lam2_grid]
    
    with Pool(processes=n_processes) as pool:
        var = pool.map(_compute_mixed_single, args_list)
    
    return lam2_grid, np.array(var)


def pfa_curve_derivative_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                                   emit_bounds: Optional[Tuple[int, int]] = None,
                                   screen_bounds: Optional[Tuple[int, int]] = None,
                                   n_processes: int = 11):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    
    # Pre-compute P_emit and Phi_screen once
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    e0 = emit_bounds or (0, max(1, int(0.9 * Nz)))
    s0 = screen_bounds or (max(0, e0[1]), Nz)
    dz = 1.0 / float(Nz)
    P_emit = np.sum(Pi_los[e0[0]:e0[1], :, :], axis=0) * dz
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * dz
    
    # Parallel computation for all lambda values
    lam2_grid = lam_grid**2
    args_list = [(lam2, P_emit, Phi_screen) for lam2 in lam2_grid]
    
    with Pool(processes=n_processes) as pool:
        var = pool.map(_compute_derivative_separated_single, args_list)
    
    return lam2_grid, np.array(var)


def pfa_curve_derivative_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                               bounds: Optional[Tuple[int, int]] = None,
                               n_processes: int = 11):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    
    # Pre-compute arrays once for efficiency
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    z0, z1 = bounds or (0, Nz)
    
    # Parallel computation: pass raw rm=φ
    lam2_grid = lam_grid**2
    args_list = [(lam2, Pi_los, phi_los, z0, z1) for lam2 in lam2_grid]
    
    with Pool(processes=n_processes) as pool:
        var = pool.map(_compute_derivative_mixed_single, args_list)
    
    return lam2_grid, np.array(var)

# -----------------------------
# Plotting (LP16-style log–log)
# -----------------------------

def fit_log_slope(x: np.ndarray, y: np.ndarray, xmin=None, xmax=None):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if xmin is not None:
        mask &= (x >= xmin)
    if xmax is not None:
        mask &= (x <= xmax)
    xlog = np.log10(x[mask]); ylog = np.log10(y[mask])
    if len(xlog) < 3:
        return np.nan, np.nan
    A = np.vstack([xlog, np.ones_like(xlog)]).T
    m, a = np.linalg.lstsq(A, ylog, rcond=None)[0]
    return m, a


def fit_log_slope_with_bounds(k, E, kmin=None, kmax=None):
    """Fit log-log slope with optional k bounds to avoid edge effects."""
    k = np.asarray(k); E = np.asarray(E)
    sel = np.isfinite(k) & np.isfinite(E) & (k>0) & (E>0)
    if kmin is not None: sel &= (k >= kmin)
    if kmax is not None: sel &= (k <= kmax)
    k = k[sel]; E = E[sel]
    if k.size < 10: return np.nan, np.nan, np.nan, (np.nan, np.nan)
    lk, lE = np.log10(k), np.log10(E)
    m, a = np.polyfit(lk, lE, 1)
    err = np.sqrt(np.mean((lE-(m*lk+a))**2) / np.sum((lk-lk.mean())**2))
    return m, a, err, (k.min(), k.max())


def fit_log_slope_window(k: np.ndarray, E: np.ndarray):
    """Pick a clean ~decade in k with minimal curvature, then fit log-log slope with error."""
    k = np.asarray(k); E = np.asarray(E)
    msk = np.isfinite(k) & np.isfinite(E) & (k>0) & (E>0)
    k, E = k[msk], E[msk]
    if k.size < 12: 
        return np.nan, np.nan, np.nan, (np.nan, np.nan)
    lk, lE = np.log10(k), np.log10(E)
    best = (np.inf, np.nan, np.nan, np.nan, (np.nan, np.nan))  # (curv, m, a, err, (lo,hi))
    for p in (35, 45, 55, 65):
        kmid = 10**np.percentile(lk, p)
        for span in (0.9, 1.0, 1.1):  # ~ one decade ±10%
            lo, hi = kmid/10**(span/2)*1.4, kmid*10**(span/2)*0.6
            s = (k>=lo)&(k<=hi)
            if s.sum()<10: 
                continue
            sl = np.diff(lE[s])/np.diff(lk[s])  # local slopes
            curv = np.nanstd(sl)
            mfit, afit = np.polyfit(lk[s], lE[s], 1)
            # Estimate error from residuals
            yfit = mfit * lk[s] + afit
            residuals = lE[s] - yfit
            mse = np.mean(residuals**2)
            err = np.sqrt(mse / np.sum((lk[s] - np.mean(lk[s]))**2))
            if curv < best[0]:
                best = (curv, mfit, afit, err, (lo,hi))
    return best[1], best[2], best[3], best[4]


def plot_pfa(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
             title: str = "PFA: $\\langle|P|^2\\rangle$ vs $\\lambda^2$"):
    plt.figure(figsize=(6.5, 5.0))
    for var, lab in zip(var_list, labels):
        plt.loglog(lam2, var, label=lab)
    if var_list:
        m, a = fit_log_slope(lam2, var_list[0])
        if np.isfinite(m):
            mid = 10**np.nanmedian(np.log10(lam2))
            ymid = 10**(a + m * np.log10(mid))
            xline = np.array([mid/5, mid*5])
            yline = ymid * (xline/mid)**m
            plt.loglog(xline, yline, '--', linewidth=1, label=f"fit slope ≈ {m:.2f}")
    lam_mark2 = cfg.lam_mark**2
    for var in var_list:
        vmark = np.interp(lam_mark2, lam2, var)
        plt.scatter([lam_mark2], [vmark], marker='o', s=10)
    plt.xlabel("$\\lambda^2$ (arb. units)")
    plt.ylabel("$\\langle |P|^2 \\rangle$ (arb. units)")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def plot_pfa_derivative(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
                        title: str = r"Derivative measure $\langle|dP/d\lambda^2|^2\rangle$"):
    plt.figure(figsize=(6.5, 5.0))
    for var, lab in zip(var_list, labels):
        plt.loglog(lam2, var, label=lab)
    if var_list:
        m, a = fit_log_slope(lam2, var_list[0])
        if np.isfinite(m):
            mid = 10**np.nanmedian(np.log10(lam2))
            ymid = 10**(a + m * np.log10(mid))
            xline = np.array([mid/5, mid*5])
            yline = ymid * (xline/mid)**m
            plt.loglog(xline, yline, '--', linewidth=1, label=f"fit slope ≈ {m:.2f}")
    lam_mark2 = cfg.lam_mark**2
    for var in var_list:
        vmark = np.interp(lam_mark2, lam2, var)
        plt.scatter([lam_mark2], [vmark], marker='o', s=10)
    plt.xlabel(r"$\lambda^2$ (arb. units)")
    plt.ylabel(r"$\langle |dP/d\lambda^2|^2 \rangle$ (arb. units)")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def plot_combined_pfa_and_derivative(lam2_pfa: np.ndarray, var_pfa_list: list, 
                                    lam2_der: np.ndarray, var_der_list: list,
                                    labels: list, cfg: PFAConfig):
    """Plot both PFA and derivative measures on the same figure with subplots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # PFA plot
    for var, lab in zip(var_pfa_list, labels):
        ax1.loglog(lam2_pfa, var, label=lab)
    if var_pfa_list:
        m, a = fit_log_slope(lam2_pfa, var_pfa_list[0])
        if np.isfinite(m):
            mid = 10**np.nanmedian(np.log10(lam2_pfa))
            ymid = 10**(a + m * np.log10(mid))
            xline = np.array([mid/5, mid*5])
            yline = ymid * (xline/mid)**m
            ax1.loglog(xline, yline, '--', linewidth=1, label=f"fit slope ≈ {m:.2f}")
    lam_mark2 = cfg.lam_mark**2
    for var in var_pfa_list:
        vmark = np.interp(lam_mark2, lam2_pfa, var)
        ax1.scatter([lam_mark2], [vmark], marker='o', s=10)
    ax1.set_xlabel("$\\lambda^2$ (arb. units)")
    ax1.set_ylabel("$\\langle |P|^2 \\rangle$ (arb. units)")
    ax1.set_title("PFA: $\\langle|P|^2\\rangle$ vs $\\lambda^2$")
    ax1.grid(True, which='both', alpha=0.2)
    ax1.legend()
    
    # Derivative plot
    for var, lab in zip(var_der_list, labels):
        ax2.loglog(lam2_der, var, label=lab)
    if var_der_list:
        m, a = fit_log_slope(lam2_der, var_der_list[0])
        if np.isfinite(m):
            mid = 10**np.nanmedian(np.log10(lam2_der))
            ymid = 10**(a + m * np.log10(mid))
            xline = np.array([mid/5, mid*5])
            yline = ymid * (xline/mid)**m
            ax2.loglog(xline, yline, '--', linewidth=1, label=f"fit slope ≈ {m:.2f}")
    for var in var_der_list:
        vmark = np.interp(lam_mark2, lam2_der, var)
        ax2.scatter([lam_mark2], [vmark], marker='o', s=10)
    ax2.set_xlabel(r"$\lambda^2$ (arb. units)")
    ax2.set_ylabel(r"$\langle |dP/d\lambda^2|^2 \rangle$ (arb. units)")
    ax2.set_title(r"Derivative measure $\langle|dP/d\lambda^2|^2\rangle$")
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()


# --- replace your plot_spatial_psa_comparison signature & body ---
def plot_spatial_psa_comparison(k_P: np.ndarray, Ek_P: np.ndarray, 
                                k_dP: np.ndarray, Ek_dP: np.ndarray,
                                lam: float, case: str = "Mixed",
                                spectrum: str = "psd",      # "psd" or "energy"
                                kfit_bounds: tuple = (4, 25) # match diagnostics
                                ):
    """
    Plot spatial spectra with consistent settings and fit bounds.
    'spectrum' just labels the y-axis; the actual E(k) vs P(k) choice must
    be done upstream by calling psa_of_map(return_energy_like=bool).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # PSA of P
    ax1.loglog(k_P, Ek_P, 'o-', markersize=2, label=f'{spectrum} of $P(\\lambda={lam:.1f})$')
    m_P, a_P, err_P, (klo_P, khi_P) = fit_log_slope_with_bounds(k_P, Ek_P, *kfit_bounds)
    if np.isfinite(m_P):
        k_fit = np.array([klo_P, khi_P])
        E_fit = 10**(a_P + m_P * np.log10(k_fit))
        ax1.loglog(k_fit, E_fit, '--', linewidth=2, color='red',
                   label=f'slope = {m_P:.2f} ± {err_P:.2f}')
        ax1.axvspan(klo_P, khi_P, alpha=0.08, color='red', label='fit range')
    ax1.set_xlabel('$k$'); ax1.set_ylabel('$E(k)$' if spectrum=='energy' else '$P(k)$')
    ax1.set_title(f'{case}: {spectrum} of $P$ at $\\lambda={lam:.1f}$')
    ax1.grid(True, which='both', alpha=0.2); ax1.legend()

    # PSA of dP/dλ²
    ax2.loglog(k_dP, Ek_dP, 's-', markersize=2, color='darkorange',
               label=f'{spectrum} of $\\partial P/\\partial\\lambda^2$')
    m_dP, a_dP, err_dP, (klo_D, khi_D) = fit_log_slope_with_bounds(k_dP, Ek_dP, *kfit_bounds)
    if np.isfinite(m_dP):
        k_fit = np.array([klo_D, khi_D])
        E_fit = 10**(a_dP + m_dP * np.log10(k_fit))
        ax2.loglog(k_fit, E_fit, '--', linewidth=2, color='red',
                   label=f'slope = {m_dP:.2f} ± {err_dP:.2f}')
        ax2.axvspan(klo_D, khi_D, alpha=0.08, color='red', label='fit range')
    ax2.set_xlabel('$k$'); ax2.set_ylabel('$E(k)$' if spectrum=='energy' else '$P(k)$')
    ax2.set_title(f'{case}: {spectrum} of $\\partial P/\\partial\\lambda^2$ at $\\lambda={lam:.1f}$')
    ax2.grid(True, which='both', alpha=0.2); ax2.legend()

    plt.tight_layout()

    # Console summary
    print(f"\n{case} case at λ={lam:.1f}:")
    print(f"  {spectrum} of P slope: {m_P:.2f} ± {err_P:.2f}")
    print(f"  {spectrum} of dP/dλ² slope: {m_dP:.2f} ± {err_dP:.2f}")
    print(f"  → Difference: {abs(m_dP - m_P):.2f}")
    return m_P, m_dP, err_P, err_dP


# === Directional spectrum: P_dir(k;λ) = |FFT[cos 2χ]|^2 + |FFT[sin 2χ]|^2 ===
# We build A=cos(2χ), B=sin(2χ) from the complex polarization map P = Q+iU at each λ.
# Implemented to share the same radial binning/fit machinery used by PSA.

def _ring_average_power(P2: np.ndarray, ring_bins: int = 24,
                        k_min: float = 6.0, min_counts_per_ring: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Radial (ring) average of a 2D power array P2 on integer-log bins."""
    ky = np.fft.fftshift(np.fft.fftfreq(P2.shape[0])) * P2.shape[0]
    kx = np.fft.fftshift(np.fft.fftfreq(P2.shape[1])) * P2.shape[1]
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY).ravel()
    # geometric k-bins similar to psa_of_map defaults
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

def _prep_window_and_pad(M: np.ndarray, pad: int = 1, apodize: bool = True) -> np.ndarray:
    """Hann-apodize and zero-pad a 2D map."""
    Y = M - np.nanmean(M)
    if apodize:
        wy = np.hanning(Y.shape[0]); wx = np.hanning(Y.shape[1])
        Y = Y * np.outer(wy, wx)
    if pad > 1:
        Yp = np.zeros((pad * Y.shape[0], pad * Y.shape[1]), dtype=float)
        Yp[:Y.shape[0], :Y.shape[1]] = Y
        Y = Yp
    return Y

def directional_spectrum_of_map(P_map: np.ndarray,
                                ring_bins: int = 24, pad: int = 1, apodize: bool = True,
                                k_min: float = 6.0, min_counts_per_ring: int = 8,
                                beam_sigma_px: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the directional spectrum P_dir(k;λ) = |Â|^2+|B̂|^2 with
    A=cos(2χ), B=sin(2χ), χ = 0.5*arg P, from a single complex map P_map.
    """
    Q = P_map.real; U = P_map.imag
    amp = np.hypot(Q, U)
    eps = np.finfo(amp.dtype).eps
    amp = np.maximum(amp, eps)
    A = Q / amp
    B = U / amp

    A = _prep_window_and_pad(A, pad=pad, apodize=apodize)
    B = _prep_window_and_pad(B, pad=pad, apodize=apodize)

    FA = np.fft.fft2(A); FB = np.fft.fft2(B)
    if beam_sigma_px > 0:
        ky = np.fft.fftfreq(A.shape[0]) * A.shape[0]
        kx = np.fft.fftfreq(A.shape[1]) * A.shape[1]
        KX, KY = np.meshgrid(kx, ky)
        G = np.exp(-0.5 * beam_sigma_px**2 * (KX**2 + KY**2))
        FA *= G; FB *= G
    FA = np.fft.fftshift(FA); FB = np.fft.fftshift(FB)
    P2 = (FA * np.conj(FA)).real + (FB * np.conj(FB)).real
    return _ring_average_power(P2, ring_bins=ring_bins, k_min=k_min, min_counts_per_ring=min_counts_per_ring)

def dir_slopes_vs_lambda(Pi: np.ndarray,
                         phi: np.ndarray,
                         cfg: PFAConfig,
                         geometry: str = "mixed",
                         lam_list: Optional[Iterable[float]] = None,
                         ring_bins: int = 24, pad: int = 1,
                         k_min: float = 6.0, min_counts_per_ring: int = 8
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each λ, compute P_map (mixed/separated), form directional spectrum P_dir(k;λ),
    fit a power-law slope, and collect slope ±1σ vs λ.
    """
    lam_arr = np.array(list(lam_list if lam_list is not None else cfg.lam_grid), dtype=float)
    mDir, eDir = [], []
    for lam in lam_arr:
        if geometry.lower().startswith("mix"):
            Pmap = P_map_mixed(Pi, phi, lam, cfg)
        else:
            Pmap = P_map_separated(Pi, phi, lam, cfg)
        kD, EDir = directional_spectrum_of_map(Pmap, ring_bins=ring_bins, pad=pad,
                                               apodize=True, k_min=k_min, min_counts_per_ring=min_counts_per_ring)
        m_i, a_i, err_i, _ = fit_log_slope_window(kD, EDir)
        mDir.append(m_i); eDir.append(err_i if np.isfinite(err_i) else np.nan)
    return lam_arr, np.array(mDir), np.array(eDir)

def plot_dir_slopes_vs_lambda(lam_arr: np.ndarray,
                             mDir: np.ndarray, eDir: np.ndarray,
                             x_is_lambda2: bool = True,
                             title: Optional[str] = None,
                             label: str = r"Directional $|\widehat{\cos2\chi}|^2+|\widehat{\sin2\chi}|^2$ slope",
                             sigma_phi: Optional[float] = None):
    # X-axis: either λ²/λ or χ=2σ_φ λ² if sigma_phi is provided
    if sigma_phi is not None:
        x = 2.0 * (lam_arr**2) * float(sigma_phi)
        xlab = r"$2\,\sigma_\phi\,\lambda^2$"
        # Mask to 0 < χ < 20 as requested
        mask = np.isfinite(x) & (x > 0) & (x < 20)
    else:
        x = lam_arr**2 if x_is_lambda2 else lam_arr
        xlab = r"$\lambda^2$" if x_is_lambda2 else r"$\lambda$"
        mask = np.isfinite(x)
    ttl = title or (r"Directional-spectrum slope" if x_is_lambda2 else r"Directional-spectrum slope vs $\\lambda$")
    plt.figure(figsize=(7.2, 5.0))
    if np.any(np.isfinite(eDir)):
        ok = np.isfinite(mDir) & mask
        plt.errorbar(x[ok], mDir[ok], yerr=eDir[ok], fmt='d-', capsize=3, label=label)
    else:
        plt.plot(x[mask], mDir[mask], 'd-', label=label)
    plt.axhline(0, color='k', lw=0.6, alpha=0.4)
    plt.xlabel(f"{xlab} (arb. units)")
    plt.ylabel(r"slope $m$ in $P_{\rm dir}(k)\propto k^{\,m}$")
    plt.title(ttl)
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()


# === PSA & derivative-PSA vs lambda ==========================================
def psa_slopes_vs_lambda(Pi: np.ndarray,
                         phi: np.ndarray,
                         cfg: PFAConfig,
                         geometry: str = "mixed",   # "mixed" or "separated"
                         lam_list: Optional[Iterable[float]] = None,
                         ring_bins: int = 64,
                         pad: int = 2,
                         k_min: float = 6.0,
                         min_counts_per_ring: int = 10,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each λ, build P(λ) and dP/dλ² maps, compute their plane-of-sky PSA,
    fit a single power-law slope in k using fit_log_slope_window, and collect
    the slopes (with 1σ errors) vs λ.

    Returns:
        lam_arr, mP, eP, mD, eD
        where mP ≈ slope of E_P(k;λ) and mD ≈ slope of E_{dP}(k;λ).
    """
    lam_arr = np.array(list(lam_list if lam_list is not None else cfg.lam_grid), dtype=float)

    mP, eP, mD, eD = [], [], [], []
    for lam in lam_arr:
        if geometry.lower().startswith("mix"):
            Pmap  = P_map_mixed(Pi, phi, lam, cfg)
            dPmap = dP_map_mixed(Pi, phi, lam, cfg)
        else:
            Pmap  = P_map_separated(Pi, phi, lam, cfg)
            dPmap = dP_map_separated(Pi, phi, lam, cfg)

        kP, EP   = psa_of_map(Pmap,  ring_bins=ring_bins, pad=pad,
                              apodize=True, k_min=k_min, min_counts_per_ring=min_counts_per_ring)
        kD, ED   = psa_of_map(dPmap, ring_bins=ring_bins, pad=pad,
                              apodize=True, k_min=k_min, min_counts_per_ring=min_counts_per_ring)

        mP_i, _, eP_i, _ = fit_log_slope_window(kP, EP)
        mD_i, _, eD_i, _ = fit_log_slope_window(kD, ED)

        mP.append(mP_i); eP.append(eP_i if np.isfinite(eP_i) else np.nan)
        mD.append(mD_i); eD.append(eD_i if np.isfinite(eD_i) else np.nan)

    return lam_arr, np.array(mP), np.array(eP), np.array(mD), np.array(eD)


def plot_psa_slopes_vs_lambda(lam_arr: np.ndarray,
                             mP: np.ndarray, eP: np.ndarray,
                             mD: np.ndarray, eD: np.ndarray,
                             x_is_lambda2: bool = True,
                             title: Optional[str] = None,
                             label_P: str = r"PSA slope of $P$",
                             label_D: str = r"PSA slope of $\\partial P/\\partial\\lambda^2$",
                             sigma_phi: Optional[float] = None):
    """
    Plot the fitted PSA slopes for P and dP/dλ² as a function of χ=2σ_φλ² if sigma_phi is provided,
    otherwise as a function of λ² (or λ).
    """
    if sigma_phi is not None:
        print("SIGMA_PHI", sigma_phi)
        x = 2.0 * (lam_arr**2) * float(sigma_phi)
        xlab = r"$2\,\sigma_\phi\,\lambda^2$"
        mask = np.isfinite(x) & (x > 0) & (x < 20)
    else:
        x = lam_arr**2 if x_is_lambda2 else lam_arr
        xlab = r"$\lambda^2$" if x_is_lambda2 else r"$\lambda$"
        mask = np.isfinite(x)
    ttl = title or (r"PSA slopes" if x_is_lambda2 else r"PSA slopes vs $\\lambda$")

    plt.figure(figsize=(7.2, 5.0))
    # Use error bars if available; fall back to lines if NaNs
    okP = np.isfinite(mP)
    okD = np.isfinite(mD)

    if np.any(np.isfinite(eP)):
        plt.errorbar(x[okP & mask], mP[okP & mask], yerr=eP[okP & mask], fmt='o-', capsize=3, label=label_P)
    else:
        plt.plot(x[okP & mask], mP[okP & mask], 'o-', label=label_P)

    if np.any(np.isfinite(eD)):
        plt.errorbar(x[okD & mask], mD[okD & mask], yerr=eD[okD & mask], fmt='s-', capsize=3, label=label_D)
    else:
        plt.plot(x[okD & mask], mD[okD & mask], 's-', label=label_D)

    plt.axhline(0, color='k', lw=0.6, alpha=0.4)
    plt.xlabel(f"{xlab} (arb. units)")
    plt.ylabel(r"slope $m$ in $E(k)\propto k^{\,m}$")
    plt.title(ttl)
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()


def plot_three_measures_vs_lambda(lam2_pfa: np.ndarray,
                                  pfa_var: np.ndarray,
                                  lam_arr_psa: np.ndarray,
                                  mP: np.ndarray, eP: np.ndarray,
                                  mD: np.ndarray, eD: np.ndarray,
                                  title: str,
                                  sigma_phi: float,
                                  normalize_pfa: bool = True,
                                  outpath: Optional[str] = None):
    """
    Panel figure comparing three measures vs 2σ_φλ²:
      - Panel 1: PSA slopes (P and dP/dλ²)
      - Panel 2: PFA variance (log10, normalized)

    Args:
        lam2_pfa:  array of λ² used for the PFA curve
        pfa_var:   ⟨|P|²⟩(λ²) for the chosen geometry
        lam_arr_psa: λ array used for PSA slopes (from psa_slopes_vs_lambda)
        mP, eP:    PSA slope and 1σ error for P
        mD, eD:    PSA slope and 1σ error for dP/dλ²
        title:     plot title
        sigma_phi: standard deviation of integrated Faraday depth
        normalize_pfa: divide PFA by its value at the smallest λ² before log10
        outpath:   if given, save figure there (dpi=300)
    """
    x_psa = 2 * sigma_phi * lam_arr_psa**2
    x_pfa = 2 * sigma_phi * lam2_pfa

    # Filter to x < 10
    mask_psa = x_psa < 10000
    mask_pfa = x_pfa < 10000
    
    x_psa = x_psa[mask_psa]
    mP = mP[mask_psa]
    eP = eP[mask_psa]
    mD = mD[mask_psa]
    eD = eD[mask_psa]
    
    x_pfa = x_pfa[mask_pfa]
    pfa_var = pfa_var[mask_pfa]

    # Interpolate PFA onto the PSA grid (safer if grids differ)
    y_pfa = np.interp(x_psa, x_pfa, pfa_var, left=np.nan, right=np.nan)

    # Normalize PFA (so it shares a visually comparable scale) and plot in log10
    if normalize_pfa:
        # average the first few λ² points that are firmly in the small-λ plateau
        n0 = max(3, int(0.03 * x_pfa.size))
        ref = np.nanmean(y_pfa[:n0])
        y_pfa = 0.1 * (y_pfa / max(ref, 1e-300))
    y_pfa_plot = np.log10(y_pfa)
    
    # Compute χ(λ) for annotation
    chi = 2.0 * (lam_arr_psa[mask_psa]**2) * sigma_phi

    # Print plateau diagnostics
    print(f"\n[Plateau diagnostics] {title}:")
    print(f"  Small-λ plateau (first 3 points): {np.mean(y_pfa[:3]):.3g}")
    print(f"  Large-λ plateau (last 10 points): {np.mean(y_pfa[-10:]):.3g}")
    print(f"  Plateau ratio: {np.mean(y_pfa[-10:]) / np.mean(y_pfa[:3]):.3g}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    # Panel 1: PSA slopes
    okP = np.isfinite(mP)
    okD = np.isfinite(mD)

    if np.any(np.isfinite(eP)):
        ax1.errorbar(x_psa[okP], mP[okP], yerr=eP[okP], fmt='o-', capsize=3, markersize=2, label=r'PSA slope of $P$')
    else:
        ax1.plot(x_psa[okP], mP[okP], 'o-', markersize=2, label=r'PSA slope of $P$')

    if np.any(np.isfinite(eD)):
        ax1.errorbar(x_psa[okD], mD[okD], yerr=eD[okD], fmt='s-', capsize=3, markersize=2, label=r'PSA slope of $\partial P/\partial\lambda^2$')
    else:
        ax1.plot(x_psa[okD], mD[okD], 's-', markersize=2, label=r'PSA slope of $\partial P/\partial\lambda^2$')

    ax1.axhline(0, color='k', lw=0.6, alpha=0.35)
    ax1.set_ylabel(r'slope $m$ in $E(k)\propto k^{\,m}$')
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend(frameon=False, loc='best')

    # Panel 2: PFA variance (log10 normalized)
    ax2.plot(x_psa, y_pfa_plot, '-', lw=1.4, color='gray', label=r'$\log_{10}\langle |P|^2\rangle$ (PFA, norm.)')
    ax2.set_ylabel(r'$\log_{10}\langle |P|^2\rangle$ (normalized)')
    ax2.set_xlabel(r'$2\sigma_\phi \lambda^2$')
    ax2.grid(True, which='both', alpha=0.25)
    ax2.legend(frameon=False, loc='best')
    
    # Add χ(λ) annotation on secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.semilogx(x_psa, chi, alpha=0.25, color='gray', linestyle='--')
    ax2_twin.axhline(1.0, ls=':', lw=0.8, color='k', alpha=0.7)
    ax2_twin.set_ylabel(r'$\chi(\lambda)=2\,\lambda^2\sigma_\Phi$', color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')

    ax1.set_title(title)
    fig.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300)


def plot_derivative_slope_analysis(lam2_sep: np.ndarray, dvar_sep: np.ndarray,
                                  lam2_mix: np.ndarray, dvar_mix: np.ndarray,
                                  cfg: PFAConfig):
    """
    Plot and analyze derivative measure slopes in the range λ² ∈ [3×10⁻¹, 10¹].
    This focuses on the derivative measure behavior in the specified λ² range.
    """
    # Define the λ² range for analysis
    lam2_min = 0.3  # 3×10⁻¹
    lam2_max = 10.0  # 10¹
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Plot derivative measure for separated case
    ax1.loglog(lam2_sep, dvar_sep, 'o-', markersize=1, label='Separated (screen)')
    
    # Fit slope in the specified range
    m_sep, a_sep = fit_log_slope(lam2_sep, dvar_sep, xmin=lam2_min, xmax=lam2_max)
    if np.isfinite(m_sep):
        lam2_fit = np.array([lam2_min, lam2_max])
        dvar_fit = 10**(a_sep + m_sep * np.log10(lam2_fit))
        ax1.loglog(lam2_fit, dvar_fit, '--', linewidth=2, color='red',
                  label=f'slope = {m_sep:.2f} (λ²∈[0.3,10])')
        # Mark the fitting range
        ax1.axvspan(lam2_min, lam2_max, alpha=0.1, color='red', label='fit range')
    
    ax1.set_xlabel('$\\lambda^2$ (arb. units)')
    ax1.set_ylabel('$\\langle |dP/d\\lambda^2|^2 \\rangle$ (arb. units)')
    ax1.set_title('Derivative Measure: Separated Case')
    ax1.grid(True, which='both', alpha=0.2)
    ax1.legend()
    
    # Plot derivative measure for mixed case
    ax2.loglog(lam2_mix, dvar_mix, 's-', markersize=1, color='darkorange', 
              label='Mixed (in-situ)')
    
    # Fit slope in the specified range
    m_mix, a_mix = fit_log_slope(lam2_mix, dvar_mix, xmin=lam2_min, xmax=lam2_max)
    if np.isfinite(m_mix):
        lam2_fit = np.array([lam2_min, lam2_max])
        dvar_fit = 10**(a_mix + m_mix * np.log10(lam2_fit))
        ax2.loglog(lam2_fit, dvar_fit, '--', linewidth=2, color='red',
                  label=f'slope = {m_mix:.2f} (λ²∈[0.3,10])')
        # Mark the fitting range
        ax2.axvspan(lam2_min, lam2_max, alpha=0.1, color='red', label='fit range')
    
    ax2.set_xlabel('$\\lambda^2$ (arb. units)')
    ax2.set_ylabel('$\\langle |dP/d\\lambda^2|^2 \\rangle$ (arb. units)')
    ax2.set_title('Derivative Measure: Mixed Case')
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    
    # Print analysis results
    print(f"\nDerivative measure slope analysis (λ² ∈ [0.3, 10]):")
    print(f"  Separated case slope: {m_sep:.2f}")
    print(f"  Mixed case slope:     {m_mix:.2f}")
    print(f"  Difference:           {abs(m_mix - m_sep):.2f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if np.isfinite(m_sep) and np.isfinite(m_mix):
        if abs(m_sep) < 0.1:
            print("  → Separated case: nearly flat (expected for external screen)")
        if abs(m_mix) > 0.5:
            print("  → Mixed case: significant slope (shows λ-dependence)")
        print("  → LP16 theory: derivative measure should show different behavior")
        print("    between separated (flat) and mixed (sloped) cases")
    
    return m_sep, m_mix


# -----------------------------
# Runner / demo
# -----------------------------

def run(h5_path: str, force_random_dominated: bool = False, decorrelate: bool = False, beam_sigma_px: float = 0.0, force_sequential: bool = False):
    """
    Run full LP16 PFA + derivative analysis with both one-point and two-point statistics.
    
    Args:
        h5_path: path to HDF5 file with MHD fields
        force_random_dominated: if True, subtract mean B_parallel to force σ_φ >> |φ̄|
    """
    keys = FieldKeys()
    cfg = PFAConfig()

    print("LP16 PFA + DERIVATIVE ANALYSIS")

    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity_lp16(Bx, By, gamma=cfg.gamma)
    
    # Choose B_parallel based on cfg.los_axis (corrected mapping)
    if cfg.los_axis == 0:
        Bpar = Bx
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz
    
    # Option to force random-dominated regime
    if force_random_dominated:
        print("\n⚠ FORCING RANDOM-DOMINATED REGIME (subtracting mean B_∥)")
        Bpar = Bpar - Bpar.mean()
    
    # provisional φ with C=1 for measuring σ_Φ
    phi = faraday_density(ne, Bpar, C=1.0)

    if cfg.use_auto_faraday:
        # integrate φ along LOS to get total Φ per LOS (Ny×Nx map), using Δz=1/Nz
        Nz = Bx.shape[cfg.los_axis]
        dz = 1.0 / float(Nz)
        Phi_tot = np.sum(np.moveaxis(phi, cfg.los_axis, 0), axis=0) * dz
        sigma_Phi = float(Phi_tot.std())
        # choose C so that χ(λ)=2 λ² σ_Φ ≈ 1 at λ² = lam2_break_target
        if sigma_Phi > 0:
            C = 1.0 / (max(cfg.lam2_break_target, 1e-30) * sigma_Phi)
        else:
            C = 1.0
        phi *= C
        print(f"[PFA] σΦ={sigma_Phi:.3g}, C={C:.3g}")
        print(f"[diag] σΦ = {sigma_Phi:.3g} ⇒ λ²_break ~ 1/(2σΦ) = {1/(2*sigma_Phi):.3g}")
    else:
        # fallback: honor user-specified constant
        phi *= cfg.faraday_const
        sigma_Phi = sigma_phi_total(phi, cfg.los_axis)
        print(f"[PFA] Using user-specified faraday_const={cfg.faraday_const}")
        print(f"[diag] σΦ = {sigma_Phi:.3g} ⇒ λ²_break ~ 1/(2σΦ) = {1/(2*sigma_Phi):.3g}")
    
    # Diagnose regime on the *native* φ (so r_i is physical)
    phi_info = compute_faraday_regime(phi, verbose=True)
    
    if decorrelate:
        print("⚙  Decorrelating φ from Pi along LOS (roll; preserves r_i)")
        phi = decorrelate_phi_along_z(phi, cfg.los_axis, seed=42)
    LOS_depth = Pi.shape[cfg.los_axis]
    
    # Choose optimal wavelengths for analysis
    lam_long = choose_lambda_for_regime(phi_info, LOS_depth, target_regime="long")  # small nudge
    lam_short = choose_lambda_for_regime(phi_info, LOS_depth, target_regime="short")
    
    print(f"λ: short={lam_short:.2f}, long={lam_long:.2f}")
    
    L_eff_long = effective_faraday_depth(lam_long, phi_info)
    L_eff_short = effective_faraday_depth(lam_short, phi_info)
    print(f"L_eff: short={L_eff_short:.1f}px, long={L_eff_long:.1f}px (r_i={phi_info['r_i']:.1f}px)")

    # Zhang Fig.2 style: λ² ∈ [1e-3, 1e3] (⇒ 6 decades), use ~180 samples
    lam2 = np.logspace(-3, 3, 180)
    cfg.lam_grid = tuple(np.sqrt(lam2))

    # Use minimal multiprocessing for faster computation
    n_processes = min(11, cpu_count())  # Reduced to 2 processes
    print(f"\nUsing {n_processes} parallel processes for computation")
    
    # Fallback: disable multiprocessing if memory issues occur
    use_multiprocessing = True
    if force_sequential:
        print("⚠ Sequential computation forced by user")
        use_multiprocessing = False
    elif len(cfg.lam_grid) > 50:  # Large grid might cause memory issues
        print("⚠ Large λ-grid detected, using sequential computation to avoid memory issues")
        use_multiprocessing = True#False

    # ========================================
    # Combined three-measures plot only
    # ========================================
    print("\n" + "="*70)
    print("COMPUTING THREE-MEASURES COMPARISON")
    print("="*70)
    print("Computing PFA variance and PSA slopes for combined plot...")
    
    # Compute PFA curves
    print("Computing PFA curves...")
    if use_multiprocessing:
        lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg, n_processes=n_processes)
        lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg, n_processes=n_processes)
    else:
        lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg, n_processes=1)
        lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg, n_processes=1)
    
    # Compute PSA slopes
    print("Computing PSA slopes...")
    lam_subset = cfg.lam_grid[::2]  # Every 2nd point for efficiency
    
    # Mixed case
    lam_arr_mix, mP_mix, eP_mix, mD_mix, eD_mix = psa_slopes_vs_lambda(
        Pi, phi, cfg, geometry="mixed", lam_list=lam_subset,
        ring_bins=24, pad=1, k_min=6.0, min_counts_per_ring=8)
    
    # Separated case
    lam_arr_sep, mP_sep, eP_sep, mD_sep, eD_sep = psa_slopes_vs_lambda(
        Pi, phi, cfg, geometry="separated", lam_list=lam_subset,
        ring_bins=24, pad=1, k_min=6.0, min_counts_per_ring=8)
    
    # Create combined plots
    print("Creating combined three-measures plots...")
    suffix = "_random" if force_random_dominated else ""
    
    # ========================================
    # Part 3.5: Directional-spectrum slopes vs λ (NEW, 4th measure)
    # ========================================
    print("\n" + "="*70)
    print("PART 3.5: Directional-spectrum slopes vs λ")
    print("="*70)
    print("Computing slopes of P_dir(k;λ) = |Â|^2 + |B̂|^2 with A=cos2χ, B=sin2χ ...")

    # Mixed
    lam_arr_dir_m, mDir_mix, eDir_mix = dir_slopes_vs_lambda(
        Pi, phi, cfg, geometry="mixed", lam_list=lam_subset,
        ring_bins=24, pad=1, k_min=6.0, min_counts_per_ring=8)
    plot_dir_slopes_vs_lambda(lam_arr_dir_m, mDir_mix, eDir_mix, x_is_lambda2=True,
                              title=r"Directional-spectrum slope (Mixed)")
    plt.savefig(f"lp2016_outputs/dir_slopes_vs_lambda_mixed{suffix}.png", dpi=300)

    # Separated
    lam_arr_dir_s, mDir_sep, eDir_sep = dir_slopes_vs_lambda(
        Pi, phi, cfg, geometry="separated", lam_list=lam_subset,
        ring_bins=24, pad=1, k_min=6.0, min_counts_per_ring=8)
    plot_dir_slopes_vs_lambda(lam_arr_dir_s, mDir_sep, eDir_sep, x_is_lambda2=True,
                              title=r"Directional-spectrum slope (Separated)")
    plt.savefig(f"lp2016_outputs/dir_slopes_vs_lambda_separated{suffix}.png", dpi=300)

    # Optional overlay of the three k-slope measures vs λ² (PSA of P, PSA of dP/dλ², Directional)
    try:
        plt.figure(figsize=(7.6, 5.0))
        xmix = lam_arr_mix**2
        plt.errorbar(xmix, mP_mix,  eP_mix,  fmt='o-', capsize=3, label=r"PSA slope of $P$")
        plt.errorbar(xmix, mD_mix,  eD_mix,  fmt='s-', capsize=3, label=r"PSA slope of $\partial P/\partial\lambda^2$")
        plt.errorbar(lam_arr_dir_m**2, mDir_mix, eDir_mix, fmt='d-', capsize=3, label=r"Directional slope")
        plt.axhline(0, color='k', lw=0.6, alpha=0.4)
        plt.xlabel(r"$\lambda^2$ (arb. units)"); plt.ylabel(r"slope $m$")
        plt.title(r"Spatial $k$-slope measures (Mixed)")
        plt.grid(True, which='both', alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(f"lp2016_outputs/slopes_three_measures_mixed{suffix}.png", dpi=300)
    except Exception as _e:
        pass
    
    # Mixed geometry
    sigPhi = sigma_phi_total(phi, cfg.los_axis)
    plot_three_measures_vs_lambda(
        lam2_pfa=lam2_mix,
        pfa_var=var_mix,
        lam_arr_psa=lam_arr_mix,
        mP=mP_mix, eP=eP_mix,
        mD=mD_mix, eD=eD_mix,
        title=r"Mixed",
        sigma_phi=sigPhi,
        normalize_pfa=True,
        outpath=f"lp2016_outputs/three_measures_vs_lambda_mixed{suffix}.png"
    )
    print(f"  Saved mixed case to three_measures_vs_lambda_mixed{suffix}.png")
    
    # Separated geometry
    plot_three_measures_vs_lambda(
        lam2_pfa=lam2_sep,
        pfa_var=var_sep,
        lam_arr_psa=lam_arr_sep,
        mP=mP_sep, eP=eP_sep,
        mD=mD_sep, eD=eD_sep,
        title=r"Separated",
        sigma_phi=sigPhi,
        normalize_pfa=True,
        outpath=f"lp2016_outputs/three_measures_vs_lambda_separated{suffix}.png"
    )
    print(f"  Saved separated case to three_measures_vs_lambda_separated{suffix}.png")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nRegime: {phi_info['regime']}")
    print(f"  |φ̄|/σ_φ = {phi_info['ratio']:.2f}")
    
    print(f"\nFour-measures comparison completed:")
    print("   ✓ Generated combined plots showing PSA slopes and PFA variance vs λ²")
    print("   ✓ Added directional spectrum slopes (4th measure)")
    print("   ✓ Mixed and separated geometries compared")
    print("   → Shows how spatial (PSA, directional) and amplitude (PFA) statistics relate")
    
    print(f"\nInterpretation:")
    if phi_info['regime'] == 'random-dominated':
        print("   • Random-dominated regime: derivative PSA should better recover turbulence slope")
        print("   • Look for differences between P and dP/dλ² PSA slopes")
        print("   • PFA variance shows λ-dependence from Faraday rotation effects")
    else:
        print("   • Mean-dominated regime: variance already shows slope ∝ λ^{-(2+2m)}")
        print("   • PSA slopes may be similar between P and dP/dλ²")
        print("   • PFA variance shows strong λ-dependence from mean field effects")
    
    print(f"\nFiles generated:")
    print(f"   • three_measures_vs_lambda_mixed{suffix}.png")
    print(f"   • three_measures_vs_lambda_separated{suffix}.png")
    print(f"   • dir_slopes_vs_lambda_mixed{suffix}.png")
    print(f"   • dir_slopes_vs_lambda_separated{suffix}.png")
    print(f"   • slopes_three_measures_mixed{suffix}.png")
    
    print(f"\nNext steps:")
    print("   • Compare the three-measures plots between mixed and separated geometries")
    print("   • Look for regime transitions where PSA slopes change behavior")
    print("   • Analyze how PFA variance relates to PSA slope evolution")

    
    print("\n" + "="*70)

    #plt.show()


if __name__ == "__main__":
    import sys
    
    print("sys.argv:", sys.argv)
    # Parse command line arguments
    force_random = "--random" in sys.argv or "-r" in sys.argv
    both_regimes = "--both" in sys.argv or "-b" in sys.argv
    decorrelate = "--decorrelate" in sys.argv or "-d" in sys.argv
    sequential = "--sequential" in sys.argv or "-s" in sys.argv
    beam = 0.0
    for a in sys.argv:
        if a.startswith("--beam="):
            try: beam = float(a.split("=")[1])
            except: pass
    
    if both_regimes:
        print("\n" + "="*70)
        print("RUNNING ANALYSIS FOR BOTH REGIMES")
        print("="*70)
        
        print("\n\n" + "#"*70)
        print("# ANALYSIS 1: NATURAL REGIME (as-is)")
        print("#"*70)
        run(H5_PATH, force_random_dominated=False, decorrelate=decorrelate, beam_sigma_px=beam, force_sequential=sequential)
        
        print("\n\n" + "#"*70)
        print("# ANALYSIS 2: FORCED RANDOM-DOMINATED REGIME")
        print("#"*70)
        run(H5_PATH, force_random_dominated=True, decorrelate=decorrelate, beam_sigma_px=beam, force_sequential=sequential)
        
        print("\n\n" + "="*70)
        print("DONE! Check lp2016_outputs/ for comparison:")
        print("  • *_variance.png vs *_variance_random.png")
        print("  • *_spatial.png vs *_spatial_random.png")
        print("="*70)
    else:
        run(H5_PATH, force_random_dominated=force_random, decorrelate=decorrelate, beam_sigma_px=beam, force_sequential=sequential)
    
    # Usage help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("\nUsage:")
        print("  python pfa_and_derivative_lp_16_like.py         # Natural regime")
        print("  python pfa_and_derivative_lp_16_like.py -r      # Force random-dominated")
        print("  python pfa_and_derivative_lp_16_like.py -d      # Decorrelate Pi-Φ")
        print("  python pfa_and_derivative_lp_16_like.py -r -d   # Random + decorrelate")
        print("  python pfa_and_derivative_lp_16_like.py --beam=1.5  # Add beam smoothing")
        print("  python pfa_and_derivative_lp_16_like.py -s      # Force sequential (avoid memory issues)")
        print("  python pfa_and_derivative_lp_16_like.py -b     # Run both for comparison")
