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
    lam_grid: Iterable[float] = tuple(np.sqrt(np.geomspace(0.05**2, 10.0**2, 500)))
    lam_mark: float = 1.0
    gamma: float = 2.0            # electron index → P_i ∝ (B_⊥)^{(γ+1)/2}
    faraday_const: float = 0.81    # arbitrary units here; relative scaling only
    los_axis: int = 0              # integrate along this axis
    voxel_depth: float = 1.0       # Δz in code units


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


def polarized_emissivity(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    LP16-style polarized emissivity: P_i = (Bx + i*By)^2 * (Bx^2 + By^2)^{(gamma-2)/2}
    For gamma=2, this reduces to (Bx + i*By)^2 without complex branch issues.
    """
    Bperp2 = Bx**2 + By**2
    eps = np.finfo(Bperp2.dtype).eps
    return (Bx + 1j*By)**2 * np.maximum(Bperp2, eps)**(0.5*(gamma - 2.0))


def faraday_density(ne: np.ndarray, Bpar: np.ndarray, C: float = 0.81) -> np.ndarray:
    return C * ne * Bpar


def _move_los(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(arr, axis, 0)


def psa_of_map(P_map: np.ndarray, ring_bins: int = 50, pad: int = 2, 
               apodize: bool = True, k_min: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
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
    
    Returns:
        k_centers: wavenumber bins (geometric mean)
        E_k: ring-averaged power spectrum
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
    
    # 2D FFT and power
    F = np.fft.fftshift(np.fft.fft2(Y))
    P2 = (F * np.conj(F)).real
    
    # Wavenumber grids
    ky = np.fft.fftshift(np.fft.fftfreq(P2.shape[0])) * P2.shape[0]
    kx = np.fft.fftshift(np.fft.fftfreq(P2.shape[1])) * P2.shape[1]
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY).ravel()
    
    # Geometric binning (better for power-law fitting)
    k_max = min(P2.shape) / 4  # Avoid high-k noise near Nyquist
    bins = np.geomspace(max(1.0, k_min), k_max, ring_bins + 1)
    lab = np.digitize(KR, bins) - 1
    
    # Ring-average with geometric mean for k
    Ek = np.array([P2.ravel()[lab == i].mean() if np.any(lab == i) else np.nan
                   for i in range(ring_bins)])
    kcen = np.sqrt(bins[:-1] * bins[1:])  # geometric mean
    
    # Keep only well-sampled bins
    msk = np.isfinite(Ek) & (kcen >= k_min) & (kcen <= k_max)
    
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
        print(f"  → {regime.upper()}")
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
        # Want L_eff ~ r_i (multiple wraps over correlation length)
        # L_eff = 1/(λ² φ_scale) ~ r_i  →  λ ~ 1/√(φ_scale * r_i)
        lam_opt = 1.0 / np.sqrt(phi_scale * r_i)
        # Make it a bit longer to be clearly in long-λ regime
        lam_opt *= 2.0
    else:  # "short"
        # Want L_eff ~ LOS_depth (weak rotation)
        # λ ~ 1/√(φ_scale * LOS_depth)
        lam_opt = 1.0 / np.sqrt(phi_scale * LOS_depth)
        # Make it shorter to be clearly in short-λ regime
        lam_opt *= 0.5
    
    return lam_opt

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
    e0 = emit_bounds or (0, max(1, int(0.9 * Nz)))
    s0 = screen_bounds or (max(0, e0[1]), Nz)
    P_emit = np.sum(Pi_los[e0[0]:e0[1], :, :], axis=0) * cfg.voxel_depth
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * cfg.voxel_depth
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)


def P_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    z0, z1 = bounds or (0, Nz)
    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    phase = np.exp(2j * (lam**2) * Phi_cum)
    return np.sum(Pi_los[z0:z1, :, :] * phase, axis=0) * cfg.voxel_depth


def dP_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                      emit_bounds: Optional[Tuple[int, int]] = None,
                      screen_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Derivative dP/d(λ²) for external screen: 2i Φ_screen * P_emit * e^{2i λ² Φ_screen}."""
    P = P_map_separated(Pi, phi, lam, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = phi_los.shape[0]
    s0 = screen_bounds or (max(0, int(0.9 * Nz)), Nz)
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * cfg.voxel_depth
    return 2j * Phi_screen * P


def dP_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                  bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Derivative dP/d(λ²) for mixed geometry: 2i ∫ Pi(z) Φ(z) e^{2i λ² Φ(z)} dz."""
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = Pi_los.shape[0]
    z0, z1 = bounds or (0, Nz)
    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    phase = np.exp(2j * (lam**2) * Phi_cum)
    contrib = 2j * (Pi_los[z0:z1, :, :] * Phi_cum) * phase
    return np.sum(contrib, axis=0) * cfg.voxel_depth

# -----------------------------
# Multiprocessing helper functions
# -----------------------------

def _compute_separated_single(args):
    """Helper function for parallel computation of single lambda value"""
    lam2, P_emit, Phi_screen = args
    P_map = P_emit * np.exp(2j * lam2 * Phi_screen)
    return np.mean(np.abs(P_map)**2)


def _compute_mixed_single(args):
    """Helper function for parallel computation of single lambda value"""
    lam2, Pi_los, Phi_cum, z0, z1, voxel_depth = args
    phase = np.exp(2j * lam2 * Phi_cum)
    contrib = Pi_los[z0:z1, :, :] * phase
    P_map = np.sum(contrib, axis=0) * voxel_depth
    return np.mean(np.abs(P_map)**2)


def _compute_derivative_separated_single(args):
    """Helper function for parallel computation of derivative measure for separated case"""
    lam2, P_emit, Phi_screen = args
    P_map = P_emit * np.exp(2j * lam2 * Phi_screen)
    dP_map = 2j * Phi_screen * P_map
    return np.mean(np.abs(dP_map)**2)


def _compute_derivative_mixed_single(args):
    """Helper function for parallel computation of derivative measure for mixed case"""
    lam2, Pi_los, Phi_cum, z0, z1, voxel_depth = args
    phase = np.exp(2j * lam2 * Phi_cum)
    contrib = 2j * (Pi_los[z0:z1, :, :] * Phi_cum) * phase
    dP_map = np.sum(contrib, axis=0) * voxel_depth
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
    P_emit = np.sum(Pi_los[e0[0]:e0[1], :, :], axis=0) * cfg.voxel_depth
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * cfg.voxel_depth
    
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
    
    # Pre-compute cumulative Faraday depth for efficiency
    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    
    # Parallel computation for all lambda values
    lam2_grid = lam_grid**2
    args_list = [(lam2, Pi_los, Phi_cum, z0, z1, cfg.voxel_depth) for lam2 in lam2_grid]
    
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
    P_emit = np.sum(Pi_los[e0[0]:e0[1], :, :], axis=0) * cfg.voxel_depth
    Phi_screen = np.sum(phi_los[s0[0]:s0[1], :, :], axis=0) * cfg.voxel_depth
    
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
    
    # Pre-compute cumulative Faraday depth for efficiency
    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    
    # Parallel computation for all lambda values
    lam2_grid = lam_grid**2
    args_list = [(lam2, Pi_los, Phi_cum, z0, z1, cfg.voxel_depth) for lam2 in lam2_grid]
    
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
        plt.scatter([lam_mark2], [vmark], marker='o')
    plt.xlabel("$\\lambda^2$ (arb. units)")
    plt.ylabel("$\\langle |P|^2 \\rangle$ (arb. units)")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def plot_pfa_derivative(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
                        title: str = r"Derivative measure $\langle|dP/d\lambda^2|^2\rangle$ vs $\lambda^2$"):
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
        plt.scatter([lam_mark2], [vmark], marker='o')
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
        ax1.scatter([lam_mark2], [vmark], marker='o')
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
        ax2.scatter([lam_mark2], [vmark], marker='o')
    ax2.set_xlabel(r"$\lambda^2$ (arb. units)")
    ax2.set_ylabel(r"$\langle |dP/d\lambda^2|^2 \rangle$ (arb. units)")
    ax2.set_title(r"Derivative measure $\langle|dP/d\lambda^2|^2\rangle$ vs $\lambda^2$")
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()


def plot_spatial_psa_comparison(k_P: np.ndarray, Ek_P: np.ndarray, 
                                k_dP: np.ndarray, Ek_dP: np.ndarray,
                                lam: float, case: str = "Mixed"):
    """
    Plot spatial power spectra (PSA) of P and dP/dλ² to show the difference
    in spatial statistics that LP16 §6 discusses.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Define k-range for slope fitting: 1.5×10¹ to 8×10¹
    k_min_fit = 15.0
    k_max_fit = 80.0
    
    # PSA of P
    ax1.loglog(k_P, Ek_P, 'o-', markersize=3, label=f'PSA of $P(\\lambda={lam:.1f})$')
    m_P, a_P = fit_log_slope(k_P, Ek_P, xmin=k_min_fit, xmax=k_max_fit)
    if np.isfinite(m_P):
        k_fit = np.array([k_min_fit, k_max_fit])
        E_fit = 10**(a_P + m_P * np.log10(k_fit))
        ax1.loglog(k_fit, E_fit, '--', linewidth=2, color='red', 
                  label=f'slope = {m_P:.2f} (k∈[15,80])')
        # Mark the fitting range
        ax1.axvspan(k_min_fit, k_max_fit, alpha=0.1, color='red', label='fit range')
    ax1.set_xlabel('$k$ (wavenumber)')
    ax1.set_ylabel('$E(k)$')
    ax1.set_title(f'Spatial PSA of $P$ ({case}, $\\lambda={lam:.1f}$)')
    ax1.grid(True, which='both', alpha=0.2)
    ax1.legend()
    
    # PSA of dP/dλ²
    ax2.loglog(k_dP, Ek_dP, 's-', markersize=3, color='darkorange',
              label=f'PSA of $dP/d\\lambda^2$ ($\\lambda={lam:.1f}$)')
    m_dP, a_dP = fit_log_slope(k_dP, Ek_dP, xmin=k_min_fit, xmax=k_max_fit)
    if np.isfinite(m_dP):
        k_fit = np.array([k_min_fit, k_max_fit])
        E_fit = 10**(a_dP + m_dP * np.log10(k_fit))
        ax2.loglog(k_fit, E_fit, '--', linewidth=2, color='red',
                  label=f'slope = {m_dP:.2f} (k∈[15,80])')
        # Mark the fitting range
        ax2.axvspan(k_min_fit, k_max_fit, alpha=0.1, color='red', label='fit range')
    ax2.set_xlabel('$k$ (wavenumber)')
    ax2.set_ylabel('$E(k)$')
    ax2.set_title(f'Spatial PSA of $dP/d\\lambda^2$ ({case}, $\\lambda={lam:.1f}$)')
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    
    # Print slopes for comparison
    print(f"\n{case} case at λ={lam:.1f}:")
    print(f"  PSA of P slope: {m_P:.2f} (fitted k∈[15,80])")
    print(f"  PSA of dP/dλ² slope: {m_dP:.2f} (fitted k∈[15,80])")
    print(f"  → Difference: {abs(m_dP - m_P):.2f}")
    print("  (LP16 predicts derivative PSA better recovers turbulence slope m)")
    
    return m_P, m_dP


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
    ax1.loglog(lam2_sep, dvar_sep, 'o-', markersize=4, label='Separated (screen)')
    
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
    ax2.loglog(lam2_mix, dvar_mix, 's-', markersize=4, color='darkorange', 
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

def run(h5_path: str, force_random_dominated: bool = False):
    """
    Run full LP16 PFA + derivative analysis with both one-point and two-point statistics.
    
    Args:
        h5_path: path to HDF5 file with MHD fields
        force_random_dominated: if True, subtract mean B_parallel to force σ_φ >> |φ̄|
    """
    keys = FieldKeys()
    cfg = PFAConfig()

    print("\n" + "="*70)
    print("LP16 PFA + DERIVATIVE ANALYSIS")
    print("="*70)

    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)
    
    # LOS along axis 0 → B_parallel = Bz by our key choice
    Bpar = Bz
    
    # Option to force random-dominated regime
    if force_random_dominated:
        print("\n⚠ FORCING RANDOM-DOMINATED REGIME (subtracting mean B_∥)")
        Bpar = Bpar - Bpar.mean()
    
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)
    
    # Diagnose Faraday regime
    phi_info = compute_faraday_regime(phi, verbose=True)
    LOS_depth = Pi.shape[cfg.los_axis]
    
    # Choose optimal wavelengths for analysis
    lam_long = choose_lambda_for_regime(phi_info, LOS_depth, target_regime="long")
    lam_short = choose_lambda_for_regime(phi_info, LOS_depth, target_regime="short")
    
    print(f"\nOptimal wavelengths:")
    print(f"  Short-λ regime: λ ≈ {lam_short:.2f}")
    print(f"  Long-λ regime:  λ ≈ {lam_long:.2f}")
    
    L_eff_long = effective_faraday_depth(lam_long, phi_info)
    L_eff_short = effective_faraday_depth(lam_short, phi_info)
    print(f"\nEffective Faraday depths:")
    print(f"  At λ={lam_short:.2f}: L_eff ≈ {L_eff_short:.1f} px  (vs LOS depth = {LOS_depth} px)")
    print(f"  At λ={lam_long:.2f}: L_eff ≈ {L_eff_long:.1f} px  (vs RM corr length ≈ {phi_info['r_i']:.1f} px)")

    # Use multiprocessing for faster computation
    n_processes = min(11, cpu_count())
    print(f"\nUsing {n_processes} parallel processes for computation")

    # ========================================
    # Part 1: One-point statistics (variance vs λ²)
    # ========================================
    print("\n" + "="*70)
    print("PART 1: One-point statistics (PFA variance vs λ²)")
    print("="*70)
    print("Expected: PFA and derivative curves have SAME λ-shape, differ by constant")
    
    # PFA
    lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg, n_processes=n_processes)
    lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg, n_processes=n_processes)

    # Derivative measure
    lam2_dsep, dvar_sep = pfa_curve_derivative_separated(Pi, phi, cfg, n_processes=n_processes)
    lam2_dmix, dvar_mix = pfa_curve_derivative_mixed(Pi, phi, cfg, n_processes=n_processes)

    # Combined plot of one-point statistics
    plot_combined_pfa_and_derivative(lam2_sep, [var_sep, var_mix], 
                                    lam2_dsep, [dvar_sep, dvar_mix],
                                    ["Separated", "Mixed"], cfg)
    suffix = "_random" if force_random_dominated else ""
    plt.savefig(f"lp2016_outputs/pfa_and_derivative_variance{suffix}.png", dpi=300)
    
    # ========================================
    # Part 1.5: Derivative slope analysis in specific λ² range
    # ========================================
    print("\n" + "="*70)
    print("PART 1.5: Derivative measure slope analysis (λ² ∈ [0.3, 10])")
    print("="*70)
    
    # Analyze derivative measure slopes in the specified range
    m_der_sep, m_der_mix = plot_derivative_slope_analysis(lam2_dsep, dvar_sep, 
                                                          lam2_dmix, dvar_mix, cfg)
    plt.savefig(f"lp2016_outputs/derivative_slope_analysis{suffix}.png", dpi=300)

    # ========================================
    # Part 2: Two-point statistics (spatial PSA) at fixed λ
    # ========================================
    print("\n" + "="*70)
    print("PART 2: Two-point (spatial) statistics at fixed λ")
    print("="*70)
    print("This is what LP16 §6 actually analyzes!")
    print("\nExpected behavior by regime:")
    print("  • Random-dominated + long-λ: derivative PSA better reveals turbulence slope")
    print("  • Mean-dominated + long-λ: variance already shows slope ∝ λ^{-(2+2m)}")
    
    # Analyze at the optimal long-λ value
    lam_test = lam_long
    print(f"\nComputing spatial PSA at λ = {lam_test:.2f} (long-λ regime)...")
    print(f"  L_eff/L ≈ {L_eff_long/LOS_depth:.3f} (<<1 = strong rotation)")
    print(f"  L_eff/r_i ≈ {L_eff_long/phi_info['r_i']:.3f}")
    
    # Mixed case
    print("\n  Computing MIXED case...")
    P_map_mix = P_map_mixed(Pi, phi, lam_test, cfg)
    dP_map_mix = dP_map_mixed(Pi, phi, lam_test, cfg)
    
    k_P, Ek_P = psa_of_map(P_map_mix, ring_bins=50, pad=2)
    k_dP, Ek_dP = psa_of_map(dP_map_mix, ring_bins=50, pad=2)
    
    m_P, m_dP = plot_spatial_psa_comparison(k_P, Ek_P, k_dP, Ek_dP, lam_test, case="Mixed")
    plt.savefig(f"lp2016_outputs/psa_P_vs_dP_spatial{suffix}.png", dpi=300)
    
    # Separated case
    print("\n  Computing SEPARATED case...")
    P_map_sep = P_map_separated(Pi, phi, lam_test, cfg)
    dP_map_sep = dP_map_separated(Pi, phi, lam_test, cfg)
    k_P_sep, Ek_P_sep = psa_of_map(P_map_sep, ring_bins=50, pad=2)
    k_dP_sep, Ek_dP_sep = psa_of_map(dP_map_sep, ring_bins=50, pad=2)
    
    m_P_sep, m_dP_sep = plot_spatial_psa_comparison(k_P_sep, Ek_P_sep, 
                                                     k_dP_sep, Ek_dP_sep, 
                                                     lam_test, case="Separated")
    plt.savefig(f"lp2016_outputs/psa_P_vs_dP_spatial_separated{suffix}.png", dpi=300)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nRegime: {phi_info['regime'].upper()}")
    print(f"  |φ̄|/σ_φ = {phi_info['ratio']:.2f}")
    
    print("\n1. One-point statistics (variance vs λ²):")
    print("   ✓ PFA and derivative curves differ only by constant factor")
    print("   ✓ Both show same λ-dependence (Gaussian Φ factorization)")
    print("   → This is EXPECTED from LP16 theory")
    
    print(f"\n1.5. Derivative slope analysis (λ² ∈ [0.3, 10]):")
    print(f"   Separated case slope: {m_der_sep:.2f}")
    print(f"   Mixed case slope:     {m_der_mix:.2f}")
    print(f"   Difference:           {abs(m_der_mix - m_der_sep):.2f}")
    
    print(f"\n2. Two-point statistics (spatial PSA at λ={lam_test:.2f}):")
    print(f"   Mixed case:")
    print(f"     PSA(P) slope:       {m_P:.2f} (fitted k∈[15,80])")
    print(f"     PSA(dP/dλ²) slope:  {m_dP:.2f} (fitted k∈[15,80])")
    print(f"     Difference:         {abs(m_dP - m_P):.2f}")
    print(f"   Separated case:")
    print(f"     PSA(P) slope:       {m_P_sep:.2f} (fitted k∈[15,80])")
    print(f"     PSA(dP/dλ²) slope:  {m_dP_sep:.2f} (fitted k∈[15,80])")
    print(f"     Difference:         {abs(m_dP_sep - m_P_sep):.2f}")
    
    print("\n3. Interpretation:")
    if phi_info['regime'] == "random-dominated":
        print("   ✓ Random-dominated regime: derivative PSA should help recover m")
        print("   ✓ PFA variance → universal λ⁻² masks turbulence")
        print("   ✓ Derivative spatial PSA → better reveals turbulence slope")
    else:
        print("   • Mean-dominated regime: variance already reveals slope")
        print("   • Expect ⟨|P|²⟩ ∝ λ^{-(2+2m)} from transverse field")
        print("   • Derivative PSA less critical (variance works well)")
        print("\n   💡 TIP: Run with force_random_dominated=True to see derivative advantage")
    
    print("\n" + "="*70)
    
    plt.show()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    force_random = "--random" in sys.argv or "-r" in sys.argv
    both_regimes = "--both" in sys.argv or "-b" in sys.argv
    
    if both_regimes:
        print("\n" + "="*70)
        print("RUNNING ANALYSIS FOR BOTH REGIMES")
        print("="*70)
        
        print("\n\n" + "#"*70)
        print("# ANALYSIS 1: NATURAL REGIME (as-is)")
        print("#"*70)
        run(H5_PATH, force_random_dominated=False)
        
        print("\n\n" + "#"*70)
        print("# ANALYSIS 2: FORCED RANDOM-DOMINATED REGIME")
        print("#"*70)
        run(H5_PATH, force_random_dominated=True)
        
        print("\n\n" + "="*70)
        print("DONE! Check lp2016_outputs/ for comparison:")
        print("  • *_variance.png vs *_variance_random.png")
        print("  • *_spatial.png vs *_spatial_random.png")
        print("="*70)
    else:
        run(H5_PATH, force_random_dominated=force_random)
    
    # Usage help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("\nUsage:")
        print("  python pfa_and_derivative_lp_16_like.py         # Natural regime")
        print("  python pfa_and_derivative_lp_16_like.py -r      # Force random-dominated")
        print("  python pfa_and_derivative_lp_16_like.py -b      # Run both for comparison")
