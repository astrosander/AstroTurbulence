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
    # P_i ∝ (B_x + i B_y)^{(γ+1)/2}
    p = (gamma + 1.0) / 2.0
    complex_B = Bx + 1j * By
    return np.power(complex_B, p)


def faraday_density(ne: np.ndarray, Bpar: np.ndarray, C: float = 0.81) -> np.ndarray:
    return C * ne * Bpar


def _move_los(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(arr, axis, 0)

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

# -----------------------------
# Runner / demo
# -----------------------------

def run(h5_path: str):
    keys = FieldKeys()
    cfg = PFAConfig()

    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)
    # LOS along axis 0 → B_parallel = Bz by our key choice; change if you rotate LOS.
    Bpar = Bz
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    # Use multiprocessing for faster computation
    n_processes = min(11, cpu_count())
    print(f"Using {n_processes} parallel processes for computation")

    # PFA
    lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg, n_processes=n_processes)
    lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg, n_processes=n_processes)

    # Derivative measure
    lam2_dsep, dvar_sep = pfa_curve_derivative_separated(Pi, phi, cfg, n_processes=n_processes)
    lam2_dmix, dvar_mix = pfa_curve_derivative_mixed(Pi, phi, cfg, n_processes=n_processes)

    # Combined plot
    plot_combined_pfa_and_derivative(lam2_sep, [var_sep, var_mix], 
                                    lam2_dsep, [dvar_sep, dvar_mix],
                                    ["Separated", "Mixed"], cfg)

    plt.savefig("lp2016_outputs/pfa_and_derivative_spectra.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run(H5_PATH)
