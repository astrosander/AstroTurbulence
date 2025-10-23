"""
PFA (Polarization Frequency Analysis) following LP16 (Lazarian & Pogosyan 2016)
from a sub-Alfvénic turbulence cube stored in an HDF5 file.

Two geometries at a grid of wavelengths λ (user may choose λ=1.0 or a range):
1) External Faraday screen (separated regions): emission and Faraday rotation occur
   in different volumes (LP16 Appendix C). Resulting |P(λ)|^2 is independent of λ.
2) Mixed volume: emission and Faraday rotation occur within the same volume
   (LP16 main text); here |P(λ)|^2 generally decreases with λ and exhibits
   asymptotic power laws described in LP16/PFA tests.

This script computes ⟨|P(λ)|^2⟩ vs λ^2 (the PFA “spectrum”) and styles the plot
as in LP16: log–log axes, slope fit, and a marker at λ=1.0.

Author: (you)
"""

from __future__ import annotations
import h5py
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Iterable
import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

# -----------------------------
# Data I/O and field primitives
# -----------------------------

@dataclass
class FieldKeys:
    bx: str = "i_mag_field"
    by: str = "j_mag_field"
    bz: str = "k_mag_field"
    ne: str = "gas_density"

@dataclass
class PFAConfig:
    faraday_const: float = 1.0   # φ = C * n_e * B_parallel
    lam_grid: Iterable[float] = tuple(np.geomspace(0.05, 5.0, 40))
    lam_mark: float = 1.0        # highlight this λ on the curve
    gamma: float = 2.0           # emissivity exponent (|B_perp|^γ); γ=2 ⇒ P_i = (Bx+iBy)^2
    los_axis: int = 0            # which array axis is LOS integration axis
    voxel_depth: float = 1.0     # Δz in arbitrary units
    detrend_emit: bool = True    # subtract mean of P_emit in separated case before forming variance


def load_fields(h5_path: str, keys: FieldKeys) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        Bx = f[keys.bx][()]
        By = f[keys.by][()]
        Bz = f[keys.bz][()]
        ne = f[keys.ne][()]
    return Bx, By, Bz, ne


# -----------------------------------------
# Synchrotron emissivity & Faraday quantities
# -----------------------------------------

def polarized_emissivity(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    B_perp2 = Bx**2 + By**2
    eps = np.finfo(B_perp2.dtype).eps
    phase_term = (Bx + 1j * By)**2
    if gamma == 2.0:
        return phase_term
    amp = np.power(np.maximum(B_perp2, eps), 0.5 * (gamma - 2.0))
    return amp * phase_term


def faraday_density(ne: np.ndarray, B_parallel: np.ndarray, C: float = 1.0) -> np.ndarray:
    return C * ne * B_parallel


# -------------------------------------
# P(λ) maps for two geometries
# -------------------------------------

def _move_los(arr: np.ndarray, los_axis: int) -> np.ndarray:
    return np.moveaxis(arr, los_axis, 0)


def P_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                    emit_bounds: Optional[Tuple[int, int]] = None,
                    screen_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """External Faraday screen: P = e^{2 i λ^2 Φ_screen(X)} * ∫ Pi dz (no internal rotation).
    Note |P| is independent of λ; we retain λ for completeness and consistency.
    """
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz, Ny, Nx = Pi_los.shape

    if emit_bounds is None:
        emit_bounds = (0, Nz)
    if screen_bounds is None:
        scr_N = max(1, int(0.1 * Nz))
        screen_bounds = (0, scr_N)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    P_emit = np.sum(Pi_los[z0e:z1e, :, :], axis=0) * cfg.voxel_depth
    if cfg.detrend_emit:
        P_emit = P_emit - P_emit.mean()

    Phi_screen = np.sum(phi_los[z0s:z1s, :, :], axis=0) * cfg.voxel_depth
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)


def P_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Mixed emission & Faraday rotation in same volume: P = ∫ Pi e^{2 i λ^2 Φ(z)} dz."""
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz, Ny, Nx = Pi_los.shape
    if bounds is None:
        bounds = (0, Nz)
    z0, z1 = bounds

    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    phase = np.exp(2j * (lam**2) * Phi_cum)
    contrib = Pi_los[z0:z1, :, :] * phase
    P_map = np.sum(contrib, axis=0) * cfg.voxel_depth
    return P_map


# -----------------------------
# PFA: variance vs λ^2
# -----------------------------

def pfa_curve_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                        emit_bounds: Optional[Tuple[int, int]] = None,
                        screen_bounds: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        P = P_map_separated(Pi, phi, lam, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
        var.append(np.mean(np.abs(P)**2))  # ⟨|P|^2⟩ over map (average many LOS)
    return lam_grid**2, np.array(var)


def pfa_curve_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                    bounds: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        P = P_map_mixed(Pi, phi, lam, cfg, bounds=bounds)
        var.append(np.mean(np.abs(P)**2))
    return lam_grid**2, np.array(var)


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


def pfa_curve_derivative_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                                   emit_bounds: Optional[Tuple[int, int]] = None,
                                   screen_bounds: Optional[Tuple[int, int]] = None):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        dP = dP_map_separated(Pi, phi, lam, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
        var.append(np.mean(np.abs(dP)**2))
    return lam_grid**2, np.array(var)


def pfa_curve_derivative_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                               bounds: Optional[Tuple[int, int]] = None):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        dP = dP_map_mixed(Pi, phi, lam, cfg, bounds=bounds)
        var.append(np.mean(np.abs(dP)**2))
    return lam_grid**2, np.array(var)


def fit_log_slope(x: np.ndarray, y: np.ndarray, xmin: float = 1.0, xmax: float = 30.0) -> Tuple[float, float]:
    """Fit slope over a specific x range (xmin to xmax)."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= xmin) & (x <= xmax)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 4:
        return np.nan, np.nan
    m, a = np.polyfit(np.log10(xv), np.log10(yv), 1)
    return m, a


# --------------
# High-level driver
# --------------

def compute_pfa_from_file(h5_path: str,
                          keys: FieldKeys = FieldKeys(),
                          cfg: PFAConfig = PFAConfig(),
                          separated: bool = False,
                          emit_bounds: Optional[Tuple[int, int]] = None,
                          screen_bounds: Optional[Tuple[int, int]] = None,
                          mixed_bounds: Optional[Tuple[int, int]] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Load data, build Pi and φ, and compute PFA curve (λ^2 vs ⟨|P|^2⟩)."""
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)

    # Choose B_parallel for Faraday density according to LOS axis
    if cfg.los_axis == 0:
        Bpar = Bz
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bx
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    if separated:
        return pfa_curve_separated(Pi, phi, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
    else:
        return pfa_curve_mixed(Pi, phi, cfg, bounds=mixed_bounds)


# --------------
# Plotting helper
# --------------

def plot_pfa(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
             title: str = "PFA (variance) vs $2\\lambda^2\\sigma_{\\Phi}$", sigma_phi: float = 1.9101312160491943):
             
    plt.figure(figsize=(6.5, 5.0))
    
    # Convert λ² to 2λ²σ_Φ
    x_axis = 2 * lam2 * sigma_phi
    
    for var, lab in zip(var_list, labels):
        plt.loglog(x_axis, var, label=lab)

    # Slope guide for the mixed case (second curve)
    if len(var_list) > 1:
        m, a = fit_log_slope(x_axis, var_list[1], xmin=1.0, xmax=30.0)  # Use mixed case (index 1)
        if np.isfinite(m):
            # Draw fitted line over the fitting range (1 to 30)
            xline = np.linspace(1.0, 30.0, 100)
            yline = 10**(a + m * np.log10(xline))
            plt.loglog(xline, yline, '--', linewidth=2, label=f"slope: {m:.2f}", color="red")

    # Mark λ = lam_mark
    lam_mark2 = cfg.lam_mark**2
    x_mark = 2 * lam_mark2 * sigma_phi
    for var, lab in zip(var_list, labels):
        # interpolate value at 2λ²σ_Φ
        vmark = np.interp(x_mark, x_axis, var)
        # plt.scatter([x_mark], [vmark], marker='o')

    plt.xlabel(r"$2\lambda^2\sigma_{\Phi}$")
    plt.ylabel(r"$\langle |P|^2 \rangle$")
    plt.title(title)
    
    # Set meaningful x-axis ticks for 2λ²σ_Φ
    x_min, x_max = x_axis.min(), x_axis.max()
    # Create ticks at round numbers: 0.1, 0.5, 1, 2, 5, 10, etc.
    tick_values = []
    for exp in range(-2, 3):  # 0.01 to 100
        for mult in [1, 2, 5]:
            val = mult * 10**exp
            if x_min <= val <= x_max:
                tick_values.append(val)
    
    if tick_values:
        plt.xticks(tick_values, [f"{val:.1f}" if val < 1 else f"{val:.0f}" for val in tick_values])
    
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def plot_pfa_derivative(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
                        title: str = r"Derivative measure $\langle|dP/d\lambda^2|^2\rangle$ vs $2\lambda^2\sigma_{\Phi}$",
                        sigma_phi: float = 1.9101312160491943):
    plt.figure(figsize=(6.5, 5.0))
    
    # Convert λ² to 2λ²σ_Φ
    x_axis = 2 * lam2 * sigma_phi
    
    for var, lab in zip(var_list, labels):
        plt.loglog(x_axis, var, label=lab)
    
    # Slope guide for the mixed case (second curve)
    if len(var_list) > 1:
        m, a = fit_log_slope(x_axis, var_list[1], xmin=1.0, xmax=30.0)  # Use mixed case (index 1)
        if np.isfinite(m):
            # Draw fitted line over the fitting range (1 to 30)
            xline = np.linspace(1.0, 30.0, 100)
            yline = 10**(a + m * np.log10(xline))
            plt.loglog(xline, yline, '--', linewidth=2, label=f"slope: {m:.2f}", color="red")
    
    # Mark λ = lam_mark
    lam_mark2 = cfg.lam_mark**2
    x_mark = 2 * lam_mark2 * sigma_phi
    for var, lab in zip(var_list, labels):
        # interpolate value at 2λ²σ_Φ
        vmark = np.interp(x_mark, x_axis, var)
        # plt.scatter([x_mark], [vmark], marker='o')
    
    plt.xlabel(r"$2\lambda^2\sigma_{\Phi}$")
    plt.ylabel(r"$\langle |dP/d\lambda^2|^2 \rangle$")
    plt.title(title)
    
    # Set meaningful x-axis ticks for 2λ²σ_Φ
    x_min, x_max = x_axis.min(), x_axis.max()
    # Create ticks at round numbers: 0.1, 0.5, 1, 2, 5, 10, etc.
    tick_values = []
    for exp in range(-2, 3):  # 0.01 to 100
        for mult in [1, 2, 5]:
            val = mult * 10**exp
            if x_min <= val <= x_max:
                tick_values.append(val)
    
    if tick_values:
        plt.xticks(tick_values, [f"{val:.1f}" if val < 1 else f"{val:.0f}" for val in tick_values])
    
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def demo(h5_path: str):
    keys = FieldKeys()
    cfg = PFAConfig(lam_grid=np.geomspace(0.05, 5.0, 50), lam_mark=1.0, los_axis=0)

    # Build Pi and φ once
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)
    Bpar = Bz  # assuming LOS along axis-0 uses Bz as parallel component
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    # Derivative measure only
    lam2_dsep, dvar_sep = pfa_curve_derivative_separated(Pi, phi, cfg)
    lam2_dmix, dvar_mix = pfa_curve_derivative_mixed(Pi, phi, cfg)
    assert np.allclose(lam2_dsep, lam2_dmix)
    plot_pfa_derivative(lam2_dsep, [dvar_sep, dvar_mix], ["separated", "mixed"], cfg,
                        title="Derivative measure $\\langle|dP/d\\lambda^2|^2\\rangle$ vs $2\\lambda^2\\sigma_{\\Phi}$")
    plt.savefig("lp2016_outputs/derivative_spectra.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    H5_PATH = r"..\\faradays_angles_stats\\lp_structure_tests\\ms01ma08.mhd_w.00300.vtk.h5"
    demo(H5_PATH)