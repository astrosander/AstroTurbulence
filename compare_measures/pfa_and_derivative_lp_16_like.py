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
import matplotlib as mpl
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

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
    lam_grid: Iterable[float] = tuple(np.geomspace(0.1, 3.1, 30))
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
# PFA curves (variance over sky vs λ²)
# -----------------------------

def pfa_curve_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                        emit_bounds: Optional[Tuple[int, int]] = None,
                        screen_bounds: Optional[Tuple[int, int]] = None):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        P = P_map_separated(Pi, phi, lam, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
        var.append(np.mean(np.abs(P)**2))
    return lam_grid**2, np.array(var)


def pfa_curve_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                    bounds: Optional[Tuple[int, int]] = None):
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        P = P_map_mixed(Pi, phi, lam, cfg, bounds=bounds)
        var.append(np.mean(np.abs(P)**2))
    return lam_grid**2, np.array(var)


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

# -----------------------------
# Plotting (LP16-style log–log)
# -----------------------------

def fit_log_slope(x: np.ndarray, y: np.ndarray, xmin: float = 0.2, xmax: float = 10.0) -> Tuple[float, float]:
    """Fit slope over a specific x range (xmin to xmax)."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= xmin) & (x <= xmax)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 4:
        return np.nan, np.nan
    m, a = np.polyfit(np.log10(xv), np.log10(yv), 1)
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
    
    # Slope guide for the mixed case (second curve)
    if len(var_list) > 1:
        m, a = fit_log_slope(lam2, var_list[1], xmin=0.2, xmax=10.0)  # Use mixed case (index 1)
        if np.isfinite(m):
            # Draw fitted line over the fitting range
            xline = np.linspace(0.2, 10.0, 100)
            yline = 10**(a + m * np.log10(xline))
            plt.loglog(xline, yline, '--', linewidth=2, label=f"slope: {m:.2f}", color="red")
    
    # Mark λ = lam_mark
    lam_mark2 = cfg.lam_mark**2
    for var, lab in zip(var_list, labels):
        # interpolate value at λ²
        vmark = np.interp(lam_mark2, lam2, var)
        # plt.scatter([lam_mark2], [vmark], marker='o')
    
    plt.xlabel(r"$\lambda^2$")
    plt.ylabel(r"$\langle |dP/d\lambda^2|^2 \rangle$")
    plt.title(title)
    
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
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

    # Derivative measure only
    lam2_dsep, dvar_sep = pfa_curve_derivative_separated(Pi, phi, cfg)
    lam2_dmix, dvar_mix = pfa_curve_derivative_mixed(Pi, phi, cfg)
    plot_pfa_derivative(lam2_dsep, [dvar_sep, dvar_mix], ["separated", "mixed"], cfg,
                        title="Derivative measure $\\langle|dP/d\\lambda^2|^2\\rangle$")
    plt.savefig("lp2016_outputs/derivative_spectra.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run(H5_PATH)
