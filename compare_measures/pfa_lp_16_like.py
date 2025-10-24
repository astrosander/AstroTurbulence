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
from multiprocessing import Pool, cpu_count
import functools

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
    lam_grid: Iterable[float] = tuple(np.sqrt(np.geomspace(0.05**2, 5.0**2, 500)))
    lam_mark: float = 1.0        # highlight this λ on the curve
    gamma: float = 2.0           # emissivity exponent (|B_perp|^γ); γ=2 ⇒ P_i = (Bx+iBy)^2
    los_axis: int = 0            # which array axis is LOS integration axis
    voxel_depth: float = 1.0     # Δz in arbitrary units
    detrend_emit: bool = True    # subtract mean of P_emit in separated case before forming variance


def load_fields(h5_path: str, keys: FieldKeys) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        # Load all fields at once to minimize file access
        Bx = np.asarray(f[keys.bx], dtype=np.float32)  # Use float32 for memory efficiency
        By = np.asarray(f[keys.by], dtype=np.float32)
        Bz = np.asarray(f[keys.bz], dtype=np.float32)
        ne = np.asarray(f[keys.ne], dtype=np.float32)
    return Bx, By, Bz, ne


# -----------------------------------------
# Synchrotron emissivity & Faraday quantities
# -----------------------------------------

def polarized_emissivity(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    LP16 emissivity:
      P_i = (Bx + i By)^2 * |B_perp|^(gamma-3)/2
    For gamma=2: P_i = (Bx + i By)^2 / |B_perp|
    """
    B2 = Bx**2 + By**2
    eps = np.finfo(B2.dtype).tiny
    amp = np.power(np.maximum(B2, eps), 0.5*(gamma - 3.0))
    return ((Bx + 1j*By)**2 * amp).astype(np.complex128)


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

def _compute_separated_single(args):
    """Helper function for parallel computation of single lambda value"""
    lam2, P_emit, Phi_screen = args
    P_map = P_emit * np.exp(2j * lam2 * Phi_screen)
    return np.mean(np.abs(P_map)**2)


def pfa_curve_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                        emit_bounds: Optional[Tuple[int, int]] = None,
                        screen_bounds: Optional[Tuple[int, int]] = None,
                        n_processes: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    
    # Pre-compute P_emit and Phi_screen once
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
    
    # Parallel computation for all lambda values
    lam2_grid = lam_grid**2
    args_list = [(lam2, P_emit, Phi_screen) for lam2 in lam2_grid]
    
    with Pool(processes=n_processes) as pool:
        var = pool.map(_compute_separated_single, args_list)
    
    return lam2_grid, np.array(var)


def _compute_mixed_single(args):
    """Helper function for parallel computation of single lambda value"""
    lam2, Pi_los, Phi_cum, z0, z1, voxel_depth = args
    # Calculate phase factor for this lambda
    phase = np.exp(2j * lam2 * Phi_cum)
    # Multiply emissivity by phase and integrate along LOS
    contrib = Pi_los[z0:z1, :, :] * phase
    P_map = np.sum(contrib, axis=0) * voxel_depth
    # Calculate variance (mean of |P|^2)
    return np.mean(np.abs(P_map)**2)


def pfa_curve_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                    bounds: Optional[Tuple[int, int]] = None,
                    n_processes: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    
    # Pre-compute arrays once for efficiency
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz, Ny, Nx = Pi_los.shape
    if bounds is None:
        bounds = (0, Nz)
    z0, z1 = bounds

    # Pre-compute cumulative Faraday depth for efficiency
    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    
    # Parallel computation for all lambda values
    lam2_grid = lam_grid**2
    args_list = [(lam2, Pi_los, Phi_cum, z0, z1, cfg.voxel_depth) for lam2 in lam2_grid]
    
    with Pool(processes=n_processes) as pool:
        var = pool.map(_compute_mixed_single, args_list)
    
    return lam2_grid, np.array(var)


def fit_log_slope(x: np.ndarray, y: np.ndarray, frac: Tuple[float, float] = (0.3, 0.8)) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 10:
        return np.nan, np.nan
    i0 = int(frac[0] * xv.size)
    i1 = max(i0 + 10, int(frac[1] * xv.size))
    xv = xv[i0:i1]
    yv = yv[i0:i1]
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
                          n_processes: int = 11,
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
        return pfa_curve_separated(Pi, phi, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds, n_processes=n_processes)
    else:
        return pfa_curve_mixed(Pi, phi, cfg, bounds=mixed_bounds, n_processes=n_processes)


# --------------
# Plotting helper
# --------------

def plot_pfa(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
             title: str = "PFA (variance) vs $2\\lambda^2\\sigma_\\phi$", 
             sigma_phi: float = 1.9101312160491943):
    plt.figure(figsize=(6.5, 5.0))
    
    # Convert lambda^2 to 2*lambda^2*sigma_phi
    x_axis = 2 * lam2 * sigma_phi
    
    for var, lab in zip(var_list, labels):
        plt.loglog(x_axis, var, label=lab)

    # Slope guide for the mixed case (second curve if available, otherwise first)
    if var_list:
        # Use mixed case (second curve) if available, otherwise first curve
        curve_idx = 1 if len(var_list) > 1 else 0
        
        xmin = 1.0
        xmax = 40.0
        # Filter data for the specified range: 1 to 40 in new units (2*lambda^2*sigma_phi)
        range_mask = (x_axis >= xmin) & (x_axis <= xmax)
        
        if np.any(range_mask) and np.sum(range_mask) >= 3:
            x_filtered = x_axis[range_mask]
            y_filtered = var_list[curve_idx][range_mask]
            m, a = fit_log_slope(x_filtered, y_filtered)
        else:
            # Fall back to using all data if range is too restrictive
            m, a = fit_log_slope(x_axis, var_list[curve_idx])
            
        if np.isfinite(m):
            # Use the actual range bounds (1 to 40) for the slope line
            xline = np.array([xmin, xmax])
            yline = 10**(a + m * np.log10(xline))
            plt.loglog(xline, yline, '--', linewidth=2, color='red', label=f"slope: {m:.2f}")

    # Mark λ = lam_mark
    lam_mark2 = cfg.lam_mark**2
    lam_mark_x = 2 * lam_mark2 * sigma_phi
    for var, lab in zip(var_list, labels):
        # interpolate value at the marked point
        vmark = np.interp(lam_mark_x, x_axis, var)
        # plt.scatter([lam_mark_x], [vmark], marker='o')

    plt.xlabel(r"$2\lambda^2\sigma_\phi$")
    plt.ylabel(r"$\langle |P|^2 \rangle$")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def validate_spectrum_accuracy(lam2: np.ndarray, var: np.ndarray, case_name: str) -> bool:
    """Validate that the spectrum has reasonable properties"""
    # Check for finite values
    if not np.all(np.isfinite(var)):
        print(f"Warning: {case_name} has non-finite values")
        return False
    
    # Check for monotonic behavior (mixed case should generally decrease)
    if case_name == "Mixed" and len(var) > 3:
        # Check if variance generally decreases (allowing for some noise)
        diff = np.diff(np.log(var))
        decreasing_fraction = np.sum(diff < 0) / len(diff)
        if decreasing_fraction < 0.3:  # Less than 30% decreasing
            print(f"Warning: {case_name} spectrum may not be monotonic enough")
    
    # Check for reasonable range
    if np.max(var) / np.min(var) < 1.1:  # Less than 10% variation
        print(f"Warning: {case_name} spectrum has very little variation")
        return False
        
    return True


def demo(h5_path: str):
    keys = FieldKeys()
    cfg = PFAConfig(lam_grid=np.sqrt(np.geomspace(0.1**2, 10.0**2, 500)), lam_mark=1.0, los_axis=0)

    # Build Pi and φ once
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)
    Bpar = Bz  # assuming LOS along axis-0 uses Bz as parallel component
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    # Use 11 processes or available CPU count, whichever is smaller
    n_processes = min(11, cpu_count())
    print(f"Using {n_processes} parallel processes for computation")
    
    lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg, n_processes=n_processes)
    lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg, n_processes=n_processes)

    # Validate spectrum accuracy
    validate_spectrum_accuracy(lam2_sep, var_sep, "Separated")
    validate_spectrum_accuracy(lam2_mix, var_mix, "Mixed")

    assert np.allclose(lam2_sep, lam2_mix)
    plot_pfa(lam2_sep, [var_sep, var_mix], ["Separated", "Mixed"], cfg,
             title="Polarization Frequency Analysis (PFA): $\\langle |P|^2 \\rangle$", 
             sigma_phi=1.9101312160491943)
    plt.savefig("lp2016_outputs/pfa_spectra.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    H5_PATH = r"..\\faradays_angles_stats\\lp_structure_tests\\ms01ma08.mhd_w.00300.vtk.h5"
    demo(H5_PATH)
