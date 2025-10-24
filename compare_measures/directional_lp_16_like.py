"""
Directional spectrum computation following LP16 (Lazarian & Pogosyan 2016)
from a sub-Alfvénic turbulence cube stored in an HDF5 file.

Two projection modes are supported at a single wavelength λ:
1) Separated regions ("external Faraday screen"): synchrotron emission and Faraday rotation occur in
   different volumes (LP16 Appendix C). The observed map is P(X, λ) = e^{2 i λ^2 Φ_screen(X)} ∫ dz P_i(X, z).
2) Mixed volume: synchrotron emission and Faraday rotation occur within the same volume (LP16 main text).
   The observed map is P(X, λ) = ∫ dz P_i(X, z) e^{2 i λ^2 Φ(X, z)}, with Φ(X, z) = ∫_{0}^{z} φ(X, z') dz'.

Here P_i is the complex polarized emissivity density; φ is the Faraday RM density ∝ n_e H_∥.

The code computes the directional spectrum P_dir(k) = |FFT(cos 2χ)|² + |FFT(sin 2χ)|², where
χ = 0.5·arg(P) is the polarization angle. This measure is sensitive to the spatial structure
of magnetic field orientations.

Author: (you)
"""

from __future__ import annotations
import h5py
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

@dataclass
class FieldKeys:
    bx: str = "i_mag_field"
    by: str = "j_mag_field"
    bz: str = "k_mag_field"
    ne: str = "gas_density"

@dataclass
class DirectionalConfig:
    lam: float = 1.0  # wavelength (same units as implicit in φ), typical choice here is arbitrary
    gamma: float = 2.0  # synchrotron emissivity exponent for |B_perp|^gamma; gamma=2 gives P_i = (Bx+iBy)^2
    faraday_const: float = 1.0  # proportionality constant for φ = C * n_e * B_parallel
    los_axis: int = 0  # axis index for the line-of-sight (0, 1, or 2)
    voxel_depth: float = 1.0  # Δz in arbitrary units for line integration
    ring_bins: int = 64  # number of bins for the radial average
    slope_fit_kmin: float = None  # minimum k for slope fitting (None = auto)
    slope_fit_kmax: float = None  # maximum k for slope fitting (None = auto, uses first half)


def load_fields(h5_path: str, keys: FieldKeys) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Bx, By, Bz, n_e arrays from an HDF5 file.

    Returns arrays in their native shape (Nz, Ny, Nx) or similar; orientation handled later.
    """
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
    """Complex polarized emissivity density P_i.

    For general gamma: P_i = |B_perp|^{gamma-2} * (Bx + i By)^2, which ensures amplitude ∝ |B_perp|^gamma
    and polarization angle 2χ set by the projected B direction. For gamma=2 this reduces to (Bx + i By)^2.
    """
    B_perp2 = Bx**2 + By**2
    # Avoid division by zero by adding tiny epsilon
    eps = np.finfo(B_perp2.dtype).eps
    phase_term = (Bx + 1j * By)**2
    if gamma == 2.0:
        return phase_term
    amp = np.power(np.maximum(B_perp2, eps), 0.5 * (gamma - 2.0))
    return amp * phase_term


def faraday_density(ne: np.ndarray, B_parallel: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Faraday RM density φ ∝ n_e * B_parallel. Units are arbitrary here (C absorbs physical constants)."""
    return C * ne * B_parallel


# -------------------------------------
# Projection: P(X, λ) maps for 2 cases
# -------------------------------------

def project_separated(
    Pi: np.ndarray,
    phi: np.ndarray,
    cfg: DirectionalConfig,
    emit_bounds: Optional[Tuple[int, int]] = None,
    screen_bounds: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Separated emission and Faraday screen (LP16 Appendix C).

    P(X, λ) = e^{2 i λ^2 Φ_screen(X)} * ∫_{emit_bounds} dz P_i(X, z)

    Args:
        Pi: complex emissivity density, shape (N_los, Ny, Nx) after moving LOS to axis 0
        phi: Faraday density, same shape
        cfg: DirectionalConfig
        emit_bounds: (z0, z1) indices [inclusive, exclusive) for emission region along LOS
        screen_bounds: (z0, z1) for the Faraday-rotating screen along LOS
    Returns:
        Complex 2D polarization map P(X, λ)
    """
    Pi_los = np.moveaxis(Pi, cfg.los_axis, 0)
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz, Ny, Nx = Pi_los.shape

    if emit_bounds is None:
        emit_bounds = (0, Nz)
    if screen_bounds is None:
        # default: first 10% of slab acts as screen
        scr_N = max(1, int(0.1 * Nz))
        screen_bounds = (0, scr_N)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    # Emission integral (no internal Faraday rotation in the emitting region in this separated case)
    P_emit = np.sum(Pi_los[z0e:z1e, :, :], axis=0) * cfg.voxel_depth

    # Screen RM Φ_screen = ∫ φ dz over the screen bounds
    Phi_screen = np.sum(phi_los[z0s:z1s, :, :], axis=0) * cfg.voxel_depth

    # Observed P map at λ
    P_map = P_emit * np.exp(2j * (cfg.lam**2) * Phi_screen)
    return P_map


def project_mixed(
    Pi: np.ndarray,
    phi: np.ndarray,
    cfg: DirectionalConfig,
    bounds: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Mixed emission + Faraday rotation in the same volume.

    P(X, λ) = ∫_{bounds} dz P_i(X, z) e^{2 i λ^2 Φ(X, z)}, with Φ the cumulative RM to depth z.
    """
    Pi_los = np.moveaxis(Pi, cfg.los_axis, 0)
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz, Ny, Nx = Pi_los.shape
    if bounds is None:
        bounds = (0, Nz)
    z0, z1 = bounds

    # Cumulative Φ along LOS (prefix sum)
    # Φ[k] = ∑_{0..k} φ * Δz
    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)

    # Contribution at each z layer with its own Faraday phase
    phase = np.exp(2j * (cfg.lam**2) * Phi_cum)
    contrib = Pi_los[z0:z1, :, :] * phase

    P_map = np.sum(contrib, axis=0) * cfg.voxel_depth
    return P_map


# ------------------------------------
# Directional spectrum computation
# ------------------------------------

def map_angles(P: np.ndarray) -> np.ndarray:
    """Polarization angle χ = 0.5 * arg(P)."""
    return 0.5 * np.angle(P)


def ring_average_power2d(field2d: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (k_centers, P1D(k)) from isotropic ring-average of |FFT(field2d)|^2."""
    ny, nx = field2d.shape
    F = np.fft.rfft2(field2d)
    P2 = np.abs(F)**2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.rfftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX**2 + KY**2)
    kmax = 0.5 * min(nx, ny)
    k_edges = np.linspace(0.0, kmax, nbins+1)
    idx = np.digitize(KR.ravel(), k_edges) - 1
    idx = np.clip(idx, 0, nbins-1)
    counts = np.bincount(idx, minlength=nbins)
    sums = np.bincount(idx, weights=P2.ravel(), minlength=nbins)
    P1D = np.zeros(nbins, dtype=float)
    valid = counts > 0
    P1D[valid] = sums[valid] / counts[valid]
    k_centers = 0.5*(k_edges[:-1] + k_edges[1:])
    return k_centers, P1D


def directional_spectrum(P_map: np.ndarray, cfg: DirectionalConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute 1D ring-averaged directional spectrum P_dir(k) from complex P_map.
    
    P_dir(k) = |FFT(cos 2χ)|² + |FFT(sin 2χ)|², where χ = 0.5·arg(P).
    
    Returns (k_centers, P_dir, extras) where extras contains the 2D fields for optional plotting.
    """
    chi = map_angles(P_map)
    
    # Compute cos and sin components
    A = np.cos(2.0 * chi)
    B = np.sin(2.0 * chi)
    
    # Ring average each component's power spectrum
    kA, PA = ring_average_power2d(A, cfg.ring_bins)
    kB, PB = ring_average_power2d(B, cfg.ring_bins)
    
    # Combined directional spectrum
    Pdir = PA + PB
    
    extras = {
        "chi": chi,
        "cos_2chi": A,
        "sin_2chi": B,
        "PA": PA,
        "PB": PB,
    }
    return kA, Pdir, extras


def fit_log_slope(k: np.ndarray, Ek: np.ndarray, kmin: float, kmax: float) -> Tuple[float, float]:
    """Fit slope m in log10(E) = a + m log10(k) over a specific k range."""
    valid = np.isfinite(Ek) & (Ek > 0) & (k > 0)
    kv = k[valid]
    Ev = Ek[valid]
    if kv.size < 8:
        return np.nan, np.nan
    
    # Select k range based on kmin and kmax
    mask = (kv >= kmin) & (kv <= kmax)
    kv_range = kv[mask]
    Ev_range = Ev[mask]
    
    if kv_range.size < 8:
        return np.nan, np.nan
    
    X = np.log10(kv_range)
    Y = np.log10(Ev_range)
    m, a = np.polyfit(X, Y, 1)
    return m, a


# ----------------
# High-level driver
# ----------------

def compute_directional_from_file(
    h5_path: str,
    keys: FieldKeys = FieldKeys(),
    cfg: DirectionalConfig = DirectionalConfig(),
    separated: bool = True,
    emit_bounds: Optional[Tuple[int, int]] = None,
    screen_bounds: Optional[Tuple[int, int]] = None,
    mixed_bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Convenience wrapper: load fields, build P_map for the requested geometry, and compute P_dir(k)."""
    Bx, By, Bz, ne = load_fields(h5_path, keys)

    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)
    # Choose B_parallel along LOS
    if cfg.los_axis == 0:
        Bpar = Bz  # LOS along axis-0 means z is LOS
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bx

    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    if separated:
        P_map = project_separated(Pi, phi, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
    else:
        P_map = project_mixed(Pi, phi, cfg, bounds=mixed_bounds)

    k, Pdir, extras = directional_spectrum(P_map, cfg)
    return k, Pdir, extras


# --------------
# Plotting helper
# --------------

def plot_directional_spectra(k: np.ndarray, Pdir_list: list, labels: list, cfg: DirectionalConfig, title: str = "Directional @ λ=1", ax=None):
    if ax is None:
        plt.figure(figsize=(6.5, 5.0))
        ax = plt.gca()
    
    for Pdir, lab in zip(Pdir_list, labels):
        ax.loglog(k[1:], Pdir[1:], label=lab, color="blue")  # skip k=0 bin

    # Optional slope fits for the first spectrum
    if Pdir_list:
        # Auto-calculate k range if not specified
        k_valid = k[1:]  # skip k=0 bin
        if cfg.slope_fit_kmin is None:
            kmin_auto = k_valid.min()
        else:
            kmin_auto = cfg.slope_fit_kmin
            
        if cfg.slope_fit_kmax is None:
            kmax_auto = k_valid.max() / 4  # first third of k range
        else:
            kmax_auto = cfg.slope_fit_kmax
            
        m, a = fit_log_slope(k[1:], Pdir_list[0][1:], kmin_auto, kmax_auto)
        if np.isfinite(m):
            # Draw the fitted line over the same range used for fitting
            k_fit_range = np.linspace(kmin_auto, kmax_auto, 100)
            y_fit_range = 10**(a + m * np.log10(k_fit_range))
            ax.loglog(k_fit_range, y_fit_range, linestyle='--', linewidth=1.5, 
                     label=f"slope: {m:.2f}", color="red")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P_{\mathrm{dir}}(k)$")
    # ax.set_title(title)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend()
    
    if ax is None:  # Only call tight_layout if we created the figure
        plt.tight_layout()


def demo(h5_path: str):
    """Example usage with defaults similar to the user's keys and λ=1.0.
    Adjust los_axis if your LOS is not the first dimension.
    """

    sigma_phi = 1.9101312160491943
    
    keys = FieldKeys()
    lambda_values = [np.sqrt(0.01/2/sigma_phi), np.sqrt(2/2/sigma_phi), np.sqrt(4/2/sigma_phi)]
    
    # Load fields once to calculate sigma_phi
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    
    # Calculate sigma_phi from the Faraday rotation measure
    # Following the same approach as lp2016_synchrotron_sim_sep.py
    cfg_temp = DirectionalConfig(los_axis=0)
    if cfg_temp.los_axis == 0:
        Bpar = Bz
    elif cfg_temp.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bx
    
    # Calculate Faraday depth density (same as faraday_depth_increment in main sim)
    K_RM = 0.81  # rad m^-2 per (cm^-3 μG pc) - same constant as main sim
    phi_inc = K_RM * ne * Bpar * cfg_temp.voxel_depth
    
    # Move LOS to axis 0 for consistency
    phi_inc_los = np.moveaxis(phi_inc, cfg_temp.los_axis, 0)
    Nz = phi_inc_los.shape[0]
    
    # For separated case, use first 10% as screen region (matching main sim logic)
    scr_N = max(1, int(0.1 * Nz))
    phi_inc_screen = phi_inc_los[:scr_N, :, :]
    
    # Calculate cumulative Faraday depth (same as cumulative_phi in main sim)
    phi_cum = np.cumsum(phi_inc_screen, axis=0)
    phi_total_map = phi_cum[-1]  # to far boundary
    
    # Create 2x3 subplot layout (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle('Directional spectrum $P_{{\\mathrm{{dir}}}}(k)$\n$P_{{\\mathrm{{dir}}}}(k) = |\\mathrm{{FFT}}(\\cos 2\\chi)|^2 + |\\mathrm{{FFT}}(\\sin 2\\chi)|^2$, where $\\chi = \\frac{1}{2}\\arg(P)$', 
                 fontsize=14, y=0.98)
    
    # Move panel to the right by 1%
    plt.subplots_adjust(left=0.11)
    
    for i, lam in enumerate(lambda_values):
        cfg = DirectionalConfig(lam=lam, gamma=2.0, los_axis=0)
        
        # Separated case: screen = first 10% slices; emission = remaining 90%
        k_sep, Pdir_sep, _ = compute_directional_from_file(h5_path, keys, cfg, separated=True)
        
        # Mixed case: whole depth participates
        k_mix, Pdir_mix, _ = compute_directional_from_file(h5_path, keys, cfg, separated=False)
        
        # Calculate 2*lambda^2*sigma_phi for labeling
        # lambda_sigma_phi = 2 * lam**2 * sigma_phi
        
        # Plot separated case (top row)
        ax_top = axes[0, i]
        plot_directional_spectra(k_sep, [Pdir_sep], [f"Separated ($2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$)"], cfg,
                        title=fr"Separated $2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$", ax=ax_top)
        
        # Plot mixed case (bottom row)
        ax_bottom = axes[1, i]
        plot_directional_spectra(k_mix, [Pdir_mix], [f"Mixed ($2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$)"], cfg,
                        title=fr"Mixed $2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$", ax=ax_bottom)
    
    # Add vertical text labels for rows
    fig.text(0.0075, 0.75, 'Separated', rotation=90, fontsize=12,  
             ha='center', va='center', color='red')
    fig.text(0.0075, 0.25, 'Mixed', rotation=90, fontsize=12,  
             ha='center', va='center', color='red')
    
    plt.tight_layout()
    plt.savefig("lp2016_outputs/directional_spectra.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Example: set your path then run `python directional_lp_16_like.py`
    H5_PATH = r"..\\faradays_angles_stats\\lp_structure_tests\\ms01ma08.mhd_w.00300.vtk.h5"
    demo(H5_PATH)
