"""
PSA (Polarization Spatial Analysis) spectrum computation following LP16 (Lazarian & Pogosyan 2016)
from a sub-Alfvénic turbulence cube stored in an HDF5 file.

Two projection modes are supported at a single wavelength λ:
1) Separated regions ("external Faraday screen"): synchrotron emission and Faraday rotation occur in
   different volumes (LP16 Appendix C). The observed map is P(X, λ) = e^{2 i λ^2 Φ_screen(X)} ∫ dz P_i(X, z).
2) Mixed volume: synchrotron emission and Faraday rotation occur within the same volume (LP16 main text).
   The observed map is P(X, λ) = ∫ dz P_i(X, z) e^{2 i λ^2 Φ(X, z)}, with Φ(X, z) = ∫_{0}^{z} φ(X, z') dz'.

Here P_i is the complex polarized emissivity density; φ is the Faraday RM density ∝ n_e H_∥.

The code aims to reproduce the PSA spectrum styling used in LP16: log–log, ring-averaged 1D spectrum
from the 2D Fourier power of P(X, λ), with optional slope fits.

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
class PSAConfig:
    lam: float = 1.0  # wavelength (same units as implicit in φ), typical choice here is arbitrary
    gamma: float = 2.0  # synchrotron emissivity exponent for |B_perp|^gamma; gamma=2 gives P_i = (Bx+iBy)^2
    faraday_const: float = 1.0  # proportionality constant for φ = C * n_e * B_parallel
    los_axis: int = 0  # axis index for the line-of-sight (0, 1, or 2)
    voxel_depth: float = 1.0  # Δz in arbitrary units for line integration
    apodize: bool = True  # apply 2D Hann window before FFT
    detrend: bool = True  # subtract mean P before FFT
    ring_bins: int = 64  # number of bins for the radial average
    slope_fit_frac: Tuple[float, float] = (0.1, 0.5)  # fit range as fraction of k span (exclude extreme bins)


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

def project_psa_separated(
    Pi: np.ndarray,
    phi: np.ndarray,
    cfg: PSAConfig,
    emit_bounds: Optional[Tuple[int, int]] = None,
    screen_bounds: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Separated emission and Faraday screen (LP16 Appendix C).

    P(X, λ) = e^{2 i λ^2 Φ_screen(X)} * ∫_{emit_bounds} dz P_i(X, z)

    Args:
        Pi: complex emissivity density, shape (N_los, Ny, Nx) after moving LOS to axis 0
        phi: Faraday density, same shape
        cfg: PSAConfig
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


def project_psa_mixed(
    Pi: np.ndarray,
    phi: np.ndarray,
    cfg: PSAConfig,
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
# 2D→1D spectrum (ring-average PSA)
# ------------------------------------

def _hann2d(ny: int, nx: int) -> np.ndarray:
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return np.outer(wy, wx)


def psa_spectrum(P_map: np.ndarray, cfg: PSAConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute 1D ring-averaged PSA spectrum E(k) from complex P_map.

    Returns (k_centers, E_k, extras) where extras contains kx, ky, and 2D power for optional plotting.
    """
    # Remove mean (detrend)
    P = P_map - (np.mean(P_map) if cfg.detrend else 0.0)

    # Optional apodization window to reduce ringing/leakage
    if cfg.apodize:
        win = _hann2d(P.shape[0], P.shape[1])
        P = P * win

    # 2D FFT and power
    F = nfft.fftshift(nfft.fft2(P))
    P2D = (F * np.conj(F)).real  # |F|^2

    ny, nx = P.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny  # dimensionless spatial frequency index
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY)

    # Radial bins
    kmax = KR.max()
    bins = np.linspace(0.0, kmax, cfg.ring_bins + 1)
    which_bin = np.digitize(KR.ravel(), bins) - 1

    Ek = np.zeros(cfg.ring_bins, dtype=float)
    counts = np.zeros(cfg.ring_bins, dtype=int)

    flatP = P2D.ravel()
    for b in range(cfg.ring_bins):
        mask = which_bin == b
        if np.any(mask):
            Ek[b] = flatP[mask].mean()
            counts[b] = mask.sum()
        else:
            Ek[b] = np.nan

    # Bin centers (exclude the first bin where k=0 by default later)
    kcenters = 0.5 * (bins[:-1] + bins[1:])

    extras = {
        "kx": kx,
        "ky": ky,
        "power2d": P2D,
        "counts": counts,
    }
    return kcenters, Ek, extras


def fit_log_slope(k: np.ndarray, Ek: np.ndarray, frac_range: Tuple[float, float]) -> Tuple[float, float]:
    """Fit slope m in log10(E) = a + m log10(k) over a fractional range of valid (finite) bins."""
    valid = np.isfinite(Ek) & (Ek > 0) & (k > 0)
    kv = k[valid]
    Ev = Ek[valid]
    if kv.size < 8:
        return np.nan, np.nan
    i0 = int(frac_range[0] * kv.size)
    i1 = max(i0 + 8, int(frac_range[1] * kv.size))
    kv = kv[i0:i1]
    Ev = Ev[i0:i1]
    if kv.size < 8:
        return np.nan, np.nan
    X = np.log10(kv)
    Y = np.log10(Ev)
    m, a = np.polyfit(X, Y, 1)
    return m, a


# ----------------
# High-level driver
# ----------------

def compute_psa_from_file(
    h5_path: str,
    keys: FieldKeys = FieldKeys(),
    cfg: PSAConfig = PSAConfig(),
    separated: bool = True,
    emit_bounds: Optional[Tuple[int, int]] = None,
    screen_bounds: Optional[Tuple[int, int]] = None,
    mixed_bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Convenience wrapper: load fields, build P_map for the requested geometry, and compute E(k)."""
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
        P_map = project_psa_separated(Pi, phi, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
    else:
        P_map = project_psa_mixed(Pi, phi, cfg, bounds=mixed_bounds)

    k, Ek, extras = psa_spectrum(P_map, cfg)
    return k, Ek, extras


# --------------
# Plotting helper
# --------------

def plot_psa_spectra(k: np.ndarray, Ek_list: list, labels: list, cfg: PSAConfig, title: str = "PSA @ λ=1", ax=None):
    if ax is None:
        plt.figure(figsize=(6.5, 5.0))
        ax = plt.gca()
    
    for Ek, lab in zip(Ek_list, labels):
        ax.loglog(k[1:], Ek[1:], label=lab, color="blue")  # skip k=0 bin

    # Optional slope fits for the first spectrum
    if Ek_list:
        m, a = fit_log_slope(k[1:], Ek_list[0][1:], cfg.slope_fit_frac)
        if np.isfinite(m):
            # Draw a guide line with the fitted slope through a mid-point
            mid = np.nanmedian(k[np.isfinite(Ek_list[0]) & (k > 0)])
            ymid = 10**(a + m * np.log10(mid))
            kline = np.array([mid/4, mid*4])
            yline = ymid * (kline / mid)**m
            ax.loglog(kline, yline, linestyle='--', linewidth=1, label=f"slope: {m:.2f}", color="red")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
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
    lambda_values = [np.sqrt(0.5/2/sigma_phi), np.sqrt(2/2/sigma_phi), np.sqrt(4/2/sigma_phi)]
    
    # Load fields once to calculate sigma_phi
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    
    # Calculate sigma_phi from the Faraday rotation measure
    # Following the same approach as lp2016_synchrotron_sim_sep.py
    cfg_temp = PSAConfig(los_axis=0)
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
    
    for i, lam in enumerate(lambda_values):
        cfg = PSAConfig(lam=lam, gamma=2.0, los_axis=0)
        
        # Separated case: screen = first 10% slices; emission = remaining 90%
        k_sep, Ek_sep, _ = compute_psa_from_file(h5_path, keys, cfg, separated=True)
        
        # Mixed case: whole depth participates
        k_mix, Ek_mix, _ = compute_psa_from_file(h5_path, keys, cfg, separated=False)
        
        # Calculate 2*lambda^2*sigma_phi for labeling
        # lambda_sigma_phi = 2 * lam**2 * sigma_phi
        
        # Plot separated case (top row)
        ax_top = axes[0, i]
        plot_psa_spectra(k_sep, [Ek_sep], [f"Separated ($2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$)"], cfg,
                        title=fr"Separated $2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$", ax=ax_top)
        
        # Plot mixed case (bottom row)
        ax_bottom = axes[1, i]
        plot_psa_spectra(k_mix, [Ek_mix], [f"Mixed ($2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$)"], cfg,
                        title=fr"Mixed $2\\lambda^2 \\sigma_{{\\Phi}}={2*lam**2*sigma_phi:.1f}$", ax=ax_bottom)
    
    plt.tight_layout()
    plt.savefig("lp2016_outputs/psa_spectra.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Example: set your path then run `python psa_lp16_like.py`
    H5_PATH = r"..\\faradays_angles_stats\\lp_structure_tests\\ms01ma08.mhd_w.00300.vtk.h5"
    demo(H5_PATH)
