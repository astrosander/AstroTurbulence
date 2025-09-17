#!/usr/bin/env python3
"""
Polarization Angle Structure Function from Energy Spectrum

This script computes the polarization angle structure function D_φ(R) from the 
energy spectrum E(k) of magnetic field data. It combines the energy spectrum 
calculation approach from energy_spectrum.py with structure function methodology.

The workflow:
1. Load magnetic field data from HDF5 files
2. Compute energy spectrum E(k) from the FFT of the field
3. Convert to 2D power spectrum via isotropy assumption
4. Compute Faraday depth correlation from power spectrum
5. Calculate polarization angle structure function D_φ(R)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
from numpy.fft import fftn, fftshift, fftfreq, fft2, ifft2
from scipy.special import j0

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Data sources
    h5_athena: str = "mhd_fields.h5"
    h5_synthetic: str = "two_slope_2D_s3_r00.h5"
    
    # Field dataset name
    field_dataset: str = "k_mag_field"
    
    # Physical parameters
    C_RM: float = 0.81  # Rotation measure coefficient
    wavelengths_m: Tuple[float, ...] = (0.21,0.1,0.05,)  # Observing wavelengths
    
    # Spectrum calculation
    nbins_spectrum: int = 128
    
    # Structure function calculation  
    nbins_R: int = 300
    R_min_pix: float = 1.0
    R_max_frac: float = 0.45
    
    # Output
    output_dir: str = "fig/polarization_angle_sf"
    dpi: int = 160
    
    # Analysis ranges for slope fitting
    k_ranges: dict = None
    
    def __post_init__(self):
        if self.k_ranges is None:
            self.k_ranges = {
                'athena_low': (20/256, None),  # Will be set dynamically
                'athena_high': (None, None),   # Will be set dynamically  
                'synthetic_low': (20/256, None),
                'synthetic_high': (None, None)
            }

C = Config()

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────
def _dx(a, f=1.0):
    """Extract grid spacing from coordinate array."""
    try:
        u = np.unique(np.asarray(a).ravel())
        d = np.diff(np.sort(u))
        d = d[d > 0]
        if d.size:
            return float(np.median(d))
    except:
        pass
    return float(f)

def load_field(path: str, dataset: str):
    """Load field data from HDF5 file."""
    with h5py.File(path, "r") as f:
        x = f[dataset][:].astype(float)
        # Extract grid spacing (currently hardcoded to 1.0)
        dx = dy = _dx(f["x_coor"][:,0,0]) if "x_coor" in f else 1.0
        dz = _dx(f["z_coor"][0,0,:]) if "z_coor" in f else 1.0
    
    # Override with unit spacing for consistency
    dx = dy = dz = 1.0
    return x, (dx, dy, dz)

def load_density_and_field(path: str):
    """Load both density and magnetic field from HDF5."""
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = dz = 1.0  # Unit spacing
    return ne, bz, dx, dz

# ─────────────────────────────────────────────────────────────
# Energy Spectrum Calculation
# ─────────────────────────────────────────────────────────────
def compute_energy_spectrum(x, spacing, nbins=128):
    """
    Compute energy spectrum E(k) from 3D field data.
    
    Parameters:
    -----------
    x : ndarray
        3D field data
    spacing : tuple
        Grid spacing (dx, dy, dz)
    nbins : int
        Number of bins for radial averaging
        
    Returns:
    --------
    k_centers : ndarray
        Wavenumber bin centers
    E_k : ndarray  
        Energy spectrum E(k)
    variance : float
        Field variance
    """
    x = x - x.mean()
    F = fftn(x)
    P = np.abs(F)**2
    
    if x.ndim == 3:
        nz, ny, nx = x.shape
        dx, dy, dz = spacing
        kx = 2*np.pi*fftfreq(nx, dx)
        ky = 2*np.pi*fftfreq(ny, dy) 
        kz = 2*np.pi*fftfreq(nz, dz)
        KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        N = nx * ny * nz
    else:
        ny, nx = x.shape
        dx, dy, _ = spacing
        kx = 2*np.pi*fftfreq(nx, dx)
        ky = 2*np.pi*fftfreq(ny, dy)
        KY, KX = np.meshgrid(ky, kx, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)
        N = nx * ny
    
    K = fftshift(K)
    P = fftshift(P) / (N**2)
    
    # Radial binning
    k_flat = K.ravel()
    p_flat = P.ravel()
    k_pos = k_flat[np.isfinite(k_flat) & (k_flat > 0)]
    k_min, k_max = k_pos.min(), k_pos.max()
    
    bins = np.logspace(np.log10(k_min), np.log10(k_max), nbins + 1)
    idx = np.digitize(k_flat, bins) - 1
    good = np.isfinite(p_flat) & (idx >= 0) & (idx < nbins)
    
    sums = np.bincount(idx[good], weights=p_flat[good], minlength=nbins)
    counts = np.bincount(idx[good], minlength=nbins).astype(float)
    
    k_centers = 0.5 * (bins[1:] + bins[:-1])
    dk = np.diff(bins)
    
    with np.errstate(invalid="ignore", divide="ignore"):
        E_k = sums / dk
    
    # Normalize to preserve variance
    valid = np.isfinite(E_k)
    if np.any(valid) and np.trapz(E_k[valid], k_centers[valid]) > 0:
        norm_factor = x.var() / np.trapz(E_k[valid], k_centers[valid])
    else:
        norm_factor = 1.0
        
    return k_centers, E_k * norm_factor, x.var()

# ─────────────────────────────────────────────────────────────
# Structure Function from Energy Spectrum
# ─────────────────────────────────────────────────────────────
def build_faraday_depth_map(ne: np.ndarray, bz: np.ndarray, dz: float, C_RM: float) -> np.ndarray:
    """Build Faraday depth map Φ(x,y) = C_RM ∑_z n_e B_z dz."""
    return C_RM * (ne * bz).sum(axis=2) * dz

def energy_spectrum_to_p2d_model(ne: np.ndarray, bz: np.ndarray):
    """
    Convert 3D energy spectrum to 2D power spectrum model.
    
    Uses isotropy assumption: P3D(k) = E1D(k)/(4π k²)
    """
    q = ne * bz  # Faraday depth integrand
    nz, ny, nx = q.shape
    
    # 3D FFT and power spectrum
    Qk = fftn(q)
    P3 = (Qk * np.conj(Qk)).real
    
    # Shell-averaged 3D power spectrum
    k_shell, P3_shell = shell_average_3d(P3, dx=1.0, dy=1.0, dz=1.0, nbins=100)
    
    # Energy spectrum E1D(k) = 4π k² P3D(k)
    E1D = 4.0 * np.pi * k_shell**2 * P3_shell
    
    # Build 2D model on map grid (kz=0 slice)
    kx = 2.0*np.pi*fftfreq(nx, d=1.0)
    ky = 2.0*np.pi*fftfreq(ny, d=1.0)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K_perp = np.sqrt(KX**2 + KY**2)
    
    # Interpolate E1D onto 2D grid and convert to P2D
    log_k = np.log(np.maximum(k_shell, 1e-12))
    log_E = np.log(np.maximum(E1D, 1e-300))
    
    def P3D_of_k(k_abs):
        k_safe = np.maximum(k_abs, k_shell[0])
        E_interp = np.exp(np.interp(np.log(k_safe), log_k, log_E, left=log_E[0], right=log_E[-1]))
        return E_interp / (4.0 * np.pi * np.maximum(k_safe**2, 1e-24))
    
    P2_model = P3D_of_k(K_perp)
    P2_model[0,0] = 0.0  # Remove DC component
    
    return P2_model, (k_shell, E1D)

def shell_average_3d(P3: np.ndarray, dx=1.0, dy=1.0, dz=1.0, nbins=80):
    """Compute isotropic shell average of 3D power spectrum."""
    nz, ny, nx = P3.shape
    kx = 2.0*np.pi*fftfreq(nx, d=dx)
    ky = 2.0*np.pi*fftfreq(ny, d=dy)
    kz = 2.0*np.pi*fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    k_min = np.max([np.min(K[K > 0]), 1e-8])
    k_max = np.max(K)
    bins = np.logspace(np.log10(k_min), np.log10(k_max), nbins + 1)
    
    idx = np.digitize(K.ravel(), bins) - 1
    y = P3.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(y)
    sums = np.bincount(idx[good], weights=y[good], minlength=nbins)
    counts = np.bincount(idx[good], minlength=nbins)
    
    prof = np.full(nbins, np.nan)
    nz_mask = counts > 0
    prof[nz_mask] = sums[nz_mask] / counts[nz_mask]
    k_centers = 0.5 * (bins[1:] + bins[:-1])
    
    return k_centers[nz_mask], prof[nz_mask]

def radial_average_map(Map2D: np.ndarray, nbins: int, r_min_pix: float, r_max_frac: float):
    """Compute radial average of 2D map."""
    ny, nx = Map2D.shape
    yy = (np.arange(ny) - ny//2)[:, None]
    xx = (np.arange(nx) - nx//2)[None, :]
    R_pix = np.hypot(yy, xx)
    r_max = R_pix.max() * float(r_max_frac)
    
    eps = 1e-8
    bins = np.logspace(np.log10(max(r_min_pix, eps)), np.log10(r_max), nbins + 1)
    idx = np.digitize(R_pix.ravel(), bins) - 1
    
    vals = Map2D.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(vals)
    sums = np.bincount(idx[good], weights=vals[good], minlength=nbins)
    counts = np.bincount(idx[good], minlength=nbins)
    
    prof = np.full(nbins, np.nan)
    nz_mask = counts > 0
    prof[nz_mask] = sums[nz_mask] / counts[nz_mask]
    
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    sel = nz_mask & (r_centers >= r_min_pix)
    return r_centers[sel], prof[sel]

def dphi_from_energy_spectrum(ne: np.ndarray, bz: np.ndarray, dz: float, C_RM: float, 
                              lam: float, nbins_R: int, R_min_pix: float, R_max_frac: float):
    """
    Compute polarization angle structure function D_φ(R) from energy spectrum.
    
    Steps:
    1. Build Faraday depth map Φ(x,y)
    2. Extract 3D energy spectrum E(k) from q = n_e * B_z
    3. Convert to 2D power spectrum P2D via isotropy
    4. Compute correlation function C(R) via inverse FFT
    5. Calculate D_Φ(R) = 2(σ_Φ² - C(R))
    6. Convert to angle structure function D_φ(R)
    """
    # Build Faraday depth map
    Phi = build_faraday_depth_map(ne, bz, dz, C_RM)
    sigma_phi2 = Phi.var(ddof=0)
    
    # Build P2D model from energy spectrum
    P2_model, (k_shell, E1D) = energy_spectrum_to_p2d_model(ne, bz)
    
    # Compute correlation function from power spectrum
    ny, nx = P2_model.shape
    C_map_unnorm = ifft2(P2_model).real
    C0_model = C_map_unnorm[0,0] / (nx * ny)
    
    # Normalize to match actual Φ variance
    A = (sigma_phi2 / C0_model) if (np.isfinite(C0_model) and C0_model != 0.0) else 1.0
    C_map = np.fft.fftshift(A * C_map_unnorm / (nx * ny))
    
    # Radial average
    R_pix, C_R = radial_average_map(C_map, nbins_R, R_min_pix, R_max_frac)
    
    # Structure functions
    D_Phi = np.maximum(2.0 * (sigma_phi2 - C_R), 0.0)  # Faraday depth SF
    D_phi = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * D_Phi))  # Angle SF
    
    return R_pix, D_phi, D_Phi, sigma_phi2, (k_shell, E1D)

# ─────────────────────────────────────────────────────────────
# Slope Analysis
# ─────────────────────────────────────────────────────────────
def fit_slope_and_plot(ax, k, E, k1, k2, label_prefix="", color="red", linestyle="--"):
    """Fit and plot power law slope in given k range."""
    mask = (k > 0) & (E > 0) & np.isfinite(k) & np.isfinite(E) & (k >= k1) & (k <= k2)
    if not np.any(mask):
        return None
        
    kk, ee = k[mask], E[mask]
    x = np.log(kk)
    y = np.log(ee)
    x0 = x.mean()
    k0 = np.exp(x0)
    
    # Fit slope
    dx = x - x0
    slope = np.sum((y - y.mean()) * dx) / np.sum(dx * dx)
    c = np.mean(y - slope * dx)
    A = np.exp(c)
    
    # Plot fit line
    ax.loglog(kk, A * (kk/k0)**slope, color=color, linestyle=linestyle, 
              label=f"{label_prefix} slope: {slope:.2f}")
    
    return slope

def plot_reference_slopes(ax, k, E, k_ref=None, E_ref=None):
    """Plot reference slopes optimized for best fit to the data."""
    
    # Filter valid data points
    valid = np.isfinite(k) & np.isfinite(E) & (k > 0) & (E > 0)
    k_valid = k[valid]
    E_valid = E[valid]
    
    if len(k_valid) < 10:
        return
    
    # Define segments automatically based on data characteristics
    k_mid = np.sqrt(k_valid[0] * k_valid[-1])  # Geometric mean as transition
    
    # Left segment (low k): find best k_ref for -5/3 slope
    k_left = k_valid[k_valid <= k_mid]
    E_left = E_valid[k_valid <= k_mid]
    
    if len(k_left) > 3:
        # Find optimal k_ref by minimizing fitting error for -5/3 slope
        best_error = np.inf
        best_k_ref_left = k_left[len(k_left)//2]
        best_E_ref_left = np.interp(best_k_ref_left, k_valid, E_valid)
        
        # Try different reference points in the left segment
        for i in range(max(1, len(k_left)//4), min(len(k_left), 3*len(k_left)//4)):
            k_ref_test = k_left[i]
            E_ref_test = E_left[i]
            
            # Compute theoretical line with -5/3 slope
            E_theory = E_ref_test * (k_left / k_ref_test)**(-5/3)
            
            # Calculate fitting error (in log space)
            log_error = np.mean((np.log10(E_left) - np.log10(E_theory))**2)
            
            if log_error < best_error:
                best_error = log_error
                best_k_ref_left = k_ref_test
                best_E_ref_left = E_ref_test
        
        # Plot -5/3 reference line
        E_53 = best_E_ref_left * (k_left / best_k_ref_left)**(5/3)
        ax.loglog(k_left, E_53, 'k:', alpha=0.7, label=r"$k^{5/3}$")
    
    # Right segment (high k): find best k_ref for -2/3 slope (or another slope)
    k_right = k_valid[k_valid >= k_mid]
    E_right = E_valid[k_valid >= k_mid]
    
    if len(k_right) > 3:
        # Find optimal k_ref for the right segment
        best_error = np.inf
        best_k_ref_right = k_right[len(k_right)//2]
        best_E_ref_right = np.interp(best_k_ref_right, k_valid, E_valid)
        
        # Try different reference points and slopes
        slopes_to_try = [-2/3, -1, -3/2, -2]  # Different possible slopes
        best_slope = -2/3
        
        for slope in slopes_to_try:
            for i in range(max(1, len(k_right)//4), min(len(k_right), 3*len(k_right)//4)):
                k_ref_test = k_right[i]
                E_ref_test = E_right[i]
                
                # Compute theoretical line
                E_theory = E_ref_test * (k_right / k_ref_test)**(slope)
                
                # Calculate fitting error (in log space)
                log_error = np.mean((np.log10(E_right) - np.log10(E_theory))**2)
                
                if log_error < best_error:
                    best_error = log_error
                    best_k_ref_right = k_ref_test
                    best_E_ref_right = E_ref_test
                    best_slope = slope
        
        # Plot right segment reference line
        E_right_ref = best_E_ref_right * (k_right / best_k_ref_right)**(best_slope)
        slope_str = f"{best_slope:.2f}" if abs(best_slope - (-2/3)) > 0.01 else "-2/3"
        ax.loglog(k_right, E_right_ref, 'gray', linestyle=':', alpha=0.7, 
                 label=fr"$k^{{{slope_str}}}$")
        
        print(f"Optimal reference points: k_left={best_k_ref_left:.3f}, k_right={best_k_ref_right:.3f}")
        print(f"Best fit slopes: left=-5/3, right={best_slope:.3f}")

# ─────────────────────────────────────────────────────────────
# Main Analysis Pipeline
# ─────────────────────────────────────────────────────────────
def analyze_and_plot_energy_spectra(C=C):
    """Analyze and plot energy spectra for both datasets."""
    os.makedirs(C.output_dir, exist_ok=True)
    
    # Load data and compute energy spectra
    print("Loading Athena data...")
    try:
        x_athena, spc_athena = load_field(C.h5_athena, C.field_dataset)
        k_athena, E_athena, var_athena = compute_energy_spectrum(x_athena, spc_athena, C.nbins_spectrum)
        print(f"Athena: shape={x_athena.shape}, variance={var_athena:.3e}")
    except Exception as e:
        print(f"Warning: Could not load Athena data: {e}")
        k_athena = E_athena = var_athena = None
    
    print("Loading synthetic data...")
    try:
        x_synth, spc_synth = load_field(C.h5_synthetic, C.field_dataset)
        k_synth, E_synth, var_synth = compute_energy_spectrum(x_synth, spc_synth, C.nbins_spectrum)
        print(f"Synthetic: shape={x_synth.shape}, variance={var_synth:.3e}")
    except Exception as e:
        print(f"Warning: Could not load synthetic data: {e}")
        k_synth = E_synth = var_synth = None
    
    # Plot energy spectra comparison
    plt.figure(figsize=(8, 6))
    
    if k_athena is not None:
        plt.loglog(k_athena, E_athena, label=f"Athena (var={var_athena:.3e})", 
                   color="dodgerblue", linewidth=2)
    
    if k_synth is not None:
        plt.loglog(k_synth, E_synth, label=f"Synthetic (var={var_synth:.3e})", 
                   color="salmon", linewidth=2)
    
    # Add reference slopes and fit analysis
    if k_synth is not None:
        # Automatically determine optimal fitting ranges for synthetic data
        k_valid = k_synth[np.isfinite(k_synth) & np.isfinite(E_synth) & (k_synth > 0) & (E_synth > 0)]
        E_valid = E_synth[np.isfinite(k_synth) & np.isfinite(E_synth) & (k_synth > 0) & (E_synth > 0)]
        
        if len(k_valid) > 10:
            # Use geometric mean as break point
            k1_s = k_valid[0]
            k2_s = k_valid[-1] 
            ks = np.sqrt(k1_s * k2_s)
            
            # Adjust ranges to get good fitting segments
            k1_s = max(k1_s, k_valid[1])  # Skip first point which might be noisy
            k2_s = min(k2_s, k_valid[-2])  # Skip last point which might be noisy
            
            # Fit slopes for synthetic data
            fit_slope_and_plot(plt, k_synth, E_synth, k1_s, ks, "Synth low", "crimson", "--")
            fit_slope_and_plot(plt, k_synth, E_synth, ks, k2_s, "Synth high", "darkred", "--")
            plt.axvline(ks, color="gray", linestyle=":", alpha=0.8, label=f"k_break={ks:.3f}")
            
            print(f"Synthetic data fitting ranges: k_low=[{k1_s:.3f}, {ks:.3f}], k_high=[{ks:.3f}, {k2_s:.3f}]")
        
        # Reference slopes with optimized k_ref
        plot_reference_slopes(plt, k_synth, E_synth)
    
    if k_athena is not None:
        # Automatically determine optimal fitting ranges for Athena data
        k_valid_a = k_athena[np.isfinite(k_athena) & np.isfinite(E_athena) & (k_athena > 0) & (E_athena > 0)]
        E_valid_a = E_athena[np.isfinite(k_athena) & np.isfinite(E_athena) & (k_athena > 0) & (E_athena > 0)]
        
        if len(k_valid_a) > 10:
            # Use geometric mean as break point
            k1_a = k_valid_a[0]
            k2_a = k_valid_a[-1]
            ka = np.sqrt(k1_a * k2_a)
            
            # Adjust ranges to get good fitting segments
            k1_a = max(k1_a, k_valid_a[1])  # Skip first point which might be noisy
            k2_a = min(k2_a, k_valid_a[-2])  # Skip last point which might be noisy
            
            # For Athena data, we might want to exclude very low-k regime
            k1_a = max(k1_a, 20/256) if len(k_valid_a) > 20 else k1_a
            
            # Fit slopes for Athena data
            fit_slope_and_plot(plt, k_athena, E_athena, k1_a, ka, "Athena low", "royalblue", "--")
            fit_slope_and_plot(plt, k_athena, E_athena, ka, k2_a, "Athena high", "navy", "--")
            plt.axvline(ka, color="purple", linestyle=":", alpha=0.8, label=f"k_break={ka:.3f}")
            
            print(f"Athena data fitting ranges: k_low=[{k1_a:.3f}, {ka:.3f}], k_high=[{ka:.3f}, {k2_a:.3f}]")
        
        # Reference slopes with optimized k_ref
        plot_reference_slopes(plt, k_athena, E_athena)
    
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.title("Energy Spectrum Comparison")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2)
    plt.xlim(20/256, None)
    plt.tight_layout()
    
    plt.savefig(os.path.join(C.output_dir, "energy_spectrum_comparison.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.output_dir, "energy_spectrum_comparison.pdf"))
    #plt.show()
    plt.close()

def analyze_structure_functions(C=C):
    """Compute and plot polarization angle structure functions."""
    os.makedirs(C.output_dir, exist_ok=True)
    
    # Try to load data with density for structure function calculation
    datasets_to_try = [
        (C.h5_athena, "Athena"),
        (C.h5_synthetic, "Synthetic")
    ]
    
    for filepath, dataset_name in datasets_to_try:
        try:
            print(f"\nAnalyzing {dataset_name} structure functions...")
            ne, bz, dx, dz = load_density_and_field(filepath)
            print(f"  Loaded: ne.shape={ne.shape}, bz.shape={bz.shape}")
            
            # Compute structure functions for different wavelengths
            plt.figure(figsize=(8, 6))
            
            for i, lam in enumerate(C.wavelengths_m):
                R, D_phi, D_Phi, sigma_phi2, (k_shell, E1D) = dphi_from_energy_spectrum(
                    ne, bz, dz, C.C_RM, lam, C.nbins_R, C.R_min_pix, C.R_max_frac
                )
                
                print(f"  λ={lam:.2f}m: max D_Φ={np.nanmax(D_Phi):.3e}, max D_φ={np.nanmax(D_phi):.3e}")
                
                # Plot angle structure function
                plt.loglog(R, D_phi, linewidth=2, label=f"λ={lam:.2f}m")
            
            # Add reference slope
            if len(C.wavelengths_m) > 0:
                R_ref = R[R > 5]  # Avoid small R noise
                if len(R_ref) > 0:
                    D_ref = D_phi[R > 5][0] if np.any(R > 5) else D_phi[len(D_phi)//4]
                    R_ref_val = R_ref[0] if len(R_ref) > 0 else R[len(R)//4]
                    slope_53_line = D_ref * (R / R_ref_val)**(5/3)
                    plt.loglog(R, slope_53_line, 'k--', alpha=0.7, label=r"$R^{5/3}$")
            
            plt.xlabel(r"$R$ [pixels]")
            plt.ylabel(r"$D_\varphi(R)$")
            plt.title(f"Polarization Angle Structure Function ({dataset_name})")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            
            # Save plot
            safe_name = dataset_name.lower().replace(" ", "_")
            plt.savefig(os.path.join(C.output_dir, f"angle_structure_function_{safe_name}.png"), dpi=C.dpi)
            plt.savefig(os.path.join(C.output_dir, f"angle_structure_function_{safe_name}.pdf"))
            #plt.show()
            plt.close()
            
            # Plot corresponding energy spectrum
            plt.figure(figsize=(7, 5))
            plt.loglog(k_shell, E1D, linewidth=2, color="darkblue", label=f"{dataset_name} E(k)")
            plot_reference_slopes(plt, k_shell, E1D)
            plt.xlabel(r"$k$")
            plt.ylabel(r"$E(k)$")
            plt.title(f"Energy Spectrum from q=n_e B_z ({dataset_name})")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(C.output_dir, f"energy_spectrum_{safe_name}.png"), dpi=C.dpi)
            plt.savefig(os.path.join(C.output_dir, f"energy_spectrum_{safe_name}.pdf"))
            #plt.show()
            plt.close()
            
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            continue

def main(C=C):
    """Main analysis pipeline."""
    print("=== Polarization Angle Structure Function Analysis ===")
    print(f"Output directory: {C.output_dir}")
    
    # Analyze energy spectra
    print("\n1. Energy Spectrum Analysis")
    analyze_and_plot_energy_spectra(C)
    
    # Analyze structure functions
    print("\n2. Structure Function Analysis")
    analyze_structure_functions(C)
    
    print(f"\nAnalysis complete. Results saved to: {os.path.abspath(C.output_dir)}")

if __name__ == "__main__":
    main()
