#!/usr/bin/env python3
"""
Standalone script to run plot_spatial_psa_comparison function only.
This script generates synthetic data and creates the spatial PSA comparison plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from scipy import ndimage

# Add the compare_measures directory to the path to import the function
sys.path.append(os.path.join(os.path.dirname(__file__), 'compare_measures'))

# Import the required function and dependencies
from pfa_and_derivative_lp_16_like import (
    plot_spatial_psa_comparison,
    load_fields,
    polarized_emissivity_simple,
    polarized_emissivity_lp16,
    faraday_density,
    P_map_mixed,
    dP_map_mixed,
    psa_of_map,
    fit_log_slope_window,
    linear_tracer,
    PFAConfig,
    FieldKeys
)

def load_real_data():
    """
    Load real data from the HDF5 file and compute the spatial PSA.
    """
    # Try different possible paths
    possible_paths = [
        r"faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
        r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
        os.path.join(os.path.dirname(__file__), "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
        os.path.join(os.path.dirname(__file__), "..", "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
    ]
    
    h5_path = None
    for path in possible_paths:
        if os.path.exists(path):
            h5_path = path
            break
    
    if h5_path is None:
        raise FileNotFoundError(f"Could not find HDF5 file. Tried: {possible_paths}")
    
    print(f"Loading MHD fields from: {h5_path}")
    
    # Load the data
    keys = FieldKeys()
    cfg = PFAConfig()
    
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    print(f"Loaded fields with shape: Bx={Bx.shape}, By={By.shape}, Bz={Bz.shape}, ne={ne.shape}")
    
    # Compute polarization emissivity
    Pi = polarized_emissivity_simple(Bx, By, gamma=cfg.gamma)
    
    # Choose B_parallel based on cfg.los_axis (corrected mapping)
    if cfg.los_axis == 0:
        Bpar = Bx
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz
    
    # Compute Faraday density
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)
    
    # Choose a specific lambda value for analysis
    lam = 1.000000
    
    print(f"Computing polarization maps for λ = {lam}...")
    
    # Compute P and dP/dλ² maps for mixed geometry
    P_map = P_map_mixed(Pi, phi, lam, cfg)
    dP_map = dP_map_mixed(Pi, phi, lam, cfg)
    
    print(f"Computed maps: P_map shape={P_map.shape}, dP_map shape={dP_map.shape}")
    
    print("Computing spatial power spectra...")
    
    # Compute PSA for both maps using consistent settings (match diagnostics)
    k_P, Ek_P = psa_of_map(P_map, ring_bins=48, pad=1, apodize=True, 
                           k_min=3.0, min_counts_per_ring=10, return_energy_like=False)
    k_dP, Ek_dP = psa_of_map(dP_map, ring_bins=48, pad=1, apodize=True, 
                             k_min=3.0, min_counts_per_ring=10, return_energy_like=False)
    
    print(f"Generated PSA data: {len(k_P)} points for P, {len(k_dP)} points for dP/dλ²")
    
    # Sanity check: verify slopes match diagnostics expectations
    print(f"\n[Sanity] Using PSD + Hann, k∈[4,25] like diagnostics")
    mP, aP, eP, _ = fit_log_slope_with_bounds(k_P, Ek_P, kmin=4, kmax=25)
    mD, aD, eD, _ = fit_log_slope_with_bounds(k_dP, Ek_dP, kmin=4, kmax=25)
    print(f"  P  slope = {mP:.2f} ± {eP:.2f}")
    print(f"  dP slope = {mD:.2f} ± {eD:.2f}")
    print(f"  Expected: ~-3.5 for PSD (should match diagnostics)")
    
    return k_P, Ek_P, k_dP, Ek_dP

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


# === Separated screen transitional prediction ===
def radial_structure_function(phi2d, nbins=40, rmin=1, rmax=None):
    """Estimate D_phi(r) ≈ ⟨[Φ(x+r)-Φ(x)]^2⟩, radially averaged."""
    ny, nx = phi2d.shape
    if rmax is None:
        rmax = min(nx, ny) // 4
    # FFT-based autocovariance
    f = np.fft.rfftn(phi2d - phi2d.mean())
    S = (f * np.conj(f)).real
    C = np.fft.irfftn(S, s=phi2d.shape) / (nx * ny)  # autocorrelation
    var = C[0, 0]
    D = 2 * (var - C)  # structure function on grid
    # radial binning
    yy, xx = np.indices(phi2d.shape)
    rr = np.hypot(xx - nx//2, yy - ny//2)
    r = np.linspace(rmin, rmax, nbins)
    Dr = []
    for i in range(nbins-1):
        m = (rr >= r[i]) & (rr < r[i+1])
        if m.any():
            Dr.append(D[m].mean())
        else:
            Dr.append(np.nan)
    r_mid = 0.5 * (r[:-1] + r[1:])
    Dr = np.array(Dr)
    sel = np.isfinite(Dr)
    return r_mid[sel], Dr[sel]


def fit_small_r_powerlaw(r, D, rmax_fit=None):
    """Fit D(r) ≈ A r^m on small r."""
    if rmax_fit is None:
        rmax_fit = np.percentile(r, 30)  # use inner 30% radii
    sel = (r > 0) & (r <= rmax_fit) & np.isfinite(D) & (D > 0)
    rfit, Dfit = r[sel], D[sel]
    if len(rfit) < 5:
        return None, None
    m, a = np.polyfit(np.log(rfit), np.log(Dfit), 1)
    A = np.exp(a)
    return A, m  # D ≈ A r^m


def predict_psa_slope_curve(chi, m0, k1=4.0, k2=25.0, A=None, m=None, q=1.0):
    """m_fit(chi) = m0 / (1 + (k_phi(chi)/k_*)^q),  k_phi = (A)^{1/m} chi^{2/m}."""
    if A is None or m is None:
        return None
    kstar = np.sqrt(k1 * k2)
    kphi = (A ** (1.0 / m)) * (np.array(chi) ** (2.0 / m))
    s = 1.0 / (1.0 + (kphi / kstar) ** q)
    return m0 * s


def predict_and_overlay_transitional_curve(Phi_scr, chi_values, slopes_P, ax, 
                                           m0=None, k1=4, k2=25, q=1.0, label="prediction"):
    """Compute A,m from Φ_screen, build m_fit(χ), and overlay."""
    # 1) D_phi(r) fit
    r, D = radial_structure_function(Phi_scr, nbins=48)
    A, m = fit_small_r_powerlaw(r, D)
    if A is None or m is None:
        print(f"Warning: Could not fit structure function power law")
        return None, None, None
    # 2) emitter-only slope if not provided
    if m0 is None:
        m0 = slopes_P[0] if len(slopes_P) > 0 else -3.5  # use first point as proxy
    # 3) prediction
    mfit = predict_psa_slope_curve(np.array(chi_values), m0, k1=k1, k2=k2, A=A, m=m, q=q)
    if mfit is not None:
        ax.plot(chi_values, mfit, 'k-', lw=1.5, label=label + f" (m≈{m:.2f})")
    return A, m, mfit


#
# === NEW: external-screen (separated) geometry helpers ===
#
def P_map_separated(Pi, phi, lam, cfg,
                    emit_bounds=None,
                    screen_bounds=None):
    """
    External screen: P(X,λ) = [∫_emit Pi dz] * exp{ 2i λ^2 Φ_screen(X) }.
    |P| does NOT depend on λ; all λ-dependence is in the phase.
    """
    Pi_los  = np.moveaxis(Pi,  cfg.los_axis, 0)
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz, Ny, Nx = Pi_los.shape

    # default: take front 10% as screen, rest as emitter
    if screen_bounds is None:
        scr_N = max(1, int(0.10 * Nz))
        screen_bounds = (0, scr_N)
    if emit_bounds is None:
        emit_bounds = (screen_bounds[1], Nz)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    # emission (no internal rotation)
    P_emit = Pi_los[z0e:z1e].sum(axis=0)
    # screen RM
    Phi_screen = phi_los[z0s:z1s].sum(axis=0)
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)

def dP_map_separated(Pi, phi, lam, cfg,
                     emit_bounds=None,
                     screen_bounds=None):
    """
    dP/d(λ²) for external screen: 2i Φ_screen * P(X,λ).
    """
    Pi_los  = np.moveaxis(Pi,  cfg.los_axis, 0)
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz, Ny, Nx = Pi_los.shape
    if screen_bounds is None:
        scr_N = max(1, int(0.10 * Nz))
        screen_bounds = (0, scr_N)
    if emit_bounds is None:
        emit_bounds = (screen_bounds[1], Nz)
    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds
    P = P_map_separated(Pi, phi, lam, cfg, emit_bounds, screen_bounds)
    Phi_screen = phi_los[z0s:z1s].sum(axis=0)
    return 2j * Phi_screen * P

def P_map_thin_slice(Bx, By, z0=0, dz=8):
    """Create a linear tracer thin slice for comparison with column integration."""
    # Returns a linear tracer slice: Q+iU ≈ Bx + i By integrated over dz
    slab = slice(z0, z0+dz)
    return (Bx[:,:,slab].sum(axis=2) + 1j*By[:,:,slab].sum(axis=2))


def lambda_zero_diagnostics(Pi, phi, Bx, By, cfg, case="Mixed"):
    """Enhanced λ=0 diagnostics with comprehensive comparisons."""
    print(f"\n{'='*60}")
    print(f"λ=0 DIAGNOSTICS ({case} case)")
    print(f"{'='*60}")
    
    # Create λ=0 polarization maps
    # (for external screen, λ=0 just returns the emitter sum)
    P0 = P_map_separated(Pi, phi, lam=0.0, cfg=cfg)
    Pi_linear = linear_tracer(Bx, By)
    P0_linear = P_map_separated(Pi_linear, phi, lam=0.0, cfg=cfg)
    Pi_lp16 = polarized_emissivity_lp16(Bx, By, gamma=2.0)
    P0_lp16 = P_map_separated(Pi_lp16, phi, lam=0.0, cfg=cfg)
    
    # Add thin slice comparison
    P0_linear_thin = P_map_thin_slice(Bx, By, z0=Bx.shape[2]//2, dz=8)
    
    # Test different conventions and windowing options
    tests = [
        ("PSD, no window", False, False, 3),
        ("PSD, Hann window", True, False, 6),
        ("Energy-like, Hann", True, True, 6),
    ]
    
    # Create plots for each emissivity type
    emissivity_cases = [
        ("Quadratic", P0, "Quadratic emissivity (P_i = (B_x + iB_y)^2)"),
        ("Linear", P0_linear, "Linear tracer (P_i = B_x + iB_y) - Column integration"),
        ("Linear-Thin", P0_linear_thin, "Linear tracer thin slice (dz=8) - SANITY CHECK"),
        ("LP16", P0_lp16, "LP16 emissivity (γ=2, with amplitude factor)")
    ]
    
    slopes_data = {}
    
    # Create single 4x3 figure (4 emissivity types, 3 test cases)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('λ=0 PSA Analysis (Separated Screen): All Emissivity Types and Test Cases', fontsize=16)
    
    for emissivity_idx, (emissivity_name, P_map, description) in enumerate(emissivity_cases):
        print(f"\n{description}:")
        slopes_data[emissivity_name] = {}
        
        for test_idx, (label, apod, energy_like, kmin) in enumerate(tests):
            k, S = psa_of_map(P_map, apodize=apod, return_energy_like=energy_like, k_min=kmin)
            # Use bounded fitting to avoid edge effects
            m, a, err, (k_min_fit, k_max_fit) = fit_log_slope_with_bounds(k, S, kmin=4, kmax=25)
            
            slopes_data[emissivity_name][label] = {'slope': m, 'error': err, 'k': k, 'S': S}
            
            print(f"  {label:20s}: slope {m:6.2f} ± {err:.2f}")
            
            # Plot the spectrum
            ax = axes[emissivity_idx, test_idx]
            ax.loglog(k, S, 'o-', markersize=2, alpha=0.7, label=f'Data')
            
            # Plot fit line if valid
            if np.isfinite(m):
                k_fit = np.array([k_min_fit, k_max_fit])
                S_fit = 10**(a + m * np.log10(k_fit))
                ax.loglog(k_fit, S_fit, '--', linewidth=2, color='red', 
                         label=f'Fit: slope = {m:.2f} ± {err:.2f}')
                # Mark the fitting range
                ax.axvspan(k_min_fit, k_max_fit, alpha=0.1, color='red')
            
            ax.set_xlabel('$k$ (wavenumber)')
            if energy_like:
                ax.set_ylabel('$E_{2D}(k)$ (energy-like)')
            else:
                ax.set_ylabel('$P_{2D}(k)$ (PSD)')
            
            # Set titles
            if emissivity_idx == 0:  # Top row
                ax.set_title(f'{label}', fontsize=12)
            if test_idx == 0:  # Left column
                ax.text(-0.15, 0.5, emissivity_name, transform=ax.transAxes, 
                       rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            
            ax.grid(True, which='both', alpha=0.2)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "compare_measures/lp2016_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lambda_zero_diagnostics_all_cases.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive plot: {output_path}")
    plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Emissivity':<12} {'Test Case':<20} {'Slope':<10} {'Error':<8}")
    print(f"{'-'*80}")
    
    for emissivity_name in ["Quadratic", "Linear", "Linear-Thin", "LP16"]:
        for label in ["PSD, no window", "PSD, Hann window", "Energy-like, Hann"]:
            data = slopes_data[emissivity_name][label]
            print(f"{emissivity_name:<12} {label:<20} {data['slope']:<10.2f} {data['error']:<8.2f}")
    
    print(f"\nEXPECTED SLOPES (theoretical):")
    print(f"  Linear thin-slice, 2D-PSD:  ~-2.67 (projection of 3D Kolmogorov β_B=11/3)")
    print(f"  Linear thin-slice, 2D-energy: ~-1.67 (PSD slope + 1)")
    print(f"  Linear column, 2D-PSD:     ~-3.0  (steeper due to cancellations)")
    print(f"  Quadratic, 2D-PSD:        ~-3.33 (baseline: β_f=13/3, projection)")
    print(f"  Quadratic, 2D-energy:     ~-2.33 (PSD slope + 1)")
    print(f"  Observed ~-4.5:           steeper due to spin-2 + windowing + finite map")
    
    print(f"\nINTERPRETATION:")
    print(f"  • ~-4.5 slopes are EXPECTED for quadratic spin-2 projected fields")
    print(f"  • LP16/Zhang Kolmogorov predictions are for PFA variance vs λ², NOT spatial PSA at λ=0")
    print(f"  • Linear thin-slice should show slopes closest to projection expectation (~-2.7)")
    print(f"  • Linear column integration steepens slopes due to cancellations")
    print(f"  • Windowing bias can steepen slopes by ~0.3-1.0")
    print(f"  • Energy-like spectrum slopes are PSD slopes + 1")
    print(f"  • Bounded fitting (k∈[4,25]) avoids edge effects")

def main():
    """
    Main function to run the spatial PSA comparison plot with λ² sweep
    and create slope vs χ plot.
    """
    print("="*70)
    print("SPATIAL PSA COMPARISON ANALYSIS - λ² SWEEP (Separated Screen)")
    print("="*70)
    
    # Load data
    keys = FieldKeys()
    cfg = PFAConfig()
    
    # Try different possible paths
    possible_paths = [
        r"faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
        r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
        os.path.join(os.path.dirname(__file__), "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
        os.path.join(os.path.dirname(__file__), "..", "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
    ]
    
    h5_path = None
    for path in possible_paths:
        if os.path.exists(path):
            h5_path = path
            break
    
    if h5_path is None:
        raise FileNotFoundError(f"Could not find HDF5 file. Tried: {possible_paths}")
    
    print(f"Loading MHD fields from: {h5_path}")
    
    # Load the data
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    print(f"Loaded fields with shape: Bx={Bx.shape}, By={By.shape}, Bz={Bz.shape}, ne={ne.shape}")
    
    # 0) Consistent LOS and regime
    cfg.los_axis = 2
    gamma = cfg.gamma = 2.0
    Bpar = Bz if cfg.los_axis == 2 else (By if cfg.los_axis == 1 else Bx)
    Bpar = Bpar - Bpar.mean()  # random-dominated
    
    # 1) Build φ with chosen C, and choose separated bounds
    phi_unit = ne * Bpar
    Nz = phi_unit.shape[cfg.los_axis]
    scr_N = max(1, int(0.10 * Nz))          # 10% of depth as screen
    screen_bounds = (0, scr_N)               # front slab → screen
    emit_bounds   = (scr_N, Nz)              # rest → emitter
    
    # 2) Set Faraday constant to reasonable value
    cfg.faraday_const = 1.0  # Keep C=1 for simplicity (can be CLI’d)
    phi = cfg.faraday_const * phi_unit
    
    # 1b) σΦ from the SCREEN ONLY (for χ = 2 σΦ λ² in separated case)
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Phi_screen = phi_los[screen_bounds[0]:screen_bounds[1]].sum(axis=0)
    sigmaPhi_screen = float(Phi_screen.std())
    
    print(f"Using C={cfg.faraday_const:.3g}; σΦ(screen)={sigmaPhi_screen:.3g} (bounds={screen_bounds})")
    
    # 3) PSA settings (use the same everywhere)
    psa_kwargs = dict(ring_bins=48, pad=1, apodize=True, k_min=6.0, min_counts_per_ring=10)
    fit_bounds = dict(kmin=4, kmax=25)
    
    # 4) Define λ² grid for sweep (uniform in λ², covering χ ∈ [0.05, 20])
    lam2_min = 0.05 / (2.0 * sigmaPhi_screen)
    lam2_max = 20.0 / (2.0 * sigmaPhi_screen)
    n_points = 100
    lam2_values = np.linspace(lam2_min, lam2_max, n_points)
    
    print(f"\nλ² sweep: {lam2_min:.3f} to {lam2_max:.3f} ({n_points} points)")
    print(f"Corresponding χ range: {2*sigmaPhi_screen*lam2_min:.2f} to {2*sigmaPhi_screen*lam2_max:.2f}")
    
    # 5) Initialize arrays to store results
    chi_values = []
    slopes_P = []
    slopes_dP = []
    errors_P = []
    errors_dP = []
    
    # 6) Sweep over λ² values
    print(f"\nComputing PSA slopes for each λ²...")
    for i, lam2 in enumerate(lam2_values):
        lam = np.sqrt(lam2)
        chi = 2.0 * sigmaPhi_screen * cfg.faraday_const * lam2
        
        print(f"  λ²={lam2:.3f}, χ={chi:.2f} ({i+1}/{n_points})")
        
        # Build maps for this λ (SEPARATED SCREEN)
        Pi_now = polarized_emissivity_simple(Bx, By, gamma=gamma)
        P_map  = P_map_separated(Pi_now, phi, lam, cfg,
                                 emit_bounds=emit_bounds, screen_bounds=screen_bounds)
        dP_map = dP_map_separated(Pi_now, phi, lam, cfg,
                                  emit_bounds=emit_bounds, screen_bounds=screen_bounds)
        
        # Compute PSA
        k_P, Ek_P = psa_of_map(P_map, return_energy_like=False, **psa_kwargs)
        k_dP, Ek_dP = psa_of_map(dP_map, return_energy_like=False, **psa_kwargs)
        
        # Fit slopes
        mP, aP, eP, _ = fit_log_slope_with_bounds(k_P, Ek_P, **fit_bounds)
        mD, aD, eD, _ = fit_log_slope_with_bounds(k_dP, Ek_dP, **fit_bounds)
        
        # Store results
        chi_values.append(chi)
        slopes_P.append(mP)
        slopes_dP.append(mD)
        errors_P.append(eP)
        errors_dP.append(eD)
        
        print(f"    P slope: {mP:.2f}±{eP:.2f}, dP/dλ² slope: {mD:.2f}±{eD:.2f}")
    
    # 7) Compute emitter-only baseline slope for separated geometry
    print(f"\nComputing emitter-only baseline slope (separated prediction)...")
    Pi_los = np.moveaxis(polarized_emissivity_simple(Bx, By, gamma=gamma), cfg.los_axis, 0)
    P_emit = Pi_los[emit_bounds[0]:emit_bounds[1]].sum(axis=0)
    # Detrend emitter map so PSA isn't DC dominated
    P_emit = P_emit - P_emit.mean()
    
    k0, S0 = psa_of_map(P_emit, return_energy_like=False, **psa_kwargs)
    m0, a0, e0, _ = fit_log_slope_with_bounds(k0, S0, **fit_bounds)
    
    print(f"[Separated prediction] emitter PSA slope m0 = {m0:.2f} ± {e0:.2f} (fit k=[{fit_bounds['kmin']},{fit_bounds['kmax']}])")
    print(f"[Separated prediction] small-χ asymptote: m ≈ {m0:.2f}")
    print(f"[Separated prediction] large-χ asymptote: m → 0")
    
    # Compute transitional prediction curve from screen structure function
    print(f"\nComputing transitional prediction from screen structure function...")
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Phi_scr = phi_los[screen_bounds[0]:screen_bounds[1]].sum(axis=0)
    
    # Fit structure function to get A, m
    r, D = radial_structure_function(Phi_scr, nbins=48)
    A, m = fit_small_r_powerlaw(r, D)
    if A is not None and m is not None:
        print(f"[Separated prediction] D_Φ(r) ≈ {A:.3e} r^{m:.2f}")
    else:
        print(f"[Separated prediction] Warning: Could not fit structure function")
        A, m = None, None

    # 7b) Save results to NPZ for later reproducibility
    output_dir = r"D:\Рабочая папка\GitHub\AstroTurbulence\compare_measures\lp2016_outputs"
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "psa_slopes_vs_chi_data.npz")
    np.savez(
        npz_path,
        lam2_values=np.array(lam2_values),
        chi_values=np.array(chi_values),
        slopes_P=np.array(slopes_P),
        slopes_dP=np.array(slopes_dP),
        errors_P=np.array(errors_P),
        errors_dP=np.array(errors_dP),
        psa_kwargs=psa_kwargs,
        fit_bounds=fit_bounds,
        sigmaPhi_screen=np.array(sigmaPhi_screen),
        faraday_const=np.array(cfg.faraday_const),
        los_axis=np.array(cfg.los_axis),
        gamma=np.array(gamma),
        geometry=np.array("separated"),
        screen_bounds=np.array(screen_bounds),
        emit_bounds=np.array(emit_bounds),
        emitter_baseline_slope=np.array(m0),
        emitter_baseline_error=np.array(e0),
        structure_function_A=np.array(A) if A is not None else np.array(np.nan),
        structure_function_m=np.array(m) if m is not None else np.array(np.nan),
    )
    print(f"\nSaved sweep data to: {npz_path}")

    # 8) Create slope vs χ plot
    print(f"\nCreating slope vs χ plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Set y-axis limits for better visualization
    ymin, ymax = -3.8, 0.5
    
    # Plot P slopes
    ax1.errorbar(chi_values, slopes_P, yerr=errors_P, fmt='o-', capsize=3, 
                 label='PSA of $P$', color='blue', alpha=0.7)
    # Add horizontal reference lines for separated geometry (two asymptotes)
    ax1.axhline(m0, ls='--', lw=1.0, color='k', alpha=0.6, 
                label='separated (emitter baseline, small-χ)')
    ax1.axhline(0.0, ls=':', lw=1.0, color='k', alpha=0.7, 
                label='large-χ limit (flat)')
    # Add transitional prediction curve
    if A is not None and m is not None:
        predict_and_overlay_transitional_curve(
            Phi_scr, chi_values, slopes_P, ax1, m0=m0, 
            k1=fit_bounds['kmin'], k2=fit_bounds['kmax'], q=1.0,
            label="separated-screen prediction"
        )
    ax1.set_xlabel('$\\chi = 2\\sigma_\\Phi \\lambda^2$')
    ax1.set_ylabel('PSA Slope of $P$')
    ax1.set_title('PSA Slope of $P$ vs $\\chi$ (Separated Screen)')
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot dP/dλ² slopes
    ax2.errorbar(chi_values, slopes_dP, yerr=errors_dP, fmt='s-', capsize=3, 
                 label='PSA of $dP/d\\lambda^2$', color='red', alpha=0.7)
    # Add horizontal reference lines (same asymptotes for separated geometry)
    ax2.axhline(m0, ls='--', lw=1.0, color='k', alpha=0.6, 
                label='separated (emitter baseline, small-χ)')
    ax2.axhline(0.0, ls=':', lw=1.0, color='k', alpha=0.7, 
                label='large-χ limit (flat)')
    # Add transitional prediction curve (same A, m as for P)
    if A is not None and m is not None:
        predict_and_overlay_transitional_curve(
            Phi_scr, chi_values, slopes_dP, ax2, m0=m0, 
            k1=fit_bounds['kmin'], k2=fit_bounds['kmax'], q=1.0,
            label="separated-screen prediction"
        )
    ax2.set_xlabel('$\\chi = 2\\sigma_\\Phi \\lambda^2$')
    ax2.set_ylabel('PSA Slope of $dP/d\\lambda^2$')
    ax2.set_title('PSA Slope of $dP/d\\lambda^2$ vs $\\chi$ (Separated Screen)')
    ax2.set_ylim(ymin, ymax)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the slope vs χ plot
    slope_plot_path = os.path.join(output_dir, "psa_slopes_vs_chi.png")
    
    print(f"Saving slope vs χ plot to: {slope_plot_path}")
    plt.savefig(slope_plot_path, dpi=300, bbox_inches='tight')
    
    # 8) Also create a combined plot
    plt.figure(figsize=(10, 6))
    ax_combined = plt.gca()
    ax_combined.errorbar(chi_values, slopes_P, yerr=errors_P, fmt='o-', capsize=3, 
                         label='PSA of $P$', color='blue', alpha=0.7)
    ax_combined.errorbar(chi_values, slopes_dP, yerr=errors_dP, fmt='s-', capsize=3, 
                         label='PSA of $dP/d\\lambda^2$', color='red', alpha=0.7)
    # Add horizontal reference lines for separated geometry (two asymptotes)
    ax_combined.axhline(m0, ls='--', lw=1.0, color='k', alpha=0.6, 
                       label='separated baseline (small-χ)')
    ax_combined.axhline(0.0, ls=':', lw=1.0, color='k', alpha=0.7, 
                       label='large-χ limit (flat)')
    # Add transitional prediction curve
    if A is not None and m is not None:
        predict_and_overlay_transitional_curve(
            Phi_scr, chi_values, slopes_P, ax_combined, m0=m0, 
            k1=fit_bounds['kmin'], k2=fit_bounds['kmax'], q=1.0,
            label="separated-screen prediction"
        )
    ax_combined.set_xlabel('$\\chi = 2\\sigma_\\Phi \\lambda^2$')
    ax_combined.set_ylabel('PSA Slope')
    ax_combined.set_title('PSA Slopes vs $\\chi$ (Separated Screen)')
    ax_combined.set_ylim(ymin, ymax)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend()
    
    combined_plot_path = os.path.join(output_dir, "psa_slopes_vs_chi_combined.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saving combined plot to: {combined_plot_path}")
    
    # 9) Print summary
    print(f"\n" + "="*70)
    print(f"SUMMARY - PSA SLOPES vs χ")
    print(f"="*70)
    print(f"χ range: {min(chi_values):.2f} to {max(chi_values):.2f}")
    print(f"P slopes range: {min(slopes_P):.2f} to {max(slopes_P):.2f}")
    print(f"dP/dλ² slopes range: {min(slopes_dP):.2f} to {max(slopes_dP):.2f}")
    print(f"Average slope difference: {np.mean(np.abs(np.array(slopes_dP) - np.array(slopes_P))):.2f}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
