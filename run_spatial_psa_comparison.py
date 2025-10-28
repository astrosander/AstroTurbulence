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
    
    # Choose B_parallel based on cfg.los_axis
    if cfg.los_axis == 0:
        Bpar = Bz
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bx
    
    # Compute Faraday density
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)
    
    # Choose a specific lambda value for analysis
    lam = 0.000000
    
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
    P0 = P_map_mixed(Pi, phi, lam=0.0, cfg=cfg)
    Pi_linear = linear_tracer(Bx, By)
    P0_linear = P_map_mixed(Pi_linear, phi, lam=0.0, cfg=cfg)
    Pi_lp16 = polarized_emissivity_lp16(Bx, By, gamma=2.0)
    P0_lp16 = P_map_mixed(Pi_lp16, phi, lam=0.0, cfg=cfg)
    
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
    fig.suptitle('λ=0 PSA Analysis: All Emissivity Types and Test Cases', fontsize=16)
    
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
    Main function to run the spatial PSA comparison plot.
    """
    print("="*70)
    print("SPATIAL PSA COMPARISON ANALYSIS")
    print("="*70)
    
    # Load data and run diagnostics
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
    
    # Compute polarization emissivity
    Pi = polarized_emissivity_simple(Bx, By, gamma=cfg.gamma)
    
    # Choose B_parallel based on cfg.los_axis
    if cfg.los_axis == 0:
        Bpar = Bz
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bx
    
    # Compute Faraday density
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)
    
    # Run λ=0 diagnostics
    lambda_zero_diagnostics(Pi, phi, Bx, By, cfg, case="Mixed")
    
    # Try to load real data first, fall back to synthetic if needed
    k_P, Ek_P, k_dP, Ek_dP = load_real_data()
    
    # Set the lambda value for the plot
    lam = 0.000000
    
    print(f"\nCreating spatial PSA comparison plot for λ = {lam}...")
    
    # Create the plot with unified settings (PSD)
    m_P, m_dP, err_P, err_dP = plot_spatial_psa_comparison(
        k_P=k_P, 
        Ek_P=Ek_P, 
        k_dP=k_dP, 
        Ek_dP=Ek_dP, 
        lam=lam, 
        case="Mixed",
        spectrum="psd",
        kfit_bounds=(4, 25)
    )
    
    # Also test energy-like spectrum for comparison
    print(f"\nTesting energy-like spectrum for consistency...")
    
    # Recompute with energy-like spectrum
    P_map = P_map_mixed(Pi, phi, lam, cfg)
    dP_map = dP_map_mixed(Pi, phi, lam, cfg)
    
    k_P_E, Ek_P_E = psa_of_map(P_map, ring_bins=48, pad=1, apodize=True, 
                               k_min=3.0, min_counts_per_ring=10, return_energy_like=True)
    k_dP_E, Ek_dP_E = psa_of_map(dP_map, ring_bins=48, pad=1, apodize=True, 
                                 k_min=3.0, min_counts_per_ring=10, return_energy_like=True)
    
    # Create energy-like plot
    m_P_E, m_dP_E, err_P_E, err_dP_E = plot_spatial_psa_comparison(
        k_P=k_P_E, 
        Ek_P=Ek_P_E, 
        k_dP=k_dP_E, 
        Ek_dP=Ek_dP_E, 
        lam=lam, 
        case="Mixed",
        spectrum="energy",
        kfit_bounds=(4, 25)
    )
    
    # Save the plot to the specified output path
    output_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\compare_measures\lp2016_outputs\derivative_slope_analysis_random.png"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nPlot saved successfully!")
    print(f"\nCONSISTENCY CHECK:")
    print(f"PSD spectrum:")
    print(f"  PSA of P slope: {m_P:.2f} ± {err_P:.2f}")
    print(f"  PSA of dP/dλ² slope: {m_dP:.2f} ± {err_dP:.2f}")
    print(f"  Difference: {abs(m_dP - m_P):.2f}")
    
    print(f"\nEnergy-like spectrum:")
    print(f"  PSA of P slope: {m_P_E:.2f} ± {err_P_E:.2f}")
    print(f"  PSA of dP/dλ² slope: {m_dP_E:.2f} ± {err_dP_E:.2f}")
    print(f"  Difference: {abs(m_dP_E - m_P_E):.2f}")
    
    print(f"\nExpected relationship: Energy-like slope = PSD slope + 1")
    print(f"  P: {m_P_E:.2f} vs {m_P + 1:.2f} (diff: {abs(m_P_E - (m_P + 1)):.2f})")
    print(f"  dP: {m_dP_E:.2f} vs {m_dP + 1:.2f} (diff: {abs(m_dP_E - (m_dP + 1)):.2f})")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
