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
    Main function to run the spatial PSA comparison plot with λ² sweep
    and create slope vs χ plot.
    """
    print("="*70)
    print("SPATIAL PSA COMPARISON ANALYSIS - λ² SWEEP")
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
    
    # 1) Compute sigmaPhi0 with C=1
    phi_unit = ne * Bpar
    Phi_tot  = np.moveaxis(phi_unit, cfg.los_axis, 0).sum(axis=0) / phi_unit.shape[cfg.los_axis]
    sigmaPhi0 = float(Phi_tot.std())
    
    # 2) Set Faraday constant to reasonable value
    cfg.faraday_const = 1.0  # Keep C=1 for simplicity
    phi = cfg.faraday_const * phi_unit
    
    print(f"Using C={cfg.faraday_const:.3g}; σΦ₀={sigmaPhi0:.3g}")
    
    # 3) PSA settings (use the same everywhere)
    psa_kwargs = dict(ring_bins=48, pad=1, apodize=True, k_min=6.0, min_counts_per_ring=10)
    fit_bounds = dict(kmin=4, kmax=25)
    
    # 4) Define λ² grid for sweep (uniform in λ², covering χ ∈ [0.1, 20])
    lam2_min = 0.05 / (2.0 * sigmaPhi0)  # χ ≈ 0.1
    lam2_max = 20.0 / (2.0 * sigmaPhi0)  # χ ≈ 20
    n_points = 100
    lam2_values = np.linspace(lam2_min, lam2_max, n_points)
    
    print(f"\nλ² sweep: {lam2_min:.3f} to {lam2_max:.3f} ({n_points} points)")
    print(f"Corresponding χ range: {2*sigmaPhi0*lam2_min:.2f} to {2*sigmaPhi0*lam2_max:.2f}")
    
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
        chi = 2.0 * sigmaPhi0 * cfg.faraday_const * lam2
        
        print(f"  λ²={lam2:.3f}, χ={chi:.2f} ({i+1}/{n_points})")
        
        # Build maps for this λ
        P_map = P_map_mixed(polarized_emissivity_simple(Bx, By, gamma=gamma), phi, lam, cfg)
        dP_map = dP_map_mixed(polarized_emissivity_simple(Bx, By, gamma=gamma), phi, lam, cfg)
        
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
    
    # 7) Save results to NPZ for later reproducibility
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
        sigmaPhi0=np.array(sigmaPhi0),
        faraday_const=np.array(cfg.faraday_const),
        los_axis=np.array(cfg.los_axis),
        gamma=np.array(gamma),
    )
    print(f"\nSaved sweep data to: {npz_path}")

    # 8) Create slope vs χ plot
    print(f"\nCreating slope vs χ plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # (plots already configured) 
    # (plots already configured)
    # (plots already configured)
    # (plots already configured)
    
    # Plot P slopes
    ax1.errorbar(chi_values, slopes_P, yerr=errors_P, fmt='o-', capsize=3, 
                 label='PSA of P', color='blue', alpha=0.7)
    ax1.set_xlabel('χ = 2σΦλ²')
    ax1.set_ylabel('PSA Slope of P')
    ax1.set_title('PSA Slope of P vs χ (Mixed Geometry, Random-Dominated)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot dP/dλ² slopes
    ax2.errorbar(chi_values, slopes_dP, yerr=errors_dP, fmt='s-', capsize=3, 
                 label='PSA of dP/dλ²', color='red', alpha=0.7)
    ax2.set_xlabel('χ = 2σΦλ²')
    ax2.set_ylabel('PSA Slope of dP/dλ²')
    ax2.set_title('PSA Slope of dP/dλ² vs χ (Mixed Geometry, Random-Dominated)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the slope vs χ plot
    slope_plot_path = os.path.join(output_dir, "psa_slopes_vs_chi.png")
    
    print(f"Saving slope vs χ plot to: {slope_plot_path}")
    plt.savefig(slope_plot_path, dpi=300, bbox_inches='tight')
    
    # 8) Also create a combined plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(chi_values, slopes_P, yerr=errors_P, fmt='o-', capsize=3, 
                 label='PSA of P', color='blue', alpha=0.7)
    plt.errorbar(chi_values, slopes_dP, yerr=errors_dP, fmt='s-', capsize=3, 
                 label='PSA of dP/dλ²', color='red', alpha=0.7)
    plt.xlabel('χ = 2σΦλ²')
    plt.ylabel('PSA Slope')
    plt.title('PSA Slopes vs χ (Mixed Geometry, Random-Dominated)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
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
