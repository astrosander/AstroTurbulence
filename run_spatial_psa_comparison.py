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
    
    # Compute PSA for both maps
    k_P, Ek_P = psa_of_map(P_map, ring_bins=48, pad=1, apodize=True, k_min=6.0)
    k_dP, Ek_dP = psa_of_map(dP_map, ring_bins=48, pad=1, apodize=True, k_min=6.0)
    
    print(f"Generated PSA data: {len(k_P)} points for P, {len(k_dP)} points for dP/dλ²")
    
    return k_P, Ek_P, k_dP, Ek_dP

def lambda_zero_diagnostics(Pi, phi, Bx, By, cfg, case="Mixed"):
    """Simple λ=0 diagnostics for the comparison script."""
    print(f"\n{'='*60}")
    print(f"λ=0 DIAGNOSTICS ({case} case)")
    print(f"{'='*60}")
    
    # Create λ=0 polarization map
    P0 = P_map_mixed(Pi, phi, lam=0.0, cfg=cfg)
    
    # Test different conventions
    tests = [
        ("PSD, no window", False, False),
        ("PSD, Hann window", True, False),
        ("Energy-like, Hann", True, True),
    ]
    
    print(f"\nQuadratic emissivity (P_i = (B_x + iB_y)^2):")
    for label, apod, energy_like in tests:
        k, S = psa_of_map(P0, apodize=apod, return_energy_like=energy_like, k_min=3)
        m, a, err, _ = fit_log_slope_window(k, S)
        print(f"  {label:20s}: slope {m:6.2f} ± {err:.2f}")
    
    # Test linear tracer
    print(f"\nLinear tracer (P_i = B_x + iB_y):")
    Pi_linear = linear_tracer(Bx, By)
    P0_linear = P_map_mixed(Pi_linear, phi, lam=0.0, cfg=cfg)
    
    for label, apod, energy_like in tests:
        k, S = psa_of_map(P0_linear, apodize=apod, return_energy_like=energy_like, k_min=3)
        m, a, err, _ = fit_log_slope_window(k, S)
        print(f"  {label:20s}: slope {m:6.2f} ± {err:.2f}")
    
    # Test LP16 emissivity
    print(f"\nLP16 emissivity (γ=2, with amplitude factor):")
    Pi_lp16 = polarized_emissivity_lp16(Bx, By, gamma=2.0)
    P0_lp16 = P_map_mixed(Pi_lp16, phi, lam=0.0, cfg=cfg)
    
    for label, apod, energy_like in tests:
        k, S = psa_of_map(P0_lp16, apodize=apod, return_energy_like=energy_like, k_min=3)
        m, a, err, _ = fit_log_slope_window(k, S)
        print(f"  {label:20s}: slope {m:6.2f} ± {err:.2f}")
    
    print(f"\nExpected slopes:")
    print(f"  Linear tracer, 2D-PSD:     ~-2.67 (projection of 3D Kolmogorov)")
    print(f"  Linear tracer, 2D-energy:   ~-1.67 (PSD slope + 1)")
    print(f"  Quadratic, 2D-PSD:         ~-4.4  (quadratic + projection + spin-2)")
    print(f"  Quadratic, 2D-energy:      ~-3.4  (PSD slope + 1)")
    print(f"  LP16 emissivity:           slightly flatter than pure quadratic")

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
    
    # Create the plot
    m_P, m_dP, err_P, err_dP = plot_spatial_psa_comparison(
        k_P=k_P, 
        Ek_P=Ek_P, 
        k_dP=k_dP, 
        Ek_dP=Ek_dP, 
        lam=lam, 
        case="Mixed"
    )
    
    # Save the plot to the specified output path
    output_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\compare_measures\lp2016_outputs\derivative_slope_analysis_random.png"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nPlot saved successfully!")
    print(f"PSA of P slope: {m_P:.2f} ± {err_P:.2f}")
    print(f"PSA of dP/dλ² slope: {m_dP:.2f} ± {err_dP:.2f}")
    print(f"Difference: {abs(m_dP - m_P):.2f}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
