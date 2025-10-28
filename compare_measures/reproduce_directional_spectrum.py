#!/usr/bin/env python3
"""
Reproduce plots from saved NPZ data files.

This script loads NPZ files from various analysis scripts and recreates the plots:
1. curve_mixed_chi.npz from compute_pfa_vs_chi_lp16.py
2. dir_slopes_vs_lambda_mixed_data.npz from run_directional_spectrum_vs_lambda.py  
3. psa_slopes_vs_chi_data.npz from run_spatial_psa_comparison.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make project root importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from compare_measures.pfa_and_derivative_lp_16_like import plot_dir_slopes_vs_lambda


def load_npz_data(npz_path: str):
    """Load data from NPZ file and return relevant arrays."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Determine data type based on available keys
    if 'lam_arr' in data and 'slopes_dir' in data:
        # Directional spectrum data (original mode)
        lam_arr = data['lam_arr']
        slopes_dir = data['slopes_dir']
        errors_dir = data['errors_dir']
        chi_values = data['chi_values']
        sigmaPhi0 = float(data['sigmaPhi0'])
        geometry = str(data['geometry']) if 'geometry' in data else 'mixed'
        data_type = 'directional_original'
    elif 'lam2_values' in data and 'slopes_dir' in data:
        # Directional spectrum data (sweep mode)
        lam2_values = data['lam2_values']
        chi_values = data['chi_values']
        slopes_dir = data['slopes_dir']
        errors_dir = data['errors_dir']
        lam_arr = np.sqrt(lam2_values)
        sigmaPhi0 = float(data['sigmaPhi0'])
        geometry = str(data['geometry']) if 'geometry' in data else 'mixed'
        data_type = 'directional_sweep'
    elif 'chi_values' in data and 'slopes_P' in data:
        # PSA slopes data
        chi_values = data['chi_values']
        slopes_P = data['slopes_P']
        slopes_dP = data['slopes_dP']
        errors_P = data['errors_P']
        errors_dP = data['errors_dP']
        sigmaPhi0 = float(data['sigmaPhi0'])
        data_type = 'psa_slopes'
    elif 'chi_values' in data and 'pfa_var' in data:
        # PFA variance data
        chi_values = data['chi_values']
        pfa_var = data['pfa_var']
        sigmaPhi0 = float(data['sigmaPhi0'])
        data_type = 'pfa_variance'
    else:
        # Generic fallback: try to infer PFA variance style data
        keys = list(data.keys())
        # Prefer keys containing these substrings
        def pick_key(candidates):
            for c in candidates:
                for k in keys:
                    if c.lower() in k.lower():
                        return k
            return None
        chi_key = pick_key(["chi_values", "chi", "x", "lambda2", "lam2"])
        pfa_key = pick_key(["pfa_var", "pfa", "variance", "y", "pvar", "PFA"])
        if chi_key is not None and pfa_key is not None:
            data_type = 'pfa_variance_generic'
        else:
            raise ValueError(f"Unknown NPZ data format in {npz_path}")
    
    return data, data_type


def reproduce_directional_original_plot(npz_path: str, output_path: str = None):
    """Reproduce the directional spectrum original mode plot (λ vs slopes)."""
    data, data_type = load_npz_data(npz_path)
    
    if data_type != 'directional_original':
        print(f"Warning: Expected directional original data, got {data_type}")
    
    lam_arr = data['lam_arr']
    slopes_dir = data['slopes_dir']
    errors_dir = data['errors_dir']
    chi_values = data['chi_values']
    sigmaPhi0 = float(data['sigmaPhi0'])
    geometry = str(data['geometry']) if 'geometry' in data else 'mixed'
    
    print(f"Reproducing directional original mode plot from: {npz_path}")
    print(f"Data points: {len(lam_arr)}")
    print(f"Lambda range: {lam_arr.min():.3f} to {lam_arr.max():.3f}")
    print(f"Chi range: {chi_values.min():.2f} to {chi_values.max():.2f}")
    
    # Create the plot using the same function as the original script
    title = f"Directional-spectrum slope ({geometry.capitalize()}, χ ∈ [0, 20])"
    plot_dir_slopes_vs_lambda(lam_arr, slopes_dir, errors_dir, 
                             x_is_lambda2=True, title=title, sigma_phi=sigmaPhi0)
    
    # Save the plot
    if output_path is None:
        output_dir = os.path.join(THIS_DIR, "lp2016_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"reproduced_dir_slopes_vs_lambda_{geometry}.png")
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved reproduced plot: {output_path}")
    return output_path


def reproduce_directional_sweep_plot(npz_path: str, output_path: str = None):
    """Reproduce the directional spectrum sweep mode plot (χ vs slopes)."""
    data, data_type = load_npz_data(npz_path)
    
    if data_type != 'directional_sweep':
        print(f"Warning: Expected directional sweep data, got {data_type}")
    
    chi_values = data['chi_values']
    slopes_dir = data['slopes_dir']
    errors_dir = data['errors_dir']
    sigmaPhi0 = float(data['sigmaPhi0'])
    geometry = str(data['geometry']) if 'geometry' in data else 'mixed'
    
    print(f"Reproducing directional sweep mode plot from: {npz_path}")
    print(f"Data points: {len(chi_values)}")
    print(f"Chi range: {chi_values.min():.2f} to {chi_values.max():.2f}")
    print(f"Slopes range: {slopes_dir.min():.2f} to {slopes_dir.max():.2f}")
    
    # Create the plot with the same styling as the original script
    plt.figure(figsize=(10, 6))
    plt.errorbar(chi_values, slopes_dir, yerr=errors_dir, fmt='d-', capsize=3, 
                 label=f'Directional spectrum ({geometry})', color='green', alpha=0.7)
    plt.xlabel('χ = 2σ_φ λ²')
    plt.ylabel('Directional Spectrum Slope')
    plt.title(f'Directional Spectrum Slope vs χ ({geometry.capitalize()} Geometry, χ ∈ [0, 20])')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    if output_path is None:
        output_dir = os.path.join(THIS_DIR, "lp2016_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"reproduced_dir_slopes_vs_chi_{geometry}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reproduced plot: {output_path}")
    return output_path


def reproduce_psa_slopes_plot(npz_path: str, output_path: str = None):
    """Reproduce PSA slopes plot from run_spatial_psa_comparison.py data."""
    data, data_type = load_npz_data(npz_path)
    
    if data_type != 'psa_slopes':
        print(f"Warning: Expected PSA slopes data, got {data_type}")
    
    chi_values = data['chi_values']
    slopes_P = data['slopes_P']
    slopes_dP = data['slopes_dP']
    errors_P = data['errors_P']
    errors_dP = data['errors_dP']
    sigmaPhi0 = float(data['sigmaPhi0'])
    
    print(f"Reproducing PSA slopes plot from: {npz_path}")
    print(f"Data points: {len(chi_values)}")
    print(f"Chi range: {chi_values.min():.2f} to {chi_values.max():.2f}")
    print(f"P slopes range: {slopes_P.min():.2f} to {slopes_P.max():.2f}")
    print(f"dP slopes range: {slopes_dP.min():.2f} to {slopes_dP.max():.2f}")
    
    # Create combined plot (same as run_spatial_psa_comparison.py)
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
    
    # Save the plot
    if output_path is None:
        output_dir = os.path.join(THIS_DIR, "lp2016_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "reproduced_psa_slopes_vs_chi_combined.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reproduced plot: {output_path}")
    return output_path


def reproduce_pfa_variance_plot(npz_path: str, output_path: str = None):
    """Reproduce PFA variance plot from compute_pfa_vs_chi_lp16.py data."""
    data, data_type = load_npz_data(npz_path)
    
    if data_type not in ('pfa_variance', 'pfa_variance_generic'):
        print(f"Warning: Expected PFA variance data, got {data_type}")
    
    if data_type == 'pfa_variance':
        chi_values = data['chi_values']
        pfa_var = data['pfa_var']
    else:
        # generic fallback: pick best-matching keys
        keys = list(data.keys())
        def pick_key(candidates):
            for c in candidates:
                for k in keys:
                    if c.lower() in k.lower():
                        return k
            return None
        chi_key = pick_key(["chi_values", "chi", "x", "lambda2", "lam2"])
        pfa_key = pick_key(["pfa_var", "pfa", "variance", "y", "pvar", "PFA"])
        chi_values = data[chi_key]
        pfa_var = data[pfa_key]
    
    print(f"Reproducing PFA variance plot from: {npz_path}")
    print(f"Data points: {len(chi_values)}")
    print(f"Chi range: {chi_values.min():.2f} to {chi_values.max():.2f}")
    print(f"PFA variance range: {pfa_var.min():.3g} to {pfa_var.max():.3g}")
    
    # Create PFA variance plot
    plt.figure(figsize=(10, 6))
    plt.loglog(chi_values, pfa_var, 'o-', markersize=3, alpha=0.7, 
               label='PFA variance', color='purple')
    plt.xlabel('χ = 2σ_φ λ²')
    plt.ylabel('⟨|P|²⟩ (PFA variance)')
    plt.title('PFA Variance vs χ')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot: default next to NPZ as PFA.png
    if output_path is None:
        npz_dir = os.path.dirname(npz_path)
        output_path = os.path.join(npz_dir, "PFA.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reproduced plot: {output_path}")
    return output_path


def reproduce_all_plots(npz_dir: str = None):
    """Reproduce all available plots from NPZ data files."""
    if npz_dir is None:
        npz_dir = os.path.join(THIS_DIR, "lp2016_outputs")
    
    reproduced_files = []
    
    # Define all possible NPZ files and their corresponding plot functions
    plot_configs = [
        # Directional spectrum plots
        ("dir_slopes_vs_lambda_mixed_data.npz", reproduce_directional_original_plot),
        ("dir_slopes_vs_lambda_separated_data.npz", reproduce_directional_original_plot),
        ("dir_slopes_vs_chi_mixed_data.npz", reproduce_directional_sweep_plot),
        ("dir_slopes_vs_chi_separated_data.npz", reproduce_directional_sweep_plot),
        # PSA slopes plot
        ("psa_slopes_vs_chi_data.npz", reproduce_psa_slopes_plot),
        # PFA variance plot
        ("curve_mixed_chi.npz", reproduce_pfa_variance_plot),
    ]
    
    for npz_file, plot_func in plot_configs:
        npz_path = os.path.join(npz_dir, npz_file)
        
        if os.path.exists(npz_path):
            try:
                output_path = plot_func(npz_path)
                reproduced_files.append(output_path)
            except Exception as e:
                print(f"Error reproducing {npz_file}: {e}")
        else:
            print(f"Data file not found: {npz_path}")
    
    return reproduced_files


def main():
    """Reproduce all available plots from NPZ data files."""
    # Check both lp2016_outputs and npz directories
    npz_dirs = [
        os.path.join(THIS_DIR, "lp2016_outputs"),
        os.path.join(THIS_DIR, "npz")
    ]
    
    print("="*70)
    print("REPRODUCING PLOTS FROM NPZ DATA FILES")
    print("="*70)
    
    reproduced_files = []

    # Priority: reproduce PFA from compare_measures/npz/curve_mixed_chi.npz → PFA.png
    pfa_npz_path = os.path.join(THIS_DIR, "npz", "curve_mixed_chi.npz")
    if os.path.exists(pfa_npz_path):
        try:
            pfa_png_path = os.path.join(THIS_DIR, "npz", "PFA.png")
            print(f"\nReproducing PFA from: {pfa_npz_path}")
            reproduce_pfa_variance_plot(pfa_npz_path, pfa_png_path)
            reproduced_files.append(pfa_png_path)
        except Exception as e:
            print(f"Error reproducing PFA from {pfa_npz_path}: {e}")
    
    # Define all possible NPZ files and their corresponding plot functions
    plot_configs = [
        # Directional spectrum plots
        ("dir_slopes_vs_lambda_mixed_data.npz", reproduce_directional_original_plot),
        ("dir_slopes_vs_lambda_separated_data.npz", reproduce_directional_original_plot),
        ("dir_slopes_vs_chi_mixed_data.npz", reproduce_directional_sweep_plot),
        ("dir_slopes_vs_chi_separated_data.npz", reproduce_directional_sweep_plot),
        # PSA slopes plot
        ("psa_slopes_vs_chi_data.npz", reproduce_psa_slopes_plot),
        # PFA variance plot
        ("curve_mixed_chi.npz", reproduce_pfa_variance_plot),
    ]
    
    for npz_file, plot_func in plot_configs:
        npz_path = None
        found_dir = None
        
        # Look for the file in all directories
        for npz_dir in npz_dirs:
            test_path = os.path.join(npz_dir, npz_file)
            if os.path.exists(test_path):
                npz_path = test_path
                found_dir = npz_dir
                break
        
        if npz_path:
            try:
                print(f"\nReproducing from: {npz_file} (found in {found_dir})")
                output_path = plot_func(npz_path)
                reproduced_files.append(output_path)
            except Exception as e:
                print(f"Error reproducing {npz_file}: {e}")
        else:
            print(f"Data file not found: {npz_file} (searched in {npz_dirs})")
    
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if reproduced_files:
        print(f"Successfully reproduced {len(reproduced_files)} plots:")
        for file_path in reproduced_files:
            print(f"  • {file_path}")
    else:
        print("No NPZ data files found to reproduce.")
    
    print("="*70)


if __name__ == "__main__":
    main()
