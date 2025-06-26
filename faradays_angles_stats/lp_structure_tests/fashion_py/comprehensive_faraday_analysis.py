#!/usr/bin/env python3
"""
comprehensive_faraday_analysis.py
================================

Comprehensive analysis of Faraday screen statistics addressing Dr. Lazarian's concerns:

1. Systematic lambda dependence study
2. 2π ambiguity handling for large rotation measures
3. Comparison between RM structure function and angle structure function  
4. Mean field vs fluctuation-only analysis
5. Pure power law validation using synthetic data

Author : <you>
Date   : 2025-06-25
License: MIT
"""

from pathlib import Path
import argparse
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftn, irfftn, fftshift
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter


# ────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────
def _axis_spacing(coord1d, name="axis"):
    uniq = np.unique(coord1d.ravel())
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    if d.size:
        return float(np.median(d))
    print(f"[!] {name}: spacing undetermined – using 1")
    return 1.0


def radial_average(map2d, dx=1.0, nbins=64, r_min=1e-3, log_bins=True):
    ny, nx = map2d.shape
    y, x   = np.indices((ny, nx))
    y      = y - ny // 2
    x      = x - nx // 2
    R      = np.hypot(x, y) * dx
    r_max  = R.max() / 2

    if log_bins:
        bins = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
    else:
        bins = np.linspace(0.0, r_max, nbins + 1)

    prof, _, _ = binned_statistic(R.ravel(), map2d.ravel(),
                                  statistic="mean", bins=bins)
    R_cent = 0.5 * (bins[1:] + bins[:-1])
    mask   = ~np.isnan(prof) & (R_cent > r_min)
    return R_cent[mask], prof[mask]


def structure_function_2d(field2d, dx=1.0, **kwa):
    """2D structure function via autocorrelation."""
    f   = field2d - field2d.mean()
    acf = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    acf = fftshift(acf)
    D   = 2.0 * f.var() - 2.0 * acf
    D[D < 0] = 0
    return radial_average(D, dx=dx, **kwa)


def autocorr_complex(field2d, dx=1.0, **kwa):
    """Complex autocorrelation, normalized so S(R=0)=1."""
    ac = irfftn(np.abs(rfftn(field2d))**2, s=field2d.shape) / field2d.size
    ac = fftshift(ac).real
    ac /= ac[ac.shape[0] // 2, ac.shape[1] // 2]
    return radial_average(ac, dx=dx, **kwa)


def local_log_slope(R, D, win=9, poly=2):
    """Smoothed derivative d ln D / d ln R using Savitzky-Golay."""
    logR  = np.log10(R)
    logD  = np.log10(D)
    logDs = savgol_filter(logD, win, poly)
    slope = np.gradient(logDs, logR, edge_order=2)
    return slope


def angle_structure_function(D_phi, lam):
    """D_φ(R, λ) = ½ [1 − exp(−2 λ⁴ D_Φ)]"""
    return 0.5 * (1.0 - np.exp(-2.0 * lam**4 * D_phi))


def check_two_pi_ambiguity(Phi, threshold_factor=0.5):
    """
    Check if rotation measure map has 2π ambiguity issues.
    Returns fraction of pixels where |Φ| > threshold_factor * π.
    """
    phi_max = np.abs(Phi).max()
    ambiguous_pixels = np.sum(np.abs(Phi) > threshold_factor * np.pi)
    total_pixels = Phi.size
    fraction = ambiguous_pixels / total_pixels
    
    print(f"RM statistics:")
    print(f"  max |Φ| = {phi_max:.3f} rad")
    print(f"  pixels with |Φ| > {threshold_factor}π: {fraction:.1%}")
    print(f"  potential 2π ambiguity: {'YES' if phi_max > np.pi else 'NO'}")
    
    return fraction, phi_max


# ────────────────────────────────────────────────────────────────────
# Main analysis functions
# ────────────────────────────────────────────────────────────────────
def analyze_cube(cube_path: Path, 
                 ne_key="gas_density",
                 bz_key="k_mag_field",
                 lambda_range=(0.02, 0.5),
                 n_lambda=10,
                 nbins=72,
                 test_fluctuations_only=True):
    """
    Comprehensive analysis of a single cube.
    """
    cube_path = cube_path.expanduser()
    results = {}
    
    with h5py.File(cube_path, "r") as f:
        ne = f[ne_key][:]
        bz = f[bz_key][:]
        
        # Check if fluctuations-only field exists
        bz_fluct = None
        if test_fluctuations_only and "k_mag_field_fluctuations" in f:
            bz_fluct = f["k_mag_field_fluctuations"][:]
            
        dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0
        dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0
        
        # Store metadata if available
        if hasattr(f, 'attrs'):
            for key in ['beta_ne', 'beta_bz', 'mean_bz']:
                if key in f.attrs:
                    results[key] = f.attrs[key]

    print(f"\nAnalyzing {cube_path.name}:")
    print(f"  cube shape: {ne.shape}")
    print(f"  dx, dz: {dx}, {dz}")
    
    # Main analysis with total field (mean + fluctuations)
    Phi_total = (ne * bz).sum(axis=2) * dz
    results['total_field'] = analyze_phi_map(Phi_total, dx, lambda_range, n_lambda, nbins, "Total field (B + ΔB)")
    
    # Analysis with fluctuations only (if available)
    if bz_fluct is not None:
        Phi_fluct = (ne * bz_fluct).sum(axis=2) * dz
        results['fluctuations_only'] = analyze_phi_map(Phi_fluct, dx, lambda_range, n_lambda, nbins, "Fluctuations only (ΔB)")
    
    return results


def analyze_phi_map(Phi, dx, lambda_range, n_lambda, nbins, title):
    """Analyze a single Φ map."""
    print(f"\n--- {title} ---")
    
    if Phi.std() == 0:
        print("  WARNING: Φ has zero variance!")
        return None
        
    # Check for 2π ambiguity
    ambig_fraction, phi_max = check_two_pi_ambiguity(Phi)
    
    # Direct structure function of Φ
    R, D_phi = structure_function_2d(Phi, dx=dx, nbins=nbins, log_bins=True)
    
    # Lambda-dependent analysis
    lambda_values = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambda)
    
    angle_structure_data = []
    collapse_data = []
    
    for lam in lambda_values:
        # Complex polarization field
        P = np.exp(1j * 2.0 * lam**2 * Phi)
        R_S, S = autocorr_complex(P, dx=dx, nbins=nbins, log_bins=True)
        
        # Collapse test: -ln S / (2λ⁴) ≈ D_Φ
        S_interp = np.interp(R, R_S, S)
        valid = (S_interp > 1e-6)  # Avoid log of very small numbers
        
        D_est = np.empty_like(S_interp)
        D_est[:] = np.nan
        D_est[valid] = -np.log(S_interp[valid]) / (2.0 * lam**4)
        
        collapse_data.append({
            'lambda': lam,
            'R': R,
            'D_estimated': D_est,
            'valid_mask': valid
        })
        
        # Angle structure function
        D_angle = angle_structure_function(D_phi, lam)
        angle_structure_data.append({
            'lambda': lam,
            'R': R,
            'D_angle': D_angle
        })
    
    # Local slope analysis
    slope = local_log_slope(R, D_phi, win=11, poly=2)
    
    return {
        'R': R,
        'D_phi': D_phi,
        'slope': slope,
        'lambda_values': lambda_values,
        'angle_structure_data': angle_structure_data,
        'collapse_data': collapse_data,
        'phi_max': phi_max,
        'ambiguity_fraction': ambig_fraction,
        'title': title
    }


def plot_comprehensive_results(results_dict, output_dir=Path(".")):
    """Generate comprehensive plots addressing Dr. Lazarian's concerns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Lambda dependence study
    plot_lambda_dependence(results_dict, output_dir)
    
    # 2. Collapse test comparison
    plot_collapse_comparison(results_dict, output_dir)
    
    # 3. Power law range analysis
    plot_powerlaw_range(results_dict, output_dir)
    
    # 4. Mean field vs fluctuation comparison
    plot_field_comparison(results_dict, output_dir)


def plot_lambda_dependence(results_dict, output_dir):
    """Plot systematic lambda dependence as requested by Dr. Lazarian."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Systematic Lambda Dependence Study", fontsize=14)
    
    for cube_name, cube_results in results_dict.items():
        if 'total_field' not in cube_results:
            continue
            
        data = cube_results['total_field']
        if data is None:
            continue
            
        # Plot angle structure functions for different lambdas
        ax = axes[0, 0]
        for i, ang_data in enumerate(data['angle_structure_data'][::2]):  # Every other lambda
            ax.loglog(ang_data['R'], ang_data['D_angle'], 
                     label=f"λ={ang_data['lambda']:.3f}", alpha=0.7)
        ax.set_xlabel("R")
        ax.set_ylabel("D_φ(R,λ)")
        ax.set_title("Angle Structure Functions")
        ax.legend(fontsize=8)
        
        # Lambda^4 scaling test
        ax = axes[0, 1]
        mid_idx = len(data['R']) // 2
        R_mid = data['R'][mid_idx]
        scaling_values = []
        
        for ang_data in data['angle_structure_data']:
            if mid_idx < len(ang_data['D_angle']):
                scaling_values.append(ang_data['D_angle'][mid_idx])
        
        if scaling_values:
            ax.loglog(data['lambda_values'], scaling_values, 'o-', label=cube_name)
            ax.loglog(data['lambda_values'], 
                     scaling_values[0] * (data['lambda_values'] / data['lambda_values'][0])**4,
                     '--', alpha=0.5, label=r'∝ λ^4')
        ax.set_xlabel("λ")
        ax.set_ylabel(f"D_φ(R={R_mid:.1f}, λ)")
        ax.set_title("Lambda^4 Scaling Test")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "lambda_dependence_study.pdf", bbox_inches='tight')
    plt.close()


def plot_collapse_comparison(results_dict, output_dir):
    """Plot collapse test comparing different methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Logarithmic Collapse Test: Direct vs Angle-based Methods", fontsize=14)
    
    for cube_name, cube_results in results_dict.items():
        if 'total_field' not in cube_results:
            continue
            
        data = cube_results['total_field']
        if data is None:
            continue
            
        ax = axes[0]
        ax.loglog(data['R'], data['D_phi'], 'k-', lw=2, label=f"{cube_name}: Direct D_Φ")
        
        # Show collapse for few selected lambdas
        selected_lambdas = [data['lambda_values'][i] for i in [1, len(data['lambda_values'])//2, -2]]
        
        for i, lam in enumerate(selected_lambdas):
            collapse_item = next(item for item in data['collapse_data'] if np.isclose(item['lambda'], lam))
            valid = collapse_item['valid_mask']
            ax.loglog(collapse_item['R'][valid], collapse_item['D_estimated'][valid],
                     '--', alpha=0.7, label=f"λ={lam:.3f}")
        
        # Reference 5/3 slope
        ax.loglog(data['R'], data['D_phi'][5] * (data['R'] / data['R'][5])**(5/3),
                 ':', color='red', alpha=0.5, label=r'∝ R^{5/3}')
        
        ax.set_xlabel("R")
        ax.set_ylabel("D_Φ")
        ax.set_title("Collapse Test")
        ax.legend(fontsize=8)
        
        # Local slope
        ax = axes[1]
        ax.semilogx(data['R'], data['slope'], label=f"{cube_name}")
        ax.axhline(5/3, color='red', ls=':', alpha=0.5, label='5/3')
        ax.fill_between(data['R'], 5/3 - 0.15, 5/3 + 0.15, 
                       color='red', alpha=0.1)
        ax.set_xlabel("R")
        ax.set_ylabel("d ln D / d ln R")
        ax.set_title("Local Power Law Index")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "collapse_comparison.pdf", bbox_inches='tight')
    plt.close()


def plot_powerlaw_range(results_dict, output_dir):
    """Analyze and plot power law range as Dr. Lazarian requested."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for cube_name, cube_results in results_dict.items():
        if 'total_field' not in cube_results:
            continue
            
        data = cube_results['total_field']
        if data is None:
            continue
            
        # Find inertial range where slope is close to 5/3
        slope_target = 5/3
        slope_tolerance = 0.2
        
        inertial_mask = np.abs(data['slope'] - slope_target) < slope_tolerance
        if np.any(inertial_mask):
            inertial_range = data['R'][inertial_mask]
            r_min, r_max = inertial_range.min(), inertial_range.max()
            range_factor = r_max / r_min
            
            ax.loglog(data['R'], data['D_phi'], label=f"{cube_name} (range: {range_factor:.1f}×)")
            ax.axvspan(r_min, r_max, alpha=0.2, label=f"Inertial range")
            
    ax.loglog(data['R'], data['D_phi'][5] * (data['R'] / data['R'][5])**(5/3),
             'k--', alpha=0.5, label=r'R^{5/3}')
    ax.set_xlabel("R")
    ax.set_ylabel("D_Φ")
    ax.set_title("Power Law Range Comparison")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "powerlaw_range.pdf", bbox_inches='tight')
    plt.close()


def plot_field_comparison(results_dict, output_dir):
    """Compare mean field vs fluctuation-only results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Mean Field vs Fluctuations-Only Comparison", fontsize=14)
    
    for cube_name, cube_results in results_dict.items():
        if 'total_field' not in cube_results or 'fluctuations_only' not in cube_results:
            continue
            
        total_data = cube_results['total_field']
        fluct_data = cube_results['fluctuations_only']
        
        if total_data is None or fluct_data is None:
            continue
        
        # Structure functions
        ax = axes[0, 0]
        ax.loglog(total_data['R'], total_data['D_phi'], 'b-', label=f"{cube_name}: Total")
        ax.loglog(fluct_data['R'], fluct_data['D_phi'], 'r--', label=f"{cube_name}: Fluctuations")
        ax.set_xlabel("R")
        ax.set_ylabel("D_Φ")
        ax.set_title("Structure Functions")
        ax.legend()
        
        # Local slopes
        ax = axes[0, 1]
        ax.semilogx(total_data['R'], total_data['slope'], 'b-', label=f"{cube_name}: Total")
        ax.semilogx(fluct_data['R'], fluct_data['slope'], 'r--', label=f"{cube_name}: Fluctuations")
        ax.axhline(5/3, color='k', ls=':', alpha=0.5)
        ax.set_xlabel("R")
        ax.set_ylabel("Local slope")
        ax.set_title("Power Law Indices")
        ax.legend()
        
        # 2π ambiguity comparison
        ax = axes[1, 0]
        ax.bar([f"{cube_name}\nTotal", f"{cube_name}\nFluct"], 
               [total_data['ambiguity_fraction'], fluct_data['ambiguity_fraction']])
        ax.set_ylabel("Fraction > π/2")
        ax.set_title("2π Ambiguity Risk")
        
        # Max RM comparison
        ax = axes[1, 1]
        ax.bar([f"{cube_name}\nTotal", f"{cube_name}\nFluct"], 
               [total_data['phi_max'], fluct_data['phi_max']])
        ax.axhline(np.pi, color='red', ls='--', label='π threshold')
        ax.set_ylabel("Max |Φ| (rad)")
        ax.set_title("Maximum Rotation Measure")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "field_comparison.pdf", bbox_inches='tight')
    plt.close()


# ────────────────────────────────────────────────────────────────────
# Main function
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Faraday screen analysis addressing Dr. Lazarian's concerns"
    )
    parser.add_argument("cubes", nargs="+", help="HDF5 cube files to analyze")
    parser.add_argument("--lambda-range", nargs=2, type=float, default=[0.02, 0.5],
                       help="Lambda range for systematic study (default: 0.02 0.5)")
    parser.add_argument("--n-lambda", type=int, default=10,
                       help="Number of lambda values to test (default: 10)")
    parser.add_argument("--output-dir", type=Path, default="analysis_results",
                       help="Output directory for plots")
    parser.add_argument("--ne-key", default="gas_density", help="Electron density dataset name")
    parser.add_argument("--bz-key", default="k_mag_field", help="Magnetic field dataset name")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Analyze all cubes
    all_results = {}
    
    for cube_file in args.cubes:
        cube_path = Path(cube_file)
        if not cube_path.exists():
            print(f"Warning: {cube_path} does not exist, skipping.")
            continue
            
        try:
            results = analyze_cube(
                cube_path,
                ne_key=args.ne_key,
                bz_key=args.bz_key,
                lambda_range=args.lambda_range,
                n_lambda=args.n_lambda
            )
            all_results[cube_path.stem] = results
        except Exception as e:
            print(f"Error analyzing {cube_path}: {e}")
            continue
    
    if not all_results:
        print("No cubes successfully analyzed!")
        return
        
    # Generate comprehensive plots
    print(f"\nGenerating plots in {args.output_dir}/")
    plot_comprehensive_results(all_results, args.output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 