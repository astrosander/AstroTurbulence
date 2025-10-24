"""
Replot measures from saved NPZ data.

This module provides functionality to load previously computed measures
and recreate the 1x3 panel plots without re-running the expensive calculations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

def load_and_replot(npz_path: str, output_dir: str = "lp2016_outputs", save_png: bool = True):
    """
    Load saved measures data and recreate the 2x3 panel plot showing all 6 measures.
    
    Parameters:
    -----------
    npz_path : str
        Path to the saved NPZ file containing measures data
    output_dir : str
        Directory to save the replot image
    save_png : bool
        Whether to save the plot as PNG file
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load data
    data = np.load(npz_path)
    lambda_list = data['lambda_list']
    derivative_lambda_mid = data['derivative_lambda_mid']
    derivative_variance = data['derivative_variance']
    pfa_variances = data.get('pfa_variances', None)
    derivative_spectrum_variances = data.get('derivative_spectrum_variances', None)
    phi_total_map = data['phi_total_map']
    regime_classification = data['regime_classification']
    
    # Compute slopes from full power spectra data
    print("Computing PSA slopes from full power spectra...")
    psa_slopes = []
    for lam in lambda_list:
        k_key = f'psa_k_lambda_{lam:.3f}'
        P_key = f'psa_P_lambda_{lam:.3f}'
        
        if k_key in data and P_key in data:
            k = data[k_key]
            P = data[P_key]
            # Fit slope using the k-range (5.0-200.0)
            m, a, k_fit, P_fit = fit_log_slope(k, P, kmin=5.0, kmax=200.0)
            psa_slopes.append(m)
        else:
            psa_slopes.append(np.nan)
    
    print("Computing directional spectrum slopes from full power spectra...")
    directional_slopes = []
    for lam in lambda_list:
        k_key = f'dir_k_lambda_{lam:.3f}'
        P_key = f'dir_P_lambda_{lam:.3f}'
        
        if k_key in data and P_key in data:
            k = data[k_key]
            P = data[P_key]
            # Fit slope using the k-range (5.0-200.0)
            m, a, k_fit, P_fit = fit_log_slope(k, P, kmin=5.0, kmax=200.0)
            directional_slopes.append(m)
        else:
            directional_slopes.append(np.nan)
    
    psa_slopes = np.array(psa_slopes)
    directional_slopes = np.array(directional_slopes)
    
    # Create 2x3 panel layout for all 6 measures
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get regime information
    regime_colors = {'weak': 'lightblue', 'transition': 'yellow', 'strong': 'lightcoral'}
    regime_wavelengths = {'weak': [], 'transition': [], 'strong': []}
    for i, lam in enumerate(lambda_list):
        regime = regime_classification[i]
        regime_wavelengths[regime].append(lam)
    
    # Panel 1: PSA slopes
    ax1 = axes[0]
    ax1.plot(lambda_list, psa_slopes, 'b.-', linewidth=1.5, markersize=4, label='PSA slope')
    ax1.set_xlabel("$\\lambda$ (m)", fontsize=12)
    ax1.set_ylabel("PSA slope", fontsize=12)
    ax1.set_title("PSA Slope vs $\\lambda$", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Derivative measure
    ax2 = axes[1]
    ax2.plot(derivative_lambda_mid, derivative_variance, 'g.-', linewidth=1.5, markersize=4, label='Derivative variance')
    ax2.set_xlabel("$\\lambda$ (m)", fontsize=12)
    ax2.set_ylabel("Var(|∂P/∂λ²|)", fontsize=12)
    ax2.set_title("Derivative Variance vs $\\lambda$", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Directional spectrum slope
    ax3 = axes[2]
    ax3.plot(lambda_list, directional_slopes, 'm.-', linewidth=1.5, markersize=4, label='Directional spectrum slope')
    ax3.set_xlabel("$\\lambda$ (m)", fontsize=12)
    ax3.set_ylabel("Directional slope", fontsize=12)
    ax3.set_title("Directional Spectrum Slope vs $\\lambda$", fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: PFA variance
    ax4 = axes[3]
    if pfa_variances is not None:
        ax4.plot(lambda_list, pfa_variances, 'r.-', linewidth=1.5, markersize=4, label='PFA variance')
        ax4.set_xlabel("$\\lambda$ (m)", fontsize=12)
        ax4.set_ylabel("⟨|P|²⟩", fontsize=12)
        ax4.set_title("PFA Variance vs $\\lambda$", fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'PFA variance data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("PFA Variance (not available)", fontsize=14, fontweight='bold')
    
    # Panel 5: Derivative spectrum variance
    ax5 = axes[4]
    if derivative_spectrum_variances is not None:
        # Filter out NaN values
        valid_mask = ~np.isnan(derivative_spectrum_variances)
        if np.any(valid_mask):
            ax5.plot(lambda_list[valid_mask], derivative_spectrum_variances[valid_mask], 'c.-', 
                    linewidth=1.5, markersize=4, label='Derivative spectrum variance')
            ax5.set_xlabel("$\\lambda$ (m)", fontsize=12)
            ax5.set_ylabel("⟨|dP/dλ²|²⟩", fontsize=12)
            ax5.set_title("Derivative Spectrum Variance vs $\\lambda$", fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No valid derivative spectrum data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title("Derivative Spectrum Variance (no data)", fontsize=14, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Derivative spectrum data not available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title("Derivative Spectrum Variance (not available)", fontsize=14, fontweight='bold')
    
    # Panel 6: Combined normalized measures
    ax6 = axes[5]
    
    # Normalize all measures to 0-1 range
    psa_norm = (psa_slopes - np.min(psa_slopes)) / (np.max(psa_slopes) - np.min(psa_slopes) + 1e-10)
    deriv_norm = (derivative_variance - np.min(derivative_variance)) / (np.max(derivative_variance) - np.min(derivative_variance) + 1e-10)
    dir_norm = (directional_slopes - np.nanmin(directional_slopes)) / (np.nanmax(directional_slopes) - np.nanmin(directional_slopes) + 1e-10)
    
    ax6.plot(lambda_list, psa_norm, 'b.-', linewidth=1.5, markersize=4, label='PSA slope')
    ax6.plot(derivative_lambda_mid, deriv_norm, 'g.-', linewidth=1.5, markersize=4, label='Derivative variance')
    ax6.plot(lambda_list, dir_norm, 'm.-', linewidth=1.5, markersize=4, label='Directional slope')
    
    if pfa_variances is not None:
        pfa_norm = (pfa_variances - np.nanmin(pfa_variances)) / (np.nanmax(pfa_variances) - np.nanmin(pfa_variances) + 1e-10)
        ax6.plot(lambda_list, pfa_norm, 'r.-', linewidth=1.5, markersize=4, label='PFA variance')
    
    if derivative_spectrum_variances is not None:
        valid_mask = ~np.isnan(derivative_spectrum_variances)
        if np.any(valid_mask):
            deriv_spec_norm = (derivative_spectrum_variances - np.nanmin(derivative_spectrum_variances)) / (np.nanmax(derivative_spectrum_variances) - np.nanmin(derivative_spectrum_variances) + 1e-10)
            ax6.plot(lambda_list, deriv_spec_norm, 'c.-', linewidth=1.5, markersize=4, label='Derivative spectrum')
    
    ax6.set_xlabel("$\\lambda$ (m)", fontsize=12)
    ax6.set_ylabel("Normalized measure value", fontsize=12)
    ax6.set_title("All Measures Combined (Normalized)", fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.1, 1.1)
    
    # Add regime backgrounds to all panels
    for ax in axes:
        for regime, wavelengths in regime_wavelengths.items():
            if wavelengths:
                wavelengths = sorted(wavelengths)
                start_idx = 0
                for i in range(1, len(wavelengths)):
                    if wavelengths[i] - wavelengths[i-1] > 0.05:
                        if i > start_idx:
                            ax.axvspan(wavelengths[start_idx], wavelengths[i-1], 
                                      color=regime_colors[regime], alpha=0.3, zorder=0)
                        start_idx = i
                if start_idx < len(wavelengths):
                    ax.axvspan(wavelengths[start_idx], wavelengths[-1], 
                              color=regime_colors[regime], alpha=0.3, zorder=0)
    
    # Add regime legend to the combined panel
    from matplotlib.patches import Patch
    regime_patches = [
        Patch(facecolor='lightblue',  alpha=0.25, label=r'$2\lambda^{2}\sigma_{\Phi} < 1$'),
        Patch(facecolor='yellow',     alpha=0.25, label=r'$1 < 2\lambda^{2}\sigma_{\Phi} < 3$'),
        Patch(facecolor='lightcoral', alpha=0.25, label=r'$2\lambda^{2}\sigma_{\Phi} > 3$')
    ]
    axes[5].legend(handles=regime_patches, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_png:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "measures_1x3_panel.png"), dpi=160, bbox_inches="tight")
        print(f"All 6 measures panel saved to: {os.path.join(output_dir, 'measures_1x3_panel.png')}")
    
    plt.show()
    print("All 6 measures replot completed!")

def fit_log_slope(k: np.ndarray, P: np.ndarray, kmin: float = 5.0, kmax: float = 200.0):
    """Fit slope m in log10(P) = a + m log10(k) over a specific k range (5.0-200.0 k)."""
    valid = np.isfinite(P) & (P > 0) & (k > 0)
    if np.sum(valid) < 5:
        return np.nan, np.nan, np.nan, np.nan
    
    kv = k[valid]
    Pv = P[valid]
    
    # Select k range based on absolute k values (4.2-20 k)
    mask = (kv >= kmin) & (kv <= kmax)
    
    if np.sum(mask) < 5:
        return np.nan, np.nan, np.nan, np.nan
    
    kv_range = kv[mask]
    Pv_range = Pv[mask]
    
    X = np.log10(kv_range)
    Y = np.log10(Pv_range)
    m, a = np.polyfit(X, Y, 1)
    
    # Generate fitted line for plotting over the fitting range
    k_fit = np.logspace(np.log10(kmin), np.log10(kmax), 100)
    P_fit = 10**(a + m * np.log10(k_fit))
    
    return m, a, k_fit, P_fit

def plot_full_spectra_panel(npz_path: str, output_dir: str = "lp2016_outputs", save_png: bool = True):
    """
    Plot full power spectra for all wavelengths as a panel with slope lines.
    
    Parameters:
    -----------
    npz_path : str
        Path to the saved NPZ file containing measures data
    output_dir : str
        Directory to save the replot image
    save_png : bool
        Whether to save the plot as PNG file
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load data
    data = np.load(npz_path)
    lambda_list = data['lambda_list']
    regime_classification = data['regime_classification']
    
    # Create 2x1 panel layout (PSA and Directional spectra)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Full Power Spectra with Slope Lines (k = 5.0-200.0) for All Wavelengths', fontsize=16, fontweight='bold')
    
    # Define colors for different wavelengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_list)))
    
    # Panel 1: PSA Power Spectra with slope lines
    ax1 = axes[0]
    psa_slopes = []
    for i, lam in enumerate(lambda_list):
        k_key = f'psa_k_lambda_{lam:.3f}'
        P_key = f'psa_P_lambda_{lam:.3f}'
        
        if k_key in data and P_key in data:
            k = data[k_key]
            P = data[P_key]
            # Only plot valid points
            mask = (P > 0) & np.isfinite(P)
            if np.any(mask):
                # Plot the spectrum
                ax1.loglog(k[mask], P[mask], '.-', color=colors[i], 
                          linewidth=1.5, markersize=3, alpha=0.8,
                          label=f'$\\lambda = {lam:.3f}$ m')
                
                # Fit and plot slope line using k-range (5.0-200.0)
                m, a, k_fit, P_fit = fit_log_slope(k[mask], P[mask], kmin=5.0, kmax=200.0)
                if not np.isnan(m):
                    psa_slopes.append(m)
                    ax1.loglog(k_fit, P_fit, '--', color=colors[i], 
                              linewidth=2, alpha=0.9)
                else:
                    psa_slopes.append(np.nan)
            else:
                psa_slopes.append(np.nan)
        else:
            psa_slopes.append(np.nan)
    
    ax1.set_xlabel("$k$ (pixel$^{-1}$)", fontsize=12)
    ax1.set_ylabel("$P_{|P|}(k)$ (arbitrary units)", fontsize=12)
    ax1.set_title("PSA Power Spectra: $P_{|P|}(k)$ with Fitted Slopes (k = 5.0-200.0)", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Panel 2: Directional Power Spectra with slope lines
    ax2 = axes[1]
    dir_slopes = []
    for i, lam in enumerate(lambda_list):
        k_key = f'dir_k_lambda_{lam:.3f}'
        P_key = f'dir_P_lambda_{lam:.3f}'
        
        if k_key in data and P_key in data:
            k = data[k_key]
            P = data[P_key]
            # Only plot valid points
            mask = (P > 0) & np.isfinite(P)
            if np.any(mask):
                # Plot the spectrum
                ax2.loglog(k[mask], P[mask], '.-', color=colors[i], 
                          linewidth=1.5, markersize=3, alpha=0.8,
                          label=f'$\\lambda = {lam:.3f}$ m')
                
                # Fit and plot slope line using k-range (5.0-200.0)
                m, a, k_fit, P_fit = fit_log_slope(k[mask], P[mask], kmin=5.0, kmax=200.0)
                if not np.isnan(m):
                    dir_slopes.append(m)
                    ax2.loglog(k_fit, P_fit, '--', color=colors[i], 
                              linewidth=2, alpha=0.9)
                else:
                    dir_slopes.append(np.nan)
            else:
                dir_slopes.append(np.nan)
        else:
            dir_slopes.append(np.nan)
    
    ax2.set_xlabel("$k$ (pixel$^{-1}$)", fontsize=12)
    ax2.set_ylabel("$P_{dir}(k)$ (arbitrary units)", fontsize=12)
    ax2.set_title("Directional Power Spectra: $P_{dir}(k)$ with Fitted Slopes (k = 5.0-200.0)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Add slope information as text
    slope_info = []
    for i, lam in enumerate(lambda_list):
        regime = regime_classification[i]
        psa_slope = psa_slopes[i] if i < len(psa_slopes) else np.nan
        dir_slope = dir_slopes[i] if i < len(dir_slopes) else np.nan
        
        slope_text = f'$\\lambda = {lam:.3f}$ m: {regime}'
        if not np.isnan(psa_slope):
            slope_text += f', PSA slope = {psa_slope:.2f}'
        if not np.isnan(dir_slope):
            slope_text += f', Dir slope = {dir_slope:.2f}'
        slope_info.append(slope_text)
    
    # Add slope information as text box
    slope_text = '\n'.join(slope_info[:5])  # Show first 5 for brevity
    if len(slope_info) > 5:
        slope_text += f'\n... and {len(slope_info)-5} more'
    
    fig.text(0.02, 0.02, f'Regime Classification and Slopes (k = 5.0-200.0):\n{slope_text}', 
             fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_png:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "full_spectra_panel.png"), dpi=160, bbox_inches="tight")
        print(f"Full spectra panel with slopes saved to: {os.path.join(output_dir, 'full_spectra_panel.png')}")
    
    plt.show()
    print("Full spectra panel with slope lines completed!")

def plot_all_measures_combined(npz_path: str, output_dir: str = "lp2016_outputs", save_png: bool = True):
    """
    Create the all_measures_combined.png plot from NPZ data showing all 6 measures.
    
    This recreates the combined plot showing PSA slopes, derivative variance, 
    directional slopes, PFA variance, and derivative spectrum variance with regime backgrounds.
    PSA slopes are computed from full spectra data using the new k-range (5.0-200.0).
    
    Parameters:
    -----------
    npz_path : str
        Path to the saved NPZ file containing measures data
    output_dir : str
        Directory to save the replot image
    save_png : bool
        Whether to save the plot as PNG file
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load data
    data = np.load(npz_path)
    lambda_list = data['lambda_list']
    derivative_lambda_mid = data['derivative_lambda_mid']
    derivative_variance = data['derivative_variance']
    pfa_variances = data.get('pfa_variances', None)
    derivative_spectrum_variances = data.get('derivative_spectrum_variances', None)
    regime_classification = data['regime_classification']
    
    # Compute PSA slopes from full spectra data using k-range (5.0-200.0)
    print("Computing PSA slopes from full spectra data...")
    psa_slopes = []
    for lam in lambda_list:
        k_key = f'psa_k_lambda_{lam:.3f}'
        P_key = f'psa_P_lambda_{lam:.3f}'
        
        if k_key in data and P_key in data:
            k = data[k_key]
            P = data[P_key]
            # Fit slope using the k-range (5.0-200.0)
            m, a, k_fit, P_fit = fit_log_slope(k, P, kmin=5.0, kmax=200.0)
            psa_slopes.append(m)
        else:
            psa_slopes.append(np.nan)
    
    # Compute directional spectrum slopes from full spectra data
    print("Computing directional spectrum slopes from full spectra data...")
    directional_slopes = []
    for lam in lambda_list:
        k_key = f'dir_k_lambda_{lam:.3f}'
        P_key = f'dir_P_lambda_{lam:.3f}'
        
        if k_key in data and P_key in data:
            k = data[k_key]
            P = data[P_key]
            # Fit slope using the k-range (5.0-200.0)
            m, a, k_fit, P_fit = fit_log_slope(k, P, kmin=5.0, kmax=200.0)
            directional_slopes.append(m)
        else:
            directional_slopes.append(np.nan)
    
    psa_slopes = np.array(psa_slopes)
    directional_slopes = np.array(directional_slopes)
    
    # Create the combined plot with all 6 measures
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # PSA slopes (normalized to 0-1 range)
    slopes_norm = np.array(psa_slopes)
    slopes_norm = (slopes_norm - np.min(slopes_norm)) / (np.max(slopes_norm) - np.min(slopes_norm) + 1e-10)
    ax.plot(lambda_list, slopes_norm, 'b.-', linewidth=1.5, markersize=4, label='PSA slope')
    
    # Derivative measure (normalized to 0-1 range)
    var_dPdl2_norm = (derivative_variance - np.min(derivative_variance)) / (np.max(derivative_variance) - np.min(derivative_variance) + 1e-10)
    ax.plot(derivative_lambda_mid, var_dPdl2_norm, 'g.-', linewidth=1.5, markersize=4, label='Derivative variance')
    
    # Directional-spectrum slope (normalized)
    pdir_norm = np.array(directional_slopes, dtype=float)
    pdir_norm = (pdir_norm - np.nanmin(pdir_norm)) / (np.nanmax(pdir_norm) - np.nanmin(pdir_norm) + 1e-10)
    ax.plot(lambda_list, pdir_norm, 'm.-', linewidth=1.5, markersize=4, label='Directional spectrum slope')
    
    # PFA variance (normalized)
    if pfa_variances is not None:
        pfa_norm = (pfa_variances - np.nanmin(pfa_variances)) / (np.nanmax(pfa_variances) - np.nanmin(pfa_variances) + 1e-10)
        ax.plot(lambda_list, pfa_norm, 'r.-', linewidth=1.5, markersize=4, label='PFA variance')
    
    # Derivative spectrum variance (normalized)
    if derivative_spectrum_variances is not None:
        valid_mask = ~np.isnan(derivative_spectrum_variances)
        if np.any(valid_mask):
            deriv_spec_norm = (derivative_spectrum_variances - np.nanmin(derivative_spectrum_variances)) / (np.nanmax(derivative_spectrum_variances) - np.nanmin(derivative_spectrum_variances) + 1e-10)
            ax.plot(lambda_list, deriv_spec_norm, 'c.-', linewidth=1.5, markersize=4, label='Derivative spectrum variance')
    
    ax.set_xlabel("$\\lambda$ (m)", fontsize=12)
    ax.set_ylabel("Normalized measure value", fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Add regime background colors
    regime_colors = {'weak': 'lightblue', 'transition': 'yellow', 'strong': 'lightcoral'}
    
    # Group wavelengths by regime
    regime_wavelengths = {'weak': [], 'transition': [], 'strong': []}
    for i, lam in enumerate(lambda_list):
        regime = regime_classification[i]
        regime_wavelengths[regime].append(lam)
    
    # Fill background regions for each regime
    y_min, y_max = -0.1, 1.1
    for regime, wavelengths in regime_wavelengths.items():
        if wavelengths:
            # Find consecutive wavelength ranges for this regime
            wavelengths = sorted(wavelengths)
            start_idx = 0
            for i in range(1, len(wavelengths)):
                if wavelengths[i] - wavelengths[i-1] > 0.05:  # Gap larger than typical spacing
                    # Fill the previous range
                    if i > start_idx:
                        ax.axvspan(wavelengths[start_idx], wavelengths[i-1], 
                                  color=regime_colors[regime], alpha=0.3, zorder=0)
                        # Fill between curves and x-axis
                        ax.fill_between([wavelengths[start_idx], wavelengths[i-1]], 
                                      y_min, y_max, 
                                      color=regime_colors[regime], alpha=0.2, zorder=0)
                    start_idx = i
            # Fill the last range
            if start_idx < len(wavelengths):
                ax.axvspan(wavelengths[start_idx], wavelengths[-1], 
                          color=regime_colors[regime], alpha=0.3, zorder=0)
                # Fill between curves and x-axis
                ax.fill_between([wavelengths[start_idx], wavelengths[-1]], 
                              y_min, y_max, 
                              color=regime_colors[regime], alpha=0.2, zorder=0)
    
    # Add regime legend
    from matplotlib.patches import Patch
    regime_patches = [
        Patch(facecolor='lightblue',  alpha=0.25, label=r'$2\lambda^{2}\sigma_{\Phi} < 1$'),
        Patch(facecolor='yellow',     alpha=0.25, label=r'$1 < 2\lambda^{2}\sigma_{\Phi} < 3$'),
        Patch(facecolor='lightcoral', alpha=0.25, label=r'$2\lambda^{2}\sigma_{\Phi} > 3$')
    ]
    # Primary curve legend + equation-based patches
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + regime_patches, labels + [p.get_label() for p in regime_patches],
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_png:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "all_measures_combined.png"), dpi=160, bbox_inches="tight")
        print(f"All 6 measures combined plot saved to: {os.path.join(output_dir, 'all_measures_combined.png')}")
    
    plt.show()
    print("All 6 measures combined plot completed!")

def main():
    """Example usage of the replot function."""
    # Example usage
    npz_file = "lp2016_outputs/measures_data.npz"
    output_dir = "lp2016_outputs"
    
    if os.path.exists(npz_file):
        print("Creating measures replot...")
        load_and_replot(npz_file, output_dir)
        
        print("\nCreating all measures combined plot...")
        plot_all_measures_combined(npz_file, output_dir)
        
        print("\nCreating full spectra panel...")
        plot_full_spectra_panel(npz_file, output_dir)
    else:
        print(f"NPZ file not found: {npz_file}")
        print("Please run the main simulation first to generate the data file.")

if __name__ == "__main__":
    main()

