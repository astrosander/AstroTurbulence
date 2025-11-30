"""
Recreate Fig. 12 from saved spectra data.

This script loads the spectra data saved by fig12.py and recreates
the figure with the same design and styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.labelsize": 16,           # axis label text
    "xtick.labelsize": 16,          # x-tick labels
    "ytick.labelsize": 16,          # y-tick labels
})

def load_spectra_data(npz_path):
    """
    Load spectra data from npz file and reconstruct the spectra.
    
    Parameters
    ----------
    npz_path : str
        Path to the spectra_data_seed*.npz file.
    
    Returns
    -------
    all_panel_spectra : list
        List of panels, each containing list of (lam, k, E) tuples.
    metadata : dict
        Dictionary with metadata (lam_min, lam_max, etc.)
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract metadata
    metadata = {
        "lam_min": float(data["lam_min"]),
        "lam_max": float(data["lam_max"]),
        "dloglam": float(data["dloglam"]),
        "n": int(data["n"]),
        "seed": int(data["seed"]),
        "C_phi": float(data["C_phi"]),
        "kfit_min": float(data["kfit_min"]),
        "kfit_max": float(data["kfit_max"]),
        "n_panels": int(data["n_panels"]),
    }
    
    # Reconstruct spectra for each panel
    all_panel_spectra = []
    n_panels = metadata["n_panels"]
    
    for panel_idx in range(n_panels):
        # Load lambda values
        lam_arr = data[f"panel{panel_idx}_lambda"]
        
        # Load flattened arrays and lengths
        k_flat = data[f"panel{panel_idx}_k_flat"]
        E_flat = data[f"panel{panel_idx}_E_flat"]
        k_lengths = data[f"panel{panel_idx}_k_lengths"]
        E_lengths = data[f"panel{panel_idx}_E_lengths"]
        
        # Reconstruct individual spectra
        panel_spectra = []
        k_start = 0
        E_start = 0
        
        for i, lam in enumerate(lam_arr):
            k_len = int(k_lengths[i])
            E_len = int(E_lengths[i])
            
            k = k_flat[k_start:k_start + k_len]
            E = E_flat[E_start:E_start + E_len]
            
            panel_spectra.append((lam, k, E))
            
            k_start += k_len
            E_start += E_len
        
        all_panel_spectra.append(panel_spectra)
    
    return all_panel_spectra, metadata


def fit_loglog_slope(k, E, kmin, kmax):
    """
    Fit a power-law slope to log10 E vs log10 k between
    (kmin, kmax).
    
    Returns slope (d logE / d logk).
    """
    k = np.asarray(k)
    E = np.asarray(E)
    mask = (k >= kmin) & (k <= kmax) & (E > 0)
    if mask.sum() < 2:
        return np.nan
    x = np.log10(k[mask])
    y = np.log10(E[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return slope


def recreate_fig12(npz_path, output_path=None, dpi=150):
    """
    Recreate Fig. 12 from saved spectra data.
    
    Parameters
    ----------
    npz_path : str
        Path to the spectra_data_seed*.npz file.
    output_path : str or None
        Path to save the figure. If None, saves next to npz file.
    dpi : int
        Resolution for saved figure.
    """
    # Load data
    print(f"Loading spectra data from {npz_path}...")
    all_panel_spectra, metadata = load_spectra_data(npz_path)
    
    lam_min = metadata["lam_min"]
    lam_max = metadata["lam_max"]
    kfit_min = metadata["kfit_min"]
    kfit_max = metadata["kfit_max"]
    
    # Panel titles (same as in original)
    panel_titles = [
        r"Emission: $\beta_B=\beta_{n_e}=4$, Faraday: $\beta_B=\beta_{n_e}=7/2$",
        r"Emission: $\beta_B=\beta_{n_e}=7/2$, Faraday: $\beta_B=\beta_{n_e}=4$",
    ]
    
    # Create figure with same layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Modern color scheme
    # Use a modern colormap with better color separation (turbo: cycles through distinct colors)
    try:
        # For newer matplotlib versions
        cmap = mpl.colormaps['turbo']
    except (AttributeError, KeyError):
        # For older matplotlib versions or if turbo not available
        try:
            cmap = plt.cm.get_cmap('turbo')
        except ValueError:
            # Fallback to gist_rainbow if turbo not available
            cmap = plt.cm.get_cmap('gist_rainbow')
    
    # Create logarithmic normalization for full lambda range (for colorbar)
    norm_full = mpl.colors.LogNorm(lam_min, lam_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_full)
    sm.set_array([])
    
    for ax, panel_spectra, title in zip(axes, all_panel_spectra, panel_titles):
        # Get lambda range for colormap normalization (use logarithmic)
        norm = mpl.colors.LogNorm(lam_min, lam_max)
        
        # Plot all spectra with modern color scheme
        for lam, k, E in panel_spectra:
            if np.any(E <= 0):
                continue
            if np.isclose(lam, lam_min):
                # Blue for minimum wavelength
                ax.loglog(k, E, color="#2E86AB", lw=2.5, label=r"$\lambda_{\rm min}$", zorder=10)
            elif np.isclose(lam, lam_max):
                # Red-orange for maximum wavelength
                ax.loglog(k, E, color="#F24236", lw=2.5, ls="--", label=r"$\lambda_{\rm max}$", zorder=10)
            else:
                # Use colormap for intermediate spectra
                color = cmap(norm(lam))
                ax.loglog(k, E, color=color, lw=1.0, alpha=0.6)
        
        ax.set_xlabel(r"$k$", fontsize=18)
        ax.set_title(title, fontsize=16)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.set_xlim(np.min(k), 256)
        ax.tick_params(labelsize=14)
        
        # Overplot reference slopes (same as original)
        k0 = 10.0
        lam0, k0_arr, E0_arr = panel_spectra[0]
        j0 = np.argmin(np.abs(k0_arr - k0))
        E0 = E0_arr[j0]
        
        # Reference slopes: k^{-3}, k^{-5/2}, k^{+1}
        # Use distinct modern colors for each reference slope
        slope_colors = {
            -3.0: "#FF6B6B",   # Coral red - distinct from viridis
            -2.5: "#2C3E50",   # Dark blue-gray - highly visible
            1.0: "#E74C3C"     # Bright red - distinct and highly visible
        }
        for slope_ref, ls, label in [(-3.0, ":",  r"$k^{-3}$"),
                                     (-2.5, "--", r"$k^{-5/2}$"),
                                     ( 1.0, "-.", r"$k^{1}$")]:
            # line: E = E0 * (k/k0)^{slope_ref}
            E_ref = E0 * (k0_arr / k0) ** slope_ref
            # For k^1 line: move down and plot only from 40% to 80% of range
            if slope_ref == 1.0:
                E_ref = E_ref / 50000000.0
                # Get indices for 40% to 80% of the range
                n = len(k0_arr)
                start_idx = int(0.05 * n)
                end_idx = int(0.4 * n)
                k_plot = k0_arr[start_idx:end_idx]
                E_plot = E_ref[start_idx:end_idx]
            else:
                k_plot = k0_arr
                E_plot = E_ref
            ax.loglog(k_plot, E_plot, ls=ls, color=slope_colors[slope_ref], 
                     alpha=0.9, lw=2.0, label=label, zorder=15)
        
        # Optional: compute and print slopes for verification
        slopes = []
        for lam, k, E in panel_spectra:
            slope = fit_loglog_slope(k, E, kfit_min, kfit_max)
            slopes.append((lam, slope))
        
        print(f"  Panel: {title}")
        print(f"  Measured slopes in k=[{kfit_min},{kfit_max}] for a few λ:")
        for lam, slope in slopes[::max(1, len(slopes)//5)]:
            print(f"    λ={lam:5.2f}  slope≈{slope:6.3f}")
    
    axes[0].set_ylabel(r"$E_{2D}(k; dP/d\lambda^2)$", fontsize=18)
    axes[0].legend(fontsize=12, loc="lower left")
    
    plt.tight_layout()
    
    # Add colorbar for lambda values outside the figure
    # Adjust subplots to make minimal room for colorbar
    plt.subplots_adjust(right=0.94)
    # Create colorbar with minimal width (logarithmic color mapping, linear tick labels)
    cbar = plt.colorbar(sm, ax=axes, pad=0.01, fraction=0.015)
    cbar.set_label(r"$\lambda$", fontsize=14, rotation=0, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    # Override log formatter to show actual lambda values (not log format)
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # Adjust colorbar axes position to remove extra space
    cbar.ax.set_position([0.95, cbar.ax.get_position().y0, 0.02, cbar.ax.get_position().height])
    
    # Save figure
    if output_path is None:
        base_name = os.path.splitext(npz_path)[0]
        output_path = f"{base_name}_recreated.png"
    
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved recreated figure to {output_path}")
    
    return fig, axes


if __name__ == "__main__":
    # Default path
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        # Look for spectra_data_seed1.npz in current directory
        npz_path = "spectra_data_seed1.npz"
        if not os.path.exists(npz_path):
            # Try in parent directory
            npz_path = os.path.join("..", "spectra_data_seed1.npz")
    
    if not os.path.exists(npz_path):
        print(f"Error: Could not find {npz_path}")
        print("Usage: python recreate_fig12.py [path_to_spectra_data_seed*.npz]")
        sys.exit(1)
    
    fig, axes = recreate_fig12(npz_path)
    
    plt.show()

