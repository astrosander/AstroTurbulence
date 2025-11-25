import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set up matplotlib style
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

def plot_slopes_vs_lambda(csv_path=None, save_path="slopes_vs_lambda.png", show_plot=True, 
                          plot_slopes=['k_lt_K_i', 'K_i_lt_k_lt_K_phi', 'k_gt_K_phi']):
    """Plot slopes as a function of chi from the CSV file.
    
    chi = 2 * sigma_RM * lambda^2
    
    Parameters:
    -----------
    csv_path : str or Path, optional
        Path to the CSV file. Defaults to 'slopes_vs_lambda.csv' in script directory.
    save_path : str or Path, optional
        Path to save the plot. If None, plot is not saved.
    show_plot : bool, default True
        Whether to display the plot.
    plot_slopes : list, default ['k_lt_K_i', 'K_i_lt_k_lt_K_phi', 'k_gt_K_phi']
        Which slopes to plot. Options: 'k_lt_K_i', 'K_i_lt_k_lt_K_phi', 'k_gt_K_phi'
    """
    
    if csv_path is None:
        script_dir = Path(__file__).parent
        csv_path = script_dir / "slopes_vs_lambda.csv"
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Sort by chi
    df = df.sort_values('chi')
    
    # Extract data - handle each slope separately to filter out NaN values
    chi_vals_all = df['chi'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot selected slopes - filter out NaN values for each slope individually
    slope_configs = {
        'k_lt_K_i': ('slope_k_lt_K_i', 'o-', '#7F8C8D', r'$k < K_i$'),
        'K_i_lt_k_lt_K_phi': ('slope_K_i_lt_k_lt_K_phi', 's-', '#E67E22', r'$K_i < k < K_\phi$'),
        'k_gt_K_phi': ('slope_k_gt_K_phi', '^-', '#E74C3C', r'$k > K_\phi$')
    }
    
    for slope_key in plot_slopes:
        if slope_key in slope_configs:
            col_name, marker_style, color, label = slope_configs[slope_key]
            # Filter out rows where this specific slope is NaN/empty
            mask = df[col_name].notna()
            if mask.sum() > 0:  # Only plot if there's at least one valid data point
                chi_vals = df.loc[mask, 'chi'].values
                data = df.loc[mask, col_name].values
                ax.plot(chi_vals, data, color=color, lw=2.5, 
                       markersize=6, label=label)
    
    ax.set_xlabel(r'$\chi = 2\sigma_\Phi\lambda^2$', fontsize=22)
    ax.set_ylabel('Slope', fontsize=22)
    ax.set_title('Slopes vs $\chi$', fontsize=24, pad=15)
    ax.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9, loc='best')
    plt.xlim(0, 10)
    plt.ylim(-5,1)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

if __name__ == "__main__":
    # Plot slopes from CSV
    plot_slopes_vs_lambda(show_plot=True)

