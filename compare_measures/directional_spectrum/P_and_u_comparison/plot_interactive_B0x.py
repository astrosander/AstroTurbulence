import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os
import re

# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 24

# Compute d(log P) / d(log k) function (copied from plot_from_points.py)
def compute_dlogP_dlogk(k, Pk, window_fraction=0.15, min_points=5, max_points=11):
    """
    Compute d(log P) / d(log k) using local power-law fitting with adaptive windowing.
    """
    # Filter out invalid values
    mask = (k > 0) & (Pk > 0)
    k_valid = k[mask]
    Pk_valid = Pk[mask]
    
    if len(k_valid) < min_points:
        return np.array([]), np.array([])
    
    log_k = np.log(k_valid)
    log_Pk = np.log(Pk_valid)
    
    # Determine window size in log space
    log_k_range = log_k.max() - log_k.min()
    window_size_log = window_fraction * log_k_range
    
    n_points = len(k_valid)
    dlogP_dlogk = np.zeros(n_points)
    k_centers = np.zeros(n_points)
    
    for i in range(n_points):
        # Find points within window_size_log in log space
        log_k_center = log_k[i]
        log_k_low = log_k_center - window_size_log / 2
        log_k_high = log_k_center + window_size_log / 2
        
        # Find indices within this range
        in_window = (log_k >= log_k_low) & (log_k <= log_k_high)
        window_indices = np.where(in_window)[0]
        
        # Ensure we have enough points
        if len(window_indices) < min_points:
            # Expand window if needed
            window_indices = np.arange(max(0, i - max_points//2), 
                                      min(n_points, i + max_points//2 + 1))
        elif len(window_indices) > max_points:
            # Limit to max_points closest to center
            distances = np.abs(log_k[window_indices] - log_k_center)
            sorted_idx = np.argsort(distances)
            window_indices = window_indices[sorted_idx[:max_points]]
        
        # Extract window data
        k_window = log_k[window_indices]
        Pk_window = log_Pk[window_indices]
        
        # Fit linear (power law in log space): log P = alpha * log k + const
        if len(k_window) >= 2:
            # Use Gaussian weights centered on current point
            distances = np.abs(k_window - log_k_center)
            sigma = window_size_log / 3.0  # 3-sigma covers most of window
            weights = np.exp(-0.5 * (distances / sigma)**2)
            weights = weights / (weights.sum() + 1e-30)
            
            # Weighted linear fit
            k_mean = np.average(k_window, weights=weights)
            Pk_mean = np.average(Pk_window, weights=weights)
            
            numerator = np.sum(weights * (k_window - k_mean) * (Pk_window - Pk_mean))
            denominator = np.sum(weights * (k_window - k_mean)**2)
            
            if abs(denominator) > 1e-10:
                alpha = numerator / denominator
            else:
                # Fallback to simple linear fit
                alpha = np.polyfit(k_window, Pk_window, 1)[0]
            
            dlogP_dlogk[i] = alpha
            k_centers[i] = k_valid[i]
        else:
            dlogP_dlogk[i] = np.nan
            k_centers[i] = k_valid[i]
    
    # Remove NaN values
    valid_mask = ~np.isnan(dlogP_dlogk)
    return k_centers[valid_mask], dlogP_dlogk[valid_mask]

def load_spectrum_files(spectrum_dir='spectrum'):
    """
    Load all B0x spectrum files and return sorted data.
    Returns: (B0x_values, data_dict) where data_dict[B0x] contains the npz data
    """
    files = glob.glob(os.path.join(spectrum_dir, 'B0x_*.npz'))
    
    B0x_values = []
    data_dict = {}
    
    for file in files:
        # Extract B0x value from filename
        match = re.search(r'B0x_([0-9.]+)\.npz', file)
        if match:
            B0x = float(match.group(1))
            B0x_values.append(B0x)
            data_dict[B0x] = np.load(file)
    
    # Sort by B0x value
    B0x_values = sorted(B0x_values)
    
    print(f"Loaded {len(B0x_values)} spectrum files")
    print(f"B0x range: {B0x_values[0]:.6f} to {B0x_values[-1]:.6f}")
    
    return B0x_values, data_dict

def plot_spectrum_data(ax, data, B0x_value):
    """Plot spectrum and derivative for given data"""
    # Extract data
    kP = data['kP']
    Pk_amp = data['Pk_amp']
    ku = data['ku']
    Pk_u = data['Pk_u']
    
    # Clear axes
    ax.clear()
    
    # Plot power spectra
    ax.loglog(kP, Pk_amp, 'o', color='blue', label=r'$P(k)$ for $|P_i|$', markersize=4, linewidth=1.5, alpha=0.7)
    ax.loglog(ku, Pk_u, 's', color='red', label=r'$P(k)$ for $u_i$', markersize=4, linewidth=1.5, alpha=0.7)
    
    # Add reference line at -11/3
    k_all = np.concatenate([kP, ku])
    k_min = np.min(k_all[k_all > 0])
    k_max = np.max(k_all)
    k_ref = np.geomspace(k_min, k_max, 100)
    anchor_k = np.sqrt(k_min * k_max)
    
    # Reference line for P_i
    anchor_idx_P = np.argmin(np.abs(kP - anchor_k))
    if anchor_idx_P >= len(Pk_amp):
        anchor_idx_P = len(Pk_amp) - 1
    C_P = Pk_amp[anchor_idx_P] * (kP[anchor_idx_P] ** (11.0/3.0))
    Pk_ref_P = C_P * k_ref ** (-11.0/3.0)
    ax.loglog(k_ref, Pk_ref_P, '--', color='blue', linewidth=1.5, label=r'$k^{-11/3}$ ($|P_i|$)', alpha=0.7)

    # C_P = Pk_amp[anchor_idx_P] * (kP[anchor_idx_P] ** (3.5))
    # Pk_ref_P = C_P * k_ref ** (-3.5)
    # ax.loglog(k_ref, Pk_ref_P, '--', color='blue', linewidth=1.5, label=r'$k^{-7/2}$ ($|P_i|$)', alpha=0.7)
    
    # Reference line for u_i
    anchor_idx_u = np.argmin(np.abs(ku - anchor_k))
    if anchor_idx_u >= len(Pk_u):
        anchor_idx_u = len(Pk_u) - 1
    C_u = Pk_u[anchor_idx_u] * (ku[anchor_idx_u] ** (11.0/3.0))
    Pk_ref_u = C_u * k_ref ** (-11.0/3.0)
    ax.loglog(k_ref, Pk_ref_u, '--', color='red', linewidth=1.5, label=r'$k^{-11/3}$ ($u_i$)', alpha=0.7)
    
    # C_u = Pk_u[anchor_idx_u] * (ku[anchor_idx_u] ** (7.0/2.0))
    # Pk_ref_u = C_u * k_ref ** (-7.0/2.0)
    # ax.loglog(k_ref, Pk_ref_u, '--', color='red', linewidth=1.5, label=r'$k^{-7/2}$ ($u_i$)', alpha=0.7)
    
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P(k)$')
    ax.set_title(rf'Power Spectra: B0x = {B0x_value:.6f}', fontsize=20)
    ax.legend(fontsize=18)
    k_data_all = np.concatenate([kP[kP > 0], ku[ku > 0]])
    Pk_data_all = np.concatenate([Pk_amp[Pk_amp > 0], Pk_u[Pk_u > 0]])
    ax.set_xlim(np.min(k_data_all), np.max(k_data_all))
    ax.set_ylim(np.min(Pk_data_all), np.max(Pk_data_all))
    ax.grid(True, alpha=0.3, which='both')

def plot_derivative_data(ax, data, B0x_value):
    """Plot derivative for given data"""
    # Extract data
    kP = data['kP']
    Pk_amp = data['Pk_amp']
    ku = data['ku']
    Pk_u = data['Pk_u']
    
    # Clear axes
    ax.clear()
    
    # Compute derivatives
    kP_deriv, dlogP_dlogk_P = compute_dlogP_dlogk(kP, Pk_amp, window_fraction=0.15, min_points=5, max_points=11)
    ku_deriv, dlogP_dlogk_u = compute_dlogP_dlogk(ku, Pk_u, window_fraction=0.15, min_points=5, max_points=11)
    
    # Plot derivatives
    if len(kP_deriv) > 0:
        ax.semilogx(kP_deriv, dlogP_dlogk_P, 'o', color='blue', label=r'$\frac{d\log P}{d\log k}$ for $|P_i|$', markersize=4, linewidth=1.5, alpha=0.7)
    if len(ku_deriv) > 0:
        ax.semilogx(ku_deriv, dlogP_dlogk_u, 's', color='red', label=r'$\frac{d\log P}{d\log k}$ for $u_i$', markersize=4, linewidth=1.5, alpha=0.7)
    
    # Add reference line at -11/3
    k_all_deriv = np.concatenate([kP_deriv, ku_deriv]) if len(kP_deriv) > 0 and len(ku_deriv) > 0 else (kP_deriv if len(kP_deriv) > 0 else ku_deriv)
    if len(k_all_deriv) > 0:
        ax.axhline(-11.0/3.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$-11/3$ reference')
        # ax.axhline(-7/2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$-7/2$ reference')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\frac{d\log P}{d\log k}$')
    ax.set_title(rf'Derivative: B0x = {B0x_value:.6f}', fontsize=20)
    ax.legend(fontsize=18)
    if len(k_all_deriv) > 0:
        ax.set_xlim(np.min(k_all_deriv), np.max(k_all_deriv))
        ax.set_ylim(-4, -3)
    ax.grid(True, alpha=0.3, which='both')

def update_plots(val):
    """Update plots when slider changes"""
    idx = int(slider.val)
    B0x = B0x_values[idx]
    data = data_dict[B0x]
    
    plot_spectrum_data(ax1, data, B0x)
    plot_derivative_data(ax2, data, B0x)
    
    fig.canvas.draw_idle()

if __name__ == "__main__":
    # Load all spectrum files
    B0x_values, data_dict = load_spectrum_files('spectrum_vtk(hu)')
    
    if len(B0x_values) == 0:
        print("No spectrum files found!")
        exit(1)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    # Initial plot (first B0x value)
    initial_idx = 0
    initial_B0x = B0x_values[initial_idx]
    initial_data = data_dict[initial_B0x]
    
    plot_spectrum_data(ax1, initial_data, initial_B0x)
    plot_derivative_data(ax2, initial_data, initial_B0x)
    
    # Create slider
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    
    # Custom formatter function
    def format_slider(val):
        idx = int(val)
        return f'{idx} (B0x={B0x_values[idx]:.6f})'
    
    slider = Slider(
        ax_slider,
        'B0x Index',
        0,
        len(B0x_values) - 1,
        valinit=initial_idx,
        valstep=1
    )
    
    # Set initial slider text
    slider.valtext.set_text(format_slider(initial_idx))
    
    # Update slider label to show current B0x value
    def update_slider_label(val):
        idx = int(val)
        slider.valtext.set_text(format_slider(val))
    
    slider.on_changed(update_plots)
    slider.on_changed(update_slider_label)
    
    # Add text showing current B0x value
    text_ax = fig.text(0.5, 0.08, f'Current B0x: {initial_B0x:.6f}', 
                       ha='center', fontsize=16, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update_text(val):
        idx = int(val)
        text_ax.set_text(f'Current B0x: {B0x_values[idx]:.6f}')
    
    slider.on_changed(update_text)
    
    # Add keyboard controls
    def on_key(event):
        if event.key == 'right' or event.key == 'up':
            if slider.val < len(B0x_values) - 1:
                slider.set_val(slider.val + 1)
        elif event.key == 'left' or event.key == 'down':
            if slider.val > 0:
                slider.set_val(slider.val - 1)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\nControls:")
    print("  Slider: Move between B0x values")
    print("  Left/Right or Up/Down arrows: Navigate between values")
    print("  Close window to exit")
    
    plt.show()

