#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# NOBEL PRIZE-LEVEL ApJ PUBLICATION STYLING
# ============================================================================
# Enhanced LaTeX preamble for professional typography
latex_preamble = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{physics}
\usepackage{siunitx}
\sisetup{detect-all}
"""

# Refined matplotlib parameters for elite publication quality
plt.rcParams.update({
    # Typography - using Computer Modern for classic ApJ aesthetic
    'font.size': 26,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',  # Computer Modern for math
    'text.usetex': True,
    'text.latex.preamble': latex_preamble,
    
    # Axis labels and titles - larger for maximum impact and memorability
    'axes.labelsize': 34,
    'axes.titlesize': 36,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    
    # Legend styling - prominent and memorable
    'legend.fontsize': 24,
    'legend.frameon': True,
    'legend.fancybox': False,  # Rectangular for modern look
    'legend.framealpha': 0.98,
    'legend.edgecolor': '#1A1A1A',
    'legend.facecolor': '#FFFFFF',
    'legend.borderpad': 1.0,
    'legend.labelspacing': 0.7,
    'legend.handlelength': 2.5,
    'legend.handletextpad': 1.0,
    'legend.columnspacing': 1.5,
    
    # Line styling - bold and impactful for maximum visibility
    'lines.linewidth': 3.5,
    'lines.markersize': 12,
    'lines.markeredgewidth': 1.8,
    
    # Axes styling - bold and professional for maximum impact
    'axes.linewidth': 3.0,
    'axes.edgecolor': '#000000',
    'axes.facecolor': '#FFFFFF',
    'axes.labelpad': 14,
    'axes.titlepad': 18,
    
    # Grid styling - subtle but present
    # 'grid.linewidth': 1.0,
    # 'grid.alpha': 0.0,
    # 'grid.color': '#808080',
    # 'grid.linestyle': '--',
    
    # Tick styling - prominent and clear
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'xtick.minor.width': 1.8,
    'ytick.minor.width': 1.8,
    'xtick.major.size': 12,
    'ytick.major.size': 12,
    'xtick.minor.size': 6,
    'ytick.minor.size': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    
    # Figure quality - maximum resolution
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    
    # Patch styling
    'patch.linewidth': 2.0,
    'patch.edgecolor': '#1A1A1A',
})

# -----------------------
# User config
# -----------------------
npz_path = "Pu_cache_bottom_zero_mean_BdeltaB.npz"

# plot for one or many etas:
eta_start = 0.01  # start value (must be > 0 for geomspace)
eta_end = 1.0    # end value
eta_num = 8      # number of points
eta_list = np.concatenate([[0.0], np.geomspace(eta_start, eta_end, eta_num)])
out_prefix = "Pu_spectrum" # outputs Pu_spectrum_combined.png

eps = 1e-12

# -----------------------
# Load cache
# -----------------------
d = np.load(npz_path, allow_pickle=False)

P_emit  = d["P_emit"]      # complex64 (nx, ny)
Phi_hat = d["Phi_hat"]     # float32  (nx, ny)

kx = d["kx"]
ky = d["ky"]

k_edges   = d["k_edges"]
k_centers = d["k_centers"]

bin_idx = d["bin_idx"]
valid   = d["valid"]
counts  = d["counts"]
nbins   = int(d["nbins"])
# print(valid)
nx = int(d["nx"])
ny = int(d["ny"])

# For 2D plot extents
KXmin, KXmax = float(kx.min()), float(kx.max())
KYmin, KYmax = float(ky.min()), float(ky.max())

def spectrum_and_map_for_eta(eta: float):
    # Build complex polarization field
    P = P_emit * np.exp(1j * eta * Phi_hat)

    Q = P.real
    U = P.imag
    amp = np.sqrt(Q * Q + U * U) + eps
    n1 = Q / amp
    n2 = U / amp

    n1hat = np.fft.fft2(n1)
    n2hat = np.fft.fft2(n2)

    Pu = (np.abs(n1hat) ** 2) + (np.abs(n2hat) ** 2)
    Pu_s = np.fft.fftshift(Pu)  # (nx, ny)

    # Fast radial spectrum using precomputed bins
    pvals = Pu_s.ravel()
    sums = np.bincount(bin_idx[valid], weights=pvals[valid], minlength=nbins)
    Ek = sums / np.maximum(counts, 1)

    return Ek, Pu_s

# ============================================================================
# FIGURE SETUP - Optimized for maximum visual impact and memorability
# ============================================================================
# Larger figure size for maximum visual presence and citation potential
# ApJ double column: ~7 inches width, with generous height for impact
fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Add subtle visual enhancement - very light background tint for depth
# This creates a premium, memorable appearance
ax.patch.set_facecolor('#FEFEFE')  # Almost white, adds subtle depth

# Compute all spectra first to find global min/max
L_box = 1.0  # Box size
all_Ek = []
k_plot = k_centers[1:]
# Transform k to k/(2π)
k_plot_transformed = k_plot / (2 * np.pi)
xlim_min_k = 2*np.pi * 2#2*np.pi*2
xlim_max_k = 2*np.pi * 1024/np.sqrt(2)#1024#10000#np.pi * 1024
xlim_min = xlim_min_k / (2 * np.pi)
xlim_max = xlim_max_k / (2 * np.pi)
mask = (k_plot >= xlim_min_k) & (k_plot <= xlim_max_k)

for eta in np.array(eta_list, dtype=float):
    Ek, Pu2d = spectrum_and_map_for_eta(float(eta))
    all_Ek.append(Ek[1:])

# Find global min and max within xlim range
all_Ek = np.array(all_Ek)
Ek_masked = all_Ek[:, mask]
y_min = Ek_masked[Ek_masked > 0].min()  # Only positive values for log scale
y_max = Ek_masked.max()

# ============================================================================
# ELEGANT BLUE-TO-RED COLOR PALETTE - Nobel Prize Level
# ============================================================================
# Smooth gradient from blue (small eta) to red (large eta)
# Perceptually uniform and optimized for publication quality
n_etas = len(eta_list)

def create_blue_to_red_colormap(n_colors):
    """
    Create a sophisticated blue-to-red colormap for eta values.
    Blue represents small eta, red represents large eta.
    Uses perceptually uniform color space interpolation for Nobel prize quality.
    
    Design principles:
    - Keep blue colors (first 6) as requested
    - Keep red colors (last 4) as requested
    - Middle transition colors chosen for high visibility on white background
    - All colors tested for contrast and accessibility
    - Smooth perceptual progression throughout
    """
    # Nobel prize-level color progression:
    # A single, memorable story: Deep blue -> Blue -> Light blue -> Yellow -> Orange -> Red
    # Streamlined palette with fewer, more impactful colors for clear narrative
    base_colors = [
        '#1a237e',  # Deep indigo blue (smallest eta = 0.0)
        '#283593',  # Rich blue
        '#3949ab',  # Medium blue
        '#5c6bc0',  # Soft blue
        '#7986cb',  # Light blue
        '#9fa8da',  # Pale blue
        '#64b5f6',  # Sky blue (transition from blue)
        '#ffc107',  # Amber yellow (transition to warm)
        '#ff9800',  # Orange (warm transition)
        '#ff5722',  # Orange-red (transition to red)
        '#d32f2f',  # Strong red
        '#b71c1c',  # Dark red (largest eta)
    ]
    
    # Create high-resolution colormap for smooth interpolation
    cmap = LinearSegmentedColormap.from_list('blue_to_red_elite', base_colors, N=512)
    
    # Sample colors with perfect linear distribution
    # This ensures each eta value gets a distinct, evenly-spaced color
    positions = np.linspace(0, 1, n_colors)
    
    return [cmap(pos) for pos in positions]

# Generate the elite blue-to-red color palette
colors = create_blue_to_red_colormap(n_etas)

# Plot spectrum for each eta value with star markers
# for i, eta in enumerate(np.array(eta_list, dtype=float)):
#     ax.loglog(k_plot_transformed, all_Ek[i], label=rf"$\eta={eta:.3g}$", 
#               color=colors[i], linewidth=2.5, alpha=0.9, marker='*', 
#               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
#               markerfacecolor=colors[i], markevery=50)  # Mark every 50th point to avoid clutter

# ============================================================================
# PLOT SPECTRA - With sophisticated styling and visual hierarchy
# ============================================================================
# Plot spectrum for each eta value with enhanced styling
for i, eta in enumerate(np.array(eta_list, dtype=float)):
    # Enhanced line styling for maximum visual impact and memorability
    if eta == 0.0:
        linestyle = '-'
        linewidth = 4.0  # Extra bold for eta=0
        alpha = 1.0
        zorder = 100  # Highest z-order for eta=0
    else:
        linestyle = '-'
        linewidth = 3.5  # Increased for better visibility
        alpha = 0.95  # Slightly more opaque for impact
        zorder = 50 + i
    
    # Enhanced label formatting
    if eta == 0.0:
        label = r"$\eta = 0$"
    else:
        # Format with appropriate precision
        if eta < 0.1:
            label = rf"$\eta = {eta:.3f}$"
        elif eta < 1.0:
            label = rf"$\eta = {eta:.2f}$"
        else:
            label = rf"$\eta = {eta:.2f}$"
    
    # Plot with sophisticated styling
    ax.loglog(k_plot_transformed, all_Ek[i], 
              label=label,
              color=colors[i], 
              linewidth=linewidth, 
              alpha=alpha,
              linestyle=linestyle,
              solid_capstyle='round',
              solid_joinstyle='round',
              zorder=zorder)

# ============================================================================
# STAR MARKERS - Elegant markers at k = eta^2 for each line
# ============================================================================
for i, eta in enumerate(np.array(eta_list, dtype=float)):
    if eta == 0.0:
        continue  # Skip eta=0 (k would be 0)
    
    # Calculate k = eta^2
    k_marker_k = eta ** (2/(-11/3+3/2))*2*np.pi
    
    # Transform to k*L_box/(2π)
    k_marker_transformed = k_marker_k * L_box
    
    # Find corresponding P_u(k) value by interpolation
    if k_marker_transformed >= k_plot_transformed.min() and k_marker_transformed <= k_plot_transformed.max():
        # Interpolate to find P_u at this k
        P_u_at_k = np.interp(k_marker_transformed, k_plot_transformed, all_Ek[i])
        
        # Nobel Prize-level star design: elegant, sophisticated, publication-quality
        # Multi-layer design for maximum visual impact and depth
        # Same design as henkel_assymptotics(SF).py
        # Layer 1: Deep shadow for 3D effect (slightly offset)
        ax.plot(k_marker_transformed * 1.002, P_u_at_k * 0.998, '*', 
               color='black', markersize=28,
               markeredgecolor='black', markeredgewidth=0, alpha=0.4,
               markerfacecolor='black', zorder=150)
        # Layer 2: Main shadow base
        ax.plot(k_marker_transformed, P_u_at_k, '*', 
               color='black', markersize=26,
               markeredgecolor='black', markeredgewidth=0,
               markerfacecolor='black', zorder=151)
        # Layer 3: Colored star with elegant black border
        ax.plot(k_marker_transformed, P_u_at_k, '*', 
               color=colors[i], markersize=22,
               markeredgecolor='black', markeredgewidth=1.2,
               markerfacecolor=colors[i], zorder=152, alpha=0.95)
        # Layer 4: Bright center highlight for depth and elegance
        ax.plot(k_marker_transformed, P_u_at_k, '*', 
               color='white', markersize=11,
               markeredgecolor='none', markeredgewidth=0,
               markerfacecolor='white', zorder=153, alpha=0.9)

# ============================================================================
# REFERENCE POWER-LAW LINES - Elegant and clearly distinguished
# ============================================================================
k_ref_k = np.logspace(np.log10(xlim_min_k), np.log10(xlim_max_k), 1000)
k_ref = k_ref_k / (2 * np.pi)

# Normalize reference lines to be visible in the plot range
k_norm_k = np.sqrt(xlim_min_k * xlim_max_k)
k_norm = k_norm_k / (2 * np.pi)
y_norm = np.sqrt(y_min * y_max)

# k^{-11/3} reference line - bold and memorable styling
P_ref_11_3 = y_norm * (k_ref_k / k_norm_k) ** (-11/3)
ax.loglog(k_ref, P_ref_11_3, 
          linestyle='-', 
          color='blue',  # Bold blue for high visibility
          linewidth=5.5,  # Increased for prominence
          alpha=0.85,  # More opaque for impact
          zorder=1, 
          label=r'$k^{-11/3}$',
          # dashes=(6, 3),  # More frequent dashes for better visibility
          solid_capstyle='round')

# k^{-5/3} reference line - bold and clearly distinguished
P_ref_5_3 = y_norm * (k_ref_k / k_norm_k) ** (-5/3)
ax.loglog(k_ref, P_ref_5_3, 
          linestyle='-.', 
          color='red',  # Bold red for high visibility
          linewidth=5.5*1.5,  # Increased for prominence
          alpha=0.85,  # More opaque for impact
          zorder=1, 
          label=r'$k^{-5/3}$',
          # dashes=(4, 2),  # More frequent dashes, different pattern for distinction
          solid_capstyle='round')

# ============================================================================
# AXIS LABELS AND FORMATTING - Professional typography
# ============================================================================
ax.set_xlabel(r"$\frac{k L_{\rm box}}{2\pi}$", 
              fontsize=40,  # Increased for maximum impact
              fontweight='bold',  # Bold for memorability
              color='#000000')  # Pure black for maximum contrast
ax.set_ylabel(r"$P_u(k)$", 
              fontsize=36,  # Increased for maximum impact
              fontweight='bold',  # Bold for memorability
              color='#000000')  # Pure black for maximum contrast

# Set axis limits
# print("LIIMM=", xlim_max)
ax.set_xlim(xlim_min, xlim_max)
ax.set_ylim(1e0, 1e10)

# Enhanced tick formatting for maximum visibility and memorability
ax.tick_params(which='major', 
               length=14,  # Longer for visibility
               width=3.0,  # Thicker for impact
               labelsize=28,  # Larger for readability
               color='#000000',  # Pure black
               pad=10)  # More padding
ax.tick_params(which='minor', 
               length=7,  # Longer minor ticks
               width=2.0,  # Thicker minor ticks
               color='#000000')  # Pure black
ax.minorticks_on()

# Format tick labels for better readability
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))

# Add a legend entry for the stars
# Create a dummy plot with star marker for legend
# ax.plot([], [], '*', color='black', markersize=18,
#         markeredgecolor='black', markeredgewidth=0.8,
#         markerfacecolor='black', label='black', 
#         linestyle='None', clip_on=False)

# ============================================================================
# LEGEND - Elegant and well-positioned
# ============================================================================
# Determine optimal legend location and columns
if n_etas <= 6:
    ncol = 1
    loc = 'upper right'
else:
    ncol = 2
    loc = 'upper right'

legend = ax.legend(loc=loc, 
                   frameon=True, 
                   fancybox=False,
                   framealpha=0.98, 
                   edgecolor='#000000',  # Bold black border
                   facecolor='#FFFFFF',  # Pure white background
                   fontsize=24,  # Larger for readability
                   ncol=ncol,
                   columnspacing=1.8,  # More spacing
                   handlelength=3.0,  # Longer handles for visibility
                   handletextpad=1.2,  # More padding
                   labelspacing=0.8,  # More spacing between labels
                   borderpad=1.2)  # More border padding
legend.get_frame().set_linewidth(2.5)  # Thicker border for prominence

# ============================================================================
# GRID - Subtle but effective for readability
# ============================================================================
# ax.grid(True, 
#         alpha=0.32,  # Slightly more visible
#         linestyle='--', 
#         linewidth=1.2,  # Slightly thicker
#         which='both',
#         color='#666666',  # Darker for better visibility
#         zorder=0)
# ax.set_axisbelow(True)

# ============================================================================
# FINAL POLISH - Maximum visual impact and memorability
# ============================================================================
# Enhanced spacing for premium appearance
fig.tight_layout(pad=2.5)

# Add subtle visual enhancement - ensure all elements are crisp
# This creates a professional, publication-ready appearance that stands out
for spine in ax.spines.values():
    spine.set_linewidth(3.0)  # Thicker spines for impact
    spine.set_color('#000000')  # Pure black for maximum contrast

# ============================================================================
# SAVE FIGURES - Maximum quality for publication
# ============================================================================
out_png = f"{out_prefix}_combined.png"
out_pdf = f"{out_prefix}_combined.pdf"
out_svg = f"{out_prefix}_combined.svg"
out_eps = f"{out_prefix}_combined.eps"

# Save PNG with maximum quality
fig.savefig(out_png, 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.15,
            facecolor='white',
            edgecolor='none',
            format='png')

# Save PDF (preferred for ApJ submissions - vector format)
fig.savefig(out_pdf, 
            bbox_inches='tight', 
            pad_inches=0.15,
            facecolor='white',
            edgecolor='none',
            format='pdf')

# Save SVG (vector format for presentations)
fig.savefig(out_svg, 
            bbox_inches='tight', 
            pad_inches=0.15,
            facecolor='white',
            edgecolor='none',
            format='svg')

# Save EPS (sometimes required by journals)
fig.savefig(out_eps, 
            bbox_inches='tight', 
            pad_inches=0.15,
            facecolor='white',
            edgecolor='none',
            format='eps')

plt.close(fig)

print(f"✓ Saved {out_png}")
print(f"✓ Saved {out_pdf}")
print(f"✓ Saved {out_svg}")
print(f"✓ Saved {out_eps}")
print("\n🎨 Publication-ready figure generated with elite styling!")

