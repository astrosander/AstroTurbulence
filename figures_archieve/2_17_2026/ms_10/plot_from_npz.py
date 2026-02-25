#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set matplotlib parameters for ApJ publication quality with LaTeX
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'serif',
    'text.usetex': True,  # Enable LaTeX rendering for ApJ
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'lines.linewidth': 2.5,
    'axes.linewidth': 2.0,
    'grid.linewidth': 1.2,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.minor.width': 1.5,
    'ytick.minor.width': 1.5,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
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

# Create single figure for all eta values
# Larger figure size to accommodate large fonts
# Single column width for ApJ: ~3.5 inches, double column: ~7 inches
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# Compute all spectra first to find global min/max
L_box = 1.0  # Box size
all_Ek = []
k_plot = k_centers[1:]
# Transform k to k*L_box/(2*pi)
k_plot_transformed = k_plot * L_box / (1 * np.pi)
xlim_min_k = 2*np.pi*2
xlim_max_k = np.pi * 1024
xlim_min = xlim_min_k * L_box / (1 * np.pi)
xlim_max = xlim_max_k * L_box / (1 * np.pi)
mask = (k_plot >= xlim_min_k) & (k_plot <= xlim_max_k)

for eta in np.array(eta_list, dtype=float):
    Ek, Pu2d = spectrum_and_map_for_eta(float(eta))
    all_Ek.append(Ek[1:])

# Find global min and max within xlim range
all_Ek = np.array(all_Ek)
Ek_masked = all_Ek[:, mask]
y_min = Ek_masked[Ek_masked > 0].min()  # Only positive values for log scale
y_max = Ek_masked.max()

# Use a modern, unique colormap for better distinction
# Options: 'turbo' (modern vibrant), 'plasma' (modern purple-yellow), 
# 'inferno' (dark to bright), 'tab10' (qualitative discrete)
n_etas = len(eta_list)
# Using 'turbo' for a modern, vibrant colormap that distinguishes well
colors = plt.cm.turbo(np.linspace(0, 1, n_etas))

# Plot spectrum for each eta value with star markers
for i, eta in enumerate(np.array(eta_list, dtype=float)):
    ax.loglog(k_plot_transformed, all_Ek[i], #label=rf"$\eta={eta:.3g}$", 
              color=colors[i], linewidth=2.5, alpha=0.9, marker='*', 
              markersize=8, markeredgecolor='black', markeredgewidth=0.5,
              markerfacecolor=colors[i], markevery=50)  # Mark every 50th point to avoid clutter

# Add markers at k = eta^2 for each line
for i, eta in enumerate(np.array(eta_list, dtype=float)):
    if eta == 0.0:
        continue  # Skip eta=0 (k would be 0)
    
    # Calculate k = eta^2
    k_marker_k = eta ** (2/(-11/3+3/2))*2*np.pi
    
    # Transform to k*L_box/(2π)
    k_marker_transformed = k_marker_k * L_box# / (1 * np.pi)
    
    # Find corresponding P_u(k) value by interpolation
    if k_marker_transformed >= k_plot_transformed.min() and k_marker_transformed <= k_plot_transformed.max():
        # Interpolate to find P_u at this k
        P_u_at_k = np.interp(k_marker_transformed, k_plot_transformed, all_Ek[i])
        
        # Plot marker with elegant, publication-quality styling
        # Use filled star markers with beautiful contrast and depth
        # First plot a visible shadow for depth and contrast
        ax.plot(k_marker_transformed, P_u_at_k, '*', 
               color='black', markersize=20, 
               markeredgecolor='black', markeredgewidth=0,
               markerfacecolor='black', 
               zorder=9, alpha=0.5, clip_on=False)
        # Then plot the main colored star with super small black border
        ax.plot(k_marker_transformed, P_u_at_k, '*', 
               color=colors[i], markersize=18, 
               markeredgecolor='black', markeredgewidth=0.8,
               markerfacecolor=colors[i], 
               zorder=10, alpha=0.7, clip_on=False)
        # Add a bright inner highlight for extra visibility and polish
        ax.plot(k_marker_transformed, P_u_at_k, '*', 
               color='white', markersize=9, 
               markeredgecolor='none', markeredgewidth=0,
               markerfacecolor='white', 
               zorder=11, alpha=0.6, clip_on=False)

# Add reference power-law lines: k^{-11/3} and k^{-3/2}
k_ref_k = np.logspace(np.log10(xlim_min_k), np.log10(xlim_max_k), 1000)
k_ref = k_ref_k * L_box / (1 * np.pi)

# Normalize reference lines to be visible in the plot range
# Choose a normalization point in the middle of the k range
k_norm_k = np.sqrt(xlim_min_k * xlim_max_k)
k_norm = k_norm_k * L_box / (1 * np.pi)
# Normalize to pass through a point in the middle of the y range
y_norm = np.sqrt(y_min * y_max)

# k^{-11/3} reference line (normalized to pass through (k_norm, y_norm))
# Note: power law is in terms of original k, so use k_ref_k
P_ref_11_3 = y_norm * (k_ref_k / k_norm_k) ** (-11/3)
ax.loglog(k_ref, P_ref_11_3, '--', color='red', linewidth=6, 
          alpha=0.7, zorder=0, label=r'$k^{-11/3}$')

# k^{-3/2} reference line (normalized to pass through (k_norm, y_norm))
P_ref_3_2 = y_norm * (k_ref_k / k_norm_k) ** (-3/2)
ax.loglog(k_ref, P_ref_3_2, '--', color='blue', linewidth=8, 
          alpha=0.7, zorder=0, label=r'$k^{-3/2}$')

ax.set_xlabel(r"$\frac{k L_{\rm box}}{\pi}$", fontsize=28)
ax.set_ylabel(r"$P_u(k)$", fontsize=28)
# ax.set_xlim(xlim_min, xlim_max)
ax.set_xlim(np.pi, 1024)
# ax.set_xlim(3, 1024)
ax.set_ylim(1e0, 4e10)  # Set ylim from 10^0 to 10^2

# Format ticks for publication
ax.tick_params(which='major', length=10, width=2.0, labelsize=24)
ax.tick_params(which='minor', length=5, width=1.5)
ax.minorticks_on()

# Add a legend entry for the stars
# Create a dummy plot with star marker for legend
ax.plot([], [], '*', color='black', markersize=18,
        markeredgecolor='black', markeredgewidth=0.8,
        markerfacecolor='black', label='black', 
        linestyle='None', clip_on=False)

# Legend with proper formatting
ax.legend(loc='best', frameon=True, fancybox=True, 
          framealpha=0.9, edgecolor='black', facecolor='white',
          fontsize=20, ncol=1 if n_etas <= 6 else 2)

# Subtle grid for publication
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, which='both')
ax.set_axisbelow(True)

fig.tight_layout()

# Save as both PNG (high-res) and PDF (preferred for publications)
out_png = f"{out_prefix}_combined.png"
out_pdf = f"{out_prefix}_combined.pdf"
out_svg = f"{out_prefix}_combined.svg"
fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.1)
fig.savefig(out_svg, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

print(f"wrote {out_png}")
print(f"wrote {out_pdf}")

