import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def compute_P_from_B(bx, by, bz=None):
    """
    Compute complex polarization P = Q + iU from B field components.
    P = (bx + i*by)^2 integrated along line of sight (z-axis).
    Returns P_i as a 2D complex array.
    """
    # Compute complex B_perp^2: (bx + i*by)^2
    B_complex = (bx + 1j*by)**2
    
    # Integrate along z-axis (axis 2) to get 2D polarization
    P_i = np.sum(B_complex, axis=2)
    
    return P_i

def compute_u_i_from_P(P_i, eps=1e-30):
    """
    Compute unit polarization u_i = P_i / |P_i|
    """
    return P_i / (np.abs(P_i) + eps)

if __name__ == "__main__":
    # Load data from npz file
    input_file = '256.npz'  # Update with actual path
    print(f"Loading data from {input_file}...")
    data = np.load(input_file)
    
    # Extract B field components
    bx = data['bx_cube']
    by = data['by_cube']
    bz = data['bz_cube'] if 'bz_cube' in data else None
    print(bx)
    # Extract parameters
    N = int(data['N'])
    L = float(data['L'])
    
    print(f"Loaded B field cubes:")
    print(f"  Shape: {bx.shape}")
    print(f"  N: {N}, L: {L}")
    
    # Optional: Add mean magnetic field
    B0x = 0.0  # Can be modified
    B0y = 0.0
    B0z = 0.0
    
    if B0x != 0.0 or B0y != 0.0 or B0z != 0.0:
        print(f"Adding mean magnetic field: B0x={B0x}, B0y={B0y}, B0z={B0z}")
        bx = bx + B0x
        by = by + B0y
        if bz is not None:
            bz = bz + B0z
    
    # Compute P_i = Q + iU along line of sight
    print("Computing P_i = Q + iU...")
    P_i = compute_P_from_B(bx, by, bz)
    
    # Compute u_i = P_i / |P_i|
    print("Computing u_i = P_i / |P_i|...")
    u_i = compute_u_i_from_P(P_i)
    
    print(f"u_i statistics:")
    print(f"  Shape: {u_i.shape}")
    print(f"  |u_i| mean: {np.abs(u_i).mean():.6f} (should be ~1.0)")
    print(f"  |u_i| std: {np.abs(u_i).std():.6f}")
    
    # Compute differences between nearby lines of sight
    # Choose offset distance (number of cells)
    offset = 16  # Distance between LOS pairs (can be modified)
    
    print(f"\nComputing u_i differences for offset = {offset} cells...")
    
    # Compute differences in horizontal direction (x-direction)
    u_i_diff_h = np.zeros((N - offset, N), dtype=complex)
    for i in range(N - offset):
        for j in range(N):
            u_i_diff_h[i, j] = u_i[i, j] - u_i[i + offset, j]
    
    # Compute differences in vertical direction (y-direction)
    u_i_diff_v = np.zeros((N, N - offset), dtype=complex)
    for i in range(N):
        for j in range(N - offset):
            u_i_diff_v[i, j] = u_i[i, j] - u_i[i, j + offset]
    
    # Flatten differences
    u_i_diff_h_flat = u_i_diff_h.flatten()
    u_i_diff_v_flat = u_i_diff_v.flatten()
    
    # Compute magnitude of differences
    u_i_diff_h_mag = np.abs(u_i_diff_h_flat)
    u_i_diff_v_mag = np.abs(u_i_diff_v_flat)
    
    # Remove invalid values
    u_i_diff_h_mag_valid = u_i_diff_h_mag[np.isfinite(u_i_diff_h_mag)]
    u_i_diff_v_mag_valid = u_i_diff_v_mag[np.isfinite(u_i_diff_v_mag)]
    
    print(f"\nDifference statistics (horizontal, offset={offset}):")
    print(f"  |u_i1 - u_i2| mean: {u_i_diff_h_mag_valid.mean():.6f}")
    print(f"  |u_i1 - u_i2| median: {np.median(u_i_diff_h_mag_valid):.6f}")
    print(f"  |u_i1 - u_i2| std: {u_i_diff_h_mag_valid.std():.6f}")
    print(f"  |u_i1 - u_i2| max: {u_i_diff_h_mag_valid.max():.6f}")
    
    print(f"\nDifference statistics (vertical, offset={offset}):")
    print(f"  |u_i1 - u_i2| mean: {u_i_diff_v_mag_valid.mean():.6f}")
    print(f"  |u_i1 - u_i2| median: {np.median(u_i_diff_v_mag_valid):.6f}")
    print(f"  |u_i1 - u_i2| std: {u_i_diff_v_mag_valid.std():.6f}")
    print(f"  |u_i1 - u_i2| max: {u_i_diff_v_mag_valid.max():.6f}")
    
    # Create figure with histogram and PDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Histogram of |u_i1 - u_i2|
    n_bins = 100
    
    # Combine horizontal and vertical differences
    all_diff_mag = np.concatenate([u_i_diff_h_mag_valid, u_i_diff_v_mag_valid])
    
    # Use log bins for better visualization
    log_bins = np.logspace(np.log10(all_diff_mag.min() + 1e-10), 
                          np.log10(all_diff_mag.max()), 
                          n_bins + 1)
    
    counts_h, bins_h, patches_h = ax1.hist(u_i_diff_h_mag_valid, bins=log_bins, density=False, 
                                          alpha=0.6, color='blue', edgecolor='black', linewidth=0.5,
                                          label=f'Horizontal')
    # counts_v, bins_v, patches_v = ax1.hist(u_i_diff_v_mag_valid, bins=log_bins, density=False, 
    #                                       alpha=0.6, color='red', edgecolor='black', linewidth=0.5,
    #                                       label=f'Vertical (offset={offset})')
    
    ax1.set_xlabel(r'$|u_{i1} - u_{i2}|$')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_xlim(1e-8, 1e1)
    
    # Plot 2: PDF (Probability Density Function)
    counts_pdf_h, bins_pdf_h = np.histogram(u_i_diff_h_mag_valid, bins=log_bins, density=True)
    counts_pdf_v, bins_pdf_v = np.histogram(u_i_diff_v_mag_valid, bins=log_bins, density=True)
    bin_centers_h = 0.5 * (bins_pdf_h[:-1] + bins_pdf_h[1:])
    bin_centers_v = 0.5 * (bins_pdf_v[:-1] + bins_pdf_v[1:])
    
    # Filter out zero counts
    mask_h = counts_pdf_h > 0
    mask_v = counts_pdf_v > 0
    ax2.loglog(bin_centers_h[mask_h], counts_pdf_h[mask_h], 'o', color='blue', 
              markersize=4, linewidth=1.5, alpha=0.7, label=f'Horizontal')
    # ax2.loglog(bin_centers_v[mask_v], counts_pdf_v[mask_v], 's', color='red', 
    #           markersize=4, linewidth=1.5, alpha=0.7, label=f'Vertical (offset={offset})')
    
    ax2.set_xlabel(r'$|u_{i1} - u_{i2}|$')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.set_xlim(1e-8, 1e1)
    plt.tight_layout()
    
    # Save figure
    output_file = f'u_i_diff_histogram_offset{offset}.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nSaved figure to {output_file}")
    
    # Also save as PNG
    output_file_png = f'u_i_diff_histogram_offset{offset}.png'
    plt.savefig(output_file_png, bbox_inches='tight', dpi=150)
    print(f"Saved figure to {output_file_png}")
    
    plt.show()

