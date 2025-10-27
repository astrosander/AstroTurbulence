# recreate_fig2_from_npz.py
# Recreates the figure from saved .npz files in the npz/ directory
# Usage: Run after plot_fig2_zhang2016.py has saved the .npz files

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main():
    # Find all .npz files in npz/ directory
    npz_files = glob.glob("npz/curve_*.npz")
    
    if not npz_files:
        print("No .npz files found in npz/ directory")
        return
    
    # Load data from each .npz file and extract resolution for sorting
    curves_data_unsorted = []
    for npz_file in npz_files:
        data = np.load(npz_file)
        label = str(data['label'])
        # Extract resolution number (last number in the label, e.g., "128×128×256" -> 256)
        resolution = int(label.split('×')[-1])
        curves_data_unsorted.append({
            'lam2': data['lam2'],
            'y': data['y'],
            'label': label,
            'resolution': resolution
        })
        print(f"Loaded {npz_file}: {label}")
    
    # Sort by resolution (smallest to largest)
    curves_data = sorted(curves_data_unsorted, key=lambda x: x['resolution'])
    
    # Define styles to match original
    styles = [":","--","-.", (0,(3,1,1,1)),"-"]
    
    fig, ax = plt.subplots(figsize=(6*1.5, 4.0*1.5))
    
    # Plot each curve
    for i, data in enumerate(curves_data):
        lam2 = data['lam2']
        y = data['y']
        label = str(data['label'])
        ls = styles[i % len(styles)]
        ax.plot(np.log10(lam2), np.log10(y), ls=ls, label=label)
    
    # Draw λ^{-2} reference line (filtered to x-axis range 1 to 2)
    if curves_data:
        lam2 = curves_data[0]['lam2']  # Use lam2 from first curve
        # Use the last curve (typically longest) for reference
        ref_i = len(curves_data) - 1
        x0_log = 0.8
        i0 = int(np.argmin(np.abs(np.log10(lam2) - x0_log)))
        y0 = curves_data[ref_i]['y'][i0]
        ref = y0 * (lam2 / lam2[i0])**(-1.0)
        # Filter to x-axis range from 1 to 2 (log10)
        log10_lam2 = np.log10(lam2)
        mask = (log10_lam2 >= 1) & (log10_lam2 <= 1.6)
        ax.plot(log10_lam2[mask], np.log10(ref)[mask] - 0.1, color="k", linewidth=1.0, label=r"$\lambda^{-2}$")
    
    ax.set_xlabel(r"$\log_{10}[\lambda^2]$")
    ax.set_ylabel(r"$\log_{10}\langle |P|^2\rangle$")
    ax.set_ylim(-4,0)
    ax.set_xlim(-3,3)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("fig2_like.png", dpi=300)
    print("\nFigure saved to fig2_like.png")
    plt.show()

if __name__ == "__main__":
    main()

