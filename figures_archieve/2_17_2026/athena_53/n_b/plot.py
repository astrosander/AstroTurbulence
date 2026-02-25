import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from pathlib import Path

# Configure matplotlib for publication-quality plots with LaTeX
# Note: LaTeX rendering requires LaTeX installation. If unavailable, matplotlib
# will automatically fall back to its built-in math rendering.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.5,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi": 600,  # Higher DPI for publication
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 100,
    "pdf.fonttype": 42,  # TrueType fonts for PDF
    "ps.fonttype": 42,   # TrueType fonts for PS
})

# ============================================================
# Files to compare
# ============================================================
F1 = "spectrum_spectra_ms10.npz"
F2 = "spectrum_spectra_ms1.npz"
F3 = "Phi_neBz_integral_spectrum_compare.npz"  # Phi = int(ne*Bz*dz) spectrum
F4_PU_MS10 = "Pu_spectrum_ms10.npz"  # Pu spectrum for Ms=10
F4_PU_MS1 = "Pu_spectrum_ms1.npz"   # Pu spectrum for Ms=1

LABEL1 = r"$M_s=10$"
LABEL2 = r"$M_s=1$"
OUTFIG = "compare_density_magnetic_ms10_vs_ms1.svg"


def load_npz(npz_path):
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")
    return np.load(p)


def get_spectrum_keys(d, kind):
    """
    Support both newer files (density: k_d,E_d) and older files if present.
    kind = 'density' or 'magnetic'
    """
    if kind == "density":
        candidates = [("k_d", "E_d"), ("k_rho", "E_rho")]
    elif kind == "magnetic":
        candidates = [("k_b", "E_b")]
    else:
        raise ValueError("kind must be 'density' or 'magnetic'")

    for kkey, ekey in candidates:
        if kkey in d and ekey in d:
            return kkey, ekey

    raise KeyError(
        f"Could not find {kind} spectrum keys in file. "
        f"Available keys: {list(d.keys())}"
    )


def get_spectrum(d, kind):
    kkey, ekey = get_spectrum_keys(d, kind)
    return d[kkey], d[ekey]


def main():
    d1 = load_npz(F1)
    d2 = load_npz(F2)

    # Load density spectra
    k1d, E1d = get_spectrum(d1, "density")
    k2d, E2d = get_spectrum(d2, "density")

    # Load magnetic spectra
    k1b, E1b = get_spectrum(d1, "magnetic")
    k2b, E2b = get_spectrum(d2, "magnetic")

    # Load Phi_neBz integral spectrum
    try:
        d3 = load_npz(F3)
        k3_ms10 = np.asarray(d3["ms10_k_centers"]) / (2.0 * np.pi)  # Divide by 2π
        E3_ms10 = np.asarray(d3["ms10_E"])
        k3_ms1 = np.asarray(d3["ms1_k_centers"]) / (2.0 * np.pi)  # Divide by 2π
        E3_ms1 = np.asarray(d3["ms1_E"])
        has_phi_data = True
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load {F3}: {e}")
        has_phi_data = False
        k3_ms10 = None
        E3_ms10 = None
        k3_ms1 = None
        E3_ms1 = None

    # Load Pu spectrum
    try:
        d4_ms10 = load_npz(F4_PU_MS10)
        d4_ms1 = load_npz(F4_PU_MS1)
        # Extract k_centers and E_u (E_u has shape (1, n_k), so take [0])
        k4_ms10 = np.asarray(d4_ms10["k_centers"]) / (2.0 * np.pi)  # Divide by 2π
        E4_ms10 = np.asarray(d4_ms10["E_u"])[0]  # Extract first (and only) spectrum
        k4_ms1 = np.asarray(d4_ms1["k_centers"]) / (2.0 * np.pi)  # Divide by 2π
        E4_ms1 = np.asarray(d4_ms1["E_u"])[0]  # Extract first (and only) spectrum
        has_pu_data = True
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load Pu spectrum files: {e}")
        has_pu_data = False
        k4_ms10 = None
        E4_ms10 = None
        k4_ms1 = None
        E4_ms1 = None

    # Professional color palette (distinguishable, publication-friendly)
    color1 = "blue"  # Blue
    color2 = "red"  # Red
    # Both curves use solid lines
    linestyle1 = "-"
    linestyle2 = "-"

    # Create figure with optimal dimensions for ApJ double-column (7.0" width max)
    # 2x2 panel layout
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), 
                                        constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # -------------------------
    # Left panel: Density
    # -------------------------
    ax1.loglog(k1d, E1d, color=color1, linestyle=linestyle1, 
               linewidth=2.2, label=LABEL1, zorder=4, alpha=0.95)
    ax1.loglog(k2d, E2d, color=color2, linestyle=linestyle2, 
               linewidth=2.2, label=LABEL2, zorder=4, alpha=0.95)
    ax1.set_xlabel(r"$\frac{k L_{\rm box}}{2\pi}$", fontsize=16, labelpad=3)
    ax1.set_ylabel(r"$E_{n}(k)$", fontsize=16, labelpad=3)
    ax1.set_xlim(1e0, 256)
    ax1.set_ylim(1e-7, 1e-1)
    
    # Add panel label
    ax1.text(0.05, 0.95, r"\textbf{(a)}", transform=ax1.transAxes,
             fontsize=11, verticalalignment="top", 
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", 
                      edgecolor="none", alpha=0.8))
    
    # Fit power law to blue curve (k1d, E1d)
    # Filter valid data points in the fit range (k from 1 to 50)
    k_fit_min = 1.0
    k_fit_max = 30.0
    mask = (k1d >= k_fit_min) & (k1d <= k_fit_max) & \
           (E1d >= ax1.get_ylim()[0]) & (E1d <= ax1.get_ylim()[1]) & \
           np.isfinite(k1d) & np.isfinite(E1d) & (k1d > 0) & (E1d > 0)
    k1d_fit = k1d[mask]
    E1d_fit = E1d[mask]
    
    if len(k1d_fit) > 1:
        # Fit log(E) = log(A) + alpha * log(k)
        log_k = np.log10(k1d_fit)
        log_E = np.log10(E1d_fit)
        # Use polyfit for linear fit in log space
        coeffs = np.polyfit(log_k, log_E, 1)
        alpha = coeffs[0]  # Power law exponent
        log_A = coeffs[1]  # Intercept
        A = 10**log_A
        
        # Generate fitted curve only in the fit range
        k_fit = np.logspace(np.log10(k_fit_min), 
                           np.log10(k_fit_max), 100)
        E_fit = A * k_fit**alpha
        
        # Plot the fit
        ax1.loglog(k_fit, E_fit, color="black", linestyle=(0, (3, 1, 1, 1)), 
                   linewidth=1.6, alpha=0.85, zorder=3,
                   label=rf"fit: $k^{{{alpha:.2f}}}$")
    
    # Add -5/3 reference line (Kolmogorov power law)
    k_ref = np.logspace(np.log10(ax1.get_xlim()[0]), np.log10(ax1.get_xlim()[1]), 200)
    # Normalize at a point in the middle of the plot range
    k_align = 10.0
    y_align = 1e-4  # Choose a point in the middle of the y-range
    y_ref_scaled = y_align * (k_ref / k_align)**(-5/3)
    ax1.loglog(k_ref, y_ref_scaled, color="#228B22", linestyle=(0, (5, 2)), 
               linewidth=4.4, alpha=0.9, zorder=1,
               label=r"$k^{-5/3}$")
    
    # Add grid for better readability (subtle, professional)
    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.15, zorder=0)
    
    # Format ticks professionally (publication quality)
    # Set explicit tick locators for consistent formatting (less dense)
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=100))
    ax1.tick_params(which="both", direction="in", top=True, right=True, 
                   pad=4, width=1.0)
    ax1.tick_params(which="major", length=5, width=1.0, labelsize=9)
    ax1.tick_params(which="minor", length=3, width=0.8)
    
    # Professional legend with optimal placement (compact)
    ax1.legend(loc="upper right", frameon=True, fancybox=False, 
              shadow=False, framealpha=1.0, edgecolor="black",
              facecolor="white", borderpad=0.3, labelspacing=0.4,
              handlelength=1.8, handletextpad=0.4, fontsize=8.5)

    # -------------------------
    # Right panel: Magnetic field
    # -------------------------
    ax2.loglog(k1b, E1b, color=color1, linestyle=linestyle1, 
               linewidth=2.2, label=LABEL1, zorder=4, alpha=0.95)
    ax2.loglog(k2b, E2b, color=color2, linestyle=linestyle2, 
               linewidth=2.2, label=LABEL2, zorder=4, alpha=0.95)
    ax2.set_xlabel(r"$\frac{k L_{\rm box}}{2\pi}$", fontsize=16, labelpad=3)
    ax2.set_ylabel(r"$E_B(k)$", fontsize=16, labelpad=3)
    ax2.set_xlim(1e0, 256)
    ax2.set_ylim(1e-7, 1e1)  # Match ax1 y-axis range
    
    # Add panel label
    ax2.text(0.05, 0.95, r"\textbf{(b)}", transform=ax2.transAxes,
             fontsize=11, verticalalignment="top", 
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", 
                      edgecolor="none", alpha=0.8))
    
    # Fit power law to blue curve (k1b, E1b)
    # Filter valid data points in the fit range (k from 1 to 30)
    k_fit_min2 = 1.0
    k_fit_max2 = 30.0
    mask2 = (k1b >= k_fit_min2) & (k1b <= k_fit_max2) & \
            (E1b >= ax2.get_ylim()[0]) & (E1b <= ax2.get_ylim()[1]) & \
            np.isfinite(k1b) & np.isfinite(E1b) & (k1b > 0) & (E1b > 0)
    k1b_fit = k1b[mask2]
    E1b_fit = E1b[mask2]
    
    if len(k1b_fit) > 1:
        # Fit log(E) = log(A) + alpha * log(k)
        log_k2 = np.log10(k1b_fit)
        log_E2 = np.log10(E1b_fit)
        # Use polyfit for linear fit in log space
        coeffs2 = np.polyfit(log_k2, log_E2, 1)
        alpha2 = coeffs2[0]  # Power law exponent
        log_A2 = coeffs2[1]  # Intercept
        A2 = 10**log_A2
        
        # Generate fitted curve only in the fit range
        k_fit2 = np.logspace(np.log10(k_fit_min2), 
                            np.log10(k_fit_max2), 100)
        E_fit2 = A2 * k_fit2**alpha2
        
        # Plot the fit
        # ax2.loglog(k_fit2, E_fit2, color="black", linestyle=(0, (3, 1, 1, 1)), 
        #            linewidth=1.6, alpha=0.85, zorder=3,
        #            label=rf"fit: $k^{{{alpha2:.2f}}}$")
    
    # Add -5/3 reference line (Kolmogorov power law)
    k_ref2 = np.logspace(np.log10(ax2.get_xlim()[0]), np.log10(ax2.get_xlim()[1]), 200)
    # Normalize to be visible in the plot range (matching ax1)
    k_align2 = 10.0
    y_align2 = 1e-4  # Match ax1 normalization
    y_ref_scaled2 = y_align2 * (k_ref2 / k_align2)**(-5/3)
    ax2.loglog(k_ref2, y_ref_scaled2, color="#228B22", linestyle=(0, (5, 2)), 
               linewidth=4.4, alpha=0.9, zorder=1,
               label=r"$k^{-5/3}$")
    
    # Add grid for better readability (subtle, professional)
    ax2.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)
    ax2.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.15, zorder=0)
    
    # Format ticks professionally (publication quality)
    # Set explicit tick locators to match ax1 (less dense)
    ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=100))
    ax2.tick_params(which="both", direction="in", top=True, right=True, 
                   pad=4, width=1.0)
    ax2.tick_params(which="major", length=5, width=1.0, labelsize=9)
    ax2.tick_params(which="minor", length=3, width=0.8)
    
    # Professional legend with optimal placement (compact)
    ax2.legend(loc="upper right", frameon=True, fancybox=False, 
              shadow=False, framealpha=1.0, edgecolor="black",
              facecolor="white", borderpad=0.3, labelspacing=0.4,
              handlelength=1.8, handletextpad=0.4, fontsize=8.5)

    # -------------------------
    # Third panel: Phi = int(ne*Bz*dz) spectrum
    # -------------------------
    if has_phi_data:
        # Filter valid data points
        m_ms10 = np.isfinite(k3_ms10) & np.isfinite(E3_ms10) & (k3_ms10 > 0) & (E3_ms10 > 0)
        m_ms1 = np.isfinite(k3_ms1) & np.isfinite(E3_ms1) & (k3_ms1 > 0) & (E3_ms1 > 0)
        
        ax3.loglog(k3_ms10[m_ms10], E3_ms10[m_ms10], color=color1, linestyle=linestyle1, 
                   linewidth=2.2, label=LABEL1, zorder=4, alpha=0.95)
        ax3.loglog(k3_ms1[m_ms1], E3_ms1[m_ms1], color=color2, linestyle=linestyle2, 
                   linewidth=2.2, label=LABEL2, zorder=4, alpha=0.95)
        ax3.set_xlabel(r"$\frac{k L_{\rm box}}{2\pi}$", fontsize=16, labelpad=3)
        ax3.set_ylabel(r"$P_{\int n_e B_z dz}(k)$", fontsize=16, labelpad=3)
        ax3.set_xlim(1e0, 256)
        # Auto-scale y-axis based on data range
        if np.any(m_ms10) and np.any(m_ms1):
            y_min = min(np.min(E3_ms10[m_ms10]), np.min(E3_ms1[m_ms1]))
            y_max = max(np.max(E3_ms10[m_ms10]), np.max(E3_ms1[m_ms1]))
            y_min = 10**(np.floor(np.log10(y_min)))
            y_max = 10**(np.ceil(np.log10(y_max)))
            ax3.set_ylim(y_min, y_max)
        else:
            ax3.set_ylim(1e-7, 1e1)

        ax3.set_ylim(1e-8, 1e4)
        
        # Add panel label
        ax3.text(0.05, 0.95, r"\textbf{(c)}", transform=ax3.transAxes,
                 fontsize=11, verticalalignment="top", 
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", 
                          edgecolor="none", alpha=0.8))
        
        # Fit power law to both curves (k from 20 to 200)
        k_fit_min3 = 2.0
        k_fit_max3 = 20.0
        
        # Fit for ms10 (blue curve)
        mask3_ms10 = (k3_ms10 >= k_fit_min3) & (k3_ms10 <= k_fit_max3) & \
                     (E3_ms10 >= ax3.get_ylim()[0]) & (E3_ms10 <= ax3.get_ylim()[1]) & \
                     np.isfinite(k3_ms10) & np.isfinite(E3_ms10) & (k3_ms10 > 0) & (E3_ms10 > 0)
        k3_ms10_fit = k3_ms10[mask3_ms10]
        E3_ms10_fit = E3_ms10[mask3_ms10]
        
        
        if len(k3_ms10_fit) > 1:
            log_k3_ms10 = np.log10(k3_ms10_fit)
            log_E3_ms10 = np.log10(E3_ms10_fit)
            coeffs3_ms10 = np.polyfit(log_k3_ms10, log_E3_ms10, 1)
            alpha3_ms10 = coeffs3_ms10[0]
            log_A3_ms10 = coeffs3_ms10[1]
            A3_ms10 = 10**log_A3_ms10
            
            k_fit3_ms10 = np.logspace(np.log10(k_fit_min3), np.log10(k_fit_max3), 100)
            E_fit3_ms10 = A3_ms10 * k_fit3_ms10**alpha3_ms10
            
            # Fit line for ms10: use blue color with dashed style
            ax3.loglog(k_fit3_ms10, E_fit3_ms10, color=color1, linestyle=(0, (3, 1, 1, 1)), 
                       linewidth=1.6, alpha=0.85, zorder=3,
                       label=rf"fit $M_s=10$: $k^{{{alpha3_ms10:.2f}}}$")
        
        # Fit for ms1 (red curve)
        mask3_ms1 = (k3_ms1 >= k_fit_min3) & (k3_ms1 <= k_fit_max3) & \
                    (E3_ms1 >= ax3.get_ylim()[0]) & (E3_ms1 <= ax3.get_ylim()[1]) & \
                    np.isfinite(k3_ms1) & np.isfinite(E3_ms1) & (k3_ms1 > 0) & (E3_ms1 > 0)
        k3_ms1_fit = k3_ms1[mask3_ms1]
        E3_ms1_fit = E3_ms1[mask3_ms1]
        
        if len(k3_ms1_fit) > 1:
            log_k3_ms1 = np.log10(k3_ms1_fit)
            log_E3_ms1 = np.log10(E3_ms1_fit)
            coeffs3_ms1 = np.polyfit(log_k3_ms1, log_E3_ms1, 1)
            alpha3_ms1 = coeffs3_ms1[0]
            log_A3_ms1 = coeffs3_ms1[1]
            A3_ms1 = 10**log_A3_ms1
            
            k_fit3_ms1 = np.logspace(np.log10(k_fit_min3), np.log10(k_fit_max3), 100)
            E_fit3_ms1 = A3_ms1 * k_fit3_ms1**alpha3_ms1
            
            # Fit line for ms1: use red color with dash-dot style
            # ax3.loglog(k_fit3_ms1, E_fit3_ms1, color=color2, linestyle=(0, (5, 1, 1, 1)), 
            #            linewidth=1.6, alpha=0.85, zorder=3,
            #            label=rf"fit $M_s=1$: $k^{{{alpha3_ms1:.2f}}}$")
        
        # Add -5/3 reference line (Kolmogorov power law)
        k_ref3 = np.logspace(np.log10(ax3.get_xlim()[0]), np.log10(ax3.get_xlim()[1]), 200)
        # Normalize to be visible in the plot range
        k_align3 = 10.0
        y_align3 = np.mean(ax3.get_ylim())  # Use middle of y-range
        y_ref_scaled3 = y_align3 * (k_ref3 / k_align3)**(-8/3)*0.001
        ax3.loglog(k_ref3, y_ref_scaled3, color="purple", linestyle=(0, (5, 2)), 
                   linewidth=4.4, alpha=0.9, zorder=1,
                   label=r"$k^{-8/3}$")
        
        # Add grid for better readability (subtle, professional)
        ax3.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)
        ax3.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.15, zorder=0)
        
        # Format ticks professionally (publication quality)
        ax3.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax3.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=100))
        ax3.tick_params(which="both", direction="in", top=True, right=True, 
                       pad=4, width=1.0)
        ax3.tick_params(which="major", length=5, width=1.0, labelsize=9)
        ax3.tick_params(which="minor", length=3, width=0.8)
        
        # Professional legend with optimal placement (compact)
        ax3.legend(loc="upper right", frameon=True, fancybox=False, 
                  shadow=False, framealpha=1.0, edgecolor="black",
                  facecolor="white", borderpad=0.3, labelspacing=0.4,
                  handlelength=1.8, handletextpad=0.4, fontsize=8.5)
    else:
        # Hide third panel if data not available
        ax3.axis('off')
        ax3.text(0.5, 0.5, f"Data not available:\n{F3}", 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor="lightgray"))

    # -------------------------
    # Fourth panel: Pu spectrum
    # -------------------------
    if has_pu_data:
        # Filter valid data points
        m_pu_ms10 = np.isfinite(k4_ms10) & np.isfinite(E4_ms10) & (k4_ms10 > 0) & (E4_ms10 > 0)
        m_pu_ms1 = np.isfinite(k4_ms1) & np.isfinite(E4_ms1) & (k4_ms1 > 0) & (E4_ms1 > 0)
        
        ax4.loglog(k4_ms10[m_pu_ms10], E4_ms10[m_pu_ms10], color=color1, linestyle=linestyle1, 
                   linewidth=2.2, label=LABEL1, zorder=4, alpha=0.95)
        ax4.loglog(k4_ms1[m_pu_ms1], E4_ms1[m_pu_ms1], color=color2, linestyle=linestyle2, 
                   linewidth=2.2, label=LABEL2, zorder=4, alpha=0.95)
        ax4.set_xlabel(r"$\frac{k L_{\rm box}}{2\pi}$", fontsize=16, labelpad=3)
        ax4.set_ylabel(r"$P_u(k)$", fontsize=16, labelpad=3)
        ax4.set_xlim(1e0, 256)
        # Auto-scale y-axis based on data range
        if np.any(m_pu_ms10) and np.any(m_pu_ms1):
            y_min = min(np.min(E4_ms10[m_pu_ms10]), np.min(E4_ms1[m_pu_ms1]))
            y_max = max(np.max(E4_ms10[m_pu_ms10]), np.max(E4_ms1[m_pu_ms1]))
            y_min = 10**(np.floor(np.log10(y_min)))
            y_max = 10**(np.ceil(np.log10(y_max)))
            ax4.set_ylim(y_min, y_max)
        else:
            ax4.set_ylim(1e-7, 1e1)
        
        # Add panel label
        ax4.text(0.05, 0.95, r"\textbf{(d)}", transform=ax4.transAxes,
                 fontsize=11, verticalalignment="top", 
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", 
                          edgecolor="none", alpha=0.8))
        
        # Fit power law to both curves
        k_fit_min4 = 1.0
        k_fit_max4 = 30.0
        
        # Fit for ms10 (blue curve)
        mask4_ms10 = (k4_ms10 >= k_fit_min4) & (k4_ms10 <= k_fit_max4) & \
                     (E4_ms10 >= ax4.get_ylim()[0]) & (E4_ms10 <= ax4.get_ylim()[1]) & \
                     np.isfinite(k4_ms10) & np.isfinite(E4_ms10) & (k4_ms10 > 0) & (E4_ms10 > 0)
        k4_ms10_fit = k4_ms10[mask4_ms10]
        E4_ms10_fit = E4_ms10[mask4_ms10]
        
        if len(k4_ms10_fit) > 1:
            log_k4_ms10 = np.log10(k4_ms10_fit)
            log_E4_ms10 = np.log10(E4_ms10_fit)
            coeffs4_ms10 = np.polyfit(log_k4_ms10, log_E4_ms10, 1)
            alpha4_ms10 = coeffs4_ms10[0]
            log_A4_ms10 = coeffs4_ms10[1]
            A4_ms10 = 10**log_A4_ms10
            
            k_fit4_ms10 = np.logspace(np.log10(k_fit_min4), np.log10(k_fit_max4), 100)
            E_fit4_ms10 = A4_ms10 * k_fit4_ms10**alpha4_ms10
            
            # Fit line for ms10: use blue color with dashed style
            ax4.loglog(k_fit4_ms10, E_fit4_ms10, color=color1, linestyle=(0, (3, 1, 1, 1)), 
                       linewidth=1.6, alpha=0.85, zorder=3,
                       label=rf"fit $M_s=10$: $k^{{{alpha4_ms10:.2f}}}$")
        
        # Fit for ms1 (red curve)
        mask4_ms1 = (k4_ms1 >= k_fit_min4) & (k4_ms1 <= k_fit_max4) & \
                    (E4_ms1 >= ax4.get_ylim()[0]) & (E4_ms1 <= ax4.get_ylim()[1]) & \
                    np.isfinite(k4_ms1) & np.isfinite(E4_ms1) & (k4_ms1 > 0) & (E4_ms1 > 0)
        k4_ms1_fit = k4_ms1[mask4_ms1]
        E4_ms1_fit = E4_ms1[mask4_ms1]
        
        if len(k4_ms1_fit) > 1:
            log_k4_ms1 = np.log10(k4_ms1_fit)
            log_E4_ms1 = np.log10(E4_ms1_fit)
            coeffs4_ms1 = np.polyfit(log_k4_ms1, log_E4_ms1, 1)
            alpha4_ms1 = coeffs4_ms1[0]
            log_A4_ms1 = coeffs4_ms1[1]
            A4_ms1 = 10**log_A4_ms1
            
            k_fit4_ms1 = np.logspace(np.log10(k_fit_min4), np.log10(k_fit_max4), 100)
            E_fit4_ms1 = A4_ms1 * k_fit4_ms1**alpha4_ms1
            
            # Fit line for ms1: use red color with dash-dot style (optional, commented out)
            # ax4.loglog(k_fit4_ms1, E_fit4_ms1, color=color2, linestyle=(0, (5, 1, 1, 1)), 
            #            linewidth=1.6, alpha=0.85, zorder=3,
            #            label=rf"fit $M_s=1$: $k^{{{alpha4_ms1:.2f}}}$")
        
        # Add -5/3 reference line (Kolmogorov power law)
        k_ref4 = np.logspace(np.log10(ax4.get_xlim()[0]), np.log10(ax4.get_xlim()[1]), 200)
        # Normalize to be visible in the plot range
        k_align4 = 10.0
        y_align4 = np.mean(ax4.get_ylim())  # Use middle of y-range
        y_ref_scaled4 = y_align4 * (k_ref4 / k_align4)**(-8/3)*0.001
        ax4.loglog(k_ref4, y_ref_scaled4, color="#228B22", linestyle=(0, (5, 2)), 
                   linewidth=4.4, alpha=0.9, zorder=1,
                   label=r"$k^{-8/3}$")
        
        # Add grid for better readability (subtle, professional)
        ax4.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)
        ax4.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.15, zorder=0)
        
        # Format ticks professionally (publication quality)
        ax4.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax4.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=100))
        ax4.tick_params(which="both", direction="in", top=True, right=True, 
                       pad=4, width=1.0)
        ax4.tick_params(which="major", length=5, width=1.0, labelsize=9)
        ax4.tick_params(which="minor", length=3, width=0.8)
        
        # Professional legend with optimal placement (compact)
        ax4.legend(loc="upper right", frameon=True, fancybox=False, 
                  shadow=False, framealpha=1.0, edgecolor="black",
                  facecolor="white", borderpad=0.3, labelspacing=0.4,
                  handlelength=1.8, handletextpad=0.4, fontsize=8.5)
    else:
        # Hide fourth panel if data not available
        ax4.axis('off')
        ax4.text(0.5, 0.5, f"Data not available:\nPu spectrum files", 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor="lightgray"))

    # Save with publication-quality settings
    # Using constrained_layout, so no need for tight_layout
    plt.savefig(OUTFIG, dpi=600, bbox_inches="tight", pad_inches=0.05, 
                facecolor="white", edgecolor="none", format="svg")
    # Also save as PDF for publication (ApJ prefers PDF)
    if OUTFIG.endswith(".svg"):
        pdf_fig = OUTFIG.replace(".svg", ".pdf")
        plt.savefig(pdf_fig, dpi=600, bbox_inches="tight", pad_inches=0.05, 
                    facecolor="white", edgecolor="none", format="pdf")
        print(f"Saved: {pdf_fig}")
    # plt.show()

    print(f"Saved: {OUTFIG}")


if __name__ == "__main__":
    main()