import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, types
from numba.extending import register_jitable
from scipy.special import j0

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    # "savefig.transparent": True,
})


@jit(nopython=True)
def f_reg(R, r0, m):
    x = (R / r0)**m
    return x / (1.0 + x)

@jit(nopython=True)
def trapezoidal_integral(y, x):
    n = len(x)
    result = 0.0
    for i in range(n - 1):
        result += (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2.0
    return result

@jit(nopython=True, parallel=True)
def compute_Pdir_parallel(k_values, R, j0_matrix, mPhi, rphi, chi, mpsi, R0, A_P):
    n_k = len(k_values)
    results = np.zeros(n_k, dtype=np.complex128)
    
    for i in prange(n_k):
        n_R = len(R)
        integrand_vals = np.zeros(n_R, dtype=np.complex128)
        
        for j in range(n_R):
            R_val = R[j]
            fPhi = f_reg(R_val, rphi, mPhi)
            fPsi = f_reg(R_val, R0, mpsi)
            j0_val = j0_matrix[i, j]
            integrand_vals[j] = A_P * (1.0 - fPsi) * np.exp(-(chi**2) * fPhi) * j0_val * R_val
        
        val = trapezoidal_integral(integrand_vals, R)
        results[i] = 2.0 * np.pi * val
    
    return results

def compute_Pdir(k, integrand, Rmax, n_points=50000, Rmin=1e-8):
    R = np.logspace(np.log10(Rmin), np.log10(Rmax), n_points)
    integrand_values = integrand(R, k)
    val = trapezoidal_integral(integrand_values, R)
    return 2*np.pi * val


def main():
    Rmax = 1024*2#256.0

    rphi = 1#33.8
    chi_values = [0,0.25, 0.5,1,2,4]#, 8]#0.8, 2.0, 5.0]

    mpsi = 5.0/3.0
    R0   = 2#55.4
    A_P  = 1

    k_values = np.logspace(np.log10(0.1), np.log10(10000.0), 400)

    mPhi_values = [2]#[5.0/3.0, 1.0]
    colors_mPhi53 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors_mPhi1 = ['#e377c2', '#17becf', '#bcbd22', '#ff9896', '#c5b0d5', '#ffbb78']
    linestyles = ['-', '--']
    
    all_Pdir = []
    
    plt.figure(figsize=(7, 4.5))
    
    R = np.logspace(np.log10(1e-5), np.log10(Rmax), 500000)
    
    kR_matrix = np.outer(k_values, R)
    j0_matrix = j0(kR_matrix)
    
    all_k_plotted = []
    all_Pdir_plotted = []
    
    for i, mPhi in enumerate(mPhi_values):
        for j, chi in enumerate(chi_values):
            Pdir = compute_Pdir_parallel(k_values, R, j0_matrix, mPhi, rphi, chi, mpsi, R0, A_P)
            all_Pdir.append(Pdir)
            
            if i == 0:
                color = colors_mPhi53[j]
            else:
                color = colors_mPhi1[j]
            
            if chi < 1.1:
                k_mask = k_values <= 50
            elif chi<2.5:
                k_mask = k_values <= 70
            else:
                k_mask = np.ones(len(k_values), dtype=bool)
            
            k_plot = k_values[k_mask]
            Pdir_plot = np.abs(Pdir[k_mask])
            valid = (np.isfinite(Pdir_plot)) & (Pdir_plot > 0)
            
            if np.any(valid):
                all_k_plotted.extend(k_plot[valid])
                all_Pdir_plotted.extend(Pdir_plot[valid])
            
            plt.loglog(k_plot[valid], Pdir_plot[valid], color=color, 
                      linestyle=linestyles[i], lw=2, alpha=0.8,
                      label=rf"$\chi={chi}$")#$m_\psi={mpsi:.1f}$, $m_\Phi={mPhi:.1f}$, 
            
            # Calculate and plot k_\times(chi) point
            if chi > 0:  # Skip chi=0 as it would give infinite k_\times
                k_cross = ((A_P * R0**mpsi) / (chi**2 * rphi**mPhi))**(1.0 / (mpsi - mPhi))
                # Find the index in k_values closest to k_cross
                idx_cross = np.argmin(np.abs(k_values - k_cross))
                k_cross_actual = k_values[idx_cross]
                Pdir_cross = np.abs(Pdir[idx_cross])
                
                # Only plot if the point is valid and within the plotted range
                if np.isfinite(Pdir_cross) and Pdir_cross > 0:
                    plt.loglog(k_cross_actual, Pdir_cross, 'o', color=color, 
                              markersize=8, markeredgecolor='black', markeredgewidth=1.5,
                              zorder=10, alpha=0.9)

    k0 = 5.0
    i0 = np.argmin(np.abs(k_values - k0))
    mask_k_gt_1 = k_values > 600
    mask_k_gt_2 = (k_values > 10) & (k_values < 50)
    
    for i, mPhi in enumerate(mPhi_values):
        slope_ref = -(mPhi + 2.0)
        Pdir_ref = all_Pdir[i * len(chi_values)]
        ref = np.abs(Pdir_ref[i0]) * (k_values / k_values[i0])**(slope_ref)*150
        ref_plot = ref[mask_k_gt_1]
        k_ref_plot = k_values[mask_k_gt_1]
        valid_ref = (np.isfinite(ref_plot)) & (ref_plot > 0)
        
        if np.any(valid_ref):
            all_k_plotted.extend(k_ref_plot[valid_ref])
            all_Pdir_plotted.extend(ref_plot[valid_ref])
        
        plt.loglog(k_ref_plot[valid_ref], ref_plot[valid_ref], "-.", color="black", lw=1.5, alpha=1,
                   label=rf"$k^{{{slope_ref:.2f}}}$" if i == 0 else "")
    
    slope_11_3 = -11.0/3.0
    Pdir_ref_11_3 = all_Pdir[0]
    ref_11_3 = np.abs(Pdir_ref_11_3[i0]) * (k_values / k_values[i0])**(slope_11_3)*0.2
    ref_11_3_plot = ref_11_3[mask_k_gt_2]
    k_ref_11_3_plot = k_values[mask_k_gt_2]
    valid_ref_11_3 = (np.isfinite(ref_11_3_plot)) & (ref_11_3_plot > 0)
    
    if np.any(valid_ref_11_3):
        all_k_plotted.extend(k_ref_11_3_plot[valid_ref_11_3])
        all_Pdir_plotted.extend(ref_11_3_plot[valid_ref_11_3])
    
    plt.loglog(k_ref_11_3_plot[valid_ref_11_3], ref_11_3_plot[valid_ref_11_3], "--", color="green", lw=1.5, alpha=1,
               label=r"$k^{-11/3}$")

    # Plot k^(-2-mPhi) reference line normalized at k=20
    k_norm = 20.0
    i_norm = np.argmin(np.abs(k_values - k_norm))
    for i, mPhi in enumerate(mPhi_values):
        slope_ref_20 = -(2.0 + mPhi)
        # Use the first chi=0 result for normalization
        Pdir_norm = all_Pdir[i * len(chi_values)]
        norm_value = np.abs(Pdir_norm[i_norm])
        ref_20 = norm_value * (k_values / k_norm)**(slope_ref_20)
        # Plot over a reasonable k range
        mask_k_ref_20 = (k_values > 1) & (k_values < 200)
        ref_20_plot = ref_20[mask_k_ref_20]
        k_ref_20_plot = k_values[mask_k_ref_20]
        valid_ref_20 = (np.isfinite(ref_20_plot)) & (ref_20_plot > 0)
        
        if np.any(valid_ref_20):
            all_k_plotted.extend(k_ref_20_plot[valid_ref_20])
            all_Pdir_plotted.extend(ref_20_plot[valid_ref_20])
        
        plt.loglog(k_ref_20_plot[valid_ref_20], ref_20_plot[valid_ref_20], 
                  ":", color="red", lw=1.5, alpha=1,
                  label=rf"$k^{{{slope_ref_20:.2f}}}$" if i == 0 else "")

    if len(all_k_plotted) > 0 and len(all_Pdir_plotted) > 0:
        k_min = np.min(all_k_plotted)
        k_max = np.max(all_k_plotted)
        Pdir_min = np.min(all_Pdir_plotted)
        Pdir_max = np.max(all_Pdir_plotted)
        plt.xlim(k_min, k_max)
        plt.ylim(Pdir_min, Pdir_max)

    plt.xlabel(r"$k$")
    plt.ylabel(r"$|P_{\rm dir}(k)|$")
    # plt.title(r"$P_{\rm dir}(k)=2\pi\int_{0}^{R_{\max}}A_P\left(1 - f_\Psi(R)\right)e^{-\chi^2 f_\Phi(R)}J_0(kR) R  dR$", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("henkel.png", dpi=300, bbox_inches="tight")
    plt.savefig("henkel.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    main()