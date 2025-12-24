import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, types
from numba.extending import register_jitable
from scipy.special import j0

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
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

def compute_Pdir(k, integrand, Rmax, n_points=50000, Rmin=1e-6):
    R = np.logspace(np.log10(Rmin), np.log10(Rmax), n_points)
    integrand_values = integrand(R, k)
    val = trapezoidal_integral(integrand_values, R)
    return 2*np.pi * val


def main():
    Rmax = 1024*2#256.0

    rphi = 0.3#33.8
    chi_values = [0, 1, 2, 4]#0.8, 2.0, 5.0]

    mpsi = 5.0/3.0
    R0   = 1#55.4
    A_P  = 1

    k_values = np.logspace(np.log10(0.1), np.log10(500.0), 400)

    mPhi_values = [1]#[5.0/3.0, 1.0]
    colors_mPhi53 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors_mPhi1 = ['#e377c2', '#17becf', '#bcbd22', '#ff9896', '#c5b0d5', '#ffbb78']
    linestyles = ['-', '--']
    
    all_Pdir = []
    
    plt.figure(figsize=(7*1.2, 4.5*1.2))
    
    R = np.logspace(np.log10(1e-5), np.log10(Rmax), 500000)
    
    kR_matrix = np.outer(k_values, R)
    j0_matrix = j0(kR_matrix)
    
    for i, mPhi in enumerate(mPhi_values):
        for j, chi in enumerate(chi_values):
            Pdir = compute_Pdir_parallel(k_values, R, j0_matrix, mPhi, rphi, chi, mpsi, R0, A_P)
            all_Pdir.append(Pdir)
            
            if i == 0:
                color = colors_mPhi53[j]
            else:
                color = colors_mPhi1[j]
            
            plt.loglog(k_values, np.abs(Pdir), color=color, 
                      linestyle=linestyles[i], lw=2, alpha=0.8,
                      label=rf"$\chi={chi}$")#$m_\psi={mpsi:.1f}$, $m_\Phi={mPhi:.1f}$, 

    k0 = 5.0
    i0 = np.argmin(np.abs(k_values - k0))
    mask_k_gt_1 = k_values > 10
    mask_k_gt_2 = (k_values > 10) & (k_values < 50)
    
    for i, mPhi in enumerate(mPhi_values):
        slope_ref = -(mPhi + 2.0)
        Pdir_ref = all_Pdir[i * len(chi_values)]
        ref = np.abs(Pdir_ref[i0]) * (k_values / k_values[i0])**(slope_ref)*50
        plt.loglog(k_values[mask_k_gt_1], ref[mask_k_gt_1], "-.", color="black", lw=1.5, alpha=1,
                   label=rf"$k^{{{slope_ref:.2f}}}$" if i == 0 else "")
    
    slope_11_3 = -11.0/3.0
    Pdir_ref_11_3 = all_Pdir[0]
    ref_11_3 = np.abs(Pdir_ref_11_3[i0]) * (k_values / k_values[i0])**(slope_11_3)*0.2
    plt.loglog(k_values[mask_k_gt_2], ref_11_3[mask_k_gt_2], "--", color="green", lw=1.5, alpha=1,
               label=r"$k^{-11/3}$")

    valid_mask = np.ones(len(k_values), dtype=bool)
    for Pdir in all_Pdir:
        valid_mask &= (np.isfinite(np.abs(Pdir)) & (np.abs(Pdir) > 0))
    
    if np.any(valid_mask):
        k_min = k_values[valid_mask].min()
        k_max = k_values[valid_mask].max()
        plt.xlim(k_min, k_max)

    plt.xlabel(r"$k$")
    plt.ylabel(r"$|P_{\rm dir}(k)|$")
    plt.title(r"$P_{\rm dir}(k)=2\pi\int_{0}^{R_{\max}}A_P\left(1 - f_\Psi(R)\right)e^{-\chi^2 f_\Phi(R)}J_0(kR) R  dR$", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("henkel.png", dpi=300, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    main()
