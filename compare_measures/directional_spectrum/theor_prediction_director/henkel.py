import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
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


def f_reg(R, r0, m):
    x = (R / r0)**m
    return x / (1.0 + x)

def make_integrand(mPhi=5/3, rphi=33.8, chi=5.0,
                   mpsi=5/3, R0=55.4, A_P=0.3):
    def integrand(R, k):
        fPhi = f_reg(R, rphi, mPhi)
        fPsi = f_reg(R, R0,  mpsi)

        Re_xiP = 1.0 - A_P * fPsi
        Cdir   = Re_xiP * np.exp(-(chi**2) * fPhi)

        Cinf = (1.0 - A_P) * np.exp(-(chi**2))

        return (Cdir - Cinf) * j0(k * R) * R

    return integrand

def compute_Pdir(k, integrand, Rmax):
    val, err = quad(integrand, 0.0, Rmax, args=(k,), limit=600)
    return 2*np.pi * val


def main():
    Rmax = 256.0

    rphi = 0.8#33.8
    chi_values = [0.8, 2.0, 5.0]

    mpsi = 5.0/3.0
    R0   = 1#55.4
    A_P  = 1

    k_values = np.logspace(np.log10(0.1), np.log10(100.0), 400)

    mPhi_values = [1]#[5.0/3.0, 1.0]
    colors_mPhi53 = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_mPhi1 = ['#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--']
    
    all_Pdir = []
    
    plt.figure(figsize=(7, 4.5))
    
    for i, mPhi in enumerate(mPhi_values):
        for j, chi in enumerate(chi_values):
            integrand = make_integrand(mPhi=mPhi, rphi=rphi, chi=chi,
                                      mpsi=mpsi, R0=R0, A_P=A_P)
            Pdir = np.array([compute_Pdir(k, integrand, Rmax) for k in k_values])
            all_Pdir.append(Pdir)
            
            if i == 0:
                color = colors_mPhi53[j]
            else:
                color = colors_mPhi1[j]
            
            plt.loglog(k_values, np.abs(Pdir), color=color, 
                      linestyle=linestyles[i], lw=2, alpha=0.8,
                      label=rf"$m_\psi={mpsi:.1f}$, $m_\Phi={mPhi:.1f}$, $\chi={chi}$")

    k0 = 5.0
    i0 = np.argmin(np.abs(k_values - k0))
    
    for i, mPhi in enumerate(mPhi_values):
        slope_ref = -(mPhi + 2.0)
        Pdir_ref = all_Pdir[i * len(chi_values)]
        ref = np.abs(Pdir_ref[i0]) * (k_values / k_values[i0])**(slope_ref)
        plt.loglog(k_values, ref, "-.", color="black", lw=1.5, alpha=0.5,
                   label=rf"$k^{{{slope_ref:.2f}}}$" if i == 0 else "")

    valid_mask = np.ones(len(k_values), dtype=bool)
    for Pdir in all_Pdir:
        valid_mask &= (np.isfinite(np.abs(Pdir)) & (np.abs(Pdir) > 0))
    
    if np.any(valid_mask):
        k_min = k_values[valid_mask].min()
        k_max = k_values[valid_mask].max()
        plt.xlim(k_min, k_max)

    plt.xlabel(r"$k$")
    plt.ylabel(r"$|P_{\rm dir}(k)|$")
    plt.title(r"$P_{\rm dir}(k)=2\pi\int_{0}^{R_{\max}}\left[\left(1 - A_P f_\Psi(R)\right)e^{-\chi^2 f_\Phi(R)}-(1 - A_P) e^{-\chi^2}\right]J_0(kR) R  dR$", fontsize=16)#$f_\Phi(R)=\frac{(R/r_\phi)^{m_\Phi}}{1 + (R/r_\phi)^{m_\Phi}},\qquadf_\Psi(R)=\frac{(R/R_0)^{m_\Psi}}{1 + (R/R_0)^{m_\Psi}}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("henkel.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
