import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import j0

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

    mPhi = 5.0/3.0
    rphi = 33.8
    chi  = 5.0

    mpsi = 5.0/3.0
    R0   = 55.4
    A_P  = 1

    integrand = make_integrand(mPhi=mPhi, rphi=rphi, chi=chi,
                               mpsi=mpsi, R0=R0, A_P=A_P)

    k_values = np.logspace(np.log10(0.1), np.log10(100.0), 400)
    Pdir = np.array([compute_Pdir(k, integrand, Rmax) for k in k_values])

    plt.figure(figsize=(7, 4.5))
    plt.loglog(k_values, np.abs(Pdir), color="blue", lw=2,
               label=rf"$P_{{\rm dir}}(k)$ (Hankel), $\chi={chi}$")

    slope_ref = -(mPhi + 2.0)
    k0 = 5.0
    i0 = np.argmin(np.abs(k_values - k0))
    ref = np.abs(Pdir[i0]) * (k_values / k_values[i0])**(slope_ref)
    plt.loglog(k_values, ref, "-.", color="black", lw=1.5, alpha=0.7,
               label=rf"$k^{{{slope_ref:.2f}}}$ (ref)")

    plt.xlabel(r"$k$")
    plt.ylabel(r"$|P_{\rm dir}(k)|$")
    plt.title(r"Hankel transform of $C_{\rm dir}(R)=\Re\xi_P(R)\,e^{-\chi^2 f_\Phi(R)}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("henkel.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
