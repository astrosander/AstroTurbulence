import numpy as np
import matplotlib.pyplot as plt

def xi_sat(R, dz, r0, m, sigma2=1.0):
    r = np.sqrt(R*R + dz*dz)
    return sigma2 * (r0**m) / (r0**m + r**m)

def cumulative_trapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*(x[1:]-x[:-1]))
    return out

def xiP_numeric(R,
                r_i=1.0, m=2/3,
                r_phi=0.1, m_phi=2/3,
                L_sf=0.07, L=100.0,
                Nz=450, Nd=300):

    zmin = L_sf/200.0
    zplus = np.concatenate(([0.0], np.logspace(np.log10(zmin), np.log10(L/2), Nz-1)))

    xi0 = xi_sat(0.0, zplus, r_phi, m_phi, sigma2=1.0)
    xiR = xi_sat(R,   zplus, r_phi, m_phi, sigma2=1.0)

    delta = xi0 - xiR

    I1 = cumulative_trapz(delta, zplus)
    I2 = cumulative_trapz(delta*zplus, zplus)
    Dplus = 2.0*(zplus*I1 - I2)

    xiR0 = xi_sat(R, 0.0, r_phi, m_phi, sigma2=1.0)
    Lambda = xi0 - xiR + 2.0*xiR0

    outer = np.zeros_like(zplus)
    for k, zp in enumerate(zplus):
        if zp == 0.0:
            outer[k] = 0.0
            continue

        lam = Lambda[k]
        dz_lim = 2.0*zp
        dz = np.linspace(0.0, dz_lim, Nd)

        xi_i = xi_sat(R, dz, r_i, m, sigma2=1.0)
        w = np.exp(-lam*dz*dz/(2.0*L_sf*L_sf))
        inner = 2.0*np.trapz(xi_i*w, dz)

        outer[k] = np.exp(-2.0*Dplus[k]/(L_sf*L_sf)) * inner

    xiP = 2.0*np.trapz(outer, zplus)
    return xiP

def xiP_eq43(R, r_i, m, L_sf, amp=1.0):
    return amp * (L_sf**2) * xi_sat(R, 0.0, r_i, m, sigma2=1.0)

def xiP_eq42(R, r_i, m, r_phi, m_phi, L_sf, amp=1.0):
    mtilde = min(m_phi, 1.0)
    return amp * (L_sf**(2.0+mtilde)) * (r_phi**mtilde) * xi_sat(R, 0.0, r_i, m, sigma2=1.0) / (R**(1.0+mtilde))

def xiP_tail(R, r_i, m, L_sf, amp=1.0):
    return amp * (L_sf**2) * (r_i/R)**m

def main():

    r_i = 1.0
    L_sf = 0.07*r_i
    L = 100.0*r_i
    m_phi = 2/3

    Rvals = np.logspace(-3, 3, 60)

    r_phi_left = 0.1*r_i
    m_left = 2/3

    xi_num_left = np.array([xiP_numeric(R, r_i=r_i, m=m_left,
                                        r_phi=r_phi_left, m_phi=m_phi,
                                        L_sf=L_sf, L=L)
                            for R in Rvals])

    amp43 = np.median(xi_num_left[(Rvals>1) & (Rvals<100)] /
                      (L_sf**2 * xi_sat(Rvals[(Rvals>1)&(Rvals<100)], 0.0, r_i, m_left)))
    amp42 = 1.0

    xi42 = xiP_eq42(Rvals, r_i, m_left, r_phi_left, m_phi, L_sf, amp=amp42)
    xi43 = xiP_eq43(Rvals, r_i, m_left, L_sf, amp=amp43)
    xitail = xiP_tail(Rvals, r_i, m_left, L_sf, amp=amp43)

    configs = [
        ("black", r_i, 2/3, "-",  "rφ = ri, m=2/3"),
        ("black", r_i, 1/2, ":",  "rφ = ri, m=1/2"),
        ("tab:blue", 0.1*r_i, 2/3, "-", "rφ = 0.1 ri, m=2/3"),
        ("tab:blue", 0.1*r_i, 1/2, ":", "rφ = 0.1 ri, m=1/2"),
    ]

    right_curves = {}
    for color, rphi, mval, ls, label in configs:
        right_curves[label] = np.array([xiP_numeric(R, r_i=r_i, m=mval,
                                                   r_phi=rphi, m_phi=m_phi,
                                                   L_sf=L_sf, L=L)
                                        for R in Rvals])

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    ax = axs[0]
    ax.loglog(Rvals/r_i, xi_num_left, "k.", ms=5, label="Numerical (Eq.36 approx)")
    ax.loglog(Rvals/r_i, xi42, color="steelblue", lw=2, label="Eq.(42) asymptote")
    ax.loglog(Rvals/r_i, xi43, color="green", lw=2, label="Eq.(43) asymptote")
    ax.loglog(Rvals/r_i, xitail, color="orange", lw=2, label=r"Tail $\propto R^{-m}$")

    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\xi_P(R)$")
    ax.set_title(r"Strong turbulent FR: $\mathcal{L}_{\sigma_\phi}=0.07\,r_i,\ r_\phi=0.1\,r_i$")
    ax.set_ylim(1e-4, 1e1)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    ax = axs[1]
    for label, y in right_curves.items():
        if "0.1" in label:
            color = "tab:blue"
        else:
            color = "black"
        ls = "-" if "2/3" in label else ":"
        ax.loglog(Rvals/r_i, y, color=color, ls=ls, lw=2)

    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\xi_P(R)$")
    ax.set_title(r"Dependence on $m$ and $r_\phi$  (same $\mathcal{L}_{\sigma_\phi}$)")
    ax.set_ylim(1e-4, 1e1)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure4_reproduction.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
