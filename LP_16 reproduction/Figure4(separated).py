import numpy as np
import matplotlib.pyplot as plt


def xi_sat_3d(R, z, r0, m, sigma2=1.0):
    rr = np.sqrt(R * R + z * z)
    return sigma2 * (r0**m) / (r0**m + rr**m)


def X_i_projected(R, R_i=1.0, M_i=2/3, X0=1.0):
    return X0 / (1.0 + (R / R_i) ** M_i)


def D_foreground(R_array, L, r_phi, m_phi, Nz=4000):
    R_array = np.atleast_1d(R_array).astype(float)

    z = np.linspace(0.0, L, Nz)

    xi0 = xi_sat_3d(0.0, z, r_phi, m_phi, sigma2=1.0)[None, :]
    xiR = xi_sat_3d(R_array[:, None], z[None, :], r_phi, m_phi, 1.0)

    integrand = 2.0 * (L - z)[None, :] * (xi0 - xiR)
    D = np.trapz(integrand, z, axis=1)
    return D


def xiP_separated(R_array, *,
                  R_i=1.0, M_i=2/3, X0=1.0,
                  L=100.0, r_phi=0.1, m_phi=2/3,
                  Lsf=0.07,
                  Nz=4000):
    R_array = np.atleast_1d(R_array).astype(float)
    Xi = X_i_projected(R_array, R_i=R_i, M_i=M_i, X0=X0)
    Df = D_foreground(R_array, L=L, r_phi=r_phi, m_phi=m_phi, Nz=Nz)
    return Xi * np.exp(-2.0 * Df / (Lsf * Lsf))


def main():
    r_i = 1.0
    Lsf = 0.07 * r_i
    L_screen = 100.0 * r_i
    m_phi = 2/3

    R = np.logspace(-3, 3, 70) * r_i

    r_phi_left = 0.1 * r_i
    M_left = 2/3
    R_i_left = r_i

    xiP_left = xiP_separated(R,
                             R_i=R_i_left, M_i=M_left, X0=1.0,
                             L=L_screen, r_phi=r_phi_left, m_phi=m_phi,
                             Lsf=Lsf, Nz=5000)

    Xi_left = X_i_projected(R, R_i=R_i_left, M_i=M_left, X0=1.0)
    Df_left = D_foreground(R, L=L_screen, r_phi=r_phi_left, m_phi=m_phi, Nz=5000)
    damp_left = np.exp(-2.0 * Df_left / (Lsf * Lsf))

    tail_left = (R / R_i_left) ** (-M_left)

    configs = [
        ("black", r_i,     2/3, "-", "rφ = ri,   M=2/3"),
        ("black", r_i,     1/2, ":", "rφ = ri,   M=1/2"),
        ("tab:blue", 0.1*r_i, 2/3, "-", "rφ = 0.1 ri, M=2/3"),
        ("tab:blue", 0.1*r_i, 1/2, ":", "rφ = 0.1 ri, M=1/2"),
    ]

    right_curves = []
    for color, rphi, Mi, ls, label in configs:
        y = xiP_separated(R,
                          R_i=r_i, M_i=Mi, X0=1.0,
                          L=L_screen, r_phi=rphi, m_phi=m_phi,
                          Lsf=Lsf, Nz=4000)
        right_curves.append((color, ls, label, y))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    ax = axs[0]
    ax.loglog(R / r_i, xiP_left, "k.", ms=5, label=r"$\xi_P(R)$ (Eq.145–146, numeric)")
    ax.loglog(R / r_i, Xi_left, color="green", lw=2, label=r"$X_i(R)$ (Eq.144)")
    ax.loglog(R / r_i, damp_left * Xi_left[0], color="steelblue", lw=2,
              label=r"$X_i(0)\,e^{-2D^F/L_{\sigma\phi}^2}$ (Eq.145–146)")
    ax.loglog(R / r_i, tail_left * 0.2, color="orange", lw=2,
              label=rf"ref. tail $\propto R^{{-{M_left}}}$ (from Eq.144)")

    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\xi_P(R)$")
    ax.set_title(r"Separated regions (Appendix C): $L_{\sigma\phi}=0.07\,r_i,\ r_\phi=0.1\,r_i$")
    ax.set_ylim(1e-6, 2e0)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    ax = axs[1]
    for color, ls, label, y in right_curves:
        ax.loglog(R / r_i, y, color=color, ls=ls, lw=2)

    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\xi_P(R)$")
    ax.set_title(r"Appendix C comparison: vary $M$ and $r_\phi$ (same style as Fig.4)")
    ax.set_ylim(1e-6, 2e0)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure4_style_separated_regions.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
