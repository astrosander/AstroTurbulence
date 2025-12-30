import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# Publication-ready font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 14

def Xi_i(R, Xi0=1.0, Ri=1.0, Mi=2/3):
    return Xi0 / (1.0 + (R / Ri) ** Mi)


def xi_phi(R, dz, sigma2=1.0, r_phi=10.0, m_phi=2/3):
    r = np.sqrt(R * R + dz * dz)
    return sigma2 / (1.0 + (r / r_phi) ** m_phi)


def D_DeltaPhi_numeric(R, L, r_phi, m_phi, sigma2=1.0, Nu=5000):
    R = np.atleast_1d(R).astype(float)

    Nu_log = int(0.35 * Nu)
    Nu_lin = Nu - Nu_log
    u_min = 1e-8
    u_log = np.geomspace(u_min, 1e-2, Nu_log)
    u_lin = np.linspace(1e-2, 1.0, Nu_lin)
    u = np.unique(np.concatenate(([0.0], u_log, u_lin)))
    dz = L * u

    xi0 = xi_phi(0.0, dz, sigma2=sigma2, r_phi=r_phi, m_phi=m_phi)[None, :]
    xiR = xi_phi(R[:, None], dz[None, :], sigma2=sigma2, r_phi=r_phi, m_phi=m_phi)

    integrand = 2.0 * (L - dz)[None, :] * (xi0 - xiR)
    return np.trapz(integrand, dz, axis=1)


def D_asym_thick(R, L, r_phi, m_phi, sigma2=1.0):
    mtil = min(m_phi, 1.0)
    D148 = sigma2 * L * R * (R / r_phi) ** mtil
    D149 = sigma2 * L * R * (R / r_phi) ** (-mtil)
    D150 = sigma2 * (L ** 2) * (L / r_phi) ** (-mtil) * np.ones_like(R)
    return D148, D149, D150


def D_asym_thin(R, L, r_phi, m_phi, sigma2=1.0):
    mtil = min(m_phi, 1.0)
    D151 = sigma2 * L * R * (R / r_phi) ** mtil
    D152 = sigma2 * (L ** 2) * (R / r_phi) ** mtil
    D153 = sigma2 * (L ** 2) * np.ones_like(R)
    return D151, D152, D153


def xiP_from_D(Xi, D, lam=1.0, clip_exp=(-700.0, 50.0)):
    expo = -4.0 * (lam ** 4) * D
    expo = np.clip(expo, clip_exp[0], clip_exp[1])
    return Xi * np.exp(expo)


def plot_one_screen(screen_type, params, outfile_png):
    r_i = params["r_i"]
    Ri = params["Ri"]
    Mi = params["Mi"]
    Xi0 = params.get("Xi0", 1.0)

    r_phi = params["r_phi"]
    L = params["L"]
    m_phi = params["m_phi"]
    sigma2 = params["sigma_phi2"]
    lam = params["lam"]

    Rmin = params.get("Rmin", 1e-3) * r_i
    Rmax = params.get("Rmax", 1e3) * r_i
    NR = params.get("NR", 500)

    R = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)
    x = R / r_i

    Xi = Xi_i(R, Xi0=Xi0, Ri=Ri, Mi=Mi)

    Dnum = D_DeltaPhi_numeric(R, L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2, Nu=6000)

    y_num = xiP_from_D(Xi, Dnum, lam=lam)

    if screen_type == "thick":
        D1, D2, D3 = D_asym_thick(R, L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        m1 = R < r_phi
        m2 = (R >= r_phi) & (R < L)
        m3 = R >= L
        labels = [r"Eq. (148)", r"Eq. (149)", r"Eq. (150)"]
        title = "Thick Faraday screen ($L > r_\\phi$), separated regions (Appendix C)"
        
        R_match1 = Rmin
        R_match2 = r_phi
        R_match3 = Rmax
    else:
        D1, D2, D3 = D_asym_thin(R, L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        m1 = R < L
        m2 = (R >= L) & (R < r_phi)
        m3 = R >= r_phi
        labels = [r"Eq. (151)", r"Eq. (152)", r"Eq. (153)"]
        title = "Thin Faraday screen ($L < r_\\phi$), separated regions (Appendix C)"
        
        R_match1 = Rmin
        R_match2 = L
        R_match3 = Rmax
    
    Dnum_match1 = D_DeltaPhi_numeric(np.array([R_match1]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2, Nu=6000)[0]
    Dnum_match2 = D_DeltaPhi_numeric(np.array([R_match2]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2, Nu=6000)[0]
    Dnum_match3 = D_DeltaPhi_numeric(np.array([R_match3]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2, Nu=6000)[0]
    
    if screen_type == "thick":
        D1_asym_match, _, _ = D_asym_thick(np.array([R_match1]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        _, D2_asym_match, _ = D_asym_thick(np.array([R_match2]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        _, _, D3_asym_match = D_asym_thick(np.array([R_match3]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
    else:
        D1_asym_match, _, _ = D_asym_thin(np.array([R_match1]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        _, D2_asym_match, _ = D_asym_thin(np.array([R_match2]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        _, _, D3_asym_match = D_asym_thin(np.array([R_match3]), L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
    
    norm1 = Dnum_match1 / D1_asym_match[0] if D1_asym_match[0] > 0 else 1.0
    norm2 = Dnum_match2 / D2_asym_match[0] if D2_asym_match[0] > 0 else 1.0
    norm3 = Dnum_match3 / D3_asym_match[0] if D3_asym_match[0] > 0 else 1.0
    
    D1 = D1 * norm1
    D2 = D2 * norm2
    D3 = D3 * norm3

    y1 = xiP_from_D(Xi, D1, lam=lam)
    y2 = xiP_from_D(Xi, D2, lam=lam)
    y3 = xiP_from_D(Xi, D3, lam=lam)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.loglog(x, y_num, "o", ms=3.0, color="k", label="Numerical (Eq. 146 integral)")

    colors = ["C0", "C2", "C1"]
    for y, mask, c, lab in zip([y1, y2, y3], [m1, m2, m3], colors, labels):
        ax.loglog(x, y, ls="--", lw=1.0, color=c, alpha=0.35)
        # ax.loglog(x[mask], y[mask], ls="-", lw=2.2, color=c, label=lab)

    ax.loglog(x, Xi, lw=1.2, color="0.55", ls=":", label=r"$\Xi_i(R)$ (Eq. 144)")
    ax.axvline(r_phi / r_i, color="0.6", lw=1.0, ls="--")
    ax.axvline(L / r_i, color="0.4", lw=1.0, ls="-.")

    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\xi_P(R)$")
    ax.set_title(title)

    y_all = np.concatenate([y_num, y1, y2, y3, Xi])
    y_pos = y_all[np.isfinite(y_all) & (y_all > 0)]
    ymin = max(1e-12, y_pos.min() * 0.5)
    ymax = min(2.0, y_pos.max() * 1.2)

    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(1e-12, ymax*10)
    ax.set_xlim(min(x), max(x))
    
    ax.text(r_phi / r_i, ax.get_ylim()[1]*0.6, r"$r_\phi$", rotation=90,
            va="top", ha="right", color="0.35", fontsize=12)
    ax.text(L / r_i, ax.get_ylim()[1]*0.6, r"$L$", rotation=90,
            va="top", ha="right", color="0.35", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile_png, dpi=160)
    print(f"Saved: {outfile_png}")
    # plt.show()


def main():
    base = dict(
        r_i=1.0,
        Xi0=1.0,
        Ri=1.0,
        Mi=2/3,
        m_phi=2/3,
        Rmin=1e-1,
        Rmax=1e6,
        NR=516,
    )

    params_thick = dict(base)
    params_thick.update(
        r_phi=10.0,
        L=1000.0,
        lam=1.0,
        sigma_phi2=0.00003,
    )

    params_thin = dict(base)
    params_thin.update(
        r_phi=100.0,
        L=3.0,
        lam=1.5,
        sigma_phi2=0.1,
    )

    plot_one_screen("thin",  params_thin,  "xiP_separated_thin.png")
    plot_one_screen("thick", params_thick, "xiP_separated_thick.png")


if __name__ == "__main__":
    main()
