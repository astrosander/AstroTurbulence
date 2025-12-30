import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# Publication-ready font sizes
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 22

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


def plot_one_screen(screen_type, params, outfile_pdf):
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
        title = "Thick Faraday screen ($L > r_\\phi$), separated regions"
        
        R_match1 = 0.1 * r_phi
        R_match2 = L#np.sqrt(r_phi * L)
        R_match3 = Rmax#10.0 * L
    else:
        D1, D2, D3 = D_asym_thin(R, L=L, r_phi=r_phi, m_phi=m_phi, sigma2=sigma2)
        m1 = R < L
        m2 = (R >= L) & (R < r_phi)
        m3 = R >= r_phi
        labels = [r"Eq. (151)", r"Eq. (152)", r"Eq. (153)"]
        title = "Thin Faraday screen ($L < r_\\phi$), separated regions"
        
        R_match1 = 0.1 * L
        R_match2 = np.sqrt(L * r_phi)
        R_match3 = Rmax#10.0 * r_phi
    
    R_match1 = np.clip(R_match1, Rmin, Rmax)
    R_match2 = np.clip(R_match2, Rmin, Rmax)
    R_match3 = np.clip(R_match3, Rmin, Rmax)
    
    if screen_type == "thick":
        fit1 = (R < 0.3*r_phi)
        fit2 = (R > 0.2*L) & (R < 0.3*L)
        fit3 = (R > 50*L)
    else:
        fit1 = (R < 0.3*L)
        fit2 = (R > 3*L) & (R < 0.3*r_phi)
        fit3 = (R > 100*r_phi)
    
    def fit_norm(Dnum, Dasym, mask):
        sel = mask & np.isfinite(Dnum) & np.isfinite(Dasym) & (Dnum > 0) & (Dasym > 0)
        if sel.sum() < 10:
            return 1.0
        return np.exp(np.mean(np.log(Dnum[sel]) - np.log(Dasym[sel])))
    
    norm1 = fit_norm(Dnum, D1, fit1)
    norm2 = fit_norm(Dnum, D2, fit2)
    norm3 = fit_norm(Dnum, D3, fit3)
    
    D1 = D1 * norm1
    D2 = D2 * norm2
    D3 = D3 * norm3

    y1 = xiP_from_D(Xi, D1, lam=lam)
    y2 = xiP_from_D(Xi, D2, lam=lam)
    y3 = xiP_from_D(Xi, D3, lam=lam)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.0))
    
    ax = ax1

    ax.loglog(x, y_num, "o", ms=3.0, color="k", label="Numerical (Eq. 146 integral)")

    colors = ["C0", "C2", "C1"]
    for y, mask, c, lab in zip([y1, y2, y3], [m1, m2, m3], colors, labels):
        ax.loglog(x, y, ls="--", lw=2.0, color=c, alpha=0.35)
        ax.loglog(x[mask], y[mask], ls="-", lw=2.2, color=c, label=lab)

    ax.loglog(x, Xi, lw=2.2, color="0.55", ls=":", label=r"$\Xi_i(R)$ (Eq. 144)")
    ax.axvline(r_phi / r_i, color="0.6", lw=2.0, ls="--")
    ax.axvline(L / r_i, color="0.4", lw=2.0, ls="-.")

    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\xi_P(R)$")
    ax.set_title(title)

    y_all = np.concatenate([y_num, y1, y2, y3, Xi])
    y_pos = y_all[np.isfinite(y_all) & (y_all > 0)]
    ymin = max(1e-12, y_pos.min() * 0.5)
    ymax = min(2.0, y_pos.max() * 1.2)

    # ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=14)
    ax.set_ylim(1e-12, ymax*10)
    ax.set_xlim(min(x), max(x))
    
    ax.text(r_phi / r_i, ax.get_ylim()[1]*0.6, r"$r_\phi$", rotation=90,
            va="top", ha="right",  fontsize=24)
    ax.text(L / r_i, ax.get_ylim()[1]*0.6, r"$L$", rotation=90,
            va="top", ha="right", fontsize=24)
    
    ax = ax2
    ax.loglog(x, Dnum, "o", ms=3.0, color="k", label=r"$D_{\Delta\Phi}$ Numerical (Eq. 146)")
    
    colors = ["C0", "C2", "C1"]
    for D, mask, c, lab in zip([D1, D2, D3], [m1, m2, m3], colors, labels):
        ax.loglog(x, D, ls="--", lw=2.0, color=c, alpha=0.35)
        ax.loglog(x[mask], D[mask], ls="-", lw=2.2, color=c, label=lab)
    
    ax.axvline(r_phi / r_i, color="0.6", lw=2.0, ls="--")
    ax.axvline(L / r_i, color="0.4", lw=2.0, ls="-.")
    
    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$D_{\Delta\Phi}(R)$")
    ax.set_title(r"$D_{\Delta\Phi}$ comparison")
    
    D_all = Dnum#np.concatenate([Dnum, D1, D2, D3])
    D_pos = D_all[np.isfinite(D_all) & (D_all > 0)]
    Dmin = max(1e-10, D_pos.min() * 0.5)
    Dmax = D_pos.max() * 3
    
    # ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=16)
    ax.set_ylim(Dmin, Dmax)
    ax.set_xlim(min(x), max(x))
    
    ax.text(r_phi / r_i, ax.get_ylim()[1]*0.6, r"$r_\phi$", rotation=90,
            va="top", ha="right",  fontsize=24)
    ax.text(L / r_i, ax.get_ylim()[1]*0.6, r"$L$", rotation=90,
            va="top", ha="right",  fontsize=24)
    
    fig.tight_layout()
    fig.savefig(outfile_pdf, dpi=160)
    print(f"Saved: {outfile_pdf}")
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
        r_phi=50.0,
        L=10000.0,
        lam=1.0,
        sigma_phi2=0.0000004,
    )

    params_thin = dict(base)
    params_thin.update(
        r_phi=1000.0,
        L=10.0,
        lam=1.5,
        sigma_phi2=0.008,
    )

    plot_one_screen("thin",  params_thin,  "xiP_separated_thin.pdf")
    plot_one_screen("thick", params_thick, "xiP_separated_thick.pdf")
    plot_one_screen("thin",  params_thin,  "xiP_separated_thin.png")
    plot_one_screen("thick", params_thick, "xiP_separated_thick.png")


if __name__ == "__main__":
    main()
