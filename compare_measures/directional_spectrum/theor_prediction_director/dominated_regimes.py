import numpy as np
import matplotlib.pyplot as plt

def xi_i(R, Xi0=1.0, r_i=1.0, m_i=2/3):
    x = (R / r_i) ** m_i
    return Xi0 / (1.0 + x)

def xi_phi_local(R, dz, sigma_phi2=1.0, r_phi=10.0, m_phi=2/3):
    rr = np.sqrt(R * R + dz * dz)
    return sigma_phi2 / (1.0 + (rr / r_phi) ** m_phi)

def make_dz_grid(L, r_phi, Nu=6000):
    Nu = int(Nu)
    Nu_log = int(0.45 * Nu)
    Nu_lin = Nu - Nu_log

    dz_min = 1e-10 * L
    dz_knee = min(0.05 * L, 5.0 * r_phi, L)
    if dz_knee <= 0:
        dz_knee = 1e-6 * L

    dz_log = np.geomspace(dz_min, max(dz_knee, dz_min * 10), Nu_log)
    dz_lin = np.linspace(max(dz_knee, dz_min * 10), L, Nu_lin)
    dz = np.unique(np.concatenate(([0.0], dz_log, dz_lin)))
    return dz

def xi_DeltaPhi(R, L, r_phi, m_phi, sigma_phi2=1.0, Nu=6000):
    R = np.atleast_1d(R).astype(float)
    dz = make_dz_grid(L, r_phi, Nu=Nu)

    w = 2.0 * (L - dz)
    xiR = xi_phi_local(R[:, None], dz[None, :], sigma_phi2, r_phi, m_phi)
    xi0 = xi_phi_local(0.0, dz, sigma_phi2, r_phi, m_phi)[None, :]

    xi_DF = np.trapz(w[None, :] * xiR, dz, axis=1)
    xi_DF0 = np.trapz(w[None, :] * xi0, dz, axis=1)[0]

    D_DF = xi_DF0 - xi_DF
    return xi_DF, D_DF, xi_DF0

def xi_dP_dlam2(R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=6000):
    XiR = xi_i(R, Xi0=Xi0, r_i=r_i, m_i=m_i)
    xi_DF, D_DF, _ = xi_DeltaPhi(R, L, r_phi, m_phi, sigma_phi2, Nu=Nu)

    expo = -4.0 * (lam ** 4) * D_DF
    expo = np.clip(expo, -800.0, 80.0)
    return XiR * (xi_DF + (lam ** 4) * (D_DF ** 2)) * np.exp(expo), xi_DF, D_DF

def SF_from_corr(xiR):
    xi0 = np.real(xiR[0])
    return 2.0 * (xi0 - np.real(xiR))

def asymptotic_terms_derivative(screen_type, R, r_i, m_i, L, r_phi, m_phi):
    mtil = min(m_phi, 1.0)

    term_int = (R / r_i) ** m_i

    if screen_type == "thick":
        pref = (r_phi / L) ** (1.0 - mtil)
        term_far = pref * (R / r_phi) ** (1.0 + mtil)
        return term_int, term_far, None

    if screen_type == "thin":
        term_far1 = np.full_like(R, np.nan, dtype=float)
        term_far2 = np.full_like(R, np.nan, dtype=float)

        m1 = R < L
        m2 = (R >= L) & (R < r_phi)

        term_far1[m1] = (r_phi / L) * (R[m1] / r_phi) ** (1.0 + mtil)
        term_far2[m2] = (R[m2] / r_phi) ** (mtil)

        return term_int, term_far1, term_far2

    raise ValueError("screen_type must be 'thick' or 'thin'")

def normalize_asymptotic_to_numeric(R, y_num, y_asym, R_match):
    idx = np.argmin(np.abs(np.log(R) - np.log(R_match)))
    if y_asym[idx] <= 0 or not np.isfinite(y_asym[idx]) or not np.isfinite(y_num[idx]) or y_num[idx] <= 0:
        return y_asym, 1.0
    fac = y_num[idx] / y_asym[idx]
    if not np.isfinite(fac) or fac <= 0:
        return y_asym, 1.0
    result = y_asym * fac
    return result, fac

def plot_derivative_measure(screen_type, params, outfile):
    Xi0 = params.get("Xi0", 1.0)
    r_i = params["r_i"]
    m_i = params["m_i"]

    L = params["L"]
    r_phi = params["r_phi"]
    m_phi = params["m_phi"]
    sigma_phi2 = params["sigma_phi2"]

    lam = params["lam"]
    Nu = params.get("Nu", 6000)

    Rmin = params.get("Rmin", 1e-2) * r_i
    Rmax = params.get("Rmax", 1e4) * r_i
    NR = params.get("NR", 450)

    R = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)

    xiPp, xi_DF, D_DF = xi_dP_dlam2(
        R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=Nu
    )
    SF_num = SF_from_corr(xiPp)

    term_int, term_far, term_far2 = asymptotic_terms_derivative(screen_type, R, r_i, m_i, L, r_phi, m_phi)

    if screen_type == "thick":
        R_match = min(0.2 * r_phi, 5.0 * r_i)
        title = r"Separated screen, thick Faraday screen ($L>r_\phi$): $P'\equiv dP/d\lambda^2$"
    else:
        R_match = min(0.2 * L, 5.0 * r_i)
        title = r"Separated screen, thin Faraday screen ($L<r_\phi$): $P'\equiv dP/d\lambda^2$"

    Rmin_rel = params["Rmin"]
    Rmax_rel = params["Rmax"]

    Rmax_abs = Rmax_rel * r_i * 0.01
    Rmin_100 = 100.0 * Rmin_rel * r_i
    
    Rmax_abs = max(R[0], min(R[-1], Rmax_abs))
    Rmin_100 = max(R[0], min(R[-1], Rmin_100))

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.loglog(R[1:] / r_i, SF_num[1:], color="k", lw=2.2, label=r"Numerical from LP16 Eq. (159)")

    if screen_type == "thick":
        term_far_safe = np.where(np.isfinite(term_far), term_far, 0.0)
        SF_asym_raw = term_int + term_far_safe
        
        SF_asym, fac_asym = normalize_asymptotic_to_numeric(R, SF_num, SF_asym_raw, Rmin_100)
        SF_far, fac_far = normalize_asymptotic_to_numeric(R, SF_num, term_far_safe, Rmax_abs)
        SF_int = term_int * fac_asym

        ax.loglog(R[1:] / r_i, SF_asym[1:], ls="--", lw=2.0, color="C0",
                  label=r"Asymptotic sum (LP16 Eqs. 162–164, normalized)")
        ax.loglog(R[1:] / r_i, SF_int[1:], ls=":", lw=2.0, color="C2",
                  label=r"Intrinsic term $\propto (R/r_i)^{m_i}$")
        ax.loglog(R[1:] / r_i, SF_far[1:], ls="-.", lw=2.0, color="C1",
                  label=r"Faraday term (LP16 Eqs. 162–164)")
    else:
        term_far1_safe = np.where(np.isfinite(term_far), term_far, 0.0)
        term_far2_safe = np.where(np.isfinite(term_far2), term_far2, 0.0)
        SF_asym_raw = term_int + term_far1_safe + term_far2_safe
        
        mtil = min(m_phi, 1.0)
        R_norm_far1 = min(0.5 * L, Rmax_abs) if L > 0 else Rmax_abs
        R_norm_far2 = Rmax_abs
        R_norm_far1 = max(R[0], min(R[-1], R_norm_far1))
        R_norm_far2 = max(R[0], min(R[-1], R_norm_far2))
        
        SF_asym, fac_asym = normalize_asymptotic_to_numeric(R, SF_num, SF_asym_raw, Rmin_100)
        SF_int = term_int * fac_asym

        idx_far1 = np.argmin(np.abs(R - R_norm_far1))
        idx_far2 = np.argmin(np.abs(R - R_norm_far2))
        
        SF_norm_far1 = SF_num[idx_far1] if idx_far1 < len(SF_num) else SF_num[-1]
        SF_norm_far2 = SF_num[idx_far2] if idx_far2 < len(SF_num) else SF_num[-1]

        R_plot = R[1:] / r_i
        mask_far1 = R[1:] < L
        mask_far2 = (R[1:] >= L) & (R[1:] < r_phi)
        R_far1_abs = R[1:][mask_far1]
        R_far2_abs = R[1:][mask_far2]
        
        if len(R_far1_abs) > 0:
            slope_far1 = 1.0 + mtil
            SF_far1_line = SF_norm_far1 * (R_far1_abs / R_norm_far1) ** slope_far1
            ax.loglog(R_far1_abs / r_i, SF_far1_line, ls="-.", lw=2.0, color="C1",
                      label=r"Faraday term (R < L, LP16 Eq. 163)")
        
        if len(R_far2_abs) > 0:
            slope_far2 = mtil
            SF_far2_line = SF_norm_far2 * (R_far2_abs / R_norm_far2) ** slope_far2
            ax.loglog(R_far2_abs / r_i, SF_far2_line, ls="-.", lw=2.0, color="C3",
                      label=r"Faraday term (L < R < r_φ, LP16 Eq. 164)")

        ax.loglog(R[1:] / r_i, SF_asym[1:], ls="--", lw=2.0, color="C0",
                  label=r"Asymptotic sum (LP16 Eqs. 162–164, normalized)")
        ax.loglog(R[1:] / r_i, SF_int[1:], ls="-.", lw=2.0, color="C2",
                  label=r"Intrinsic term $\propto (R/r_i)^{m_i}$")

    ax.set_xlim(params["Rmin"]*100, params["Rmax"]*0.1)
    ax.set_ylim(1e-15, 1e-7)

    ax.set_title(title + rf"\n($\lambda={lam}$, $m_i={m_i}$, $m_\phi={m_phi}$)")
    ax.set_xlabel(r"$R/r_i$")
    ax.set_ylabel(r"$\left\langle\left|P'(X)-P'(X+R)\right|^2\right\rangle$")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")
    plt.show()

def main():
    params_thick = dict(
        Xi0=1.0,
        r_i=1000.0,
        m_i=2/3,
        L=200.0,
        r_phi=10.0,
        m_phi=2/3,
        sigma_phi2=5e-10,
        lam=100.0,
        Rmin=1e-12,
        Rmax=1e-3,
        NR=520,
        Nu=7000,
    )

    params_thin = dict(
        Xi0=1.0,
        r_i=1000.0,
        m_i=2/3,
        r_phi=900.0,
        L=1.0,
        m_phi=1/3,
        sigma_phi2=5e-10,
        lam=100.0,
        Rmin=1e-10,
        Rmax=1e0,
        NR=520,
        Nu=7000,
    )

    plot_derivative_measure("thick", params_thick, "LP16_sep_derivativeSF_thick.png")
    plot_derivative_measure("thin", params_thin, "LP16_sep_derivativeSF_thin.png")
if __name__ == "__main__":
    main()
