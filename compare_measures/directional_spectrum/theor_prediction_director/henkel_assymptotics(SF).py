import math
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


def Xi_i(R, A_P=1.0, R0=1.0, m_psi=2/3):
    return A_P / (1.0 + (R / R0) ** m_psi)


def f_saturating_powerlaw(R, Rb, m):
    x = (R / Rb) ** m
    return x / (1.0 + x)


def chi2_from_lambda(lam, sigma_phi2, chi2_factor=2.0):
    print(np.sqrt(chi2_factor * (lam ** 4) * sigma_phi2))
    return chi2_factor * (lam ** 4) * sigma_phi2


def xi_P(R, A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2, chi2_factor=2.0):
    Xi = Xi_i(R, A_P=A_P, R0=R0, m_psi=m_psi)
    fphi = f_saturating_powerlaw(R, r_phi, m_phi)
    chi2 = chi2_from_lambda(lam, sigma_phi2, chi2_factor=chi2_factor)
    return Xi * np.exp(-chi2 * fphi)


def Du_half(R, A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2, chi2_factor=2.0):
    return A_P - xi_P(
        R, A_P=A_P, R0=R0, m_psi=m_psi,
        r_phi=r_phi, m_phi=m_phi,
        lam=lam, sigma_phi2=sigma_phi2,
        chi2_factor=chi2_factor
    )


def sf_spectrum_proxy(k, R, Du2, sigma_ln=0.35):
    lnR = np.log(R)
    dlnR = np.diff(lnR)

    ln_kR = np.log(k[:, None]) + lnR[None, :]

    W = np.exp(-(ln_kR ** 2) / (2.0 * sigma_ln ** 2)) / (math.sqrt(2.0 * math.pi) * sigma_ln)

    integrand = Du2[None, :] * W
    return np.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dlnR[None, :], axis=1)


def compute_scales(A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2,
                   chi2_factor=2.0, eps=0.1, F=30.0, x_eff=1.0):
    chi2 = chi2_from_lambda(lam, sigma_phi2, chi2_factor=chi2_factor)

    R_psi = R0 * (eps ** (1.0 / m_psi))
    R_phi = r_phi * (eps ** (1.0 / m_phi))
    R_exp = r_phi * ((eps / chi2) ** (1.0 / m_phi)) if chi2 > 0 else np.inf

    R_asym = min(R_psi, R_phi, R_exp)
    k_asym = x_eff / R_asym

    k_inert_min = x_eff * max(1.0 / R0, 1.0 / r_phi)

    if abs(m_psi - m_phi) < 1e-12:
        R_x = np.nan
        k_x = np.nan
        delta = 0.0
    else:
        R_x = (chi2 * (R0 ** m_psi) / (r_phi ** m_phi)) ** (1.0 / (m_psi - m_phi))
        k_x = x_eff / R_x
        delta = m_phi - m_psi

    def k_at_ratio(target):
        return k_x * (target ** (-1.0 / delta))

    k_far_sure = (np.nan, np.nan)
    k_int_sure = (np.nan, np.nan)
    if np.isfinite(k_x) and abs(delta) > 1e-12:
        kF = k_at_ratio(F)
        kInvF = k_at_ratio(1.0 / F)
        k_lo = min(kF, kInvF)
        k_hi = max(kF, kInvF)

        k_test = 10.0 * k_x
        W_test = (k_x / k_test) ** delta

        if W_test < 1.0:
            k_far_sure = (0.0, k_lo)
            k_int_sure = (k_hi, np.inf)
        else:
            k_far_sure = (k_hi, np.inf)
            k_int_sure = (0.0, k_lo)

        def intersect_with_asym(interval):
            lo, hi = interval
            lo = max(lo, k_asym)
            if hi < lo:
                return (np.nan, np.nan)
            return (lo, hi)

        k_far_sure = intersect_with_asym(k_far_sure)
        k_int_sure = intersect_with_asym(k_int_sure)

    return dict(
        chi2=chi2,
        eps=eps,
        F=F,
        x_eff=x_eff,
        R_psi=R_psi,
        R_phi=R_phi,
        R_exp=R_exp,
        R_asym=R_asym,
        R_x=R_x,
        delta=delta,
        k_inert_min=k_inert_min,
        k_asym=k_asym,
        k_x=k_x,
        k_far_sure=k_far_sure,
        k_int_sure=k_int_sure,
    )


def suggest_sigma_phi2_for_kx(kx_desired, R0, m_psi, r_phi, m_phi, lam,
                              chi2_factor=2.0):
    R_x = 1.0 / kx_desired
    if abs(m_psi - m_phi) < 1e-12:
        raise ValueError("m_psi == m_phi: crossover is prefactor-only; kx is not set by slope balance.")
    chi2 = (r_phi ** m_phi / R0 ** m_psi) * (R_x ** (m_psi - m_phi))
    return chi2 / (chi2_factor * lam ** 4)
def run_and_plot(params,
                 Rmin=1e-4, Rmax=1e3, NR=1400,
                 Nk=900,
                 eps=0.1, F=30.0,
                 sigma_ln=0.35,
                 out_prefix="sf_run",
                 save_scales=True,
                 show=True):
    A_P = params["A_P"]
    R0 = params["R0"]
    m_psi = params["m_psi"]
    r_phi = params["r_phi"]
    m_phi = params["m_phi"]
    lam = params["lam"]
    sigma_phi2 = params["sigma_phi2"]
    chi2_factor = params.get("chi2_factor", 2.0)

    R = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)
    Xi = Xi_i(R, A_P=A_P, R0=R0, m_psi=m_psi)
    xi = xi_P(R, A_P=A_P, R0=R0, m_psi=m_psi,
              r_phi=r_phi, m_phi=m_phi,
              lam=lam, sigma_phi2=sigma_phi2,
              chi2_factor=chi2_factor)
    Du2 = A_P - xi

    kmin = 1.0 / Rmax
    kmax = 1.0 / Rmin
    k = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

    M = sf_spectrum_proxy(k, R, Du2, sigma_ln=sigma_ln)

    scales = compute_scales(
        A_P=A_P, R0=R0, m_psi=m_psi,
        r_phi=r_phi, m_phi=m_phi,
        lam=lam, sigma_phi2=sigma_phi2,
        chi2_factor=chi2_factor,
        eps=eps, F=F,
        x_eff=1.0
    )

    fig1, ax1 = plt.subplots(figsize=(7.6, 5.0))
    ax1.loglog(R, Du2, "k.", lw=2.2, label=r"$\frac{D_u(R;\lambda)}{2}$")
    ax1.loglog(R, Xi, color="1", ls=":", lw=2.5, label=r"$\Xi_i(R)$")
    ax1.loglog(R, xi, color="0.25", ls="--", lw=2.0, alpha=0.8, label=r"$\xi_P(R)$")

    chi2 = scales["chi2"]
    Du2_int = A_P * (R / R0) ** m_psi
    Du2_far = A_P * chi2 * (R / r_phi) ** m_phi
    ax1.loglog(R, Du2_int, ls="--", lw=2.2, alpha=0.65, label=rf"$\propto R^{{m_\psi}}$")
    ax1.loglog(R, Du2_far, ls="--", lw=2.2, alpha=0.65, label=rf"$\propto R^{{m_\Phi}}$")

    ax1.set_xlim(Rmin, Rmax)
    ax1.set_ylim(1e-4, 2e0)
    
    def mark_R(v, color, ls, text):
        if np.isfinite(v) and (Rmin < v < Rmax):
            ax1.axvline(v, color=color, ls=ls, lw=2.2)
            # print(text)
            ax1.text(v, ax1.get_ylim()[0] * 1000, text, rotation=90,
                     va="top", ha="right", color=color)

    # mark_R(R0, "C0", "-.", r"$R_0$")
    mark_R(r_phi, "C2", "-.", r"$r_\phi$")
    mark_R(scales["R_asym"], "red", ":", r"$R_{\rm asym}$")
    mark_R(scales["R_x"], "green", ":", r"$R_\times$")
    
    ax1.set_xlabel(r"$R$")
    ax1.set_ylabel(r"structure function")
    # ax1.set_title("Real-space (no Hankel transforms)")
    # ax1.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)
    ax1.legend(loc="best", fontsize=20)
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_realspace.png", dpi=220)
    fig1.savefig(f"{out_prefix}_realspace.svg", dpi=220)
    print(f"Saved {out_prefix}_realspace.svg")

    fig2, ax2 = plt.subplots(figsize=(7.6, 5.0))
    ax2.loglog(k, M, "k.", lw=2.2,
               label=r"$\mathcal{M}(k)=\int [D_u/2]\,W(kR)\,d\ln R$")

    k_anchor = math.sqrt(kmin * kmax)
    y_anchor = M[np.argmin(np.abs(k - k_anchor))]
    ax2.loglog(k, y_anchor * (k / k_anchor) ** (-m_psi),
               ls="--", alpha=1, label=rf"ref $k^{{-m_\psi}}$")
    ax2.loglog(k, y_anchor * (k / k_anchor) ** (-m_phi),
               ls="--", alpha=1, label=rf"ref $k^{{-m_\Phi}}$")

    ax2.set_xlim(kmin, kmax)
    ax2.set_ylim(1e-4, 2e0)
    
    ax2.axvline(scales["k_inert_min"], color="red", ls=":", lw=2.2, label=r"$k_{\rm inert,min}$")
    ax2.axvline(scales["k_asym"],      color="red", ls="--", lw=2.2, label=r"$k_{\rm asym}$")
    if np.isfinite(scales["k_x"]):
        ax2.axvline(scales["k_x"], color="green", ls="-.", lw=2.2, label=r"$k_\times$")

    for name, (lo, hi) in [("Faraday sure", scales["k_far_sure"]),
                           ("Intrinsic sure", scales["k_int_sure"])]:
        for v in (lo, hi):
            if np.isfinite(v) and (kmin < v < kmax):
                ax2.axvline(v, color="blue", ls="--", lw=0.9)

    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$\mathcal{M}(k)$")
    ax2.set_title("Non-oscillatory k-proxy from the structure function")
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_kproxy.png", dpi=220)
    fig2.savefig(f"{out_prefix}_kproxy.svg", dpi=220)
    print(f"Saved {out_prefix}_kproxy.svg")

    if show:
        pass
        # plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return scales


if __name__ == "__main__":
    params = dict(
        A_P=1.0,
        R0=1.0,
        m_psi=5/3,
        r_phi=1.0,
        m_phi=1/2,
        lam=1,
        sigma_phi2=0.01,
        chi2_factor=2.0,
    )

    scales = run_and_plot(
        params,
        Rmin=1e-4, Rmax=1e1, NR=1400,
        Nk=900,
        eps=0.1,
        F=30.0,
        sigma_ln=0.35,
        out_prefix="sf_demo",
        save_scales=True,
        show=True
    )

    print("\nComputed scales:")
    for k, v in scales.items():
        print(f"{k:>12s} : {v}")
