import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0


def f_powerlaw(R, r0, m):
    x = (R / r0) ** m
    return x / (1.0 + x)


def xi_u_dir(R, A_P, R0, m_psi, r_phi, m_phi, chi):
    fPsi = f_powerlaw(R, R0, m_psi)
    fPhi = f_powerlaw(R, r_phi, m_phi)
    return A_P * (1.0 - fPsi) * np.exp(-(chi ** 2) * fPhi)


def make_R_grid(R0, r_phi, Rmax_factor=2048.0, Rmin_factor=1e-6, Nlog=5000, Nlin=7000):
    Rmax = Rmax_factor * max(R0, r_phi)
    Rmin = Rmin_factor * min(R0, r_phi)
    Rmid = 10.0 * max(R0, r_phi)

    Rlog = np.geomspace(Rmin, Rmid, Nlog)
    Rlin = np.linspace(Rmid, Rmax, Nlin)

    R = np.unique(np.concatenate(([0.0], Rlog, Rlin)))
    return R


def P_dir_numeric(k, R, xiR, chunk=128):
    k = np.atleast_1d(k).astype(float)
    out = np.empty_like(k)

    for i0 in range(0, len(k), chunk):
        kk = k[i0:i0 + chunk][:, None]
        J = j0(kk * R[None, :])
        integrand = xiR[None, :] * J * R[None, :]
        out[i0:i0 + chunk] = 2.0 * np.pi * np.trapz(integrand, R, axis=1)

    return out


def P_dir_numeric_xgrid(k, xi_func, x_max=800.0, Nx=20000, chunk=128):
    k = np.atleast_1d(k).astype(float)

    x = np.linspace(0.0, x_max, Nx)
    J = j0(x)
    base = J * x

    out = np.empty_like(k)

    for i0 in range(0, len(k), chunk):
        kk = k[i0:i0+chunk]

        R = x[None, :] / kk[:, None]
        xi = xi_func(R)

        integrand = xi * base[None, :]
        out[i0:i0+chunk] = (2.0*np.pi) * np.trapz(integrand, x, axis=1) / (kk**2)

    return out


def dominance_scales(A_P, R0, m_psi, r_phi, m_phi, chi, dominance_factor=10.0, eta_inert=10.0):
    a_psi = A_P / (R0 ** m_psi)
    a_phi = A_P / (r_phi ** m_phi)
    delta = m_phi - m_psi
    if abs(delta) < 1e-14:
        raise ValueError("m_phi == m_psi -> no slope crossover in this two-term model.")

    kx = (a_phi * chi ** 2 / a_psi) ** (1.0 / delta)

    k_inert_min = eta_inert * max(1.0 / R0, 1.0 / r_phi)

    F = float(dominance_factor)
    eps = 1.0 / F

    k_W_eq_F = (a_phi * chi ** 2 / (a_psi * F)) ** (1.0 / delta)
    k_W_eq_invF = (a_phi * chi ** 2 * F / a_psi) ** (1.0 / delta)

    if delta > 0:
        k_far_sure = (k_inert_min, k_W_eq_F)
        k_int_sure = (k_W_eq_invF, np.inf)
    else:
        k_far_sure = (k_W_eq_F, np.inf)
        k_int_sure = (k_inert_min, k_W_eq_invF)

    return dict(
        a_psi=a_psi,
        a_phi=a_phi,
        delta=delta,
        kx=kx,
        k_inert_min=k_inert_min,
        k_far_sure=k_far_sure,
        k_int_sure=k_int_sure,
        dominance_factor=F,
    )


def pick_geom_mean(window, kmin, kmax):
    a, b = window
    lo = max(kmin, a)
    hi = min(kmax, b if np.isfinite(b) else kmax)
    if not (lo < hi):
        return None
    return 10.0 ** (0.5 * (np.log10(lo) + np.log10(hi)))


def chi_for_target_kx(A_P, R0, m_psi, r_phi, m_phi, kx_target):
    a_psi = A_P / (R0 ** m_psi)
    a_phi = A_P / (r_phi ** m_phi)
    delta = m_phi - m_psi
    chi2 = (a_psi / a_phi) * (kx_target ** delta)
    return np.sqrt(chi2)


def plot_Pdir_with_asymptotics(
    A_P=1.0,
    R0=1.0,
    m_psi=2/3,
    r_phi=0.1,
    m_phi=5/3,
    chi=None,
    kmin=1e-1,
    kmax=1e5,
    Nk=360,
    dominance_factor=10.0,
    eta_inert=10.0,
    x_max=800.0,
    Nx=20000,
    save_npz=None,
    save_png=None,
):
    if chi is None:
        kx_target = np.sqrt(kmin * kmax)
        chi = chi_for_target_kx(A_P, R0, m_psi, r_phi, m_phi, kx_target)

    k = np.geomspace(kmin, kmax, Nk)

    def xi_of_R(R):
        return xi_u_dir(R, A_P, R0, m_psi, r_phi, m_phi, chi)

    P = P_dir_numeric_xgrid(k, xi_of_R, x_max=x_max, Nx=Nx)
    Pabs = np.abs(P)

    scales = dominance_scales(A_P, R0, m_psi, r_phi, m_phi, chi,
                              dominance_factor=dominance_factor, eta_inert=eta_inert)

    k_F_lo, k_F_hi = scales["k_far_sure"]
    k_I_lo, k_I_hi = scales["k_int_sure"]
    kx = scales["kx"]
    
    kN_far = None
    kN_int = None
    
    if np.isfinite(k_F_hi) and (kmin < k_F_hi < kmax):
        kN_far = np.sqrt(kx * k_F_hi)
        kN_far = np.clip(kN_far, kmin, kmax)
    
    if np.isfinite(k_I_lo) and (kmin < k_I_lo < kmax):
        kN_int = k_I_lo
        kN_int = np.clip(kN_int, kmin, kmax)

    P_far_as = None
    P_int_as = None

    if kN_far is not None:
        Pk = np.abs(P_dir_numeric_xgrid(np.array([kN_far]), xi_of_R, x_max=x_max, Nx=Nx))[0]
        C = Pk * (kN_far ** (m_phi + 2.0))
        P_far_as = C * k ** (-(m_phi + 2.0))

    if kN_int is not None:
        Pk = np.abs(P_dir_numeric_xgrid(np.array([kN_int]), xi_of_R, x_max=x_max, Nx=Nx))[0]
        C = Pk * (kN_int ** (m_psi + 2.0))
        P_int_as = C * k ** (-(m_psi + 2.0))

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.loglog(k, Pabs, "o", ms=3.0, color="k", label=r"$|P_{\rm dir}(k)|$")

    if P_int_as is not None:
        ax.loglog(k, P_int_as, lw=2.4, label=rf"Intrinsic slope $k^{{-(m_\Psi+2)}}$, $m_\Psi={m_psi:.2f}$")
    else:
        ax.text(0.03, 0.07, "No intrinsic 'sure' window in this k-range",
                transform=ax.transAxes)

    if P_far_as is not None:
        ax.loglog(k, P_far_as, lw=2.4, label=rf"Faraday slope $k^{{-(m_\Phi+2)}}$, $m_\Phi={m_phi:.2f}$")
    else:
        ax.text(0.03, 0.03, "No Faraday 'sure' window in this k-range",
                transform=ax.transAxes)
    
    ax.axvline(scales["k_inert_min"], color="red", ls=":", lw=1.2)
    ax.axvline(scales["kx"], color="green", ls="-.", lw=1.2, label=r"$k_\times$")
    
    F = scales["dominance_factor"]
    delta = scales["delta"]
    k_F_lo, k_F_hi = scales["k_far_sure"]
    k_I_lo, k_I_hi = scales["k_int_sure"]
    
    boundaries = []
    for v in [k_F_lo, k_F_hi, k_I_lo, k_I_hi]:
        if np.isfinite(v) and (kmin < v < kmax):
            boundaries.append(v)
    
    for v in sorted(set(boundaries)):
        ax.axvline(v, color="blue", ls="--", lw=0.9)
    
    lo, hi = scales["k_far_sure"]
    lo = max(lo, kmin)
    hi = min(hi if np.isfinite(hi) else kmax, kmax)
    if lo < hi:
        ax.axvspan(hi, scales["kx"], color="blue", alpha=0.08, label=rf"Faraday-dominated: $W\geq {F:g}$")
    
    lo, hi = scales["k_int_sure"]
    lo = max(lo, kmin)
    hi = min(hi if np.isfinite(hi) else kmax, kmax)
    if lo < hi:
        ax.axvspan(scales["kx"], lo, color="gray", alpha=0.08, label=rf"Intrinsic-dominated: $W\leq 1/{F:g}$")

    ax.set_xlabel(r"$k\,R_0$")
    ax.set_ylabel(r"$|P_{\rm dir}(k)|$")
    ax.set_title(
        rf"Directional spectrum: $\chi={chi:.3g}$, "
        rf"$m_\Psi={m_psi:.2f}$, $m_\Phi={m_phi:.2f}$, "
        rf"$F={dominance_factor:g}$"
    )
    
    ax.set_ylim(1e-10, 1e-3)
    ax.set_xlim(kmin, lo)
    
    y_top = ax.get_ylim()[1]
    # ax.text(scales["k_inert_min"], y_top*0.8, r"$k_{\rm inert,min}$", color="red",
    #         rotation=90, va="top", ha="right", fontsize=10)
    # ax.text(scales["kx"], y_top*0.8, r"$k_\times$", color="green",
    #         rotation=90, va="top", ha="right", fontsize=10)
    
    b = sorted([v for v in boundaries if np.isfinite(v)])
    if len(b) >= 2:
        below = max([v for v in b if v < scales["kx"]], default=None)
        above = min([v for v in b if v > scales["kx"]], default=None)
        
        # if below is not None:
        #     ax.text(below, y_top*0.6, rf"$W(k)={F:g}$ or $1/{F:g}$", color="blue",
        #             rotation=90, va="top", ha="right", fontsize=9)
        # if above is not None:
        #     ax.text(above, y_top*0.6, rf"$W(k)={F:g}$ or $1/{F:g}$", color="blue",
        #             rotation=90, va="top", ha="right", fontsize=9)
    
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    if save_npz is not None:
        R_save = make_R_grid(R0, r_phi)
        xiR_save = xi_u_dir(R_save, A_P, R0, m_psi, r_phi, m_phi, chi)
        np.savez(
            save_npz,
            k=k, P=P, Pabs=Pabs,
            R=R_save, xiR=xiR_save,
            **scales,
            params=np.array([A_P, R0, m_psi, r_phi, m_phi, chi, dominance_factor, eta_inert], dtype=float),
        )
        print(f"Saved data: {save_npz}")

    if save_png is not None:
        fig.savefig(save_png, dpi=200)
        print(f"Saved figure: {save_png}")
    # ax.xlim(0,1)
    print("\n--- Dominance windows (within inertial condition) ---")
    print(f"chi = {chi:.6g}")
    print(f"delta = m_phi - m_psi = {scales['delta']:.6g}")
    print(f"k_inert_min = {scales['k_inert_min']:.6g}")
    print(f"k_x = {scales['kx']:.6g}")
    print(f"Faraday sure window (W>=F):      k in [{scales['k_far_sure'][0]:.6g}, {scales['k_far_sure'][1]}]")
    print(f"Intrinsic sure window (W<=1/F):  k in [{scales['k_int_sure'][0]:.6g}, {scales['k_int_sure'][1]}]")
    print("-----------------------------------------------\n")

    return fig, scales


def main():
    A_P = 1
    R0 = 1.0
    m_psi = 2/3

    r_phi = 0.1
    m_phi = 5/3

    kx_target = 1e4
    chi = chi_for_target_kx(A_P, R0, m_psi, r_phi, m_phi, kx_target)
    chi=5

    plot_Pdir_with_asymptotics(
        A_P=A_P, R0=R0, m_psi=m_psi,
        r_phi=r_phi, m_phi=m_phi,
        chi=chi,
        kmin=5e1, kmax=1e6, Nk=380,#2.5e4
        dominance_factor=6.0,
        eta_inert=10.0,
        save_npz="Pdir_demo.npz",
        save_png="Pdir_demo.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
