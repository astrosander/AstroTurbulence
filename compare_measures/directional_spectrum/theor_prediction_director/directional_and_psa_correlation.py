import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

init_A_P  = 1.0
init_chi  = 0.14
init_R0   = 1.0
init_r_phi = 1
init_m_psi = 4/3
init_m_phi = 2/3
init_L    = 200.0
Nu        = 100
Xi0       = 1.0

plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 16


def compute_sigma_phi2_and_lambda(chi, lambda_fixed=100.0):
    sigma_phi = chi / (2.0 * lambda_fixed**2)
    sigma_phi2 = sigma_phi**2
    return sigma_phi2, lambda_fixed

def fpsi(R, R0, m_psi):
    x = (R / R0) ** m_psi
    return x / (1.0 + x)

def fphi(R, r_phi, m_phi):
    y = (R / r_phi) ** m_phi
    return y / (1.0 + y)

def xi_P(R, A_P, R0, m_psi):
    return A_P / (1.0 + (R / R0) ** m_psi)

def xi_u(R, A_P, chi, R0, r_phi, m_psi, m_phi):
    xiP = xi_P(R, A_P, R0, m_psi)
    return xiP * np.exp(-(chi**2) * fphi(R, r_phi, m_phi))

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

def D_DeltaPhi_numeric(R, L, r_phi, m_phi, sigma_phi2=1.0, Nu=6000):
    R = np.atleast_1d(R).astype(float)
    dz = make_dz_grid(L, r_phi, Nu=Nu)
    xi0 = xi_phi_local(0.0, dz, sigma_phi2=sigma_phi2, r_phi=r_phi, m_phi=m_phi)[None, :]
    xiR = xi_phi_local(R[:, None], dz[None, :], sigma_phi2=sigma_phi2, r_phi=r_phi, m_phi=m_phi)
    integrand = 2.0 * (L - dz)[None, :] * (xi0 - xiR)
    return np.trapz(integrand, dz, axis=1)

def D_asym_thick(R, L, r_phi, m_phi, sigma_phi2=1.0):
    mtil = min(m_phi, 1.0)
    D148 = sigma_phi2 * L * R * (R / r_phi) ** mtil
    D149 = sigma_phi2 * L * R * (R / r_phi) ** (-mtil)
    D150 = sigma_phi2 * (L ** 2) * (L / r_phi) ** (-mtil) * np.ones_like(R)
    return D148, D149, D150

def xiP_from_D(Xi, D, lam=1.0, clip_exp=(-700.0, 50.0)):
    expo = -4.0 * (lam ** 4) * D
    expo = np.clip(expo, clip_exp[0], clip_exp[1])
    return Xi * np.exp(expo)

def compute_R_x_xi(chi, R0, r_phi, m_psi, m_phi):
    if abs(m_psi - m_phi) > 1e-12:
        return ((chi**2) * (R0**m_psi) / (r_phi**m_phi)) ** (1.0 / (m_psi - m_phi))
    return None

R = np.logspace(-5, 1, 1500)

def compute_all_curves(A_P, chi, R0, r_phi, m_psi, m_phi, L):
    r_i = R0
    m_i = m_psi

    sigma_phi2, lam = compute_sigma_phi2_and_lambda(chi)

    xi_u_vals = xi_u(R, A_P, chi, R0, r_phi, m_psi, m_phi)
    xi_P_vals = xi_P(R, A_P, R0, m_psi)
    R_x_xi = compute_R_x_xi(chi, R0, r_phi, m_psi, m_phi)

    xi_dP_dlam2_num = None
    xi_dP_dlam2_asym1 = None
    xi_dP_dlam2_asym2 = None
    xi_dP_dlam2_asym3 = None
    mask1 = mask2 = mask3 = None
    has_numerical = False

    try:
        XiR = xi_i(R, Xi0=Xi0, r_i=r_i, m_i=m_i)
        Dnum = D_DeltaPhi_numeric(R, L, r_phi, m_phi, sigma_phi2, Nu=Nu)
        xi_dP_dlam2_num = xiP_from_D(XiR, Dnum, lam=lam)
        has_numerical = True

        if L > r_phi:
            D1, D2, D3 = D_asym_thick(R, L, r_phi, m_phi, sigma_phi2)
            mask1 = R < r_phi
            mask2 = (R >= r_phi) & (R < L)
            mask3 = R >= L

            R_min = R.min()
            R_max = R.max()
            R_match1 = np.clip(0.1 * r_phi, R_min, R_max)
            R_match2 = np.clip(L, R_min, R_max)
            R_match3 = np.clip(R_max, R_min, R_max)

            fit1 = (R < 0.3 * r_phi)
            fit2 = (R > 0.2 * L) & (R < 0.3 * L)
            fit3 = (R > 50 * L)

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

            xi_dP_dlam2_asym1 = xiP_from_D(XiR, D1, lam=lam)
            xi_dP_dlam2_asym2 = xiP_from_D(XiR, D2, lam=lam)
            xi_dP_dlam2_asym3 = xiP_from_D(XiR, D3, lam=lam)

    except Exception as e:
        pass

    return (xi_u_vals, xi_P_vals, R_x_xi,
            xi_dP_dlam2_num, xi_dP_dlam2_asym1, xi_dP_dlam2_asym2, xi_dP_dlam2_asym3,
            mask1, mask2, mask3, has_numerical)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
plt.subplots_adjust(bottom=0.20, top=0.98, left=0.08, right=0.98, wspace=0.15)

def update_plot(chi, r_phi, m_psi, m_phi):
    xi_u_vals, xi_P_vals, R_x_xi, xi_dP_dlam2_num, xi_dP_dlam2_asym1, xi_dP_dlam2_asym2, xi_dP_dlam2_asym3, mask1, mask2, mask3, has_numerical = compute_all_curves(
        init_A_P, chi, init_R0, r_phi, m_psi, m_phi, init_L
    )
    
    ax1.clear()
    ax2.clear()
    
    ax1.loglog(R, xi_u_vals, label=r"$\xi_u(R;\lambda) = \frac{A_P}{1+(R/R_0)^{m_\psi}}\exp[-\chi^2 f_\Phi(R)]$", color="black", lw=3)
    ax1.loglog(R, xi_P_vals, "--", label=r"$\xi_P(R) = \frac{A_P}{1+(R/R_0)^{m_\psi}}$", color="orange", lw=2, alpha=1)
    
    ax1.set_ylabel(r"$\xi_u(R)$", labelpad=2)
    ax1.set_xlabel("R", labelpad=2)
    ax1.legend(fontsize=12, borderpad=0.3, handlelength=1.5)
    ax1.tick_params(pad=2)
    
    R_min = 10*R.min()
    R_max = R.max()
    x_min_plot = R_min
    x_max_plot = R_max
    
    mask_x = (R >= x_min_plot) & (R <= x_max_plot)
    y1_all = xi_u_vals[mask_x]
    y1_all = y1_all[np.isfinite(y1_all) & (y1_all > 0)]
    if len(y1_all) > 0:
        ax1.set_xlim(x_min_plot, x_max_plot)
        ax1.set_ylim(y1_all.min()/2, y1_all.max()*2)
    
    if R_x_xi is not None and np.isfinite(R_x_xi) and (x_min_plot < R_x_xi < x_max_plot):
        ax1.axvline(R_x_xi, linestyle="-.", linewidth=2, color="gray")
        y0, y1 = ax1.get_ylim()
        ax1.text(R_x_xi, 2 * y0, r"$R_{\times,\xi}$", rotation=90, va="bottom", ha="right", color="gray", fontsize=12)
    
    if has_numerical and (xi_dP_dlam2_num is not None):
        ax2.loglog(R, xi_dP_dlam2_num, "o", ms=3.0, color="k", label=r"$\xi_{dP/d\lambda^2}$: numerical")
        
        if xi_dP_dlam2_asym1 is not None and mask1 is not None:
            colors = ["C0", "C2", "C1"]
            labels = [r"Eq. (148)", r"Eq. (149)", r"Eq. (150)"]
            for y, mask, c, lab in zip([xi_dP_dlam2_asym1, xi_dP_dlam2_asym2, xi_dP_dlam2_asym3], 
                                       [mask1, mask2, mask3], colors, labels):
                if y is not None and mask is not None:
                    ax2.loglog(R, y, ls="--", lw=2.0, color=c, alpha=0.35)
                    ax2.loglog(R[mask], y[mask], ls="-", lw=2.2, color=c, label=lab)
        
        if init_L > init_r_phi:
            ax2.axvline(r_phi, color="0.6", lw=2.0, ls="--")
            ax2.axvline(init_L, color="0.4", lw=2.0, ls="-.")
            y0, y1 = ax2.get_ylim()
            ax2.text(r_phi, y1 * 0.6, r"$r_\phi$", rotation=90, va="top", ha="right", fontsize=16)
            ax2.text(init_L, y1 * 0.6, r"$L$", rotation=90, va="top", ha="right", fontsize=16)
    
    ax2.set_xlabel("R", labelpad=2)
    ax2.set_ylabel(r"$\xi_{dP/d\lambda^2}(R)$", labelpad=2)
    ax2.legend(fontsize=12, borderpad=0.3, handlelength=1.5)
    ax2.tick_params(pad=2)
    
    y2_all_list = []
    if has_numerical and (xi_dP_dlam2_num is not None):
        mask_x2 = (R >= x_min_plot) & (R <= x_max_plot)
        xi_dP_dlam2_filtered = xi_dP_dlam2_num[mask_x2]
        y2_all_list.append(xi_dP_dlam2_filtered)
    
    if len(y2_all_list) > 0:
        y2_all = np.concatenate(y2_all_list)
        y2_all = y2_all[np.isfinite(y2_all) & (y2_all > 0)]
        if len(y2_all) > 0:
            ax2.set_xlim(x_min_plot, x_max_plot)
            ax2.set_ylim(y2_all.min()/2, y2_all.max()*2)
    
    fig.canvas.draw_idle()

initial_xi_u_vals, initial_xi_P_vals, initial_R_x_xi, initial_xi_dP_dlam2_num, initial_xi_dP_dlam2_asym1, initial_xi_dP_dlam2_asym2, initial_xi_dP_dlam2_asym3, initial_mask1, initial_mask2, initial_mask3, initial_has_numerical = compute_all_curves(
    init_A_P, init_chi, init_R0, init_r_phi, init_m_psi, init_m_phi, init_L
)

update_plot(init_chi, init_r_phi, init_m_psi, init_m_phi)

ax_chi = plt.axes([0.15, 0.14, 0.3, 0.02])
ax_r_phi = plt.axes([0.15, 0.11, 0.3, 0.02])
ax_m_psi = plt.axes([0.15, 0.08, 0.3, 0.02])
ax_m_phi = plt.axes([0.15, 0.05, 0.3, 0.02])

slider_chi = Slider(ax_chi, r'$\chi$', 0.01, 2.0, valinit=init_chi, valstep=0.01, color="black")
slider_r_phi = Slider(ax_r_phi, r'$r_\phi$', 0.1, 10.0, valinit=init_r_phi, valstep=0.1, color="black")
slider_m_psi = Slider(ax_m_psi, r'INTRISTIC: $m_\psi$', 0.1, 5.0, valinit=init_m_psi, valstep=0.1, color="orange")
slider_m_phi = Slider(ax_m_phi, r'FARADAY: $m_\Phi$', 0.1, 5.0, valinit=init_m_phi, valstep=0.1, color="blue")

def update(val):
    chi = slider_chi.val
    r_phi = slider_r_phi.val
    m_psi = slider_m_psi.val
    m_phi = slider_m_phi.val
    update_plot(chi, r_phi, m_psi, m_phi)

slider_chi.on_changed(update)
slider_r_phi.on_changed(update)
slider_m_psi.on_changed(update)
slider_m_phi.on_changed(update)

plt.show()

print("=== Initial Parameters ===")
print(f"A_P={init_A_P}, chi={init_chi}, R0={init_R0}, r_phi={init_r_phi}, m_psi={init_m_psi}, m_phi={init_m_phi}, L={init_L}, Nu={Nu}")
print(f"Correlation crossover R_x,xi = {initial_R_x_xi}")

