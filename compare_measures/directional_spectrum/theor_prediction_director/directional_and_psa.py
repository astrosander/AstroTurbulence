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

# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# Publication-ready font sizes

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

def F(R, A_P, chi, R0, r_phi, m_psi, m_phi):
    return A_P * (1.0 - (1.0 - fpsi(R, R0, m_psi)) * np.exp(-(chi**2) * fphi(R, r_phi, m_phi)))

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
        return term_int, term_far

    if screen_type == "thin":
        term_far = np.full_like(R, np.nan, dtype=float)
        m1 = R < L
        m2 = (R >= L) & (R < r_phi)
        term_far[m1] = (r_phi / L) * (R[m1] / r_phi) ** (1.0 + mtil)
        term_far[m2] = (R[m2] / r_phi) ** (mtil)
        return term_int, term_far

    raise ValueError("screen_type must be 'thick' or 'thin'")

def pick_match_R(R, y_num, prefer="small", R_max=None, R_min=None):
    mask = np.isfinite(y_num) & (y_num > 0) & np.isfinite(R) & (R > 0)
    if R_max is not None:
        mask &= (R <= R_max)
    if R_min is not None:
        mask &= (R >= R_min)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None
    return R[idxs[0]] if prefer == "small" else R[idxs[-1]]

def normalize_asymptotic_to_numeric(R, y_num, y_asym, R_match):
    if R_match is None:
        return y_asym, 1.0
    idx = np.argmin(np.abs(np.log(R) - np.log(R_match)))
    if (not np.isfinite(y_asym[idx])) or (y_asym[idx] <= 0) or (not np.isfinite(y_num[idx])) or (y_num[idx] <= 0):
        return y_asym, 1.0
    fac = y_num[idx] / y_asym[idx]
    if (not np.isfinite(fac)) or (fac <= 0):
        return y_asym, 1.0
    return y_asym * fac, fac

def F_small_piece_psi(R, A_P, R0, m_psi):
    return A_P * (R / R0) ** m_psi

def F_small_piece_phi(R, A_P, chi, r_phi, m_phi):
    return A_P * (chi**2) * (R / r_phi) ** m_phi

def compute_R_x_F(chi, R0, r_phi, m_psi, m_phi):
    if abs(m_psi - m_phi) > 1e-12:
        return ((chi**2) * (R0**m_psi) / (r_phi**m_phi)) ** (1.0 / (m_psi - m_phi))
    return None

def compute_R_x_SF_thick(fac_int, fac_far, r_i, m_i, L, r_phi, m_phi):
    mtil = min(m_phi, 1.0)
    p_int = m_i
    p_far = 1.0 + mtil
    if abs(p_int - p_far) < 1e-14:
        return None

    C_int = fac_int * (r_i ** (-m_i))
    pref  = (r_phi / L) ** (1.0 - mtil)
    C_far = fac_far * pref * (r_phi ** (-(1.0 + mtil)))

    if (C_int <= 0) or (C_far <= 0) or (not np.isfinite(C_int)) or (not np.isfinite(C_far)):
        return None

    return (C_far / C_int) ** (1.0 / (p_int - p_far))

R = np.logspace(-4, 1, 1500)

def compute_all_curves(A_P, chi, R0, r_phi, m_psi, m_phi, L):
    r_i = R0
    m_i = m_psi

    sigma_phi2, lam = compute_sigma_phi2_and_lambda(chi)

    F_vals = F(R, A_P, chi, R0, r_phi, m_psi, m_phi)
    F_psi  = F_small_piece_psi(R, A_P, R0, m_psi)
    F_phi  = F_small_piece_phi(R, A_P, chi, r_phi, m_phi)
    R_x_F  = compute_R_x_F(chi, R0, r_phi, m_psi, m_phi)

    SF_num = SF_int = SF_far = None
    fac_int = fac_far = 1.0
    R_match_int = R_match_far = None
    R_x_SF = None

    has_numerical = has_asymptotics = False

    try:
        xiPp, xi_DF, D_DF = xi_dP_dlam2(R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=Nu)
        SF_num = SF_from_corr(xiPp)
        has_numerical = True

        screen_type = "thick" if L > r_phi else "thin"
        term_int, term_far = asymptotic_terms_derivative(screen_type, R, r_i, m_i, L, r_phi, m_phi)

        R_min = R.min()
        R_max = R.max()
        R_match_int = 0.001 * R_max
        R_match_far = 10.0 * R_min

        SF_int, fac_int = normalize_asymptotic_to_numeric(R, SF_num, term_int, R_match_int)

        term_far_safe = np.where(np.isfinite(term_far) & (term_far > 0), term_far, np.nan)
        SF_far, fac_far = normalize_asymptotic_to_numeric(R, SF_num, term_far_safe, R_match_far)

        if screen_type == "thick":
            R_x_SF = compute_R_x_SF_thick(fac_int, fac_far, r_i, m_i, L, r_phi, m_phi)

        has_asymptotics = True

    except Exception as e:
        pass

    return (F_vals, F_psi, F_phi, R_x_F,
            SF_num, SF_int, SF_far,
            R_match_int, R_match_far, fac_int, fac_far, R_x_SF,
            has_numerical, has_asymptotics)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
plt.subplots_adjust(bottom=0.25)

def update_plot(chi, r_phi, m_psi, m_phi):
    F_vals, F_psi, F_phi, R_x_F, SF_num, SF_int, SF_far, R_match_int, R_match_far, fac_int, fac_far, R_x_SF, has_numerical, has_asymptotics = compute_all_curves(
        init_A_P, chi, init_R0, r_phi, m_psi, m_phi, init_L
    )
    
    ax1.clear()
    ax2.clear()
    
    ax1.loglog(R, F_vals, label="Directional SF: $D_u/2$", lw=3)
    ax1.loglog(R, F_psi, "--", label=r"INTRISTIC", color="orange", lw=2, alpha=1)
    ax1.loglog(R, F_phi, "--", label=r"FARADAY", color="green", lw=2, alpha=1)
    
    if R_x_F is not None and np.isfinite(R_x_F) and (R.min() < R_x_F < R.max()):
        ax1.axvline(R_x_F, linestyle="-.", linewidth=2, color="red")
        y0, y1 = ax1.get_ylim()
        ax1.text(R_x_F, y0 * 10, r"$R_{\times,F}$", rotation=90, va="bottom", ha="right", color="red")
    
    ax1.set_ylabel("F(R)")
    ax1.legend(fontsize=12)
    
    x_all = np.concatenate([R, R[1:]])
    x_min, x_max = x_all.min(), x_all.max()
    
    y1_all = np.concatenate([F_vals])#, F_psi, F_phi])
    y1_all = y1_all[np.isfinite(y1_all) & (y1_all > 0)]
    if len(y1_all) > 0:
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y1_all.min(), y1_all.max())
    
    if has_numerical and (SF_num is not None):
        ax2.loglog(R[1:], SF_num[1:], color="k", lw=3.2, label=r"PSA: numerical")
    
    if has_asymptotics:
        if SF_int is not None:
            ax2.loglog(R[1:], SF_int[1:], ls="-.", lw=2.0, color="orange", label=r"INTRISTIC")
        if SF_far is not None:
            ax2.loglog(R[1:], SF_far[1:], ls="-.", lw=2.0, color="green", label=r"FARADAY")
        
        if (R_x_SF is not None) and np.isfinite(R_x_SF) and (R.min() < R_x_SF < R.max()):
            ax2.axvline(R_x_SF, linestyle="-.", linewidth=2, color="red")
            y_cross = fac_int * (R_x_SF / init_R0) ** m_psi
            y0, y1 = ax2.get_ylim()
            ax2.text(R_x_SF, y0 * 10, r"$R_{\times,SF}$", rotation=90, va="bottom", ha="right", color="red")
    
    ax2.set_xlabel("R")
    ax2.set_ylabel("SF(R)")
    ax2.legend(fontsize=12)
    
    y2_all_list = []
    if has_numerical and (SF_num is not None):
        y2_all_list.append(SF_num[1:])
    
    if len(y2_all_list) > 0:
        y2_all = np.concatenate(y2_all_list)
        y2_all = y2_all[np.isfinite(y2_all) & (y2_all > 0)]
        if len(y2_all) > 0:
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(5e-12, y2_all.max())
    
    fig.canvas.draw_idle()

initial_F_vals, initial_F_psi, initial_F_phi, initial_R_x_F, initial_SF_num, initial_SF_int, initial_SF_far, initial_R_match_int, initial_R_match_far, initial_fac_int, initial_fac_far, initial_R_x_SF, initial_has_numerical, initial_has_asymptotics = compute_all_curves(
    init_A_P, init_chi, init_R0, init_r_phi, init_m_psi, init_m_phi, init_L
)

update_plot(init_chi, init_r_phi, init_m_psi, init_m_phi)

ax_chi = plt.axes([0.15, 0.15, 0.3, 0.03])
ax_r_phi = plt.axes([0.15, 0.11, 0.3, 0.03])
ax_m_psi = plt.axes([0.15, 0.07, 0.3, 0.03])
ax_m_phi = plt.axes([0.15, 0.03, 0.3, 0.03])

slider_chi = Slider(ax_chi, r'$\chi$', 0.01, 2.0, valinit=init_chi, valstep=0.01)
slider_r_phi = Slider(ax_r_phi, r'$r_\phi$', 0.1, 10.0, valinit=init_r_phi, valstep=0.1)
slider_m_psi = Slider(ax_m_psi, r'$m_\psi$', 0.1, 5.0, valinit=init_m_psi, valstep=0.1)
slider_m_phi = Slider(ax_m_phi, r'$m_\Phi$', 0.1, 5.0, valinit=init_m_phi, valstep=0.1)

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
print(f"F crossover R_x_F = {initial_R_x_F}")
print(f"SF match points: R_match_int={initial_R_match_int}, R_match_far={initial_R_match_far}")
print(f"SF crossover R_x_SF (thick screen, after matching) = {initial_R_x_SF}")
