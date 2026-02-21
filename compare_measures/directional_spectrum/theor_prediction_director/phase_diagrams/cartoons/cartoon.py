import numpy as np
import matplotlib.pyplot as plt

# Set up LaTeX rendering and publication-quality fonts
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 32,
    'axes.labelsize': 40,
    'axes.titlesize': 44,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 32,
    'figure.titlesize': 48,
    'lines.linewidth': 4.0,
    'axes.linewidth': 2.5,
    'grid.linewidth': 1.5,
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'xtick.minor.width': 1.5,
    'ytick.minor.width': 1.5,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 6,
    'ytick.minor.size': 6,
})

init_A_P = 1.0
init_R0 = 1.0
init_r_phi = 1.0
init_m_psi = 4/3
init_m_phi = 2/3
init_L = 200.0
Nu = 1000
Xi0 = 1.0

def compute_sigma_phi2_and_lambda(chi, lambda_fixed=100.0):
    sigma_phi = chi / (2.0 * lambda_fixed**2)
    return sigma_phi**2, lambda_fixed

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

def make_dz_grid(L, r_phi, Nu=3000):
    Nu = int(Nu)
    Nu_log = int(0.45 * Nu)
    Nu_lin = Nu - Nu_log
    dz_min = 1e-10 * L
    dz_knee = min(0.05 * L, 5.0 * r_phi, L)
    if dz_knee <= 0:
        dz_knee = 1e-6 * L
    dz_log = np.geomspace(dz_min, max(dz_knee, dz_min * 10), Nu_log)
    dz_lin = np.linspace(max(dz_knee, dz_min * 10), L, Nu_lin)
    return np.unique(np.concatenate(([0.0], dz_log, dz_lin)))

def xi_DeltaPhi(R, L, r_phi, m_phi, sigma_phi2=1.0, Nu=3000):
    R = np.atleast_1d(R).astype(float)
    dz = make_dz_grid(L, r_phi, Nu=Nu)
    w = 2.0 * (L - dz)
    xiR = xi_phi_local(R[:, None], dz[None, :], sigma_phi2, r_phi, m_phi)
    xi0 = xi_phi_local(0.0, dz, sigma_phi2, r_phi, m_phi)[None, :]
    xi_DF = np.trapz(w[None, :] * xiR, dz, axis=1)
    xi_DF0 = np.trapz(w[None, :] * xi0, dz, axis=1)[0]
    D_DF = xi_DF0 - xi_DF
    return xi_DF, D_DF

def xi_dP_dlam2(R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=3000):
    XiR = xi_i(R, Xi0=Xi0, r_i=r_i, m_i=m_i)
    xi_DF, D_DF = xi_DeltaPhi(R, L, r_phi, m_phi, sigma_phi2, Nu=Nu)
    expo = -4.0 * (lam ** 4) * D_DF
    expo = np.clip(expo, -800.0, 80.0)
    return XiR * (xi_DF + (lam ** 4) * (D_DF ** 2)) * np.exp(expo)

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
    pref = (r_phi / L) ** (1.0 - mtil)
    C_far = fac_far * pref * (r_phi ** (-(1.0 + mtil)))
    if (C_int <= 0) or (C_far <= 0) or (not np.isfinite(C_int)) or (not np.isfinite(C_far)):
        return None
    return (C_far / C_int) ** (1.0 / (p_int - p_far))

def compute_all(R, chi, A_P, R0, r_phi, m_psi, m_phi, L):
    F_vals = F(R, A_P, chi, R0, r_phi, m_psi, m_phi)
    F_psi = F_small_piece_psi(R, A_P, R0, m_psi)
    F_phi = F_small_piece_phi(R, A_P, chi, r_phi, m_phi)
    R_x_F = compute_R_x_F(chi, R0, r_phi, m_psi, m_phi)

    r_i = R0
    m_i = m_psi
    sigma_phi2, lam = compute_sigma_phi2_and_lambda(chi)
    SF_num = SF_from_corr(xi_dP_dlam2(R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=Nu))

    screen_type = "thick" if L > r_phi else "thin"
    term_int, term_far = asymptotic_terms_derivative(screen_type, R, r_i, m_i, L, r_phi, m_phi)

    R_min = R.min()
    R_max = R.max()
    R_match_int = 0.001 * R_max
    R_match_far = 10.0 * R_min

    SF_int, fac_int = normalize_asymptotic_to_numeric(R, SF_num, term_int, R_match_int)
    term_far_safe = np.where(np.isfinite(term_far) & (term_far > 0), term_far, np.nan)
    SF_far, fac_far = normalize_asymptotic_to_numeric(R, SF_num, term_far_safe, R_match_far)

    R_x_SF = None
    if screen_type == "thick":
        R_x_SF = compute_R_x_SF_thick(fac_int, fac_far, r_i, m_i, L, r_phi, m_phi)

    return F_vals, F_psi, F_phi, R_x_F, SF_num, SF_int, SF_far, R_x_SF, screen_type

R = np.logspace(-4, 1, 1500)
chis = [0.08, 0.14, 0.5]
colors = ['#2563EB', '#F59E0B', '#10B981']  # Modern blue, amber, emerald for each chi
color_m_psi = '#EF4444'  # Modern red for m_psi slopes
color_m_phi = '#8B5CF6'  # Modern purple for m_phi slopes

# Figure 1: Directional with three chi values
fig1, ax1 = plt.subplots(1, 1, figsize=(16*0.8,12*0.8))

m_psi_labeled = False
m_phi_labeled = False
for j, chi in enumerate(chis):
    F_vals, F_psi, F_phi, R_x_F, SF_num, SF_int, SF_far, R_x_SF, screen_type = compute_all(
        R, chi, init_A_P, init_R0, init_r_phi, init_m_psi, init_m_phi, init_L
    )

    ax1.loglog(R, F_vals, lw=5.0, label=rf"$\eta={chi}$", color=colors[j])
    if not m_psi_labeled:
        ax1.loglog(R, F_psi, "--", lw=3.5, color=color_m_psi, alpha=0.6, label=r"$m_\psi$ slope")
        m_psi_labeled = True
    else:
        ax1.loglog(R, F_psi, "--", lw=3.5, color=color_m_psi, alpha=0.6)
    if not m_phi_labeled:
        ax1.loglog(R, F_phi, "--", lw=3.5, color=color_m_phi, alpha=0.6, label=r"$m_\phi$ slope")
        m_phi_labeled = True
    else:
        ax1.loglog(R, F_phi, "--", lw=3.5, color=color_m_phi, alpha=0.6)
    if R_x_F is not None and np.isfinite(R_x_F) and (R.min() < R_x_F < R.max()):
        ax1.axvline(R_x_F, ls="-.", lw=3.5, color=colors[j], alpha=0.7)

ax1.set_xlabel(r"$R/r_i$", fontsize=40)
ax1.set_ylabel(r"directional", fontsize=40)
ax1.set_xlim(3e-4, 1e1)
ax1.set_ylim(1e-5, 1e0)
ax1.legend(fontsize=32, frameon=True, fancybox=True, shadow=False)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
ax1.tick_params(axis='both', which='major', labelsize=36, width=2.5, length=10)
ax1.tick_params(axis='both', which='minor', labelsize=32, width=1.5, length=6)

plt.tight_layout()
plt.savefig("directional_three_chi.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("directional_three_chi.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)

# Figure 2: dP/d lambda^2 with three chi values
# fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(1, 1, figsize=(16*0.8,12*0.8))

m_psi_labeled = False
m_phi_labeled = False
for j, chi in enumerate(chis):
    F_vals, F_psi, F_phi, R_x_F, SF_num, SF_int, SF_far, R_x_SF, screen_type = compute_all(
        R, chi, init_A_P, init_R0, init_r_phi, init_m_psi, init_m_phi, init_L
    )

    ax2.loglog(R[1:], SF_num[1:], lw=5.0, label=rf"$\eta={chi}$", color=colors[j])
    if not m_psi_labeled:
        ax2.loglog(R[1:], SF_int[1:], ls="-.", lw=3.5, color=color_m_psi, alpha=0.6, label=r"$m_\psi$ slope")
        m_psi_labeled = True
    else:
        ax2.loglog(R[1:], SF_int[1:], ls="-.", lw=3.5, color=color_m_psi, alpha=0.6)
    if not m_phi_labeled:
        ax2.loglog(R[1:], SF_far[1:], ls="-.", lw=3.5, color=color_m_phi, alpha=0.6, label=r"$m_\phi$ slope")
        m_phi_labeled = True
    else:
        ax2.loglog(R[1:], SF_far[1:], ls="-.", lw=3.5, color=color_m_phi, alpha=0.6)
    if R_x_SF is not None and np.isfinite(R_x_SF) and (R.min() < R_x_SF < R.max()):
        ax2.axvline(R_x_SF, ls="-.", lw=3.5, color=colors[j], alpha=0.7)

ax2.set_xlabel(r"$R/r_i$", fontsize=40)
ax2.set_ylabel(r"$\frac{\mathrm{d}P}{\mathrm{d}\lambda^2}$", fontsize=40)
ax2.set_xlim(3e-4, 1e1)
ax2.set_ylim(1e-12, 6e-6)
ax2.legend(fontsize=32, frameon=True, fancybox=True, shadow=False)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
ax2.tick_params(axis='both', which='major', labelsize=36, width=2.5, length=10)
ax2.tick_params(axis='both', which='minor', labelsize=32, width=1.5, length=6)

plt.tight_layout()
plt.savefig("dP_dlam2_three_chi1.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("dP_dlam2_three_chi1.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
