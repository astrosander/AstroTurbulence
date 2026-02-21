import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# ============================================================
# Plot P vs dP/d(lambda^2) in two panels (same model framework)
# ============================================================

# ---------- Style ----------
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'lines.linewidth': 3.0,
    'axes.linewidth': 2.0,
    'grid.linewidth': 1.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
})

# ---------- Model parameters ----------
init_A_P   = 1.0
init_R0    = 1.0
init_r_phi = 1.0
init_m_psi = 4/3+2
init_m_phi = 2/3
init_L     = 200.0
Xi0        = 1.0
Nu         = 1200

# chi values used for comparison
num_chis = 8
chis = np.geomspace(0.0001, 0.1, num_chis)


# ============================================================
# Helper functions (same spirit as your original script)
# ============================================================

def compute_sigma_phi2_and_lambda(chi, lambda_fixed=100.0):
    """
    Keep lambda fixed and vary sigma_phi so that chi ~ 2 lambda^2 sigma_phi.
    """
    sigma_phi = chi / (2.0 * lambda_fixed**2)
    return sigma_phi**2, lambda_fixed


def xi_i(R, Xi0=1.0, r_i=1.0, m_i=2/3):
    """Intrinsic polarization correlation model."""
    x = (R / r_i) ** m_i
    return Xi0 / (1.0 + x)


def xi_phi_local(R, dz, sigma_phi2=1.0, r_phi=10.0, m_phi=2/3):
    """Local RM-density correlation model in (R, dz)."""
    rr = np.sqrt(R * R + dz * dz)
    return sigma_phi2 / (1.0 + (rr / r_phi) ** m_phi)


def make_dz_grid(L, r_phi, Nu=3000):
    """
    Hybrid log+linear dz grid for stable LOS integrals.
    """
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
    """
    LOS-integrated Faraday-depth correlation piece xi_DF(R)
    and the associated structure function D_DF(R)=xi_DF(0)-xi_DF(R).
    """
    R = np.atleast_1d(R).astype(float)
    dz = make_dz_grid(L, r_phi, Nu=Nu)
    w = 2.0 * (L - dz)

    xiR = xi_phi_local(R[:, None], dz[None, :], sigma_phi2, r_phi, m_phi)
    xi0 = xi_phi_local(0.0, dz, sigma_phi2, r_phi, m_phi)[None, :]

    xi_DF  = np.trapz(w[None, :] * xiR, dz, axis=1)
    xi_DF0 = np.trapz(w[None, :] * xi0, dz, axis=1)[0]
    D_DF   = xi_DF0 - xi_DF
    return xi_DF, D_DF


def xi_P_corr(R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=3000):
    """
    Correlation of complex polarization P at fixed lambda (simplified PSA-style model):
        xi_P(R) ~ xi_i(R) * exp[-4 lambda^4 D_DF(R)]
    """
    XiR = xi_i(R, Xi0=Xi0, r_i=r_i, m_i=m_i)
    _, D_DF = xi_DeltaPhi(R, L, r_phi, m_phi, sigma_phi2, Nu=Nu)
    expo = np.clip(-4.0 * (lam ** 4) * D_DF, -800.0, 80.0)
    return XiR * np.exp(expo)


def xi_dP_dlam2(R, lam, Xi0, r_i, m_i, L, r_phi, m_phi, sigma_phi2, Nu=3000):
    """
    Derivative correlation model (same as in your original code).
    """
    XiR = xi_i(R, Xi0=Xi0, r_i=r_i, m_i=m_i)
    xi_DF, D_DF = xi_DeltaPhi(R, L, r_phi, m_phi, sigma_phi2, Nu=Nu)

    expo = np.clip(-4.0 * (lam ** 4) * D_DF, -800.0, 80.0)
    return XiR * (xi_DF + (lam ** 4) * (D_DF ** 2)) * np.exp(expo)


def SF_from_corr(xiR):
    """
    Structure function from correlation:
        SF(R) = 2 [xi(0) - xi(R)]
    """
    xi0 = np.real(xiR[0])
    return 2.0 * (xi0 - np.real(xiR))


# ============================================================
# Main plotting routine
# ============================================================

def make_two_panel_plot():
    R = np.logspace(-4, 1, 1500)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16, 6.8), constrained_layout=True
    )

    # Create colormap from blue (eta=0) to red (higher eta)
    cmap = LinearSegmentedColormap.from_list('blue_to_red', ['#0000FF', '#FF0000'])
    # Normalize chi values logarithmically to match geometric spacing
    norm = LogNorm(vmin=chis.min(), vmax=chis.max())

    for j, chi in enumerate(chis):
        sigma_phi2, lam = compute_sigma_phi2_and_lambda(chi, lambda_fixed=100.0)

        # Match intrinsic correlation parameters to your original convention
        r_i = init_R0
        m_i = init_m_psi

        # Pure P correlation and its structure function
        xiP = xi_P_corr(
            R, lam, Xi0, r_i, m_i,
            init_L, init_r_phi, init_m_phi, sigma_phi2, Nu=Nu
        )
        SF_P = SF_from_corr(xiP)

        # Derivative correlation and its structure function
        xid = xi_dP_dlam2(
            R, lam, Xi0, r_i, m_i,
            init_L, init_r_phi, init_m_phi, sigma_phi2, Nu=Nu
        )
        SF_d = SF_from_corr(xid)

        # Skip the first point (R=smallest) if it is zero/negative for log scale
        mP = np.isfinite(SF_P) & (SF_P > 0)
        md = np.isfinite(SF_d) & (SF_d > 0)

        # Get color from colormap based on chi value
        color = cmap(norm(chi))
        axL.loglog(R[mP], SF_P[mP], color=color, lw=3.8, label=rf"$\eta={chi:.3f}$")
        axR.loglog(R[md], SF_d[md], color=color, lw=3.8, label=rf"$\eta={chi:.3f}$")

    # ---- Left panel: P ----
    axL.set_title(r"Pure polarization $P$")
    axL.set_xlabel(r"$R/r_i$")
    axL.set_ylabel(r"$\mathrm{SF}[P] = 2\,[\xi_P(0)-\xi_P(R)]$")
    axL.set_xlim(3e-4, 1e1)
    axL.grid(True, which='both', alpha=0.25, linestyle='--')
    axL.legend(frameon=True, fancybox=True)

    # ---- Right panel: derivative ----
    axR.set_title(r"Derivative $\mathrm{d}P/\mathrm{d}\lambda^2$")
    axR.set_xlabel(r"$R/r_i$")
    axR.set_ylabel(r"$\mathrm{SF}\!\left[\frac{\mathrm{d}P}{\mathrm{d}\lambda^2}\right]$")
    axR.set_xlim(3e-4, 1e1)
    axR.grid(True, which='both', alpha=0.25, linestyle='--')
    axR.legend(frameon=True, fancybox=True)

    # Optional: uncomment to force same y-range style across panels
    axL.set_ylim(1e-11, 1e1)
    axR.set_ylim(1e-20, 1e-6)
    
    # Add reference lines for m_phi and m_psi
    # Reference lines show power-law slopes on log-log plots
    R_ref = np.logspace(-3, 0, 100)  # Reference line x-range
    
    # For m_psi reference line (intrinsic polarization index)
    # Structure function scales as R^m_psi in the appropriate regime
    y_ref_psi_L = 1e-10 * (R_ref / R_ref[0]) ** init_m_psi
    y_ref_psi_R = 1e-19 * (R_ref / R_ref[0]) ** init_m_psi
    
    # For m_phi reference line (RM-density index + 1)
    m_phi_plus_1 = init_m_phi + 1
    y_ref_phi_L = 1e-10 * (R_ref / R_ref[0]) ** m_phi_plus_1
    y_ref_phi_R = 1e-19 * (R_ref / R_ref[0]) ** m_phi_plus_1
    
    # Plot reference lines on both panels with distinct colors
    axL.loglog(R_ref, y_ref_psi_L, '--', color='purple', lw=2.5, alpha=0.7, label=rf"$m_\psi={init_m_psi:.2f}$")
    axL.loglog(R_ref, y_ref_phi_L, '-.', color='green', lw=2.5, alpha=0.7, label=rf"$m_\phi+1={m_phi_plus_1:.2f}$")
    axR.loglog(R_ref, y_ref_psi_R, '--', color='purple', lw=2.5, alpha=0.7, label=rf"$m_\psi={init_m_psi:.2f}$")
    axR.loglog(R_ref, y_ref_phi_R, '-.', color='green', lw=2.5, alpha=0.7, label=rf"$m_\phi+1={m_phi_plus_1:.2f}$")
    
    # Update legends to include reference lines
    axL.legend(frameon=True, fancybox=True)
    axR.legend(frameon=True, fancybox=True)

    # fig.suptitle(r"Comparison: $P$ vs $\mathrm{d}P/\mathrm{d}\lambda^2$", y=1.02)
    plt.savefig("P_vs_dPdlambda2_two_panel.svg", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig("P_vs_dPdlambda2_two_panel.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == "__main__":
    make_two_panel_plot()