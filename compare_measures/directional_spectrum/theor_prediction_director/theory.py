import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.integrate import simpson

# plt.rcParams.update({
#     'font.size': 11,
#     "font.family": "STIXGeneral",
#     "mathtext.fontset": "stix",
#     # 'font.family': 'serif',
#     # 'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
#     'text.usetex': False,
#     'axes.linewidth': 1.2,
#     'axes.labelsize': 12,
#     'axes.titlesize': 13,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14,
#     'legend.frameon': True,
#     'legend.fancybox': True,
#     'legend.framealpha': 0.9,
#     'figure.dpi': 100,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
# })

# import matplotlib as mpl

# # --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
# mpl.rcParams.update({
#     "text.usetex": False,          # use MathText (portable)
#     "font.family": "STIXGeneral",  # match math fonts
# })

def ring_average_2d(power2d, kmin=1.0, kmax=None, nbins=200):
    ny, nx = power2d.shape
    ky = np.arange(-ny//2, ny - ny//2)
    kx = np.arange(-nx//2, nx - nx//2)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)

    if kmax is None:
        kmax = K.max()

    edges = np.linspace(kmin, kmax, nbins + 1)
    kc = 0.5 * (edges[:-1] + edges[1:])
    Pk = np.zeros_like(kc)

    for i in range(nbins):
        m = (K >= edges[i]) & (K < edges[i+1])
        if np.any(m):
            Pk[i] = np.mean(power2d[m])
        else:
            Pk[i] = np.nan

    good = np.isfinite(Pk)
    return kc[good], Pk[good]

def D_P_model(R, A_P=1.0, R0=50.0, mpsi=1.0):
    x = (R / R0)**mpsi
    return 2.0 * A_P * x / (1.0 + x)

def D_Phi_model(R, sigma_RM=10.0, rphi=30.0, mPhi=1.0):
    x = (R / rphi)**mPhi
    return 2.0 * (sigma_RM**2) * x / (1.0 + x)

def radial_grid(N):
    y = np.arange(-N//2, N - N//2)
    x = np.arange(-N//2, N - N//2)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    return R

def corr_from_structure(D, C0):
    return C0 - 0.5 * D

def spectrum_from_corr(C):
    ny, nx = C.shape
    
    y = np.arange(-ny//2, ny - ny//2)
    x = np.arange(-nx//2, nx - nx//2)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    
    R_flat = R.flatten()
    C_flat = C.flatten()
    
    R_max = R_flat.max()
    R_bins = np.linspace(0, R_max, min(500, int(R_max) + 1))
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    
    C_radial = np.zeros(len(R_centers))
    for i in range(len(R_centers)):
        if i == 0:
            mask = (R_flat >= 0) & (R_flat < R_bins[1])
        else:
            mask = (R_flat >= R_bins[i]) & (R_flat < R_bins[i+1])
        if np.any(mask):
            C_radial[i] = np.mean(C_flat[mask])
    
    ky = np.arange(-ny//2, ny - ny//2)
    kx = np.arange(-nx//2, nx - nx//2)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)
    
    Pk = np.zeros_like(C, dtype=float)
    
    for i in range(ny):
        for j in range(nx):
            k = K[i, j]
            if k > 0 and len(R_centers) > 1:
                integrand = C_radial * j0(k * R_centers) * 2 * np.pi * R_centers
                Pk[i, j] = simpson(integrand, R_centers)
            else:
                Pk[i, j] = 0.0
    
    return Pk

def directional_spectrum_from_models(
    N=512,
    A_P=1.0,
    R0=55.0,
    mpsi=1.37,
    sigma_RM=108.0,
    rphi=33.8,
    mPhi=0.93,
    chi_list=(0.2, 1.0, 5.0, 20.0),
    nbins=200,
    kmin=2.0,
    save_figures=False,
    figure_prefix='directional_spectrum',
):
    R = radial_grid(N)

    DP = D_P_model(R, A_P=A_P, R0=R0, mpsi=mpsi)
    DPH = D_Phi_model(R, sigma_RM=sigma_RM, rphi=rphi, mPhi=mPhi)

    xi_P = corr_from_structure(DP, C0=1.0)
    xi_Phi = corr_from_structure(DPH, C0=sigma_RM**2)

    P_ui_2d = spectrum_from_corr(xi_P)
    P_Phi_2d = spectrum_from_corr(xi_Phi)

    kc_ui, P_ui = ring_average_2d(P_ui_2d, kmin=kmin, kmax=N/2, nbins=nbins)
    kc_ph, P_ph = ring_average_2d(P_Phi_2d, kmin=kmin, kmax=N/2, nbins=nbins)

    results = []

    for chi in chi_list:
        lam2 = chi / (2.0 * sigma_RM)
        lam4 = lam2**2
        C_dir = xi_P * np.exp(-2.0 * lam4 * DPH)
        Pdir_2d = spectrum_from_corr(C_dir)
        kc, Pdir = ring_average_2d(Pdir_2d, kmin=kmin, kmax=N/2, nbins=nbins)
        results.append((chi, kc, Pdir))

    fig1, ax1 = plt.subplots(figsize=(6.5, 5.0))

    chi_values = np.array([chi for chi, _, _ in results])
    if len(chi_values) > 1:
        chi_min = chi_values.min()*0.1
        chi_max = chi_values.max()*0.1
        if chi_max > chi_min:
            chi_shifted = chi_values - chi_min + 1e-6
            chi_norm_linear = (chi_shifted - chi_shifted.min()) / (chi_shifted.max() - chi_shifted.min())
            chi_norm_power = chi_norm_linear**0.3
            chi_norm_power = chi_norm_power * 0.9
            colors = plt.cm.plasma(chi_norm_power)
        else:
            colors = plt.cm.plasma(np.linspace(0, 0.85, len(results)))
    else:
        colors = plt.cm.plasma([0.5])
    
    k_all = []
    Pdir_all = []
    
    for idx, (chi, kc, Pdir) in enumerate(results):
        ax1.loglog(kc, Pdir, lw=2.0, label=rf"$\chi={chi:g}$", color=colors[idx])
        k_all.extend(kc)
        Pdir_all.extend(Pdir)

    if results:
        _, kc_ref, Pdir_ref = results[0]
        if len(kc_ref) > 0:
            k_mid = kc_ref[len(kc_ref) // 2]
            P_mid = Pdir_ref[len(Pdir_ref) // 2]
            k_ref = np.logspace(np.log10(kc_ref.min()), np.log10(kc_ref.max()), 500)
            P_ref_11 = P_mid * (k_ref / k_mid)**(-11.0/3.0) * 0.4
            P_ref_10 = P_mid * (k_ref / k_mid)**(-10.0/3.0) * 0.8
            ax1.loglog(k_ref, P_ref_11, 'g-.', lw=2.0, alpha=0.8, label=r"$k^{-11/3}$")
            ax1.loglog(k_ref, P_ref_10, 'r--', lw=2.0, alpha=0.8, label=r"$k^{-10/3}$")

    if k_all and Pdir_all:
        k_all = np.array(k_all)
        Pdir_all = np.array(Pdir_all)
        valid = np.isfinite(k_all) & np.isfinite(Pdir_all) & (k_all > 0) & (Pdir_all > 0)
        if np.any(valid):
            k_min = k_all[valid].min()
            k_max = k_all[valid].max()
            P_min = Pdir_all[valid].min()
            P_max = Pdir_all[valid].max()
            ax1.set_xlim(k_min, k_max)
            ax1.set_ylim(P_min, P_max)

    ax1.set_xlabel(r"$k$", fontsize=16)# (FFT index radius)
    ax1.set_ylabel(r"$P_{\rm dir}(k)$", fontsize=16)
    # ax1.set_title(r"Directional spectrum from $D_P(R)$ and $D_\Phi(R)$", fontsize=13)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()
    
    if save_figures or True:
        plt.savefig(f"{figure_prefix}_main.pdf", format='pdf')
        plt.savefig(f"{figure_prefix}_main.png", format='png')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6.5, 5.0))
    ax2.loglog(kc_ui, P_ui, lw=2.5, label=r"$P_{u_i}(k)$ from $\xi_P$", color='C0')
    ax2.loglog(kc_ph, P_ph, lw=2.5, label=r"$P_{\Phi}(k)$ from $\xi_\Phi$", color='C1')
    ax2.set_xlabel(r"$k$", fontsize=16)
    ax2.set_ylabel(r"$P(k)$", fontsize=16)
    # ax2.set_title("Reference spectra implied by the input structure functions", fontsize=13)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()
    
    if save_figures:
        plt.savefig(f"{figure_prefix}_reference.pdf", format='pdf')
        plt.savefig(f"{figure_prefix}_reference.png", format='png')
    plt.show()

def mPhi_from_beta3D(beta_3d):
    m3 = beta_3d - 3.0
    tilde = min(m3, 1.0)
    return 1.0 + tilde

if __name__ == "__main__":
    chi_list = [0]
    chi_list_item = 1.0#0.5
    for i in range(6):
        chi_list.append(chi_list_item)
        chi_list_item *= 2

    beta_kolm = 11.0 / 3.0
    mPhi_kolm = mPhi_from_beta3D(beta_kolm)

    directional_spectrum_from_models(
        N=1024,
        A_P=1.0,
        R0=55.38,
        mpsi=1.373,
        sigma_RM=108.251,
        rphi=33.76,
        mPhi=mPhi_kolm,
        chi_list=chi_list,
        nbins=200,
        kmin=3.0,
        save_figures=False,
        figure_prefix='directional_spectrum',
    )
