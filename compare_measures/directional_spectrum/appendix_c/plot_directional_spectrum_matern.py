import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
})


def plot_from_npz(npz_filename="validate_lp16_directional_spectrum_P_lambda.npz"):
    if not os.path.exists(npz_filename):
        raise FileNotFoundError(f"NPZ file not found: {npz_filename}")
    
    print(f"Loading data from {npz_filename}...")
    data = np.load(npz_filename)
    
    chi_values = data['chi_values']
    lam_values_thick = data['lam_values_thick']
    lam_values_thin = data['lam_values_thin']
    n_lam = int(data['n_lam'])
    
    kc_th = data['kc_th']
    Pdir_th_all = data['Pdir_th_all']
    k_th_syn = data['k_th_syn']
    P_th_syn = data['P_th_syn']
    
    kc_tn = data['kc_tn']
    Pdir_tn_all = data['Pdir_tn_all']
    k_th_rm = data['k_th_rm']
    P_th_rm = data['P_th_rm']
    
    s_syn = float(data['s_syn'])
    s_rm = float(data['s_rm'])
    
    n = int(data['n'])
    M_i = float(data['M_i'])
    tilde_m_phi = float(data['tilde_m_phi'])
    sigma_RM_thick = float(data['sigma_RM_thick'])
    sigma_RM_thin = float(data['sigma_RM_thin'])
    
    print("Data loaded successfully.")
    print(f"  n_lam = {n_lam}")
    print(f"  n = {n}")
    print(f"  M_i = {M_i:.3f}")
    print(f"  tilde_m_phi = {tilde_m_phi:.3f}")

    fig2, (ax_th, ax_tn) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_lam - 1)) for i in range(n_lam)]

    ax = ax_th
    for i, chi in enumerate(chi_values):
        label = rf"$\chi={chi:.2f}$"
        ax.loglog(kc_th, Pdir_th_all[i], '-', lw=0.5, color=colors[i])

    ax.loglog(k_th_syn, P_th_syn, 'r--', lw=2,
              label=fr"Synch ref: $k^{{-2-M_i}}=k^{{{s_syn:.2f}}}$")

    ax.set_xlim([kc_th.min(), kc_th.max()])
    ax.set_ylim([Pdir_th_all.min() * 0.8, Pdir_th_all.max() * 1.2])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P_{\mathrm{dir}}(k)$")
    ax.set_title(r"Directional spectrum of $P(X,\lambda^2)$: thick screen")
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=10, ncol=2)

    ax = ax_tn
    for i, chi in enumerate(chi_values):
        label = rf"$\chi={chi:.2f}$"
        ax.loglog(kc_tn, Pdir_tn_all[i], '-', lw=0.5, color=colors[i])

    ax.loglog(k_th_rm, P_th_rm, 'r--', lw=2,
              label=fr"RM ref: $k^{{-2-\tilde m_\phi}}=k^{{{s_rm:.2f}}}$")

    ax.set_xlim([kc_tn.min(), kc_tn.max()])
    ax.set_ylim([Pdir_tn_all.min() * 0.8, Pdir_tn_all.max() * 1.2])
    ax.set_xlabel(r"$k$")
    ax.set_title(r"Directional spectrum of $P(X,\lambda^2)$: thin screen")
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig("validate_lp16_directional_spectrum_P_lambda_matern.png",
                dpi=300, bbox_inches="tight")
    plt.savefig("validate_lp16_directional_spectrum_P_lambda_matern.svg",
                dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to validate_lp16_directional_spectrum_P_lambda_matern.png/svg")
    plt.show()


if __name__ == "__main__":
    import sys
    npz_file = sys.argv[1] if len(sys.argv) > 1 else "validate_lp16_directional_spectrum_P_lambda.npz"
    plot_from_npz(npz_file)
