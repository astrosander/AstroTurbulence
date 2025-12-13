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


def fit_powerlaw_slope(r, D, rmin, rmax):
    r = np.asarray(r)
    D = np.asarray(D)
    mask = (r >= rmin) & (r <= rmax) & (D > 0)
    if mask.sum() < 2:
        return np.nan
    x = np.log(r[mask])
    y = np.log(D[mask])
    s, b = np.polyfit(x, y, 1)
    return s


def matern_psd_2d_per_mode(k, nu, kappa):
    k = np.asarray(k, float)
    return (kappa ** (2.0 * nu)) / ((k * k + kappa * kappa) ** (nu + 1.0))


def local_log_slope(k, P):
    k = np.asarray(k, float)
    P = np.asarray(P, float)
    m = (k > 0) & (P > 0) & np.isfinite(P)
    k = k[m]
    P = P[m]
    lk = np.log(k)
    lP = np.log(P)
    s = np.gradient(lP, lk)
    return k, s


def estimate_kappa_from_half_slope(k, P, s_asym):
    if not np.isfinite(s_asym) or s_asym == 0:
        return np.nan

    k_s, s_loc = local_log_slope(k, P)
    if k_s.size < 5:
        return np.nan

    target = 0.5 * s_asym
    idx = np.where(s_loc <= target)[0]
    if idx.size == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return k_s[0]

    k1, k2 = k_s[j - 1], k_s[j]
    s1, s2 = s_loc[j - 1], s_loc[j]
    if s2 == s1:
        return k2
    frac = (target - s1) / (s2 - s1)
    return k1 + frac * (k2 - k1)


def fit_chi_star_from_highk_slopes(k, P_all, chi_values, s_syn, s_rm, kmin=None, kmax=None):
    chi = np.asarray(chi_values, float)
    if kmin is None:
        kmin = np.quantile(k, 0.60)
    if kmax is None:
        kmax = np.quantile(k, 0.90)

    slopes = []
    for i in range(len(chi)):
        slopes.append(fit_powerlaw_slope(k, P_all[i], kmin, kmax))
    slopes = np.asarray(slopes, float)

    s_target = 0.5 * (s_syn + s_rm)

    good = np.isfinite(slopes) & (chi > 0)
    if good.sum() < 3:
        return np.nan, slopes, (kmin, kmax), s_target

    chi_g = chi[good]
    s_g = slopes[good]

    y = s_g - s_target
    idx = np.where(y[:-1] * y[1:] <= 0)[0]
    if idx.size == 0:
        j = np.nanargmin(np.abs(y))
        return chi_g[j], slopes, (kmin, kmax), s_target

    j = idx[0]
    x1, x2 = np.log(chi_g[j]), np.log(chi_g[j + 1])
    y1, y2 = y[j], y[j + 1]
    if y2 == y1:
        return float(np.exp(x2)), slopes, (kmin, kmax), s_target
    frac = (0.0 - y1) / (y2 - y1)
    return float(np.exp(x1 + frac * (x2 - x1))), slopes, (kmin, kmax), s_target


def toy_Pdir_shape(k, chi, nu0, nuphi, kappa0, kapphi, chi_star):
    w = (chi * chi) / (chi * chi + chi_star * chi_star)
    return (1.0 - w) * matern_psd_2d_per_mode(k, nu0, kappa0) + w * matern_psd_2d_per_mode(k, nuphi, kapphi)


def normalize_at_pivot(k, model, data):
    k = np.asarray(k, float)
    model = np.asarray(model, float)
    data = np.asarray(data, float)
    k0 = 10.0 ** (0.5 * (np.log10(k.min()) + np.log10(k.max())))
    i0 = np.argmin(np.abs(k - k0))
    if model[i0] <= 0 or not np.isfinite(model[i0]):
        return model
    A = data[i0] / model[i0]
    return A * model


def make_theory_line(kc, Pk_ref, slope):
    kmin, kmax = kc.min(), kc.max()
    k0 = 10.0 ** (0.5 * (np.log10(kmin) + np.log10(kmax)))
    idx0 = np.argmin(np.abs(kc - k0))
    P0 = Pk_ref[idx0]

    k_th = np.linspace(kmin, kmax, 200)
    P_th = P0 * (k_th / k0)**slope
    return k_th, P_th


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
    
    nu0   = 0.5 * M_i
    nuphi = 0.5 * tilde_m_phi

    kappa0_hat  = estimate_kappa_from_half_slope(kc_th, Pdir_th_all[0],  s_syn)
    kapphi_hat  = estimate_kappa_from_half_slope(kc_tn, Pdir_tn_all[-1], s_rm)

    if not np.isfinite(kappa0_hat) or kappa0_hat <= 0:
        kappa0_hat = kc_th.min() * 2.0
    if not np.isfinite(kapphi_hat) or kapphi_hat <= 0:
        kapphi_hat = kc_tn.min() * 2.0

    L0_hat   = n / kappa0_hat
    Lphi_hat = n / kapphi_hat

    chi_star_hat, slopes_hi, (k_hi_min, k_hi_max), s_target = fit_chi_star_from_highk_slopes(
        kc_tn, Pdir_tn_all, chi_values, s_syn, s_rm
    )
    if not np.isfinite(chi_star_hat) or chi_star_hat <= 0:
        chi_star_hat = 1.0

    print("\nMatérn toy model parameter picks (from spectra):")
    print(f"  nu0   = M_i/2          ≈ {nu0:.3f}")
    print(f"  nuphi = tilde_m_phi/2  ≈ {nuphi:.3f}")
    print(f"  kappa0_hat  ≈ {kappa0_hat:.3f}  ->  L0_hat   = n/kappa0 ≈ {L0_hat:.2f} px")
    print(f"  kapphi_hat  ≈ {kapphi_hat:.3f}  ->  Lphi_hat = n/kapphi ≈ {Lphi_hat:.2f} px")
    print(f"  chi_*_hat   ≈ {chi_star_hat:.3f}  (from high-k slopes fit over k∈[{k_hi_min:.1f},{k_hi_max:.1f}])")
    print(f"  target slope for chi_*: (s_syn+s_rm)/2 = {s_target:.3f}")

    fig2, (ax_th, ax_tn) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_lam - 1)) for i in range(n_lam)]

    ax = ax_th
    for i, chi in enumerate(chi_values):
        label = rf"$\chi={chi:.2f}$"
        ax.loglog(kc_th, Pdir_th_all[i], '-', lw=0.5, color=colors[i])

    toy_idxs = np.unique(np.clip(
        [0,
         int(len(chi_values) * 0.33),
         np.argmin(np.abs(chi_values - chi_star_hat)),
         int(len(chi_values) * 0.67),
         len(chi_values) - 1],
        0, len(chi_values) - 1
    ))

    for j, idx in enumerate(toy_idxs):
        chi = chi_values[idx]
        model = toy_Pdir_shape(kc_th, chi, nu0, nuphi,
                               kappa0_hat, kapphi_hat, chi_star_hat)
        model = normalize_at_pivot(kc_th, model, Pdir_th_all[idx])
        lab = "Matérn toy model" if j == 0 else None
        ax.loglog(kc_th, model, 'k--', lw=1.6, alpha=0.8, label=lab)

    ax.text(0.03, 0.06,
            rf"$L_0={L0_hat:.1f}\,$px, $L_\phi={Lphi_hat:.1f}\,$px, $\chi_*={chi_star_hat:.2f}$",
            transform=ax.transAxes, fontsize=12, va='bottom')

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

    for j, idx in enumerate(toy_idxs):
        chi = chi_values[idx]
        model = toy_Pdir_shape(kc_tn, chi, nu0, nuphi,
                               kappa0_hat, kapphi_hat, chi_star_hat)
        model = normalize_at_pivot(kc_tn, model, Pdir_tn_all[idx])
        lab = "Matérn toy model" if j == 0 else None
        ax.loglog(kc_tn, model, 'k--', lw=1.6, alpha=0.8, label=lab)

    ax.text(0.03, 0.06,
            rf"$L_0={L0_hat:.1f}\,$px, $L_\phi={Lphi_hat:.1f}\,$px, $\chi_*={chi_star_hat:.2f}$",
            transform=ax.transAxes, fontsize=12, va='bottom')

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
