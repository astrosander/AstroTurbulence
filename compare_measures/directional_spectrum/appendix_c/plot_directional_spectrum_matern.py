import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

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


def fit_loglog_slope(k, Pk, kmin, kmax, min_pts=6):
    k = np.asarray(k, float)
    Pk = np.asarray(Pk, float)
    m = np.isfinite(k) & np.isfinite(Pk) & (Pk > 0) & (k >= kmin) & (k <= kmax)
    if m.sum() < min_pts:
        return np.nan
    x = np.log(k[m]); y = np.log(Pk[m])
    s, _ = np.polyfit(x, y, 1)
    return float(s)


def default_k_window(n, kc, frac_lo_nyq=0.03, frac_hi_nyq=0.25, kmin_floor=8.0):
    nyq = 0.5 * n
    kmin = max(kmin_floor, frac_lo_nyq * nyq)
    kmax = frac_hi_nyq * nyq
    kc = np.asarray(kc, float)
    kmin = max(kmin, np.nanmin(kc))
    kmax = min(kmax, np.nanmax(kc))
    if kmax <= kmin:
        kmin, kmax = float(np.nanmin(kc)), float(np.nanmax(kc))
    return float(kmin), float(kmax)


# ---- Corrected thick/thin screen models ----
def sphi_thick_from_tilde(tilde_m_phi):
    """Thick Faraday exponent is (1 + tilde_m_phi) in real space -> slope -(2 + (1+tilde))"""
    return -(3.0 + tilde_m_phi)

def sphi_thin_from_tilde(tilde_m_phi):
    """Thin (L<R<r_phi) branch -> slope -(2 + tilde)"""
    return -(2.0 + tilde_m_phi)

def seff_src_to_white(chi, chi2, spsi, p=2.0):
    """Monotone s: spsi -> 0 (for thick screens)."""
    chi = np.asarray(chi, float)
    w2 = (chi/chi2)**p
    return spsi / (1.0 + w2)

def seff_src_rm_white(chi, chi1, chi2, spsi, sphi, p=2.0):
    """Three-asymptote: spsi -> sphi -> 0 (for thin screens)."""
    chi = np.asarray(chi, float)
    w1 = (chi/chi1)**p
    w2 = (chi/chi2)**p
    return (spsi + sphi*w1) / (1.0 + w1 + w2)

def fit_chi2_only(chi, s_meas, spsi, p=2.0, ngrid=400):
    """Fit chi2 for thick: spsi -> 0."""
    chi = np.asarray(chi, float)
    s_meas = np.asarray(s_meas, float)
    m = np.isfinite(chi) & np.isfinite(s_meas) & (chi > 0)
    chi = chi[m]; s_meas = s_meas[m]
    if chi.size < 8:
        return np.nan, np.inf

    grid = np.logspace(np.log10(chi.min())-2, np.log10(chi.max())+2, ngrid)
    best = (np.inf, np.nan)
    for chi2 in grid:
        s_pred = seff_src_to_white(chi, chi2, spsi, p=p)
        mse = np.mean((s_meas - s_pred)**2)
        if mse < best[0]:
            best = (mse, chi2)
    return float(best[1]), float(best[0])

def fit_chi1_chi2_ordered(chi, s_meas, spsi, sphi, p=2.0, ngrid=160):
    """Fit chi1<chi2 for thin: spsi -> sphi -> 0."""
    chi = np.asarray(chi, float)
    s_meas = np.asarray(s_meas, float)
    m = np.isfinite(chi) & np.isfinite(s_meas) & (chi > 0)
    chi = chi[m]; s_meas = s_meas[m]
    if chi.size < 10:
        return np.nan, np.nan, np.inf

    grid = np.logspace(np.log10(chi.min())-2, np.log10(chi.max())+2, ngrid)

    best_mse = np.inf
    best = (np.nan, np.nan)
    for i, chi1 in enumerate(grid):
        # only allow chi2 > chi1
        for chi2 in grid[i+1:]:
            s_pred = seff_src_rm_white(chi, chi1, chi2, spsi, sphi, p=p)
            mse = np.mean((s_meas - s_pred)**2)
            if mse < best_mse:
                best_mse = mse
                best = (chi1, chi2)
    return float(best[0]), float(best[1]), float(best_mse)

def fit_with_p_scan(chi, s_meas_th, s_meas_tn, spsi, tilde_m_phi, p_list=(1.5, 2.0, 2.5, 3.0)):
    """
    Returns best fits for thick and thin, choosing p that minimizes MSE.
    Thick uses only src->white.
    Thin uses src->rm->white with rm slope appropriate to thin.
    """
    sphi_thick = sphi_thick_from_tilde(tilde_m_phi)
    sphi_thin  = sphi_thin_from_tilde(tilde_m_phi)

    best = dict(mse=np.inf)

    for p in p_list:
        # thick
        chi2_th, mse_th = fit_chi2_only(chi, s_meas_th, spsi, p=p)
        # thin
        chi1_tn, chi2_tn, mse_tn = fit_chi1_chi2_ordered(chi, s_meas_tn, spsi, sphi_thin, p=p)

        mse_tot = mse_th + mse_tn
        if mse_tot < best["mse"]:
            best = dict(
                p=p,
                # thick
                chi2_th=chi2_th, mse_th=mse_th, sphi_thick=sphi_thick,
                # thin
                chi1_tn=chi1_tn, chi2_tn=chi2_tn, mse_tn=mse_tn, sphi_thin=sphi_thin,
                mse=mse_tot
            )
    return best


def plot_from_npz(npz_filename="validate_lp16_directional_spectrum_P_lambda.npz",
                  kfit_min=None, kfit_max=None,
                  save_prefix="validate_lp16_directional_spectrum_P_lambda_matern",
                  p_list=(1.5, 2.0, 2.5, 3.0)):
    if not os.path.exists(npz_filename):
        raise FileNotFoundError(f"NPZ file not found: {npz_filename}")

    print(f"Loading data from {npz_filename}...")
    data = np.load(npz_filename, allow_pickle=True)

    chi_values = data["chi_values"]
    n_lam = int(data["n_lam"])

    kc_th = data["kc_th"]; Pdir_th_all = data["Pdir_th_all"]
    kc_tn = data["kc_tn"]; Pdir_tn_all = data["Pdir_tn_all"]

    n = int(data["n"])
    M_i = float(data["M_i"])
    tilde_m_phi = float(data["tilde_m_phi"])

    # asymptotes:
    spsi = float(data["s_syn"])  # = -(M_i+2)
    # Compute correct s_phi for thick vs thin
    sphi_thick = sphi_thick_from_tilde(tilde_m_phi)
    sphi_thin = sphi_thin_from_tilde(tilde_m_phi)

    print("Data loaded successfully.")
    print(f"  n = {n}, n_lam = {n_lam}")
    print(f"  M_i = {M_i:.3f} -> spsi = {spsi:.3f}")
    print(f"  tilde_m_phi = {tilde_m_phi:.3f}")
    print(f"    -> sphi_thick = {sphi_thick:.3f} (thick: no intermediate branch)")
    print(f"    -> sphi_thin = {sphi_thin:.3f} (thin: RM intermediate branch)")

    # choose k-fit windows
    if (kfit_min is None) or (kfit_max is None):
        kfit_min_th, kfit_max_th = default_k_window(n, kc_th)
        kfit_min_tn, kfit_max_tn = default_k_window(n, kc_tn)
    else:
        kfit_min_th = kfit_min_tn = float(kfit_min)
        kfit_max_th = kfit_max_tn = float(kfit_max)

    print(f"\nUsing k-fit window (thick): [{kfit_min_th:.2f}, {kfit_max_th:.2f}]")
    print(f"Using k-fit window (thin):  [{kfit_min_tn:.2f}, {kfit_max_tn:.2f}]")

    # sanity: enough bins?
    nbin_th = np.sum((kc_th >= kfit_min_th) & (kc_th <= kfit_max_th))
    nbin_tn = np.sum((kc_tn >= kfit_min_tn) & (kc_tn <= kfit_max_tn))
    print(f"Ring bins in window: thick={nbin_th}, thin={nbin_tn}")
    if nbin_th < 8 or nbin_tn < 8:
        warnings.warn("Too few ring points in fit window. Increase ring_bins (e.g. 192–256 for n~1024–1400) or widen window.")

    # --- plot spectra (thick only)
    # Select chi values closest to [0, 1, 2, 3, 4, 5]
    target_chis = np.array([0, 1, 2, 3, 4, 5])*2

    plot_indices = []
    for target_chi in target_chis:
        # Find index with chi value closest to target
        idx = np.argmin(np.abs(chi_values - target_chi))
        plot_indices.append(idx)
    plot_indices = np.unique(plot_indices)  # Remove duplicates if any
    
    fig2, ax_th = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(plot_indices) - 1)) for i in range(len(plot_indices))]

    for idx, i in enumerate(plot_indices):
        chi = chi_values[i]
        ax_th.loglog(kc_th, Pdir_th_all[i], "-", lw=2.5, color=colors[idx], 
                 label=fr"$\chi={chi:.2g}$")
    
    # Add black dashed reference line with slope -10/3
    k_ref = np.logspace(np.log10(kc_th.min()), np.log10(kc_th.max()), 100)
    # Normalize to match the data range (use first spectrum as reference)
    if len(plot_indices) > 0:
        P_ref_scale = np.interp(k_ref[0], kc_th, Pdir_th_all[plot_indices[0]]) * (k_ref[0] ** (10/3))
    else:
        P_ref_scale = np.interp(k_ref[0], kc_th, Pdir_th_all[0]) * (k_ref[0] ** (10/3))
    P_ref = P_ref_scale * (k_ref ** (-10/3))
    ax_th.loglog(k_ref, P_ref, "--", lw=2.0, color="red", label=r"$k^{-10/3}$")
    
    # ax_th.axvspan(kfit_min_th, kfit_max_th, alpha=0.12, label="fit window")
    ax_th.set_xlabel(r"$k$", fontsize=20); ax_th.set_ylabel(r"$P_{\rm dir}(k)$", fontsize=20)
    ax_th.set_title("Directional spectrum: thick ($m_\\phi=3, m_\\psi=11/3$)", fontsize=20)
    ax_th.tick_params(labelsize=16)
    ax_th.grid(True, which="both", ls=":")
    ax_th.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.svg", dpi=300, bbox_inches="tight")
    print(f"\nSaved spectra plot to {save_prefix}.png/svg")

    # --- measure slope vs chi
    s_meas_th = np.array([fit_loglog_slope(kc_th, Pdir_th_all[i], kfit_min_th, kfit_max_th) for i in range(len(chi_values))])
    s_meas_tn = np.array([fit_loglog_slope(kc_tn, Pdir_tn_all[i], kfit_min_tn, kfit_max_tn) for i in range(len(chi_values))])

    # --- fit with p scan (thick: src->white only, thin: src->rm->white)
    best = fit_with_p_scan(chi_values, s_meas_th, s_meas_tn, spsi, tilde_m_phi,
                           p_list=p_list)

    print("\nBest p + fits:")
    print(f"  p = {best['p']:.2f}")
    print(f"  thick: chi2={best['chi2_th']:.3g}, MSE={best['mse_th']:.3e} (src -> white)")
    print(f"  thin:  chi1={best['chi1_tn']:.3g}, chi2={best['chi2_tn']:.3g}, MSE={best['mse_tn']:.3e} (src -> RM -> white)")
    print(f"  total MSE = {best['mse']:.3e}")

    # predictions
    s_pred_th = seff_src_to_white(chi_values, best["chi2_th"], spsi, p=best["p"])
    s_pred_tn = seff_src_rm_white(chi_values, best["chi1_tn"], best["chi2_tn"],
                                  spsi, best["sphi_thin"], p=best["p"])

    # --- plot slope vs chi
    order = np.argsort(chi_values)
    chi_s = chi_values[order]

    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    ax = ax1
    ax.semilogx(chi_s, s_meas_th[order], "o", ms=4, label="measured")
    ax.semilogx(chi_s, s_pred_th[order], "-", lw=2, label=fr"pred: chi2={best['chi2_th']:.2g}, p={best['p']:.2f}")
    ax.axhline(spsi, ls="--", lw=1.2, label=fr"$s_\psi={spsi:.2f}$")
    ax.axhline(0.0, ls=":", lw=1.2, label=r"$s_\infty=0$")
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"slope $s$ in $P_{\rm dir}\propto k^s$")
    ax.set_title("Thick")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    ax = ax2
    ax.semilogx(chi_s, s_meas_tn[order], "o", ms=4, label="measured")
    ax.semilogx(chi_s, s_pred_tn[order], "-", lw=2, label=fr"pred: chi1={best['chi1_tn']:.2g}, chi2={best['chi2_tn']:.2g}, p={best['p']:.2f}")
    ax.axhline(spsi, ls="--", lw=1.2, label=fr"$s_\psi={spsi:.2f}$")
    ax.axhline(best["sphi_thin"], ls="--", lw=1.2, label=fr"$s_\phi={best['sphi_thin']:.2f}$")
    ax.axhline(0.0, ls=":", lw=1.2, label=r"$s_\infty=0$")
    ax.set_xlabel(r"$\chi$")
    ax.set_title("Thin")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = "validate_lp16_seff3_vs_chi.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    print(f"Saved slope-validation plot: {out} / .svg")
    plt.show()


if __name__ == "__main__":
    import sys
    npz_file = sys.argv[1] if len(sys.argv) > 1 else "validate_lp16_directional_spectrum_P_lambda.npz"#
    kmin = float(sys.argv[2]) if len(sys.argv) > 2 else None
    kmax = float(sys.argv[3]) if len(sys.argv) > 3 else None
    # kmin
    # kmax=100
    plot_from_npz(npz_file, kfit_min=kmin, kfit_max=kmax)
