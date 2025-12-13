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


# ---- 3-asymptote analytic slope model ----
def seff3(chi, chi1, chi2, spsi, sphi, p=2.0):
    chi = np.asarray(chi, float)
    w1 = (chi / chi1) ** p
    w2 = (chi / chi2) ** p
    return (spsi + sphi * w1 + 0.0 * w2) / (1.0 + w1 + w2)


def fit_chi1_chi2_grid(chi, s_meas, spsi, sphi, p=2.0, ngrid=140):
    """
    Fit (chi1, chi2) by 2D grid search in log-space.
    Returns best (chi1, chi2) and best MSE.
    """
    chi = np.asarray(chi, float)
    s_meas = np.asarray(s_meas, float)
    m = np.isfinite(chi) & np.isfinite(s_meas) & (chi > 0)
    chi = chi[m]; s_meas = s_meas[m]
    if chi.size < 8:
        return np.nan, np.nan, np.inf

    lo = np.log10(chi.min()) - 2.0
    hi = np.log10(chi.max()) + 2.0
    grid = np.logspace(lo, hi, ngrid)

    best_mse = np.inf
    best = (np.nan, np.nan)
    # cheap-ish 2D search: ngrid^2 ~ 2e4 if ngrid=140 (OK)
    for chi1 in grid:
        # precompute w1 contribution for speed
        w1 = (chi / chi1) ** p
        for chi2 in grid:
            w2 = (chi / chi2) ** p
            s_pred = (spsi + sphi * w1) / (1.0 + w1 + w2)
            mse = np.mean((s_meas - s_pred) ** 2)
            if mse < best_mse:
                best_mse = mse
                best = (chi1, chi2)

    return float(best[0]), float(best[1]), float(best_mse)


def plot_from_npz(npz_filename="validate_lp16_directional_spectrum_P_lambda.npz",
                  kfit_min=None, kfit_max=None,
                  save_prefix="validate_lp16_directional_spectrum_P_lambda_matern",
                  p_mix=2.0):
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

    # asymptotes you used before:
    spsi = float(data["s_syn"])  # = -(M_i+2)
    sphi = float(data["s_rm"])   # your "RM-like" reference (may be regime-dependent)

    print("Data loaded successfully.")
    print(f"  n = {n}, n_lam = {n_lam}")
    print(f"  M_i = {M_i:.3f} -> spsi = {spsi:.3f}")
    print(f"  tilde_m_phi = {tilde_m_phi:.3f} -> sphi = {sphi:.3f}")

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

    # --- plot spectra (same as before, with fit window shading)
    fig2, (ax_th, ax_tn) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_lam - 1)) for i in range(n_lam)]

    ax = ax_th
    for i in range(len(chi_values)):
        ax.loglog(kc_th, Pdir_th_all[i], "-", lw=0.5, color=colors[i])
    ax.axvspan(kfit_min_th, kfit_max_th, alpha=0.12, label="fit window")
    ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$P_{\rm dir}(k)$")
    ax.set_title("Directional spectrum: thick")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    ax = ax_tn
    for i in range(len(chi_values)):
        ax.loglog(kc_tn, Pdir_tn_all[i], "-", lw=0.5, color=colors[i])
    ax.axvspan(kfit_min_tn, kfit_max_tn, alpha=0.12, label="fit window")
    ax.set_xlabel(r"$k$")
    ax.set_title("Directional spectrum: thin")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.svg", dpi=300, bbox_inches="tight")
    print(f"\nSaved spectra plot to {save_prefix}.png/svg")

    # --- measure slope vs chi
    s_meas_th = np.array([fit_loglog_slope(kc_th, Pdir_th_all[i], kfit_min_th, kfit_max_th) for i in range(len(chi_values))])
    s_meas_tn = np.array([fit_loglog_slope(kc_tn, Pdir_tn_all[i], kfit_min_tn, kfit_max_tn) for i in range(len(chi_values))])

    # --- fit (chi1, chi2) separately for thick and thin
    chi1_th, chi2_th, mse_th = fit_chi1_chi2_grid(chi_values, s_meas_th, spsi, sphi, p=p_mix)
    chi1_tn, chi2_tn, mse_tn = fit_chi1_chi2_grid(chi_values, s_meas_tn, spsi, sphi, p=p_mix)

    print("\n3-asymptote fit results (src -> RM -> white):")
    print(f"  thick: chi1={chi1_th:.3g}, chi2={chi2_th:.3g}, MSE={mse_th:.3e}")
    print(f"  thin:  chi1={chi1_tn:.3g}, chi2={chi2_tn:.3g}, MSE={mse_tn:.3e}")

    s_pred_th = seff3(chi_values, chi1_th, chi2_th, spsi, sphi, p=p_mix)
    s_pred_tn = seff3(chi_values, chi1_tn, chi2_tn, spsi, sphi, p=p_mix)

    # --- plot slope vs chi
    order = np.argsort(chi_values)
    chi_s = chi_values[order]

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax = ax1
    ax.semilogx(chi_s, s_meas_th[order], "o", ms=4, label="measured")
    ax.semilogx(chi_s, s_pred_th[order], "-", lw=2, label=fr"pred: chi1={chi1_th:.2g}, chi2={chi2_th:.2g}")
    ax.axhline(spsi, ls="--", lw=1.2, label=fr"$s_\psi={spsi:.2f}$")
    ax.axhline(sphi, ls="--", lw=1.2, label=fr"$s_\phi={sphi:.2f}$")
    ax.axhline(0.0, ls=":", lw=1.2, label=r"$s_\infty=0$")
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"slope $s$ in $P_{\rm dir}\propto k^s$")
    ax.set_title("Thick: slope vs χ")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    ax = ax2
    ax.semilogx(chi_s, s_meas_tn[order], "o", ms=4, label="measured")
    ax.semilogx(chi_s, s_pred_tn[order], "-", lw=2, label=fr"pred: chi1={chi1_tn:.2g}, chi2={chi2_tn:.2g}")
    ax.axhline(spsi, ls="--", lw=1.2, label=fr"$s_\psi={spsi:.2f}$")
    ax.axhline(sphi, ls="--", lw=1.2, label=fr"$s_\phi={sphi:.2f}$")
    ax.axhline(0.0, ls=":", lw=1.2, label=r"$s_\infty=0$")
    ax.set_xlabel(r"$\chi$")
    ax.set_title("Thin: slope vs χ")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = "validate_lp16_seff3_vs_chi.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".svg"), dpi=300, bbox_inches="tight")
    print(f"Saved slope-validation plot: {out} / .svg")
    plt.show()


if __name__ == "__main__":
    import sys
    npz_file = sys.argv[1] if len(sys.argv) > 1 else "validate_lp16_directional_spectrum_P_lambda.npz"
    kmin = float(sys.argv[2]) if len(sys.argv) > 2 else None
    kmax = float(sys.argv[3]) if len(sys.argv) > 3 else None
    plot_from_npz(npz_file, kfit_min=kmin, kfit_max=kmax, p_mix=2.0)
