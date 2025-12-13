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


# -------------------------
# helpers: slopes + fit χ*
# -------------------------

def fit_loglog_slope(k, Pk, kmin, kmax, min_pts=6):
    """Fit Pk ~ k^s on [kmin,kmax] in log-log space."""
    k = np.asarray(k, float)
    Pk = np.asarray(Pk, float)
    m = np.isfinite(k) & np.isfinite(Pk) & (Pk > 0) & (k >= kmin) & (k <= kmax)
    if m.sum() < min_pts:
        return np.nan
    x = np.log(k[m])
    y = np.log(Pk[m])
    s, _b = np.polyfit(x, y, 1)
    return float(s)


def seff_model(chi, chistar, spsi, sphi):
    """s_eff(chi) = (spsi + sphi*(chi/chistar)^2) / (1 + (chi/chistar)^2)."""
    chi = np.asarray(chi, float)
    w = (chi / chistar) ** 2
    return (spsi + sphi * w) / (1.0 + w)


def fit_chistar_grid(chi, s_meas, spsi, sphi, ngrid=400):
    """
    Fit χ* by grid search in log-space:
      minimize mean squared error between s_meas and seff_model.
    """
    chi = np.asarray(chi, float)
    s_meas = np.asarray(s_meas, float)
    m = np.isfinite(chi) & np.isfinite(s_meas) & (chi > 0)
    chi = chi[m]
    s_meas = s_meas[m]
    if chi.size < 5:
        return np.nan, None

    # search χ* over a broad range around data
    lo = np.log10(chi.min()) - 2.0
    hi = np.log10(chi.max()) + 2.0
    grid = np.logspace(lo, hi, ngrid)

    best = (np.inf, np.nan)
    for chistar in grid:
        s_pred = seff_model(chi, chistar, spsi, sphi)
        mse = np.mean((s_meas - s_pred) ** 2)
        if mse < best[0]:
            best = (mse, chistar)

    chistar_best = best[1]
    return float(chistar_best), grid


def default_k_window(n, kc, frac_lo_nyq=0.03, frac_hi_nyq=0.25, kmin_floor=8.0):
    """
    Sensible default inertial-range window for large maps.
    Nyquist ~ n/2 (because your k indices run in pixels^-1 units of FFT index).
    """
    nyq = 0.5 * n
    kmin = max(kmin_floor, frac_lo_nyq * nyq)
    kmax = frac_hi_nyq * nyq

    # clamp into available kc range
    kc = np.asarray(kc, float)
    kmin = max(kmin, np.nanmin(kc))
    kmax = min(kmax, np.nanmax(kc))
    if kmax <= kmin:
        kmin = np.nanmin(kc)
        kmax = np.nanmax(kc)
    return float(kmin), float(kmax)


# -------------------------
# main
# -------------------------

def plot_from_npz(npz_filename="validate_lp16_directional_spectrum_P_lambda.npz",
                  kfit_min=None, kfit_max=None,
                  save_prefix="validate_lp16_directional_spectrum_P_lambda_matern"):
    if not os.path.exists(npz_filename):
        raise FileNotFoundError(f"NPZ file not found: {npz_filename}")

    print(f"Loading data from {npz_filename}...")
    data = np.load(npz_filename, allow_pickle=True)

    chi_values = data["chi_values"]
    n_lam = int(data["n_lam"])

    kc_th = data["kc_th"]
    Pdir_th_all = data["Pdir_th_all"]

    kc_tn = data["kc_tn"]
    Pdir_tn_all = data["Pdir_tn_all"]

    # Asymptote slopes (your saved refs)
    s_syn = float(data["s_syn"])   # = -(M_i + 2) in your generator
    s_rm  = float(data["s_rm"])    # whatever you saved as RM-like ref

    # extra diagnostics
    n = int(data["n"])
    M_i = float(data["M_i"])
    tilde_m_phi = float(data["tilde_m_phi"])

    print("Data loaded successfully.")
    print(f"  n_lam = {n_lam}")
    print(f"  n = {n}")
    print(f"  M_i = {M_i:.3f} -> s_syn(saved) = {s_syn:.3f}")
    print(f"  tilde_m_phi = {tilde_m_phi:.3f} -> s_rm(saved) = {s_rm:.3f}")

    # -------------------------
    # choose k-fit window
    # -------------------------
    if (kfit_min is None) or (kfit_max is None):
        kfit_min_th, kfit_max_th = default_k_window(n, kc_th)
        kfit_min_tn, kfit_max_tn = default_k_window(n, kc_tn)
    else:
        kfit_min_th = kfit_min_tn = float(kfit_min)
        kfit_max_th = kfit_max_tn = float(kfit_max)

    print(f"\nUsing k-fit window (thick): [{kfit_min_th:.2f}, {kfit_max_th:.2f}]")
    print(f"Using k-fit window (thin):  [{kfit_min_tn:.2f}, {kfit_max_tn:.2f}]")

    # sanity: do we have enough k bins?
    nbin_th = np.sum((kc_th >= kfit_min_th) & (kc_th <= kfit_max_th))
    nbin_tn = np.sum((kc_tn >= kfit_min_tn) & (kc_tn <= kfit_max_tn))
    if nbin_th < 6 or nbin_tn < 6:
        warnings.warn(
            f"Fit window has too few ring bins (thick={nbin_th}, thin={nbin_tn}). "
            "Increase ring_bins in generation (e.g. 128–512 for n~1400), "
            "or widen the fit window."
        )

    # -------------------------
    # 1) plot spectra (your original figure)
    # -------------------------
    fig2, (ax_th, ax_tn) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_lam - 1)) for i in range(n_lam)]

    ax = ax_th
    for i, chi in enumerate(chi_values):
        ax.loglog(kc_th, Pdir_th_all[i], "-", lw=0.5, color=colors[i])
    ax.axvspan(kfit_min_th, kfit_max_th, alpha=0.12, label="fit window")
    ax.set_xlim([kc_th.min(), kc_th.max()])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P_{\mathrm{dir}}(k)$")
    ax.set_title(r"Directional spectrum: thick")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    ax = ax_tn
    for i, chi in enumerate(chi_values):
        ax.loglog(kc_tn, Pdir_tn_all[i], "-", lw=0.5, color=colors[i])
    ax.axvspan(kfit_min_tn, kfit_max_tn, alpha=0.12, label="fit window")
    ax.set_xlim([kc_tn.min(), kc_tn.max()])
    ax.set_xlabel(r"$k$")
    ax.set_title(r"Directional spectrum: thin")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.svg", dpi=300, bbox_inches="tight")
    print(f"\nSaved spectra plot to {save_prefix}.png/svg")

    # -------------------------
    # 2) measure slope vs chi
    # -------------------------
    s_meas_th = np.array([fit_loglog_slope(kc_th, Pdir_th_all[i], kfit_min_th, kfit_max_th)
                          for i in range(len(chi_values))])
    s_meas_tn = np.array([fit_loglog_slope(kc_tn, Pdir_tn_all[i], kfit_min_tn, kfit_max_tn)
                          for i in range(len(chi_values))])

    # -------------------------
    # 3) fit χ* for the seff(χ) model
    # -------------------------
    # Use the *two asymptotes* you want to validate:
    #   sψ = s_syn  (intrinsic / synch-like)
    #   sϕ = s_rm   (RM-like)
    spsi = s_syn
    sphi = s_rm

    chistar_th, _grid_th = fit_chistar_grid(chi_values, s_meas_th, spsi, sphi)
    chistar_tn, _grid_tn = fit_chistar_grid(chi_values, s_meas_tn, spsi, sphi)

    s_pred_th = seff_model(chi_values, chistar_th, spsi, sphi) if np.isfinite(chistar_th) else np.full_like(chi_values, np.nan)
    s_pred_tn = seff_model(chi_values, chistar_tn, spsi, sphi) if np.isfinite(chistar_tn) else np.full_like(chi_values, np.nan)

    print("\nFitted χ* (one-parameter slope-mixing model):")
    print(f"  thick: χ* = {chistar_th:.4g}")
    print(f"  thin:  χ* = {chistar_tn:.4g}")

    # -------------------------
    # 4) plot slope vs chi: measured vs predicted
    # -------------------------
    order = np.argsort(chi_values)
    chi_s = chi_values[order]

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax = ax1
    ax.semilogx(chi_s, s_meas_th[order], "o", ms=4, label="measured")
    ax.semilogx(chi_s, s_pred_th[order], "-", lw=2, label=fr"pred: χ*={chistar_th:.3g}")
    ax.axhline(spsi, ls="--", lw=1.5, label=fr"$s_\psi={spsi:.2f}$")
    ax.axhline(sphi, ls="--", lw=1.5, label=fr"$s_\phi={sphi:.2f}$")
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"slope $s$ in $P_{\rm dir}(k)\propto k^s$")
    ax.set_title("Thick: slope vs χ")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    ax = ax2
    ax.semilogx(chi_s, s_meas_tn[order], "o", ms=4, label="measured")
    ax.semilogx(chi_s, s_pred_tn[order], "-", lw=2, label=fr"pred: χ*={chistar_tn:.3g}")
    ax.axhline(spsi, ls="--", lw=1.5, label=fr"$s_\psi={spsi:.2f}$")
    ax.axhline(sphi, ls="--", lw=1.5, label=fr"$s_\phi={sphi:.2f}$")
    ax.set_xlabel(r"$\chi$")
    ax.set_title("Thin: slope vs χ")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = "validate_lp16_seff_vs_chi.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".svg"), dpi=300, bbox_inches="tight")
    print(f"Saved slope-validation plot: {out} / .svg")

    plt.show()


if __name__ == "__main__":
    import sys
    npz_file = sys.argv[1] if len(sys.argv) > 1 else "validate_lp16_directional_spectrum_P_lambda.npz"

    # optional: python script.py file.npz kmin kmax
    kmin = float(sys.argv[2]) if len(sys.argv) > 2 else None
    kmax = float(sys.argv[3]) if len(sys.argv) > 3 else None
    kmin=21.0
    kmax=175.0
    plot_from_npz(npz_file, kfit_min=kmin, kfit_max=kmax)
