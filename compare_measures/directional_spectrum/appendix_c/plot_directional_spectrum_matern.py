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


# -----------------------------
# helpers: fitting + predictions
# -----------------------------

def fit_loglog_slope(k, Pk, kmin, kmax):
    """Fit Pk ~ k^s on [kmin,kmax]. Returns s (float)."""
    k = np.asarray(k)
    Pk = np.asarray(Pk)
    m = (k >= kmin) & (k <= kmax) & np.isfinite(Pk) & (Pk > 0) & np.isfinite(k)
    if m.sum() < 3:
        return np.nan
    x = np.log(k[m])
    y = np.log(Pk[m])
    s, _b = np.polyfit(x, y, 1)
    return float(s)


def pick_default_kfit(kc, nx=None, ny=None, frac_lo=0.20, frac_hi=0.55, kmin_abs=10.0, kmax_frac_nyq=0.35):
    """
    Heuristic inertial-range window:
      - ignore smallest-k (finite box + apodization)
      - ignore largest-k (Nyquist / window / noise)
    
    Parameters:
    -----------
    kc : array
        k values from the spectrum
    nx, ny : int, optional
        Map dimensions. If provided, kmax scales with Nyquist frequency.
        If None, falls back to a conservative fixed cap.
    frac_lo, frac_hi : float
        Quantile fractions for selecting k range
    kmin_abs : float
        Minimum k to consider (avoids box/apodization effects)
    kmax_frac_nyq : float
        Maximum k as fraction of Nyquist (e.g., 0.35 means k_max = 0.35 * k_nyq)
    """
    kc = np.asarray(kc)
    kc = kc[np.isfinite(kc)]
    if kc.size < 10:
        return float(np.nanmin(kc)), float(np.nanmax(kc))
    
    # Compute Nyquist-based cap if dimensions provided
    if nx is not None and ny is not None:
        k_nyq = 0.5 * min(nx, ny)
        kmax_abs = kmax_frac_nyq * k_nyq
    else:
        # Fallback: conservative fixed cap (for backward compatibility)
        kmax_abs = 200.0
        if nx is None and ny is None:
            warnings.warn("pick_default_kfit: nx/ny not provided, using fixed kmax=200.0. "
                         "For better results, provide map dimensions.")
    
    k_lo = max(kmin_abs, np.quantile(kc, frac_lo))
    k_hi = min(kmax_abs, np.quantile(kc, frac_hi))
    if k_hi <= k_lo:
        k_lo = max(kmin_abs, np.quantile(kc, 0.15))
        k_hi = min(kmax_abs, np.quantile(kc, 0.65))
    return float(k_lo), float(k_hi)


def predict_directional_slope_vs_chi(
    chi_values,
    *,
    M_i,
    R_i,
    # thick
    alpha_phi_thick,
    r_phi_thick,
    # thin
    alpha_phi_thin=None,
    r_phi_thin=None,
    L_thin=None,
    # fit window (in k)
    kfit_min,
    kfit_max,
    # model knobs
    randomize_thresh=1.0,
    use_thin_piecewise=True,
    smooth=True,
    smooth_decay_scale=2.0,
):
    """
    Predict the *measured* slope in a fixed k-fit window, using a minimal
    analytic model for the direction field u = exp(i*theta), theta = arg(P)=2chi.

    Core idea (small-angle regime):
      D_u(R) ~ D_theta(R) = D_i(R) + D_F(R)
    with
      D_i(R)   ~ (R/R_i)^{alpha_i},   alpha_i = M_i
      D_F(R)   ~ 2 * chi^2 * (R/r_phi)^{alpha_phi}   (R<r_phi), saturating at 2*chi^2

    Then:
      - if D_theta(R0) > randomize_thresh: slope -> 0 (random direction field)
      - else decide (or mix) between intrinsic and Faraday slopes

    Returns:
      pred_thick_slope(chi), pred_thin_slope(chi), plus diagnostic arrays.
    """
    chi_values = np.asarray(chi_values, float)

    alpha_i = float(M_i)
    s_src = -(alpha_i + 2.0)  # consistent with your s_syn = -(M_i+2)

    # choose a representative real-space scale for this k-fit window
    k0 = 10.0 ** (0.5 * (np.log10(kfit_min) + np.log10(kfit_max)))  # geometric mean
    R0 = 1.0 / k0  # pixels (up to 2pi factors; OK for regime logic)

    # intrinsic increment model normalized so D_i(R_i)=1
    def D_i(R):
        if R_i <= 0 or not np.isfinite(R_i):
            return np.nan
        return (R / R_i) ** alpha_i

    # Faraday increment model; normalized so D_phi(r_phi) ~ 2*sigma^2 -> D_theta_F(r_phi) ~ 2*chi^2
    def D_F(R, chi, alpha_phi, r_phi):
        if r_phi is None or r_phi <= 0 or not np.isfinite(r_phi):
            return np.nan
        val = 2.0 * (chi ** 2) * (R / r_phi) ** alpha_phi
        # saturation for R>=r_phi in a structure-function sense
        return min(val, 2.0 * (chi ** 2))

    # "faraday slope" for the direction field in small-angle regime:
    # if D(R) ~ R^{alpha_phi} => P(k) ~ k^{-(alpha_phi+2)}
    def s_far(alpha_phi):
        return -(alpha_phi + 2.0)

    # predictive mixing at R0
    D_i0 = D_i(R0)

    # --- thick prediction ---
    pred_th = np.full_like(chi_values, np.nan)
    Dtot_th = np.full_like(chi_values, np.nan)
    w_th = np.full_like(chi_values, np.nan)

    for j, chi in enumerate(chi_values):
        DF0 = D_F(R0, chi, float(alpha_phi_thick), float(r_phi_thick))
        Dtot = D_i0 + DF0
        Dtot_th[j] = Dtot

        # weight of Faraday in the phase increments at the probe scale
        w = DF0 / (Dtot + 1e-30)
        w_th[j] = w

        sF = s_far(alpha_phi_thick)

        if not smooth:
            if Dtot > randomize_thresh:
                pred_th[j] = 0.0
            else:
                pred_th[j] = sF if (DF0 > D_i0) else s_src
        else:
            # smooth blend + smooth suppression when directions randomize
            s_mix = (1.0 - w) * s_src + w * sF
            # decay toward 0 when Dtot grows past ~1
            decay = np.exp(-Dtot / float(smooth_decay_scale))
            pred_th[j] = s_mix * decay

    # --- thin prediction ---
    pred_tn = None
    Dtot_tn = None
    w_tn = None
    if (alpha_phi_thin is not None) and (r_phi_thin is not None):
        pred_tn = np.full_like(chi_values, np.nan)
        Dtot_tn = np.full_like(chi_values, np.nan)
        w_tn = np.full_like(chi_values, np.nan)

        for j, chi in enumerate(chi_values):
            alpha_eff = float(alpha_phi_thin)
            if use_thin_piecewise and (L_thin is not None) and np.isfinite(L_thin) and (L_thin > 0):
                # heuristic: beyond the screen thickness, "effective" scaling can flatten by ~1 power.
                # (This mirrors the A/B split in Appendix C; turn off with use_thin_piecewise=False.)
                if R0 > float(L_thin):
                    alpha_eff = max(0.05, alpha_eff - 1.0)

            DF0 = D_F(R0, chi, alpha_eff, float(r_phi_thin))
            Dtot = D_i0 + DF0
            Dtot_tn[j] = Dtot

            w = DF0 / (Dtot + 1e-30)
            w_tn[j] = w

            sF = s_far(alpha_eff)

            if not smooth:
                if Dtot > randomize_thresh:
                    pred_tn[j] = 0.0
                else:
                    pred_tn[j] = sF if (DF0 > D_i0) else s_src
            else:
                s_mix = (1.0 - w) * s_src + w * sF
                decay = np.exp(-Dtot / float(smooth_decay_scale))
                pred_tn[j] = s_mix * decay

    diagnostics = dict(
        kfit_center=k0,
        R_probe=R0,
        s_src=s_src,
        D_i_probe=D_i0,
        Dtot_th=Dtot_th,
        w_th=w_th,
        Dtot_tn=Dtot_tn,
        w_tn=w_tn,
    )
    return pred_th, pred_tn, diagnostics


# -----------------------------
# main plotting/validation
# -----------------------------

def plot_from_npz(npz_filename="validate_lp16_directional_spectrum_P_lambda.npz",
                  kfit_min=None, kfit_max=None,
                  smooth_pred=True,
                  use_thin_piecewise=True):
    if not os.path.exists(npz_filename):
        raise FileNotFoundError(f"NPZ file not found: {npz_filename}")

    print(f"Loading data from {npz_filename}...")
    data = np.load(npz_filename, allow_pickle=True)

    # required keys from your saved file
    chi_values = data['chi_values']
    n_lam = int(data['n_lam'])

    kc_th = data['kc_th']
    Pdir_th_all = data['Pdir_th_all']

    kc_tn = data['kc_tn']
    Pdir_tn_all = data['Pdir_tn_all']

    # stored reference slopes (from your original code)
    s_syn_saved = float(data['s_syn'])
    s_rm_saved = float(data['s_rm'])

    # exponents from simulation fits
    M_i = float(data['M_i'])
    tilde_m_phi = float(data['tilde_m_phi'])

    # optional geometry + SF scalings (recommended)
    def get_optional(key, default=None):
        return data[key].item() if (key in data and np.ndim(data[key]) == 0) else (data[key] if key in data else default)

    # Map dimensions (for Nyquist calculation)
    n = get_optional("n", default=None)
    if n is not None:
        nx = int(n)
        ny = int(n)
    else:
        # Try to infer from data if not saved
        nx = ny = None
        warnings.warn("Map dimensions (n) not found in NPZ. Fit window selection may be suboptimal.")

    R_i = get_optional("R_i", default=np.nan)
    r_phi_thick = get_optional("r_phi_thick", default=np.nan)
    r_phi_thin = get_optional("r_phi_thin", default=np.nan)
    L_thin = get_optional("L_thin", default=np.nan)
    L_thick = get_optional("L_thick", default=np.nan)
    alpha_phi_thick = get_optional("alpha_phi_thick", default=(tilde_m_phi + 1.0))
    alpha_phi_thin = get_optional("alpha_phi_thin", default=np.nan)

    print("Data loaded.")
    print(f"  n_lam = {n_lam}")
    print(f"  M_i = {M_i:.3f}  (=> s_src = -(M_i+2) = {-(M_i+2):.3f})")
    print(f"  tilde_m_phi = {tilde_m_phi:.3f}")
    print(f"  alpha_phi_thick = {float(alpha_phi_thick):.3f}")
    if np.isfinite(alpha_phi_thin):
        print(f"  alpha_phi_thin  = {float(alpha_phi_thin):.3f}")
    print(f"  R_i = {float(R_i):.2f}, r_phi_thick={float(r_phi_thick):.2f}, r_phi_thin={float(r_phi_thin):.2f}, L_thin={float(L_thin):.2f}")

    # choose k-fit windows
    if kfit_min is None or kfit_max is None:
        kfit_min_th, kfit_max_th = pick_default_kfit(kc_th, nx=nx, ny=ny)
        kfit_min_tn, kfit_max_tn = pick_default_kfit(kc_tn, nx=nx, ny=ny)
    else:
        kfit_min_th, kfit_max_th = float(kfit_min), float(kfit_max)
        kfit_min_tn, kfit_max_tn = float(kfit_min), float(kfit_max)

    # Compute diagnostics for the fit window
    k0_th = 10.0 ** (0.5 * (np.log10(kfit_min_th) + np.log10(kfit_max_th)))
    R0_th = 1.0 / k0_th
    k0_tn = 10.0 ** (0.5 * (np.log10(kfit_min_tn) + np.log10(kfit_max_tn)))
    R0_tn = 1.0 / k0_tn
    
    if nx is not None and ny is not None:
        k_nyq = 0.5 * min(nx, ny)
        print(f"\nMap dimensions: {nx}×{ny} → k_Nyquist ≈ {k_nyq:.1f}")
    else:
        k_nyq = None
    
    print(f"Using k-fit window (thick): [{kfit_min_th:.2f}, {kfit_max_th:.2f}]")
    print(f"  → k_probe ≈ {k0_th:.2f}, R_probe ≈ {R0_th:.4f} pixels")
    if k_nyq is not None:
        print(f"  → k_max/k_Nyq = {kfit_max_th/k_nyq:.3f} (safe if < 0.4)")
    if R0_th < 1.0:
        warnings.warn(f"  ⚠️  R_probe ({R0_th:.4f}) < 1 pixel! Fit window may be in discretization regime.")
    
    print(f"Using k-fit window (thin):  [{kfit_min_tn:.2f}, {kfit_max_tn:.2f}]")
    print(f"  → k_probe ≈ {k0_tn:.2f}, R_probe ≈ {R0_tn:.4f} pixels")
    if k_nyq is not None:
        print(f"  → k_max/k_Nyq = {kfit_max_tn/k_nyq:.3f} (safe if < 0.4)")
    if R0_tn < 1.0:
        warnings.warn(f"  ⚠️  R_probe ({R0_tn:.4f}) < 1 pixel! Fit window may be in discretization regime.")

    # ---- 1) measure slopes vs chi from the saved spectra ----
    slopes_th = np.array([fit_loglog_slope(kc_th, Pdir_th_all[i], kfit_min_th, kfit_max_th) for i in range(len(chi_values))])
    slopes_tn = np.array([fit_loglog_slope(kc_tn, Pdir_tn_all[i], kfit_min_tn, kfit_max_tn) for i in range(len(chi_values))])

    # ---- 2) analytic slope predictions vs chi (direction-field model) ----
    # sanity checks for optional geometry
    if not (np.isfinite(R_i) and R_i > 0):
        warnings.warn("R_i missing/invalid in NPZ; predictions will be unreliable. Save R_i in the NPZ for best results.")
    if not (np.isfinite(r_phi_thick) and r_phi_thick > 0):
        warnings.warn("r_phi_thick missing/invalid in NPZ; predictions will be unreliable. Save r_phi_thick in the NPZ.")

    pred_th, pred_tn, diag_th = predict_directional_slope_vs_chi(
        chi_values,
        M_i=M_i,
        R_i=float(R_i),
        alpha_phi_thick=float(alpha_phi_thick),
        r_phi_thick=float(r_phi_thick),
        alpha_phi_thin=(float(alpha_phi_thin) if np.isfinite(alpha_phi_thin) else None),
        r_phi_thin=(float(r_phi_thin) if np.isfinite(r_phi_thin) else None),
        L_thin=(float(L_thin) if np.isfinite(L_thin) else None),
        kfit_min=kfit_min_th,  # use thick window to define the probe scale
        kfit_max=kfit_max_th,
        smooth=bool(smooth_pred),
        use_thin_piecewise=bool(use_thin_piecewise),
    )

    # ---- 3) plot the original spectra panels (as before, but without old reference arrays) ----
    fig_spec, (ax_th, ax_tn) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(chi_values) - 1)) for i in range(len(chi_values))]

    # thick spectra
    ax = ax_th
    for i, chi in enumerate(chi_values):
        ax.loglog(kc_th, Pdir_th_all[i], '-', lw=0.5, color=colors[i])
    ax.axvspan(kfit_min_th, kfit_max_th, alpha=0.12, label="fit window")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P_{\mathrm{dir}}(k)$")
    ax.set_title(r"Directional spectrum of $P(X,\lambda^2)$: thick")
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=10)

    # thin spectra
    ax = ax_tn
    for i, chi in enumerate(chi_values):
        ax.loglog(kc_tn, Pdir_tn_all[i], '-', lw=0.5, color=colors[i])
    ax.axvspan(kfit_min_tn, kfit_max_tn, alpha=0.12, label="fit window")
    ax.set_xlabel(r"$k$")
    ax.set_title(r"Directional spectrum of $P(X,\lambda^2)$: thin")
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("validate_lp16_directional_spectrum_P_lambda_matern.png", dpi=300, bbox_inches="tight")
    plt.savefig("validate_lp16_directional_spectrum_P_lambda_matern.svg", dpi=300, bbox_inches="tight")
    print("\nSaved spectra plot: validate_lp16_directional_spectrum_P_lambda_matern.png/svg")

    # ---- 4) plot slope-vs-chi: measured vs predicted (THIS is the analytic validation) ----
    # sort by chi for a cleaner curve
    order = np.argsort(chi_values)
    chi_s = chi_values[order]
    s_th_s = slopes_th[order]
    s_tn_s = slopes_tn[order]
    p_th_s = pred_th[order]
    p_tn_s = pred_tn[order] if pred_tn is not None else None

    # asymptotes (for context)
    s_src = -(M_i + 2.0)
    s_far_th = -(float(alpha_phi_thick) + 2.0)  # direction-field faraday asymptote (small-angle)
    # your previously used “RM ref” in the spectrum plot (kept for comparison)
    s_rm_old = s_rm_saved

    fig_s, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # thick slope panel
    ax = ax1
    ax.semilogx(chi_s, s_th_s, 'o', ms=4, label="measured slope (thick)")
    ax.semilogx(chi_s, p_th_s, '-', lw=2, label="predicted slope (thick)")
    ax.axhline(s_src, ls='--', lw=1.5, label=fr"$s_{{src}}=-(M_i+2)={s_src:.2f}$")
    ax.axhline(s_far_th, ls='--', lw=1.5, label=fr"$s_{{F}}=-(\alpha_\phi+2)={s_far_th:.2f}$")
    ax.axhline(0.0, ls=':', lw=1.2, label="randomized limit (0)")
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"fit slope $s$ in $P_{\rm dir}\propto k^s$")
    ax.set_title("Thick: slope vs χ")
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=10)

    # thin slope panel
    ax = ax2
    ax.semilogx(chi_s, s_tn_s, 'o', ms=4, label="measured slope (thin)")
    if p_tn_s is not None:
        ax.semilogx(chi_s, p_tn_s, '-', lw=2, label="predicted slope (thin)")
    ax.axhline(s_src, ls='--', lw=1.5, label=fr"$s_{{src}}={s_src:.2f}$")
    if np.isfinite(alpha_phi_thin):
        s_far_tn = -(float(alpha_phi_thin) + 2.0)
        ax.axhline(s_far_tn, ls='--', lw=1.5, label=fr"$s_{{F}}=-(\alpha_{{\phi,thin}}+2)={s_far_tn:.2f}$")
    ax.axhline(s_rm_old, ls='-.', lw=1.2, label=fr"(old ref) $-(\tilde m_\phi+2)={s_rm_old:.2f}$")
    ax.axhline(0.0, ls=':', lw=1.2, label="randomized limit (0)")
    ax.set_xlabel(r"$\chi$")
    ax.set_title("Thin: slope vs χ")
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("validate_lp16_directional_slope_vs_chi.png", dpi=300, bbox_inches="tight")
    plt.savefig("validate_lp16_directional_slope_vs_chi.svg", dpi=300, bbox_inches="tight")
    print("Saved slope validation plot: validate_lp16_directional_slope_vs_chi.png/svg")

    # ---- 5) quick textual diagnostics ----
    print("\nDiagnostics (prediction model):")
    print(f"  k_probe (geom mean of fit window) = {diag_th['kfit_center']:.3f}")
    print(f"  R_probe = 1/k_probe = {diag_th['R_probe']:.5f} pixels (scale-proxy)")
    print(f"  intrinsic D_i(R_probe) = {diag_th['D_i_probe']:.3e}")
    print(f"  predicted src slope = {diag_th['s_src']:.3f}")
    print("  Notes:")
    print("    - Prediction uses a minimal u=exp(i theta) model + small-angle logic.")
    print("    - If you want less/greater randomization, adjust randomize_thresh and/or smooth_decay_scale.")
    print("    - For thin screen, use_thin_piecewise toggles the heuristic A/B exponent drop.")

    plt.show()


if __name__ == "__main__":
    import sys

    npz_file = sys.argv[1] if len(sys.argv) > 1 else "validate_lp16_directional_spectrum_P_lambda.npz"

    # optional CLI: python script.py file.npz kmin kmax
    kmin = float(sys.argv[2]) if len(sys.argv) > 2 else None
    kmax = float(sys.argv[3]) if len(sys.argv) > 3 else None

    plot_from_npz(
        npz_file,
        kfit_min=kmin,
        kfit_max=kmax,
        smooth_pred=True,          # set False for hard-regime (step) predictions
        use_thin_piecewise=True,   # set False to use alpha_phi_thin everywhere
    )
