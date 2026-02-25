from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Journal-style plotting setup
# ----------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})


def _slope_guide(k, E, exponent=-11/3, frac_lo=0.07, frac_hi=0.35, scale=1.2):
    klo = max(1, int(len(k) * frac_lo))
    khi = max(klo + 1, int(len(k) * frac_hi))
    kseg = k[klo:khi]
    kref = np.median(kseg) if len(kseg) else k[max(1, len(k) // 3)]
    Eref = np.interp(kref, k, E)
    A = scale * Eref / (kref ** exponent)
    return A * (k ** exponent)


def fit_power_law_small_k(k, spec, k_min=20, k_max=200, min_points=5):
    """
    Fit a power law spec(k) = A * k^exponent to the small k region.
    
    Parameters:
    -----------
    k : np.ndarray
        Wavenumber array
    spec : np.ndarray
        Spectrum array
    k_min : float
        Minimum k value for fitting range
    k_max : float
        Maximum k value for fitting range
    min_points : int
        Minimum number of points to use for fitting
    
    Returns:
    --------
    A : float
        Amplitude of the power law
    exponent : float
        Exponent of the power law
    k_fit : np.ndarray
        k values used for fitting
    spec_fit : np.ndarray
        Fitted power law values
    """
    # Select k range from k_min to k_max
    mask = (k >= k_min) & (k <= k_max)
    
    if mask.sum() < min_points:
        # Not enough points in range, return None
        return None, None, None, None
    
    k_small = k[mask]
    spec_small = spec[mask]
    
    # Filter out invalid values
    valid = (k_small > 0) & (spec_small > 0) & np.isfinite(k_small) & np.isfinite(spec_small)
    if valid.sum() < 3:
        # Not enough valid points, return None
        return None, None, None, None
    
    k_fit = k_small[valid]
    spec_fit = spec_small[valid]
    
    # Fit in log-log space: log(spec) = log(A) + exponent * log(k)
    log_k = np.log(k_fit)
    log_spec = np.log(spec_fit)
    
    # Linear regression
    coeffs = np.polyfit(log_k, log_spec, 1)
    exponent = coeffs[0]
    log_A = coeffs[1]
    A = np.exp(log_A)
    
    # Generate fitted curve over the fitting range
    k_fit_range = np.linspace(k_fit.min(), k_fit.max(), 100)
    spec_fit_curve = A * (k_fit_range ** exponent)
    
    return A, exponent, k_fit_range, spec_fit_curve


def _normalize_npz_data(data):
    """
    Normalize NPZ data to common (k, spec, kind, meta).

    Supported formats:
      A) old:         keys: k, Pk
      B) PSA:         keys: k, Pk_lambda, lambda2_list
      C) directional: keys: k_centers, E_u (and optionally lam2/eta)
      D) neB:         keys: k_centers, E_neB

    Returns
    -------
    k : (n_k,)
    spec : (n_k,) or (n_lambda, n_k)
    Lxy : float or None
    param_list : np.ndarray or None   (lambda2_list, lam2, or eta; only for multi-spectra)
    kind : str  in {"old","psa","directional","neB"}
    meta : dict  (extra fields you might want)
    """
    keys = set(data.keys())
    meta = {}

    # D) neB format
    if "k_centers" in keys and "E_neB" in keys:
        k = np.asarray(data["k_centers"])
        spec = np.asarray(data["E_neB"])
        # Single spectrum; keep it 1D
        Lxy = data.get("Lxy", None)
        if Lxy is None:
            # You stored dx,dy; dx alone is not Lxy, but keep for legacy compatibility
            meta["dx"] = float(data.get("dx", np.nan))
            meta["dy"] = float(data.get("dy", np.nan))
        meta["case_tag"] = str(data.get("case_tag", ""))
        return k, spec, Lxy, None, "neB", meta

    # C) directional spectrum from your u-field code
    if "k_centers" in keys and "E_u" in keys:
        k = np.asarray(data["k_centers"])
        spec = np.asarray(data["E_u"])
        # E_u can be (n_eta, n_k) or (n_k,)
        if spec.ndim == 2:
            # treat as multi-spectra; param could be eta or lam2 if present
            if "eta" in keys:
                param_list = np.asarray(data["eta"])
            elif "lam2" in keys:
                param_list = np.asarray(data["lam2"])
            else:
                param_list = np.arange(spec.shape[0])
            Lxy = data.get("Lxy", None)
            return k, spec, Lxy, param_list, "psa", meta
        else:
            Lxy = data.get("Lxy", None)
            return k, spec, Lxy, None, "directional", meta

    # B) PSA format
    if "k" in keys and "Pk_lambda" in keys and "lambda2_list" in keys:
        k = np.asarray(data["k"])
        spec = np.asarray(data["Pk_lambda"])
        Lxy = data.get("Lxy", None)
        param_list = np.asarray(data["lambda2_list"])
        return k, spec, Lxy, param_list, "psa", meta

    # A) old format
    if "k" in keys and "Pk" in keys:
        k = np.asarray(data["k"])
        spec = np.asarray(data["Pk"])
        Lxy = data.get("Lxy", None)
        return k, spec, Lxy, None, "old", meta

    raise KeyError(
        f"Unknown NPZ format. Found keys: {list(data.keys())}. "
        f"Expected one of: "
        f"(k,Pk), (k,Pk_lambda,lambda2_list), (k_centers,E_u), (k_centers,E_neB)."
    )


def plot_multiple_spectra(
    spectra_list,
    out_prefix="spectrum_comparison",
    ylabel=r"$E(k)$",
    show_slope=True,
    slope_exp=-11/3,
):
    plt.figure(figsize=(7.0, 4.0))
    ax = plt.subplot(1, 1, 1)

    k_min = np.inf
    k_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    for k, spec, label, color in spectra_list:
        ax.loglog(k, spec, label=label, color=color, linewidth=2.5)
        
        # Fit and plot power law for small k (k from 20 to 200)
        A, exponent, k_fit, spec_fit = fit_power_law_small_k(k, spec, k_min=30, k_max=200)
        if A is not None and k_fit is not None:
            # Create label with fitted exponent in $\propto k^{x}$ format
            fit_label = f"$\\propto k^{{{exponent:.2f}}}$"
            ax.loglog(k_fit, spec_fit, linestyle=":", linewidth=2.0, 
                     alpha=0.7, color=color, label=fit_label)
            print(f"  Fit for {label} (k: 20-200): E(k) = {A:.2e} * k^{exponent:.3f}")
        
        k_min = min(k_min, float(np.min(k)))
        k_max = max(k_max, float(np.max(k)))
        y_min = min(y_min, float(np.min(spec)))
        y_max = max(y_max, float(np.max(spec)))

    if show_slope and len(spectra_list) > 0:
        k0, s0 = spectra_list[0][0], spectra_list[0][1]
        g = _slope_guide(k0, s0, exponent=slope_exp)
        ax.loglog(k0, g, linestyle="--", linewidth=3.5, label=rf"$k^{{{slope_exp:.3g}}}$", color="black")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left")
    plt.tight_layout()

    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()
    return {"png": png, "svg": svg}


def main():
    # Update these paths for your machine
    npz_paths = [
        Path(r"D:\Downloads\2_12_2026\ne_b\neB_spectrum_ms1.npz"),
        Path(r"D:\Downloads\2_12_2026\ne_b\neB_spectrum_ms10.npz"),
        Path(r"D:\Downloads\2_12_2026\ne_b\neB_spectrum_ms10_plus_ms1.npz"),
    ]

    labels = [
        r"$M_s=1,\ M_A=0.8$",
        r"$M_s=10,\ M_A=0.8,\ t=0.25$",
        r"$M_s=10,\ M_A=0.8,\ t=0.25$ \& $M_s=1,\ M_A=0.8$",
    ]
    colors = ["blue", "red", "green"]

    spectra_list = []

    # Your previous filter choice
    k_min_cut = 2 * np.pi
    k_max_cut = np.pi * 512

    for i, npz_path in enumerate(npz_paths):
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find {npz_path}")

        data = np.load(npz_path)
        k, spec, Lxy, param_list, kind, meta = _normalize_npz_data(data)

        # For PSA-like multi spectra, pick the first by default
        if kind == "psa" and spec.ndim == 2:
            spec = spec[0]

        # Filter k-range
        mask = (k >= k_min_cut) & (k <= k_max_cut)
        k_f = k[mask]
        s_f = spec[mask]

        label = labels[i] if i < len(labels) else f"Spectrum {i+1}"
        color = colors[i % len(colors)]
        spectra_list.append((k_f, s_f, label, color))

        print(f"Loaded {npz_path.name}: {len(k_f)} points, kind={kind}")

    out = plot_multiple_spectra(
        spectra_list,
        out_prefix="neB_spectrum_comparison",
        ylabel=r"$E_{n_e B}(k)$",
        show_slope=True,
        slope_exp=-11/3,
    )
    print("Saved:", out["png"], "and", out["svg"])


if __name__ == "__main__":
    main()
