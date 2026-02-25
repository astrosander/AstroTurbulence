from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Journal-style plotting setup
# (same as before)
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


def fit_power_law_small_k(k, spec, k_min=30, k_max=200, min_points=5):

    mask = (k >= k_min) & (k <= k_max)

    if mask.sum() < min_points:
        return None, None, None, None

    k_small = k[mask]
    spec_small = spec[mask]

    valid = (
        (k_small > 0)
        & (spec_small > 0)
        & np.isfinite(k_small)
        & np.isfinite(spec_small)
    )

    if valid.sum() < 3:
        return None, None, None, None

    k_fit = k_small[valid]
    spec_fit = spec_small[valid]

    log_k = np.log(k_fit)
    log_spec = np.log(spec_fit)

    coeffs = np.polyfit(log_k, log_spec, 1)

    exponent = coeffs[0]
    log_A = coeffs[1]
    A = np.exp(log_A)

    k_fit_range = np.linspace(k_fit.min(), k_fit.max(), 200)
    spec_fit_curve = A * (k_fit_range ** exponent)

    return A, exponent, k_fit_range, spec_fit_curve


def main():

    # -------------------------
    # Input file
    # -------------------------
    npz_path = Path("1dens_spectrum_ms10ma08_512_w_00021.npz")

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find {npz_path}")

    data = np.load(npz_path)

    if "k_centers" not in data or "E_dens" not in data:
        raise KeyError(
            f"Unexpected keys in {npz_path.name}: {list(data.keys())}"
        )

    k = np.asarray(data["k_centers"])
    spec = np.asarray(data["E_dens"])

    # -------------------------
    # Basic filtering
    # -------------------------
    valid = (
        np.isfinite(k)
        & np.isfinite(spec)
        & (k > 0)
        & (spec > 0)
    )

    k = k[valid]
    spec = spec[valid]

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(7.0, 4.0))
    ax = plt.subplot(1, 1, 1)

    ax.loglog(
        k,
        spec,
        linewidth=2.5,
        label=r"$\Sigma n$",
        color="blue"
    )

    # Power-law fit at small k
    A, exponent, k_fit, spec_fit = fit_power_law_small_k(
        k, spec, k_min=30, k_max=200
    )

    if A is not None:

        fit_label = rf"$\propto k^{{{exponent:.2f}}}$"

        ax.loglog(
            k_fit,
            spec_fit,
            linestyle=":",
            linewidth=3.0,
            # alpha=0.7,
            label=fit_label, color="red"
        )

        print(
            f"Fit (k=30–200): E(k) = {A:.2e} * k^{exponent:.3f}"
        )

    # Reference slope
    guide = _slope_guide(k, spec, exponent=-11/3)

    ax.loglog(
        k,
        guide,
        linestyle="-.",
        linewidth=1,
        color="black",
        label=r"$k^{-11/3}$", 
    )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E_{\Sigma n}(k)$")

    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(spec.min(), spec.max())

    ax.legend(loc="lower left")

    plt.tight_layout()

    # -------------------------
    # Save
    # -------------------------
    out_prefix = "dens_spectrum_ms10ma08_512_w_00021"

    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"

    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()

    print("Saved:", png, "and", svg)


if __name__ == "__main__":
    main()
