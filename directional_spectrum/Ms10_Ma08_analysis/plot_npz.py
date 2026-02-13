import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- match your plotting style ---
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 18


def _slope_guide(k, E, exponent=-5/3, frac_lo=0.07, frac_hi=0.35, scale=1.2):
    """
    Build a reference power-law curve A*k^exponent anchored to the spectrum
    around a mid-k segment.
    """
    klo = max(1, int(len(k) * frac_lo))
    khi = max(klo + 1, int(len(k) * frac_hi))
    kseg = k[klo:khi]
    kref = np.median(kseg) if len(kseg) else k[max(1, len(k) // 3)]
    Eref = np.interp(kref, k, E)
    A = scale * Eref / (kref ** exponent)
    return A * (k ** exponent)


def plot_from_npz(
    npz_path,
    json_path=None,
    out_prefix=None,
    label=r"$M_{\rm s}=1.0, M_{\rm A}=1.2$",
    xlim=(1, 128),
    ylim=(1e-4, 1e1),
    show_slope=True,
    slope_exp=-5/3,
):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find npz: {npz_path}")

    if out_prefix is None:
        # e.g. spectrum_spectra.npz -> spectrum_replot
        stem = npz_path.stem.replace("_spectra", "")
        out_prefix = f"{stem}_replot"

    # Load spectra
    data = np.load(npz_path)
    required = ["k_v", "E_v", "k_b", "E_b"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing '{k}' in {npz_path}. Found keys: {list(data.keys())}")

    kv = np.asarray(data["k_v"])
    Ev = np.asarray(data["E_v"])
    kb = np.asarray(data["k_b"])
    Eb = np.asarray(data["E_b"])

    # Optional metadata (units, Lbox, etc.)
    meta = {}
    if json_path is None:
        candidate = npz_path.with_suffix(".json")
        if candidate.exists():
            json_path = candidate
    if json_path is not None and Path(json_path).exists():
        meta = json.load(open(json_path, "r"))

    units = meta.get("units", "mode_index")
    Lbox = meta.get("Lbox", None)

    # Slope guides
    gv = _slope_guide(kv, Ev, exponent=slope_exp) if show_slope else None
    gb = _slope_guide(kb, Eb, exponent=slope_exp) if show_slope else None

    # X label
    xlabel = r"$k\frac{L_{\rm box}}{2\pi}$" if (Lbox is not None or units == "k*Lbox/(2π)") else r"$k$"

    # Plot
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.loglog(kv, Ev, linewidth=1.5, label=label, color='blue')
    if show_slope:
        ax1.loglog(kv, gv, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"$E_v(k)$")
    ax1.set_ylim(1e-6, 1e1)
    ax1.set_xlim(2, 128*2)

    ax2.set_ylim(1e-6, 1e1)
    ax2.set_xlim(2, 128*2)

    ax1.legend(loc="lower left")

    ax2.loglog(kb, Eb, linewidth=1.5, label=label, color='blue')
    if show_slope:
        ax2.loglog(kb, gb, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"$E_B(k)$")
    # ax2.set_ylim(1e-6, 1e0)
    # ax2.set_xlim(1, 128)
    ax2.legend(loc="lower left")

    plt.tight_layout()

    png = f"{out_prefix}_both.png"
    svg = f"{out_prefix}_both.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()

    return {"png": png, "svg": svg, "meta": meta}


def main():
    # --- EDIT THESE TWO PATHS ONLY ---
    NPZ = Path("spectrum_spectra.npz")      # produced by your VTK script
    JSON = Path("spectrum_spectra.json")    # optional

    out = plot_from_npz(
        NPZ,
        json_path=JSON if JSON.exists() else None,
        out_prefix="spectrum",  # output name base for replot
        label=r"$M_{\rm s}=10, M_{\rm A}=0.8$",
        xlim=(1, 128),
        ylim=(1e-4, 1e1),
        show_slope=True,
        slope_exp=-5/3,
    )
    print("Saved:", out["png"], "and", out["svg"])
    if out["meta"]:
        print("Loaded meta keys:", sorted(out["meta"].keys()))


if __name__ == "__main__":
    main()
