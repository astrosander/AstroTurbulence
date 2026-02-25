import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- match your plotting style ---
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 16


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
    ax1.set_ylim(4e-7, 1e1)
    ax1.set_xlim(2, 128*2)

    ax2.set_ylim(4e-7, 1e1)
    ax2.set_xlim(2, 128*2)

    ax1.legend(loc="lower left")

    ax2.loglog(kb, Eb, linewidth=1.5, label=label, color='blue')
    if show_slope:
        ax2.loglog(kb, gb, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"$E_B(k)$")
    # ax2.set_ylim(4e-7, 1e0)
    # ax2.set_xlim(1, 128)
    ax2.legend(loc="lower left")

    plt.tight_layout()

    png = f"{out_prefix}_both.png"
    svg = f"{out_prefix}_both.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()

    return {"png": png, "svg": svg, "meta": meta}


def plot_multiple_npz(
    npz_paths,
    labels,
    colors,
    json_paths=None,
    out_prefix="spectrum_comparison",
    show_slope=True,
    slope_exp=-5/3,
):
    """
    Plot multiple NPZ files on the same figure.
    
    Parameters:
    -----------
    npz_paths : list of Path or str
        List of paths to NPZ files
    labels : list of str
        Labels for each spectrum
    colors : list of str
        Colors for each spectrum
    json_paths : list of Path or str, optional
        Optional JSON metadata files
    out_prefix : str
        Output file prefix
    show_slope : bool
        Whether to show reference slope line
    slope_exp : float
        Exponent for reference slope
    """
    if json_paths is None:
        json_paths = [None] * len(npz_paths)
    
    # Load all spectra
    spectra_data = []
    meta_list = []
    
    for i, npz_path in enumerate(npz_paths):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find npz: {npz_path}")
        
        data = np.load(npz_path)
        data_keys = list(data.keys())
        
        # Check format: either (k_v, E_v, k_b, E_b) or (k_centers, E_u)
        if "k_v" in data and "E_v" in data and "k_b" in data and "E_b" in data:
            # Standard format
            kv = np.asarray(data["k_v"])
            Ev = np.asarray(data["E_v"])
            kb = np.asarray(data["k_b"])
            Eb = np.asarray(data["E_b"])
        elif "k_centers" in data and "E_u" in data:
            # dir.py format - use E_u for both velocity and magnetic field
            k_centers = np.asarray(data["k_centers"])
            E_u = np.asarray(data["E_u"])
            
            # E_u shape: (n_lambda, n_k) or (n_k,)
            if E_u.ndim == 1:
                E_u_1d = E_u
            else:
                E_u_1d = E_u[0]  # Use first lambda value
            
            # Use same data for both velocity and magnetic field
            kv = k_centers
            Ev = E_u_1d
            kb = k_centers
            Eb = E_u_1d
        else:
            raise KeyError(f"Unknown format in {npz_path}. Found keys: {data_keys}")
        
        # Optional metadata
        meta = {}
        json_path = json_paths[i]
        if json_path is None:
            candidate = npz_path.with_suffix(".json")
            if candidate.exists():
                json_path = candidate
        if json_path is not None and Path(json_path).exists():
            meta = json.load(open(json_path, "r"))
        
        spectra_data.append((kv, Ev, kb, Eb))
        meta_list.append(meta)
    
    # Get units/Lbox from first file
    units = meta_list[0].get("units", "mode_index") if meta_list[0] else "mode_index"
    Lbox = meta_list[0].get("Lbox", None) if meta_list[0] else None
    xlabel = r"$k\frac{L_{\rm box}}{2\pi}$" if (Lbox is not None or units == "k*Lbox/(2π)") else r"$k$"
    
    # Plot
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot velocity spectra
    for i, (kv, Ev, kb, Eb) in enumerate(spectra_data):
        ax1.loglog(kv, Ev, linewidth=1.5, label=labels[i], color=colors[i])
    
    # Add slope guide (use first spectrum)
    if show_slope and len(spectra_data) > 0:
        kv_first, Ev_first = spectra_data[0][0], spectra_data[0][1]
        gv = _slope_guide(kv_first, Ev_first, exponent=slope_exp)
        ax1.loglog(kv_first, gv, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"$E_v(k)$")
    ax1.set_ylim(4e-7, 1e1)
    ax1.set_xlim(2, 128*2)
    ax1.legend(loc="lower left")
    
    # Plot magnetic field spectra
    for i, (kv, Ev, kb, Eb) in enumerate(spectra_data):
        ax2.loglog(kb, Eb, linewidth=1.5, label=labels[i], color=colors[i])
    
    # Add slope guide (use first spectrum)
    if show_slope and len(spectra_data) > 0:
        kb_first, Eb_first = spectra_data[0][2], spectra_data[0][3]
        gb = _slope_guide(kb_first, Eb_first, exponent=slope_exp)
        ax2.loglog(kb_first, gb, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"$E_B(k)$")
    ax2.set_ylim(4e-7, 1e1)
    ax2.set_xlim(2, 128*2)
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    
    png = f"{out_prefix}_both.png"
    svg = f"{out_prefix}_both.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()
    
    return {"png": png, "svg": svg}


def main():
    # Load both files
    npz_paths = [
        Path("ms_10_21.npz"),
        Path("ms_1_71.npz"),
    ]
    
    labels = [
        r"$M_{\rm A}=0.8, M_{\rm s}=10$",
        r"$M_{\rm A}=0.8, M_{\rm s}=1$",
    ]
    
    colors = ['blue', 'red']
    
    out = plot_multiple_npz(
        npz_paths,
        labels,
        colors,
        json_paths=None,
        out_prefix="spectrum_comparison",
        show_slope=True,
        slope_exp=-5/3,
    )
    print("Saved:", out["png"], "and", out["svg"])


if __name__ == "__main__":
    main()
