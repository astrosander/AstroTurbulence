from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- match your plotting style ---
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 24


def _slope_guide(k, E, exponent=-11/3, frac_lo=0.07, frac_hi=0.35, scale=1.2):
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


def plot_directional_from_npz(
    npz_path,
    out_prefix=None,
    label="Directional Spectrum",
    xlim=None,
    ylim=None,
    show_slope=True,
    slope_exp=-11/3,
):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find npz: {npz_path}")

    if out_prefix is None:
        # e.g. spectrum_B0x0.npz -> spectrum_B0x0
        out_prefix = npz_path.stem

    # Load spectrum data
    data = np.load(npz_path)
    required = ["k", "Pk"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing '{k}' in {npz_path}. Found keys: {list(data.keys())}")

    k = np.asarray(data["k"])
    Pk = np.asarray(data["Pk"])
    Lxy = data.get("Lxy", None)

    # Slope guide
    g = _slope_guide(k, Pk, exponent=slope_exp) if show_slope else None

    # X label
    xlabel = r"$k$" if Lxy is None else r"$k$"

    # Plot
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)

    ax.loglog(k, Pk, linewidth=1.5, label=label, color='blue')
    if show_slope:
        ax.loglog(k, g, linestyle="--", linewidth=4, label=r"slope: $-11/3$", color='black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$P(k)$")
    
    # if xlim is not None:
    #     ax.set_xlim(*xlim)
    # if ylim is not None:
    #     ax.set_ylim(*ylim)
    ax.set_xlim(min(k), max(k))
    ax.set_ylim(min(Pk), max(Pk))
    ax.legend(loc="lower left")
    plt.tight_layout()

    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()

    return {"png": png, "svg": svg}


def load_and_average_spectra(npz_dir, file_start=40, file_end=51):
    """
    Load and average spectra from multiple npz files.
    
    Parameters:
    -----------
    npz_dir : Path or str
        Directory containing the npz files
    file_start : int
        Starting file number (inclusive)
    file_end : int
        Ending file number (inclusive)
    
    Returns:
    --------
    k : np.ndarray
        Wavenumber array (should be the same for all files)
    Pk_avg : np.ndarray
        Averaged power spectrum
    Lxy : float or None
        Box size (from first file)
    """
    npz_dir = Path(npz_dir)
    Pk_list = []
    k = None
    Lxy = None
    
    loaded_count = 0
    for file_num in range(file_start, file_end + 1):
        npz_path = npz_dir / f"directional_{file_num}.npz"
        
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping...")
            continue
        
        try:
            data = np.load(npz_path)
            if "k" not in data or "Pk" not in data:
                print(f"Warning: {npz_path} missing required keys, skipping...")
                continue
            
            k_current = np.asarray(data["k"])
            Pk_current = np.asarray(data["Pk"])
            
            if k is None:
                k = k_current
                Lxy = data.get("Lxy", None)
            else:
                # Check if k arrays match (they should)
                if not np.allclose(k, k_current, rtol=1e-10):
                    print(f"Warning: k arrays differ in {npz_path}, interpolating...")
                    # Interpolate to common k grid
                    Pk_current = np.interp(k, k_current, Pk_current)
            
            Pk_list.append(Pk_current)
            loaded_count += 1
            print(f"Loaded: {npz_path}")
            
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
    
    if len(Pk_list) == 0:
        raise RuntimeError(f"No valid npz files found in range {file_start}-{file_end}")
    
    # Average the spectra
    Pk_avg = np.mean(Pk_list, axis=0)
    print(f"\nAveraged {loaded_count} spectra from files {file_start}-{file_end}")
    
    return k, Pk_avg, Lxy


def compute_log_derivative(k, Pk, smooth_window=None):
    """
    Compute the log derivative d(log P)/d(log k) with optional smoothing.
    
    Parameters:
    -----------
    k : np.ndarray
        Wavenumber array
    Pk : np.ndarray
        Power spectrum array
    smooth_window : int, optional
        Window size for smoothing. If None, uses 10% of data points (min 5).
        Larger values give smoother results.
    
    Returns:
    --------
    k_deriv : np.ndarray
        Wavenumber array
    dlogP_dlogk : np.ndarray
        Log derivative array
    """
    # Use log space for numerical stability
    log_k = np.log(k)
    log_Pk = np.log(Pk)
    
    # Smooth the log spectrum before computing derivative
    if smooth_window is None:
        smooth_window = max(5, int(len(log_Pk) * 0.10))
    
    # Ensure window is odd for symmetric smoothing
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    # Use a simple moving average for smoothing
    # Pad the array to handle boundaries
    pad_size = smooth_window // 2
    padded = np.pad(log_Pk, pad_size, mode='edge')
    
    # Create a normalized boxcar kernel
    kernel = np.ones(smooth_window) / smooth_window
    smoothed_log_Pk = np.convolve(padded, kernel, mode='valid')
    
    # Compute derivative using central differences on smoothed data
    dlogP_dlogk = np.gradient(smoothed_log_Pk, log_k)
    
    return k, dlogP_dlogk


def plot_averaged_spectrum(
    k,
    Pk,
    out_prefix="spectrum_B0x0_avg",
    label="$M_s=1, M_A=0.8, 512^3$ (averaged)",
    Lxy=None,
    show_slope=True,
    slope_exp=-11/3,
):
    """
    Plot an averaged spectrum.
    """
    # Slope guide
    g = _slope_guide(k, Pk, exponent=slope_exp) if show_slope else None

    # X label
    xlabel = r"$k$" if Lxy is None else r"$k$"

    # Plot
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)

    ax.loglog(k, Pk, linewidth=2.5, label=label, color='blue')
    if show_slope:
        ax.loglog(k, g, linestyle="--", linewidth=4, label=r"$k^{-11/3}$", color='black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$P(k)$")
    
    ax.set_xlim(min(k), max(k))
    ax.set_ylim(min(Pk), max(Pk))
    ax.legend(loc="lower left")
    plt.tight_layout()

    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()

    return {"png": png, "svg": svg}


def plot_log_derivative(
    k,
    Pk,
    out_prefix="spectrum_B0x0_avg_deriv",
    label="$M_s=1, M_A=0.8, 512^3$",
    Lxy=None,
    reference_slope=-11/3,
):
    """
    Plot the log derivative d(log P)/d(log k).
    """
    k_deriv, dlogP_dlogk = compute_log_derivative(k, Pk)
    
    # X label
    xlabel = r"$k$" if Lxy is None else r"$k$"
    
    # Plot
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    
    ax.semilogx(k_deriv, dlogP_dlogk, linewidth=2.5, label=label, color='blue')
    
    # Add reference line for expected slope
    ax.axhline(y=reference_slope, linestyle="--", linewidth=4, 
               label=r"$k^{-11/3}$", color='black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{d}(\log P)/\mathrm{d}(\log k)$")
    
    ax.set_xlim(min(k_deriv), max(k_deriv))
    ax.legend(loc="best")
    plt.tight_layout()
    
    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()
    
    return {"png": png, "svg": svg}


def main():
    # Load and average spectra from files 40-51
    npz_dir = Path("npz")
    file_start = 44#44#44
    file_end = 63
    
    print(f"Loading and averaging spectra from {npz_dir}/directional_{file_start}.npz to directional_{file_end}.npz")
    print("="*60)
    
    k, Pk_avg, Lxy = load_and_average_spectra(npz_dir, file_start, file_end)
    
    print("="*60)
    print("Plotting averaged spectrum...")
    
    out = plot_averaged_spectrum(
        k,
        Pk_avg,
        out_prefix="spectrum_B0x0_avg",
        label="$M_s=1, M_A=0.8, 512^3$",
        Lxy=Lxy,
        show_slope=True,
        slope_exp=-11/3,
    )
    print("Saved:", out["png"], "and", out["svg"])
    
    print("="*60)
    print("Plotting log derivative...")
    
    out_deriv = plot_log_derivative(
        k,
        Pk_avg,
        out_prefix="spectrum_B0x0_avg_deriv",
        label="$M_s=1, M_A=0.8, 512^3$",
        Lxy=Lxy,
        reference_slope=-11/3,
    )
    print("Saved:", out_deriv["png"], "and", out_deriv["svg"])


if __name__ == "__main__":
    main()

