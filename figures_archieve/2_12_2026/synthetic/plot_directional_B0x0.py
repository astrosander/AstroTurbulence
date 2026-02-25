from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# # --- match your plotting style ---
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams["legend.frameon"] = False
# plt.rcParams['font.size'] = 24
# plt.rcParams['axes.labelsize'] = 24
# plt.rcParams['axes.titlesize'] = 24
# plt.rcParams['xtick.labelsize'] = 24
# plt.rcParams['ytick.labelsize'] = 24
# plt.rcParams['legend.fontsize'] = 24
# plt.rcParams['figure.titlesize'] = 24

import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Journal-style plotting setup
# ----------------------------

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Times New Roman", "Times", "Computer Modern"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
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


def _slope_guide_structure(r, D, exponent=5/3, frac_lo=0.07, frac_hi=0.35, scale=1.2):
    """
    Build a reference power-law curve A*r^exponent anchored to the structure function
    around a mid-r segment.
    """
    rlo = max(1, int(len(r) * frac_lo))
    rhi = max(rlo + 1, int(len(r) * frac_hi))
    rseg = r[rlo:rhi]
    rref = np.median(rseg) if len(rseg) else r[max(1, len(r) // 3)]
    Dref = np.interp(rref, r, D)
    A = scale * Dref / (rref ** exponent)
    return A * (r ** exponent)


def fit_power_law_small_k(k, Pk, k_min=10, k_max=100, min_points=5):
    """
    Fit a power law P(k) = A * k^exponent to the small k region.
    
    Parameters:
    -----------
    k : np.ndarray
        Wavenumber array
    Pk : np.ndarray
        Power spectrum array
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
    Pk_fit : np.ndarray
        Fitted power law values
    """
    # Select k range from k_min to k_max
    mask = (k >= k_min) & (k <= k_max)
    
    if mask.sum() < min_points:
        # Not enough points in range, return None
        return None, None, None, None
    
    k_small = k[mask]
    Pk_small = Pk[mask]
    
    # Filter out invalid values
    valid = (k_small > 0) & (Pk_small > 0) & np.isfinite(k_small) & np.isfinite(Pk_small)
    if valid.sum() < 3:
        # Not enough valid points, return None
        return None, None, None, None
    
    k_fit = k_small[valid]
    Pk_fit = Pk_small[valid]
    
    # Fit in log-log space: log(P) = log(A) + exponent * log(k)
    log_k = np.log(k_fit)
    log_Pk = np.log(Pk_fit)
    
    # Linear regression
    coeffs = np.polyfit(log_k, log_Pk, 1)
    exponent = coeffs[0]
    log_A = coeffs[1]
    A = np.exp(log_A)
    
    # Generate fitted curve over the fitting range
    k_fit_range = np.linspace(k_fit.min(), k_fit.max(), 100)
    Pk_fit_curve = A * (k_fit_range ** exponent)
    
    return A, exponent, k_fit_range, Pk_fit_curve


def _normalize_npz_data(data):
    """
    Normalize NPZ data from different formats to a common format.
    
    Handles three formats:
    1. Old format: k, Pk
    2. PSA format: k, Pk_lambda, lambda2_list
    3. dir.py format: k_centers, E_u, lam2
    
    Returns:
    --------
    k : np.ndarray
        Wavenumber array
    Pk : np.ndarray or np.ndarray with shape (n_lambda, n_k)
        Power spectrum (1D or 2D for PSA)
    Lxy : float or None
        Box size
    lambda2_list : np.ndarray or None
        Array of lambda^2 values (if PSA format), None otherwise
    format_type : str
        'old', 'psa', or 'dir'
    """
    # Check for dir.py format (k_centers, E_u, lam2)
    has_dir_format = "k_centers" in data and "E_u" in data
    
    if has_dir_format:
        k = np.asarray(data["k_centers"])
        E_u = np.asarray(data["E_u"])
        lam2 = np.asarray(data.get("lam2", [0.0]))  # Default to single value if missing
        
        # E_u shape: (n_lambda, n_k) or (n_k,) if single lambda
        if E_u.ndim == 1:
            # Single spectrum, reshape to (1, n_k) for consistency
            Pk = E_u.reshape(1, -1)
            lambda2_list = lam2
        else:
            # Multiple lambda values
            Pk = E_u
            lambda2_list = lam2
        
        Lxy = data.get("dx", None)  # Try dx as Lxy, or could use other fields
        if Lxy is None:
            Lxy = data.get("Lxy", None)
        
        # If we have multiple lambda values, treat as PSA-like
        if len(lambda2_list) > 1:
            return k, Pk, Lxy, lambda2_list, "psa"
        else:
            # Single spectrum, extract it
            return k, Pk[0] if Pk.ndim > 1 else Pk, Lxy, None, "old"
    
    # Check for PSA format
    has_psa = "Pk_lambda" in data and "lambda2_list" in data
    if has_psa:
        k = np.asarray(data["k"])
        Pk = np.asarray(data["Pk_lambda"])
        lambda2_list = np.asarray(data["lambda2_list"])
        Lxy = data.get("Lxy", None)
        return k, Pk, Lxy, lambda2_list, "psa"
    
    # Check for old format
    has_old = "k" in data and "Pk" in data
    if has_old:
        k = np.asarray(data["k"])
        Pk = np.asarray(data["Pk"])
        Lxy = data.get("Lxy", None)
        return k, Pk, Lxy, None, "old"
    
    # If we get here, format is unknown
    raise KeyError(f"Unknown NPZ format. Found keys: {list(data.keys())}. "
                   f"Expected one of: (k, Pk), (k, Pk_lambda, lambda2_list), "
                   f"or (k_centers, E_u, lam2)")


def plot_directional_from_npz(
    npz_path,
    out_prefix=None,
    label="Directional Spectrum",
    xlim=None,
    ylim=None,
    show_slope=True,
    slope_exp=-11/3,
    lambda2_idx=0,
):
    """
    Plot directional spectrum from NPZ file.
    
    Parameters:
    -----------
    lambda2_idx : int, optional
        Index into lambda2_list for PSA data (new format). 
        If old format (single Pk), this is ignored.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find npz: {npz_path}")

    if out_prefix is None:
        # e.g. spectrum_B0x0.npz -> spectrum_B0x0
        out_prefix = npz_path.stem

    # Load spectrum data
    data = np.load(npz_path)
    
    # Normalize data format
    k, Pk_data, Lxy, lambda2_list, format_type = _normalize_npz_data(data)
    
    # Extract the appropriate spectrum
    if format_type == "psa":
        # PSA format: Pk_data is 2D (n_lambda, n_k)
        if lambda2_idx >= len(lambda2_list):
            lambda2_idx = 0
            print(f"Warning: lambda2_idx out of range, using 0")
        
        Pk = Pk_data[lambda2_idx] if Pk_data.ndim > 1 else Pk_data
        lambda2_val = lambda2_list[lambda2_idx]
        
        if label == "Directional Spectrum":
            label = f"$\\sigma\\lambda^2 = {lambda2_val/59.494:.3f}$"
    else:
        # Old format or dir.py format: single power spectrum
        Pk = Pk_data if Pk_data.ndim == 1 else Pk_data[0]

    # Slope guide
    g = _slope_guide(k, Pk, exponent=slope_exp) if show_slope else None

    # X label
    xlabel = r"$k$" if Lxy is None else r"$k$"

    # Plot
    plt.figure(figsize=(7.0,4.0))
    ax = plt.subplot(1, 1, 1)

    ax.loglog(k, Pk, linewidth=1.5, label=label, color='blue')
    if show_slope:
        ax.loglog(k, g, linestyle="--", linewidth=4, label=r"slope: $-11/3$", color='black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$P(k)$")
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
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


def load_and_average_spectra(npz_dir, file_start=40, file_end=51, lambda2_idx=0):
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
    lambda2_idx : int, optional
        Index into lambda2_list for PSA data (new format).
        If old format (single Pk), this is ignored.
    
    Returns:
    --------
    k : np.ndarray
        Wavenumber array (should be the same for all files)
    Pk_avg : np.ndarray
        Averaged power spectrum
    Lxy : float or None
        Box size (from first file)
    lambda2_list : np.ndarray or None
        Array of lambda^2 values (if PSA format), None otherwise
    """
    npz_dir = Path(npz_dir)
    Pk_list = []
    k = None
    Lxy = None
    lambda2_list = None
    
    loaded_count = 0
    for file_num in range(file_start, file_end + 1):
        npz_path = f"Pu_spectrum.npz"#directional_ms10ma08_512.mhd_bcc.{file_num}.npz"
        
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping...")
            continue
        
        try:
            data = np.load(npz_path)
            
            # Normalize data format
            try:
                k_current, Pk_data_current, Lxy_current, lambda2_list_current, format_type = _normalize_npz_data(data)
            except KeyError as e:
                print(f"Warning: {npz_path} - {e}, skipping...")
                continue
            
            # Extract the appropriate spectrum
            if format_type == "psa":
                # PSA format: Pk_data is 2D (n_lambda, n_k)
                if lambda2_list is None:
                    lambda2_list = lambda2_list_current
                elif lambda2_list_current is not None and not np.allclose(lambda2_list, lambda2_list_current):
                    print(f"Warning: lambda2_list differs in {npz_path}")
                
                # Select the requested lambda^2 index
                if lambda2_list_current is not None:
                    if lambda2_idx >= len(lambda2_list_current):
                        print(f"Warning: lambda2_idx {lambda2_idx} out of range in {npz_path}, using 0")
                        idx = 0
                    else:
                        idx = lambda2_idx
                    Pk_current = Pk_data_current[idx] if Pk_data_current.ndim > 1 else Pk_data_current
                else:
                    Pk_current = Pk_data_current[0] if Pk_data_current.ndim > 1 else Pk_data_current
            else:
                # Old format or dir.py format: single power spectrum
                Pk_current = Pk_data_current if Pk_data_current.ndim == 1 else Pk_data_current[0]
            
            if k is None:
                k = k_current
                Lxy = Lxy_current
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
    if lambda2_list is not None:
        print(f"Using lambda^2 index {lambda2_idx} (lambda^2 = {lambda2_list[lambda2_idx]:.3f})")
    
    return k, Pk_avg, Lxy, lambda2_list


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
    label="$M_s=1, M_A=0.8, 520^3$ (averaged)",
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
    plt.figure(figsize=(7.0,4.0))
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


def plot_multiple_spectra(
    spectra_list,
    out_prefix="spectrum_comparison",
    show_slope=True,
    slope_exp=-11/3,
    Lxy=None,
):
    """
    Plot multiple spectra on the same figure.
    
    Parameters:
    -----------
    spectra_list : list of tuples
        Each tuple is (k, Pk, label, color) where:
        - k: wavenumber array
        - Pk: power spectrum array
        - label: label for the spectrum
        - color: color for the line (optional)
    out_prefix : str
        Output file prefix
    show_slope : bool
        Whether to show reference slope line
    slope_exp : float
        Exponent for reference slope
    Lxy : float or None
        Box size (for labeling)
    
    Returns:
    --------
    dict with 'png' and 'svg' file paths
    """
    # X label
    xlabel = r"$k$" if Lxy is None else r"$k$"
    
    # Plot
    plt.figure(figsize=(7.0,4.0))
    ax = plt.subplot(1, 1, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    k_min = np.inf
    k_max = -np.inf
    Pk_min = np.inf
    Pk_max = -np.inf
    
    for i, spec in enumerate(spectra_list):
        if len(spec) == 3:
            k, Pk, label = spec
            color = colors[i % len(colors)]
        else:
            k, Pk, label, color = spec
        
        ax.loglog(k, Pk, linewidth=2.5, label=label, color=color)
        
        # Fit and plot power law for small k (k from 10 to 100)
        A, exponent, k_fit, Pk_fit = fit_power_law_small_k(k, Pk, k_min=10, k_max=100)
        if A is not None and k_fit is not None:
            # Use a lighter/dashed version of the same color for the fit
            fit_color = color
            # Create label with fitted exponent
            fit_label = f"$\\propto k^{{{exponent:.2f}}}$"
            ax.loglog(k_fit, Pk_fit, linestyle=":", linewidth=2.0, 
                     alpha=0.7, color=fit_color, label=fit_label)
            print(f"  Fit for {label} (k: 10-100): P(k) = {A:.2e} * k^{exponent:.3f}")
        
        k_min = min(k_min, min(k))
        k_max = max(k_max, max(k))
        Pk_min = min(Pk_min, min(Pk))
        Pk_max = max(Pk_max, max(Pk))
    
    # Add reference slope line (use first spectrum to anchor it)
    if show_slope and len(spectra_list) > 0:
        k_ref, Pk_ref = spectra_list[0][0], spectra_list[0][1]
        g = _slope_guide(k_ref, Pk_ref, exponent=slope_exp)
        ax.loglog(k_ref, g, linestyle="--", linewidth=4, label=r"$k^{-11/3}$", color='black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$P(k)$")
    
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(Pk_min, Pk_max)
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
    label="$M_s=1, M_A=0.8, 520^3$",
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
    plt.figure(figsize=(7.0,4.0))
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


def plot_psa_spectra(
    npz_dir,
    file_start=43,
    file_end=71,
    out_prefix="psa_spectra",
    lambda2_idx=None,
):
    """
    Plot PSA power spectra at different lambda^2 values.
    
    Parameters:
    -----------
    lambda2_idx : int or None
        If None, plots all lambda^2 values. Otherwise, plots only the specified index.
    """
    npz_dir = Path(npz_dir)
    
    # Load first file to get lambda2_list
    first_file = npz_dir / f"directional_{file_start}.npz"
    if not first_file.exists():
        raise FileNotFoundError(f"Could not find {first_file}")
    
    data = np.load(first_file)
    if "lambda2_list" not in data or "Pk_lambda" not in data:
        raise ValueError(f"{first_file} does not contain PSA data")
    
    lambda2_list = np.asarray(data["lambda2_list"])
    k = np.asarray(data["k"])
    Lxy = data.get("Lxy", None)
    
    if lambda2_idx is not None:
        lambda2_indices = [lambda2_idx]
    else:
        lambda2_indices = range(len(lambda2_list))
    
    # Load and average spectra for each lambda^2
    Pk_avg_list = []
    for idx in lambda2_indices:
        _, Pk_avg, _, _ = load_and_average_spectra(npz_dir, file_start, file_end, lambda2_idx=idx)
        Pk_avg_list.append(Pk_avg)
    
    # Plot
    plt.figure()#figsize=(7.0,4.0)))
    ax = plt.subplot(1, 1, 1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda2_indices)))
    for i, idx in enumerate(lambda2_indices):
        if idx % 5 == 0:
            ax.loglog(k, Pk_avg_list[i], linewidth=2.5, 
                     label = f"$\\sigma\\lambda^2 = {lambda2_list[idx]/59.494*5:.3f}$",
                     # label=f"$\\lambda^2 = {lambda2_list[idx]:.3f}$", 
                     color=colors[i])
        else:
            ax.loglog(k, Pk_avg_list[i], linewidth=2.5, 
                     # label = f"$\\sigma\\lambda^2 = {lambda2_list[idx]/59.494*5:.3f}$",
                     # label=f"$\\lambda^2 = {lambda2_list[idx]:.3f}$", 
                     color=colors[i])
            
    # Add -11/3 reference line
    # Use the first spectrum to anchor the reference line
    if len(Pk_avg_list) > 0:
        g = _slope_guide(k, Pk_avg_list[0], exponent=-11/3)
        ax.loglog(k, g, linestyle="--", linewidth=4, 
                 label=r"$k^{-11/3}$", color='black')
    
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P(k)$")
    ax.set_xlim(min(k), max(k))
    ax.legend(loc="lower left")
    plt.tight_layout()
    
    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()
    
    return {"png": png, "svg": svg}


def plot_psa_structure_functions(
    npz_dir,
    file_start=43,
    file_end=71,
    out_prefix="psa_structure_functions",
    lambda2_idx=None,
):
    """
    Plot PSA structure functions at different lambda^2 values.
    
    Parameters:
    -----------
    lambda2_idx : int or None
        If None, plots all lambda^2 values. Otherwise, plots only the specified index.
    """
    npz_dir = Path(npz_dir)
    
    # Load first file to get lambda2_list
    first_file = npz_dir / f"directional_{file_start}.npz"
    if not first_file.exists():
        raise FileNotFoundError(f"Could not find {first_file}")
    
    data = np.load(first_file)
    if "lambda2_list" not in data or "D_lambda" not in data:
        raise ValueError(f"{first_file} does not contain PSA data")
    
    lambda2_list = np.asarray(data["lambda2_list"])
    r = np.asarray(data["r"])
    Lxy = data.get("Lxy", None)
    
    if lambda2_idx is not None:
        lambda2_indices = [lambda2_idx]
    else:
        lambda2_indices = range(len(lambda2_list))
    
    # Load and average structure functions for each lambda^2
    D_avg_list = []
    for idx in lambda2_indices:
        D_list = []
        for file_num in range(file_start, file_end + 1):
            npz_path = npz_dir / f"directional_{file_num}.npz"
            if not npz_path.exists():
                continue
            try:
                data = np.load(npz_path)
                if "D_lambda" in data:
                    D_lambda = np.asarray(data["D_lambda"])
                    D_list.append(D_lambda[idx])
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
        
        if len(D_list) > 0:
            D_avg = np.mean(D_list, axis=0)
            D_avg_list.append(D_avg)
    
    # Plot
    plt.figure(figsize=(7.0,4.0))
    ax = plt.subplot(1, 1, 1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda2_indices)))
    for i, idx in enumerate(lambda2_indices):
        if i < len(D_avg_list):
            if idx % 5==0:
                ax.loglog(r, D_avg_list[i], linewidth=2.5, 
                         label=f"$\\sigma\\lambda^2 = {lambda2_list[idx]/59.494*5:.3f}$", 
                     # label = f"$\\sigma\\lambda^2 = {lambda2_list[idx]/59.494*5:.3f}$",
                         color=colors[i])
            else:
                ax.loglog(r, D_avg_list[i], linewidth=2.5, 
                         # label=f"$\\sigma\\lambda^2 = {lambda2_list[idx]:.3f}$", 
                         color=colors[i])
    
    # Add r^{5/3} reference line
    # Use the first structure function to anchor the reference line
    if len(D_avg_list) > 0:
        g = _slope_guide_structure(r, D_avg_list[0], exponent=5/3)
        ax.loglog(r, g, linestyle="--", linewidth=4, 
                 label=r"$r^{5/3}$", color='black')

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$D(r)$")
    ax.set_xlim(min(r), max(r))
    ax.legend(loc="lower left")
    plt.tight_layout()
    
    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=300)
    plt.savefig(svg, dpi=300)
    plt.close()
    
    return {"png": png, "svg": svg}


def main():
    # Load all spectra
    npz_paths = [
        Path(r"0.npz"),
        Path(r"0.05.npz"),
        Path(r"0.1.npz"),
        Path(r"0.2.npz"),
        Path(r"0.4.npz"),
        # Path(r"D:\Downloads\2_12_2026\ms_10_5\Pu_spectrum.npz"),
        Path(r"1.npz"),
        # Path(r"3.npz"),
        # Path(r"10.npz"),
        
        # Path(r"D:\Downloads\2_12_2026\ms_10_1+ms_1_72\Pu_spectrum.npz"),
        # Path(r"D:\Downloads\2_12_2026\21_P_u_0.1\Pu_spectrum_ms10_plus_ms1.npz"),
    ]

    
    labels = [
        # "$\\eta=0.01$",
        "$\\eta=0$",
        "$\\eta=0.05$",
        "$\\eta=0.1$",
        "$\\eta=0.2$",
        "$\\eta=0.4$",
        "$\\eta=1$",
        # "$\\eta=3$",
        # "$\\eta=10$",
        # "$M_s=10, M_A=0.8, t=0.05$",
        # "$M_A=0.8, t=1: M_s=10$",
        # "$M_s=10, M_A=0.8, t=0.05$ \& $M_s=1, M_A=0.8$",
        # "$M_A=0.8, t=1; (n_e) M_s=10\\,\&\\, (\\vec{B})M_s=1$",
    ]

    #     # Load all spectra
    # npz_paths = [
    #     # Path(r"b1_0.01.npz"),
    #     # Path(r"b1_0.1.npz"),
    #     # Path(r"b1_1.npz"),
    #     # Path(r"b1_4.npz"),
    #     # Path(r"b1_10.npz"),
    #     # Path(r"P0_0.05.npz"),
    #     # Path(r"P0_0.1.npz"),
    #     # Path(r"P0_0.2.npz"),
    #     # Path(r"P0_1.npz"),
    #     Path(r"0.1.npz"),
    #     Path(r"0.2.npz"),
    #     Path(r"0.4.npz"),
    #     Path(r"1.npz"),
    #     Path(r"3.npz"),
    #     Path(r"10.npz"),
        
    #     # Path(r"D:\Downloads\2_12_2026\ms_10_1+ms_1_72\Pu_spectrum.npz"),
    #     # Path(r"D:\Downloads\2_12_2026\21_P_u_0.1\Pu_spectrum_ms10_plus_ms1.npz"),
    # ]

    
    # labels = [
    #     # "$\\eta=0.01$",
    #     "$\\eta=0.01$",
    #     "$\\eta=0.1$",
    #     "$\\eta=1$",
    #     "$\\eta=4$",
    #     "$\\eta=10$",
    #     # "$\\eta=0.05$",
    #     # # "$\\eta=0.05$",
    #     # "$\\eta=0.1$",
    #     # "$\\eta=0.2$",
    #     # "$\\eta=1$",
    #     # "$\\eta=0.2$",
    #     # "$\\eta=0.4$",
    #     # "$\\eta=1$",
    #     # "$\\eta=3$",
    #     # "$\\eta=10$",
    #     # "$M_s=10, M_A=0.8, t=0.05$",
    #     # "$M_A=0.8, t=1: M_s=10$",
    #     # "$M_s=10, M_A=0.8, t=0.05$ \& $M_s=1, M_A=0.8$",
    #     # "$M_A=0.8, t=1; (n_e) M_s=10\\,\&\\, (\\vec{B})M_s=1$",
    # ]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    spectra_list = []
    
    print("Loading spectra...")
    print("="*60)
    
    for i, npz_path in enumerate(npz_paths):
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find {npz_path}")
        
        print(f"Loading spectrum from {npz_path}")
        data = np.load(npz_path)
        k, Pk_data, Lxy, lambda2_list, format_type = _normalize_npz_data(data)
        
        # Extract spectrum
        if format_type == "psa" and Pk_data.ndim > 1:
            # Use first lambda value for single file plot
            Pk = Pk_data[0]
        else:
            Pk = Pk_data if Pk_data.ndim == 1 else Pk_data[0]
        
        # Filter out data points where k < 2pi or k > pi*512
        k_min = 2 * np.pi
        k_max = np.pi * 512
        mask = (k >= k_min) & (k <= k_max)
        k = k[mask]
        Pk = Pk[mask]
        
        label = labels[i] if i < len(labels) else f"Spectrum {i+1}"
        color = colors[i] if i < len(colors) else None
        
        spectra_list.append((k, Pk, label, color))
        print(f"  Loaded: {len(k)} points (filtered {k_min:.2f} <= k <= {k_max:.2f}), label: {label}")
    
    print("="*60)
    print("Plotting spectra on same figure...")
    
    out = plot_multiple_spectra(
        spectra_list,
        out_prefix="spectrum_comparison",
        show_slope=True,
        slope_exp=-11/3,
        Lxy=Lxy,
    )
    print("Saved:", out["png"], "and", out["svg"])


if __name__ == "__main__":
    main()

