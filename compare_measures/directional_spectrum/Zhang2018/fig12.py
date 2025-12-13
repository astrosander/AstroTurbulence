import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. Utilities: 3D Gaussian fields with power-law spectrum
# ============================================================

def gaussian_3d_field(n, beta, rng=None):
    """
    Generate a 3D Gaussian random field with 3D power spectrum
        P_3D(k) ∝ k^{-beta}
    on an n^3 grid.

    Parameters
    ----------
    n : int
        Grid size.
    beta : float
        3D spectral index (e.g. 11/3, 4, 7/2).
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    f : ndarray, shape (n, n, n)
        Real-valued 3D field with zero mean and unit variance.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Discrete wavenumbers
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kz = np.fft.fftfreq(n) * n
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)

    amp = np.zeros_like(kk, dtype=np.float64)
    mask = kk > 0
    amp[mask] = kk[mask] ** (-beta / 2.0)

    noise_real = rng.normal(size=(n, n, n))
    noise_imag = rng.normal(size=(n, n, n))
    Fk = (noise_real + 1j * noise_imag) * amp

    f = np.fft.ifftn(Fk).real
    f -= f.mean()
    std = f.std()
    if std > 0:
        f /= std
    return f


# ============================================================
# 2. Build SEFR cubes for the "spatially separated" case
# ============================================================

def build_separated_sefr_cubes(
    n,
    beta_B_emit,
    beta_ne_emit,
    beta_B_far,
    beta_ne_far,
    B0_emit=(0.0, 0.0, 0.0),
    B0_far=(0.0, 0.0, 0.0),
    seed=0,
):
    """
    Build synthetic cubes for the spatially separated SEFR case,
    following Zhang et al. (2018) and LP16.

    We create two independent regions:
      - synchrotron-emitting region ("emit")
      - Faraday-rotating region ("far")

    Each region has 3D turbulent magnetic field components
    (Bx, By, Bz) and a thermal electron density ne, with
    prescribed 3D spectral indices.

    Parameters
    ----------
    n : int
        Grid size (n^3).
    beta_B_emit, beta_ne_emit : float
        3D spectral indices in the emitting region.
    beta_B_far, beta_ne_far : float
        3D spectral indices in the Faraday rotation region.
    B0_emit, B0_far : 3-tuple of floats
        Mean magnetic field components added to each region.
    seed : int
        Seed for RNG.

    Returns
    -------
    emitter : (Bx_e, By_e, Bz_e, ne_e)
    far     : (Bx_f, By_f, Bz_f, ne_f)
    """
    rng = np.random.default_rng(seed)

    # Emitting region
    Bx_e = gaussian_3d_field(n, beta_B_emit, rng=rng) + B0_emit[0]
    By_e = gaussian_3d_field(n, beta_B_emit, rng=rng) + B0_emit[1]
    Bz_e = gaussian_3d_field(n, beta_B_emit, rng=rng) + B0_emit[2]
    ne_e = gaussian_3d_field(n, beta_ne_emit, rng=rng)

    # Faraday rotation region
    Bx_f = gaussian_3d_field(n, beta_B_far, rng=rng) + B0_far[0]
    By_f = gaussian_3d_field(n, beta_B_far, rng=rng) + B0_far[1]
    Bz_f = gaussian_3d_field(n, beta_B_far, rng=rng) + B0_far[2]
    ne_f = gaussian_3d_field(n, beta_ne_far, rng=rng)

    emitter = (Bx_e, By_e, Bz_e, ne_e)
    far     = (Bx_f, By_f, Bz_f, ne_f)
    return emitter, far


def compute_P_emit_and_RM(emitter, far, los_axis=0, C_phi=1.0):
    """
    Spatially separated SEFR setup:

    - intrinsic polarized emissivity density:
         P_i(x, y, z) ∝ (B_x + i B_y)^2   (p = 3 electron index)
    - complex emissivity integrated over the emitting region:
         P_emit(X) = ∫ P_i(X, z) dz
    - Faraday depth (RM) from the foreground region:
         Φ(X) ∝ ∫ ne(X, z) B_parallel(X, z) dz

    Parameters
    ----------
    emitter : tuple
        (Bx_e, By_e, Bz_e, ne_e) for emitting region.
    far : tuple
        (Bx_f, By_f, Bz_f, ne_f) for Faraday region.
    los_axis : int
        Axis to integrate along (0 -> z).
    C_phi : float
        Overall normalization constant for Φ, absorbs physical
        units (0.812 ne B L etc.) and grid spacing.

    Returns
    -------
    P_emit : 2D complex array
        Intrinsic complex polarization integrated along LOS.
    Phi_map : 2D real array
        Faraday depth map (foreground screen).
    """
    Bx_e, By_e, Bz_e, ne_e = emitter
    Bx_f, By_f, Bz_f, ne_f = far

    # Intrinsic complex emissivity density
    P_i = (Bx_e + 1j * By_e) ** 2

    # Integrate along LOS (simple Riemann sum with Δz=1)
    P_emit = np.sum(P_i, axis=los_axis)

    # Faraday rotation density (proportional to ne * B_parallel)
    phi_3d = C_phi * ne_f * Bz_f
    Phi_map = np.sum(phi_3d, axis=los_axis)

    return P_emit, Phi_map


# ============================================================
# 3. Polarization, derivative, and E_2D(k) spectra
# ============================================================

def ring_E2D(field2d, kbins=64, kmin=1.0, kmax=None, apodize=True):
    """
    Compute the ring-integrated 1D spectrum E_2D(k) for a
    2D complex field (e.g., P, dP/dλ^2).

    This follows the definition used in Zhang et al. (2018):
    - compute 2D Fourier power P_2D(kx, ky) = |F(kx, ky)|^2
    - integrate P_2D over rings in Fourier space

    Parameters
    ----------
    field2d : 2D array (complex or real)
    kbins : int
        Number of radial bins.
    kmin : float
        Minimal radial k to include.
    kmax : float or None
        Maximal radial k. If None, determined from grid.
    apodize : bool
        If True, apply a 2D Hann window before FFT.

    Returns
    -------
    k_centers : 1D array
        Centers of the k bins.
    E2D : 1D array
        Ring-integrated power E_2D(k).
    """
    f = np.asarray(field2d)
    ny, nx = f.shape

    if apodize:
        wy = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(ny) / (ny - 1)))
        wx = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(nx) / (nx - 1)))
        window = wy[:, None] * wx[None, :]
        f = f * window

    Fk = np.fft.fft2(f)
    P2 = (Fk * np.conj(Fk)).real / (nx * ny) ** 2

    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    kr = np.sqrt(KX ** 2 + KY ** 2)

    if kmax is None:
        kmax = kr.max()

    edges = np.linspace(kmin, kmax, kbins + 1)
    k_centers = 0.5 * (edges[:-1] + edges[1:])
    E2D = np.zeros_like(k_centers)

    for i in range(kbins):
        mask = (kr >= edges[i]) & (kr < edges[i + 1])
        if np.any(mask):
            # Sum over the ring -> ∝ 2π k P_2D(k)
            E2D[i] = P2[mask].sum()

    good = E2D > 0
    return k_centers[good], E2D[good]


def compute_P_lambda(P_emit, Phi_map, lam):
    """
    Complex polarization map at wavelength λ (code units) for
    the spatially separated SEFR case:

        P(X, λ^2) = P_emit(X) * exp( 2 i λ^2 Φ(X) )

    Parameters
    ----------
    P_emit : 2D complex array
    Phi_map : 2D real array
    lam : float
        Wavelength λ (same units as used to define Φ).

    Returns
    -------
    P_map : 2D complex array
    """
    lam2 = lam * lam
    return P_emit * np.exp(2j * lam2 * Phi_map)


def compute_dPdlambda2_numeric(P_maps, lam_list):
    """
    Compute dP/dλ^2 maps using finite differences in λ^2.

    Parameters
    ----------
    P_maps : list of 2D complex arrays
        P_maps[i] corresponds to λ_i in lam_list.
    lam_list : 1D array
        Wavelengths λ_i.

    Returns
    -------
    dP_maps : list of 2D complex arrays
        Same length as lam_list; derivative at each λ_i.
    """
    lam_list = np.asarray(lam_list)
    lam2 = lam_list ** 2
    nlam = len(lam_list)
    dP_maps = []

    for i in range(nlam):
        if i == 0:
            dP = (P_maps[i + 1] - P_maps[i]) / (lam2[i + 1] - lam2[i])
        elif i == nlam - 1:
            dP = (P_maps[i] - P_maps[i - 1]) / (lam2[i] - lam2[i - 1])
        else:
            dP = (P_maps[i + 1] - P_maps[i - 1]) / (lam2[i + 1] - lam2[i - 1])
        dP_maps.append(dP)
    return dP_maps


def fit_loglog_slope(k, E, kmin, kmax):
    """
    Fit a power-law slope to log10 E vs log10 k between
    (kmin, kmax).

    Returns slope (d logE / d logk).
    """
    k = np.asarray(k)
    E = np.asarray(E)
    mask = (k >= kmin) & (k <= kmax) & (E > 0)
    if mask.sum() < 2:
        return np.nan
    x = np.log10(k[mask])
    y = np.log10(E[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return slope


# ============================================================
# 4. Driver to reproduce Fig. 12 qualitatively
# ============================================================

def run_fig12_like_sim(
    n=256,
    lam_min=0.1,
    lam_max=20.0,
    dloglam=0.1,
    C_phi=1.0,
    kfit_min=15.0,
    kfit_max=100.0,
    seed=0,
    save_dir=None,
):
    """
    Run a synthetic experiment that mimics Zhang+2018 Fig. 12:
    E_2D(k) of dP/dλ^2 for spatially separated SEFR regions.

    Parameters
    ----------
    n : int
        Grid size (n x n x n). Use n=512 for close reproduction.
    lam_min, lam_max : float
        Minimal and maximal λ in code units.
    dloglam : float
        Step in log10(λ). (0.1 in the paper.)
    C_phi : float
        Normalization constant for Faraday depth Φ.
        Adjust this so that 2π λ^2 σ_Φ falls in the k range
        where you want the transition.
    kfit_min, kfit_max : float
        k-range used to measure slopes in log–log space.
    seed : int
        RNG seed.
    save_dir : str or None
        Directory to save cubes, spectra data, and figure.
        If None, saves in current directory.
    """
    # Setup save directory
    if save_dir is None:
        save_dir = "."
    os.makedirs(save_dir, exist_ok=True)

    loglam = np.arange(np.log10(lam_min), np.log10(lam_max) + 1e-8, dloglam)
    lam_list = 10 ** loglam

    cfg_left = dict(beta_B_emit=4.0,   beta_ne_emit=4.0,
                    beta_B_far=7.0/2.0, beta_ne_far=7.0/2.0)

    cfg_right = dict(beta_B_emit=7.0/2.0, beta_ne_emit=7.0/2.0,
                     beta_B_far=4.0,      beta_ne_far=4.0)

    configs = [cfg_left, cfg_right]
    panel_titles = [
        r"Emission: $\beta_B=\beta_{n_e}=4$, Faraday: $\beta_B=\beta_{n_e}=7/2$",
        r"Emission: $\beta_B=\beta_{n_e}=7/2$, Faraday: $\beta_B=\beta_{n_e}=4$",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Store all spectra data for saving
    all_panel_spectra = []

    for panel_idx, (ax, cfg, title) in enumerate(zip(axes, configs, panel_titles)):
        # --- (i) Build cubes
        emitter, far = build_separated_sefr_cubes(
            n=n,
            beta_B_emit=cfg["beta_B_emit"],
            beta_ne_emit=cfg["beta_ne_emit"],
            beta_B_far=cfg["beta_B_far"],
            beta_ne_far=cfg["beta_ne_far"],
            B0_emit=(0.0, 0.0, 0.0),
            B0_far=(0.0, 0.0, 0.0),
            seed=seed,
        )
        
        # Save cubes
        cube_filename = os.path.join(save_dir, f"cubes_panel{panel_idx}_seed{seed}.npz")
        Bx_e, By_e, Bz_e, ne_e = emitter
        Bx_f, By_f, Bz_f, ne_f = far
        np.savez_compressed(
            cube_filename,
            # Emitter region
            Bx_emit=Bx_e,
            By_emit=By_e,
            Bz_emit=Bz_e,
            ne_emit=ne_e,
            # Faraday region
            Bx_far=Bx_f,
            By_far=By_f,
            Bz_far=Bz_f,
            ne_far=ne_f,
            # Metadata
            n=n,
            beta_B_emit=cfg["beta_B_emit"],
            beta_ne_emit=cfg["beta_ne_emit"],
            beta_B_far=cfg["beta_B_far"],
            beta_ne_far=cfg["beta_ne_far"],
            seed=seed,
        )
        print(f"Saved cubes to {cube_filename}")

        # --- (ii) P_emit and Φ
        P_emit, Phi_map = compute_P_emit_and_RM(emitter, far, los_axis=0, C_phi=C_phi)
        sigma_phi = Phi_map.std()
        print(f"[{title}]  sigma_Phi = {sigma_phi:.3g}")

        # --- (iii) P(λ^2) maps
        P_maps = [compute_P_lambda(P_emit, Phi_map, lam) for lam in lam_list]

        # --- (iv) dP/dλ^2 maps
        dP_maps = compute_dPdlambda2_numeric(P_maps, lam_list)

        # --- (v) Spectra & slopes
        slopes = []
        all_spectra = []

        for lam, dP in zip(lam_list, dP_maps):
            k, E = ring_E2D(dP, kbins=n//2, kmin=1.0)
            all_spectra.append((lam, k, E))
            slope = fit_loglog_slope(k, E, kfit_min, kfit_max)
            slopes.append((lam, slope))

        # --- (vi) Plot a subset of λ curves, mimicking the paper
        # Highlight λ_min and λ_max as thick lines
        for lam, k, E in all_spectra:
            if np.any(E <= 0):
                continue
            if np.isclose(lam, lam_min):
                ax.loglog(k, E, color="k", lw=2.0, label=r"$\lambda_{\rm min}$")
            elif np.isclose(lam, lam_max):
                ax.loglog(k, E, color="k", lw=2.0, ls="--", label=r"$\lambda_{\rm max}$")
            else:
                ax.loglog(k, E, color="gray", lw=0.5, alpha=0.5)
        
        # Store spectra data for this panel
        all_panel_spectra.append(all_spectra)

        ax.set_xlabel(r"$k$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.3)

        # Optional: overplot reference slopes as straight lines
        # Choose some pivot k0 and amplitude from λ_min spectrum
        k0 = 10.0
        lam0, k0_arr, E0_arr = all_spectra[0]
        j0 = np.argmin(np.abs(k0_arr - k0))
        E0 = E0_arr[j0]

        # Reference slopes: k^{-3}, k^{-5/2}, k^{+1}
        for slope_ref, ls, label in [(-3.0, ":",  r"$k^{-3}$"),
                                     (-2.5, "--", r"$k^{-5/2}$"),
                                     ( 1.0, "-.", r"$k^{1}$")]:
            # line: E = E0 * (k/k0)^{slope_ref}
            E_ref = E0 * (k0_arr / k0) ** slope_ref
            ax.loglog(k0_arr, E_ref, ls=ls, color="C1", alpha=0.8, label=label)

        # Print measured slopes for sanity check
        print(f"  Measured slopes in k=[{kfit_min},{kfit_max}] for a few λ:")
        for lam, slope in slopes[::max(1, len(slopes)//5)]:
            print(f"    λ={lam:5.2f}  slope≈{slope:6.3f}")

    axes[0].set_ylabel(r"$E_{2D}(k; dP/d\lambda^2)$")
    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    
    # Save all spectra data
    spectra_filename = os.path.join(save_dir, f"spectra_data_seed{seed}.npz")
    # Convert all_spectra to arrays for saving
    spectra_data = {}
    for panel_idx, panel_spectra in enumerate(all_panel_spectra):
        # Store each panel's spectra
        # Save lambda values as a 1D array
        lam_list_panel = [lam for lam, k, E in panel_spectra]
        spectra_data[f"panel{panel_idx}_lambda"] = np.array(lam_list_panel)
        
        # Save k and E as lists (they have different lengths for each lambda)
        # We'll save them as object arrays which can be reconstructed
        k_list_panel = [k for lam, k, E in panel_spectra]
        E_list_panel = [E for lam, k, E in panel_spectra]
        
        # Save lengths for reconstruction
        k_lengths = [len(k_arr) for k_arr in k_list_panel]
        E_lengths = [len(E_arr) for E_arr in E_list_panel]
        spectra_data[f"panel{panel_idx}_k_lengths"] = np.array(k_lengths)
        spectra_data[f"panel{panel_idx}_E_lengths"] = np.array(E_lengths)
        
        # Concatenate all k and E arrays with a marker to separate them
        # We'll use a large negative value as separator (assuming k and E are positive)
        k_flat = np.concatenate(k_list_panel)
        E_flat = np.concatenate(E_list_panel)
        spectra_data[f"panel{panel_idx}_k_flat"] = k_flat
        spectra_data[f"panel{panel_idx}_E_flat"] = E_flat
    
    # Save metadata
    spectra_data["lam_min"] = lam_min
    spectra_data["lam_max"] = lam_max
    spectra_data["dloglam"] = dloglam
    spectra_data["n"] = n
    spectra_data["seed"] = seed
    spectra_data["C_phi"] = C_phi
    spectra_data["kfit_min"] = kfit_min
    spectra_data["kfit_max"] = kfit_max
    spectra_data["n_panels"] = len(all_panel_spectra)
    
    # Save as npz
    np.savez_compressed(spectra_filename, **spectra_data)
    print(f"Saved spectra data to {spectra_filename}")
    
    # Save figure as PNG
    fig_filename = os.path.join(save_dir, f"fig12_seed{seed}.png")
    fig.savefig(fig_filename, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {fig_filename}")
    
    return fig, axes


if __name__ == "__main__":
    # Example run with modest resolution. For a closer
    # reproduction of Zhang+ (2018) use n=512, but that
    # will be heavy.
    fig, axes = run_fig12_like_sim(
        n=512,         # set n=512 if your machine can handle it
        lam_min=0.1,
        lam_max=20.0,
        dloglam=0.1,
        C_phi=0.01,   # tune this so 2π λ^2 σ_Φ lands in your k-range
        kfit_min=15.0,
        kfit_max=80.0,
        seed=1,
    )
    plt.show()
