import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.labelsize": 20,           # axis label text
    "xtick.labelsize": 18,          # x-tick labels
    "ytick.labelsize": 18,          # y-tick labels
    "axes.titlesize": 20,           # title size
    "legend.fontsize": 14,          # legend font size
})

# ============================================================
# 1. 3D Gaussian random fields with power-law spectra
# ============================================================

def gaussian_3d_field(n, beta, rng=None):
    """
    Generate a 3D Gaussian random field with 3D power spectrum
        P_3D(k) ∝ k^{-beta}
    on an n^3 grid.

    Returns real-valued array f with zero mean and unit variance.
    Axes ordering: (z, y, x).
    """
    if rng is None:
        rng = np.random.default_rng()

    kz = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx = np.fft.fftfreq(n) * n
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)

    amp = np.zeros_like(kk, dtype=np.complex128)
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
    Build synthetic cubes for separated SEFR regions:
      emitter: (Bx_e, By_e, Bz_e, ne_e)
      Faraday: (Bx_f, By_f, Bz_f, ne_f)
    """
    rng = np.random.default_rng(seed)

    # Emitting region
    Bx_e = gaussian_3d_field(n, beta_B_emit, rng=rng) + B0_emit[0]
    By_e = gaussian_3d_field(n, beta_B_emit, rng=rng) + B0_emit[1]
    Bz_e = gaussian_3d_field(n, beta_B_emit, rng=rng) + B0_emit[2]
    ne_e = gaussian_3d_field(n, beta_ne_emit, rng=rng)

    # Faraday region
    Bx_f = gaussian_3d_field(n, beta_B_far, rng=rng) + B0_far[0]
    By_f = gaussian_3d_field(n, beta_B_far, rng=rng) + B0_far[1]
    Bz_f = gaussian_3d_field(n, beta_B_far, rng=rng) + B0_far[2]
    ne_f = gaussian_3d_field(n, beta_ne_far, rng=rng)

    emitter = (Bx_e, By_e, Bz_e, ne_e)
    far     = (Bx_f, By_f, Bz_f, ne_f)
    return emitter, far


# ============================================================
# 2. P_emit, Phi maps, and dP/dλ² at λ²->0
# ============================================================

def compute_P_emit_and_RM_layers(emitter, far,
                                 L_emit=None, L_far=None,
                                 C_phi=1.0,
                                 A_phi=1.0):
    """
    Construct intrinsic polarization map P_emit(X) and Faraday depth Φ(X)
    for given LOS thicknesses L_emit and L_far.
    LOS axis is the first index (z).

    P_i(z,y,x)   = (Bx_e + i By_e)^2
    P_emit(y,x)  = sum_{z=0}^{L_emit-1} P_i(z,y,x)
    Φ(y,x)       = sum_{z=0}^{L_far-1}   [C_phi * A_phi * ne_f * Bz_f]

    A_phi is an overall amplitude factor for the Faraday density, so you
    can "turn up/down" the RM fluctuations without changing their
    correlation length or scaling exponent.
    """
    Bx_e, By_e, Bz_e, ne_e = emitter
    Bx_f, By_f, Bz_f, ne_f = far

    n = Bx_e.shape[0]
    if L_emit is None:
        L_emit = n
    if L_far is None:
        L_far = n

    # Intrinsic emissivity density
    P_i = (Bx_e + 1j * By_e)**2
    P_emit = np.sum(P_i[:L_emit, :, :], axis=0)

    # Faraday density and RM with amplitude factor A_phi
    phi3d = C_phi * A_phi * ne_f * Bz_f
    Phi   = np.sum(phi3d[:L_far, :, :], axis=0)

    return P_emit, Phi


def dP_dlambda2_at_zero(P_emit, Phi):
    """
    Weak-Faraday limit derivative (λ² -> 0) for separated SEFR:

    P(X, λ²) = P_emit(X) exp(2 i λ² Φ(X))
    => dP/dλ²|_{λ²->0} = 2 i Φ(X) P_emit(X)
    """
    return 2j * Phi * P_emit


# ============================================================
# 3. 2D correlation & structure function
# ============================================================

def corr_2d(field2d):
    """
    2D correlation function C(Δx,Δy) of a complex or real 2D field
    via FFT. Returned array is real and fftshifted so that C(0,0)
    is at the center.
    """
    f = np.asarray(field2d, dtype=np.complex128)
    ny, nx = f.shape
    f = f - f.mean()

    F = np.fft.fft2(f)
    C = np.fft.ifft2(F * np.conj(F))
    C = np.fft.fftshift(C)
    C = C.real / (nx * ny)
    return C


def radial_profile_2d(array2d, nbins=64, rmin=0.0, rmax=None):
    """
    Radial average of a 2D array (centered via fftshift).

    Returns r_centers, prof(r).
    """
    A = np.asarray(array2d, float)
    ny, nx = A.shape

    y = np.arange(-ny//2, ny - ny//2)
    x = np.arange(-nx//2, nx - nx//2)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    if rmax is None:
        rmax = R.max()

    edges = np.linspace(rmin, rmax, nbins+1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.zeros_like(r_centers)

    for i in range(nbins):
        m = (R >= edges[i]) & (R < edges[i+1])
        if np.any(m):
            prof[i] = A[m].mean()
        else:
            prof[i] = np.nan

    good = np.isfinite(prof)
    return r_centers[good], prof[good]


def structure_function_2d_complex(field2d, nbins=64, rmin=1.0, rmax=None):
    """
    Isotropic second-order structure function of a complex 2D field:

        D(R) = ⟨ |f(x) - f(x+R)|^2 ⟩ = 2(C(0) - C(R))

    We compute C via FFT, then radially average.
    """
    C = corr_2d(field2d)
    C0 = C[C.shape[0]//2, C.shape[1]//2]
    D2 = 2.0 * (C0 - C)
    return radial_profile_2d(D2, nbins=nbins, rmin=rmin, rmax=rmax)


def correlation_length_2d(field2d, nbins=64, method="efold"):
    """
    Estimate a 2D correlation length r_corr from the radial
    correlation function C(R), via either e-fold or half-power.

    method = "efold" → C(R) = C(0)/e
             "half"  → C(R) = C(0)/2
    """
    C = corr_2d(field2d)
    r, Cr = radial_profile_2d(C, nbins=nbins, rmin=0.0, rmax=None)
    C0 = Cr[0]
    if method == "half":
        target = C0 / 2.0
    else:
        target = C0 / np.e

    # Find first R where Cr <= target
    idx = np.where(Cr <= target)[0]
    if idx.size == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return r[0]
    # linear interpolation in C vs r
    r1, r2 = r[j-1], r[j]
    C1, C2 = Cr[j-1], Cr[j]
    if C2 == C1:
        return r2
    frac = (target - C1) / (C2 - C1)
    return r1 + frac * (r2 - r1)


def fit_powerlaw_slope(r, D, rmin, rmax):
    """
    Fit D(r) ∝ r^s between rmin and rmax, return slope s.
    """
    r = np.asarray(r)
    D = np.asarray(D)
    mask = (r >= rmin) & (r <= rmax) & (D > 0)
    if mask.sum() < 2:
        return np.nan
    x = np.log(r[mask])
    y = np.log(D[mask])
    s, b = np.polyfit(x, y, 1)
    return s


# ============================================================
# Helper functions for directional spectrum computation
# ============================================================

def hann2d(ny, nx):
    """2D Hann window for apodization."""
    y = np.arange(ny)
    x = np.arange(nx)
    Y, X = np.meshgrid(y, x, indexing='ij')
    wy = np.sin(np.pi * Y / (ny - 1))**2 if ny > 1 else np.ones(ny)
    wx = np.sin(np.pi * X / (nx - 1))**2 if nx > 1 else np.ones(nx)
    return wy * wx


def centered_indices(ny, nx):
    """Return centered kx, ky indices for FFT."""
    iy = np.arange(-ny//2, ny - ny//2)
    ix = np.arange(-nx//2, nx - nx//2)
    return np.meshgrid(ix, iy, indexing='ij')


def ring_average(field2d, ring_bins=48, k_min=3.0, k_max=None, apod=True, energy_like=False):
    """
    Ring-average a 2D field in k-space.
    
    Returns kc (ring centers), Pk (power), and optionally S (full 2D power), kx, ky, edges.
    """
    F = field2d
    ny, nx = F.shape
    if apod:
        F = F * hann2d(ny, nx)
    Fk = np.fft.fftshift(np.fft.fft2(F))
    S = (Fk * np.conj(Fk)).real / (ny * nx)**2
    kx, ky = centered_indices(ny, nx)
    k = np.sqrt(kx**2 + ky**2)
    if k_max is None:
        k_max = min(kx.max(), ky.max())
    edges = np.linspace(k_min, k_max, ring_bins + 1)
    kc = 0.5 * (edges[1:] + edges[:-1])
    Pk = np.zeros_like(kc)
    cnt = np.zeros_like(kc, dtype=int)
    for i in range(ring_bins):
        m = (k >= edges[i]) & (k < edges[i+1])
        cnt[i] = m.sum()
        Pk[i] = S[m].mean() if cnt[i] > 0 else np.nan
    good = (cnt > 10) & np.isfinite(Pk)
    kc, Pk = kc[good], Pk[good]
    if energy_like:
        Pk = 2 * np.pi * kc * Pk
    return kc, Pk, S, kx, ky, edges


def directional_spectrum_from_complex(F, ring_bins=48, k_min=3.0, k_max=None):
    """
    Directional spectrum of a complex field F(X) (here F = dP/dλ²):

      1) χ(X) = 0.5 * arg(F)
      2) n(X) = (cos 2χ, sin 2χ)
      3) P_dir(k) = P_c2(k) + P_s2(k) from ring-averaged 2D FFT.

    Returns
    -------
    kc : 1D array
        Ring-averaged wavenumbers.
    P_dir : 1D array
        Directional spectrum P_dir(k).
    """
    chi = 0.5 * np.angle(F)
    c2 = np.cos(2.0 * chi)
    s2 = np.sin(2.0 * chi)

    kc, P_c2, *_ = ring_average(c2, ring_bins, k_min, k_max, apod=True, energy_like=False)
    _,  P_s2, *_ = ring_average(s2, ring_bins, k_min, k_max, apod=True, energy_like=False)

    P_dir = P_c2 + P_s2
    return kc, P_dir


# ============================================================
# 4. Main validation routine for Appendix C
# ============================================================

def validate_lp16_appendixC(
    n=256,
    beta_B_emit=11.0/3.0,
    beta_ne_emit=11.0/3.0,
    beta_B_far=11.0/3.0,
    beta_ne_far=11.0/3.0,
    C_phi=1.0,
    seed=1,
):
    """
    Validate LP16 Appendix C scalings (Eqs. 162–164) with synthetic
    separated SEFR cubes, respecting:
      - thick:  L > r_phi
      - thin:   L < r_phi  and separate fits for R<L and L<R<r_phi
    """

    print("Building cubes...")
    emitter, far = build_separated_sefr_cubes(
        n=n,
        beta_B_emit=beta_B_emit,
        beta_ne_emit=beta_ne_emit,
        beta_B_far=beta_B_far,
        beta_ne_far=beta_ne_far,
        seed=seed,
    )

    # ----------------------------------------------------------
    # Step 1: estimate a global r_phi by using a full-thickness
    # Faraday screen. Then choose L_thin < r_phi_global.
    # ----------------------------------------------------------
    print("\nEstimating global r_phi from full-thickness screen...")
    # full-thickness screen for RM
    _, Phi_full = compute_P_emit_and_RM_layers(
        emitter, far, L_emit=n, L_far=n, C_phi=C_phi
    )
    r_phi_global = correlation_length_2d(Phi_full, nbins=64, method="efold")
    print(f"  r_phi_global ≈ {r_phi_global:.2f} pixels")

    # Choose L_thin so that L_thin < r_phi_global
    L_thick = n
    L_thin  = max(4, int(0.5 * r_phi_global))
    print(f"  Chosen L_thick = {L_thick}, L_thin = {L_thin} (to enforce thin screen)")

    # ----------------------------------------------------------
    # Step 2: construct P_emit and Phi for thick and thin
    # ----------------------------------------------------------
    print("\nConstructing thick and thin screens...")

    # THICK screen (emitter and Faraday integrated over full depth)
    P_emit_thick, Phi_thick = compute_P_emit_and_RM_layers(
        emitter, far, L_emit=L_thick, L_far=L_thick, C_phi=C_phi
    )
    dP_thick = dP_dlambda2_at_zero(P_emit_thick, Phi_thick)

    # THIN screen (same emitter, shallower Faraday region)
    P_emit_thin, Phi_thin = compute_P_emit_and_RM_layers(
        emitter, far, L_emit=L_thick, L_far=L_thin, C_phi=C_phi
    )
    dP_thin = dP_dlambda2_at_zero(P_emit_thin, Phi_thin)

    # ----------------------------------------------------------
    # Step 3: correlation lengths from 2D correlation functions
    # ----------------------------------------------------------
    R_i        = correlation_length_2d(P_emit_thick, nbins=64, method="efold")
    r_phi_thick = correlation_length_2d(Phi_thick,  nbins=64, method="efold")
    r_phi_thin  = correlation_length_2d(Phi_thin,   nbins=64, method="efold")

    print("\nEstimated correlation lengths:")
    print(f"  R_i           ≈ {R_i:.2f}")
    print(f"  r_phi_thick   ≈ {r_phi_thick:.2f}")
    print(f"  r_phi_thin    ≈ {r_phi_thin:.2f}")
    print(f"  L_thin        =  {L_thin}")
    print(f"  L_thick       =  {L_thick}")

    print("\nRegime checks:")
    print(f"  Thick screen: L_thick / r_phi_thick ≈ {L_thick / r_phi_thick:.2f}  (should be > 1)")
    print(f"  Thin  screen: L_thin  / r_phi_thin  ≈ {L_thin  / r_phi_thin:.2f}  (should be < 1)")

    # ----------------------------------------------------------
    # Step 4: structure functions for P_emit, Phi, dP/dλ²
    # ----------------------------------------------------------
    print("\nComputing structure functions...")

    r_P,       D_P       = structure_function_2d_complex(P_emit_thick, nbins=64, rmin=1.0, rmax=n/3)
    r_Phi_th,  D_Phi_th  = structure_function_2d_complex(Phi_thick,    nbins=64, rmin=1.0, rmax=n/3)
    r_Phi_tn,  D_Phi_tn  = structure_function_2d_complex(Phi_thin,     nbins=64, rmin=1.0, rmax=n/3)
    r_dP_th,   D_dP_th   = structure_function_2d_complex(dP_thick,     nbins=64, rmin=1.0, rmax=n/3)
    r_dP_tn,   D_dP_tn   = structure_function_2d_complex(dP_thin,      nbins=64, rmin=1.0, rmax=n/3)

    # ----------------------------------------------------------
    # Step 5: choose a common R-range and measure basic exponents
    # ----------------------------------------------------------
    Rmin_global = 4.0
    Rmax_global = n / 5.0

    M_i = fit_powerlaw_slope(r_P,      D_P,      Rmin_global, Rmax_global)
    alpha_phi = fit_powerlaw_slope(r_Phi_th, D_Phi_th, Rmin_global, Rmax_global)
    # Here alpha_phi is the exponent of D_Phi(R) ∝ R^{alpha_phi}.
    # In LP16 notation, D_{ΔΦ} ~ R^{1+tilde_m_phi}, so alpha_phi ↔ (1+tilde_m_phi).
    tilde_m_phi = alpha_phi - 1.0

    print("\nBasic slopes:")
    print(f"  Source M_i          ≈ {M_i:.3f}")
    print(f"  Faraday SF exponent ≈ {alpha_phi:.3f}  (i.e. D_Φ ∝ R^{alpha_phi})")
    print(f"  ⇒ LP16 tilde m_phi ≈ {tilde_m_phi:.3f}")

    # ==========================================================
    # THICK SCREEN (Eq. 162):  R < r_phi_thick
    # ==========================================================
    Rmax_thick = min(Rmax_global, r_phi_thick)
    slope_dP_thick = fit_powerlaw_slope(r_dP_th, D_dP_th,
                                        Rmin_global, Rmax_thick)

    print("\nThick screen (Eq. 162):")
    print(f"  Fit range: R ∈ [{Rmin_global:.1f}, {Rmax_thick:.1f}]")
    print(f"  Measured slope of D[dP] ≈ {slope_dP_thick:.3f}")
    print(f"  Expect combination of M_i ≈ {M_i:.3f} and Faraday exponent ≈ {alpha_phi:.3f}")

    # ==========================================================
    # THIN SCREEN (Eqs. 163 & 164)
    # Region A: R < L_thin
    # Region B: L_thin < R < r_phi_thin
    # ==========================================================

    # Region A: Rmin_global < R < min(L_thin, r_phi_thin, Rmax_global)
    Rmax_thin_A = min(L_thin, r_phi_thin, Rmax_global)
    if Rmax_thin_A > Rmin_global:
        slope_thin_A = fit_powerlaw_slope(r_dP_tn, D_dP_tn,
                                          Rmin_global, Rmax_thin_A)
    else:
        slope_thin_A = np.nan

    # Region B: max(L_thin, Rmin_global) < R < min(r_phi_thin, Rmax_global)
    Rmin_thin_B = max(L_thin, Rmin_global)
    Rmax_thin_B = min(r_phi_thin, Rmax_global)
    if Rmax_thin_B > Rmin_thin_B:
        slope_thin_B = fit_powerlaw_slope(r_dP_tn, D_dP_tn,
                                          Rmin_thin_B, Rmax_thin_B)
    else:
        slope_thin_B = np.nan

    print("\nThin screen (Eqs. 163 & 164):")
    print(f"  Region A (R<L):")
    print(f"    Fit range: R ∈ [{Rmin_global:.1f}, {Rmax_thin_A:.1f}]")
    print(f"    Measured slope of D[dP] ≈ {slope_thin_A:.3f}")
    print(f"    Expect mix of M_i ≈ {M_i:.3f} and Faraday exponent ≈ {alpha_phi:.3f}")
    print(f"  Region B (L<R<r_phi):")
    print(f"    Fit range: R ∈ [{Rmin_thin_B:.1f}, {Rmax_thin_B:.1f}]")
    print(f"    Measured slope of D[dP] ≈ {slope_thin_B:.3f}")
    print(f"    Expect mix of M_i~{M_i:.3f} and tilde_m_phi~{tilde_m_phi:.3f}")

    # ----------------------------------------------------------
    # Step 6: plotting, with non-overlapping fit segments
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Panel 1: D_P(R)
    ax = axes[0]
    ax.loglog(r_P, D_P, 'k-', lw=1.5, label=r'$D_{P_{\rm emit}}(R)$')
    rline = np.linspace(Rmin_global, Rmax_global, 50)
    D0 = D_P[np.argmin(np.abs(r_P - Rmin_global))]
    ref = D0 * (rline / Rmin_global)**M_i
    ax.loglog(rline, ref, 'r--', lw=1.5, label=fr'slope $M_i={M_i:.2f}$')
    ax.set_xlim([r_P.min(), r_P.max()])
    ax.set_ylim([D_P.min() * 0.8, D_P.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$D(R)$')
    ax.set_title('Source structure function')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    # Panel 2: thick screen D_dP
    ax = axes[1]
    ax.loglog(r_dP_th, D_dP_th, 'b-', lw=1.5, label=r'$D_{dP}^{\rm thick}(R)$')
    rfit = np.linspace(Rmin_global, Rmax_thick, 50)
    D0_t = D_dP_th[np.argmin(np.abs(r_dP_th - Rmin_global))]
    line_t = D0_t * (rfit / Rmin_global)**slope_dP_thick
    ax.loglog(rfit, line_t, 'r--', lw=1.5,
              label=fr'fit: slope$={ slope_dP_thick:.2f}$')
    ax.axvline(r_phi_thick, color='gray', ls=':', label=r'$r_\phi$')
    ax.set_xlim([r_dP_th.min(), r_dP_th.max()])
    ax.set_ylim([D_dP_th.min() * 0.8, D_dP_th.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_title('Thick screen (Eq. 162)')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    # Panel 3: thin screen D_dP with two non-overlapping segments
    ax = axes[2]
    ax.loglog(r_dP_tn, D_dP_tn, 'm-', lw=1.5, label=r'$D_{dP}^{\rm thin}(R)$')

    # Region A fit line: [Rmin_global, Rmax_thin_A]
    if not np.isnan(slope_thin_A) and Rmax_thin_A > Rmin_global:
        rA = np.linspace(Rmin_global, Rmax_thin_A, 50)
        D0_A = D_dP_tn[np.argmin(np.abs(r_dP_tn - Rmin_global))]
        line_A = D0_A * (rA / Rmin_global)**slope_thin_A
        ax.loglog(rA, line_A, 'r--', lw=1.5,
                  label=fr'A: slope$={ slope_thin_A:.2f}$')

    # Region B fit line: [Rmin_thin_B, Rmax_thin_B]
    if not np.isnan(slope_thin_B) and Rmax_thin_B > Rmin_thin_B:
        rB = np.linspace(Rmin_thin_B, Rmax_thin_B, 50)
        D0_B = D_dP_tn[np.argmin(np.abs(r_dP_tn - Rmin_thin_B))]
        line_B = D0_B * (rB / Rmin_thin_B)**slope_thin_B
        ax.loglog(rB, line_B, 'g--', lw=1.5,
                  label=fr'B: slope$={ slope_thin_B:.2f}$')

    ax.axvline(L_thin,     color='k',    ls=':', label=r'$L_{\rm thin}$')
    ax.axvline(r_phi_thin, color='gray', ls=':', label=r'$r_\phi$')
    ax.set_xlim([r_dP_tn.min(), r_dP_tn.max()])
    ax.set_ylim([D_dP_tn.min() * 0.8, D_dP_tn.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_title('Thin screen (Eqs. 163 & 164)')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    plt.tight_layout()
    plt.savefig('validate_lp16_appendixC.png', dpi=300, bbox_inches='tight')
    plt.show()

def validate_lp16_appendixC_theory_overlay(
    n=256,
    beta_B_emit=11.0/3.0,
    beta_ne_emit=11.0/3.0,
    beta_B_far=11.0/3.0,
    beta_ne_far=11.0/3.0,
    C_phi=1.0,
    A_phi=1.0,
    seed=1,
):
    """
    Validate LP16 Appendix C (Eqs. 162–164) by overlaying *theoretical*
    structure functions (not fitted straight lines) on top of the measured
    D_{dP}(R), using your current SEFR setup.

    - Uses dP/dλ^2 at λ^2 → 0: dP/dλ^2 = 2 i Φ P_emit.
    - Extracts M_i, tilde m_phi, R_i, r_phi, L_thin, L_thick from data.
    - Builds theory curves exactly from LP16, with a single amplitude
      normalization per regime.
    """

    print("Building cubes...")
    emitter, far = build_separated_sefr_cubes(
        n=n,
        beta_B_emit=beta_B_emit,
        beta_ne_emit=beta_ne_emit,
        beta_B_far=beta_B_far,
        beta_ne_far=beta_ne_far,
        seed=seed,
    )

    # ---------- Step 1: choose geometry: thick and thin ----------
    print("\nEstimating global r_phi for geometry...")
    # Full RM for r_phi_global
    _, Phi_full = compute_P_emit_and_RM_layers(
        emitter, far, L_emit=n, L_far=n, C_phi=C_phi, A_phi=A_phi
    )
    r_phi_global = correlation_length_2d(Phi_full, nbins=64, method="efold")
    print(f"  r_phi_global ≈ {r_phi_global:.2f} pixels")

    L_thick = n
    L_thin  = max(4, int(0.5 * r_phi_global))  # enforce L_thin < r_phi
    print(f"  L_thick = {L_thick}, L_thin = {L_thin}")

    # ---------- Step 2: construct maps ----------
    print("\nConstructing thick and thin screens...")

    # Thick
    P_emit_thick, Phi_thick = compute_P_emit_and_RM_layers(
        emitter, far, L_emit=L_thick, L_far=L_thick, C_phi=C_phi, A_phi=A_phi
    )
    dP_thick = dP_dlambda2_at_zero(P_emit_thick, Phi_thick)

    # Thin
    P_emit_thin, Phi_thin = compute_P_emit_and_RM_layers(
        emitter, far, L_emit=L_thick, L_far=L_thin, C_phi=C_phi, A_phi=A_phi
    )
    dP_thin = dP_dlambda2_at_zero(P_emit_thin, Phi_thin)

    # ---------- Step 3: correlation lengths ----------
    R_i         = correlation_length_2d(P_emit_thick, nbins=64, method="efold")
    r_phi_thick = correlation_length_2d(Phi_thick,   nbins=64, method="efold")
    r_phi_thin  = correlation_length_2d(Phi_thin,    nbins=64, method="efold")

    print("\nEstimated correlation lengths (pixels):")
    print(f"  R_i           ≈ {R_i:.2f}")
    print(f"  r_phi_thick   ≈ {r_phi_thick:.2f}")
    print(f"  r_phi_thin    ≈ {r_phi_thin:.2f}")
    print(f"  L_thin        =  {L_thin}")
    print(f"  L_thick       =  {L_thick}")

    print("\nRegime checks:")
    print(f"  Thick screen: L_thick / r_phi_thick ≈ {L_thick / r_phi_thick:.2f}  (should be > 1)")
    print(f"  Thin  screen: L_thin  / r_phi_thin  ≈ {L_thin  / r_phi_thin:.2f}  (should be < 1)")

    # ---------- Step 4: structure functions ----------
    print("\nComputing structure functions...")

    r_P,   D_P   = structure_function_2d_complex(P_emit_thick, nbins=64, rmin=1.0, rmax=n/3)
    r_Phi, D_Phi = structure_function_2d_complex(Phi_thick,    nbins=64, rmin=1.0, rmax=n/3)
    r_dP_th, D_dP_th = structure_function_2d_complex(dP_thick, nbins=64, rmin=1.0, rmax=n/3)
    r_dP_tn, D_dP_tn = structure_function_2d_complex(dP_thin,  nbins=64, rmin=1.0, rmax=n/3)

    # ---------- Step 5: extract M_i and tilde m_phi ----------
    Rmin_global = 4.0#4.0
    Rmax_global = n / 10.0#5

    M_i       = fit_powerlaw_slope(r_P,   D_P,   Rmin_global, Rmax_global)
    alpha_phi = fit_powerlaw_slope(r_Phi, D_Phi, Rmin_global, Rmax_global)
    tilde_m_phi = alpha_phi - 1.0

    print("\nBasic slopes from simulation:")
    print(f"  Source M_i          ≈ {M_i:.3f}")
    print(f"  Faraday SF exponent ≈ {alpha_phi:.3f}  (D_Φ ∝ R^{alpha_phi:.3f})")
    print(f"  ⇒ LP16 tilde m_phi  ≈ {tilde_m_phi:.3f}")

    # ---------- Helper: theory curves with 1-parameter normalization ----------

    def theory_thick(R):
        """
        LP16 Eq. (162) for thick screen, up to a global factor K_thick.
        """
        return (R / R_i)**M_i + (r_phi_thick / L_thick)**(1.0 - tilde_m_phi) * \
               (R / r_phi_thick)**(1.0 + tilde_m_phi)

    def theory_thin_A(R):
        """
        LP16 Eq. (163): thin screen, R < L_thin, up to global factor K_A.
        """
        return (R / R_i)**M_i + (r_phi_thin / L_thin) * \
               (R / r_phi_thin)**(1.0 + tilde_m_phi)

    def theory_thin_B(R):
        """
        LP16 Eq. (164): thin screen, L_thin < R < r_phi_thin, up to K_B.
        """
        return (R / R_i)**M_i + (R / r_phi_thin)**(tilde_m_phi)

    # ---------- Step 6: define R-ranges and normalize theory curves ----------

    # Thick: R ∈ [Rmin_global, min(r_phi_thick, Rmax_global)]
    Rmax_thick = min(r_phi_thick, Rmax_global)
    R_th = np.linspace(Rmin_global, Rmax_thick, 200)

    # Choose pivot for normalization (geometric mean of range)
    R0_th = np.sqrt(Rmin_global * Rmax_thick)
    idx0_th = np.argmin(np.abs(r_dP_th - R0_th))
    D0_th = D_dP_th[idx0_th]
    S0_th = theory_thick(R0_th)
    K_th = D0_th / S0_th if S0_th != 0 else 1.0
    D_th_theory = K_th * theory_thick(R_th)

    # Thin Region A: R ∈ [Rmin_global, Rmax_thin_A]
    Rmax_thin_A = min(L_thin, r_phi_thin, Rmax_global)
    R_A = np.linspace(Rmin_global, Rmax_thin_A, 200)

    R0_A = np.sqrt(Rmin_global * Rmax_thin_A)
    idx0_A = np.argmin(np.abs(r_dP_tn - R0_A))
    D0_A = D_dP_tn[idx0_A]
    S0_A = theory_thin_A(R0_A)
    K_A = D0_A / S0_A if S0_A != 0 else 1.0
    D_A_theory = K_A * theory_thin_A(R_A)

    # Thin Region B: R ∈ [Rmin_thin_B, Rmax_thin_B]
    Rmin_thin_B = max(L_thin, Rmin_global)
    Rmax_thin_B = min(r_phi_thin, Rmax_global)
    R_B = np.linspace(Rmin_thin_B, Rmax_thin_B, 200)

    R0_B = np.sqrt(Rmin_thin_B * Rmax_thin_B)
    idx0_B = np.argmin(np.abs(r_dP_tn - R0_B))
    D0_B = D_dP_tn[idx0_B]
    S0_B = theory_thin_B(R0_B)
    K_B = D0_B / S0_B if S0_B != 0 else 1.0
    D_B_theory = K_B * theory_thin_B(R_B)

    print("\nNormalization (one factor per regime):")
    print(f"  Thick:  K_th ≈ {K_th:.3e}, pivot R0_th≈{R0_th:.2f}")
    print(f"  Thin A: K_A  ≈ {K_A:.3e}, pivot R0_A ≈{R0_A:.2f}")
    print(f"  Thin B: K_B  ≈ {K_B:.3e}, pivot R0_B ≈{R0_B:.2f}")

    # ---------- Step 7: plotting (like Fig. 12, but in R-space) ----------

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
    ax11, ax12, ax21, ax22 = axes.ravel()

    # Panel 1: source SF with its measured slope
    ax = ax11
    ax.loglog(r_P, D_P, 'k-', lw=2, label=r'$D_{P_{\mathrm{emit}}}(R)$')
    rline = np.linspace(Rmin_global, Rmax_global, 100)
    D0_src = D_P[np.argmin(np.abs(r_P - Rmin_global))]
    ax.loglog(rline,
              D0_src * (rline / Rmin_global)**M_i,
              'r--', lw=2,
              label=fr'$R^{{M_i}}$, $M_i={M_i:.2f}$')
    ax.set_xlim([r_P.min(), r_P.max()])
    ax.set_ylim([D_P.min() * 0.8, D_P.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$D(R)$')
    ax.set_title(r'Source SF: $D_{P_{\mathrm{emit}}}(R)$')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    # Panel 2: Faraday SF with its measured exponent
    ax = ax12
    ax.loglog(r_Phi, D_Phi, 'C0-', lw=2, label=r'$D_{\Phi}(R)$')
    D0_phi = D_Phi[np.argmin(np.abs(r_Phi - Rmin_global))]
    ax.loglog(rline,
              D0_phi * (rline / Rmin_global)**alpha_phi,
              'C3--', lw=2,
              label=fr'$R^{{1+\tilde m_\phi}}$, $R^{{{alpha_phi:.2f}}}$')
    ax.set_xlim([r_Phi.min(), r_Phi.max()])
    ax.set_ylim([D_Phi.min() * 0.8, D_Phi.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_title('Faraday SF: $D_{\Phi}(R)$')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    # Panel 3: thick screen – data + *theory* (Eq. 162)
    ax = ax21
    ax.loglog(r_dP_th, D_dP_th, 'b-', lw=2, label=r'$D_{dP}^{\mathrm{thick}}(R)$')
    ax.loglog(R_th, D_th_theory * 1.1, 'r--', lw=2,
              label=r'Theory (Eq. 162)')
    ax.axvline(r_phi_thick, color='gray', ls=':', label=r'$r_\phi$')
    ax.set_xlim([r_dP_th.min(), r_dP_th.max()])
    ax.set_ylim([D_dP_th.min() * 0.8, D_dP_th.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$D(R)$')
    ax.set_title('Thick screen: $D_{dP}(R)$ vs theory')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    # Panel 4: thin screen – data + theory for regions A and B
    ax = ax22
    ax.loglog(r_dP_tn, D_dP_tn, 'm-', lw=2, label=r'$D_{dP}^{\mathrm{thin}}(R)$')

    # Theory in region A (R<L_thin): Eq. 163
    ax.loglog(R_A, D_A_theory * 1.1, 'r--', lw=2,
              label=r'Theory A: Eq. 163')
    # Theory in region B (L<R<r_phi): Eq. 164
    ax.loglog(R_B, D_B_theory, 'g--', lw=2,
              label=r'Theory B: Eq. 164')

    ax.axvline(L_thin,     color='k',    ls=':', label=r'$L_{\mathrm{thin}}$')
    ax.axvline(r_phi_thin, color='gray', ls=':', label=r'$r_\phi$')
    ax.set_xlim([r_dP_tn.min(), r_dP_tn.max()])
    ax.set_ylim([D_dP_tn.min() * 0.8, D_dP_tn.max() * 1.2])
    ax.set_xlabel(r'$R$')
    ax.set_title('Thin screen: $D_{dP}(R)$ vs theory')
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls=':')

    plt.tight_layout()
    plt.savefig('validate_lp16_appendixC_theory_overlay.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- Step 8: directional spectra of dP/dλ² (thick & thin) ----------

    print("\nComputing directional spectra for dP/dλ²...")

    # Directional spectra for thick and thin screens
    kc_th, Pdir_th = directional_spectrum_from_complex(dP_thick, ring_bins=48, k_min=3.0)
    kc_tn, Pdir_tn = directional_spectrum_from_complex(dP_thin,  ring_bins=48, k_min=3.0)

    # Predicted slopes from LP16 Appendix C exponents:
    # D(R) ∝ R^α  ⇒  E_2D(k) ∝ k^{-(α+2)} in 2D
    s_src      = -(M_i + 2.0)            # source term: α = M_i
    s_rm_A     = -(tilde_m_phi + 3.0)    # RM term in Eqs. 162,163: α = 1+tilde_m_phi
    s_rm_B     = -(tilde_m_phi + 2.0)    # RM term in Eq. 164:     α = tilde_m_phi

    print("\nDirectional-spectrum theoretical slopes (from LP16 exponents):")
    print(f"  Source-like:        P_dir(k) ∝ k^{s_src:.3f}  (α = M_i ≈ {M_i:.3f})")
    print(f"  RM-like (small R):  P_dir(k) ∝ k^{s_rm_A:.3f} (α = 1+tilde_m ≈ {1+tilde_m_phi:.3f})")
    print(f"  RM-like (L<R<rφ):   P_dir(k) ∝ k^{s_rm_B:.3f} (α = tilde_m ≈ {tilde_m_phi:.3f})")

    # Choose pivot scales in k to normalize the theory lines
    def make_theory_line(kc, Pdir, slope, k_fraction_low=0.3, k_fraction_high=0.7):
        """
        Build a power-law line P_th(k) = A * k^slope, normalized at a pivot k0
        chosen from the data (single-parameter normalization; exponent fixed).
        """
        kmin, kmax = kc.min(), kc.max()
        k0 = 10.0 ** (0.5 * (np.log10(kmin) + np.log10(kmax)))  # geometric mean
        idx0 = np.argmin(np.abs(kc - k0))
        P0 = Pdir[idx0]
        k_th = np.linspace(kmin, kmax, 200)
        P_th = P0 * (k_th / k0) ** slope
        return k_th, P_th

    # Build theory lines for thick (source vs RM in Eq. 162)
    k_th_src, P_th_src = make_theory_line(kc_th, Pdir_th, s_src)
    k_th_rmA, P_th_rmA = make_theory_line(kc_th, Pdir_th, s_rm_A)

    # Build theory lines for thin:
    #   - Region A (R<L): source vs RM with slope s_rm_A
    #   - Region B (L<R<rφ): source vs RM with slope s_rm_B
    k_tn_src, P_tn_src = make_theory_line(kc_tn, Pdir_tn, s_src)
    k_tn_rmB, P_tn_rmB = make_theory_line(kc_tn, Pdir_tn, s_rm_B)

    # Optionally, measure empirical slopes of directional spectra for reference
    def fit_k_slope(kc, Pk, kmin_fit_frac=0.2, kmax_fit_frac=0.8):
        kmin_fit = kc[int(len(kc) * kmin_fit_frac)]
        kmax_fit = kc[int(len(kc) * kmax_fit_frac)]
        return fit_powerlaw_slope(kc, Pk, kmin_fit, kmax_fit)

    slope_dir_th = fit_k_slope(kc_th, Pdir_th)
    slope_dir_tn = fit_k_slope(kc_tn, Pdir_tn)

    print("\nMeasured directional-spectrum slopes (single broad fit):")
    print(f"  Thick screen: slope ≈ {slope_dir_th:.3f}")
    print(f"  Thin  screen: slope ≈ {slope_dir_tn:.3f}")

    # ---------- Step 9: plot directional spectra + LP16-based predictions ----------

    fig2, (ax_th, ax_tn) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Thick screen panel
    ax = ax_th
    ax.loglog(kc_th, Pdir_th, 'b-', lw=2, label=r'$P_{\mathrm{dir}}^{\mathrm{thick}}(k)$')
    ax.loglog(k_th_src, P_th_src, 'r--', lw=2,
              label=fr'Source: $k^{{{s_src:.2f}}}$')
    ax.loglog(k_th_rmA, P_th_rmA, 'g--', lw=2,
              label=fr'RM (Eq. 162): $k^{{{s_rm_A:.2f}}}$')
    ax.set_xlim([kc_th.min(), kc_th.max()])
    ax.set_ylim([Pdir_th.min() * 0.8, Pdir_th.max() * 1.2])
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P_{\mathrm{dir}}(k)$')
    ax.set_title('Directional spectrum: thick screen')
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=14)

    # Thin screen panel
    ax = ax_tn
    ax.loglog(kc_tn, Pdir_tn, 'm-', lw=2, label=r'$P_{\mathrm{dir}}^{\mathrm{thin}}(k)$')
    ax.loglog(k_tn_src, P_tn_src, 'r--', lw=2,
              label=fr'Source: $k^{{{s_src:.2f}}}$')
    ax.loglog(k_tn_rmB, P_tn_rmB, 'g--', lw=2,
              label=fr'RM (Eq. 164): $k^{{{s_rm_B:.2f}}}$')
    ax.set_xlim([kc_tn.min(), kc_tn.max()])
    ax.set_ylim([Pdir_tn.min() * 0.8, Pdir_tn.max() * 1.2])
    ax.set_xlabel(r'$k$')
    ax.set_title('Directional spectrum: thin screen')
    ax.grid(True, which='both', ls=':')
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig('validate_lp16_directional_spectrum.png',
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Use n=256 or 512 for a serious test (512 is heavy in RAM).
    # validate_lp16_appendixC(
    #     n=256,
    #     beta_B_emit=11.0/3.0,
    #     beta_ne_emit=11.0/3.0,
    #     beta_B_far=11.0/3.0,
    #     beta_ne_far=11.0/3.0,
    #     C_phi=1.0,
    #     seed=1,
    # )

    # validate_lp16_appendixC_multislopes(
    #     n=256,
    #     beta_B_emit=11.0/3.0,
    #     beta_ne_emit=11.0/3.0,
    #     beta_B_far=11.0/3.0,
    #     beta_ne_far=11.0/3.0,
    #     C_phi=1.0,
    #     seed=1,
    # )
    
    validate_lp16_appendixC_theory_overlay(
        n=256,
        beta_B_emit=11.0/3.0,
        beta_ne_emit=11.0/3.0,
        beta_B_far=11.0/3.0,
        beta_ne_far=11.0/3.0,
        C_phi=1.0,
        A_phi=1.0,
        seed=1,
    )
