import numpy as np
import matplotlib.pyplot as plt

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
                                 C_phi=1.0):
    """
    Construct intrinsic polarization map P_emit(X) and Faraday depth Φ(X)
    for given LOS thicknesses L_emit and L_far.
    LOS axis is the first index (z).

    P_i(z,y,x) = (B_x + i B_y)^2
    P_emit(y,x) = sum_{z=0}^{L_emit-1} P_i(z,y,x)
    Φ(y,x)      = sum_{z=0}^{L_far-1}   [C_phi * ne(z,y,x) * Bz(z,y,x)]
    """
    Bx_e, By_e, Bz_e, ne_e = emitter
    Bx_f, By_f, Bz_f, ne_f = far

    n = Bx_e.shape[0]
    if L_emit is None:
        L_emit = n
    if L_far is None:
        L_far = n

    # Intrinsic emissivity
    P_i = (Bx_e + 1j * By_e)**2
    P_emit = np.sum(P_i[:L_emit, :, :], axis=0)

    # Faraday density and RM
    phi3d = C_phi * ne_f * Bz_f
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

    print("\nEstimated correlation lengths (pixels):")
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
    ax.loglog(rline, ref, 'r--', lw=1.5, label=fr'slope $M_i\approx{M_i:.2f}$')
    ax.set_xlabel(r'$R$ (pixels)')
    ax.set_ylabel(r'$D(R)$')
    ax.set_title('Source structure function')
    ax.legend()
    ax.grid(True, which='both', ls=':')

    # Panel 2: thick screen D_dP
    ax = axes[1]
    ax.loglog(r_dP_th, D_dP_th, 'b-', lw=1.5, label=r'$D_{dP}^{\rm thick}(R)$')
    rfit = np.linspace(Rmin_global, Rmax_thick, 50)
    D0_t = D_dP_th[np.argmin(np.abs(r_dP_th - Rmin_global))]
    line_t = D0_t * (rfit / Rmin_global)**slope_dP_thick
    ax.loglog(rfit, line_t, 'r--', lw=1.5,
              label=fr'fit: slope$\approx{ slope_dP_thick:.2f}$')
    ax.axvline(r_phi_thick, color='gray', ls=':', label=r'$r_\phi$')
    ax.set_xlabel(r'$R$ (pixels)')
    ax.set_title('Thick screen (Eq. 162)')
    ax.legend()
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
                  label=fr'A: slope$\approx{ slope_thin_A:.2f}$')

    # Region B fit line: [Rmin_thin_B, Rmax_thin_B]
    if not np.isnan(slope_thin_B) and Rmax_thin_B > Rmin_thin_B:
        rB = np.linspace(Rmin_thin_B, Rmax_thin_B, 50)
        D0_B = D_dP_tn[np.argmin(np.abs(r_dP_tn - Rmin_thin_B))]
        line_B = D0_B * (rB / Rmin_thin_B)**slope_thin_B
        ax.loglog(rB, line_B, 'g--', lw=1.5,
                  label=fr'B: slope$\approx{ slope_thin_B:.2f}$')

    ax.axvline(L_thin,     color='k',    ls=':', label=r'$L_{\rm thin}$')
    ax.axvline(r_phi_thin, color='gray', ls=':', label=r'$r_\phi$')
    ax.set_xlabel(r'$R$ (pixels)')
    ax.set_title('Thin screen (Eqs. 163 & 164)')
    ax.legend()
    ax.grid(True, which='both', ls=':')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use n=256 or 512 for a serious test (512 is heavy in RAM).
    validate_lp16_appendixC(
        n=256,
        beta_B_emit=11.0/3.0,
        beta_ne_emit=11.0/3.0,
        beta_B_far=11.0/3.0,
        beta_ne_far=11.0/3.0,
        C_phi=1.0,
        seed=1,
    )
