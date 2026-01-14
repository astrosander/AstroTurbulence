import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Power-law random-field helpers
# -----------------------------

def _kgrid_fftn(N, L):
    """
    k-grid for fftn with shape (N, N, N).
    L is the box size (same units as desired physical coordinates).
    """
    kx = 2*np.pi * np.fft.fftfreq(N, d=L/N)
    ky = 2*np.pi * np.fft.fftfreq(N, d=L/N)
    kz = 2*np.pi * np.fft.fftfreq(N, d=L/N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    return KX, KY, KZ, K


def make_powerlaw_scalar_field_3d(N, L, beta, rng=None):
    """
    Make a real 3D Gaussian random field with 3D power spectrum P(k) ∝ k^{-beta}.

    Returns: field[N,N,N] real
    """
    rng = np.random.default_rng() if rng is None else rng
    KX, KY, KZ, K = _kgrid_fftn(N, L)

    amp = np.zeros_like(K, dtype=np.complex128)
    mask = K > 0
    amp[mask] = K[mask] ** (-beta / 2.0)

    noise = rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape)
    F = noise * amp

    F[0, 0, 0] = 0.0

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if K[i, j, k] > 0:
                    i_neg = (-i) % N
                    j_neg = (-j) % N
                    k_neg = (-k) % N
                    if (i, j, k) < (i_neg, j_neg, k_neg):
                        F[i_neg, j_neg, k_neg] = np.conj(F[i, j, k])
                    elif (i, j, k) == (i_neg, j_neg, k_neg):
                        F[i, j, k] = np.real(F[i, j, k])

    field = np.real(np.fft.ifftn(F, axes=(0, 1, 2)))

    field -= field.mean()
    field /= (field.std() + 1e-30)
    return field


def make_solenoidal_vector_field_3d(N, L, beta, rng=None):
    """
    Make an approximately divergence-free (solenoidal) 3D vector field with P(k) ∝ k^{-beta}.
    Returns Bx, By, Bz as real cubes.
    """
    rng = np.random.default_rng() if rng is None else rng
    KX, KY, KZ, K = _kgrid_fftn(N, L)

    amp = np.zeros_like(K, dtype=np.complex128)
    mask = K > 0
    amp[mask] = K[mask] ** (-beta / 2.0)

    Fx = (rng.normal(size=K.shape) + 1j*rng.normal(size=K.shape)) * amp
    Fy = (rng.normal(size=K.shape) + 1j*rng.normal(size=K.shape)) * amp
    Fz = (rng.normal(size=K.shape) + 1j*rng.normal(size=K.shape)) * amp

    k2 = KX**2 + KY**2 + KZ**2
    k2[~mask] = 1.0

    kdotF = KX*Fx + KY*Fy + KZ*Fz
    Fx = Fx - KX * kdotF / k2
    Fy = Fy - KY * kdotF / k2
    Fz = Fz - KZ * kdotF / k2
    Fx[~mask] = 0.0
    Fy[~mask] = 0.0
    Fz[~mask] = 0.0

    Fx[0, 0, 0] = 0.0
    Fy[0, 0, 0] = 0.0
    Fz[0, 0, 0] = 0.0

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if K[i, j, k] > 0:
                    i_neg = (-i) % N
                    j_neg = (-j) % N
                    k_neg = (-k) % N
                    if (i, j, k) < (i_neg, j_neg, k_neg):
                        Fx[i_neg, j_neg, k_neg] = np.conj(Fx[i, j, k])
                        Fy[i_neg, j_neg, k_neg] = np.conj(Fy[i, j, k])
                        Fz[i_neg, j_neg, k_neg] = np.conj(Fz[i, j, k])
                    elif (i, j, k) == (i_neg, j_neg, k_neg):
                        Fx[i, j, k] = np.real(Fx[i, j, k])
                        Fy[i, j, k] = np.real(Fy[i, j, k])
                        Fz[i, j, k] = np.real(Fz[i, j, k])

    bx = np.real(np.fft.ifftn(Fx, axes=(0, 1, 2)))
    by = np.real(np.fft.ifftn(Fy, axes=(0, 1, 2)))
    bz = np.real(np.fft.ifftn(Fz, axes=(0, 1, 2)))

    for arr in (bx, by, bz):
        arr -= arr.mean()
        arr /= (arr.std() + 1e-30)

    return bx, by, bz


def make_lognormal_from_gaussian(g, mean=1.0, sigma_ln=1.0):
    """
    Turn a standardized Gaussian field into a lognormal-positive field.
    Note: this can slightly distort the exact power spectrum at high sigma_ln.
    """
    # g is assumed mean~0, std~1
    x = np.exp(sigma_ln * g)
    x *= mean / (x.mean() + 1e-30)
    return x


# -----------------------------
# Intrinsic polarization P_i(x,y)
# -----------------------------

def intrinsic_polarization_from_density_and_Bperp(
    n_cube,
    bx_cube,
    by_cube,
    p0=0.7,
    alpha_B=2.0,
    alpha_n=1.0,
    dz=1.0,
    evpa_from_B=True,
    chi_offset=0.0,
):
    """
    Build intrinsic complex polarization P_i(x,y) by LOS integration of a polarized emissivity.

    Model:
      emissivity amplitude:  eps ∝ n^alpha_n * (B_perp)^alpha_B
      polarization angle χ from projected B (optional):
        χ = atan2(By, Bx) + pi/2   (synchrotron E-vector ⟂ B)
      complex polarization contribution per cell:
        dP = p0 * eps * exp(2 i χ) * dz

    Returns:
      P_i: complex array [N,N]
    """
    print(np.mean(bx_cube))
    bperp = np.sqrt(bx_cube**2 + by_cube**2) + 1e-30
    #eps = (n_cube**alpha_n) * (bperp**alpha_B)
    eps = (bperp**2)

    if evpa_from_B:
        # B angle in plane; E-vector is perpendicular for synchrotron
        phiB = np.arctan2(by_cube, bx_cube)
        chi = phiB + 0.5*np.pi + chi_offset
    else:
        # if you have your own angle model, plug it in here instead
        chi = chi_offset

    dP = (bx_cube + 1j*by_cube)**2
    P_i = np.sum(dP, axis=2)
    return P_i


def unit_polarization(u_from_P, eps=1e-30):
    """
    u_i(X) = P_i(X) / |P_i(X)|
    """
    return u_from_P / (np.abs(u_from_P) + eps)


# -----------------------------
# 2D isotropic power spectrum
# -----------------------------

def isotropic_power_spectrum_2d(field2d, Lxy, nbins=40):
    """
    Isotropic 2D power spectrum of a real or complex field.
    Returns (k_centers, Pk_binavg).

    Normalization is consistent up to a constant factor; slopes are the main target.
    """
    N = field2d.shape[0]
    assert field2d.shape == (N, N)

    F = np.fft.fft2(field2d)
    P2 = np.abs(F)**2 / (N*N)

    kx = 2*np.pi * np.fft.fftfreq(N, d=Lxy/N)
    ky = 2*np.pi * np.fft.fftfreq(N, d=Lxy/N)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    kmax = K.max()
    bins = np.linspace(0.0, kmax, nbins+1)
    which = np.digitize(K.ravel(), bins) - 1

    Pk = np.zeros(nbins)
    Nk = np.zeros(nbins, dtype=int)

    P_flat = P2.ravel()
    for i in range(nbins):
        m = which == i
        if np.any(m):
            Pk[i] = P_flat[m].mean()
            Nk[i] = m.sum()

    k_centers = 0.5*(bins[:-1] + bins[1:])
    good = (Nk > 0) & (k_centers > 0)
    return k_centers[good], Pk[good]


# -----------------------------
# Structure function via FFT correlation
# -----------------------------

def isotropic_structure_function_2d(field2d, Lxy, nbins=40):
    """
    Direct structure function for a complex (or real) field:
      D(r) = < |f(x+r) - f(x)|^2 >
           = 2( <|f|^2> - Re(C(r)) )
    where C(r) = < f(x) f*(x+r) > (autocorrelation)

    Returns (r_centers, D_binavg)
    """
    N = field2d.shape[0]
    assert field2d.shape == (N, N)

    F = np.fft.fft2(field2d)
    # autocorrelation (circular) via Wiener–Khinchin
    C = np.fft.ifft2(np.abs(F)**2) / (N*N)
    C = np.fft.fftshift(C)

    C0 = np.real(C[N//2, N//2])
    D = 2.0 * (C0 - np.real(C))

    # radius grid in real space
    dx = Lxy / N
    x = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    rmax = R.max()
    bins = np.linspace(0.0, rmax, nbins+1)
    which = np.digitize(R.ravel(), bins) - 1

    Dr = np.zeros(nbins)
    Nr = np.zeros(nbins, dtype=int)

    D_flat = D.ravel()
    for i in range(nbins):
        m = which == i
        if np.any(m):
            Dr[i] = D_flat[m].mean()
            Nr[i] = m.sum()

    r_centers = 0.5*(bins[:-1] + bins[1:])
    good = Nr > 0
    return r_centers[good], Dr[good]


# -----------------------------
# Convenience: spectra for |P_i| and u_i
# -----------------------------

def spectrum_P_complex(P_i, Lxy, nbins=40):
    """
    Spectrum of complex P_i using |FFT{Q+iU}|^2 where Q+iU = P_i
    """
    return isotropic_power_spectrum_2d(P_i, Lxy, nbins=nbins)


def spectrum_u_i(u_i, Lxy, nbins=40):
    """
    Spectrum of u_i using |FFT{(Q+iU)/sqrt(Q^2+U^2)}|^2
    where u_i = (Q+iU)/sqrt(Q^2+U^2) is the unit polarization
    """
    return isotropic_power_spectrum_2d(u_i, Lxy, nbins=nbins)


# -----------------------------
# Example pipeline
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1234)

    # Grid / box
    N = 256
    L = 1.0  # physical size (same in x,y,z here)

    # --- 1) Density cube with power-law spectrum ---
    beta_n = 11/3  # e.g. Kolmogorov-like 3D slope for a passive scalar (adjust as needed)
    g_n = make_powerlaw_scalar_field_3d(1, L, beta=beta_n, rng=rng)
    n = make_lognormal_from_gaussian(g_n, mean=1.0, sigma_ln=1.0)

    # --- 2) Perpendicular B-field cube with power-law spectrum ---
    beta_B = 11/3
    bx, by, bz = make_solenoidal_vector_field_3d(N, L, beta=beta_B, rng=rng)

    print(f"RMS before adding mean field:")
    print(f"  bx RMS: {np.sqrt(np.mean(bx**2)):.6e}")
    print(f"  by RMS: {np.sqrt(np.mean(by**2)):.6e}")
    print(f"  bz RMS: {np.sqrt(np.mean(bz**2)):.6e}")

    # --- 2a) Add mean magnetic field ---
    
    B0x = 0.0  # mean field in x direction
    B0y = 0.0  # mean field in y direction
    B0z = 0.0  # mean field in z direction (along LOS)
    bx = bx + B0x
    by = by + B0y
    bz = bz + B0z

    # --- 3) Build intrinsic complex polarization screen P_i(x,y) ---
    P_i = intrinsic_polarization_from_density_and_Bperp(
        n_cube=n,
        bx_cube=bx,
        by_cube=by,
        p0=0.7,
        alpha_B=2.0,  # emissivity dependence on B_perp
        alpha_n=1.0,  # emissivity dependence on density
        dz=L/N,
        evpa_from_B=True,
        chi_offset=0.0,
    )

    # --- 4) u_i(x,y) = P_i/|P_i| ---
    u_i = unit_polarization(P_i)

    # --- 4a) Mean subtraction for spectra (to avoid k=0 dominance) ---
    P_i_fluc = P_i - P_i.mean()
    u_i_fluc = u_i - u_i.mean()

    # --- 5) Spectra: complex P_i and u_i ---
    kP, Pk_amp = spectrum_P_complex(P_i_fluc, Lxy=L, nbins=50)
    ku, Pk_u = spectrum_u_i(u_i_fluc, Lxy=L, nbins=50)

    # --- 6) Structure functions directly ---
    rP, DP = isotropic_structure_function_2d(P_i, Lxy=L, nbins=50)
    ru, Du = isotropic_structure_function_2d(u_i, Lxy=L, nbins=50)

    # At this point, you can fit slopes in log-log space, plot, etc.
    print("Computed:")
    print(f"  |P_i| spectrum bins: {len(kP)}")
    print(f"  u_i spectrum bins:   {len(ku)}")
    print(f"  P_i structure bins:  {len(rP)}")
    print(f"  u_i structure bins:  {len(ru)}")

    # --- 7) Save all data to npz file ---
    output_file = f'{N}.npz'
    np.savez(
        output_file,
        # Parameters
        N=N,
        L=L,
        beta_n=beta_n,
        beta_B=beta_B,
        p0=0.7,
        alpha_B=2.0,
        alpha_n=1.0,
        B0x=B0x,
        B0y=B0y,
        B0z=B0z,
        # 3D cubes
        n_cube=n,
        bx_cube=bx,
        by_cube=by,
        bz_cube=bz,
        # 2D polarization fields
        P_i=P_i,
        u_i=u_i,
        # Power spectra
        kP=kP,
        Pk_amp=Pk_amp,
        ku=ku,
        Pk_u=Pk_u,
        # Structure functions
        rP=rP,
        DP=DP,
        ru=ru,
        Du=Du,
    )
    print(f"  Saved all data to {output_file}")

    # --- 8) Plot structure functions ---
    plt.figure(figsize=(10, 6))
    plt.loglog(rP, DP, 'o-', label='D_P (P_i structure function)', markersize=4, linewidth=1.5)
    plt.loglog(ru, Du, 's-', label='D_u (u_i structure function)', markersize=4, linewidth=1.5)
    plt.xlabel('r (separation distance)', fontsize=12)
    plt.ylabel('D(r) (structure function)', fontsize=12)
    plt.title('Structure Functions: P_i vs u_i', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()
