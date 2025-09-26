import numpy as np, matplotlib.pyplot as plt

# ------------ synthesis helpers ------------
def fbm2d(ny, nx, alpha2D, seed=0):
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(nx); ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k = np.sqrt(KX**2 + KY**2); k[0,0] = 1.0
    amp = 1.0 / (k ** (alpha2D/2.0))
    phase = np.exp(1j * 2*np.pi * rng.random((ny, nx)))
    f = np.fft.ifft2(amp * phase).real
    f -= f.mean(); f /= (f.std() + 1e-12)
    return f

def fbm3d(nz, ny, nx, beta3D, seed=0):
    rng = np.random.default_rng(seed)
    kz = np.fft.fftfreq(nz); ky = np.fft.fftfreq(ny); kx = np.fft.fftfreq(nx)

    # Build grids with z as FIRST axis -> shapes (nz, ny, nx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    k[0,0,0] = 1.0

    amp = 1.0 / (k ** (beta3D / 2.0))             # (nz, ny, nx)
    phase = np.exp(1j * 2*np.pi * rng.random((nz, ny, nx)))  # (nz, ny, nx)

    f = np.fft.ifftn(amp * phase).real            # (nz, ny, nx)
    f -= f.mean()
    f /= (f.std() + 1e-12)
    return f


def cumulative_to_observer(phi_cells, axis=0):
    return np.flip(np.cumsum(np.flip(phi_cells, axis=axis), axis=axis), axis=axis)

# ------------ spectra & fitting ------------
def directional_spectrum(P):
    Q = np.real(P); U = np.imag(P)
    amp = np.sqrt(Q**2 + U**2) + 1e-30
    cos2, sin2 = Q/amp, U/amp
    F1 = np.fft.fftshift(np.fft.fft2(cos2))
    F2 = np.fft.fftshift(np.fft.fft2(sin2))
    P2 = (np.abs(F1)**2 + np.abs(F2)**2) / P.size
    ny, nx = P2.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny)); kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2).ravel(); p = P2.ravel()
    idx = np.argsort(kr); kr = kr[idx]; p = p[idx]
    n_bins = max(24, int(np.sqrt(nx*ny)))
    edges = np.linspace(kr.min(), kr.max(), n_bins+1)
    centers = 0.5*(edges[1:]+edges[:-1])
    Pk = np.zeros_like(centers); cnt = np.zeros_like(centers)
    ind = np.digitize(kr, edges) - 1
    v = (ind>=0) & (ind<n_bins)
    for i,val in zip(ind[v], p[v]): Pk[i]+=val; cnt[i]+=1
    cnt[cnt==0]=1; Pk/=cnt
    return centers, Pk

def fit_log_slope(k, Pk, kmin=5e-3, kmax=2e-1):
    m = (k>kmin) & (k<kmax) & (Pk>0)
    x = np.log10(k[m]); y = np.log10(Pk[m])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept

# ------------ physics blocks ------------
def polarization_map_mixed_constant_emissivity(ne, Bpar_const, dz, lam, K=1.0, zaxis=0,
                                               chi_rms_internal=0.7):
    """
    Mixed (internal emit+rotate), with emissivity = const per cell.
    Scale ne so std(lambda^2 * total_phi) = chi_rms_internal.
    total_phi := sum_z K * ne * Bpar_const * dz
    """
    # scale ne to set the total Faraday depth rms
    total_phi = np.sum(K * ne * Bpar_const * dz, axis=zaxis)
    s = chi_rms_internal / (lam**2 * np.std(total_phi) + 1e-12)
    ne_scaled = ne * s

    phi_cells = K * ne_scaled * Bpar_const * dz
    phi_to_obs = cumulative_to_observer(phi_cells, axis=zaxis)
    phase = np.exp(2j * (lam**2) * phi_to_obs)
    P_back = np.sum(1.0 * np.exp(2j*0.0) * phase, axis=zaxis)  # emissivity = const, psi0 = 0
    return P_back, ne_scaled

def apply_screen_from_density(P_in, ne2d, lam, Bpar_screen_const=1.0, chi_rms_screen=0.3):
    RM = Bpar_screen_const * ne2d
    s = chi_rms_screen / (lam**2 * np.std(RM) + 1e-12)
    RM *= s
    return P_in * np.exp(2j * (lam**2) * RM), RM

# ------------ experiment runner ------------
def run_mixed_plus_screen(beta3D_ne_list=(11/3, 3.0, 2.6),
                          alpha_screen=1.67,
                          nz=64, ny=512, nx=512,
                          lam=0.5, dz=1.0,
                          chi_rms_internal=0.7, chi_rms_screen=0.3,
                          seed=42, outfile="Pdir_mixed_plus_screen_density_slope.pdf"):
    rng = np.random.default_rng(seed)
    plt.figure()

    for beta_ne in beta3D_ne_list:
        # 3D density for emitting slab (controls internal Faraday depth structure)
        ne3d = fbm3d(nz, ny, nx, beta3D=beta_ne, seed=rng.integers(1e9))
        # make strictly positive but keep fluctuations (not essential for emissivity=const)
        ne3d = (ne3d - ne3d.min()) + 0.1

        # internal mixed slab with B_par constant
        P_back, _ = polarization_map_mixed_constant_emissivity(
            ne3d, Bpar_const=1.0, dz=dz, lam=lam, K=1.0, zaxis=0,
            chi_rms_internal=chi_rms_internal
        )

        # foreground screen built from 2D density with chosen slope
        ne2d_scr = fbm2d(ny, nx, alpha2D=alpha_screen, seed=rng.integers(1e9))
        P_obs, RM = apply_screen_from_density(
            P_back, ne2d_scr, lam, Bpar_screen_const=1.0, chi_rms_screen=chi_rms_screen
        )

        # spectrum & slope
        k, Pk = directional_spectrum(P_obs)
        slope, _ = fit_log_slope(k, Pk)
        lbl = fr"density slope $\beta_n={beta_ne:.2f}$; fit $\beta_{{P}}={slope:.2f}$"
        m = (k>0) & (Pk>0)
        plt.loglog(k[m], Pk[m], label=lbl)

    plt.xlabel("k")
    plt.ylabel(r"$P_{\rm dir}(k)$")
    plt.title("Polarization Spectrum: Emitting Volume with Foreground Screen")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.savefig(outfile.replace("pdf","png"), dpi=200)
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    # try a few emitting-volume slopes; screen kept at alpha=1.67 ("5/3"-like 2D)
    run_mixed_plus_screen(beta3D_ne_list=(11/3, 3.0, 2.6),
                          alpha_screen=1.67,
                          nz=64, ny=512, nx=512,
                          lam=0.5, dz=1.0,
                          chi_rms_internal=0.7, chi_rms_screen=0.3,
                          outfile="Pdir_mixed_plus_screen_density_slope.pdf")
