import numpy as np, matplotlib.pyplot as plt

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

def directional_spectrum(P):
    Q = np.real(P); U = np.imag(P)
    amp = np.sqrt(Q**2 + U**2) + 1e-30
    cos2 = Q/amp; sin2 = U/amp
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

def screen_only_from_density(ny=1024, nx=1024, alpha_n=1.67, lam=0.5,
                             B0=1.0, chi_rms=0.5, seed=123):
    """
    Build a pure Faraday screen with RM = n_e * B0 (B_par const).
    Background polarization angle is constant (no emission structure).
    Scale RM so std(lambda^2 * RM) = chi_rms to make the screen dominate.
    """
    ne2d = fbm2d(ny, nx, alpha_n, seed=seed)
    RM = B0 * ne2d
    scale = chi_rms / ( (lam**2) * np.std(RM) + 1e-12 )
    RM *= scale
    P = np.exp(2j * (lam**2) * RM)   # constant amplitude, angle = lambda^2 RM
    return P, RM

def main():
    ny = nx = 1024
    lam = 0.5
    alpha_list = [1.10, 1.30, 1.67]   # shallower -> 1.10; "5/3" -> 1.67
    colors = ["tab:green","tab:orange","tab:blue"]

    plt.figure()
    for a, c in zip(alpha_list, colors):
        P, RM = screen_only_from_density(ny=ny, nx=nx, alpha_n=a, lam=lam,
                                         B0=1.0, chi_rms=0.7, seed=int(1000*a))
        k, Pk = directional_spectrum(P)
        slope, _ = fit_log_slope(k, Pk)
        plt.loglog(k[(k>0)], Pk[(k>0)], label=fr"screen: slope $\alpha_n={a:.2f}$; fit $\beta={slope:.2f}$", color=c, alpha=0.9)
    plt.xlabel("k"); plt.ylabel(r"$P_{\rm dir}(k)$")
    plt.title("Polarization Spectrum: Faraday Screen")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("Pdir_screen_density_slope.pdf", dpi=200)
    plt.savefig("Pdir_screen_density_slope.png", dpi=200)
    print("Saved: Pdir_screen_density_slope.pdf")

if __name__ == "__main__":
    main()
