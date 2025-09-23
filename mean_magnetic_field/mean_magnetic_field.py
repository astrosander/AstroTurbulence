import numpy as np, h5py, matplotlib.pyplot as plt

def load_density_and_field(path):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = dz = 1.0
    return ne, bz, dx, dz

def ensure3d(a):
    if a.ndim==3: return a
    if a.ndim==2: return a[:,None,:]
    return a[:,None,None]

def cumulative_to_observer(phi_cells, axis):
    return np.flip(np.cumsum(np.flip(phi_cells, axis=axis), axis=axis), axis=axis)

def polarization_map_mixed(ne, bz, dz, lam, psi, K=1.0, emissivity="density", zaxis=0, B0_par=0.0, psi0=0.0):
    if emissivity=="constant":
        p_i = np.ones_like(ne, dtype=np.float64)
    elif emissivity=="density":
        p_i = ne.astype(np.float64)
    else:
        raise ValueError
    bz_tot = bz + B0_par
    phi_cells = K * ne * bz_tot * dz
    phi_to_obs = cumulative_to_observer(phi_cells, zaxis)
    phase = np.exp(2j * (lam**2) * phi_to_obs)
    P = np.sum(p_i * np.exp(2j*(psi+psi0)) * phase, axis=zaxis)
    return P

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
    n_bins = max(16, int(np.sqrt(nx*ny)))
    edges = np.linspace(kr.min(), kr.max(), n_bins+1)
    centers = 0.5*(edges[1:]+edges[:-1])
    Pk = np.zeros_like(centers); cnt = np.zeros_like(centers)
    ind = np.digitize(kr, edges) - 1
    v = (ind>=0) & (ind<n_bins)
    for i,val in zip(ind[v], p[v]): Pk[i]+=val; cnt[i]+=1
    cnt[cnt==0]=1; Pk/=cnt
    return centers, Pk

def fbm3d(nz, ny, nx, H, seed=1234):
    rng = np.random.default_rng(seed)
    kz = np.fft.fftfreq(nz); ky = np.fft.fftfreq(ny); kx = np.fft.fftfreq(nx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    k = np.sqrt(KX**2 + KY**2 + KZ**2); k[0,0,0]=1.0
    beta = 2*H + 3.0
    amp = 1.0/(k**(beta/2.0))
    phase = np.exp(1j*2*np.pi*rng.random((nz,ny,nx)))
    f = np.fft.ifftn(amp*phase).real
    f -= f.mean(); f /= (f.std()+1e-12)
    return f

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\synthetic_kolmogorov_normal.h5"
    ne,bz,dx,dz = load_density_and_field(h5_path)
    ne = ensure3d(ne); bz = ensure3d(bz)
    nz,ny,nx = ne.shape
    psi = np.zeros((nz,ny,nx), dtype=np.float64)
    lam = 0.5
    K = 1.0; zaxis = 0
    sbz = float(np.std(bz))
    B0_list = [0.0, 10.0*sbz, 1000.0**0.5*sbz, 100.0*sbz, 100000.0**0.5*sbz, 1000.0*sbz]
    plt.figure()
    for B0 in B0_list:
        P = polarization_map_mixed(ne,bz,dz,lam,psi,K,"density",zaxis,B0_par=B0,psi0=0.0)
        k, Pk = directional_spectrum(P)
        m = (k>0) & (Pk>0)
        plt.loglog(
            k[m],
            Pk[m],
            label=fr"$\frac{{B_{{0 \parallel}}}}{{\sigma_{{bz}}}}$={B0/sbz:.2f}"
        )

    plt.xlabel("k"); plt.ylabel("$P_{\\rm dir}(k)$")
    plt.title("Effect of mean B along LOS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Pdir_meanfield.png", dpi=200)
    plt.show()

if __name__=="__main__":
    main()
