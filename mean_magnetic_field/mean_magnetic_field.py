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

def P_emission_only(ne_or_I, psi_bg=0.0, zaxis=0, use_sum=True):
    if use_sum and ne_or_I.ndim == 3:
        I_emit = np.sum(ne_or_I.astype(np.float64), axis=zaxis)
    else:
        I_emit = ne_or_I.astype(np.float64)

    return I_emit * np.exp(2j * psi_bg)

def polarization_map_screen(ne_scr, bz_scr, dz, lam, P_emit_2d, K=1.0, zaxis=0, B0_par=0.0):
    RM = np.sum(K * ne_scr * (bz_scr + B0_par) * dz, axis=zaxis)
    return P_emit_2d * np.exp(2j * (lam**2) * RM)

def P_synchrotron_slab_plus_screen(ne_emit, psi_bg, ne_scr, bz_scr, dz, lam, K=1.0, zaxis=0, B0_par_screen=0.0):
    P_emit = P_emission_only(ne_emit, psi_bg, zaxis=zaxis, use_sum=True)
    return polarization_map_screen(ne_scr, bz_scr, dz, lam, P_emit, K, zaxis, B0_par=B0_par_screen)

def P_mixed_plus_screen(ne_back, bz_back, psi, ne_scr, bz_scr, dz, lam, K=1.0, zaxis=0, B0_par_back=0.0, B0_par_screen=0.0):
    P_back = polarization_map_mixed(ne_back, bz_back, dz, lam, psi, K, "density", zaxis, B0_par=B0_par_back, psi0=0.0)
    return polarization_map_screen(ne_scr, bz_scr, dz, lam, P_back, K, zaxis, B0_par=B0_par_screen)

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

def plot_case(title, curves, outfile):
    plt.figure()
    for label, P in curves:
        k, Pk = directional_spectrum(P)
        m = (k>0) & (Pk>0)
        plt.loglog(k[m], Pk[m], label=label)
    plt.xlabel("k"); plt.ylabel(r"$P_{\rm dir}(k)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")
    plt.close()

def main(scenario="screen"):
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
    ne,bz,dx,dz = load_density_and_field(h5_path)
    ne = ensure3d(ne); bz = ensure3d(bz)
    nz,ny,nx = ne.shape
    psi = np.zeros((nz,ny,nx), dtype=np.float64)
    lam = 0.5
    K = 1.0; zaxis = 0
    sbz = float(np.std(bz))
    mus = [0.0, 0.1, 0.3, 1, 3, 10, 20]
    B0_list = [mu * sbz for mu in mus]

    if scenario == "two_screens":
        z_split = nz//2
        ne_back, bz_back = ne[:z_split], bz[:z_split]
        ne_scr,  bz_scr  = ne[z_split:],  bz[z_split:]
        
        psi_bg = 0.0
        curves_emit = []
        curves_mixed = []
        for B0 in B0_list:
            P_emit_scr = P_synchrotron_slab_plus_screen(
                ne_emit=ne_back, psi_bg=psi_bg,
                ne_scr=ne_scr, bz_scr=bz_scr,
                dz=dz, lam=lam, K=K, zaxis=zaxis,
                B0_par_screen=B0
            )
            curves_emit.append((fr"$B_0/\sigma_{{bz}}$={B0/sbz:.2f}", P_emit_scr))
            
            P_mix_scr = P_mixed_plus_screen(
                ne_back=ne_back, bz_back=bz_back, psi=psi[:z_split],
                ne_scr=ne_scr,  bz_scr=bz_scr,
                dz=dz, lam=lam, K=K, zaxis=zaxis,
                B0_par_back=0.0,
                B0_par_screen=B0
            )
            curves_mixed.append((fr"$B_0/\sigma_{{bz}}$={B0/sbz:.2f}", P_mix_scr))
        
        plot_case("Synchrotron Emission Behind a Foreground Faraday Screen", curves_emit, "Pdir_two_screens_emit.pdf")
        plot_case("Synchrotron Emission Behind a Foreground Faraday Screen", curves_emit, "Pdir_two_screens_emit.png")
        plot_case("Emitting Volume Plus Foreground Faraday Screen", curves_mixed, "Pdir_two_screens_mixed.pdf")
        plot_case("Emitting Volume Plus Foreground Faraday Screen", curves_mixed, "Pdir_two_screens_mixed.png")
        return

    plt.figure()
    if scenario == "emission":
        I_emit = np.sum(ne, axis=zaxis)
        psi_map = np.random.uniform(0, np.pi, (ny, nx)) * 0.1
        P = I_emit * np.exp(2j * psi_map)
        k, Pk = directional_spectrum(P)
        m = (k>0) & (Pk>0)
        plt.loglog(k[m], Pk[m], label="Emission with random polarization angles")
    else:
        for B0 in B0_list:
            if scenario == "mixed":
                P = polarization_map_mixed(ne,bz,dz,lam,psi,K,"density",zaxis,B0_par=B0,psi0=0.0)
            elif scenario == "screen":
                psi_bg = 0.0
                P_emit = P_emission_only(ne, psi_bg, zaxis=zaxis, use_sum=True)
                P = polarization_map_screen(ne, bz, dz, lam, P_emit, K, zaxis, B0_par=B0)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
                
            k, Pk = directional_spectrum(P)
            m = (k>0) & (Pk>0)
            plt.loglog(
                k[m],
                Pk[m],
                label=fr"$\frac{{B_{{0 \parallel}}}}{{\sigma_{{bz}}}}$={B0/sbz:.2f}"
            )

    plt.xlabel("k"); plt.ylabel("$P_{\\rm dir}(k)$")
    title_suffix = {"mixed": "Mixed (old)", "screen": "Faraday Screen", "emission": "Emission Only"}
    plt.title(f"Effect of mean B along LOS - {title_suffix[scenario]} ($M_{{\\rm A}} = 0.8$)")
    plt.legend()
    plt.tight_layout()
    filename = f"Pdir_meanfield_{scenario}"
    plt.savefig(f"{filename}.png", dpi=200)
    plt.savefig(f"{filename}.pdf", dpi=200)
    print(f"Saved: {filename}")
    # plt.show()

if __name__=="__main__":
    scenario = "two_screens"
    main(scenario)