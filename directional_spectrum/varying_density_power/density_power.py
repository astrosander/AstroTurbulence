import numpy as np, h5py, matplotlib.pyplot as plt, json, csv

def load_density_and_field(path):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    return ne, bz, 1.0, 1.0

def ensure3d(a):
    if a.ndim==3: return a
    if a.ndim==2: return a[:,None,:]
    return a[:,None,None]

def fftfreq_grid(nz, ny, nx):
    kz = np.fft.fftfreq(nz)
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    return KX, KY, KZ

def spectrum_3d(field):
    f = field - np.mean(field)
    nz, ny, nx = f.shape
    F = np.fft.fftn(f)
    P3 = (np.abs(F)**2) / (nz*ny*nx)
    KX, KY, KZ = fftfreq_grid(nz, ny, nx)
    KR = np.sqrt(KX**2 + KY**2 + KZ**2)
    kr = KR.ravel(); p = P3.ravel()
    m = kr>0
    kr = kr[m]; p = p[m]
    idx = np.argsort(kr)
    kr = kr[idx]; p = p[idx]
    n_bins = max(24, int((nz*ny*nx)**(1/3)))
    edges = np.linspace(kr.min(), kr.max(), n_bins+1)
    centers = 0.5*(edges[1:] + edges[:-1])
    Pk = np.zeros_like(centers); cnt = np.zeros_like(centers)
    ind = np.digitize(kr, edges) - 1
    v = (ind>=0) & (ind<n_bins)
    for i,val in zip(ind[v], p[v]):
        Pk[i] += val; cnt[i] += 1
    cnt[cnt==0] = 1
    Pk = Pk / cnt
    return centers, Pk

def fit_loglog(k, pk, kmin_frac=0.05, kmax_frac=0.35):
    m = (k>0) & (pk>0)
    k = k[m]; pk = pk[m]
    if k.size<12:
        return {"slope":np.nan,"intercept":np.nan,"r2":np.nan,"xmin":np.nan,"xmax":np.nan}
    xmin = kmin_frac*k.max(); xmax = kmax_frac*k.max()
    w = (k>=xmin) & (k<=xmax)
    if w.sum()<12: w = (k>0)
    X = np.log10(k[w]); Y = np.log10(pk[w])
    A = np.vstack([X, np.ones_like(X)]).T
    c,_,_,_ = np.linalg.lstsq(A, Y, rcond=None)
    Yp = A @ c
    ssr = float(np.sum((Y-Yp)**2)); sst = float(np.sum((Y-Y.mean())**2) + 1e-30)
    return {"slope":float(c[0]), "intercept":float(c[1]), "r2":1-ssr/sst, "xmin":float(xmin), "xmax":float(xmax)}

def adjust_spectral_slope_3d(field, target_alpha):
    nz, ny, nx = field.shape
    f = field - np.mean(field)
    F = np.fft.fftn(f)
    KX, KY, KZ = fftfreq_grid(nz, ny, nx)
    KR = np.sqrt(KX**2 + KY**2 + KZ**2)
    KR[0,0,0] = 1.0
    k, pk = spectrum_3d(field)
    fit = fit_loglog(k, pk)
    alpha0 = -fit["slope"]
    delta = target_alpha - alpha0
    W = 1.0 / (KR**(delta/2.0))
    G = np.fft.ifftnn(F*W).real if hasattr(np.fft, "ifftnn") else np.fft.ifftn(F*W).real
    G -= np.mean(G)
    s = np.std(G); s = 1.0 if s==0 else s
    G /= s
    return G, alpha0

def directional_spectrum_from_n(ne, bz, dz, lam, K=1.0, zaxis=0):
    phi_cells = K * ne * bz * dz
    phi_to_obs = np.flip(np.cumsum(np.flip(phi_cells, axis=zaxis), axis=zaxis), axis=zaxis)
    P = np.sum(np.exp(2j * (lam**2) * phi_to_obs), axis=zaxis)
    Q = np.real(P); U = np.imag(P)
    A = np.sqrt(Q**2 + U**2) + 1e-30
    c2 = Q/A; s2 = U/A
    def ring_spectrum_2d(arr):
        ny, nx = arr.shape
        F = np.fft.fftshift(np.fft.fft2(arr))
        P2 = (np.abs(F)**2)/(nx*ny)
        ky = np.fft.fftshift(np.fft.fftfreq(ny))
        kx = np.fft.fftshift(np.fft.fftfreq(nx))
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
    k1, p1 = ring_spectrum_2d(c2); k2, p2 = ring_spectrum_2d(s2)
    if k1.shape==k2.shape and np.allclose(k1,k2):
        k = k1; Pk = p1+p2
    else:
        k = k1 if k1.size<=k2.size else k2
        Pk = np.interp(k, k1, p1) + np.interp(k, k2, p2)
    return k, Pk

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
    ne, bz, dx, dz = load_density_and_field(h5_path)
    ne = ensure3d(ne); bz = ensure3d(bz)
    mu = float(np.mean(ne)); rms = float(np.std(ne))
    with open("n_stats.json","w") as f:
        json.dump({"mean":mu,"rms":rms}, f, indent=2)
    k0, Pn0 = spectrum_3d(ne)
    fit0 = fit_loglog(k0, Pn0)
    with open("n_slope_original.json","w") as f:
        json.dump({"slope_Pk":float(fit0["slope"]), "alpha":float(-fit0["slope"]), "r2":float(fit0["r2"])}, f, indent=2)
    plt.figure()
    m0 = (k0>0) & (Pn0>0)
    plt.loglog(k0[m0], Pn0[m0], label="original")
    if np.isfinite(fit0["slope"]):
        xx = np.geomspace(k0[m0].min(), k0[m0].max(), 256)
        yy = 10**fit0["intercept"] * xx**fit0["slope"]
        plt.loglog(xx, yy, linestyle="--")
    plt.xlabel("k"); plt.ylabel("$P_n(k)$")
    plt.title("3D density spectrum (log–log)")
    plt.legend(); plt.tight_layout(); plt.savefig("n_power_original.png", dpi=200); plt.savefig("n_power_original.pdf", dpi=200)
    targets = [2.7, 3.0, 3.3, 3.67, 3.9, 4.0]
    out_curves = {}
    plt.figure()
    for a in targets:
        g, alpha0 = adjust_spectral_slope_3d(ne, a)
        g = mu + rms*(g/np.std(g))
        k, Pk = spectrum_3d(g)
        fit = fit_loglog(k, Pk)
        out_curves[str(a)] = {"target_alpha":a, "measured_slope":float(fit["slope"]), "measured_alpha":float(-fit["slope"]), "r2":float(fit["r2"])}
        m = (k>0) & (Pk>0)
        plt.loglog(k[m], Pk[m], label=f"target $\\alpha={a}$, fit $\\alpha={-fit['slope']:.2f}$")
        if np.isfinite(fit["slope"]):
            xx = np.geomspace(k[m].min(), k[m].max(), 200)
            yy = 10**fit["intercept"] * xx**fit["slope"]
            plt.loglog(xx, yy, linestyle="--")
    plt.xlabel("k"); plt.ylabel("$P_n(k)$")
    plt.title("3D density spectra for target slopes")
    plt.legend(); plt.tight_layout(); plt.savefig("n_power_targets.png", dpi=200); plt.savefig("n_power_targets.pdf", dpi=200)
    lam = 0.5
    k_dir0, Pdir0 = directional_spectrum_from_n(ne, bz, dz, lam)
    m = (k_dir0>0) & (Pdir0>0)
    plt.figure()
    plt.loglog(k_dir0[m], Pdir0[m], label="original n")
    for a in targets:
        g, _ = adjust_spectral_slope_3d(ne, a)
        g = mu + rms*(g/np.std(g))
        k_dir, Pdir = directional_spectrum_from_n(g, bz, dz, lam)
        md = (k_dir>0) & (Pdir>0)
        plt.loglog(k_dir[md], Pdir[md], label=f"$\\alpha_n={a}$")
    plt.xlabel("k"); plt.ylabel(r"$P_{\rm dir}(k)$")
    plt.title("Directional spectra vs. imposed density spectral slope")
    plt.legend(); plt.tight_layout(); plt.savefig("Pdir_vs_n_slope.png", dpi=200); plt.savefig("Pdir_vs_n_slope.pdf", dpi=200)
    with open("n_power_targets_summary.json","w") as f:
        json.dump({"original_alpha":float(-fit0["slope"]), "targets":out_curves}, f, indent=2)
    with open("n_power_original.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["k","P_n(k)"]); w.writerows(zip(k0.tolist(), Pn0.tolist()))
    for a in targets:
        g, _ = adjust_spectral_slope_3d(ne, a)
        g = mu + rms*(g/np.std(g))
        k, Pk = spectrum_3d(g)
        with open(f"n_power_alpha_{a}.csv","w",newline="") as fcsv:
            w=csv.writer(fcsv); w.writerow(["k","P_n(k)"]); w.writerows(zip(k.tolist(), Pk.tolist()))

if __name__=="__main__":
    main()
