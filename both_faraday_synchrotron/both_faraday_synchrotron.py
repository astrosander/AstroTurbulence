import numpy as np, h5py, matplotlib.pyplot as plt, csv, json

def load_density_and_field(path):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = dz = 1.0
    return ne, bz, dx, dz

def fbm3d(nz, ny, nx, H, seed=42):
    rng = np.random.default_rng(seed)
    kz = np.fft.fftfreq(nz)
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    k[0,0,0] = 1.0
    beta = 2*H + 3.0
    amp = 1.0 / (k**(beta/2.0))
    phase = np.exp(1j * 2*np.pi * rng.random((nz, ny, nx)))
    fourier = amp * phase
    f = np.fft.ifftn(fourier).real
    f -= f.mean()
    f /= (f.std() + 1e-12)
    return f

def cumulative_to_observer(phi_cells, axis):
    return np.flip(np.cumsum(np.flip(phi_cells, axis=axis), axis=axis), axis=axis)

def polarization_map_mixed(ne, bz, dz, lam, psi, K=1.0, emissivity="density", zaxis=0):
    if emissivity == "constant":
        p_i = np.ones_like(ne, dtype=np.float64)
    elif emissivity == "density":
        p_i = ne.astype(np.float64)
    else:
        raise ValueError("emissivity must be 'constant' or 'density'")
    phi_cells = K * ne * bz * dz
    phi_to_obs = cumulative_to_observer(phi_cells, zaxis)
    phase = np.exp(2j * (lam**2) * phi_to_obs)
    P = np.sum(p_i * np.exp(2j*psi) * phase, axis=zaxis)
    return P

def polarization_cube_mixed(ne, bz, dz, lam_grid, psi, K=1.0, emissivity="density", zaxis=0):
    if emissivity == "constant":
        p_i = np.ones_like(ne, dtype=np.float64)
    elif emissivity == "density":
        p_i = ne.astype(np.float64)
    else:
        raise ValueError("emissivity must be 'constant' or 'density'")
    phi_cells = K * ne * bz * dz
    phi_to_obs = cumulative_to_observer(phi_cells, zaxis)
    Pλ = []
    for lam in lam_grid:
        phase = np.exp(2j * (lam**2) * phi_to_obs)
        Pλ.append(np.sum(p_i * np.exp(2j*psi) * phase, axis=zaxis))
    return np.array(Pλ)

def ring_spectrum(field2d):
    f = field2d
    ny, nx = f.shape
    F = np.fft.fftshift(np.fft.fft2(f))
    P2 = (np.abs(F)**2) / (nx*ny)
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)
    k_flat = kr.ravel(); p_flat = P2.ravel()
    idx = np.argsort(k_flat)
    k_sorted = k_flat[idx]; p_sorted = p_flat[idx]
    n_bins = max(16, int(np.sqrt(nx*ny)))
    k_edges = np.linspace(k_sorted.min(), k_sorted.max(), n_bins+1)
    k_centers = 0.5*(k_edges[1:] + k_edges[:-1])
    Pk = np.zeros_like(k_centers); counts = np.zeros_like(k_centers, dtype=int)
    inds = np.digitize(k_sorted, k_edges) - 1
    valid = (inds>=0) & (inds<n_bins)
    for i, val in zip(inds[valid], p_sorted[valid]):
        Pk[i] += val; counts[i] += 1
    counts[counts==0] = 1
    Pk = Pk / counts
    return k_centers, Pk

def fit_loglog_slope(x, y, xmin_frac=0.1, xmax_frac=0.3):
    x = np.asarray(x); y = np.asarray(y)
    msk = (x>0) & (y>0)
    x = x[msk]; y = y[msk]
    if len(x) < 10:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "xmin": np.nan, "xmax": np.nan}
    xmin = xmin_frac * x.max(); xmax = xmax_frac * x.max()
    m = (x>=xmin) & (x<=xmax)
    if m.sum() < 10:
        m = (x>0)
    X = np.log10(x[m]); Y = np.log10(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    slope, intercept = float(coeff[0]), float(coeff[1])
    Yp = A @ coeff
    ssr = float(np.sum((Y - Yp)**2)); sst = float(np.sum((Y - Y.mean())**2) + 1e-30)
    r2 = 1 - ssr/sst
    return {"slope": slope, "intercept": intercept, "r2": r2, "xmin": float(xmin), "xmax": float(xmax)}

def psa_spectrum(P):
    k, Pk = ring_spectrum(P)
    fit = fit_loglog_slope(k, Pk)
    return k, Pk, fit

def directional_spectrum(P):
    Q = np.real(P); U = np.imag(P)
    amp = np.sqrt(Q**2 + U**2) + 1e-30
    cos2 = Q/amp; sin2 = U/amp
    F1 = np.fft.fftshift(np.fft.fft2(cos2))
    F2 = np.fft.fftshift(np.fft.fft2(sin2))
    P2 = (np.abs(F1)**2 + np.abs(F2)**2) / (cos2.size)
    ny, nx = cos2.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)
    k_flat = kr.ravel(); p_flat = P2.ravel()
    idx = np.argsort(k_flat)
    k_sorted = k_flat[idx]; p_sorted = p_flat[idx]
    n_bins = max(16, int(np.sqrt(nx*ny)))
    k_edges = np.linspace(k_sorted.min(), k_sorted.max(), n_bins+1)
    k_centers = 0.5*(k_edges[1:] + k_edges[:-1])
    Pk = np.zeros_like(k_centers); counts = np.zeros_like(k_centers, dtype=int)
    inds = np.digitize(k_sorted, k_edges) - 1
    valid = (inds>=0) & (inds<n_bins)
    for i, val in zip(inds[valid], p_sorted[valid]):
        Pk[i] += val; counts[i] += 1
    counts[counts==0] = 1
    Pk = Pk / counts
    fit = fit_loglog_slope(k_centers, Pk)
    return k_centers, Pk, fit

def variance_over_sightlines(Pλ):
    if Pλ.ndim == 3:
        nlam, ny, nx = Pλ.shape
        flat = Pλ.reshape(nlam, ny*nx)
    else:
        nlam, n = Pλ.shape
        flat = Pλ
    meanP = np.mean(flat, axis=1)
    meanP2 = np.mean(np.abs(flat)**2, axis=1)
    varP = meanP2 - np.abs(meanP)**2
    return varP

def finite_diff_derivative(Pλ, lam2):
    nlam = len(lam2)
    if Pλ.ndim == 3:
        nlam_chk, ny, nx = Pλ.shape
        flat = Pλ.reshape(nlam_chk, ny*nx)
    else:
        nlam_chk, n = Pλ.shape
        flat = Pλ
    dP = np.empty_like(flat, dtype=np.complex128)
    for i in range(nlam):
        if i == 0:
            dP[i] = (flat[i+1] - flat[i]) / (lam2[i+1] - lam2[i])
        elif i == nlam-1:
            dP[i] = (flat[i] - flat[i-1]) / (lam2[i] - lam2[i-1])
        else:
            dP[i] = (flat[i+1] - flat[i-1]) / (lam2[i+1] - lam2[i-1])
    return dP

def fit_piecewise_loglog(x, y):
    xm = np.median(x)
    left = x <= xm; right = x > xm
    def fit_seg(xs, ys):
        m = (xs>0) & (ys>0)
        X = np.log10(xs[m]); Y = np.log10(ys[m]+1e-30)
        A = np.vstack([X, np.ones_like(X)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        slope, intercept = float(coeff[0]), float(coeff[1])
        Yp = A @ coeff
        ssr = float(np.sum((Y - Yp)**2))
        n = int(m.sum())
        return slope, intercept, ssr, n
    s1,b1,ssr1,n1 = fit_seg(x[left], y[left])
    s2,b2,ssr2,n2 = fit_seg(x[right], y[right])
    s_all,b_all,ssr_all,n_all = fit_seg(x, y)
    k1,k2 = 2,4
    aic1 = n_all*np.log(ssr_all/max(n_all,1)+1e-30) + 2*k1
    aic2 = n1*np.log(ssr1/max(n1,1)+1e-30) + n2*np.log(ssr2/max(n2,1)+1e-30) + 2*k2
    return {"slope_all": s_all, "intercept_all": b_all, "AIC1": aic1, "left_slope": s1, "right_slope": s2, "AIC2": aic2, "split_x": float(xm)}

def save_csv(path, cols, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(cols)

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
    lam_single = 0.5
    lam_min = 0.05
    lam_max = 1.0
    nlam = 64
    emissivity = "density"
    K = 1.0
    zaxis = 0
    H_em = 0.6
    psi_sigma = 0.5
    out = "MIXED"
    ne, bz, dx, dz = load_density_and_field(h5_path)
    shp = ne.shape
    psi = fbm3d(shp[0], shp[1], shp[2] if ne.ndim==3 else 1, H_em)
    if ne.ndim == 2:
        psi = psi[:, :, 0]
    psi = psi_sigma * psi
    P = polarization_map_mixed(ne, bz, dz, lam_single, psi, K=K, emissivity=emissivity, zaxis=zaxis)
    k_dir, Pk_dir, fit_dir = directional_spectrum(P)
    save_csv(f"{out}_dirPk.csv", list(zip(k_dir, Pk_dir)), ["k", "P_dir(k)"])
    with open(f"{out}_dir_fit.json", "w") as f:
        json.dump(fit_dir, f, indent=2)
    # Modern directional spectrum plot
    plt.figure(figsize=(10, 8))
    m = (k_dir>0) & (Pk_dir>0)
    plt.loglog(k_dir[m], Pk_dir[m], marker='o', linestyle='none', markersize=6, 
               alpha=0.7, label='Data', color='blue')
    if np.isfinite(fit_dir["slope"]):
        xx = np.linspace(max(fit_dir["xmin"], k_dir[m].min()), fit_dir["xmax"], 200)
        yy = 10**(fit_dir["intercept"]) * xx**(fit_dir["slope"])
        plt.loglog(xx, yy, ':', linewidth=2, 
                  label=f'Slope = {fit_dir["slope"]:.2f}, R² = {fit_dir["r2"]:.3f}', color='green')
    plt.xlabel("k", fontsize=12)
    plt.ylabel("P_dir(k)", fontsize=12)
    plt.title("Directional Spectrum (Mixed, Single λ)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}_dirPk.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{out}_dirPk.pdf", dpi=150, bbox_inches='tight')
    k_psa, Pk_psa, fit_psa = psa_spectrum(P)
    save_csv(f"{out}_psaPk.csv", list(zip(k_psa, Pk_psa)), ["k", "P(k)"])
    with open(f"{out}_psa_fit.json", "w") as f:
        json.dump(fit_psa, f, indent=2)
    # Modern PSA spectrum plot
    plt.figure(figsize=(10, 8))
    m = (k_psa>0) & (Pk_psa>0)
    plt.loglog(k_psa[m], Pk_psa[m], marker='o', linestyle='none', markersize=6, 
               alpha=0.7, label='Data', color='blue')
    if np.isfinite(fit_psa["slope"]):
        xx = np.linspace(max(fit_psa["xmin"], k_psa[m].min()), fit_psa["xmax"], 200)
        yy = 10**(fit_psa["intercept"]) * xx**(fit_psa["slope"])
        plt.loglog(xx, yy, ':', linewidth=2, 
                  label=f'Slope = {fit_psa["slope"]:.2f}, R² = {fit_psa["r2"]:.3f}', color='green')
    plt.xlabel("k", fontsize=12)
    plt.ylabel("PSA: |P̂(k)|²", fontsize=12)
    plt.title("PSA Spectrum (Mixed, Single λ)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}_psaPk.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{out}_psaPk.pdf", dpi=150, bbox_inches='tight')
    lam_grid = np.linspace(lam_min, lam_max, nlam)
    lam2 = lam_grid**2
    Pλ = polarization_cube_mixed(ne, bz, dz, lam_grid, psi, K=K, emissivity=emissivity, zaxis=zaxis)
    varP = variance_over_sightlines(Pλ)
    save_csv(f"{out}_pfa_var.csv", list(zip(lam2, varP)), ["lambda2", "VarP"])
    fits_pfa = fit_piecewise_loglog(lam2, varP)
    with open(f"{out}_pfa_fit.json", "w") as f:
        json.dump(fits_pfa, f, indent=2)
    # Modern PFA/PVA plot
    plt.figure(figsize=(10, 8))
    m = (lam2>0) & (varP>0)
    plt.loglog(lam2[m], varP[m], marker='o', linestyle='none', markersize=6, 
               alpha=0.7, label='Data', color='blue')
    if np.isfinite(fits_pfa["slope_all"]):
        xx = np.linspace(lam2[m].min(), lam2.max(), 200)
        yy = 10**(fits_pfa["intercept_all"]) * xx**(fits_pfa["slope_all"])
        plt.loglog(xx, yy, ':', linewidth=2, 
                  label=f'Slope = {fits_pfa["slope_all"]:.2f}', color='green')
    plt.xlabel(r"$\lambda^2$", fontsize=12)
    plt.ylabel(r"Var$[P(\lambda)]$", fontsize=12)
    plt.title("PFA/PVA (Mixed)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}_pfa_var.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{out}_pfa_var.pdf", dpi=150, bbox_inches='tight')
    dP = finite_diff_derivative(Pλ, lam2)
    var_dP = variance_over_sightlines(dP)
    save_csv(f"{out}_der_var.csv", list(zip(lam2, var_dP)), ["lambda2", "Var_dP"])
    fit_der = fit_loglog_slope(lam2[lam2>0], var_dP[lam2>0])
    with open(f"{out}_der_fit.json", "w") as f:
        json.dump(fit_der, f, indent=2)
    # Modern derivative statistic plot
    plt.figure(figsize=(10, 8))
    m = (lam2>0) & (var_dP>0)
    plt.loglog(lam2[m], var_dP[m], marker='o', linestyle='none', markersize=6, 
               alpha=0.7, label='Data', color='blue')
    if np.isfinite(fit_der["slope"]):
        xx = np.linspace(lam2[m].min(), lam2.max(), 200)
        yy = 10**(fit_der["intercept"]) * xx**(fit_der["slope"])
        plt.loglog(xx, yy, ':', linewidth=2, 
                  label=f'Slope = {fit_der["slope"]:.2f}, R² = {fit_der["r2"]:.3f}', color='green')
    plt.xlabel(r"$\lambda^2$", fontsize=12)
    plt.ylabel(r"Var$[\partial P/\partial(\lambda^2)]$", fontsize=12)
    plt.title("Derivative Statistic (Mixed)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}_der_var.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{out}_der_var.pdf", dpi=150, bbox_inches='tight')
    phi_cells = K * ne * bz * dz
    phi_total = np.sum(phi_cells, axis=zaxis)
    sigma_phi = np.std(phi_total)
    N_rms = (lam_single**2) * sigma_phi / (2*np.pi)
    sat_info = {"N_rms": float(N_rms), "sigma_Phi_total": float(sigma_phi)}
    with open(f"{out}_saturation.json", "w") as f:
        json.dump(sat_info, f, indent=2)
    rows = [
        ["measure", "slope", "R2_or_AIC", "notes"],
        ["Directional", f"{fit_dir['slope']:.6g}", f"R2={fit_dir['r2']:.6g}", ""],
        ["PSA", f"{fit_psa['slope']:.6g}", f"R2={fit_psa['r2']:.6g}", ""],
        ["PFA_all", f"{fits_pfa['slope_all']:.6g}", f"AIC1={fits_pfa['AIC1']:.6g}", f"split_x={fits_pfa['split_x']:.6g}"],
        ["PFA_piece", f"left={fits_pfa['left_slope']:.6g}; right={fits_pfa['right_slope']:.6g}", f"AIC2={fits_pfa['AIC2']:.6g}", ""],
        ["Derivative", f"{fit_der['slope']:.6g}", f"R2={fit_der['r2']:.6g}", ""],
        ["N_rms", "", f"{N_rms:.6g}", ""],
    ]
    with open(f"{out}_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    summary_json = {"Directional": fit_dir, "PSA": fit_psa, "PFA": fits_pfa, "Derivative": fit_der, "N_rms": sat_info}
    with open(f"{out}_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

if __name__ == "__main__":
    main()
