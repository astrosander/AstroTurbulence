#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import json
from dataclasses import dataclass

def load_density_and_field(path: str):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = dz = 1.0
    return ne, bz, dx, dz

def cumulative_faraday_depth(ne, bz, dz, K=1.0, axis=0):
    phi_per_cell = K * ne * bz * dz
    return np.cumsum(phi_per_cell, axis=axis)

def polarization_map(ne, bz, dz, lam, K=1.0, emissivity="constant", zaxis=0):
    Φ = cumulative_faraday_depth(ne, bz, dz, K=K, axis=zaxis)
    if emissivity == "constant":
        p_i = np.ones_like(ne, dtype=np.float64)
    elif emissivity == "density":
        p_i = ne.astype(np.float64)
    else:
        raise ValueError("emissivity must be 'constant' or 'density'")
    phase = np.exp(2j * (lam**2) * Φ)
    P = np.sum(p_i * phase, axis=zaxis)
    return P

def azimuthal_average_power(field2d):
    arr = field2d
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    ny, nx = arr.shape
    fft = np.fft.fftshift(np.fft.fft2(arr))
    power2d = np.abs(fft)**2 / (nx*ny)
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)
    k_flat = kr.ravel()
    p_flat = power2d.ravel()
    idx = np.argsort(k_flat)
    k_sorted = k_flat[idx]; p_sorted = p_flat[idx]
    n_bins = int(np.sqrt(nx*ny))
    if n_bins < 16: n_bins = 16
    k_edges = np.linspace(k_sorted.min(), k_sorted.max(), n_bins+1)
    k_centers = 0.5*(k_edges[1:] + k_edges[:-1])
    Pk = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers, dtype=int)
    inds = np.digitize(k_sorted, k_edges) - 1
    valid = (inds>=0) & (inds<n_bins)
    for i, val in zip(inds[valid], p_sorted[valid]):
        Pk[i] += val; counts[i] += 1
    counts[counts==0] = 1
    Pk = Pk / counts
    return k_centers, Pk

def fit_loglog_slope(k, pk, kmin_frac=0.05, kmax_frac=0.45):
    mask = (k > 0) & (pk > 0)
    k = k[mask]; pk = pk[mask]
    if len(k) < 10:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "k_min": np.nan, "k_max": np.nan}
    kmin = kmin_frac * k.max(); kmax = kmax_frac * k.max()
    m = (k>=kmin) & (k<=kmax)
    if m.sum() < 10:
        m = (k>0)
    x = np.log10(k[m]); y = np.log10(pk[m] + 1e-30)
    A = np.vstack([x, np.ones_like(x)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coeff[0]), float(coeff[1])
    y_pred = A @ coeff
    ss_res = float(np.sum((y - y_pred)**2))
    ss_tot = float(np.sum((y - y.mean())**2) + 1e-30)
    r2 = 1 - ss_res/ss_tot
    return {"slope": slope, "intercept": intercept, "r2": r2, "k_min": float(kmin), "k_max": float(kmax)}

def directional_spectrum(P):
    Q = np.real(P); U = np.imag(P)
    amp = np.sqrt(Q**2 + U**2) + 1e-30
    cos2 = Q / amp
    sin2 = U / amp
    fft_cos = np.fft.fftshift(np.fft.fft2(cos2))
    fft_sin = np.fft.fftshift(np.fft.fft2(sin2))
    power2d = (np.abs(fft_cos)**2 + np.abs(fft_sin)**2) / (cos2.shape[0]*cos2.shape[1])
    ny, nx = power2d.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)
    k_flat = kr.ravel(); p_flat = power2d.ravel()
    idx = np.argsort(k_flat)
    k_sorted = k_flat[idx]; p_sorted = p_flat[idx]
    n_bins = int(np.sqrt(nx*ny)); n_bins = max(n_bins, 16)
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

def psa_spectrum(P):
    k, Pk = azimuthal_average_power(P)
    fit = fit_loglog_slope(k, Pk)
    return k, Pk, fit


def polarization_cube(ne, bz, dz, lam_grid, K=1.0, emissivity="density", zaxis=0):
    Φ = cumulative_faraday_depth(ne, bz, dz, K=K, axis=zaxis)
    if emissivity == "constant":
        p_i = np.ones_like(ne, dtype=np.float64)
    elif emissivity == "density":
        p_i = ne.astype(np.float64)
    else:
        raise ValueError("emissivity must be 'constant' or 'density'")
    Pλ = []
    for lam in lam_grid:
        phase = np.exp(2j * (lam**2) * Φ)
        Pmap = np.sum(p_i * phase, axis=zaxis)
        Pλ.append(Pmap)
    return np.array(Pλ)

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

def fit_piecewise_loglog(x, y):
    xm = np.median(x)
    left = x <= xm; right = x > xm
    def fit(x_, y_):
        m = (x_>0) & (y_>0)
        xlog = np.log10(x_[m]); ylog = np.log10(y_[m]+1e-30)
        A = np.vstack([xlog, np.ones_like(xlog)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, ylog, rcond=None)
        slope, intercept = float(coeff[0]), float(coeff[1])
        ypred = A @ coeff
        ssr = float(np.sum((ylog - ypred)**2))
        return slope, intercept, ssr, int(m.sum())
    s1, b1, ssr1, n1 = fit(x[left], y[left])
    s2, b2, ssr2, n2 = fit(x[right], y[right])
    s_all, b_all, ssr_all, n_all = fit(x, y)
    k1, k2 = 2, 4
    aic1 = n_all*np.log(ssr_all/max(n_all,1) + 1e-30) + 2*k1
    aic2 = (n1*np.log(ssr1/max(n1,1) + 1e-30) + n2*np.log(ssr2/max(n2,1) + 1e-30) + 2*k2)
    return {"slope_all": s_all, "intercept_all": b_all, "AIC1": aic1,
            "left_slope": s1, "right_slope": s2, "AIC2": aic2, "split_x": float(xm)}


def finite_diff_derivative(Pλ, lam2):
    nlam = len(lam2)
    shp = Pλ.shape
    if Pλ.ndim == 3:
        nlam_chk, ny, nx = shp
        flat = Pλ.reshape(nlam_chk, ny*nx)
    else:
        nlam_chk, n = shp
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

def var_vs_lambda2(arr_by_lambda):
    mean = np.mean(arr_by_lambda, axis=1)
    mean2 = np.mean(np.abs(arr_by_lambda)**2, axis=1)
    var = mean2 - np.abs(mean)**2
    return var

def fit_loglog(x, y):
    m = (x>0) & (y>0)
    xlog = np.log10(x[m]); ylog = np.log10(y[m]+1e-30)
    A = np.vstack([xlog, np.ones_like(xlog)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, ylog, rcond=None)
    slope, intercept = float(coeff[0]), float(coeff[1])
    ypred = A @ coeff
    ssr = float(np.sum((ylog - ypred)**2))
    sst = float(np.sum((ylog - ylog.mean())**2) + 1e-30)
    r2 = 1 - ssr/sst
    return {"slope": slope, "intercept": intercept, "r2": r2}

def saturation_N_rms(ne, bz, dz, lam, K=1.0, zaxis=0):
    phi_per_cell = K * ne * bz * dz
    phi = np.moveaxis(phi_per_cell, zaxis, 0)
    Nz = phi.shape[0]
    sigma_cell = np.std(phi, axis=0, ddof=1)
    sigma_LOS = np.sqrt(Nz) * np.nanmean(sigma_cell)
    N_rms = (lam**2) * sigma_LOS / (2*np.pi)
    return float(N_rms), float(sigma_LOS), int(Nz)

def save_csv(path, cols, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(cols)

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\synthetic_kolmogorov_normal.h5"
    lam = 0.5
    lam_min = 0.05
    lam_max = 1.0
    nlam = 64
    kfaraday = 1.0
    emissivity = "density"
    zaxis = 0
    out_prefix = "COMP"

    ne, bz, dx, dz = load_density_and_field(h5_path)
    P = polarization_map(ne, bz, dz, lam, K=kfaraday, emissivity=emissivity, zaxis=zaxis)

    k_dir, Pk_dir, fit_dir = directional_spectrum(P)
    save_csv(f"{out_prefix}_dirPk.csv", list(zip(k_dir, Pk_dir)), ["k", "P_dir(k)"])
    with open(f"{out_prefix}_dir_fit.json", "w") as f:
        json.dump(fit_dir, f, indent=2)
    plt.figure()
    m = (k_dir>0) & (Pk_dir>0)
    plt.loglog(k_dir[m], Pk_dir[m], marker=".", linestyle="none")
    if np.isfinite(fit_dir["slope"]):
        xx = np.linspace(max(fit_dir["k_min"], k_dir[m].min()), fit_dir["k_max"], 200)
        yy = 10**(fit_dir["intercept"]) * xx**(fit_dir["slope"])
        plt.loglog(xx, yy, label=f"slope={fit_dir['slope']:.2f}")
        plt.legend()
    plt.xlabel("k"); plt.ylabel("P_dir(k)")
    plt.title("new directional spectrum (single λ)")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_dirPk.pdf", dpi=150)

    k_psa, Pk_psa, fit_psa = psa_spectrum(P)
    save_csv(f"{out_prefix}_psaPk.csv", list(zip(k_psa, Pk_psa)), ["k", "P(k)"])
    with open(f"{out_prefix}_psa_fit.json", "w") as f:
        json.dump(fit_psa, f, indent=2)
    plt.figure()
    m = (k_psa>0) & (Pk_psa>0)
    plt.loglog(k_psa[m], Pk_psa[m], marker=".", linestyle="none")
    if np.isfinite(fit_psa["slope"]):
        xx = np.linspace(max(fit_psa["k_min"], k_psa[m].min()), fit_psa["k_max"], 200)
        yy = 10**(fit_psa["intercept"]) * xx**(fit_psa["slope"])
        plt.loglog(xx, yy, label=f"slope={fit_psa['slope']:.2f}, R^2={fit_psa['r2']:.3f}")
        plt.legend()
    plt.xlabel("k"); plt.ylabel("P(k) of P-map")
    plt.title("PSA spectrum (single λ)")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_psaPk.png", dpi=150)

    lam_grid = np.linspace(lam_min, lam_max, nlam)
    lam2 = lam_grid**2
    Pλ = polarization_cube(ne, bz, dz, lam_grid, K=kfaraday, emissivity=emissivity, zaxis=zaxis)
    varP = variance_over_sightlines(Pλ)
    save_csv(f"{out_prefix}_pfa_var.csv", list(zip(lam2, varP)), ["lambda2", "VarP"])
    fits_pfa = fit_piecewise_loglog(lam2, varP)
    with open(f"{out_prefix}_pfa_fit.json", "w") as f:
        json.dump(fits_pfa, f, indent=2)
    plt.figure()
    m = (lam2>0) & (varP>0)
    plt.loglog(lam2[m], varP[m], marker=".", linestyle="none")
    if np.isfinite(fits_pfa["slope_all"]):
        xx = np.linspace(lam2[m].min(), lam2.max(), 200)
        yy = 10**(fits_pfa["intercept_all"]) * xx**(fits_pfa["slope_all"])
        plt.loglog(xx, yy, label=f"all slope={fits_pfa['slope_all']:.2f}")
        plt.legend()
    plt.xlabel(r"$\lambda^2$"); plt.ylabel(r"Var$[P(\lambda)]$")
    plt.title("PFA/PVA: variance vs $\\lambda^2$")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_pfa_var.png", dpi=150)

    dP = finite_diff_derivative(Pλ, lam2)
    var_dP = var_vs_lambda2(dP)
    save_csv(f"{out_prefix}_der_var.csv", list(zip(lam2, var_dP)), ["lambda2", "Var_dP"])
    fit_der = fit_loglog(lam2, var_dP)
    with open(f"{out_prefix}_der_fit.json", "w") as f:
        json.dump(fit_der, f, indent=2)
    plt.figure()
    m = (lam2>0) & (var_dP>0)
    plt.loglog(lam2[m], var_dP[m], marker=".", linestyle="none")
    if np.isfinite(fit_der["slope"]):
        xx = np.linspace(lam2[m].min(), lam2.max(), 200)
        yy = 10**(fit_der["intercept"]) * xx**(fit_der["slope"])
        plt.loglog(xx, yy, label=f"slope={fit_der['slope']:.2f}, R^2={fit_der['r2']:.3f}")
        plt.legend()
    plt.xlabel(r"$\lambda^2$"); plt.ylabel(r"Var$[\partial P/\partial(\lambda^2)]$")
    plt.title("Derivative-based statistic")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_der_var.png", dpi=150)

    N_rms, sigma_LOS, Nz = saturation_N_rms(ne, bz, dz, lam, K=kfaraday, zaxis=zaxis)
    sat_info = {"N_rms": N_rms, "sigma_Phi_LOS": sigma_LOS, "Nz": Nz}
    with open(f"{out_prefix}_saturation.json", "w") as f:
        json.dump(sat_info, f, indent=2)

    rows = [
        ["measure", "slope", "R2_or_AIC", "notes"],
        ["NEW_directional", f"{fit_dir['slope']:.6g}", f"R2={fit_dir['r2']:.6g}", ""],
        ["PSA",             f"{fit_psa['slope']:.6g}", f"R2={fit_psa['r2']:.6g}", ""],
        ["PFA_all",         f"{fits_pfa['slope_all']:.6g}", f"AIC1={fits_pfa['AIC1']:.6g}", f"split_x={fits_pfa['split_x']:.6g}"],
        ["PFA_piecewise",   f"left={fits_pfa['left_slope']:.6g}; right={fits_pfa['right_slope']:.6g}", f"AIC2={fits_pfa['AIC2']:.6g}", ""],
        ["DER_var",         f"{fit_der['slope']:.6g}", f"R2={fit_der['r2']:.6g}", ""],
        ["saturation",      "", f"N_rms={N_rms:.6g}", f"sigma_LOS={sigma_LOS:.6g}; Nz={Nz}"]
    ]
    with open(f"{out_prefix}_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    summary_json = {
        "NEW_directional": fit_dir,
        "PSA": fit_psa,
        "PFA": fits_pfa,
        "DER": fit_der,
        "saturation": sat_info
    }
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print("== Comparison complete ==")
    print(f"Saved summary: {out_prefix}_summary.csv / {out_prefix}_summary.json")
    print(f"Saturation diagnostic: N_rms={N_rms:.3f} (lambda={lam})")

if __name__ == "__main__":
    main()
