#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass

@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    k_min: float
    k_max: float

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
    contrib = p_i * phase
    P = np.sum(contrib, axis=zaxis)
    return P

def azimuthal_average_power(img_complex):
    arr = img_complex
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D P map after integrating along z; got shape {arr.shape}")
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
    k_sorted = k_flat[idx]
    p_sorted = p_flat[idx]
    n_bins = int(np.sqrt(nx*ny))
    k_edges = np.linspace(k_sorted.min(), k_sorted.max(), n_bins+1)
    k_centers = 0.5*(k_edges[1:] + k_edges[:-1])
    Pk = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers, dtype=int)
    inds = np.digitize(k_sorted, k_edges) - 1
    valid = (inds>=0) & (inds<n_bins)
    for i, val in zip(inds[valid], p_sorted[valid]):
        Pk[i] += val
        counts[i] += 1
    counts[counts==0] = 1
    Pk /= counts
    return k_centers, Pk

def fit_loglog_slope(k, pk, kmin_frac=0.05, kmax_frac=0.45):
    mask = (k > 0)
    k = k[mask]
    pk = pk[mask]
    if len(k) < 10:
        return FitResult(np.nan, np.nan, np.nan, np.nan, np.nan)
    kmin = kmin_frac * k.max()
    kmax = kmax_frac * k.max()
    m = (k>=kmin) & (k<=kmax)
    if m.sum() < 10:
        m = mask
    x = np.log10(k[m])
    y = np.log10(pk[m] + 1e-30)
    A = np.vstack([x, np.ones_like(x)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coeff[0], coeff[1]
    y_pred = A @ coeff
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1 - ss_res/ss_tot
    return FitResult(slope, intercept, r2, float(kmin), float(kmax))

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
    lam = 0.5
    kfaraday = 1.0
    emissivity = "density"
    zaxis = 0
    out_prefix = "PSA_out"

    ne, bz, dx, dz = load_density_and_field(h5_path)
    P = polarization_map(ne, bz, dz, lam, K=kfaraday, emissivity=emissivity, zaxis=zaxis)
    k, Pk = azimuthal_average_power(P)
    fit = fit_loglog_slope(k, Pk)

    np.savetxt(f"{out_prefix}_Pk.csv", np.c_[k, Pk], delimiter=",", header="k,Pk", comments="")
    with open(f"{out_prefix}_fit.json", "w") as f:
        json.dump({"slope": fit.slope, "intercept": fit.intercept, "r2": fit.r2,
                   "k_min": fit.k_min, "k_max": fit.k_max}, f, indent=2)

    plt.figure()
    plt.loglog(k[k>0], Pk[k>0], marker=".", linestyle="none")
    if np.isfinite(fit.slope):
        xx = np.linspace(max(fit.k_min, k[k>0].min()), fit.k_max, 100)
        yy = 10**(fit.intercept) * xx**(fit.slope)
        plt.loglog(xx, yy, label=f"slope={fit.slope:.2f}, R^2={fit.r2:.3f}")
        plt.legend()
    plt.xlabel("k (arb. units)")
    plt.ylabel("P(k) of polarization map")
    plt.title("PSA: Spatial Power Spectrum at fixed λ")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_Pk.png", dpi=150)
    print(f"[PSA] Saved: {out_prefix}_Pk.csv, {out_prefix}_fit.json, {out_prefix}_Pk.png")

if __name__ == "__main__":
    main()
