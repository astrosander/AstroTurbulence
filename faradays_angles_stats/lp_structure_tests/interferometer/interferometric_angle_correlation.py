#!/usr/bin/env python3

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from numpy.fft import fft2, ifft2, fftshift, fftfreq, rfftn, irfftn

@dataclass
class DatasetSpec:
    path: str
    label: str
    f_key: Optional[str] = None
    phi_key: Optional[str] = "Phi"
    ne_key: Optional[str] = "gas_density"
    bz_key: Optional[str] = "k_mag_field"
    x_key: Optional[str] = "x_coor"
    z_key: Optional[str] = "z_coor"

DATASETS: List[DatasetSpec] = [
    DatasetSpec(path="../ms01ma08.mhd_w.00300.vtk.h5", label="Athena"),
    DatasetSpec(path="../synthetic_kolmogorov_normal.h5", label="Synthetic cube"),
]

LAMBDAS_M: Tuple[float, ...] = (0.06, 0.11, 0.21, 0.40)

NBINS     = 240
LOG_BINS  = True
R_MIN     = 1e-3
R_MAX_FRAC= 0.45
K_MIN     = 1e-3
K_MAX_FRAC= 1.0

OUT_DIR   = "figures"
OUT_PREFIX= "interf_corr"

COMPUTE_DPHI_REFERENCE = True
K_HIGHPASS      = 0.0
HIGHPASS_SOFT   = True
HIGHPASS_ORDER  = 4

FLATTEN_LOWK = True
K_FLAT_TO    = 1e-2

def _axis_spacing(coord_1d, name="axis") -> float:
    c = np.unique(coord_1d.ravel())
    dif = np.diff(np.sort(c))
    dif = dif[dif > 0]
    if dif.size: return float(np.median(dif))
    print(f"[!] {name}: could not determine spacing – using 1.0")
    return 1.0

def load_map(spec: DatasetSpec) -> Tuple[np.ndarray, float, bool]:
    path = os.path.expanduser(spec.path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".npz", ".npy"):
        if ext == ".npz":
            data = np.load(path)
            if spec.f_key and spec.f_key in data:
                f_map = data[spec.f_key]
                if f_map.ndim != 2: raise ValueError("f map must be 2D")
                return f_map.astype(float), 1.0, True
            key = spec.phi_key if spec.phi_key else "Phi"
            if key not in data: raise KeyError(f"'{key}' not found in {path}")
            phi = data[key]
            if phi.ndim != 2: raise ValueError("Phi map must be 2D")
            return phi.astype(float), 1.0, False
        else:
            arr = np.load(path)
            if arr.ndim != 2: raise ValueError("Numpy array must be 2D")
            is_f = bool(spec.f_key)
            return arr.astype(float), 1.0, is_f

    with h5py.File(path, "r") as f:
        if spec.f_key and spec.f_key in f:
            f_map = f[spec.f_key][:]
            if f_map.ndim != 2: raise ValueError("f map must be 2D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            return f_map.astype(float), dx, True
        
        if spec.phi_key and spec.phi_key in f:
            Phi = f[spec.phi_key][:]
            if Phi.ndim != 2: raise ValueError(f"{path}:{spec.phi_key} not 2D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            return Phi.astype(float), dx, False

        if "Phi" in f:
            Phi = f["Phi"][:]
            if Phi.ndim != 2: raise ValueError(f"{path}: 'Phi' not 2D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            return Phi.astype(float), dx, False

        if (spec.ne_key in f) and (spec.bz_key in f):
            ne = f[spec.ne_key][:]
            bz = f[spec.bz_key][:]
            if ne.shape != bz.shape or ne.ndim != 3:
                raise ValueError(f"{path}: ne/bz shapes mismatch or not 3D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            dz = _axis_spacing(f[spec.z_key][0,0,:], "z_coor") if (spec.z_key and spec.z_key in f) else 1.0
            Phi = (ne * bz).sum(axis=2) * dz
            return Phi.astype(float), dx, False

    raise RuntimeError(f"{path}: could not load Φ")

def radial_bin_map(Map2D: np.ndarray, dx: float, nbins=240, r_min=1e-3, r_max_frac=0.45, log_bins=True):
    Ny, Nx = Map2D.shape
    y = (np.arange(Ny) - Ny//2)[:,None]
    x = (np.arange(Nx) - Nx//2)[None,:]
    R = np.hypot(y, x) * dx
    r_max = R.max() * float(r_max_frac)
    if log_bins:
        bins = np.logspace(np.log10(max(r_min,1e-8)), np.log10(r_max), nbins+1)
    else:
        bins = np.linspace(0.0, r_max, nbins+1)
    r = R.ravel(); m = Map2D.ravel()
    idx = np.digitize(r, bins) - 1
    good = (idx >= 0) & (idx < len(bins)-1) & np.isfinite(m)
    sums  = np.bincount(idx[good], weights=m[good], minlength=len(bins)-1)
    counts= np.bincount(idx[good], minlength=len(bins)-1)
    prof  = np.full(len(bins)-1, np.nan, float)
    mask  = counts > 0
    prof[mask] = sums[mask] / counts[mask]
    centers = 0.5*(bins[1:] + bins[:-1])
    mask2 = ~np.isnan(prof) & (centers > r_min)
    return centers[mask2], prof[mask2]

def isotropic_spectrum(P2D: np.ndarray, dx: float, nbins=240, k_min=1e-3, k_max_frac=1.0, log_bins=True):
    Ny, Nx = P2D.shape
    ky = fftfreq(Ny, d=dx)
    kx = fftfreq(Nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KY, KX)

    k_max = K.max() * float(k_max_frac)
    if log_bins:
        bins = np.logspace(np.log10(max(k_min,1e-8)), np.log10(k_max), nbins+1)
    else:
        bins = np.linspace(0.0, k_max, nbins+1)

    k = K.ravel(); p = P2D.ravel()
    idx = np.digitize(k, bins) - 1
    good = (idx >= 0) & (idx < len(bins)-1) & np.isfinite(p)
    sums  = np.bincount(idx[good], weights=p[good], minlength=len(bins)-1)
    counts= np.bincount(idx[good], minlength=len(bins)-1)
    prof  = np.full(len(bins)-1, np.nan, float)
    mask  = counts > 0
    prof[mask] = sums[mask] / counts[mask]
    centers = 0.5*(bins[1:] + bins[:-1])
    mask2 = ~np.isnan(prof) & (centers > k_min)
    return centers[mask2], prof[mask2]

def fit_loglog(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float):
    m = (x>=xmin) & (x<=xmax) & np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    if np.count_nonzero(m) < 3: return np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(np.exp(b))

def build_A_B_from_f(f_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    twof = 2.0 * f_map
    return np.cos(twof).astype(float), np.sin(twof).astype(float)

def _make_highpass(shape: Tuple[int,int], dx: float, k0: float, soft: bool=True, order: int=4) -> np.ndarray:
    Ny, Nx = shape
    ky = fftfreq(Ny, d=dx)
    kx = fftfreq(Nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KY, KX)
    if k0 <= 0:
        return np.ones_like(K, dtype=float)
    if soft:
        eps = 1e-30
        H = 1.0 / (1.0 + (k0 / (K + eps))**order)
        H[K == 0.0] = 0.0
        return H.astype(float)
    else:
        return (K >= k0).astype(float)

def interferometric_spectrum(A: np.ndarray, B: np.ndarray, dx: float,
                             k_highpass: float=0.0, hp_soft: bool=True, hp_order: int=4) -> np.ndarray:
    FA = fft2(A); FB = fft2(B)
    if k_highpass > 0.0:
        H = _make_highpass(FA.shape, dx, k_highpass, soft=hp_soft, order=hp_order)
        FA = FA * H
        FB = FB * H
    return (FA*np.conj(FA)).real + (FB*np.conj(FB)).real

def _flatten_low_k(k1d: np.ndarray, Pk: np.ndarray, k0: float) -> np.ndarray:
    if not (np.isfinite(k0) and k0 > 0):
        return Pk
    kmin, kmax = np.nanmin(k1d), np.nanmax(k1d)
    if not (np.isfinite(kmin) and np.isfinite(kmax)):
        return Pk
    Pk_out = np.array(Pk, copy=True)
    low = (k1d < k0) & np.isfinite(Pk_out)
    if not np.any(low):
        return Pk_out
    if k0 <= kmin:
        ref_val = Pk_out[np.nanargmin(k1d)]
    else:
        valid = np.isfinite(k1d) & np.isfinite(Pk_out)
        pos   = valid & (k1d > 0) & (Pk_out > 0)
        if np.sum(pos) >= 2:
            ref_val = float(np.exp(np.interp(np.log(k0), np.log(k1d[pos]), np.log(Pk_out[pos]))))
        else:
            ref_val = float(np.interp(k0, k1d[valid], Pk_out[valid]))
    Pk_out[low] = ref_val
    return Pk_out

def correlation_from_spectrum(P2D: np.ndarray) -> np.ndarray:
    Ny, Nx = P2D.shape
    C = ifft2(P2D).real / (Nx*Ny)
    return fftshift(C)

def run_once_on_f(f_map: np.ndarray, dx: float, label: str, lam_label: str):
    A, B = build_A_B_from_f(f_map)

    P2D = interferometric_spectrum(
        A, B, dx=dx,
        k_highpass=K_HIGHPASS,
        hp_soft=HIGHPASS_SOFT,
        hp_order=HIGHPASS_ORDER
    )

    k1d, Pk = isotropic_spectrum(P2D, dx=dx, nbins=NBINS, k_min=K_MIN, k_max_frac=K_MAX_FRAC, log_bins=LOG_BINS)
    if FLATTEN_LOWK:
        Pk = _flatten_low_k(k1d, Pk, K_FLAT_TO)

    S_map = correlation_from_spectrum(P2D)
    R1d, S_R = radial_bin_map(S_map, dx=dx, nbins=NBINS, r_min=R_MIN, r_max_frac=R_MAX_FRAC, log_bins=LOG_BINS)

    center = np.unravel_index(np.argmax(S_map), S_map.shape)

    if COMPUTE_DPHI_REFERENCE:
        Dphi_R = 0.5*(1.0 - S_R)
    else:
        Dphi_R = None

    return dict(k1d=k1d, Pk=Pk, R1d=R1d, S_R=S_R, Dphi_R=Dphi_R, lam_label=lam_label)

def create_combined_plots(all_results: List[dict], label: str, base: str, ds_idx: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(8,6))
    for res in all_results:
        plt.loglog(res["k1d"], res["Pk"], lw=1.6, label=res["lam_label"])
    
    if len(all_results) > 1 and len(all_results[0]["k1d"]) > 10:
        mid_idx = len(all_results[0]["k1d"]) // 3
        if mid_idx < len(all_results[0]["k1d"]) and all_results[0]["Pk"][mid_idx] > 0:
            k_ref = all_results[0]["k1d"][mid_idx] 
            P_ref = all_results[0]["Pk"][mid_idx]
            
            k_range = np.logspace(np.log10(all_results[0]["k1d"].min()), 
                                  np.log10(all_results[0]["k1d"].max()), 200)
            
            plt.loglog(
                k_range, 
                P_ref * (k_range/k_ref)**(-11/3), 
                "--", lw=1.0, alpha=0.7, 
                label=r"$\propto k^{-11/3}$"
            )

    plt.xlabel(r"$k$")
    plt.ylabel(r"$P_S(k)$")
    plt.title(f"{label}: $P_s(k)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2, loc='best')
    plt.tight_layout()
    out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_spectrum_k")
    plt.savefig(out_prefix + ".png", dpi=160); plt.savefig(out_prefix + ".pdf")
    plt.close()

    plt.figure(figsize=(8,6))
    for res in all_results:
        plt.loglog(res["R1d"], res["S_R"], lw=1.6, label=res["lam_label"])
    plt.xlabel(r"$R$")
    plt.ylabel(r"$S(R)$")
    plt.title(f"{label}: interferometric correlation S(R)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2, loc='best')
    plt.tight_layout()
    out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_S_of_R")
    plt.savefig(out_prefix + ".png", dpi=160); plt.savefig(out_prefix + ".pdf")
    plt.close()

    if COMPUTE_DPHI_REFERENCE:
        plt.figure(figsize=(8,6))
        for res in all_results:
            if res["Dphi_R"] is not None:
                plt.loglog(res["R1d"], res["Dphi_R"], lw=1.6, label=res["lam_label"])
        
        if len(all_results) > 1 and len(all_results[0]["R1d"]) > 10:
            mid_idx = len(all_results[0]["R1d"]) // 3
            if mid_idx < len(all_results[0]["R1d"]) and all_results[0]["Dphi_R"][mid_idx] > 0:
                R_ref = all_results[0]["R1d"][mid_idx]
                D_ref = all_results[0]["Dphi_R"][mid_idx]
                R_range = np.logspace(np.log10(R_ref), np.log10(all_results[0]["R1d"].max()), 20)
                plt.loglog(R_range, D_ref * (R_range/R_ref)**(5/3), "--", lw=1.0, alpha=0.7, 
                          label=r"$\propto R^{5/3}$")
        
        plt.xlabel(r"$R$")
        plt.ylabel(r"$D_\varphi(R)$")
        plt.title(f"{label}: $D_\\varphi(R)=\\frac{{1}}{{2}}(1-S)$")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False, ncol=2, loc='best')
        plt.tight_layout()
        out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_Dphi_of_R")
        plt.savefig(out_prefix + ".png", dpi=160); plt.savefig(out_prefix + ".pdf")
        plt.close()

def main():
    if not DATASETS:
        print("[!] Please set DATASETS in the script to point to your Φ or f maps.")
    os.makedirs(OUT_DIR, exist_ok=True)

    summary: Dict[str, dict] = {"params": {
        "datasets": [asdict(d) for d in DATASETS],
        "lambdas_m": LAMBDAS_M,
        "nbins": NBINS, "log_bins": LOG_BINS,
        "R_MIN": R_MIN, "R_MAX_FRAC": R_MAX_FRAC,
        "K_MIN": K_MIN, "K_MAX_FRAC": K_MAX_FRAC,
        "compute_Dphi_reference": COMPUTE_DPHI_REFERENCE,
    }}

    for ds_idx, spec in enumerate(DATASETS):
        arr2d, dx, is_f = load_map(spec)
        label = spec.label

        sig_phi = float(np.std(arr2d)) if not is_f else np.nan
        base = f"ds{ds_idx}_{os.path.splitext(os.path.basename(spec.path))[0]}"

        if is_f:
            lam_label = "f-map input"
            res = run_once_on_f(arr2d, dx, label, lam_label)
            summary[f"{base}_fmap"] = {
                "dx": dx, "k1d": res["k1d"], "Pk": res["Pk"], "R1d": res["R1d"], "S_R": res["S_R"],
                "Dphi_R": (res["Dphi_R"] if COMPUTE_DPHI_REFERENCE else None),
                "lam_label": lam_label
            }
            create_combined_plots([res], label, base, ds_idx)
        else:
            all_results = []
            for lam in LAMBDAS_M:
                f_map = (lam**2) * arr2d
                N_rms = (lam**2 * sig_phi) / (2.0*np.pi) if np.isfinite(sig_phi) else np.nan
                lam_label = fr"$\mathcal{{N}}_{{\rm rms}}={N_rms:.3g}$"
                res = run_once_on_f(f_map, dx, label, lam_label)
                res["lambda_m"] = lam
                res["N_rms"] = N_rms
                all_results.append(res)
                summary[f"{base}_lam{lam:.3f}"] = {
                    "dx": dx, "lambda_m": lam, "N_rms": N_rms,
                    "k1d": res["k1d"], "Pk": res["Pk"], "R1d": res["R1d"], "S_R": res["S_R"],
                    "Dphi_R": (res["Dphi_R"] if COMPUTE_DPHI_REFERENCE else None),
                    "sigma_Phi": sig_phi
                }
            
            create_combined_plots(all_results, label, base, ds_idx)

    out_npz = os.path.join(OUT_DIR, f"{OUT_PREFIX}_summary.npz")
    to_save = {"meta": json.dumps(summary["params"], indent=2)}
    for k, v in summary.items():
        if k == "params": continue
        for kk, vv in v.items():
            key = f"{k}__{kk}"
            if isinstance(vv, (list, tuple, np.ndarray)):
                to_save[key] = np.array(vv)

    np.savez_compressed(out_npz, **to_save)
    print(f"[done] Saved summary: {out_npz}")
    print(f"[done] Figures saved to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
