#!/usr/bin/env python3

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from numpy.fft import rfftn, irfftn, fftshift

@dataclass
class DatasetSpec:
    path: str
    label: str
    phi_key: Optional[str] = None
    ne_key: Optional[str] = "gas_density"
    bz_key: Optional[str] = "k_mag_field"
    x_key: Optional[str] = "x_coor"
    z_key: Optional[str] = "z_coor"

DATASETS: List[DatasetSpec] = [
    # Example placeholders; replace with your files and labels:
    DatasetSpec(path="../ms01ma08.mhd_w.00300.vtk.h5", label="Athena"),
    DatasetSpec(path="../synthetic_kolmogorov_normal.h5", label="Synthetic cube"),
    # DatasetSpec(path="../synthetic_kolmogorov_nz.h5", label="$n_e$ = const"),
    # DatasetSpec(path="Phi_map.npz", label="Precomputed Φ", phi_key="Phi", ne_key=None, bz_key=None),
]

WAVELENGTHS_M: Tuple[float, ...] = (0.06, 0.11, 0.21, 0.40)
FREQUENCIES_MHZ: Tuple[float, ...] = ()

NBINS     = 480
LOG_BINS  = True
R_MIN     = 1e-3
R_MAX_FRAC= 0.45
FIT_R_MIN = 4.0
FIT_R_MAX = 32.0

OUT_DIR   = "figures"
OUT_PREFIX= "freq_regime_separation"

def _axis_spacing(coord_1d, name="axis") -> float:
    c = np.unique(coord_1d.ravel())
    dif = np.diff(np.sort(c))
    dif = dif[dif > 0]
    if dif.size: return float(np.median(dif))
    print(f"[!] {name}: could not determine spacing – using 1.0")
    return 1.0

def _bin_mean(x, y, bins):
    idx = np.digitize(x, bins) - 1
    good = (idx >= 0) & (idx < len(bins)-1) & np.isfinite(y)
    sums  = np.bincount(idx[good], weights=y[good], minlength=len(bins)-1)
    counts= np.bincount(idx[good], minlength=len(bins)-1)
    out   = np.full(len(bins)-1, np.nan, float)
    m = counts > 0
    out[m] = sums[m]/counts[m]
    centers = 0.5*(bins[1:]+bins[:-1])
    return centers, out

def structure_function_2d(field2d: np.ndarray, dx: float, nbins=480, r_min=1e-3, r_max_frac=0.45, log_bins=True):
    f = field2d.astype(float)
    f = f - f.mean()
    var = float(np.var(f))
    F  = rfftn(f)
    ac = irfftn(np.abs(F)**2, s=f.shape) / f.size
    ac = fftshift(ac)
    Dmap = 2.0*(var - ac)
    Dmap[Dmap < 0] = 0.0

    ny, nx = f.shape
    y = np.arange(ny)[:,None] - ny//2
    x = np.arange(nx)[None,:] - nx//2
    R = np.hypot(y, x) * dx

    r_max = float(R.max()) * float(r_max_frac)
    if log_bins:
        bins = np.logspace(np.log10(max(r_min,1e-8)), np.log10(r_max), nbins+1)
    else:
        bins = np.linspace(0.0, r_max, nbins+1)
    Rc, Dr = _bin_mean(R.ravel(), Dmap.ravel(), bins)
    m = ~np.isnan(Dr) & (Rc > r_min)
    return Rc[m], Dr[m], var

def load_phi(spec: DatasetSpec) -> Tuple[np.ndarray, float, float]:
    path = os.path.expanduser(spec.path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".npz", ".npy"):
        if ext == ".npz":
            data = np.load(path)
            key = spec.phi_key if spec.phi_key else "Phi"
            if key not in data: raise KeyError(f"{path}: no '{key}' in npz")
            Phi = data[key]
        else:
            Phi = np.load(path)
        if Phi.ndim != 2: raise ValueError(f"{path}: expected 2D Φ, got {Phi.shape}")
        dx = dz = 1.0
        return Phi.astype(float), dx, dz

    with h5py.File(path, "r") as f:
        if spec.phi_key and spec.phi_key in f:
            Phi = f[spec.phi_key][:]
            if Phi.ndim != 2: raise ValueError(f"{path}:{spec.phi_key} not 2D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            return Phi.astype(float), dx, 1.0

        if "Phi" in f:
            Phi = f["Phi"][:]
            if Phi.ndim != 2: raise ValueError(f"{path}: 'Phi' not 2D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            return Phi.astype(float), dx, 1.0

        if (spec.ne_key in f) and (spec.bz_key in f):
            ne = f[spec.ne_key][:]
            bz = f[spec.bz_key][:]
            if ne.shape != bz.shape or ne.ndim != 3:
                raise ValueError(f"{path}: ne/bz shapes mismatch or not 3D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            dz = _axis_spacing(f[spec.z_key][0,0,:], "z_coor") if (spec.z_key and spec.z_key in f) else 1.0
            Phi = (ne * bz).sum(axis=2) * dz
            return Phi.astype(float), dx, dz

    raise RuntimeError(f"{path}: could not load Φ")

def Dphi_from_DPhi(DPhi: np.ndarray, lam: float) -> np.ndarray:
    return 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * DPhi))

def fit_loglog(R: np.ndarray, Y: np.ndarray, rmin: float, rmax: float) -> Tuple[float,float]:
    m = (R >= rmin) & (R <= rmax) & np.isfinite(Y) & (Y > 0)
    if np.count_nonzero(m) < 3: return np.nan, np.nan
    X = np.log(R[m]); Z = np.log(Y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, logC = np.linalg.lstsq(A, Z, rcond=None)[0]
    return float(a), float(np.exp(logC))

def rotations_rms(lam: float, sigma_phi: float) -> float:
    return (lam*lam * sigma_phi) / (2.0*np.pi)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    lam_list = list(WAVELENGTHS_M)
    if FREQUENCIES_MHZ:
        c = 299_792_458.0
        lam_from_nu = [c/(1e6*nu) for nu in FREQUENCIES_MHZ]
        lam_list += lam_from_nu
    lam_list = tuple(sorted(lam_list))

    all_out = {"params": {
        "datasets": [asdict(d) for d in DATASETS],
        "wavelengths_m": lam_list,
        "nbins": NBINS, "log_bins": LOG_BINS,
        "r_min": R_MIN, "r_max_frac": R_MAX_FRAC,
        "fit_r_min": FIT_R_MIN, "fit_r_max": FIT_R_MAX,
    }}

    for ds_idx, spec in enumerate(DATASETS):
        Phi, dx, dz = load_phi(spec)
        sigma_phi = float(Phi.std())
        R, Dphi_RM, var_phi = structure_function_2d(Phi, dx=dx, nbins=NBINS, r_min=R_MIN,
                                                   r_max_frac=R_MAX_FRAC, log_bins=LOG_BINS)

        alpha_phi, _ = fit_loglog(R, Dphi_RM, FIT_R_MIN*dx, FIT_R_MAX*dx)

        plt.figure(figsize=(6,4))
        plt.loglog(R, Dphi_RM, lw=1.8, label="$D_\\Phi(R)$")
        mid = len(R)//3 if len(R)>=3 else 1
        if mid < len(R) and Dphi_RM[mid] > 0:
            plt.loglog(R, Dphi_RM[mid]*(R/R[mid])**(5/3), "--", lw=1.0, label=r"$\propto R^{5/3}$")
        plt.xlabel("R")
        plt.ylabel(r"$D_{\Phi}(R)$")
        plt.title(f"{spec.label}: RM structure (fit: α={alpha_phi:.2f})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        out1 = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_RM_structure")
        plt.savefig(out1 + ".png", dpi=160); plt.savefig(out1 + ".pdf")
        plt.close()

        plt.figure(figsize=(6,4))
        slopes = []
        N_labels = []
        for lam in lam_list:
            Y = Dphi_from_DPhi(Dphi_RM, lam)
            a, _ = fit_loglog(R, Y, FIT_R_MIN*dx, FIT_R_MAX*dx)
            slopes.append(a)
            N = rotations_rms(lam, sigma_phi)
            N_labels.append(N)
            plt.loglog(R, Y, lw=1.35, label=fr"$\mathcal{{N}}_{{\rm rms}}={N:.3g}$")
            mid = len(R)//3 if len(R)>=3 else 1
            if mid < len(R) and Y[mid] > 0:
                plt.loglog(R, Y[mid]*(R/R[mid])**(5/3), "--", lw=0.8, alpha=0.6)

        plt.xlabel("R")
        plt.ylabel(r"$D_{\varphi}(R,\lambda)$")
        plt.title(f"Polarization-angle structure ({spec.label})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        out2 = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_Dphi_vs_R")
        plt.savefig(out2 + ".png", dpi=160); plt.savefig(out2 + ".pdf")
        plt.close()

        plt.figure(figsize=(6,4))
        for lam, N in zip(lam_list, N_labels):
            Y = Dphi_from_DPhi(Dphi_RM, lam) / (lam**4 + 1e-300)
            plt.loglog(R, Y, lw=1.35, label=fr"$\mathcal{{N}}_{{\rm rms}}={N:.3g}$")
        plt.xlabel("R")
        plt.ylabel(r"$D_{\varphi}(R,\lambda)/\lambda^4$")
        plt.title(f"$\\lambda^4$ collapse ({spec.label})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        out3 = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_Dphi_over_lambda4")
        plt.savefig(out3 + ".png", dpi=160); plt.savefig(out3 + ".pdf")
        plt.close()

        base = f"ds{ds_idx}_{os.path.splitext(os.path.basename(spec.path))[0]}"
        all_out[base] = dict(
            R=R, Dphi_RM=Dphi_RM, alpha_phi=alpha_phi, var_phi=var_phi, sigma_phi=sigma_phi,
            lambdas_m=np.array(lam_list), N_rms=np.array(N_labels),
            slopes=np.array(slopes)
        )

    out_npz = os.path.join(OUT_DIR, f"{OUT_PREFIX}_summary.npz")
    np.savez_compressed(out_npz, **{
        "meta": json.dumps(all_out["params"], indent=2),
        **{k: v for k, v in all_out.items() if k != "params"}
    })
    print(f"Saved summary: {out_npz}")
    print(f"Figures in:    {os.path.abspath(OUT_DIR)}")
    if not DATASETS:
        print("[note] No datasets listed yet. Edit DATASETS in the script.")

if __name__ == "__main__":
    main()
