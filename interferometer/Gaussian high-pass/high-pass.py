#!/usr/bin/env python3
"""
Interferometric-Style Angle Correlation with Gaussian High-Pass
===============================================================

Adds an **analytic Gaussian high-pass** on low spatial frequencies to mimic missing short
spacings (e.g., simple dish / baseline filtering):

    H_hp(k) = 1 - exp( - k^2 / (2 k_c^2) )

Applied to the Fourier transforms of A=cos 2f and B=sin 2f before forming power:

    P_S(k) = | H_hp(k) * FFT(A) |^2  +  | H_hp(k) * FFT(B) |^2

This cleanly removes large-scale (low-k) contributions *without* angle-wrap saturation.

Outputs: isotropic P_S(k) and S(R) (and optional D_phi), with **all wavelengths on the same plot**.
The legends are labeled by full rotations N_rms = λ^2 σ_Φ / (2π).

Edit CONFIG below (paths, λ, cutoff, units). No argparse.
"""

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from numpy.fft import fft2, ifft2, fftshift, fftfreq

# ──────────────────────────────────────────────────────────────────────
# CONFIG (edit)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    path: str
    label: str
    # Choose one:
    #  - If you already have an angle map f(x) [rad], set f_key and leave phi_key=None.
    #  - If you have a Φ map [rad m^-2], set phi_key (e.g. "Phi") and leave f_key=None;
    #    then we form f = λ^2 Φ for each λ in LAMBDAS_M.
    f_key: Optional[str] = None
    phi_key: Optional[str] = "Phi"
    # Optional HDF5 coordinate keys (for dx units); otherwise dx=1
    x_key: Optional[str] = "x_coor"
    ne_key: Optional[str] = "gas_density"
    bz_key: Optional[str] = "k_mag_field"
    z_key: Optional[str] = "z_coor"

# Example datasets (replace with your paths)
DATASETS: List[DatasetSpec] = [
    DatasetSpec(path="../../ms01ma08.mhd_w.00300.vtk.h5", label="Athena"),
    DatasetSpec(path="../../synthetic_kolmogorov_normal.h5", label="Synthetic cube"),
    # DatasetSpec(path="Phi_map.npz", label="Precomputed Φ", phi_key="Phi"),
]

# Wavelengths [m] used only when loading Φ maps
LAMBDAS_M: Tuple[float, ...] = (0.06, 0.11, 0.21, 0.40)

# --- Gaussian high-pass parameters ---
K_UNITS: str = "cycles"  # "cycles" (cycles/dx) or "angular" (rad/dx)
K_CUT: float = 0.09      # cutoff wavenumber k_c in chosen units (0 → disabled)
# Notes:
#   cycles/dx → real-space scale  L_cut ≈ 1/k_c
#   rad/dx    → real-space scale  L_cut ≈ 2π/k_c

# Radial binning
NBINS      = 240
LOG_BINS   = True
R_MIN      = 1e-3
R_MAX_FRAC = 0.45
K_MIN      = 1e-3
K_MAX_FRAC = 1.0

# Output
OUT_DIR    = "figures"
OUT_PREFIX = "interf_corr_gaussHP"

# Toggle to compute D_φ(R)=½(1−S) for reference
COMPUTE_DPHI_REFERENCE = True

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing(coord_1d, name="axis") -> float:
    c = np.unique(coord_1d.ravel())
    dif = np.diff(np.sort(c)); dif = dif[dif > 0]
    if dif.size: return float(np.median(dif))
    print(f"[!] {name}: could not determine spacing – using 1.0")
    return 1.0

def load_map(spec: DatasetSpec) -> Tuple[np.ndarray, float, bool]:
    """Load a 2D angle map f or a 2D Φ map. Return (array2d, dx, is_f_map)."""
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
        arr = np.load(path)
        if arr.ndim != 2: raise ValueError("Array must be 2D")
        is_f = bool(spec.f_key)
        return arr.astype(float), 1.0, is_f

    # HDF5
    with h5py.File(path, "r") as f:
        dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
        if spec.f_key and spec.f_key in f:
            f_map = f[spec.f_key][:]
            if f_map.ndim != 2: raise ValueError("f map must be 2D")
            return f_map.astype(float), dx, True
        if spec.phi_key and spec.phi_key in f:
            Phi = f[spec.phi_key][:]
            if Phi.ndim != 2: raise ValueError("Phi map must be 2D")
            return Phi.astype(float), dx, False
        if "Phi" in f:
            Phi = f["Phi"][:]
            if Phi.ndim != 2: raise ValueError("'Phi' must be 2D")
            return Phi.astype(float), dx, False
        if (spec.ne_key in f) and (spec.bz_key in f):
            ne = f[spec.ne_key][:]; bz = f[spec.bz_key][:]
            if ne.shape != bz.shape or ne.ndim != 3:
                raise ValueError("ne/bz shapes mismatch or not 3D")
            dz = _axis_spacing(f[spec.z_key][0,0,:], "z_coor") if (spec.z_key and spec.z_key in f) else 1.0
            Phi = (ne * bz).sum(axis=2) * dz
            return Phi.astype(float), dx, False
    raise RuntimeError(f"{path}: could not load f or Φ")

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

def isotropic_spectrum(P2D: np.ndarray, dx: float, nbins=240, k_min=1e-3, k_max_frac=1.0, log_bins=True, units="cycles"):
    Ny, Nx = P2D.shape
    ky = fftfreq(Ny, d=dx)
    kx = fftfreq(Nx, d=dx)
    if units == "angular":
        ky = 2*np.pi*ky; kx = 2*np.pi*kx
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

def build_A_B_from_f(f_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    twof = 2.0 * f_map
    return np.cos(twof).astype(float), np.sin(twof).astype(float)

def gaussian_highpass_window(shape: Tuple[int,int], dx: float, k_c: float, units: str="cycles") -> np.ndarray:
    """H_hp(K) = 1 - exp(-K^2 / (2 k_c^2)). k_c in 'cycles/dx' or 'rad/dx' depending on units."""
    if k_c <= 0.0:
        return np.ones(shape, float)
    Ny, Nx = shape
    ky = fftfreq(Ny, d=dx); kx = fftfreq(Nx, d=dx)
    if units == "angular":
        ky = 2*np.pi*ky; kx = 2*np.pi*kx
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KY, KX)
    H = 1.0 - np.exp(-0.5 * (K / k_c)**2)
    H[0,0] = 0.0  # ensure DC is removed
    return H.astype(float)

def interferometric_spectrum_with_gaussHP(A: np.ndarray, B: np.ndarray, dx: float,
                                          k_c: float, units: str="cycles") -> np.ndarray:
    """Apply Gaussian high-pass to FFT(A), FFT(B) before forming power spectrum."""
    FA = fft2(A); FB = fft2(B)
    H = gaussian_highpass_window(FA.shape, dx, k_c, units=units)
    FA *= H; FB *= H
    return (FA*np.conj(FA)).real + (FB*np.conj(FB)).real

def correlation_from_spectrum(P2D: np.ndarray) -> np.ndarray:
    """Average cyclic correlation S_map = IFFT(P2D)/(Nx Ny), shifted to center."""
    Ny, Nx = P2D.shape
    C = ifft2(P2D).real / (Nx*Ny)
    return fftshift(C)

def _posmask(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]

def run_once_on_f(f_map: np.ndarray, dx: float, label: str, lam_label: str, k_units: str, k_cut: float):
    # Build cos/sin maps
    A, B = build_A_B_from_f(f_map)

    # Spectrum with Gaussian high-pass
    P2D = interferometric_spectrum_with_gaussHP(A, B, dx=dx, k_c=k_cut, units=k_units)

    # Isotropic spectrum
    k1d, Pk = isotropic_spectrum(P2D, dx=dx, nbins=NBINS, k_min=K_MIN, k_max_frac=K_MAX_FRAC,
                                 log_bins=LOG_BINS, units=k_units)

    # Correlation S(R) and optional D_phi(R)
    S_map = correlation_from_spectrum(P2D)
    R1d, S_R = radial_bin_map(S_map, dx=dx, nbins=NBINS, r_min=R_MIN, r_max_frac=R_MAX_FRAC, log_bins=LOG_BINS)
    #Dphi_R = 0.5*(1.0 - S_R) if COMPUTE_DPHI_REFERENCE else None
    Dphi_R = 0.5*(S_R[0] - S_R) if COMPUTE_DPHI_REFERENCE else None
    print("S_R=", S_R[0] )
    return dict(k1d=k1d, Pk=Pk, R1d=R1d, S_R=S_R, Dphi_R=Dphi_R, lam_label=lam_label)

def cutoff_length(k_c: float, units: str="cycles") -> float:
    """Return the real-space scale L_cut corresponding to k_c."""
    if k_c <= 0: return np.inf
    return (1.0 / k_c) if (units == "cycles") else (2.0*np.pi / k_c)

def create_combined_plots(all_results: List[dict], label: str, ds_idx: int, k_units: str, k_c: float, dx: float):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Spectrum plot (all λ)
    plt.figure(figsize=(8,6))
    for res in all_results:
        x, y = _posmask(res["k1d"], res["Pk"])
        plt.loglog(x, y, lw=1.6, label=res["lam_label"])
    xlabel = r"$k$" if k_units=="cycles" else r"$k$ [rad/dx]"
    plt.xlabel(xlabel); plt.ylabel(r"$P_S(k)$")
    plt.title(f"{label}: $P_S(k)$ (Gaussian high-pass)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_spectrum_k_gaussHP")
    plt.savefig(out_prefix + ".png", dpi=160); plt.savefig(out_prefix + ".pdf")
    plt.close()

    # S(R) plot with vertical line at L_cut
    Lc = cutoff_length(k_c, k_units)
    plt.figure(figsize=(8,6))
    for res in all_results:
        x, y = _posmask(res["R1d"], res["S_R"])
        plt.loglog(x, y, lw=1.6, label=res["lam_label"])
    plt.xlabel(r"$R$"); plt.ylabel(r"$S(R)$")
    plt.title(f"{label}: $S(R)$ (Gaussian high-pass)")
    if np.isfinite(Lc):
        plt.axvline(Lc, color="k", ls="--", lw=1.0, alpha=0.8, label=r"$L_{\rm cut}$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2, loc="best")
    plt.tight_layout()
    out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_S_of_R_gaussHP")
    plt.savefig(out_prefix + ".png", dpi=160); plt.savefig(out_prefix + ".pdf")
    plt.close()

    # D_phi(R) plot (optional)
    if COMPUTE_DPHI_REFERENCE:
        plt.figure(figsize=(8,6))
        for res in all_results:
            if res["Dphi_R"] is not None:
                x, y = _posmask(res["R1d"], res["Dphi_R"])
                plt.loglog(x, y, lw=1.6, label=res["lam_label"])
        plt.xlabel(r"$R$"); plt.ylabel(r"$D_\varphi(R)$")
        plt.title(f"{label}: $D_\\varphi(R)=\\frac{{1}}{{2}}(S(0)-S(R))$ (Gaussian high-pass)")
        if np.isfinite(Lc):
            plt.axvline(Lc, color="k", ls="--", lw=1.0, alpha=0.8, label=r"$L_{\rm cut}$")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False, ncol=2, loc="best")
        plt.tight_layout()
        out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_Dphi_of_R_gaussHP")
        plt.savefig(out_prefix + ".png", dpi=160); plt.savefig(out_prefix + ".pdf")
        plt.close()

def main():
    if not DATASETS:
        print("[!] Please set DATASETS to your Φ or f maps."); return
    os.makedirs(OUT_DIR, exist_ok=True)

    summary: Dict[str, dict] = {"params": {
        "datasets": [asdict(d) for d in DATASETS],
        "lambdas_m": LAMBDAS_M,
        "nbins": NBINS, "log_bins": LOG_BINS,
        "R_MIN": R_MIN, "R_MAX_FRAC": R_MAX_FRAC,
        "K_MIN": K_MIN, "K_MAX_FRAC": K_MAX_FRAC,
        "compute_Dphi_reference": COMPUTE_DPHI_REFERENCE,
        "k_units": K_UNITS, "k_cut": K_CUT,
    }}

    for ds_idx, spec in enumerate(DATASETS):
        arr2d, dx, is_f = load_map(spec)
        label = spec.label

        # If Φ: loop over λ; if f-map: single run
        sig_phi = float(np.std(arr2d)) if not is_f else np.nan
        base = f"ds{ds_idx}_{os.path.splitext(os.path.basename(spec.path))[0]}"
        all_results = []

        if is_f:
            lam_label = "f-map input"
            res = run_once_on_f(arr2d, dx, label, lam_label, K_UNITS, K_CUT)
            all_results.append(res)
            summary[f"{base}_fmap"] = {
                "dx": dx, "k1d": res["k1d"], "Pk": res["Pk"],
                "R1d": res["R1d"], "S_R": res["S_R"],
                "Dphi_R": (res["Dphi_R"] if COMPUTE_DPHI_REFERENCE else None),
                "lam_label": lam_label
            }
        else:
            for lam in LAMBDAS_M:
                f_map = (lam**2) * arr2d
                N_rms = (lam**2 * sig_phi) / (2.0*np.pi) if np.isfinite(sig_phi) else np.nan
                lam_label = fr"$\mathcal{{N}}_{{\rm rms}}={N_rms:.3g}$"
                res = run_once_on_f(f_map, dx, label, lam_label, K_UNITS, K_CUT)
                res["lambda_m"] = lam; res["N_rms"] = N_rms
                all_results.append(res)
                summary[f"{base}_lam{lam:.3f}"] = {
                    "dx": dx, "lambda_m": lam, "N_rms": N_rms,
                    "k1d": res["k1d"], "Pk": res["Pk"],
                    "R1d": res["R1d"], "S_R": res["S_R"],
                    "Dphi_R": (res["Dphi_R"] if COMPUTE_DPHI_REFERENCE else None),
                    "sigma_Phi": sig_phi
                }

        # Make combined plots with all wavelengths
        create_combined_plots(all_results, label, ds_idx, K_UNITS, K_CUT, dx)

    # Save bundle
    out_npz = os.path.join(OUT_DIR, f"{OUT_PREFIX}_gaussHP_summary.npz")
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
    if K_CUT > 0:
        if K_UNITS == "cycles":
            L_cut = 1.0 / K_CUT
        else:
            L_cut = 2.0*np.pi / K_CUT
        print(f"[info] Using k_units='{K_UNITS}', k_cut={K_CUT}. Real-space L_cut ≈ {L_cut:.3g} (dx units).")

if __name__ == "__main__":
    main()
