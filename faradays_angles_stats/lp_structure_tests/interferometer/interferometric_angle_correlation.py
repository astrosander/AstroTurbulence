#!/usr/bin/env python3
"""
Interferometric-Style Angle Correlation via cos(2f) & sin(2f)
=============================================================

Goal
----
Compute the correlation of polarization angles WITHOUT explicitly forming D_φ,
by using the identity
    ⟨cos 2(f1 − f2)⟩ = ⟨cos 2f1 cos 2f2⟩ + ⟨sin 2f1 sin 2f2⟩.
Define maps A(x)=cos 2f(x), B(x)=sin 2f(x). Then the two-point correlation
S(R) ≡ ⟨A(x)A(x+R)⟩ + ⟨B(x)B(x+R)⟩ is exactly the desired
⟨cos 2(f1 − f2)⟩. Using Wiener–Khinchin, each term's correlation is the inverse
FFT of its power spectrum, so the spectrum of S is simply |FFT(A)|^2 + |FFT(B)|^2.

This "go-directly-to-spectrum" path mimics interferometry and avoids angle-wrap
saturation when computing D_φ = ½(1 − S) explicitly.

What the program does
---------------------
• Loads an existing 2D Φ (RM) map *or* a precomputed f-map (angle map).
• For each λ in LAMBDAS_M, constructs f = λ^2 Φ (ignored if an f-map is used).
• Builds A = cos(2f), B = sin(2f), computes 2D spectra P_A = |FFT(A)|^2, P_B = |FFT(B)|^2.
• S-spectrum P_S = P_A + P_B. (This is the Fourier representation of S.)
• Inverse-FFT → S_map (properly normalized average correlation), optional D_φ = ½(1 − S).
• Radial binning: P_S(k) vs k and S(R) vs R (both log–log), labeling each curve by
  number of FULL rotations N_rms = λ^2 σ_Φ / (2π) for interpretability.

Edit the CONFIG block below (no argparse).

Outputs (in OUT_DIR)
--------------------
  *_spectrum_k.png/.pdf     : Isotropic spectrum P_S(k) vs k (log–log)
  *_S_of_R.png/.pdf         : Isotropic correlation S(R) (log–log)
  *_Dphi_of_R.png/.pdf      : Optional D_φ(R)=½[1−S(R)] for reference (log–log)
  *_summary.npz             : k, P(k), R, S(R), D_φ(R) arrays per λ, σ_Φ, N_rms, etc.
"""

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from numpy.fft import fft2, ifft2, fftshift, fftfreq, rfftn, irfftn

# ──────────────────────────────────────────────────────────────────────
# CONFIG (edit)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    path: str
    label: str
    # Choose one of the following representations:
    #  - If you already have an angle map f(x) in radians, set f_key (e.g., "f") and leave phi_key=None.
    #  - If you have a Φ map (RM, rad m^-2), set phi_key (e.g., "Phi") and leave f_key=None;
    #    then we will form f = λ^2 Φ for each λ.
    f_key: Optional[str] = None       # e.g. "f"
    phi_key: Optional[str] = "Phi"    # e.g. "Phi" in HDF5/NPZ

    # HDF5 dataset names for coordinates (optional, used for dx units only)
    ne_key: Optional[str] = "gas_density"
    bz_key: Optional[str] = "k_mag_field"
    x_key: Optional[str] = "x_coor"
    z_key: Optional[str] = "z_coor"

# Example placeholders — replace paths/keys with your files
DATASETS: List[DatasetSpec] = [
    # Example placeholders; replace with your files and labels:
    DatasetSpec(path="../ms01ma08.mhd_w.00300.vtk.h5", label="Athena"),
    DatasetSpec(path="../synthetic_kolmogorov_normal.h5", label="Synthetic cube"),
    # DatasetSpec(path="../synthetic_kolmogorov_nz.h5", label="$n_e$ = const"),
    # DatasetSpec(path="Phi_map.npz", label="Precomputed Φ", phi_key="Phi", ne_key=None, bz_key=None),
]

# Wavelengths [meters] used only if loading Φ (RM) maps.
LAMBDAS_M: Tuple[float, ...] = (0.06, 0.11, 0.21, 0.40)

# Radial binning
NBINS     = 240
LOG_BINS  = True
R_MIN     = 1e-3        # same units as dx
R_MAX_FRAC= 0.45
K_MIN     = 1e-3        # in 1/dx
K_MAX_FRAC= 1.0

# Output
OUT_DIR   = "figures"
OUT_PREFIX= "interf_corr"

# Toggle this to also compute D_φ(R) for reference (not required to avoid saturation)
COMPUTE_DPHI_REFERENCE = True

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing(coord_1d, name="axis") -> float:
    c = np.unique(coord_1d.ravel())
    dif = np.diff(np.sort(c))
    dif = dif[dif > 0]
    if dif.size: return float(np.median(dif))
    print(f"[!] {name}: could not determine spacing – using 1.0")
    return 1.0

def load_map(spec: DatasetSpec) -> Tuple[np.ndarray, float, bool]:
    """
    Load either an angle map f(x) [radians] or a Φ (RM) map [rad m^-2].
    Returns (arr2d, dx, is_f_map).
    Supports .npz/.npy or HDF5.
    """
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
            # Assume Φ if phi_key default, else assume f-map if told so
            is_f = bool(spec.f_key)
            return arr.astype(float), 1.0, is_f

    # HDF5
    with h5py.File(path, "r") as f:
        # Prefer f-map if present
        if spec.f_key and spec.f_key in f:
            f_map = f[spec.f_key][:]
            if f_map.ndim != 2: raise ValueError("f map must be 2D")
            dx = _axis_spacing(f[spec.x_key][:,0,0], "x_coor") if (spec.x_key and spec.x_key in f) else 1.0
            return f_map.astype(float), dx, True
        
        # Try Φ
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
    """Isotropic radial profile (R vs ⟨Map(R)⟩), assuming Map centered at pixel [Ny//2,Nx//2]."""
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
    """
    Radially average a 2D power spectrum onto k=|k| bins.
    Uses wavenumbers in cycles per dx (from fftfreq).
    """
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

# ──────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────

def build_A_B_from_f(f_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return A=cos(2f), B=sin(2f) as float64 arrays."""
    twof = 2.0 * f_map
    return np.cos(twof).astype(float), np.sin(twof).astype(float)

def interferometric_spectrum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return 2D spectrum P_S(k) = |FFT(A)|^2 + |FFT(B)|^2 (no normalization here).
    For correlation via ifft2, we divide by Npix there to produce average correlation.
    """
    FA = fft2(A); FB = fft2(B)
    return (FA*np.conj(FA)).real + (FB*np.conj(FB)).real

def correlation_from_spectrum(P2D: np.ndarray) -> np.ndarray:
    """
    Average (cyclic) correlation map via Wiener–Khinchin:
        C(R) = (1/Npix) * IFFT2( P2D )
    """
    Ny, Nx = P2D.shape
    C = ifft2(P2D).real / (Nx*Ny)
    return fftshift(C)

def run_once_on_f(f_map: np.ndarray, dx: float, label: str, out_prefix: str, lam_label: str):
    # Build cos/sin maps
    A, B = build_A_B_from_f(f_map)

    # Spectrum (interferometric)
    P2D = interferometric_spectrum(A, B)

    # Isotropic spectrum
    k1d, Pk = isotropic_spectrum(P2D, dx=dx, nbins=NBINS, k_min=K_MIN, k_max_frac=K_MAX_FRAC, log_bins=LOG_BINS)

    # Correlation S(R) and optional D_phi(R)
    S_map = correlation_from_spectrum(P2D)
    R1d, S_R = radial_bin_map(S_map, dx=dx, nbins=NBINS, r_min=R_MIN, r_max_frac=R_MAX_FRAC, log_bins=LOG_BINS)

    # Sanity: S(0) ~ 1 because A^2 + B^2 = 1 pointwise
    center = np.unravel_index(np.argmax(S_map), S_map.shape)
    # (No assert, but we can print later if needed)

    # Optional D_phi(R) for reference
    if COMPUTE_DPHI_REFERENCE:
        Dphi_R = 0.5*(1.0 - S_R)
    else:
        Dphi_R = None

    # ---- Plots ----
    os.makedirs(OUT_DIR, exist_ok=True)

    # Spectrum plot
    plt.figure(figsize=(6,4))
    plt.loglog(k1d, Pk, lw=1.6, label=fr"{lam_label}")
    plt.xlabel(r"$k$ (1 / dx)")
    plt.ylabel(r"$P_S(k)$")
    plt.title(f"{label}: spectrum of ⟨cos2f cos2f + sin2f sin2f⟩")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_prefix+"_spectrum_k.png", dpi=160); plt.savefig(out_prefix+"_spectrum_k.pdf")
    plt.close()

    # S(R) plot
    plt.figure(figsize=(6,4))
    plt.loglog(R1d, S_R, lw=1.6, label=fr"{lam_label}")
    plt.xlabel(r"$R$ (dx units)")
    plt.ylabel(r"$S(R)$")
    plt.title(f"{label}: interferometric correlation S(R)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_prefix+"_S_of_R.png", dpi=160); plt.savefig(out_prefix+"_S_of_R.pdf")
    plt.close()

    # D_phi(R) (optional)
    if Dphi_R is not None:
        plt.figure(figsize=(6,4))
        plt.loglog(R1d, Dphi_R, lw=1.6, label=fr"{lam_label}")
        plt.xlabel(r"$R$ (dx units)")
        plt.ylabel(r"$D_\varphi(R)$")
        plt.title(f"{label}: reference $D_\\varphi(R)=\\frac{{1}}{{2}}(1-S)$")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_prefix+"_Dphi_of_R.png", dpi=160); plt.savefig(out_prefix+"_Dphi_of_R.pdf")
        plt.close()

    return dict(k1d=k1d, Pk=Pk, R1d=R1d, S_R=S_R, Dphi_R=Dphi_R)

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

        # If we loaded Φ, we will loop over wavelengths; if an f-map, do a single run.
        sig_phi = float(np.std(arr2d)) if not is_f else np.nan  # Φ std only if Φ
        base = f"ds{ds_idx}_{os.path.splitext(os.path.basename(spec.path))[0]}"

        if is_f:
            lam_label = "f-map input"
            out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_fmap")
            res = run_once_on_f(arr2d, dx, label, out_prefix, lam_label)
            summary[f"{base}_fmap"] = {
                "dx": dx, "k1d": res["k1d"], "Pk": res["Pk"], "R1d": res["R1d"], "S_R": res["S_R"],
                "Dphi_R": (res["Dphi_R"] if COMPUTE_DPHI_REFERENCE else None),
                "lam_label": lam_label
            }
        else:
            # Treat arr2d as Φ map; run for each λ with labels in FULL ROTATIONS N_rms
            for lam in LAMBDAS_M:
                f_map = (lam**2) * arr2d
                N_rms = (lam**2 * sig_phi) / (2.0*np.pi) if np.isfinite(sig_phi) else np.nan
                lam_label = fr"$\mathcal{{N}}_{{\rm rms}}={N_rms:.3g}$"
                out_prefix = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{ds_idx}_lam{lam:.3f}m")
                res = run_once_on_f(f_map, dx, label, out_prefix, lam_label)
                summary[f"{base}_lam{lam:.3f}"] = {
                    "dx": dx, "lambda_m": lam, "N_rms": N_rms,
                    "k1d": res["k1d"], "Pk": res["Pk"], "R1d": res["R1d"], "S_R": res["S_R"],
                    "Dphi_R": (res["Dphi_R"] if COMPUTE_DPHI_REFERENCE else None),
                    "sigma_Phi": sig_phi
                }

    # Save machine-readable bundle
    out_npz = os.path.join(OUT_DIR, f"{OUT_PREFIX}_summary.npz")
    # Convert nested dict to JSON for meta; arrays go directly
    to_save = {"meta": json.dumps(summary["params"], indent=2)}
    for k, v in summary.items():
        if k == "params": continue
        for kk, vv in v.items():
            key = f"{k}__{kk}"
            if isinstance(vv, (list, tuple, np.ndarray)):
                to_save[key] = np.array(vv)
    # The above simple flattener only stores array fields; meta holds the rest.

    np.savez_compressed(out_npz, **to_save)
    print(f"[done] Saved summary: {out_npz}")
    print(f"[done] Figures saved to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
