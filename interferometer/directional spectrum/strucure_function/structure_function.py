#!/usr/bin/env python3
"""
Dphi(R) : numerical (directional), simulation-RM, and analytic (two-slope) — compared
=====================================================================================

Input:
  h5_path → HDF5 with datasets:
      gas_density : n_e(x,y,z)
      k_mag_field : B_z(x,y,z)
      x_coor, z_coor (optional; used to read dx,dz)

Output (fig/two_slope_compare/):
  - Dphi_compare_{i}.{png,pdf}  for each wavelength λ_i

Notes
-----
• The *analytic* curve is obtained from a broken power-law 2D RM spectrum P_2D(k),
  evaluated with a *numerical Hankel integral* in *radian* k, and normalized so that
      ∫_0^∞ 2π k P_2D(k) dk = σ_Φ^2
  where σ_Φ^2 is measured directly from your RM map Φ(x,y).

• The directional (numerical) curve is computed from A=cos(2f), B=sin(2f),
  P_dir = |Â|^2 + |𝐵̂|^2, S = IFFT2(P_dir)/∑(A^2+B^2), Dφ = ½[1−S(R)].

• The “simulation-RM” curve is λ^4 times the RM structure function D_Φ(R),
  estimated from Φ via Wiener–Khinchin (same radial-averaging as S(R)).

This avoids hypergeometric functions entirely and fixes the usual “cycles vs radians”
mismatch that suppresses the analytic amplitude by orders of magnitude.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from numpy.fft import fft2, ifft2, fftfreq
from scipy.special import j0  # Bessel J0

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # <<< set your path here >>>
    h5_path: str = "../../../faradays_angles_stats/lp_structure_tests/transition_smoothness.h5"#mhd_fields.h5"

    outdir: str = "fig/two_slope_compare"

    # projected-spectrum guide params (3D→2D slope shift not used here)
    alpha3D_low: float  = +1.5
    alpha3D_high: float = -5.0/3.0
    k_break_cyc: float  = 0.06  # *cycles per dx* (from your generator)

    # RM physics
    C_RM: float = 0.81       # usual constant; only scale (shape is our focus)

    # wavelengths to analyze (meters)
    lambdas_m: Tuple[float, ...] = (0.21,)

    # binning / ranges
    nbins_R: int   = 240
    R_min: float   = 1e-2     # in dx units (avoid the central pixel)
    R_max_frac: float = 0.45  # fraction of map half-size

    # plotting
    dpi: int = 160

C = Config()

# ──────────────────────────────────────────────────────────────────────
# I/O and helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing_from_h5(arr, name: str, fallback: float) -> float:
    try:
        u = np.unique(arr.ravel())
        dif = np.diff(np.sort(u))
        dif = dif[dif > 0]
        if dif.size:
            return float(np.median(dif))
    except Exception:
        pass
    print(f"[!] {name}: using fallback dx={fallback}")
    return float(fallback)

def load_cube(path: str):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
        if "x_coor" in f:
            dx = _axis_spacing_from_h5(f["x_coor"][:,0,0], "x_coor", 1.0)
        else:
            dx = 1.0
        if "z_coor" in f:
            dz = _axis_spacing_from_h5(f["z_coor"][0,0,:], "z_coor", 1.0)
        else:
            dz = 1.0
    dx=1.0
    return ne, bz, dx, dz

def project_maps(ne: np.ndarray, bz: np.ndarray, dz: float, C_RM: float):
    Bz_proj = bz.sum(axis=2)
    Phi = C_RM * (ne * bz).sum(axis=2) * dz
    return Bz_proj, Phi

def radial_average_map(Map2D: np.ndarray, dx: float, nbins: int, r_min: float, r_max_frac: float):
    ny, nx = Map2D.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(y, x) * dx
    rmax = R.max() * float(r_max_frac)
    bins = np.logspace(np.log10(max(r_min, 1e-8)), np.log10(rmax), nbins+1)
    idx = np.digitize(R.ravel(), bins) - 1
    m = Map2D.ravel()
    nb = nbins
    good = (idx>=0) & (idx<nb) & np.isfinite(m)
    sums  = np.bincount(idx[good], weights=m[good], minlength=nb)
    cnts  = np.bincount(idx[good], minlength=nb)
    prof  = np.full(nb, np.nan, float)
    nz    = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    rcent = 0.5*(bins[1:] + bins[:-1])
    sel = nz & (rcent > r_min)
    return rcent[sel], prof[sel]

# ──────────────────────────────────────────────────────────────────────
# Directional estimator (numerical) and RM structure function
# ──────────────────────────────────────────────────────────────────────

def directional_S_and_Dphi_from_f(f_map: np.ndarray, dx: float, nbins_R: int, R_min: float, R_max_frac: float):
    A = np.cos(2.0 * f_map)
    B = np.sin(2.0 * f_map)
    FA = fft2(A); FB = fft2(B)
    P2_dir = (FA * np.conj(FA)).real + (FB * np.conj(FB)).real
    power0 = np.sum(A*A + B*B)          # normalization so that S(0)=1
    S_map  = np.fft.fftshift(ifft2(P2_dir).real) / power0
    R, S_R = radial_average_map(S_map, dx, nbins_R, R_min, R_max_frac)
    Dphi_R = 0.5 * (1.0 - S_R)
    return R, S_R, Dphi_R

def Dphi_from_RM_map(Phi: np.ndarray, lam: float, dx: float, nbins_R: int, R_min: float, R_max_frac: float):
    # C_Phi via Wiener–Khinchin
    F = fft2(Phi)
    P2 = (F * np.conj(F)).real
    C_map = np.fft.fftshift(ifft2(P2).real) / (Phi.size)
    R, C_R = radial_average_map(C_map, dx, nbins_R, R_min, R_max_frac)

    sigma2 = Phi.var(ddof=0)
    Dphi_R = (lam**4) * 2.0 * (sigma2 - C_R)   # λ^4 D_Φ(R)
    return R, Dphi_R, sigma2

# ──────────────────────────────────────────────────────────────────────
# Analytic Dphi via numeric Hankel (Bessel) integral in *radian* k
# ──────────────────────────────────────────────────────────────────────

def Dphi_analytic_R_hankel(
    R1d, lam, k_break_cyc, sigma_phi2,
    alpha3D_low=+1.5, alpha3D_high=-(5.0/3.0),
    k_span_lo=1e-4, k_span_hi=1e+4, n_k=4096
):
    """
    Analytic D_phi(R,λ) from a broken power-law P_2D(k), normalized to σ_Φ^2.
    R is in dx units; k grid is in *radians per dx* to match J0(kR).
    """
    gl = alpha3D_low  - 2.0   # -1/2
    gh = alpha3D_high - 2.0   # -11/3
    kb = 2.0 * np.pi * float(k_break_cyc)   # cycles → radians

    # log-k grid around kb (radians)
    kmin = kb * float(k_span_lo)
    kmax = kb * float(k_span_hi)
    k = np.exp(np.linspace(np.log(kmin), np.log(kmax), int(n_k)))

    # dimensionless shape
    x = k / kb
    Pshape = np.where(k < kb, x**gl, x**gh)

    # normalization to measured σ_Φ^2
    # σ^2 = ∫ 2π k P_2D(k) dk = 2π ∫ k [A Pshape] dk
    G0 = 2.0 * np.pi * k * Pshape
    # integrate in log-k space: ∫ f(k) dk = ∫ f(k) k d(ln k)
    I0 = np.trapz(G0, x=np.log(k))
    A  = sigma_phi2 / I0
    P2 = A * Pshape

    # Hankel transform for C_Φ(R)
    R = np.asarray(R1d, dtype=np.float64)
    kR = np.outer(R, k)        # (NR, Nk)
    J = j0(kR)
    # integrand sans Bessel:
    G = 2.0 * np.pi * k * P2
    # d ln k spacing (use gradient to be robust)
    dlnk = np.gradient(np.log(k))
    C_R = J @ (G * dlnk)       # ≈ ∫ 2π k P_2D(k) J0(kR) dk

    Dphi_RM = 2.0 * (sigma_phi2 - C_R)
    Dphi_RM = np.maximum(Dphi_RM, 0.0)  # clip tiny negatives
    Dphi_ang = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * Dphi_RM))
    return Dphi_ang, Dphi_RM

# ──────────────────────────────────────────────────────────────────────
# Plot util
# ──────────────────────────────────────────────────────────────────────

def save_loglog(R, curves,  title, ylabel, path):
    plt.figure(figsize=(7.0,5.4))
    for y, lab, sty in curves:
        plt.loglog(R, y, sty, lw=1.9, label=lab)
    plt.xlabel(r"$R$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path+".png", dpi=C.dpi)
    plt.savefig(path+".pdf")
    plt.close()

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    ne, bz, dx, dz = load_cube(C.h5_path)
    print(f"[load] ne,bz shapes: {ne.shape}, dx={dx}, dz={dz}")

    # build maps
    _, Phi = project_maps(ne, bz, dz, C.C_RM)

    for i, lam in enumerate(C.lambdas_m):
        # Directional (numerical)
        R1, S_R, Dphi_dir = directional_S_and_Dphi_from_f(lam**2 * Phi, dx, C.nbins_R, C.R_min, C.R_max_frac)

        # RM-based (simulation)
        R2, Dphi_rm, sigma2 = Dphi_from_RM_map(Phi, lam, dx, C.nbins_R, C.R_min, C.R_Max_frac if hasattr(C,'R_Max_frac') else C.R_max_frac)

        # Analytic (Hankel, radians, normalized to measured σ_Φ^2)
        Dphi_ana, Dphi_RM_ana = Dphi_analytic_R_hankel(
            R1, lam, C.k_break_cyc, sigma2,
            alpha3D_low=C.alpha3D_low, alpha3D_high=C.alpha3D_high,
            k_span_lo=1e-4, k_span_hi=1e4, n_k=4096
        )

        # Sanity print
        print(f"[λ={lam:.2f} m]  σ_Φ^2(measured)={sigma2:.6e}   D_Φ(large R, ana)≈{float(Dphi_RM_ana[-1]):.6e}")

        # Common R grid for plotting: use R1 (directional)
        # Interp simulation-RM onto R1
        Dphi_rm_interp = np.interp(R1, R2, Dphi_rm)

        # Plot
        save_loglog(
            R1,
            curves=[
                (Dphi_dir, r"numerical", "-"),
                # (Dphi_rm_interp, r"From RM map (simulation)", "-"),
                (Dphi_ana, r"Analytic two-slope", "--"),
            ],
            title=rf"",
            ylabel=r"$D_\varphi(R)$",
            path=os.path.join(C.outdir, f"Dphi_compare_{i}")
        )

    print(f"Saved → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()
