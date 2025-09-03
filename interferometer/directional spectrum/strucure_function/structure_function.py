#!/usr/bin/env python3
"""
2D observer analysis with full comparison:
  numerical directional Dφ  vs  RM–based (from simulation)  vs  analytic two-slope model
========================================================================================

Inputs (HDF5):
  gas_density : n_e(x,y,z)
  k_mag_field : B_z(x,y,z)
  x_coor, z_coor (optional for dx,dz)

Outputs (fig/two_slope_compare/):
  - proj_Bz_map.{png,pdf}        : LOS-summed Bz
  - proj_RM_map.{png,pdf}        : Faraday screen Φ
  - proj_Bz_E1D.{png,pdf}        : E1D of LOS Bz
  - proj_RM_E1D.{png,pdf}        : E1D of Φ
  For each λ index i:
  - Pdir_ring_i.{png,pdf}        : ring spectrum of |FT cos2f|^2+|FT sin2f|^2
  - S_of_R_i.{png,pdf}           : directional correlation S(R)
  - Dphi_compare_i.{png,pdf}     : ***key panel*** Dφ_num vs Dφ_simRM vs Dφ_ana

Author: you
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from numpy.fft import fft2, ifft2, fftfreq
from scipy.special import gamma

# from mpmath import hyper
from mpmath import mp, hyper
mp.dps = 100

_hyp1f2_scalar = lambda a,b1,b2,zz: float(hyper([a],[b1,b2], float(zz)))
hyp1f2_vec = np.vectorize(_hyp1f2_scalar, otypes=[float], excluded=[0,1,2])


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    h5_path: str = "../../../faradays_angles_stats/lp_structure_tests/mhd_fields.h5"
    # h5_path: str = "mhd_fields.h5"           # your cube (or synthetic_two_slope.h5)
    outdir: str  = "fig/two_slope_compare"

    # binning (log)
    nbins_k: int = 240
    nbins_R: int = 240
    kmin: float  = 1e-3
    kmax_frac: float = 1.0
    R_min: float = 1e-2
    R_max_frac: float = 0.45
    dpi: int     = 160

    # RM constant and wavelengths (meters)
    C_RM: float = 0.81
    lambdas_m: Tuple[float, ...] = (0.06, 0.11, 0.21)

    # Target 3D shell slopes and break (cycles/dx)
    alpha3D_low:  float = +1.5
    alpha3D_high: float = -5.0/3.0
    k_break_cyc:  float = 0.06
    guide_span:   float = 8.0

C = Config()

# ──────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing_1d(coord_1d, fallback=1.0) -> float:
    try:
        u = np.unique(np.asarray(coord_1d).ravel())
        d = np.diff(np.sort(u))
        d = d[d > 0]
        if d.size: return float(np.median(d))
    except Exception:
        pass
    return float(fallback)

def load_cube(path: str):
    with h5py.File(path, "r") as f:
        ne = np.asarray(f["gas_density"][:], dtype=np.float64)
        bz = np.asarray(f["k_mag_field"][:], dtype=np.float64)
        dx = 1.0; dz = 1.0
        if "x_coor" in f: dx = _axis_spacing_1d(f["x_coor"][:,0,0], 1.0)
        if "z_coor" in f: dz = _axis_spacing_1d(f["z_coor"][0,0,:], 1.0)
    return ne, bz, dx, dz

# ──────────────────────────────────────────────────────────────────────
# Observer maps & spectra
# ──────────────────────────────────────────────────────────────────────

def project_maps(ne, bz, dz, C_RM):
    Bz_proj = bz.sum(axis=2)
    Phi     = C_RM * (ne * bz).sum(axis=2) * dz
    return Bz_proj, Phi

def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KY, KX)
    kmax = K.max() * float(kmax_frac)
    bins = np.logspace(np.log10(max(kmin, 1e-8)), np.log10(kmax), nbins+1)
    idx  = np.digitize(K.ravel(), bins) - 1
    p    = P2D.ravel()
    nb   = nbins
    good = (idx>=0) & (idx<nb) & np.isfinite(p)
    sums = np.bincount(idx[good], weights=p[good], minlength=nb)
    cnts = np.bincount(idx[good], minlength=nb)
    prof = np.full(nb, np.nan, float)
    nz   = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    kcent = 0.5*(bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def radial_average_map(Map2D: np.ndarray, dx: float, nbins: int, r_min: float, r_max_frac: float):
    ny, nx = Map2D.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(y, x) * dx
    rmax = R.max() * float(r_max_frac)
    bins = np.logspace(np.log10(max(r_min, 1e-8)), np.log10(rmax), nbins+1)
    idx  = np.digitize(R.ravel(), bins) - 1
    m    = Map2D.ravel()
    nb   = nbins
    good = (idx>=0) & (idx<nb) & np.isfinite(m)
    sums = np.bincount(idx[good], weights=m[good], minlength=nb)
    cnts = np.bincount(idx[good], minlength=nb)
    prof = np.full(nb, np.nan, float)
    nz   = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    rcent = 0.5*(bins[1:] + bins[:-1])
    sel   = nz & (rcent > r_min)
    return rcent[sel], prof[sel]

def E1D_from_map(Map2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    F  = fft2(Map2D)
    P2 = (F * np.conj(F)).real
    k1d, Pk = ring_average_2d(P2, dx, nbins, kmin, kmax_frac)
    E1D = 2*np.pi * k1d * Pk
    return k1d, E1D, P2

def directional_spectrum_and_correlation(f: np.ndarray):
    A  = np.cos(2.0 * f)
    B  = np.sin(2.0 * f)
    FA = fft2(A); FB = fft2(B)
    P2_dir = (FA*np.conj(FA)).real + (FB*np.conj(FB)).real
    norm   = np.sum(A*A + B*B)  # exact S(0)=1 normalization
    S_map  = np.fft.fftshift(ifft2(P2_dir).real) / norm
    return P2_dir, S_map, norm

# ──────────────────────────────────────────────────────────────────────
# RM correlation/structure function from the map (simulation-based)
# ──────────────────────────────────────────────────────────────────────

def Dphi_RM_from_map(Phi: np.ndarray, dx: float, nbins: int, r_min: float, r_max_frac: float):
    F  = fft2(Phi)
    P2 = (F * np.conj(F)).real
    C_map = np.fft.fftshift(ifft2(P2).real) / Phi.size  # Wiener–Khinchin
    R1d, C_R = radial_average_map(C_map, dx, nbins, r_min, r_max_frac)
    C0 = float(np.max(C_map))  # center pixel after fftshift
    D_RM = 2.0 * (C0 - C_R)
    return R1d, D_RM, C0, C_map

# ──────────────────────────────────────────────────────────────────────
# Analytic D_Φ(R) and D_φ(R,λ)
# ──────────────────────────────────────────────────────────────────────

def C_high_constant():
    """C_high = ∫_0^∞ q^{γ_h+1}(1−J0 q)dq for γ_h=−11/3.
       Write q^{−1−m} with m=5/3 → 2^(−m) Γ(−m/2)/Γ(1+m/2)."""
    m = 5.0/3.0
    return (2.0**(-m)) * gamma(-m/2.0) / gamma(1.0 + m/2.0)

import numpy as np
from scipy.special import j0  # Bessel J0

def Dphi_analytic_R_hankel(
    R1d, lam, k_break_cyc, sigma_phi2,
    alpha3D_low=+1.5, alpha3D_high=-(5.0/3.0),
    k_span_lo=1e-4, k_span_hi=1e+4, n_k=4096
):
    """
    Analytic D_phi(R,λ) from the broken power-law *2D* screen spectrum,
    computed via a *numerical Hankel integral* in *radian* wavenumbers:

        C_Φ(R) = 2π ∫_0^∞ k P_2D(k) J0(kR) dk,
        D_Φ(R) = 2[σ_Φ^2 - C_Φ(R)],
        D_φ(R,λ) = 1/2 [1 - exp(-2 λ^4 D_Φ(R))].

    P_2D(k) is piecewise with exponents γ_low=α_low-2 and γ_high=α_high-2.
    The overall amplitude is set so that ∫ 2π k P_2D(k) dk = σ_Φ^2.

    Parameters
    ----------
    R1d : array of separations (same units as your map's dx)
    lam : wavelength (m)
    k_break_cyc : break in *cycles/dx*  → internally converted to *radians/dx*
    sigma_phi2 : measured Var[Φ] on the map (same units)
    k_span_lo/hi : log-range around k_b to integrate over
    n_k : number of log-k points (4096 is usually fine)

    Returns
    -------
    Dphi_ang : array, analytic D_phi(R,λ)
    Dphi_RM  : array, analytic D_Φ(R)
    """
    # 1) exponents
    gl = alpha3D_low  - 2.0   # -1/2
    gh = alpha3D_high - 2.0   # -11/3

    # 2) k in *radians* per dx
    kb = 2.0 * np.pi * float(k_break_cyc)

    # 3) log-k grid spanning around kb
    kmin = kb * float(k_span_lo)
    kmax = kb * float(k_span_hi)
    k = np.exp(np.linspace(np.log(kmin), np.log(kmax), n_k))  # radians/dx

    # piecewise spectrum shape (unit amplitude)
    x = k / kb
    Pshape = np.where(k < kb, x**gl, x**gh)   # dimensionless shape

    # 4) normalize to measured σ_Φ^2 with *radian* convention
    # σ_Φ^2 = ∫_0^∞ 2π k P_2D(k) dk  =  2π ∫  k * (A * Pshape) dk
    # Integrate on log-k: ∫ f(k) dk = ∫ f(k) k d(ln k)  → use trapz over ln k.
    G0 = 2.0 * np.pi * k * Pshape
    I0 = np.trapz(G0, x=np.log(k))
    A  = sigma_phi2 / I0
    P2 = A * Pshape  # properly normalized P_2D(k)

    # 5) C_Φ(R) via Hankel integral for each R (vectorized)
    R = np.asarray(R1d, dtype=np.float64)
    # precompute kR matrix efficiently: (len(R) x len(k))
    kR = np.outer(R, k)  # shape (NR, Nk) — both in consistent units
    J = j0(kR)
    G = 2.0 * np.pi * k * P2  # integrand sans Bessel
    # Hankel integral per R over log-k
    C_R = J @ (G * np.gradient(np.log(k)))    # ≈ ∫ 2π k P2(k) J0(kR) dk
    # (Using gradient for non-uniform spacing is a bit more accurate than trapz in a loop)

    # 6) D_Φ(R) and D_φ(R,λ)
    Dphi_RM = 2.0 * (sigma_phi2 - C_R)
    # clip tiny negative roundoff (|·| ~ 1e-15) to zero
    Dphi_RM = np.maximum(Dphi_RM, 0.0)
    Dphi_ang = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * Dphi_RM))
    return Dphi_ang, Dphi_RM


# ──────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────

def _imshow(img, path, title, cmap="viridis"):
    plt.figure(figsize=(5.4,4.8))
    plt.imshow(img, origin="lower", cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path+".png", dpi=C.dpi); plt.savefig(path+".pdf")
    plt.close()

def _slope_guides(ax, k_break, yref, span, slopes_with_labels):
    kr = np.logspace(np.log10(k_break/span), np.log10(k_break*span), 200)
    for s, lab in slopes_with_labels:
        ax.loglog(kr, yref*(kr/k_break)**s, "--", lw=1.0, alpha=0.8, label=lab)

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    ne, bz, dx, dz = load_cube(C.h5_path)
    print(f"[load] ne,bz shapes: {ne.shape}, dx={dx}, dz={dz}")

    # Observer-plane fields
    Bz_proj, Phi = project_maps(ne, bz, dz, C.C_RM)
    _imshow(Bz_proj, os.path.join(C.outdir, "proj_Bz_map"), r"Projected $B_z$ (sum over $z$)")
    _imshow(Phi,     os.path.join(C.outdir, "proj_RM_map"), r"Faraday screen $\Phi$")

    # Quick E1D spectra (projection-slice guides: α_2D = α_3D − 1 → +1/2 and −8/3)
    kB  = C.k_break_cyc
    s2D = [(C.alpha3D_low-1.0,  r"$+1/2$ guide"),
           (C.alpha3D_high-1.0, r"$-8/3$ guide")]

    k_bz, E_bz, _ = E1D_from_map(Bz_proj, dx, C.nbins_k, C.kmin, C.kmax_frac)
    plt.figure(figsize=(6.6,5.0))
    plt.loglog(k_bz, E_bz, lw=1.8, label=r"$E_{1\rm D}[\sum B_z]$")
    if np.isfinite(np.interp(kB, k_bz, E_bz, left=np.nan, right=np.nan)):
        _slope_guides(plt.gca(), kB, np.interp(kB, k_bz, E_bz), C.guide_span, s2D)
    plt.axvline(kB, color='k', ls=':', lw=1, label=r"$k_b$")
    plt.xlabel(r"$k$"); plt.ylabel(r"$E_{1\rm D}(k)$"); plt.title("Projected $B_z$ spectrum (2D)")
    plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "proj_Bz_E1D.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "proj_Bz_E1D.pdf")); plt.close()

    k_phi, E_phi, _ = E1D_from_map(Phi, dx, C.nbins_k, C.kmin, C.kmax_frac)
    plt.figure(figsize=(6.6,5.0))
    plt.loglog(k_phi, E_phi, lw=1.8, label=r"$E_{1\rm D}[\Phi]$")
    if np.isfinite(np.interp(kB, k_phi, E_phi, left=np.nan, right=np.nan)):
        _slope_guides(plt.gca(), kB, np.interp(kB, k_phi, E_phi), C.guide_span, s2D)
    plt.axvline(kB, color='k', ls=':', lw=1, label=r"$k_b$")
    plt.xlabel(r"$k$"); plt.ylabel(r"$E_{1\rm D}(k)$"); plt.title("Faraday screen spectrum (2D)")
    plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "proj_RM_E1D.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "proj_RM_E1D.pdf")); plt.close()

    # Measured σ_Φ^2 for analytic normalization, and D_Φ(R) directly from Φ
    sigma_phi2 = float(np.var(Phi))
    Rrm, Dphi_RM_num, C0_rm, _ = Dphi_RM_from_map(Phi, dx, C.nbins_R, C.R_min, C.R_max_frac)

    # Directional spectra & correlations with full comparison
    for i, lam in enumerate(C.lambdas_m):
        f_map = (lam**2) * Phi

        # Directional estimator
        P2_dir, S_map, _ = directional_spectrum_and_correlation(f_map)
        kdir, Pdir_ring = ring_average_2d(P2_dir, dx, C.nbins_k, C.kmin, C.kmax_frac)
        plt.figure(figsize=(6.6,5.0))
        plt.loglog(kdir, Pdir_ring, lw=1.8, label=fr"$P_{{\rm dir}}(k)$, $\lambda={lam:.2f}$ m")
        plt.xlabel(r"$k$"); plt.ylabel(r"$|\widehat{\cos2f}|^2+|\widehat{\sin2f}|^2$")
        plt.title("Directional spectrum")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"Pdir_ring_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Pdir_ring_{i}.pdf")); plt.close()

        # Directional correlation and numerical angle structure function
        R1d, S_R = radial_average_map(S_map, dx, C.nbins_R, C.R_min, C.R_max_frac)
        Dphi_num = 0.5 * (1.0 - S_R)

        # RM-based (simulation) prediction on the same R grid (interpolate D_Φ^num)
        Dphi_RM_interp = np.interp(R1d, Rrm, Dphi_RM_num, left=np.nan, right=np.nan)
        Dphi_simRM = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * Dphi_RM_interp))

        # Analytic Dφ(R,λ) on the same R grid
        # Dphi_ana, _ = Dphi_analytic_R(
        #     R1d, lam, C.k_break_cyc, sigma_phi2,
        #     alpha3D_low=C.alpha3D_low, alpha3D_high=C.alpha3D_high
        # )
        Dphi_ana, _ = Dphi_analytic_R_hankel(
            R1d, lam, C.k_break_cyc, sigma_phi2,
            alpha3D_low=C.alpha3D_low, alpha3D_high=C.alpha3D_high,
            k_span_lo=1e-4, k_span_hi=1e4, n_k=4096
        )

        # Plot S(R) (optional)
        plt.figure(figsize=(6.6,5.0))
        plt.loglog(R1d, S_R, lw=1.8, label=fr"numerical $S(R)$, $\lambda={lam:.2f}$ m")
        plt.xlabel(r"$R$"); plt.ylabel(r"$S(R)$")
        plt.title("Directional correlation")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"S_of_R_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"S_of_R_{i}.pdf")); plt.close()

        # *** KEY PANEL: numerical vs RM-based (simulation) vs analytic ***
        plt.figure(figsize=(6.8,5.2))
        plt.loglog(R1d, Dphi_num,   lw=2.0,  label=r"numerical")
        # plt.loglog(R1d, Dphi_simRM, lw=0.2,  label=r"From RM map (simulation)")
        print(Dphi_ana)
        plt.loglog(R1d, Dphi_ana,   "--", lw=2.0, label=r"Analytic two-slope")
        # visual slope guide near small R (≈ 5/3)
        if len(R1d) > 8:
            Rref = R1d[len(R1d)//6]
            yref = np.interp(Rref, R1d, Dphi_num)
            Rg   = np.logspace(np.log10(Rref/6), np.log10(Rref*6), 200)
            # plt.loglog(Rg, yref*(Rg/Rref)**(5/3), ":", lw=1.0, alpha=0.8, label=r"$\propto R^{5/3}$")
        plt.xlabel(r"$R$"); plt.ylabel(r"$D_\varphi(R)$")
        plt.title(fr"$D_\varphi$: numerical vs simulation–RM vs analytic  ($\lambda={lam:.2f}$ m)")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False, loc="best")
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"Dphi_compare_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Dphi_compare_{i}.pdf")); plt.close()

    print(f"Saved → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()
