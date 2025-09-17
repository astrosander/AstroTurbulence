#!/usr/bin/env python3
"""
D_phi(R, λ) from the 3D ENERGY SPECTRUM (isotropic) of q = n_e * B_z.
- Loads ../mhd_fields.h5 (only 'gas_density', 'k_mag_field'; dx=dz=1).
- Measures E1D(k) from the 3D FFT of q.
- Uses isotropy: P3D(k) = E1D(k) / (4π k^2).
- Builds a 2D model power on the map FFT grid via kz=0 slice.
- Normalizes to σ_Φ^2 from the actual Φ map, then computes D_Φ and D_φ.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from numpy.fft import fftn, ifft2, fft2, fftfreq

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    h5_path: str = r"D:\Рабочая папка\GitHub\AstroTurbulence\interferometer\Transition smoothness demo\h5\transition_smoothness\two_slope_2D_s3_r00.h5"#"../two_slope_2D_s3_r00.h5"
    outdir:  str = "fig/dphi_from_energy_spectrum"
    C_RM: float = 0.81
    lambdas_m: Tuple[float, ...] = (0.21,)

    nbins_R: int   = 300
    R_min_pix: float = 1.0
    R_max_frac: float = 0.45
    dpi: int = 170

C = Config()

# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────
def load_cube(path: str):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = 1.0
    dz = 1.0
    return ne, bz, dx, dz

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def radial_average_map(Map2D: np.ndarray, nbins: int, r_min_pix: float, r_max_frac: float):
    ny, nx = Map2D.shape
    yy = (np.arange(ny) - ny//2)[:, None]
    xx = (np.arange(nx) - nx//2)[None, :]
    R_pix = np.hypot(yy, xx)
    rmax = R_pix.max() * float(r_max_frac)

    eps = 1e-8
    bins = np.logspace(np.log10(max(r_min_pix, eps)), np.log10(rmax), nbins+1)
    idx = np.digitize(R_pix.ravel(), bins) - 1

    vals = Map2D.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(vals)
    sums = np.bincount(idx[good], weights=vals[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)

    prof = np.full(nbins, np.nan)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]

    rcent_pix = 0.5*(bins[1:] + bins[:-1])
    sel = nz & (rcent_pix >= r_min_pix)
    return rcent_pix[sel], prof[sel]

def shell_average_3d(P3: np.ndarray, dx=1.0, dy=1.0, dz=1.0, nbins=80):
    """Isotropic shell average of 3D power -> returns k_shell, <P3D>."""
    nz, ny, nx = P3.shape
    kx = 2.0*np.pi*fftfreq(nx, d=dx)
    ky = 2.0*np.pi*fftfreq(ny, d=dy)
    kz = 2.0*np.pi*fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    kmin = np.max([np.min(K[K>0]), 1e-8])
    kmax = np.max(K)
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)

    idx = np.digitize(K.ravel(), bins) - 1
    y = P3.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(y)
    sums = np.bincount(idx[good], weights=y[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)

    prof = np.full(nbins, np.nan)
    nzm  = cnts > 0
    prof[nzm] = sums[nzm] / cnts[nzm]
    kcent = 0.5*(bins[1:] + bins[:-1])
    return kcent[nzm], prof[nzm]

def safe_loglog(x, y, label=None, lw=2.0):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(m): return
    yclip = y[m]
    if not np.any(yclip > 0):
        yclip = np.full_like(yclip, 1e-20)
    plt.loglog(x[m], np.maximum(yclip, 1e-300), lw=lw, label=label)

# ─────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────
def build_phi(ne: np.ndarray, bz: np.ndarray, dz: float, C_RM: float) -> np.ndarray:
    """Φ(x,y) = C_RM ∑_z n_e B_z dz."""
    return C_RM * (ne * bz).sum(axis=2) * dz

def p2_model_from_energy_spectrum(ne: np.ndarray, bz: np.ndarray):
    """
    Measure the 3D ENERGY spectrum of q = n_e B_z:
      E1D(k) = 4π k^2 <P3D>_shell
    and reconstruct a 2D model power on the map grid via the kz=0 slice:
      P2D(K⊥) = P3D(K⊥) = E1D(K⊥)/(4π K⊥^2).
    """
    q = ne * bz
    nz, ny, nx = q.shape

    Qk = fftn(q)
    P3 = (Qk * np.conj(Qk)).real

    # Shell-averaged <P3D>(k)
    k_shell, P3_shell = shell_average_3d(P3, dx=1.0, dy=1.0, dz=1.0, nbins=100)

    # ENERGY spectrum E1D(k) = 4π k^2 P3D(k)
    E1D = 4.0 * np.pi * k_shell**2 * P3_shell

    # Build P2D model on the SAME k-grid as the 2D map via kz=0
    kx = 2.0*np.pi*fftfreq(nx, d=1.0)
    ky = 2.0*np.pi*fftfreq(ny, d=1.0)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    Kperp = np.sqrt(KX**2 + KY**2)

    # Interpolate E1D(k) on log–log, then P3D = E1D/(4π k^2)
    lk = np.log(np.maximum(k_shell, 1e-12))
    lE = np.log(np.maximum(E1D, 1e-300))
    def P3_of_k(kabs):
        ksafe = np.maximum(kabs, k_shell[0])
        # E1D interp in log–log
        E_interp = np.exp(np.interp(np.log(ksafe), lk, lE, left=lE[0], right=lE[-1]))
        return E_interp / (4.0 * np.pi * np.maximum(ksafe**2, 1e-24))

    P2_model = P3_of_k(Kperp)
    P2_model[0,0] = 0.0
    return P2_model, (k_shell, E1D)

def dphi_from_p2_model(P2_model: np.ndarray, sigma_phi2_target: float, lam: float,
                        nbins_R: int, R_min_pix: float, R_max_frac: float):
    """
    From model 2D power → correlation → D_Φ → D_φ.
    Normalize so C(0)=σ_Φ^2 (measured from actual Φ map).
    """
    ny, nx = P2_model.shape
    C_map_unnorm = ifft2(P2_model).real
    C0_model = C_map_unnorm[0,0] / (nx * ny)
    A = (sigma_phi2_target / C0_model) if (np.isfinite(C0_model) and C0_model != 0.0) else 1.0
    C_map = np.fft.fftshift(A * C_map_unnorm / (nx * ny))

    R_pix, C_R = radial_average_map(C_map, nbins_R, R_min_pix, R_max_frac)
    sigma2 = float(sigma_phi2_target)
    D_Phi = np.maximum(2.0 * (sigma2 - C_R), 0.0)
    Dphi  = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * D_Phi))
    return R_pix, Dphi, (R_pix, D_Phi, C_R)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    ne, bz, dx, dz = load_cube(C.h5_path)
    print(f"[load] ne,bz shapes: {ne.shape}, dx={dx}, dz={dz}")

    # Build Φ map and its variance for amplitude normalization
    Phi = build_phi(ne, bz, dz, C.C_RM)
    sigma_phi2 = Phi.var(ddof=0)
    print(f"σ_Φ^2 = {sigma_phi2:.6e}")

    # Build P2D model from the 3D ENERGY spectrum
    P2_model, (k_shell, E1D) = p2_model_from_energy_spectrum(ne, bz)

    # Plot E1D(k) for sanity
    plt.figure(figsize=(6.8, 4.8))
    safe_loglog(k_shell, E1D, label=r"$E_{1\mathrm{D}}(k)$ from $q=n_e B_z$")
    
    k0 = k_shell[len(k_shell)//3]   # pick roughly 1/3 along the spectrum
    E0 = E1D[len(E1D)//3]

    ref_line = E0 * (k_shell / k0)**(1)
    plt.loglog(k_shell, ref_line, 'k--', label=r"$k^{-5/3}$")

    kmin, kmax = 1e-1, 1e1   # set your interval
    mask = (k_shell > kmin) & (k_shell < kmax) & (E1D > 0) & np.isfinite(E1D)
    m, b = np.polyfit(np.log(k_shell[mask]), np.log(E1D[mask]), 1)
    plt.loglog(k_shell[mask], np.exp(b) * k_shell[mask]**m,
               label=fr"fit [{kmin:g},{kmax:g}]: $k^{{{m:.2f}}}$")


    kmin, kmax = 0e-1, 1e-1   # set your interval
    mask = (k_shell > kmin) & (k_shell < kmax) & (E1D > 0) & np.isfinite(E1D)
    m, b = np.polyfit(np.log(k_shell[mask]), np.log(E1D[mask]), 1)
    plt.loglog(k_shell[mask], np.exp(b) * k_shell[mask]**m,
               label=fr"fit [{kmin:g},{kmax:g}]: $k^{{{m:.2f}}}$")


    plt.xlabel(r"$k$")
    plt.ylabel(r"$E_{1\mathrm{D}}(k)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(C.outdir, "E1D_q.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "E1D_q.pdf"))
    plt.close()

    # D_phi from energy-spectrum-based P2D model
    for i, lam in enumerate(C.lambdas_m):
        R, Dphi, (R_D, D_Phi, C_R) = dphi_from_p2_model(
            P2_model=P2_model,
            sigma_phi2_target=sigma_phi2,
            lam=lam,
            nbins_R=C.nbins_R,
            R_min_pix=C.R_min_pix,
            R_max_frac=C.R_max_frac
        )
        print(f"[λ={lam:.2f} m]  max D_Φ={np.nanmax(D_Phi):.6e}, max D_φ={np.nanmax(Dphi):.6e}")

        # Plot D_phi
        plt.figure(figsize=(7.0, 5.2))
        safe_loglog(R, Dphi, label=fr"$\lambda={lam:.2f}\,$m")
        plt.xlabel(r"$R$")
        plt.ylabel(r"$D_\varphi(R)$")

            # --- Add tangent line with slope 5/3 through (R=0, Dphi=0) ---
        # Pick a reference point to draw the line (e.g. smallest R value)
        R_ref = R[0]  # avoid zero if log scale
        y_ref = (5/3) * R_ref  # slope * x
        tangent_line = (Dphi[6])/((5/3) * R[6])*(5/3) * R  # y = (5/3)*x

        plt.loglog(R, tangent_line, "--", color="k", 
                   label="$5/3$")

        # plt.title("Angle structure function from 3D ENERGY spectrum (isotropic)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        out = os.path.join(C.outdir, f"Dphi_energy_{i}")
        plt.savefig(out + ".png", dpi=C.dpi)
        plt.savefig(out + ".pdf")
        plt.close()

    print(f"Saved → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()
