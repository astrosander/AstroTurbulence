#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from numpy.fft import fft2, ifft2, fftfreq, fftn, ifft2, ifftshift


@dataclass
class Config:
    h5_path: str = "../mhd_fields.h5"
    outdir:  str = "fig/head_on_from_spectrum"
    C_RM: float = 0.81
    lambdas_m: Tuple[float, ...] = (0.21,)
    nbins_R: int = 256
    R_min_pix: float = 1.0
    R_max_frac: float = 0.45
    dpi: int = 160

C = Config()


def _axis_spacing_from_h5(arr, fallback: float) -> float:
    try:
        u = np.unique(arr.ravel())
        dif = np.diff(np.sort(u))
        dif = dif[dif > 0]
        if dif.size:
            return float(np.median(dif))
    except Exception:
        pass
    return float(fallback)

def load_cube(path: str):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
        if "x_coor" in f:
            dx = _axis_spacing_from_h5(f["x_coor"][:,0,0], 1.0)
        else:
            dx = 1.0
        if "z_coor" in f:
            dz = _axis_spacing_from_h5(f["z_coor"][0,0,:], 1.0)
        else:
            dz = 1.0
    dx=1.0
    dz=1.0
    return ne, bz, dx, dz


def radial_average_map(Map2D: np.ndarray, dx: float, nbins: int, r_min_pix: float, r_max_frac: float):
    ny, nx = Map2D.shape
    yy = (np.arange(ny) - ny//2)[:, None]
    xx = (np.arange(nx) - nx//2)[None, :]
    R_pix = np.hypot(yy, xx)
    rmax = R_pix.max() * float(r_max_frac)
    bins = np.logspace(np.log10(max(r_min_pix, 1e-8)), np.log10(rmax), nbins+1)
    idx = np.digitize(R_pix.ravel(), bins) - 1
    m = Map2D.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(m)
    sums = np.bincount(idx[good], weights=m[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)
    prof = np.full(nbins, np.nan)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    rcent_pix = 0.5*(bins[1:] + bins[:-1])
    sel = nz & (rcent_pix >= r_min_pix)
    return rcent_pix[sel]*dx, prof[sel]

def shell_average_3d(P3D: np.ndarray, dx: float, dz: float, nbins: int = 60):
    """Numerical S(R), Dφ(R) from the angle field f = λ^2 Φ."""
    nz, ny, nx = P3D.shape
    kx = 2.0*np.pi*fftfreq(nx, d=dx)
    ky = 2.0*np.pi*fftfreq(ny, d=dx)
    kz = 2.0*np.pi*fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    kmin = max(np.min(Kmag[Kmag>0]), 1e-12)
    kmax = np.max(Kmag)
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
    idx = np.digitize(Kmag.ravel(), bins) - 1
    y = P3D.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(y)
    sums = np.bincount(idx[good], weights=y[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)
    prof = np.full(nbins, np.nan)
    nzb  = cnts > 0
    prof[nzb] = sums[nzb] / cnts[nzb]
    kcent = 0.5*(bins[1:] + bins[:-1])
    return kcent[nzb], prof[nzb]


def build_phi_map(ne: np.ndarray, bz: np.ndarray, dz: float, C_RM: float) -> np.ndarray:
    return C_RM * (ne * bz).sum(axis=2) * dz

def directional_S_and_Dphi(Phi: np.ndarray, lam: float, dx: float, nbins_R: int, R_min_pix: float, R_max_frac: float):
    f = (lam*lam) * Phi
    A = np.cos(2.0 * f)
    B = np.sin(2.0 * f)
    FA = fft2(A); FB = fft2(B)
    Pdir = (FA*np.conj(FA)).real + (FB*np.conj(FB)).real
    power0 = np.sum(A*A + B*B)
    S_map = np.fft.fftshift(ifft2(Pdir).real) / power0
    R, S_R = radial_average_map(S_map, dx, nbins_R, R_min_pix, R_max_frac)
    Dphi = 0.5 * (1.0 - S_R)
    return R, Dphi, S_R

def p2d_from_measured_p3d_kz0(ne: np.ndarray, bz: np.ndarray, dx: float, dz: float):
    """
    'Head-on' projection: from measured 3D spectrum of q = n_e B_z to a 2D model spectrum
    by taking the kz=0 slice (correct for summing over the full periodic depth).
    Returns the 2D model power on the SAME k-grid as the map.
    """
    q = ne * bz
    nz, ny, nx = q.shape
    Qk = fftn(q)
    P3 = (Qk * np.conj(Qk)).real
    k1d, P3_shell = shell_average_3d(P3, dx=dx, dz=dz, nbins=80)
    kx = 2.0*np.pi*fftfreq(nx, d=dx)
    ky = 2.0*np.pi*fftfreq(ny, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    from numpy import log, exp
    kref = k1d
    yref = np.maximum(P3_shell, 1e-300)
    lk, ly = log(kref), log(yref)
    def p3_of_k(kabs):
        ksafe = np.maximum(kabs, kref[0])
        return np.exp(np.interp(np.log(ksafe), lk, ly, left=ly[0], right=ly[-1]))
    Ktot = np.sqrt(KX*KX + KY*KY)
    P2_model = p3_of_k(Ktot)
    P2_model[0,0] = 0.0
    return P2_model, (k1d, P3_shell)

def dphi_from_p2d_model(P2_model: np.ndarray, sigma_phi2_target: float, lam: float,
                        nbins_R: int, dx: float, R_min_pix: float, R_max_frac: float):
    ny, nx = P2_model.shape
    C_map_unnorm = ifft2(P2_model).real
    C0_model = C_map_unnorm[0,0] / (nx*ny)
    if not np.isfinite(C0_model) or C0_model == 0.0:
        A = 1.0
    else:
        A = float(sigma_phi2_target) / float(C0_model)
    C_map = A * C_map_unnorm / (nx*ny)
    C_map = np.fft.fftshift(C_map)
    R, C_R = radial_average_map(C_map, dx, nbins_R, R_min_pix, R_max_frac)
    sigma2 = float(sigma_phi2_target)
    D_Phi = 2.0 * (sigma2 - C_R)
    Dphi  = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * D_Phi))
    return R, Dphi, (R, D_Phi, C_R)


def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)
    ne, bz, dx, dz = load_cube(C.h5_path)
    nz, ny, nx = ne.shape
    print(f"[load] ne,bz: {ne.shape}, dx={dx:g}, dz={dz:g}")
    Phi = build_phi_map(ne, bz, dz, C.C_RM)
    sigma_phi2 = Phi.var(ddof=0)
    P2_model, spec3d_diag = p2d_from_measured_p3d_kz0(ne, bz, dx, dz)

    for i, lam in enumerate(C.lambdas_m):
        R_num, Dphi_num, _ = directional_S_and_Dphi(# Athena
            Phi, lam, dx, nbins_R=C.nbins_R, R_min_pix=C.R_min_pix, R_max_frac=C.R_max_frac
        )
        R_ana, Dphi_ana, _ = dphi_from_p2d_model(
            P2_model=P2_model,
            sigma_phi2_target=sigma_phi2,
            lam=lam,
            nbins_R=C.nbins_R,
            dx=dx,
            R_min_pix=C.R_min_pix,
            R_max_frac=C.R_max_frac
        )
        plt.figure(figsize=(7.0,5.2))
        plt.loglog(R_num, Dphi_num, '-',  lw=2.0, label="Numerical (directional)")
        plt.loglog(R_ana, Dphi_ana, '--', lw=2.0, label="From 3D spectrum (kz=0)")
        plt.xlabel(r"$R$")
        plt.ylabel(r"$D_\varphi(R)$")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        out = os.path.join(C.outdir, f"Dphi_head_on_{i}")
        plt.savefig(out + ".png", dpi=C.dpi)
        plt.savefig(out + ".pdf")
        plt.close()

    k1d, P3_shell = spec3d_diag
    plt.figure(figsize=(6.8,4.8))
    plt.loglog(k1d, P3_shell, lw=1.8)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$P_{3D}(k)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.outdir, "P3D_q_shell.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "P3D_q_shell.pdf"))
    plt.close()

    print(f"Saved → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()
