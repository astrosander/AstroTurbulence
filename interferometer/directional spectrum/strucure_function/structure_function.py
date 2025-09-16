#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from numpy.fft import fft2, ifft2, fftfreq
from scipy.special import j0


@dataclass
class Config:
    # h5_path: str = "two_slope_2D_s4_r00.h5"
    h5_path: str = "mhd_fields.h5"

    outdir: str = "fig/two_slope_compare"

    alpha3D_low: float  = -1.54#3.0/2
    alpha3D_high: float = -4.39#-5.0/3.0
    k_break_cyc: float  = 0.08692406371 #0.06*2pi

    # alpha3D_low: float  = 1.77
    # alpha3D_high: float = -1.78
    # k_break_cyc: float  = 0.06875352187

    C_RM: float = 0.81

    lambdas_m: Tuple[float, ...] = (0.21,)

    nbins_R: int   = 2560
    R_min: float   = 1e-2
    R_max_frac: float = 1.0

    dpi: int = 160

C = Config()


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
    dz=1.0
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


def directional_S_and_Dphi_from_f(f_map: np.ndarray, dx: float, nbins_R: int, R_min: float, R_max_frac: float):
    A = np.cos(2.0 * f_map)
    B = np.sin(2.0 * f_map)
    FA = fft2(A); FB = fft2(B)
    P2_dir = (FA * np.conj(FA)).real + (FB * np.conj(FB)).real
    power0 = np.sum(A*A + B*B)
    S_map  = np.fft.fftshift(ifft2(P2_dir).real) / power0
    R, S_R = radial_average_map(S_map, dx, nbins_R, R_min, R_max_frac)
    Dphi_R = 0.5 * (1.0 - S_R)
    return R, S_R, Dphi_R

def Dphi_from_RM_map(Phi: np.ndarray, lam: float, dx: float, nbins_R: int, R_min: float, R_max_frac: float):
    F = fft2(Phi)
    P2 = (F * np.conj(F)).real
    C_map = np.fft.fftshift(ifft2(P2).real) / (Phi.size)
    R, C_R = radial_average_map(C_map, dx, nbins_R, R_min, R_max_frac)

    sigma2 = Phi.var(ddof=0)
    Dphi_R = (lam**4) * 2.0 * (sigma2 - C_R)
    return R, Dphi_R, sigma2


def Dphi_analytic_R_hankel(
    R1d, lam, k_break_cyc, sigma_phi2,
    alpha3D_low=+1.5, alpha3D_high=-(5.0/3.0),
    k_span_lo=1e-4, k_span_hi=1e+4, n_k=409600
):
    gl = alpha3D_low  - 2.0
    gh = alpha3D_high - 2.0
    kb = 2.0 * np.pi * float(k_break_cyc)
    kmin = kb * float(k_span_lo)
    kmax = kb * float(k_span_hi)
    k = np.exp(np.linspace(np.log(kmin), np.log(kmax), int(n_k)))

    x = k / kb
    Pshape = np.where(k < kb, x**gl, x**gh)

    G0 = 2.0 * np.pi * k * Pshape
    I0 = np.trapz(G0, x=np.log(k))
    A  = sigma_phi2 / I0
    P2 = A * Pshape

    R = np.asarray(R1d, dtype=np.float64)
    kR = np.outer(R, k)
    J = j0(kR)
    G = 2.0 * np.pi * k * P2
    dlnk = np.gradient(np.log(k))
    C_R = J @ (G * dlnk)

    Dphi_RM = 2.0 * (sigma_phi2 - C_R)
    Dphi_RM = np.maximum(Dphi_RM, 0.0)
    Dphi_ang = Dphi_RM#0.5 * (1.0 - np.exp(-2.0 * (lam**4) * Dphi_RM))
    return Dphi_ang, Dphi_RM



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
    plt.show()
    plt.close()


def Dphi_analytic_two_slope_numeric(
    R,
    dx, Lx,
    k_break_cyc,
    alpha_low=+1.5, alpha_high=-(5.0/3.0),
    s=4.0,
    sigma_phi_map=None,
    use_pixel_window=False,
):
    # print(s)
    Nx = int(round(Lx / dx))
    L_phys = Lx * dx
    kappa_min = 2.0 * np.pi / L_phys
    kappa_max = np.pi / dx
    kappa_b = 2.0 * np.pi * k_break_cyc
    print(kappa_b)
    nk = 409600
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), nk)
    x = kappas / kappa_b

    t = 1.0 / (1.0 + x**s)
    alpha = alpha_low * t + alpha_high * (1.0 - t)

    P2 = (x ** (alpha - 1.0))

    if use_pixel_window:
        W = (np.sinc(kappas * dx / (2.0*np.pi)))**2
        P2 *= W

    dlnk = np.diff(np.log(kappas)).mean()
    dkap = kappas * dlnk

    sigma_mod = (1.0/(2.0*np.pi)) * np.sum(P2 * kappas * dlnk)
    if sigma_phi_map is None or sigma_mod == 0.0:
        C = 1.0
    else:
        C = float(sigma_phi_map) / float(sigma_mod)
    P2 *= C

    C0 = (1.0/(2.0*np.pi)) * np.sum(P2 * kappas * dlnk)
    C_R = []
    for r in np.atleast_1d(R):
        J = j0(kappas * r)
        C_R.append((1.0/(2.0*np.pi)) * np.sum(P2 * J * kappas * dlnk))
    C_R = np.array(C_R)

    D_phi_no_lambda = 2.0 * (C0 - C_R)
    return D_phi_no_lambda


def Dphi_analytic_from_grid(nx, ny, dx, k_break_cyc, alpha_low, alpha_high, s, sigma_phi2, lam,
                            nbins_R, R_min, R_max_frac):
    k_b = 2.0*np.pi*float(k_break_cyc)
    kx = 2.0*np.pi*fftfreq(nx, d=dx)
    ky = 2.0*np.pi*fftfreq(ny, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KX, KY)
    x = np.maximum(K, 1e-20) / k_b
    t = 1.0/(1.0 + x**s)
    alpha = alpha_low*t + alpha_high*(1.0 - t)

    P2_shape = (x**(alpha - 1.0))
    P2_shape[0,0] = 0.0

    # normalize so C(0)=sigma_phi2
    C0_model = (np.fft.ifft2(P2_shape).real)[0,0] / (nx*ny)
    A = sigma_phi2 / C0_model if C0_model != 0 else 1.0
    P2_model = A * P2_shape

    C_map = np.fft.fftshift(np.fft.ifft2(P2_model).real) / (nx*ny)
    R1d, C_R = radial_average_map(C_map, dx, nbins_R, R_min, R_max_frac)
    Dphi = (2.0*(sigma_phi2 - C_R))#0.5 * (1.0 - np.exp(-2.0*(lam**4) * (2.0*(sigma_phi2 - C_R))))
    return R1d, Dphi

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    ne, bz, dx, dz = load_cube(C.h5_path)
    _, Phi = project_maps(ne, bz, dz, C.C_RM)
    ny, nx = Phi.shape

    # Read synthesis parameters from the file (falls back to config if absent)
    with h5py.File(C.h5_path, "r") as f:
        attrs = dict(f.attrs)
    k_break_cyc = float(attrs.get("k_break", C.k_break_cyc))
    s_sharp     = float(attrs.get("s",        4.0))
    
    # k_break_cyc = 
    print(k_break_cyc)
    print(s_sharp)

    sigma_phi2 = Phi.var(ddof=0)

    for i, lam in enumerate(C.lambdas_m):
        # --- numerical (directional) ---
        R_num, _, Dphi_num = directional_S_and_Dphi_from_f(
            lam**2 * Phi, dx, C.nbins_R, max(C.R_min, 1.0), C.R_max_frac
        )

        # --- analytic on the *same FFT grid* ---
        R_ana, Dphi_ana = Dphi_analytic_from_grid(
            nx=nx, ny=ny, dx=dx,
            k_break_cyc=k_break_cyc,
            alpha_low=C.alpha3D_low, alpha_high=C.alpha3D_high,
            s=s_sharp,
            sigma_phi2=sigma_phi2,
            lam=lam,
            nbins_R=C.nbins_R, R_min=max(C.R_min, 1.0), R_max_frac=C.R_max_frac
        )

        # --- plot ---
        plt.figure(figsize=(7,5))
        # plt.loglog(R_num, Dphi_num, '-',  label='Numerical (synthetic)')
        plt.loglog(R_ana, Dphi_ana, '-', label="from directional spectrum")
        # plt.xlabel(r"$R$")
        plt.ylabel(r"$D_\Phi(R)$")
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        out = os.path.join(C.outdir, f"Dphi_compare_{i}")
        plt.savefig(out + ".png", dpi=C.dpi)
        plt.savefig(out + ".pdf")
        plt.show()
        plt.close()

    print(f"Saved → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()