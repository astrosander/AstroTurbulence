#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synchrotron_rgb_noargs.py
Create colorized 2D polarized synchrotron emission maps (no CLI args).

Color encoding (HSV):
  Hue        = polarization angle χ = 0.5 * arg(P)
  Saturation = polarization fraction |P| / I
  Value      = total intensity I (percentile-stretched)
"""

import numpy as np
import h5py
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ------------------ USER SETTINGS (edit here) ------------------ #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
# theta_list = [30, 45, 60]     # viewing angles (deg)
theta_list = []
for i in range(0, 361, 5):
    theta_list.append(i)

N       = 256                 # image size (pixels)
nsamp   = 192                 # samples along LOS
p       = 3.0                 # CR electron index
include_faraday = False       # internal Faraday rotation toggle
lambda_m = 0.21               # wavelength [m]
ne0      = 1.0                # uniform ne (relative)
rm_coeff = 0.812              # relative RM coefficient

add_uniform_B0 = False        # add mean B0 along +z?
B0_amp_factor  = 0.5          # B0 amplitude in units of Brms

outprefix = "synch_RGB"       # output filename prefix
dpi       = 200               # PNG DPI
clip_frac = 0.995             # intensity percentile for display
# --------------------------------------------------------------- #

def load_cube(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2,1,0).astype(np.float32)
        By = f["j_mag_field"][:].transpose(2,1,0).astype(np.float32)
        Bz = f["k_mag_field"][:].transpose(2,1,0).astype(np.float32)
        x_edges = f["x_coor"][0,0,:]
        L = float(x_edges[-1] - x_edges[0])
    nx = Bx.shape[0]
    Brms = float(np.sqrt(np.mean(Bx**2 + By**2 + Bz**2)))
    print(f"[load] cube {nx}^3; L={L:g}; Brms={Brms:g}")
    return (Bx,By,Bz), L, nx, Brms

def geom(theta_deg):
    th = np.radians(theta_deg).astype(np.float32)
    n  = np.array([np.sin(th),0.0,np.cos(th)], np.float32)   # LOS
    e1 = np.array([np.cos(th),0.0,-np.sin(th)], np.float32)  # sky x
    e2 = np.array([0.0,1.0,0.0], np.float32)                 # sky y
    return n,e1,e2

def make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat):
    i_idx, j_idx = np.indices((N,N), dtype=np.float32)
    x0 = ((i_idx-N/2)/N)[...,None]*L*e1_hat + ((j_idx-N/2)/N)[...,None]*L*e2_hat  # (N,N,3)
    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32)                          # (nsamp,)
    ds = float(s[1]-s[0])
    los_offsets = (s[:,None]*n_hat[None,:])[None,None,:,:]                        # (1,1,nsamp,3)
    pos_phys = x0[...,None,:] + los_offsets                                       # (N,N,nsamp,3)
    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = (pos_frac * (nx - 1)).reshape(-1,3).T.astype(np.float32)           # (3, N*N*nsamp)
    return pos_idx, ds

def interp(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N,N,nsamp)

def build_I_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds,
              add_uniform_B0=False, B0_amp=0.0,
              include_faraday=False, ne0=1.0, lambda_m=0.21, rm_coeff=0.812):
    """Return total intensity I and complex polarized map P=Q+iU."""
    Bx_l, By_l, Bz_l = (interp(Bx,pos_idx,N,nsamp),
                         interp(By,pos_idx,N,nsamp),
                         interp(Bz,pos_idx,N,nsamp))
    B = np.stack((Bx_l,By_l,Bz_l), axis=-1)                                    # (N,N,nsamp,3)

    if add_uniform_B0 and (B0_amp > 0):
        B += np.array([0,0,B0_amp], dtype=B.dtype)[None,None,None,:]

    # Geometry and emissivity
    B_par  = np.tensordot(B, n_hat, axes=([-1],[0]))                           # (N,N,nsamp)
    B_perp = B - B_par[...,None]*n_hat                                          # (N,N,nsamp,3)
    Bp_mag = np.linalg.norm(B_perp, axis=-1)                                    # (N,N,nsamp)
    eps    = Bp_mag ** ((p+1)/2.0)                                              # ~ synch emissivity
    psi    = 0.5*np.arctan2(B_perp[...,1], B_perp[...,0])                       # polarization angle

    # Optional internal Faraday rotation
    if include_faraday:
        Phi = rm_coeff * ne0 * np.cumsum(B_par, axis=-1) * ds                   # relative RM
        phase = 2.0*(psi + (lambda_m**2)*Phi)
    else:
        phase = 2.0*psi

    # LOS integrals
    P = np.sum(eps * np.exp(1j*phase), axis=-1) * ds                            # (N,N)
    I = np.sum(eps, axis=-1) * ds                                               # (N,N)
    return I, P

def percentile_stretch(img, p_lo=1.0, p_hi=99.5, eps=1e-12):
    finite = np.isfinite(img)
    lo, hi = np.percentile(img[finite], [p_lo, p_hi])
    if hi <= lo: hi = lo + eps
    out = np.clip((img - lo)/(hi - hi + (hi - lo)), 0, 1)  # avoid divide-by-zero
    # Oops, fix typo: correct normalization
    out = np.clip((img - lo) / (hi - lo + eps), 0, 1)
    return out

def polarization_rgb(I, P, clip_frac=0.995):
    """Map (I, P) -> RGB via HSV:
       hue = χ/(π) wrapped to [0,1], saturation = |P|/I, value = stretched I."""
    chi = 0.5 * np.angle(P)                       # [-π/2, π/2]
    hue = (chi / np.pi) % 1.0                     # [0,1)

    I_safe = np.maximum(I, 1e-20)
    pfrac = np.abs(P) / I_safe
    pfrac = np.clip(pfrac, 0.0, 1.0)

    V = percentile_stretch(I, p_lo=1.0, p_hi=100*clip_frac)
    H = hue
    S = pfrac

    HSV = np.stack([H, S, V], axis=-1).astype(np.float32)
    RGB = hsv_to_rgb(HSV)
    return RGB

# ------------------------------ MAIN ------------------------------ #
if __name__ == "__main__":
    (Bx,By,Bz), L, nx, Brms = load_cube(filename)
    B0_amp = B0_amp_factor * Brms if add_uniform_B0 else 0.0
    print(f"[cfg] N={N} nsamp={nsamp} p={p} Faraday={include_faraday} λ={lambda_m} m "
          f" addB0={add_uniform_B0} (B0_amp={B0_amp:.3g})")

    for theta_deg in theta_list:
        print(f"\n=== θ = {theta_deg:.1f}° ===")
        n_hat, e1_hat, e2_hat = geom(theta_deg)
        pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)

        I, P = build_I_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds,
                         add_uniform_B0=add_uniform_B0, B0_amp=B0_amp,
                         include_faraday=include_faraday, ne0=ne0,
                         lambda_m=lambda_m, rm_coeff=rm_coeff)

        RGB = polarization_rgb(I, P, clip_frac=clip_frac)

        fname = f"{outprefix}_theta{int(round(theta_deg))}.png"
        fname = f"img\\{int(round(theta_deg))}.png"

        plt.figure(figsize=(6,6))
        plt.imshow(RGB, origin="lower", interpolation="nearest")
        plt.title(f"Polarized Synchrotron (θ={theta_deg:.1f}°)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[save] → {fname}")
