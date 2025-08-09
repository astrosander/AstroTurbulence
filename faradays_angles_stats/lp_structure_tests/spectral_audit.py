#!/usr/bin/env python3
"""
spectral_audit.py
=================

One-stop “spectral audit”:

• 3-D isotropic power spectrum  P(k)  of   B_z   *and*   n_e
• 2-D transverse structure function D_Bz(R)  (central z-slice)
• twin log–log panels
      left  : Athena snapshot
      right : synthetic power-law cube

Tune the filenames/parameters at the top and run:

    python spectral_audit.py
"""

from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift, rfftn, irfftn

# ────────── paths & constants ────────────────────────────────────────
CUBE_ATHENA    = "ms01ma08.mhd_w.00300.vtk.h5"
CUBE_SYNTHETIC = "synthetic_tuned.h5"

NBINS_K  = 50     # bins in |k| for P(k)
NBINS_R  = 80     # bins in R   for D(R)
Z_SAMPLE = "mid"  # central slice for 2-D SF
DX_FALLBACK = 1.0
# --------------------------------------------------------------------


def _dx(coord):
    uniq = np.unique(coord)
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    return float(np.median(d)) if d.size else DX_FALLBACK


# ────────── 3-D power spectrum (isotropic) ───────────────────────────
def isotropic_pk(cube, dx, nbins=50):
    N = cube.shape[0]            # assume cubic
    vol = (N*dx)**3
    Fk  = fftn(cube) / vol
    Pk3 = np.abs(Fk)**2

    # radial grid in k-space
    kfreq = fftshift(np.fft.fftfreq(N, d=dx))
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    kmag_flat = kmag.ravel()
    Pk_flat   = fftshift(Pk3).ravel()

    kmin, kmax = kmag_flat[1], kmag_flat.max()
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
    sumP, _ = np.histogram(kmag_flat, bins=bins, weights=Pk_flat)
    counts, _ = np.histogram(kmag_flat, bins=bins)
    P_iso = sumP / np.maximum(counts, 1)
    k_cent = 0.5*(bins[1:]+bins[:-1])
    return k_cent[counts>0], P_iso[counts>0]


# ────────── 2-D transverse structure function ───────────────────────
def sf_2d(field2d, dx, nbins=80):
    f = field2d - field2d.mean()
    ac = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    D  = 2*f.var() - 2*ac
    D[D < 0] = 0

    ny, nx = f.shape
    y = (np.arange(ny)-ny//2)[:,None]
    x = (np.arange(nx)-nx//2)[None,:]
    R = np.hypot(x,y)*dx

    rmin, rmax = dx*1e-3, R.max()*0.4
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    sumD, _ = np.histogram(R, bins=bins, weights=D)
    cnts, _ = np.histogram(R, bins=bins)
    D_R = sumD / np.maximum(cnts,1)
    R_cent = 0.5*(bins[1:]+bins[:-1])
    mask = cnts>0
    return R_cent[mask], D_R[mask]


# ────────── helper to pull cubes and compute everything ──────────────
def analyse_cube(path):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:]
        bz = f["k_mag_field"][:]
        dx = _dx(f["x_coor"][:,0,0]) if "x_coor" in f else DX_FALLBACK

    # 3-D spectra
    k_ne, P_ne = isotropic_pk(ne, dx, NBINS_K)
    k_bz, P_bz = isotropic_pk(bz, dx, NBINS_K)

    # 2-D structure function (central slice)
    zidx = bz.shape[2]//2 if Z_SAMPLE=="mid" else int(Z_SAMPLE)
    R, D_bz = sf_2d(bz[:,:,zidx], dx, NBINS_R)

    return k_ne, P_ne, k_bz, P_bz, R, D_bz, dx


# ────────── run analysis on both cubes ───────────────────────────────
k_ne_A, P_ne_A, k_bz_A, P_bz_A, R_A, D_A, dxA = analyse_cube(CUBE_ATHENA)
k_ne_S, P_ne_S, k_bz_S, P_bz_S, R_S, D_S, dxS = analyse_cube(CUBE_SYNTHETIC)

# Kolmogorov reference slopes
refPk_A = P_bz_A[1]*(k_bz_A/k_bz_A[1])**(-11/3)
refSF_A = D_A[5]*(R_A/R_A[5])**(2/3)

refPk_S = P_bz_S[1]*(k_bz_S/k_bz_S[1])**(-11/3)
refSF_S = D_S[5]*(R_S/R_S[5])**(2/3)

# ────────── plot … two columns (Athena | Synthetic) ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey='row')

# Athena spectra
axes[0,0].loglog(k_bz_A, P_bz_A, label="B$_z$")
axes[0,0].loglog(k_ne_A, P_ne_A, label="n$_e$")
axes[0,0].loglog(k_bz_A, refPk_A, "--", label=r"$k^{-11/3}$")
axes[0,0].set(title="Athena ‒ power spectrum",
              ylabel="P(k)", xlabel="k")
axes[0,0].legend(frameon=False, fontsize=7)

# Athena SF
axes[1,0].loglog(R_A, D_A, label="B$_z$ slice")
axes[1,0].loglog(R_A, refSF_A, "--", label=r"$R^{2/3}$")
axes[1,0].set(title="Athena ‒ structure function",
              ylabel=r"$D_{B_z}(R)$", xlabel="R")
axes[1,0].legend(frameon=False, fontsize=7)

# Synthetic spectra
axes[0,1].loglog(k_bz_S, P_bz_S, color='tab:green', label="B$_z$")
axes[0,1].loglog(k_ne_S, P_ne_S, color='tab:orange', label="n$_e$")
axes[0,1].loglog(k_bz_S, refPk_S, "--", color='k')
axes[0,1].set(title="Synthetic ‒ power spectrum", xlabel="k")
axes[0,1].legend(frameon=False, fontsize=7)

# Synthetic SF
axes[1,1].loglog(R_S, D_S, color='tab:green')
axes[1,1].loglog(R_S, refSF_S, "--", color='k')
axes[1,1].set(title="Synthetic ‒ structure function", xlabel="R")

fig.suptitle("Spectral audit: P(k)  &  D$_{B_z}$(R)  (β ≈ 11/3 target)")
fig.tight_layout()
plt.show()
