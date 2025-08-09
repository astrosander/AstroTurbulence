# import numpy as np, yt, scipy.ndimage as ndi


# # ds        = yt.load(r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\synthetic_tuned.h5")

# import h5py

# filename = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

# with h5py.File(filename, "r") as f:
#     Bx = f["i_mag_field"][:].transpose(2, 1, 0)  # now in (x, y, z) order
#     By = f["j_mag_field"][:].transpose(2, 1, 0)
#     Bz = f["k_mag_field"][:].transpose(2, 1, 0)

#     x_edges = f["x_coor"][0, 0, :]  # axis 2 varies
#     domain_width_x = x_edges[-1] - x_edges[0]
#     print("domain_width[0] =", domain_width_x)
#     # print(f["x_coor"][:, 0, 0])

# # cg = ds.covering_grid(level=0, left_edge=[0,0,0], dims=ds.domain_dimensions)


# # Bx,By,Bz  = [ds.covering_grid(level=0, left_edge=[0,0,0],
# #                               dims=ds.domain_dimensions)[f"B{c}"] for c in "xyz"]

# def interpolate(field, pos_idx):
#     return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap")


# theta = np.radians(45)
# n_hat = np.array([np.sin(theta),0,np.cos(theta)])
# e1    = np.array([np.cos(theta),0,-np.sin(theta)])
# e2    = np.array([0,1,0])

# N   = 512;  L = domain_width_x
# P   = np.zeros((N,N), dtype=np.complex128)
# s   = np.linspace(-L/2, L/2, 256)

# for i in range(N):
#     for j in range(N):
#         x0  = ((i-N/2)/N)*L*e1 + ((j-N/2)/N)*L*e2
#         pos = x0[None,:] + s[:,None]*n_hat
#         pos_idx = (pos / L % 1.0) * (Bx.shape[0] - 1)  # shape (256, 3)
#         pos_idx = pos_idx.T  # shape (3, 256)
#         Bx_l = interpolate(Bx, pos_idx)
#         By_l = interpolate(By, pos_idx)
#         Bz_l = interpolate(Bz, pos_idx)
#         B    = np.vstack((Bx_l,By_l,Bz_l)).T
#         B_par = B @ n_hat
#         B_perp = B - B_par[:,None]*n_hat
#         eps   = np.linalg.norm(B_perp,axis=1)**2   # p≈3 → (p+1)/2=2
#         psi   = 0.5*np.arctan2(B_perp[:,1], B_perp[:,0])
#         P[i,j]= np.sum(eps*np.exp(2j*psi))*(s[1]-s[0])


# F  = np.fft.fft2(P)                # complex
# C  = np.fft.ifft2(F * F.conj())    # complex correlation map
# C  = np.fft.fftshift(C) / P.size   # centre zero-lag
# C_re, C_im = C.real, C.imag



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropy diagnostic: 45° LOS, complex correlation, polar quadrupole plot
-------------------------------------------------------------------------

Requires: numpy, scipy, h5py, matplotlib, tqdm
           (optional: numba for ~2× speed-up; see notes)

Author: <your-name>, 2025-07-31
"""

import numpy as np
import scipy.ndimage as ndi
import h5py, matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------
# 0. PARAMETERS ---------------------------------------------------------
# ----------------------------------------------------------------------
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
theta_deg  = 45                # LOS tilt angle (deg) to the +z axis
N          = 512//1               # pixels per side in the POS map
nsamp      = 256//1               # integration samples along each ray
p          = 3.0               # electron spectral index (≈3 → ε∝B⊥²)
R0_frac    = 0.25              # pick radius R0 = R0_frac * (N/2) for polar plot
savefig    = True              # write PNGs to disk
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 1. LOAD THE CUBE ------------------------------------------------------
# ----------------------------------------------------------------------
with h5py.File(filename, "r") as f:
    Bx = f["i_mag_field"][:].transpose(2, 1, 0)  # → (x,y,z)
    By = f["j_mag_field"][:].transpose(2, 1, 0)
    Bz = f["k_mag_field"][:].transpose(2, 1, 0)
    # coordinates are cell centres; assume cubic grid
    x_edges       = f["x_coor"][0, 0, :]
    domain_width  = float(x_edges[-1] - x_edges[0])      # physical size
nx = Bx.shape[0]         # cells per axis (assume cube)
print(f"Loaded cube {nx}³; physical box size L = {domain_width:g}.")

# ----------------------------------------------------------------------
# 2. GEOMETRY VECTORS ---------------------------------------------------
# ----------------------------------------------------------------------
theta  = np.radians(theta_deg)
n_hat  = np.array([np.sin(theta), 0.0, np.cos(theta)])        # LOS
e1_hat = np.array([np.cos(theta), 0.0, -np.sin(theta)])       # POS x̂
e2_hat = np.array([0.0, 1.0,  0.0])                           # POS ŷ  (kept fixed)

# Pre-compute grids -----------------------------------------------------
L  = domain_width
s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32)          # LOS samples
ds = s[1] - s[0]

# Pixel centre coordinates in POS plane (physical units)
i_idx, j_idx = np.indices((N, N))
x0_phys = ((i_idx - N/2) / N)[..., None] * L * e1_hat + \
          ((j_idx - N/2) / N)[..., None] * L * e2_hat          # (N,N,3)

# Expand to every sample along the ray: shape (N,N,nsamp,3)
pos_phys = x0_phys[..., None, :] + s[:, None] * n_hat                  # broadcast on s
# Map to fractional index within [0, nx)
pos_idx = (pos_phys / L) % 1.0
pos_idx = pos_idx * (nx - 1)                                  # (N,N,nsamp,3)
# Re-order to (3, N*N*nsamp) for ndi.map_coordinates
pos_idx_reshaped = pos_idx.reshape(-1, 3).T.astype(np.float32)

# ----------------------------------------------------------------------
# 3. INTERPOLATE B-FIELD ALONG RAYS ------------------------------------
# ----------------------------------------------------------------------
def interp_field(field):
    return ndi.map_coordinates(field,
                               pos_idx_reshaped,
                               order=1, mode="wrap").reshape(N, N, nsamp)

print("Interpolating Bx, By, Bz ...")
Bx_l, By_l, Bz_l = [interp_field(f) for f in tqdm((Bx, By, Bz))]

# ----------------------------------------------------------------------
# 4. BUILD SYNCHROTRON POLARISATION MAP P(X) ----------------------------
# ----------------------------------------------------------------------
print("Computing P(X) ...")
B_vec   = np.stack((Bx_l, By_l, Bz_l), axis=-1)               # (N,N,nsamp,3)
B_par   = np.tensordot(B_vec, n_hat, axes=([-1],[0]))         # (N,N,nsamp)
B_perp  = B_vec - B_par[..., None] * n_hat                    # (N,N,nsamp,3)

B_perp_mag = np.linalg.norm(B_perp, axis=-1)                  # |B⊥|

# Synchrotron emissivity ε ∝ |B⊥|^{(p+1)/2}
eps  = B_perp_mag ** ((p + 1) / 2.0)

# Intrinsic polarisation angle ψ = ½ arctan2(B⊥_y, B⊥_x)
psi  = 0.5 * np.arctan2(B_perp[..., 1], B_perp[..., 0])       # radians

# For a pure Faraday screen we ignore rotation: Φ=0
P = np.sum(eps * np.exp(2j * psi), axis=-1) * ds             # (N,N)
print("P(X) map done.")

# ----------------------------------------------------------------------
# 5. COMPLEX CORRELATION C(R) ------------------------------------------
# ----------------------------------------------------------------------
print("FFT → correlation ...")
F  = np.fft.fft2(P)
C  = np.fft.ifft2(F * F.conj())
C  = np.fft.fftshift(C) / P.size         # normalised, zero-lag centred
C_re, C_im = C.real, C.imag

# ----------------------------------------------------------------------
# 6. AZIMUTHAL BINNING AT R=R0 -----------------------------------------
# ----------------------------------------------------------------------
# Pixel-space radius grid
x = (np.arange(N) - N/2)
y = (np.arange(N) - N/2)
X, Y = np.meshgrid(x, y, indexing='ij')
R_pix  = np.sqrt(X**2 + Y**2)
phi    = np.arctan2(Y, X)                # [-π, π]

R0_pix = R0_frac * (N/2.0)
dR     = 0.5                             # thickness of ring in pixels
mask   = np.abs(R_pix - R0_pix) < dR

phi_bins = np.linspace(-np.pi, np.pi, 181)   # 2° bins
phi_cent = 0.5 * (phi_bins[:-1] + phi_bins[1:])

def az_average(arr):
    vals, _ = np.histogram(phi[mask], bins=phi_bins, weights=arr[mask])
    counts, _ = np.histogram(phi[mask], bins=phi_bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(vals / counts)

C_re_phi = az_average(C_re)
C_im_phi = az_average(C_im)

# ----------------------------------------------------------------------
# 7. PLOTS --------------------------------------------------------------
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection='polar')
ax.plot(phi_cent, C_re_phi, label='Re{C}', lw=1.8)
ax.plot(phi_cent, C_im_phi, label='Im{C}', lw=1.2, ls='--')
ax.set_theta_zero_location('E')          # 0 at +x
ax.set_theta_direction(-1)               # counter-clockwise positive
ax.set_title(rf"$\theta={theta_deg}^\circ$,  $R_0={R0_frac}\,R_{{\rm max}}$",
             pad=20)
ax.legend(loc='upper right')
if savefig:
    plt.savefig("polar_quadrupole.png", dpi=180, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 8. OPTIONAL: NUMBA ACCELERATION --------------------------------------
# ----------------------------------------------------------------------
"""
If the interpolation step is the bottleneck, moving it into a numba
jit-compiled function (or PyTorch CUDA tensor interpolation) can give ~2×.
Left out here for clarity, but easy to add:

from numba import njit, prange

@njit(parallel=True)
def map_coordinates_numba(field, pos):
    ...  # your own trilinear code

"""
print("Done.")
