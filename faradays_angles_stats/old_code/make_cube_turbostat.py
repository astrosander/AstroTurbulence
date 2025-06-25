#!/usr/bin/env python3
"""
make_cube.py  –  build a Kolmogorov Bz + log-normal ne slab
             plus validate that the spectra follow P(k)∝k^(-11/3)
Written 2025-04-18
Requires: turbustat, numpy, h5py
"""
import numpy as np
import h5py
from turbustat.simulator import make_3dfield

# ---------- user knobs ------------------------------------------------
N         = 512          # grid points per axis
L_pc      = 100.0        # physical box size
B_rms     = 5.0          # μG (sets σ_RM)
ne_mean   = 0.10         # cm⁻³
ln_sigma  = 0.5          # dex dispersion of log-normal ne
seed_B, seed_ne = 42, 57
# ---------------------------------------------------------------------

dx = L_pc / N

print("Generating Kolmogorov B_z …")
Bz = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=seed_B)
Bz -= Bz.mean()
Bz *= B_rms / Bz.std()

print("Generating log-normal n_e …")
ln_ne = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=seed_ne)
ne    = np.exp(np.log(ne_mean) + ln_sigma * ln_ne/ln_ne.std())

# Save to HDF5
with h5py.File("cube.h5", "w") as f:
    f["Bz"], f["ne"] = Bz.astype("f4"), ne.astype("f4")
    f.attrs["dx_pc"] = dx
    f.attrs["comment"] = (
        f"Kolmogorov fBm cube (B_rms={B_rms} μG, <ne>={ne_mean} cm⁻³)"
    )
print(f"cube.h5 saved  (N={N},  dx={dx:.3f} pc)")

# ---------------------------------------------------------------------
# validate spectrum
# ---------------------------------------------------------------------
def shell_ps(field, dx):
    """
    Compute isotropic 3D power spectrum P(k) and corresponding k-shell centers.
    """
    ft  = np.fft.fftn(field)
    psd = np.abs(ft)**2

    # build k-space grid
    kax = np.fft.fftfreq(N, d=dx) * 2*np.pi
    kx, ky, kz = np.meshgrid(kax, kax, kax, indexing='ij')
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)

    # bin in integer shells (excluding k=0)
    kmax = np.max(kmag)
    kbins = np.arange(1, int(np.floor(kmax))+1)
    shell_inds = np.digitize(kmag.flat, kbins)

    Pk = np.array([psd.flat[shell_inds == i].mean() 
                   for i in range(1, len(kbins))])
    kcen = kbins[:-1] + 0.5

    return kcen, Pk

def fit_slope(k, Pk, kmin=None, kmax=None):
    """
    Fit log P = m log k + c, returning slope m.
    """
    mask = np.ones_like(k, bool)
    if kmin is not None: mask &= (k >= kmin)
    if kmax is not None: mask &= (k <= kmax)
    logk, logP = np.log10(k[mask]), np.log10(Pk[mask])
    m, c = np.polyfit(logk, logP, 1)
    return m

print("\nValidating power spectra…")

# Bz spectrum
k_bz, P_bz = shell_ps(Bz, dx)
slope_bz = fit_slope(k_bz, P_bz, kmin=5*(2*np.pi/L_pc), kmax=0.5*(2*np.pi/dx))
print(f"  Bz spectrum slope  ≃ {slope_bz:.3f}  (target = {-11/3:.3f})")

# ln(ne) spectrum
# We expect the log-density fluctuations to also follow Kolmogorov
k_ln, P_ln = shell_ps(ln_ne, dx)
slope_ln = fit_slope(k_ln, P_ln, kmin=5*(2*np.pi/L_pc), kmax=0.5*(2*np.pi/dx))
print(f"  ln(ne) spectrum slope ≃ {slope_ln:.3f}  (target = {-11/3:.3f})")
