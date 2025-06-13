#!/usr/bin/env python
"""
Synthetic Faraday-screen observation.

Steps
1. read cube.h5
2. pick LOS = z (axis=2) and integrate Phi(X,Y) = 0.81 ∑ ne Bz dz
3. build P(λ) = P0 exp(2i λ² Φ)
4. turn into Q,U and angle φ
5. compute structure function Dφ(R) via FFT
"""

import numpy as np, h5py, astropy.io.fits as fits, scipy.fft as fft, os, sys
from pathlib import Path

def radial_average(arr):
    """Azimuthal average of a 2-D array around its centre."""
    N  = arr.shape[0]
    y, x = np.indices((N,N)) - N//2
    R    = np.hypot(x, y).astype(int)
    rmax = R.max()
    prof = np.bincount(R.ravel(), arr.ravel()) / np.bincount(R.ravel())
    return np.arange(rmax+1), prof[:rmax+1]

out = Path("results"); out.mkdir(exist_ok=True)

with h5py.File("cube.h5","r") as f:
    Bz = f["Bz"][...]                # shape (N,N,N)
    ne = f["ne"][...]
    dx = f.attrs["dx_pc"]

print("Computing RM map …")
K      = 0.81                        # rad m⁻² pc⁻¹ cm³ μG⁻¹
Phi    = K * np.cumsum(ne*Bz, axis=2) * dx   # cumulative RM
RM     = Phi[:,:,-1]                 # (N,N)

sigma_Phi = RM.std()
print(f"σ_RM = {sigma_Phi:.1f}  rad m⁻²")

# wavelength grid: from 1.5 cm to 5× saturation
lam0   = 0.015           # 1.5 cm
lam_sat= (1/(2*sigma_Phi**2))**0.25
lam    = np.logspace(np.log10(lam0), np.log10(5*lam_sat), 20)

# save RM to FITS (optional)
fits.PrimaryHDU(RM.astype("f4")).writeto(out/"RM.fits", overwrite=True)

print("Loop over  λ …")
pix   = RM.shape[0]
kx    = fft.fftfreq(pix)
ky    = kx
kx2   = kx[:,None]**2
ky2   = ky[None,:]**2
k2    = kx2+ky2
k2[0,0] = 1              # avoid /0

for lam_i in lam:
    P0       = 1.0                      # uniform background
    alpha    = 2.0 * (lam_i**2) * RM
    P        = P0 * np.exp(1j*alpha)
    Q, U     = P.real, P.imag

    # write Stokes FITS
    hdu = fits.HDUList([fits.PrimaryHDU(Q.astype("f4")),
                        fits.ImageHDU(U.astype("f4"), name="U")])
    hdu.writeto(out/f"QU_{lam_i*100:.3f}cm.fits", overwrite=True)

    # angle map
    phi      = 0.5 * np.arctan2(U, Q)          # rad

    # structure function via FFT trick
    cimg     = np.cos(phi) + 1j*np.sin(phi)
    c_k      = fft.fft2(cimg, workers=-1)
    C        = fft.ifft2(c_k * np.conj(c_k), workers=-1).real
    S        = np.fft.fftshift(C) / (pix**2)    # centred & normalised

    R, S1D   = radial_average(S)
    Dphi     = 0.5 * (1 - S1D)                  # eq. (5)

    np.save(out/f"Dphi_{lam_i*100:.3f}cm.npy", np.vstack((R*dx, Dphi)))
    print(f"  λ = {lam_i*100:.2f} cm  done.")

print("All λ finished.")
