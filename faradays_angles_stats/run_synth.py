#!/usr/bin/env python
"""
run_synth.py  –  synthetic Faraday-screen observation
----------------------------------------------------
• reads cube.h5
• integrates a *thin* screen (depth_frac)
• chooses λ-grid from σ_RM  so the first λ is safely unsaturated
• writes QU and Dφ files under ./results/
"""
import numpy as np, h5py
from pathlib import Path
from astropy.io import fits

# ---------------- analysis knobs -----------------
depth_frac = 0.125     # screen thickness = 1/8 of the box
chi_unsat  = 0.10      # 2 λ² σ_RM  for the first wavelength
nwaves     = 20
# -------------------------------------------------

out = Path("results")
out.mkdir(exist_ok=True)

with h5py.File("cube.h5", "r") as f:
    Bz = f["Bz"][...]
    ne = f["ne"][...]
    dx = float(f.attrs["dx_pc"])

Nz = Bz.shape[2]
zmax = int(depth_frac * Nz)

K = 0.81                                    # rad m⁻² pc⁻¹ cm³ μG⁻¹
Phi = K * np.cumsum(ne[:, :, :zmax] * Bz[:, :, :zmax], axis=2) * dx
RM  = Phi[:, :, -1]

fits.PrimaryHDU(RM.astype("f4")).writeto(out/"RM.fits", overwrite=True)
sigma_RM = RM.std()
print(f"σ_RM = {sigma_RM:.1f}  rad m⁻²  (depth = {depth_frac} L_box)")

# ---------- wavelength grid ----------
lam0 = np.sqrt(chi_unsat / (2*sigma_RM))          # metres
lam_grid = np.logspace(np.log10(lam0),
                       np.log10(12*lam0), nwaves) # covers unsat→sat

# ---------- helpers ----------
def radial_average(arr):
    N = arr.shape[0]
    y,x = np.indices(arr.shape) - N//2
    R   = np.hypot(x,y).astype(int)
    prof= np.bincount(R.ravel(), arr.ravel()) / np.bincount(R.ravel())
    return np.arange(prof.size), prof

# ---------- main loop ----------
for lam in lam_grid:
    lam_cm = lam*100
    alpha  = 2*lam**2 * RM
    P      = np.exp(1j*alpha)              # unit intrinsic polarisation
    Q, U   = P.real.astype("f4"), P.imag.astype("f4")

    fits.HDUList([fits.PrimaryHDU(Q),
                  fits.ImageHDU(U, name="U")]
                ).writeto(out/f"QU_{lam_cm:.3f}cm.fits", overwrite=True)

    phi = 0.5*np.arctan2(U, Q)
    fft = np.fft.fft2(phi)
    Corr= np.fft.ifft2(fft*fft.conj()).real / phi.size
    Corr= np.fft.fftshift(Corr)
    Rpx, C1D = radial_average(Corr)
    Dphi = 2*np.maximum(0, phi.var() - C1D)       # enforce ≥0
    np.save(out/f"Dphi_{lam_cm:.3f}cm.npy",
            np.vstack((Rpx*dx, Dphi)))
    print(f"λ = {lam_cm:7.2f} cm  done")

print("All λ finished.")
