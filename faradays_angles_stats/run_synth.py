#!/usr/bin/env python3
"""
run_synth.py   –   synthetic Faraday-screen observation
2025-04-18  (final)

Outputs into ./results/  per wavelength λ:
  QU_λ.fits      – Q,U maps
  Dphi_λ.npy     – angle SF           (new measure)
  DphiLn_λ.npy   – log-measure        (LP16-free slope)
  DP_λ.npy       – PSA (LP16)         (λ⁻⁴ scaling done in plotter)
  Ddphi_λ.npy    – SF of ∂φ/∂λ
Global:
  RM.fits ,  sigmaP.txt
"""
import numpy as np, h5py
from pathlib import Path
from astropy.io import fits

# ---------- analysis knobs -------------------------------------------
depth_frac = 0.125     # fraction of cube used as Faraday screen
chi_unsat  = 0.25      # 2 λ₀² σ_RM  ≈ 0.25 → safely unsaturated
nwaves     = 20        # number of λ points (log-spaced)
# ---------------------------------------------------------------------

out = Path("results"); out.mkdir(exist_ok=True)

with h5py.File("cube.h5") as f:
    Bz, ne = f["Bz"][...], f["ne"][...]
    dx     = float(f.attrs["dx_pc"])

Nz   = Bz.shape[2];  zmax = int(depth_frac * Nz)
K    = 0.81                               # rad m⁻² pc⁻¹ cm³ μG⁻¹
RM   = (K * np.cumsum(ne[:, :, :zmax] * Bz[:, :, :zmax], 2) * dx)[:, :, -1]
sigma_RM = RM.std()
fits.PrimaryHDU(RM.astype("f4")).writeto(out/"RM.fits", overwrite=True)
print(f"σ_RM = {sigma_RM:.2f}  rad m⁻²   (depth={depth_frac} L_box)")

lam0 = np.sqrt(chi_unsat / (2*sigma_RM))          # first (unsat.) λ  [m]
lam_grid = np.logspace(np.log10(lam0),
                       np.log10(12*lam0),
                       nwaves)

# ---------- helper: azimuthal average (skip R=0) ----------------------
def radial_average(arr):
    N = arr.shape[0]
    y,x = np.indices(arr.shape)
    r = np.hypot(x-N//2, y-N//2).astype(int)
    rmax = r.max()
    sum_r   = np.bincount(r.ravel(), arr.ravel(), minlength=rmax+1)
    count_r = np.bincount(r.ravel(),             minlength=rmax+1)
    prof = sum_r / np.maximum(count_r, 1)
    radii = np.arange(1, rmax+1)                 # drop bin 0
    return radii*dx, prof[1:]

# ---------- loop over λ ----------------------------------------------
sigma_tab = []

for lam in lam_grid:
    lam_cm = lam*100
    alpha  = 2 * lam**2 * RM
    P      = np.exp(1j*alpha)                         # |P| = 1 background
    Q, U   = P.real.astype("f4"), P.imag.astype("f4")

    fits.HDUList([fits.PrimaryHDU(Q),
                  fits.ImageHDU(U, name="U")]).writeto(
                      out/f"QU_{lam_cm:.3f}cm.fits", overwrite=True)

    # ------ angle SF  Dphi -------------------------------------------
    phi = 0.5*np.arctan2(U, Q)
    var_phi = phi.var()
    Cphi = np.fft.ifft2(np.fft.fft2(phi) *
                        np.conj(np.fft.fft2(phi))).real / phi.size
    R, C1 = radial_average(np.fft.fftshift(Cphi))
    Dphi = 2*np.clip(var_phi - C1, 0.0, None)
    np.save(out/f"Dphi_{lam_cm:.3f}cm.npy", np.vstack((R, Dphi)))

    # ------ log-measure  DphiLn  -------------------------------------
    S = np.exp(-2*lam**4 * (var_phi - C1))
    Dphi_ln = -np.log(S) / (2*lam**4)
    np.save(out/f"DphiLn_{lam_cm:.3f}cm.npy", np.vstack((R, Dphi_ln)))

    # ------ LP16 PSA  D_P  -------------------------------------------
    CPP  = np.fft.ifft2(np.fft.fft2(P) *
                        np.conj(np.fft.fft2(P))).real / P.size
    R, Cpp1 = radial_average(np.fft.fftshift(CPP))
    DP = 2*np.clip(CPP[0,0] - Cpp1, 0.0, None)
    np.save(out/f"DP_{lam_cm:.3f}cm.npy", np.vstack((R, DP)))

    # ------ SF of ∂φ/∂λ  ---------------------------------------------
    dphi_dlam = 4*lam * RM
    var_d = dphi_dlam.var()
    Cd = np.fft.ifft2(np.fft.fft2(dphi_dlam) *
                      np.conj(np.fft.fft2(dphi_dlam))).real / dphi_dlam.size
    R, Cd1 = radial_average(np.fft.fftshift(Cd))
    Sdphi = 2*np.clip(var_d - Cd1, 0.0, None)
    np.save(out/f"Ddphi_{lam_cm:.3f}cm.npy", np.vstack((R, Sdphi)))

    # ------ PVA table -------------------------------------------------
    sigma_tab.append((lam_cm, (np.abs(P)**2).mean()))

    print(f"λ = {lam_cm:7.2f} cm   finished")

np.savetxt(out/"sigmaP.txt", np.array(sigma_tab), header="λ_cm   σ_P²")
print("All wavelengths complete.")
