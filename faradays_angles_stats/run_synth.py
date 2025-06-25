#!/usr/bin/env python3
"""
run_synth.py  –  Faraday-screen synthetic observation
Outputs (per λ) Dphi, DphiLn, DP, Ddphi + Q/U FITS + RM map + σ_P² table
"""
import numpy as np, h5py
from pathlib import Path
from astropy.io import fits

# ---------------- analysis knobs ------------------------------------
depth_frac = 0.125        # screen depth relative to full cube
chi_unsat  = 0.50         # 2 λ₀² σ_RM  → λ₀ safely unsaturated
nwaves     = 20           # number of λ values (log-spaced)
# --------------------------------------------------------------------

out = Path("results"); out.mkdir(exist_ok=True)

with h5py.File("cube.h5") as f:
    Bz, ne = f["Bz"][...], f["ne"][...]
    dx     = float(f.attrs["dx_pc"])

# Nz, zmax = Bz.shape[2], int(depth_frac*Nz)
Nz   = Bz.shape[2];  zmax = int(depth_frac * Nz)

K  = 0.81
RM = (K*np.cumsum(ne[:,:,:zmax]*Bz[:,:,:zmax],2)*dx)[:,:,-1]
sigma_RM = RM.std()
fits.PrimaryHDU(RM.astype("f4")).writeto(out/"RM.fits", overwrite=True)
print(f"σ_RM = {sigma_RM:.2f} rad m⁻²   (depth={depth_frac} L_box)")

lam0 = np.sqrt(chi_unsat/(2*sigma_RM))          # first λ   [m]
lam_grid = np.logspace(np.log10(lam0),
                       np.log10(12*lam0),
                       nwaves)

# -------- half-pixel radial bins helper -----------------------------
def radial_average(arr):
    N = arr.shape[0]
    y,x = np.indices(arr.shape)
    r = np.hypot(x-N//2, y-N//2)
    bins = np.arange(-.5, r.max()+1.5, 1.0)     # [-0.5,0.5),[0.5,1.5)…
    which = np.digitize(r.ravel(), bins)
    prof  = np.bincount(which, arr.ravel()) / np.bincount(which)
    radii = 0.5 + np.arange(len(prof))          # bin centres
    return radii[1:]*dx, prof[1:]               # drop R<0.5 bin

# ---------------- wavelength loop -----------------------------------
sigma_tab = []

for lam in lam_grid:
    lam_cm = lam*100
    alpha  = 2*lam**2 * RM
    P      = np.exp(1j*alpha)
    Q, U   = P.real.astype("f4"), P.imag.astype("f4")

    fits.HDUList([fits.PrimaryHDU(Q),
                  fits.ImageHDU(U,name="U")]).writeto(
        out/f"QU_{lam_cm:.3f}cm.fits", overwrite=True)

    # angle SF
    phi = 0.5*np.arctan2(U,Q)
    var_phi = phi.var()
    Cphi = np.fft.ifft2(np.fft.fft2(phi) *
                        np.conj(np.fft.fft2(phi))).real / phi.size
    R, C1 = radial_average(np.fft.fftshift(Cphi))
    diff = var_phi - C1
    diff[diff < 0] = np.nan
    Dphi = 2*diff
    np.save(out/f"Dphi_{lam_cm:.3f}cm.npy", np.vstack((R,Dphi)))

    # log-measure
    S = np.exp(-2*lam**4 * diff)
    Dphi_ln = -np.log(S) / (2*lam**4)
    np.save(out/f"DphiLn_{lam_cm:.3f}cm.npy", np.vstack((R,Dphi_ln)))

    # PSA
    CPP = np.fft.ifft2(np.fft.fft2(P) *
                       np.conj(np.fft.fft2(P))).real / P.size
    R, Cpp1 = radial_average(np.fft.fftshift(CPP))
    diffP = CPP[0,0] - Cpp1
    diffP[diffP < 0] = np.nan
    DP = 2*diffP
    np.save(out/f"DP_{lam_cm:.3f}cm.npy", np.vstack((R,DP)))

    # SF of ∂φ/∂λ
    dphi_dlam = 4*lam * RM
    var_d = dphi_dlam.var()
    Cd = np.fft.ifft2(np.fft.fft2(dphi_dlam) *
                      np.conj(np.fft.fft2(dphi_dlam))).real / dphi_dlam.size
    R, Cd1 = radial_average(np.fft.fftshift(Cd))
    diffd = var_d - Cd1
    diffd[diffd < 0] = np.nan
    Sdphi = 2*diffd
    np.save(out/f"Ddphi_{lam_cm:.3f}cm.npy", np.vstack((R,Sdphi)))

    sigma_tab.append((lam_cm, (np.abs(P)**2).mean()))
    print(f"λ = {lam_cm:7.2f} cm   done")

np.savetxt(out/"sigmaP.txt", np.array(sigma_tab), header="λ_cm   σ_P²")
print("All wavelengths finished.")
