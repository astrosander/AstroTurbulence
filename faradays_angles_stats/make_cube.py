#!/usr/bin/env python3
"""
make_cube.py  (pure-NumPy version)
----------------------------------
* divergence-free 3-D magnetic field with Kolmogorov spectrum
* log-normal electron density from an independent Kolmogorov scalar field
* writes cube.h5  (datasets Bx,By,Bz,ne ; attribute dx_pc)
* prints shell-averaged spectral slopes for sanity

Requires only:  numpy >= 1.17 , h5py
"""

import numpy as np, h5py, sys

# ------------- user parameters ---------------------------------------
N        = 512            # grid points per side   (256 if RAM is tight)
L_pc     = 100.0          # physical box size      [pc]
B_rms    = 5.0            # desired rms of |B|     [μG]
ne_mean  = 0.10           # mean electron density  [cm⁻³]
ln_sigma = 0.5            # log-normal σ (dex)
seed     = 42
# ---------------------------------------------------------------------

rng = np.random.default_rng(seed)
dx  = L_pc / N
kgrid = np.fft.fftfreq(N, d=dx/L_pc)  # physical units: 1/pc
kx, ky, kz = np.meshgrid(kgrid, kgrid, kgrid, indexing="ij")
k2 = kx**2 + ky**2 + kz**2
k_mag = np.sqrt(k2, dtype=float); k_mag[0,0,0] = 1     # avoid /0

# ---------- helper: divergence-free random field ---------------------
def kolmogorov_B():
    """
    Return 3 divergence-free components with P(k) ∝ k^{-11/3}.
    Construction: generate complex vector field in k-space, then project
    out the longitudinal component  k · B_k  to enforce ∇·B = 0.
    """
    amp  = k_mag**(-11/6)                # sqrt of k^{-11/3}
    phase = rng.normal(size=k_mag.shape) + 1j*rng.normal(size=k_mag.shape)
    Vk = amp * phase                     # random complex scalar
    
    # random complex vector
    Bxk = rng.normal(size=k_mag.shape) + 1j*rng.normal(size=k_mag.shape)
    Byk = rng.normal(size=k_mag.shape) + 1j*rng.normal(size=k_mag.shape)
    Bzk = rng.normal(size=k_mag.shape) + 1j*rng.normal(size=k_mag.shape)
    
    # project onto plane perpendicular to k  :  B⊥ = B - (k·B) k / k²
    dot = kx*Bxk + ky*Byk + kz*Bzk
    Bxk -= dot * kx / k_mag**2
    Byk -= dot * ky / k_mag**2
    Bzk -= dot * kz / k_mag**2
    
    # scale each component by scalar amplitude Vk
    Bxk *= Vk;  Byk *= Vk;  Bzk *= Vk
    return (np.fft.ifftn(Bxk).real,
            np.fft.ifftn(Byk).real,
            np.fft.ifftn(Bzk).real)

print("Generating divergence-free Kolmogorov B-field …")
Bx, By, Bz = kolmogorov_B()
# set global rms to B_rms  (true rms of magnitude)
norm = np.sqrt((Bx**2+By**2+Bz**2).mean())
Bx *= B_rms / norm;  By *= B_rms / norm;  Bz *= B_rms / norm

print("Generating log-normal n_e …")
amp  = k_mag**(-11/6)
phase= rng.normal(size=k_mag.shape) + 1j*rng.normal(size=k_mag.shape)
ln_ne_k = amp * phase
ln_ne   = np.fft.ifftn(ln_ne_k).real
ne      = np.exp(np.log(ne_mean) + ln_sigma * ln_ne/ln_ne.std())

# -------------- optional: spectrum validation ------------------------
def shell_ps(field):
    F = np.fft.fftn(field)
    P = np.abs(F)**2
    # shell‐average in 3-D
    kbins = np.sqrt((kx**2+ky**2+kz**2)).flatten()
    Pbins = P.flatten()
    k_indices = np.rint(kbins/dk).astype(int)
    kmax = k_indices.max()
    pshell = np.bincount(k_indices, Pbins, minlength=kmax+1)
    counts = np.bincount(k_indices, minlength=kmax+1)
    shell = pshell / np.maximum(counts,1)
    k_vals= (np.arange(kmax+1)+0.5)*dk
    return k_vals[1:], shell[1:]        # skip k=0

def fit_slope(k, P, kmin, kmax):
    mask = (k>kmin) & (k<kmax) & np.isfinite(P) & (P>0)
    m,_ = np.polyfit(np.log10(k[mask]), np.log10(P[mask]), 1)
    return m

dk = kgrid[1] - kgrid[0]
print("Validating power spectra …")
for fld,name in zip((Bz, ln_ne), ("Bz", "ln(ne)")):
    k,P = shell_ps(fld)
    slope = fit_slope(k,P, kmin=5*(2*np.pi/L_pc), kmax=0.5*(2*np.pi/dx))
    print(f"  {name} spectrum slope ≃ {slope:.3f}  (target = {-11/3:.3f})")

# -------------- write HDF5 -------------------------------------------
with h5py.File("cube.h5","w") as f:
    f["Bx"], f["By"], f["Bz"] = Bx.astype("f4"), By.astype("f4"), Bz.astype("f4")
    f["ne"] = ne.astype("f4")
    f.attrs["dx_pc"] = dx
    f.attrs["comment"] = "Pure-NumPy Kolmogorov cube (divergence-free B)"
print("cube.h5 written.")
