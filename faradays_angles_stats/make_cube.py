#!/usr/bin/env python
"""
Generate an isothermal MHD-style cube:

- Bx, By, Bz: divergence–free Gaussian random field with P(k) ∝ k^-11/3
- ne        : log-normal with the same spectrum (σ_logne ≈ 0.5 dex)

The cube is saved as cube.h5 with a few metadata attributes.
"""
import numpy as np, h5py, scipy.fft as fft

N      = 256          # grid size along one axis (≤512 keeps RAM < 6 GB)
L_pc   = 100.0        # physical box length in parsec
dx     = L_pc / N     # cell size

rng    = np.random.default_rng(42)

def kolmogorov_spectrum(k):
    """P(k) ∝ k^-11/3 but with P(0)=0, avoids divide-by-zero."""
    Pk = np.zeros_like(k)
    mask = k > 0
    # use out / where so NumPy never evaluates 0**negative
    np.power(k, -11/3, out=Pk, where=mask)
    return Pk


def random_field():
    """Return one divergence-free real field with Kolmogorov spectrum."""
    kx = fft.fftfreq(N, d=1./N)
    ky = kx
    kz = kx
    k2 = np.add.outer(kx**2, ky**2)[:, :, None] + kz**2          # (N,N,N)
    k  = np.sqrt(k2, dtype=float)

    # build three complex components of vector potential Â(k)
    A_k = rng.normal(size=(3,N,N,N)) + 1j*rng.normal(size=(3,N,N,N))
    A_k *= kolmogorov_spectrum(k)[None]**0.5

    # B = ∇ × A in k-space: i k × Â
    kvec = np.stack(np.meshgrid(kx,ky,kz, indexing='ij'), axis=0)   # shape (3,N,N,N)
    B_k  = 1j * np.cross(kvec, A_k, axisa=0, axisb=0, axisc=0)

    # back to real space
    B = fft.ifftn(B_k, axes=(1,2,3), workers=-1).real
    # set mean field to zero, normalise rms to 1 μG
    for i in range(3):
        B[i] -= B[i].mean()
        B[i] *= 1.0 / B[i].std()          # 1 μG rms
    return B

print("Generating Kolmogorov B-field …")
Bx, By, Bz = random_field()

print("Generating log-normal electron density …")
# same Fourier amplitudes as Bx but different random phase
amp        = np.abs(fft.fftn(Bx))
phase      = np.exp(2j*np.pi*rng.random((N,N,N)))
logn_k     = amp * phase
logn       = fft.ifftn(logn_k, workers=-1).real
sigma      = 0.5                              # dex dispersion
ne         = np.exp(np.log(0.03) + sigma*logn/logn.std())   # ⟨ne⟩ ~ 0.03 cm⁻³

print("Saving HDF5 …")
with h5py.File("cube.h5","w") as f:
    f["Bx"], f["By"], f["Bz"] = Bx, By, Bz
    f["ne"]                   = ne
    f.attrs["dx_pc"]          = dx
    f.attrs["Note"]           = "Synthetic Kolmogorov MHD cube (N={}³)".format(N)
print("Done.")
