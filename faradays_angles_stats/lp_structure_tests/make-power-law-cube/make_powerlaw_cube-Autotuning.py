#!/usr/bin/env python3
"""
make_powerlaw_cube.py
=====================

Generate a synthetic MHD-like cube where the 3-D power spectrum of
each field is P(k) ∝ k^(−β). We then affinely rescale the real-space
fields to match a requested sample mean and RMS exactly.

Output (HDF5):
    gas_density  : n_e(x,y,z)     (positive)
    k_mag_field  : B_z(x,y,z)     (mean + fluctuations)
    x_coor, y_coor, z_coor        (coords)

Author : <you>
Date   : 2025-06-25
Licence: MIT
"""

from pathlib import Path
import numpy as np
import h5py
from numpy.fft import irfftn, rfftfreq


# ──────────────────────────────────────────────────────────────────────
# 1) helper: isotropic |k| array for rFFT layout
# ──────────────────────────────────────────────────────────────────────
def _k_magnitude(nx, ny, nz, dx=1.0):
    kx = np.fft.fftfreq(nx, d=dx)     # full along x
    ky = np.fft.fftfreq(ny, d=dx)     # full along y
    kz = rfftfreq(nz, d=dx)           # half along z (rFFT)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij", sparse=True)
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    k[0, 0, 0] = np.inf               # avoid div-by-zero at DC
    return k


# ──────────────────────────────────────────────────────────────────────
# 2) Gaussian field with P(k) ∝ k^(−β), unit variance in real space
# ──────────────────────────────────────────────────────────────────────
def gaussian_powerlaw_cube(N, beta, seed=None, dx=1.0):
    rng = np.random.default_rng(seed)
    k   = _k_magnitude(N, N, N, dx)
    amp = k ** (-beta / 2.0)

    noise = rng.normal(size=(N, N, N//2 + 1)) + 1j * rng.normal(size=(N, N, N//2 + 1))
    noise[0, 0, 0] = 0.0  # kill DC

    field_k = amp * noise
    cube = irfftn(field_k, s=(N, N, N)).real
    cube -= cube.mean(dtype=np.float64)
    cube /= cube.std(ddof=0, dtype=np.float64)  # unit-variance
    return cube.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 3) exact mean/RMS matching via affine scaling about the mean
#     y' = M + a (y − μ), with a chosen so RMS(y') = R
# ──────────────────────────────────────────────────────────────────────
def scale_to_mean_rms(y, mean_target, rms_target):
    y = y.astype(np.float64, copy=False)
    mu  = y.mean()
    var = y.var(ddof=0)
    M   = float(mean_target)
    R   = float(rms_target)

    if R < abs(M):
        raise ValueError(f"Impossible target: RMS ({R}) must be ≥ |mean| ({abs(M)}).")
    if var == 0.0:
        # Degenerate: all values equal. Only works if R == |M|.
        if R == abs(M):
            return np.full_like(y, M, dtype=np.float64)
        raise ValueError("Input variance is zero; cannot match the requested RMS.")

    a = np.sqrt((R*R - M*M) / var)
    y_prime = M + a * (y - mu)
    return y_prime.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 4) n_e: small-sigma log-normal → affine match to (mean, RMS)
# ──────────────────────────────────────────────────────────────────────
def make_ne_cube(N, beta, seed, mean_target=1.0, rms_target=1.0, dx=1.0):
    """
    Build positive n_e with Kolmogorov-like spatial structure.
    We start with a Gaussian field g (unit var), then X = exp(s*g) with a
    small log-normal sigma 's' picked from the target mean/RMS ratio
    (keeps the field safely positive). Finally, we rescale about the mean
    to hit the sample mean and RMS exactly.
    """
    g = gaussian_powerlaw_cube(N, beta, seed=seed, dx=dx).astype(np.float64)

    # If rms_target == mean_target, ln-ratio is 0 → s=0 → nearly uniform before final scaling.
    ratio = max(1.0, float(rms_target) / float(mean_target))
    s2 = 2.0 * np.log(ratio)    # for log-normal, RMS/mean = exp(s^2/2)
    s  = np.sqrt(s2)

    X = np.exp(s * g)                           # positive
    # crude mean set to get close:
    X *= (mean_target / X.mean())               # sample mean ≈ mean_target
    # exact mean & RMS by affine scaling around mean:
    ne = scale_to_mean_rms(X, mean_target, rms_target)
    return ne


# ──────────────────────────────────────────────────────────────────────
# 5) B_z: Gaussian field → affine match to (mean, RMS)
# ──────────────────────────────────────────────────────────────────────
def make_bz_cube(N, beta, seed, mean_target, rms_target, dx=1.0):
    g = gaussian_powerlaw_cube(N, beta, seed=seed, dx=dx)  # zero-mean, unit-std
    bz = scale_to_mean_rms(g, mean_target, rms_target)     # exact sample mean & RMS
    return bz


# ──────────────────────────────────────────────────────────────────────
# 6) coordinate arrays
# ──────────────────────────────────────────────────────────────────────
def make_coords(N, dx=1.0):
    x = (np.arange(N, dtype=np.float64) - N/2 + 0.5) * dx
    y = (np.arange(N, dtype=np.float64) - N/2 + 0.5) * dx
    z = (np.arange(N, dtype=np.float64) - N/2 + 0.5) * dx
    X = np.broadcast_to(x[:, None, None], (N, N, N)).astype(np.float32)
    Y = np.broadcast_to(y[None, :, None], (N, N, N)).astype(np.float32)
    Z = np.broadcast_to(z[None, None, :], (N, N, N)).astype(np.float32)
    return X, Y, Z


# ──────────────────────────────────────────────────────────────────────
# 7) utility: print stats
# ──────────────────────────────────────────────────────────────────────
def stats(name, arr):
    arr64 = arr.astype(np.float64, copy=False)
    mean = arr64.mean()
    rms  = np.sqrt((arr64 * arr64).mean())
    print(f"{name:12s}  mean={mean:.15g}  min={arr64.min():.9g}  max={arr64.max():.9g}  rms={rms:.15g}")


# ──────────────────────────────────────────────────────────────────────
# 8) main
# ──────────────────────────────────────────────────────────────────────
def main(
    N: int,
    beta_ne: float,
    beta_bz: float,
    mean_ne: float,
    rms_ne: float,
    mean_bz: float,
    rms_bz: float,
    dx: float,
    seed: int,
    out: Path,
    write_coords: bool = True,
):
    rng = np.random.default_rng(seed)
    seed_ne, seed_bz = rng.integers(0, 2**31, size=2)

    print(f"• generating n_e  (β={beta_ne}) …")
    ne = make_ne_cube(N, beta_ne, seed_ne, mean_ne, rms_ne, dx=dx)

    print(f"• generating B_z  (β={beta_bz}) …")
    bz = make_bz_cube(N, beta_bz, seed_bz, mean_bz, rms_bz, dx=dx)

    stats("gas_density", ne)
    stats("k_mag_field", bz)

    X = Y = Z = None
    if write_coords:
        X, Y, Z = make_coords(N, dx)

    print(f"• writing {out} …")
    with h5py.File(out, "w") as h5:
        h5.create_dataset("gas_density", data=ne, compression="gzip")
        h5.create_dataset("k_mag_field", data=bz, compression="gzip")
        h5.create_dataset("x_coor",      data=X,  compression="gzip")
        h5.create_dataset("y_coor",      data=Y,  compression="gzip")
        h5.create_dataset("z_coor",      data=Z,  compression="gzip")

    print("done.")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NE_MEAN = 1.0000001192092896
    NE_RMS  = 1.000738263130188

    BZ_MEAN = 0.11952292174100876
    BZ_RMS  = 0.12989801168441772

    main(
        N=256,
        beta_ne=11/3,
        beta_bz=11/3,
        mean_ne=NE_MEAN,
        rms_ne=NE_RMS,
        mean_bz=BZ_MEAN,
        rms_bz=BZ_RMS,
        dx=1.0,
        seed=2025,
        out=Path("synthetic_kolmogorov.h5"),
        write_coords=True,
    )
