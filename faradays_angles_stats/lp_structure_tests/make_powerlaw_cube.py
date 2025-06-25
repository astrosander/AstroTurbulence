#!/usr/bin/env python3
"""
make_powerlaw_cube.py
=====================

Generate a synthetic MHD-like cube in which the 3-D power spectrum of
each field is a pure power law  P(k) ∝ k^(-β).

Output (HDF5):
    gas_density  : n_e(x,y,z)  (always ≥ 0)
    k_mag_field  : B_z  (mean + fluct.)
    x_coor, y_coor, z_coor  : physical coordinates (optional)

Author : <you>
Date   : 2025-06-25
Licence: MIT
"""

from pathlib import Path
import argparse

import numpy as np
import h5py
from numpy.fft import rfftn, irfftn, rfftfreq

# ──────────────────────────────────────────────────────────────────────
# 1.  helper: isotropic |k| array
# ──────────────────────────────────────────────────────────────────────
def _k_magnitude(nx, ny, nz, dx=1.0):
    kx = np.fft.fftfreq(nx, d=dx)     # full spectrum
    ky = np.fft.fftfreq(ny, d=dx)
    kz = rfftfreq(nz, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij", sparse=True)
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    k[0, 0, 0] = np.inf  # avoid division by zero at the DC mode
    return k


# ──────────────────────────────────────────────────────────────────────
# 2.  power-law Gaussian field
# ──────────────────────────────────────────────────────────────────────
def gaussian_powerlaw_cube(N, beta, seed=None, dx=1.0):
    """
    Return a zero-mean, unit-variance Gaussian field with 3-D spectrum
    P(k) ∝ k^(−β).

    Parameters
    ----------
    N    : int            (cube size: N×N×N)
    beta : float          (spectral index)
    seed : int | None     (rng seed)
    dx   : float          (grid spacing)
    """
    rng = np.random.default_rng(seed)
    k   = _k_magnitude(N, N, N, dx)
    amp = k ** (-beta / 2.0)

    noise = rng.normal(size=(N, N, N//2 + 1)) \
          + 1j * rng.normal(size=(N, N, N//2 + 1))
    noise[0, 0, 0] = 0.0   # kill DC

    field_k = amp * noise
    cube    = irfftn(field_k, s=(N, N, N)).real
    cube -= cube.mean()
    cube /= cube.std(ddof=0)
    return cube


# ──────────────────────────────────────────────────────────────────────
# 3.  build electron density and Bz
# ──────────────────────────────────────────────────────────────────────
def make_ne_cube(N, beta, seed):
    """Log-normal positive density field."""
    g = gaussian_powerlaw_cube(N, beta, seed=seed)
    ne = np.exp(g)         # log-normal ⇒ always positive
    ne /= ne.mean()        # normalise <n_e>=1
    return ne.astype(np.float32)


def make_bz_cube(N, beta, seed, mean_bz=0.0):
    """Mean field + fluctuations with power-law spectrum."""
    fluctuations = gaussian_powerlaw_cube(N, beta, seed=seed)
    bz = mean_bz + fluctuations
    bz -= bz.mean() - mean_bz   # re-centred so that mean exactly = mean_bz
    return bz.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 4.  coordinate arrays (optional, but nice to have)
# ──────────────────────────────────────────────────────────────────────
def make_coords(N, dx=1.0):
    x = (np.arange(N) - N/2 + 0.5) * dx
    y = (np.arange(N) - N/2 + 0.5) * dx
    z = (np.arange(N) - N/2 + 0.5) * dx
    X = np.broadcast_to(x[:, None, None], (N, N, N)).astype(np.float32)
    Y = np.broadcast_to(y[None, :, None], (N, N, N)).astype(np.float32)
    Z = np.broadcast_to(z[None, None, :], (N, N, N)).astype(np.float32)
    return X, Y, Z


# ──────────────────────────────────────────────────────────────────────
# 5.  main
# ──────────────────────────────────────────────────────────────────────
def main(
    N: int,
    beta_ne: float,
    beta_bz: float,
    mean_bz: float,
    dx: float,
    seed: int,
    out: Path,
    save_fluctuations_only: bool = False,
):
    rng = np.random.default_rng(seed)
    seed_ne, seed_bz = rng.integers(0, 2**31, size=2)

    print(f"• generating n_e  (β={beta_ne}) …")
    ne = make_ne_cube(N, beta_ne, seed_ne)

    print(f"• generating B_z  (β={beta_bz},  mean={mean_bz}) …")
    bz = make_bz_cube(N, beta_bz, seed_bz, mean_bz)
    
    # Also save fluctuations only (without mean field) for comparison
    bz_fluctuations = bz - mean_bz if mean_bz != 0 else bz

    X, Y, Z = make_coords(N, dx)

    print(f"• writing {out} …")
    with h5py.File(out, "w") as h5:
        h5.create_dataset("gas_density", data=ne, compression="gzip")
        h5.create_dataset("k_mag_field", data=bz, compression="gzip")
        h5.create_dataset("k_mag_field_fluctuations", data=bz_fluctuations, compression="gzip")  # Add fluctuations-only field
        h5.create_dataset("x_coor",      data=X,  compression="gzip")
        h5.create_dataset("y_coor",      data=Y,  compression="gzip")
        h5.create_dataset("z_coor",      data=Z,  compression="gzip")
        
        # Store metadata about the cube
        h5.attrs["beta_ne"] = beta_ne
        h5.attrs["beta_bz"] = beta_bz
        h5.attrs["mean_bz"] = mean_bz
        h5.attrs["dx"] = dx
        h5.attrs["seed"] = seed

    print("done.")


def generate_test_suite(base_name="synthetic_test", N=512):
    """
    Generate a suite of test cubes with different parameters for systematic testing.
    Based on Dr. Lazarian's suggestions to test pure power laws.
    """
    test_cases = [
        # (beta_ne, beta_bz, mean_bz, description)
        (11/3, 11/3, 0.0, "kolmogorov_no_mean"),
        (11/3, 11/3, 1.0, "kolmogorov_with_mean"),
        (11/3, 11/3, 5.0, "kolmogorov_strong_mean"),
        (8/3, 8/3, 0.0, "steep_powerlaw"),
        (14/3, 14/3, 0.0, "shallow_powerlaw"),
    ]
    
    for i, (beta_ne, beta_bz, mean_bz, desc) in enumerate(test_cases):
        filename = f"{base_name}_{desc}.h5"
        print(f"\n=== Generating {filename} ===")
        main(
            N=N,
            beta_ne=beta_ne,
            beta_bz=beta_bz,
            mean_bz=mean_bz,
            dx=1.0,
            seed=2025 + i,
            out=Path(filename)
        )

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate a synthetic cube with power-law spectra"
    )
    p.add_argument("--N", type=int, default=256,
                   help="linear size of the cube (default 256)")
    p.add_argument("--beta_ne", type=float, default=11/3,
                   help="spectral index β for n_e (default 11/3 ≈ Kolmogorov)")
    p.add_argument("--beta_bz", type=float, default=11/3,
                   help="spectral index β for B_z (default 11/3)")
    p.add_argument("--mean_bz", type=float, default=0.0,
                   help="uniform mean B_z to add (default 0)")
    p.add_argument("--dx", type=float, default=1.0,
                   help="physical pixel size (for coordinate arrays)")
    p.add_argument("--seed", type=int, default=2025,
                   help="random seed (default 2025)")
    p.add_argument("--out", required=True,
                   help="output filename (HDF5)")
    p.add_argument("--generate-suite", action="store_true",
                   help="generate a test suite of cubes with different parameters")

    args = p.parse_args()
    
    if args.generate_suite:
        generate_test_suite(base_name="synthetic_test", N=args.N)
    else:
        main(
            N=args.N,
            beta_ne=args.beta_ne,
            beta_bz=args.beta_bz,
            mean_bz=args.mean_bz,
            dx=args.dx,
            seed=args.seed,
            out=Path(args.out),
        )
