#!/usr/bin/env python3
"""
make_two_slope_cube.py
======================

Generate synthetic 3-D cubes where the *1D shell spectrum* E_1D(k)
has two power-law segments: rising +3/2 (low-k) and falling -5/3 (high-k).
We shape the *3-D spectral density* P_3D(k) accordingly:

    E_1D(k) = 4π k^2 P_3D(k)  ⇒  to target E_1D ∝ k^α  use  P_3D ∝ k^(α−2).

We then affinely rescale the real-space fields to hit requested sample mean and RMS.

Output (HDF5):
    gas_density  : n_e(x,y,z)      (positive; log-normalized then affine)
    k_mag_field  : B_z(x,y,z)      (mean + fluctuations; affine)
    x_coor, y_coor, z_coor         (coords)

No argparse; edit the CONFIG in __main__.
"""

from pathlib import Path
import numpy as np
import h5py
from numpy.fft import irfftn, rfftfreq

# ──────────────────────────────────────────────────────────────────────
# 1) helper: isotropic |k| array for rFFT layout (cycles per dx)
# ──────────────────────────────────────────────────────────────────────
def _k_magnitude_rfft(nx, ny, nz, dx=1.0, k_floor_mult=1.0):
    """
    Build |k| on the rFFT layout (full x,y; half z). Units: cycles/dx.

    k_floor_mult provides a tiny regularization near k=0:
        k_safe = sqrt(k^2 + k0^2), with k0 = k_floor_mult / (N * dx)
    to prevent excessive low-k blowup when density exponent > -3.
    """
    kx = np.fft.fftfreq(nx, d=dx)         # shape (nx,)
    ky = np.fft.fftfreq(ny, d=dx)         # shape (ny,)
    kz = rfftfreq(nz, d=dx)               # shape (nz//2 + 1,)

    # sparse mesh to avoid big memory overhead
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij", sparse=True)
    k = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Regularize near zero to avoid huge power at tiny k (except exact DC)
    k0 = k_floor_mult / (max(nx, ny, nz) * dx)
    k_safe = np.sqrt(k*k + k0*k0)
    k_safe = np.asarray(k_safe)

    # Keep DC clean (no power at k=0)
    k_safe[0, 0, 0] = np.inf
    return k, k_safe


# ──────────────────────────────────────────────────────────────────────
# 2) smooth two-slope 3D spectral density P_3D(k)
#     target shell slopes α_low=+3/2 and α_high=−5/3 in E_1D(k)
# ──────────────────────────────────────────────────────────────────────
def two_slope_P3D(k_safe, k_break, alpha_low=+1.5, alpha_high=-(5.0/3.0),
                  sharp=8.0, eps=1e-30):
    """
    Return P_3D(k) with a smooth transition (logistic in log-k) between
    density exponents γ_low = α_low − 2 and γ_high = α_high − 2.

      At k << k_break:  P_3D ∝ k^(γ_low)   → E_1D ∝ k^(α_low)
      At k >> k_break:  P_3D ∝ k^(γ_high)  → E_1D ∝ k^(α_high)

    Parameters
    ----------
    k_safe   : array of |k| (cycles/dx), with small regularization already applied
    k_break  : scalar (cycles/dx) where the transition occurs (choose in inertial range)
    sharp    : controls the steepness of the transition (higher = sharper)
    """
    # convert target E_1D slopes to 3D spectral-density slopes
    gamma_low  = alpha_low  - 2.0   # +1.5 → -0.5
    gamma_high = alpha_high - 2.0   # -1.666… → -3.666…

    # smooth blend t ≈ 1 at k<<kb, t ≈ 0 at k>>kb
    kb = float(max(k_break, eps))
    x  = np.maximum(k_safe, kb*1e-9) / kb
    t  = 1.0 / (1.0 + x**sharp)

    # amplitude continuity at k=kb (dimensionless k/kb form)
    P_low  = x**(gamma_low)
    P_high = x**(gamma_high)

    # blended density; no DC (handled by caller)
    P = t * P_low + (1.0 - t) * P_high
    return P


# ──────────────────────────────────────────────────────────────────────
# 3) Gaussian field with prescribed two-slope P_3D(k), unit variance
# ──────────────────────────────────────────────────────────────────────
def gaussian_two_slope_cube(N, alpha_low, alpha_high, k_break_cyc,
                            sharp=8.0, seed=None, dx=1.0, k_floor_mult=1.0):
    """
    Build a real 3D field with target E_1D slopes α_low (low k) and α_high (high k).
    Uses rFFT along z and irfftn to return a real cube.

    Returns float32 unit-variance, zero-mean cube.
    """
    rng = np.random.default_rng(seed)
    # rFFT layout (full x,y; half z)
    noise = rng.normal(size=(N, N, N//2 + 1)) + 1j * rng.normal(size=(N, N, N//2 + 1))
    noise[0, 0, 0] = 0.0  # kill DC

    k, k_safe = _k_magnitude_rfft(N, N, N, dx=dx, k_floor_mult=k_floor_mult)
    P3D = two_slope_P3D(k_safe, k_break=k_break_cyc,
                        alpha_low=alpha_low, alpha_high=alpha_high, sharp=sharp)
    # amplitude = sqrt(P3D)
    amp = np.sqrt(np.maximum(P3D, 0.0))

    field_k = amp * noise
    cube = irfftn(field_k, s=(N, N, N)).real

    # standardize
    cube = cube.astype(np.float64, copy=False)
    cube -= cube.mean()
    std = cube.std(ddof=0)
    if std == 0.0:
        raise RuntimeError("Two-slope synthesis resulted in zero variance (unexpected).")
    cube /= std
    return cube.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 4) exact mean/RMS matching via affine scaling
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
        if R == abs(M):
            return np.full_like(y, M, dtype=np.float64)
        raise ValueError("Input variance is zero; cannot match the requested RMS.")

    a = np.sqrt((R*R - M*M) / var)
    y_prime = M + a * (y - mu)
    return y_prime.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 5) n_e (positive) and B_z (signed) constructors
# ──────────────────────────────────────────────────────────────────────
def make_ne_cube_two_slope(N, alpha_low, alpha_high, k_break_cyc, sharp,
                           seed, mean_target=1.0, rms_target=1.0, dx=1.0, k_floor_mult=1.0):
    """
    Build positive n_e:
      - synthesize Gaussian two-slope field g
      - exponentiate to weak log-normal X = exp(s g) to enforce positivity
      - affine rescale to hit exact sample mean and RMS
    """
    g = gaussian_two_slope_cube(N, alpha_low, alpha_high, k_break_cyc, sharp, seed, dx, k_floor_mult).astype(np.float64)

    # choose small log-normal sigma from mean/RMS ratio
    ratio = max(1.0, float(rms_target) / float(mean_target))
    s2 = 2.0 * np.log(ratio)     # RMS/mean = exp(s^2/2)
    s  = np.sqrt(s2)

    X = np.exp(s * g)
    X *= (mean_target / X.mean())  # crude mean correction
    ne = scale_to_mean_rms(X, mean_target, rms_target)
    return ne


def make_bz_cube_two_slope(N, alpha_low, alpha_high, k_break_cyc, sharp,
                           seed, mean_target, rms_target, dx=1.0, k_floor_mult=1.0):
    """
    Build signed B_z with two-slope spectrum, then affine to (mean, RMS).
    """
    g = gaussian_two_slope_cube(N, alpha_low, alpha_high, k_break_cyc, sharp, seed, dx, k_floor_mult)
    bz = scale_to_mean_rms(g, mean_target, rms_target)
    return bz


def make_field_constant(N, mean_target):
    return np.full((N, N, N), float(mean_target), dtype=np.float32)


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
    # target shell slopes and break (in cycles/dx)
    alpha_low: float,
    alpha_high: float,
    k_break_cyc: float,
    sharp: float,
    # field statistics
    mean_ne: float,
    rms_ne: float,
    mean_bz: float,
    rms_bz: float,
    dx: float,
    seed: int,
    out: Path,
    write_coords: bool = True,
    constant_bz: bool = False,
    constant_ne: bool = False,
    k_floor_mult: float = 1.0,   # regularization near k=0 (≈ 1 fundamental)
):
    rng = np.random.default_rng(seed)
    seed_ne, seed_bz = rng.integers(0, 2**31, size=2)

    if constant_ne:
        print("• generating n_e  (constant field) …")
        ne = make_field_constant(N, mean_ne)
    else:
        print(f"• generating n_e  (two-slope α_low={alpha_low:+.3f}, α_high={alpha_high:+.3f}, k_break={k_break_cyc}) …")
        ne = make_ne_cube_two_slope(N, alpha_low, alpha_high, k_break_cyc, sharp,
                                    seed_ne, mean_ne, rms_ne, dx=dx, k_floor_mult=k_floor_mult)

    if constant_bz:
        print("• generating B_z  (constant field) …")
        bz = make_field_constant(N, mean_bz)
    else:
        print(f"• generating B_z  (two-slope α_low={alpha_low:+.3f}, α_high={alpha_high:+.3f}, k_break={k_break_cyc}) …")
        bz = make_bz_cube_two_slope(N, alpha_low, alpha_high, k_break_cyc, sharp,
                                    seed_bz, mean_bz, rms_bz, dx=dx, k_floor_mult=k_floor_mult)

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
    # Example targets: +3/2 (rising) below k_break, -5/3 (falling) above k_break
    ALPHA_LOW  = +1.5
    ALPHA_HIGH = -5.0/3.0
    K_BREAK    = 0.06      # cycles per dx; choose in the inertial range
    SHARP      = 8.0       # larger → sharper transition

    # Example field stats (same as your previous script)
    NE_MEAN = 1.0000001192092896
    NE_RMS  = 1.000738263130188
    BZ_MEAN = 0.11952292174100876
    BZ_RMS  = 0.12989801168441772

    main(
        N=256,
        alpha_low=ALPHA_LOW,
        alpha_high=ALPHA_HIGH,
        k_break_cyc=K_BREAK,
        sharp=SHARP,
        mean_ne=NE_MEAN,
        rms_ne=NE_RMS,
        mean_bz=BZ_MEAN,
        rms_bz=BZ_RMS,
        dx=1.0,
        seed=2025,
        out=Path("../synthetic_two_slope.h5"),
        write_coords=True,
        k_floor_mult=1.0,          # ~ one fundamental freq as low-k regularization
    )
