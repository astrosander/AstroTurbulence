#!/usr/bin/env python3
"""
Kolmogorov‑like 3‑D velocity cube → plane‑of‑sky projection →
vector & Stokes angle statistics.

Run from the command line, e.g.
    python turbulence_angles.py --N 256 --outfile baseline.h5
Add "--solenoidal" to enforce ∇·u = 0.

Main stages
-----------
1. generate_velocity_cube()     – synthetic k^(-11/3) field
2. los_integrate()              – integrate along chosen LOS axis
3. angle_maps_vector()          – EPC angles by direct vector addition
4. angle_maps_stokes()          – angles via Stokes (Q,U) addition
5. diagnostics()                – PDFs & structure functions

Outputs
-------
• HDF5 file with raw cubes + angle maps
• *.png figures in ./figures
"""
# ---------------------------------------------------------------------
import argparse
import os
from pathlib import Path
import h5py
import numpy as np
from numpy.fft import rfftn, irfftn, fftfreq
from scipy.stats import cauchy
import matplotlib.pyplot as plt
from numba import njit, prange

# --------------------------- 1.  Velocity cube ------------------------
def _solenoidal_projection(u_k, kx, ky, kz):
    """Project Fourier velocity field onto divergence‑free (solenoidal) sub‑space."""
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # avoid divide by zero
    k_dot_u = kx * u_k[..., 0] + ky * u_k[..., 1] + kz * u_k[..., 2]
    for i, kcomp in enumerate([kx, ky, kz]):
        u_k[..., i] -= kcomp * k_dot_u / k2
    return u_k


def generate_velocity_cube(N: int = 256,
                           outer_scale: int | float | None = None,
                           solenoidal: bool = False,
                           seed: int | None = None) -> np.ndarray:
    """
    Returns u[x,y,z,3] drawn from Kolmogorov spectrum (E~k^-11/3).

    Parameters
    ----------
    N : int
        Grid size (cubic).
    outer_scale : float | None
        Characteristic energy‑containing scale in grid cells.
    solenoidal : bool
        If True, Helmholtz‑project to ∇·u = 0.
    seed : int | None
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    outer_scale = outer_scale or N / 4

    # --- Fourier grid -------------------------------------------------
    kx = fftfreq(N)[:, None, None]
    ky = fftfreq(N)[None, :, None]
    kz = fftfreq(N)[: N // 2 + 1][None, None, :]
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0, 0, 0] = 1.0  # avoid divide by zero

    # amplitude A(k) ~ k^{-11/6}
    amp = k_mag**(-11 / 6.0)
    # impose a smooth large‑scale cutoff
    amp *= np.exp(-(k_mag * outer_scale / N)**2)

    # random complex field with Gaussian real & imag
    phase = rng.standard_normal((N, N, N // 2 + 1, 3)) + 1j * rng.standard_normal((N, N, N // 2 + 1, 3))
    u_k = amp[..., None] * phase

    if solenoidal:
        u_k = _solenoidal_projection(u_k, kx, ky, kz)

    # inverse FFT –> real space
    u = irfftn(u_k, s=(N, N, N), axes=(0, 1, 2))
    return u.astype(np.float32)


# --------------------------- 2.  LOS integration ----------------------
def los_integrate(u: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Integrate velocity field along line‑of‑sight axis.

    Parameters
    ----------
    u : ndarray, shape (Nx,Ny,Nz,3)
    axis : int
        Axis to integrate (default =z).
    """
    return np.sum(u, axis=axis)


# --------------------------- 3.  Angle maps ---------------------------
def angle_maps_vector(u_int: np.ndarray) -> np.ndarray:
    """
    Map of angles via vector addition (arctan of integrated transverse components).

    Returns
    -------
    theta_v : ndarray, shape (Nx,Ny)
        Angles in [‑π, π).
    """
    ux, uy = u_int[..., 0], u_int[..., 1]
    return np.arctan2(uy, ux)


@njit(parallel=True, fastmath=True)
def _stokes_accumulate(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Accumulate Q,U along z for every (x,y).
    JIT‑compiled for speed.
    """
    Nx, Ny, Nz, _ = u.shape
    Q = np.zeros((Nx, Ny), dtype=np.float32)
    U = np.zeros((Nx, Ny), dtype=np.float32)
    for i in prange(Nx):
        for j in prange(Ny):
            q, u_local = 0.0, 0.0
            for k in range(Nz):
                ux, uy = u[i, j, k, 0], u[i, j, k, 1]
                theta = np.arctan2(uy, ux)
                q += np.cos(2.0 * theta)
                u_local += np.sin(2.0 * theta)
            Q[i, j] = q
            U[i, j] = u_local
    return Q, U


def angle_maps_stokes(u: np.ndarray) -> np.ndarray:
    """
    Angles obtained by adding Stokes vectors per cell.

    Returns
    -------
    theta_s : ndarray, shape (Nx,Ny)
        Stokes‑averaged angle field.
    """
    Q, U = _stokes_accumulate(u)
    return 0.5 * np.arctan2(U, Q)


# --------------------------- 4.  Statistics ---------------------------
def pdf_histogram(theta: np.ndarray, nbins: int = 180):
    """Return (centers, pdf) of θ with sin‑weight normalisation."""
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    counts, _ = np.histogram(theta.ravel(), bins=bins, density=False)
    centers = 0.5 * (bins[:-1] + bins[1:])
    # sinθ factor for isotropic surface element
    weights = np.sin(np.abs(centers))
    pdf = counts / (weights * counts.sum())
    return centers, pdf


def two_point_delta(theta_map: np.ndarray,
                    r_vec: tuple[int, int]) -> np.ndarray:
    """
    Δθ catalogue for one separation vector (rx,ry).
    Periodic boundary conditions.

    Parameters
    ----------
    theta_map : ndarray (Nx,Ny)
    r_vec : (int,int)
        Pixel shifts (rx,ry).
    """
    rx, ry = r_vec
    shifted = np.roll(theta_map, shift=(rx, ry), axis=(0, 1))
    return (theta_map - shifted + np.pi) % (2 * np.pi) - np.pi


def structure_function(theta_map: np.ndarray,
                       separations: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute D(R)=½⟨1‑cosΔθ⟩ vs R.

    Returns
    -------
    R : ndarray
    D : ndarray
    """
    Nx, Ny = theta_map.shape
    D = []
    for R in separations:
        deltas = []
        # sample eight lattice directions for statistical efficiency
        for dx, dy in ((R, 0), (-R, 0), (0, R), (0, -R),
                       (R, R), (-R, -R), (R, -R), (-R, R)):
            deltas.append(two_point_delta(theta_map, (dx, dy)).ravel())
        deltas = np.concatenate(deltas)
        D.append(0.5 * (1.0 - np.cos(deltas)).mean())
    return np.asarray(separations), np.asarray(D)


# --------------------------- 5.  Diagnostics --------------------------
def diagnostics(theta_v, theta_s, outdir="figures"):
    """Plot single‑point PDFs and structure functions."""
    os.makedirs(outdir, exist_ok=True)

    # --- single‑point PDFs -------------------------------------------
    for name, theta in (("vector", theta_v), ("stokes", theta_s)):
        x, pdf = pdf_histogram(theta)
        plt.figure()
        plt.semilogy(x, pdf, label=f"empirical {name}")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$P(\theta)$")
        plt.title(f"Single‑point angle PDF – {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(outdir) / f"pdf_{name}.png", dpi=200)
        plt.close()

    # --- structure functions -----------------------------------------
    N = theta_v.shape[0]
    seps = np.unique(np.logspace(0, np.log10(N // 3), 20).astype(int))

    for name, theta in (("vector", theta_v), ("stokes", theta_s)):
        R, D = structure_function(theta, seps)
        plt.figure()
        plt.loglog(R, D, "o-", label=f"{name}")
        # power‑law guides
        slope = 2 / 3 if name == "vector" else 5 / 3
        plt.loglog(R, 0.1 * R**(slope), "--",
                   label=fr"$\propto R^{{{slope:.2f}}}$")
        plt.xlabel(r"$R$  [pixels]")
        plt.ylabel(r"$D(R)$")
        plt.title(f"Angular structure function – {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(outdir) / f"structure_{name}.png", dpi=200)
        plt.close()


# --------------------------- 6.  CLI wrapper --------------------------
def main():
    p = argparse.ArgumentParser(
        description="Synthetic Kolmogorov turbulence → angle statistics")
    p.add_argument("--N", type=int, default=256,
                   help="Cube size (default 256)")
    p.add_argument("--los-axis", type=int, default=2,
                   help="0=x,1=y,2=z (default 2)")
    p.add_argument("--solenoidal", action="store_true",
                   help="Enforce ∇·u = 0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--outfile", default="simulation.h5",
                   help="HDF5 output")
    args = p.parse_args()

    # ---------- 1–2 ---------------------------------------------------
    print("Generating velocity cube …")
    u = generate_velocity_cube(
        N=args.N,
        solenoidal=args.solenoidal,
        seed=args.seed)
    print("Integrating along LOS …")
    u_int = los_integrate(u, axis=args.los_axis)
    # ---------- 3 -----------------------------------------------------
    print("Computing angle maps …")
    theta_v = angle_maps_vector(u_int)
    theta_s = angle_maps_stokes(u)
    # ---------- 4 -----------------------------------------------------
    print("Running diagnostics …")
    diagnostics(theta_v, theta_s)

    # ---------- save everything --------------------------------------
    print(f"Writing {args.outfile}")
    with h5py.File(args.outfile, "w") as h5:
        h5["u"] = u           # (N,N,N,3)
        h5["u_int"] = u_int   # (N,N,3)
        h5["theta_vector"] = theta_v
        h5["theta_stokes"] = theta_s
        h5.attrs["solenoidal"] = args.solenoidal
        h5.attrs["seed"] = args.seed or -1
    print("Done.")


if __name__ == "__main__":   # pragma: no cover
    main()
