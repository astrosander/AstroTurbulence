#!/usr/bin/env python3
"""
compare_structure.py
--------------------

• Reads test_Kolm_* Fortran arrays (I,Q,U,ang);
• Computes 2‑D angle structure functions for
      – Stokes azimuth  χS = ½ arctan(U/Q)
      – Vector azimuth  χV = ang  (already in degrees or radians)
• Computes the theoretical Δρ(R⊥) by 1‑D integration and converts it to
  Dχ(R) = 0.5 Δρ(R);
• Plots all three on the same axes.

The geometry‑only integral needs an outer‑scale depth L.  We take
    L = outer_scale_pixels = N/2   (edit as needed).
---------------------------------------------------------------------
"""

from __future__ import annotations
import struct, math
from pathlib import Path
from typing import Tuple, Generator

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ------------------------------------------------------------------
# Fortran unformatted‑record reader (unchanged)
# ------------------------------------------------------------------
def read_fortran_2d_array(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        f.read(4)               # leading record length (ignored)
        ndim = struct.unpack("i", f.read(4))[0]
        f.read(4)
        f.read(4)
        if ndim == 2:
            nx, ny = struct.unpack("ii", f.read(8))
        else:                   # protect against 4‑D outputs
            dims = struct.unpack("iiii", f.read(16))
            nx, ny = dims[:2]
        f.read(4)               # end of header record

        f.read(4)
        data = np.fromfile(f, dtype=np.float32, count=nx * ny)
        f.read(4)
    return data.reshape((nx, ny), order="F")


# ------------------------------------------------------------------
# Geometry‑only Δρ(R⊥)  (single 1‑D integral, no baseline term)
# ------------------------------------------------------------------
def delta_rho(R: float,
              L: float,
              C2: float = 1.0,
              N_int: int = 6000) -> float:
    """
    Δρ = ρ̄⊥(0) – ρ̄⊥(R)   with the regularised integrand
         ∫ [ (R²+Δs²)^{1/3} – |Δs|^{2/3} ] (1-|Δs|/L) dΔs
    Accurate trapezoid rule on y = |Δs|/L ∈ [0,1].
    """
    if R == 0:
        return 0.0
    y = np.linspace(0.0, 1.0, N_int)
    core = ((R**2 + (y*L)**2)**(1./3.) - (y*L)**(2./3.)) * (1.0 - y)
    integral = 2.0 * L * np.trapz(core, y)
    return (C2 / (2 * L**(2./3.) * L)) * integral


# ------------------------------------------------------------------
# 2‑D structure function helpers
# ------------------------------------------------------------------
def angular_difference(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Return principal‑value angle differences in (−π,π]."""
    return (phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi


def sample_pairs_2d(field: np.ndarray,
                    R: int,
                    max_pairs: int = 300_000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Random samples of angle differences separated by |R| pixels."""
    Nx, Ny = field.shape
    rng = rng or np.random.default_rng()
    dirs = [(R, 0), (-R, 0), (0, R), (0, -R),
            (R, R), (-R, -R), (R, -R), (-R, R)]
    out: list[np.ndarray] = []
    for dx, dy in dirs:
        n = int(np.sqrt(max_pairs // len(dirs))) * 1000
        xs = rng.integers(0, Nx, size=n)
        ys = rng.integers(0, Ny, size=n)
        p1 = field[xs, ys]
        p2 = field[(xs + dx) % Nx, (ys + dy) % Ny]
        out.append(angular_difference(p1, p2))
    return np.concatenate(out)


def structure_function_2d(field: np.ndarray,
                          Rs: np.ndarray,
                          max_pairs: int = 300_000,
                          harmonic: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """D(R)=½[1−⟨cos (2Δχ)⟩]   (harmonic=2)."""
    D = []
    for R in Rs:
        dphi = sample_pairs_2d(field, int(R), max_pairs=max_pairs)
        D.append(0.5 * (1 - np.cos(harmonic * dphi).mean()))
    return Rs, np.asarray(D)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_structure(Rs: np.ndarray,
                   D_sim: np.ndarray,
                   D_int: np.ndarray,
                   title: str,
                   outfile: Path,
                   expected_slope: float | None = 5/3) -> float:
    half = len(Rs) // 2
    slope, intercept, *_ = linregress(np.log(Rs[:half]), np.log(D_sim[:half]))

    plt.figure(figsize=(4.8, 3.6))
    plt.loglog(Rs, D_sim, "o", ms=4, label="simulation")
    plt.loglog(Rs, D_int+1e-2, lw=1.8, ls="-.", color="tab:red",
               label="1‑D integration")

    ref_slope = expected_slope if expected_slope is not None else slope
    C = D_sim[1] / (Rs[1] ** ref_slope)
    plt.loglog(Rs, C * Rs ** ref_slope, "--", color="black",
               label=f"$\\propto R^{{{ref_slope:.2f}}}$")

    plt.xlabel("R [pixels]")
    plt.ylabel("D(R)")
    plt.title(title, fontsize=13)
    plt.legend(frameon=False)
    plt.ylim(1e-3, 1e0)
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()
    return slope


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    outdir = Path("figures")
    outdir.mkdir(exist_ok=True)

    I = read_fortran_2d_array("input/synchrotron/test_Kolm_L512V_L512_I")
    Q = read_fortran_2d_array("input/synchrotron/test_Kolm_L512V_L512_Q")
    U = read_fortran_2d_array("input/synchrotron/test_Kolm_L512V_L512_U")
    ang = read_fortran_2d_array("input/synchrotron/test_Kolm_L512V_L512_ang")

    # ----- vector azimuth (degrees → radians if needed) -----
    ang_vec = np.radians(ang) if np.max(np.abs(ang)) > np.pi else ang
    # ----- Stokes azimuth  χ = ½ arctan(U/Q)  (already radians) -----
    ang_stokes = 0.5 * np.arctan2(U, Q)

    N = I.shape[0]
    Rs = np.unique(np.logspace(0, np.log10(N // 3), 20).astype(int))

    # -------- theoretical integration: Dχ(R) = ½ Δρ(R) -----------
    outer_scale_pixels = N / 2       # adjust if you know the true L
    D_int = np.array([0.5 * delta_rho(r, outer_scale_pixels)
                      for r in Rs])

    # -------- simulation structure functions --------------------
    _, D_stokes = structure_function_2d(ang_stokes, Rs)
    _, D_vector = structure_function_2d(ang_vec,    Rs)

    slope_stokes = plot_structure(
        Rs,
        D_stokes,
        D_int,
        title="Structure Function – Stokes Azimuth",
        outfile=outdir / "2d_structure_stokes.png",
        expected_slope=5 / 3,
    )

    slope_vector = plot_structure(
        Rs,
        D_vector,
        D_int,
        title="Structure Function – Vector Azimuth",
        outfile=outdir / "2d_structure_vector.png",
        expected_slope=5 / 3,
    )

    print(f"Stokes azimuth : {slope_stokes:.3f}")
    print(f"Vector azimuth : {slope_vector:.3f}")


if __name__ == "__main__":
    main()
