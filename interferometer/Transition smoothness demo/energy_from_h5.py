#!/usr/bin/env python3
"""
Read 2D field from .h5 and plot its 1D spectrum E(k).
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftfreq


def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int = 50, kmin: float = 1e-3):
    """
    Compute azimuthal average of 2D power spectrum.
    """
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx)
    kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KY, KX)

    kmax = K.max()
    bins = np.logspace(np.log10(max(kmin, 1e-8)), np.log10(kmax), nbins + 1)
    k = K.ravel()
    p = P2D.ravel()
    idx = np.digitize(k, bins) - 1
    good = (idx >= 0) & (idx < nbins) & np.isfinite(p)
    sums = np.bincount(idx[good], weights=p[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)

    prof = np.full(nbins, np.nan, float)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]

    kcent = 0.5 * (bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]


def plot_spectrum(h5_path: str, out_png: str = "Ek.png"):
    # --- read the map ---
    with h5py.File(h5_path, "r") as h5:
        f2 = h5["k_mag_field"][:]  # shape (Nx, Ny, 1)
        f2 = f2[:, :, 0].T         # back to (ny, nx)
        # dx = float(h5.attrs["dx"])
        dx=1.0

    # --- compute spectrum ---
    Fk = fft2(f2)
    P2D = (Fk * np.conj(Fk)).real
    k1d, Pk = ring_average_2d(P2D, dx)
    E1D = 2 * np.pi * k1d * Pk

    # --- plot ---
    plt.figure(figsize=(6, 4))
    plt.loglog(k1d, E1D, lw=1.5)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.title("1D spectrum from " + h5_path)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Example: adjust path to one of your saved .h5 files
    # plot_spectrum("h5/transition_smoothness/two_slope_2D_s3_r00.h5")
    plot_spectrum("h5/transition_smoothness/two_slope_2D_s4_r00.h5")
    # plot_spectrum("../../faradays_angles_stats/lp_structure_tests/mhd_fields.h5")