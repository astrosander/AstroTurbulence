# angle_spectrum.py  — 2025-06-29  (with reference-slope overlay)
"""
Compute the 2-D angle power-spectrum P_φ(k⊥) and plot it together
with a reference power-law slope.

Usage
-----
python angle_spectrum.py ms01ma08.mhd_w.00300.vtk.h5 synthetic_tuned.h5
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence, Tuple, Dict

import numpy as np
import h5py
import matplotlib.pyplot as plt

# ---------- configuration ----------------------------------------------------
DX_FALLBACK = 1.0            # default pixel size [same units as x,y]
NBINS       = 50             # radial bins for azimuthal average
REF_SLOPE   = -8.0 / 3.0     # theoretical P(k) ∝ k^REF_SLOPE (Kolmogorov)

# ---------- helpers ----------------------------------------------------------
def _safe_spacing(coord: np.ndarray | None) -> float:
    """Return a positive grid spacing or DX_FALLBACK."""
    if coord is None or coord.size < 2:
        return DX_FALLBACK

    diffs = np.diff(coord).astype(float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return float(np.median(diffs)) if diffs.size else DX_FALLBACK


# ---------- I/O --------------------------------------------------------------
def load_phi_map(path: Path) -> Tuple[np.ndarray, float]:
    """
    Integrate n_e · B_z along LOS -> Φ(x,y) and return (Φ, dx).
    """
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:]
        bz = f["k_mag_field"][:]
        x_line = f["x_coor"][:, 0, 0] if "x_coor" in f else None
        z_line = f["z_coor"][0, 0, :] if "z_coor" in f else None

    dx = _safe_spacing(x_line)
    dz = _safe_spacing(z_line)
    phi = (ne * bz).sum(axis=2) * dz
    return phi.astype(np.float32), dx


# ---------- spectrum math ----------------------------------------------------
def _azimuthal_average(ps2d: np.ndarray,
                       kx: np.ndarray,
                       ky: np.ndarray,
                       nbins: int = NBINS) -> Tuple[np.ndarray, np.ndarray]:
    k   = np.sqrt(kx * kx + ky * ky)
    k_min, k_max = k.min(), k.max() * (1.0 + 1e-9)
    edges = np.linspace(k_min, k_max, nbins + 1)

    idx  = np.digitize(k.ravel(), edges) - 1
    Pk   = np.bincount(idx, weights=ps2d.ravel(), minlength=nbins)
    Nk   = np.bincount(idx, minlength=nbins)
    Pk  /= np.maximum(Nk, 1)

    k_centers = 0.5 * (edges[:-1] + edges[1:])
    return k_centers, Pk


def compute_angle_spectrum(phi: np.ndarray,
                           dx: float,
                           nbins: int = NBINS) -> Tuple[np.ndarray, np.ndarray]:
    if dx <= 0.0 or not np.isfinite(dx):
        raise ValueError(f"Positive dx required, got dx={dx}")

    phi = phi.astype(float, copy=False) - phi.mean()

    ny, nx = phi.shape
    fft2   = np.fft.fft2(phi)
    ps2d   = np.abs(fft2)**2 * (dx**2) / (nx * ny)

    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing="xy")

    return _azimuthal_average(ps2d, kx, ky, nbins=nbins)


# ---------- convenience ------------------------------------------------------
def spectrum_from_file(path: Path,
                       nbins: int = NBINS) -> Tuple[np.ndarray, np.ndarray]:
    phi, dx = load_phi_map(path)
    return compute_angle_spectrum(phi, dx, nbins=nbins)


def batch_spectrum(paths: Sequence[Path],
                   nbins: int = NBINS) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for p in paths:
        try:
            out[p.stem] = spectrum_from_file(p, nbins=nbins)
        except Exception as exc:
            print(f"[WARN] Skipping {p.name}: {exc}", file=sys.stderr)
    return out


# ---------- plotting helpers -------------------------------------------------
def _plot_reference(ax: plt.Axes,
                    k: np.ndarray,
                    Pk: np.ndarray,
                    slope: float = REF_SLOPE,
                    frac: float = 0.3) -> None:
    """
    Overlay a power-law line k^slope scaled to pass through
    the spectrum at a reference point.

    Parameters
    ----------
    ax    : existing matplotlib Axes
    k,Pk  : data from one spectrum (for scaling)
    slope : theoretical exponent (default -5/3)
    frac  : choose anchor index around len(k)*frac
    """
    if k.size == 0:
        return
    anchor = int(max(min(len(k) * frac, len(k) - 1), 0))
    k0, P0 = k[anchor], Pk[anchor]
    coeff  = P0 / (k0 ** slope)
    k_line = np.array([k.min(), k.max()])
    ax.loglog(k_line, coeff * (k_line**slope)*15.7, "--", color="black",
              label=rf"$k^{{{slope:.2f}}}$")


# ---------- CLI / demo -------------------------------------------------------
def _demo(paths: Sequence[str]) -> None:
    if not paths:
        print(__doc__)
        return

    spectra = batch_spectrum([Path(p) for p in paths])
    if not spectra:
        print("[ERROR] No spectra computed.", file=sys.stderr)
        return

    fig, ax = plt.subplots()
    # plot data
    first_spec = None
    for name, (k, Pk) in spectra.items():
        ax.loglog(k, Pk, label=name)
        if first_spec is None and k.size:
            first_spec = (k, Pk)

    ax.set_xlabel(r"$k_\perp$ [rad$^{-1}$]")
    ax.set_ylabel(r"$P_\phi(k_\perp)$")

    # reference slope
    if first_spec is not None:
        _plot_reference(ax, *first_spec, slope=REF_SLOPE)

    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig_angle_spectrum.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    _demo(sys.argv[1:])
