#!/usr/bin/env python3
"""
wrap_saturation_test.py
=======================

Large-rotation (2π–ambiguity) experiment.

• Loads a single HDF5 cube (Athena or synthetic)
• Picks a list of long wavelengths λ such that λ² Φ_rms ≳ π
• Wraps the resulting polarisation angles modulo π   *or*   2π
• Computes the 2-D structure function D_φ(R,λ)
• Shows how D_φ saturates at the theoretical plateau
      π²/12  (for modulo-π)
      π²/6   (for modulo-2π)

Edit the PARAMETERS block and run

    python wrap_saturation_test.py
"""

from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import rfftn, irfftn, fftshift

# ───────── PARAMETERS ────────────────────────────────────────────────
CUBE_FILE   = "synthetic_tuned.h5"     # or Athena snapshot
LONG_WAVE   = (0.30, 0.50, 0.80)       # metres – adjust as needed
WRAP_TYPE   = "pi"                     # "pi"  or  "2pi"
NBINS       = 70
DX_FALLBACK = 1.0
# ─────────────────────────────────────────────────────────────────────


# ---------- helpers --------------------------------------------------
def _spacing(coord):
    uniq = np.unique(coord)
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    return float(np.median(d)) if d.size else DX_FALLBACK


def wrap(angle, mode="pi"):
    if mode == "pi":
        return (angle + np.pi/2) % np.pi - np.pi/2
    elif mode == "2pi":
        return (angle + np.pi) % (2*np.pi) - np.pi
    else:
        raise ValueError("mode must be 'pi' or '2pi'")


def structure_function_2d(field, dx, nbins=60):
    f  = field - field.mean()
    ac = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    D  = 2*f.var() - 2*fftshift(ac)
    D[D < 0] = 0.0

    ny, nx = f.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(x, y) * dx

    rmin, rmax = dx*1e-3, R.max()*0.4
    bins  = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    sumD, _ = np.histogram(R, bins=bins, weights=D)
    cnts, _ = np.histogram(R, bins=bins)
    D_R = sumD / np.maximum(cnts, 1)
    R_c = 0.5*(bins[1:]+bins[:-1])
    mask = cnts > 0
    return R_c[mask], D_R[mask]


# ---------- load cube & build RM map ---------------------------------
with h5py.File(Path(CUBE_FILE), "r") as f:
    ne = f["gas_density"][:].astype(np.float32)
    bz = f["k_mag_field"][:].astype(np.float32)
    dx = _spacing(f["x_coor"][:,0,0]) if "x_coor" in f else DX_FALLBACK
    dz = _spacing(f["z_coor"][0,0,:]) if "z_coor" in f else DX_FALLBACK

Phi = (ne * bz).sum(axis=2) * dz
Phi_rms = Phi.std()
print(f"Φ_rms = {Phi_rms:.3g}")

# ---------- saturation plateau value ---------------------------------
plateau = (np.pi**2) / (6 if WRAP_TYPE == "2pi" else 12)
label_plateau = r"$\pi^{2}/6$" if WRAP_TYPE=="2pi" else r"$\pi^{2}/12$"

# ---------- compute & plot ------------------------------------------
fig, ax = plt.subplots(figsize=(6,4.3))

for lam in LONG_WAVE:
    ang_map = wrap(lam**2 * Phi, WRAP_TYPE)
    R, Dphi = structure_function_2d(ang_map, dx, NBINS)
    ax.loglog(R, Dphi, label=rf"$\lambda={lam:.2f}\,$m")

ax.axhline(plateau, color="k", ls=":", lw=1.5, label=label_plateau)
ax.set_xlabel(r"$R$  (same units as $dx$)")
ax.set_ylabel(r"$D_{\varphi}(R,\lambda)$")
ax.set_title(f"Saturation in the wrapped–angle regime (mod {WRAP_TYPE})")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("wrap_saturation.pdf", bbox_inches="tight")
plt.show()
