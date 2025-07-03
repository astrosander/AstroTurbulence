#!/usr/bin/env python3
"""
bfield_structure_function.py
============================

Draw the transverse structure function of the *line-of-sight* magnetic
field  B_z(x,y,z)  for two data cubes:

    • left panel  – Athena snapshot
    • right panel – synthetic Kolmogorov cube (powerbox)

Everything is hard-wired: just edit the filenames below and run

    python bfield_structure_function.py
"""

from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import rfftn, irfftn, fftshift

# ────────── file paths & basic settings ───────────────────────────────
CUBE_ATHENA    = "ms01ma08.mhd_w.00300.vtk.h5"
CUBE_SYNTHETIC = "synthetic_tuned.h5"

NBINS     = 600        # radial bins for SF
Z_SAMPLE  = "mid"     # "mid"  → take central z-slice
                      # int    → take that slice index
DX_FALLBACK = 1.0
# ──────────────────────────────────────────────────────────────────────


def _grid_spacing(coord1d):
    uniq = np.unique(coord1d)
    step = np.diff(np.sort(uniq))
    step = step[step > 0]
    return float(np.median(step)) if step.size else DX_FALLBACK


def radial_structure_function_2d(field2d, dx, nbins=60):
    """
    Isotropic second-order SF  D(R)=⟨(f(x+R)−f(x))²⟩  for a 2-D map.
    """
    f = field2d - field2d.mean()
    ac = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    ac = fftshift(ac)
    D  = 2.0 * f.var() - 2.0 * ac
    D[D < 0] = 0

    ny, nx = f.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(x, y) * dx

    r_min, r_max = dx*1e-3, R.max()*0.4
    bins     = np.logspace(np.log10(r_min), np.log10(r_max), nbins+1)
    sumD, _  = np.histogram(R, bins=bins, weights=D)
    counts, _= np.histogram(R, bins=bins)
    D_R      = sumD / np.maximum(counts, 1)
    R_cent   = 0.5*(bins[1:]+bins[:-1])
    mask     = counts > 0
    return R_cent[mask], D_R[mask]


def load_bz_slice(cubefile):
    with h5py.File(cubefile, "r") as f:
        bz = f["k_mag_field"]
        nz = bz.shape[2]
        zidx = nz//2 if Z_SAMPLE == "mid" else int(Z_SAMPLE)
        slice2d = bz[:, :, zidx]

        dx = _grid_spacing(f["x_coor"][:, 0, 0]) if "x_coor" in f else DX_FALLBACK
    return slice2d.astype(np.float32), dx


# ────────── compute SFs ───────────────────────────────────────────────
bz_a, dx_a = load_bz_slice(Path(CUBE_ATHENA))
R_a, D_a   = radial_structure_function_2d(bz_a, dx_a, NBINS)

bz_s, dx_s = load_bz_slice(Path(CUBE_SYNTHETIC))
R_s, D_s   = radial_structure_function_2d(bz_s, dx_s, NBINS)

# Kolmogorov reference slope 2/3
ref_a = D_a[5] * (R_a / R_a[1])**(2/3)
ref_s = (D_s[1]+D_s[2])/2 * (R_s / R_s[1])**(2/3)

# ────────── plot ──────────────────────────────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

axL.loglog(R_a, D_a, label="Athena B_z slice")
axL.loglog(R_a, ref_a, "--", label=r"$\propto R^{2/3}$")
axL.set(title="Athena", xlabel="R (code units)", ylabel=r"$D_{B_z}(R)$")
axL.legend(frameon=False, fontsize=8)

axR.loglog(R_s, D_s, color="tab:green", label="Synthetic B_z slice")
axR.loglog(R_s, ref_s, "--", color="k")
axR.set(title="Synthetic Kolmogorov cube", xlabel="R (code units)")
axR.legend(frameon=False, fontsize=8)

fig.suptitle("Magnetic-field structure function  (z-slice, B$_z$)")
fig.tight_layout()
fig.savefig('figures/fig5_bfield_structure_function.pdf',  bbox_inches='tight')
plt.show()
