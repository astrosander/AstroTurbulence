#!/usr/bin/env python3
"""
lambda_scaling_demo.py
======================

Demonstrate the λ⁴ amplitude scaling of the angle–structure function
D_φ(R, λ) predicted by the Lazarian–Pogosyan Faraday-screen theory.

* Cube path, wavelength list, and reference R₀ are hard-coded.
* Produces a single matplotlib window with two panels:

  left  – D_φ(R, λ) curves (same as earlier)
  right – amplitude at R₀ versus λ + a ∝λ⁴ reference line
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import rfftn, irfftn, fftshift

# ----------------------------------------------------------------------
# Parameters you may want to tweak
# ----------------------------------------------------------------------
CUBE_FILE   = "synthetic_kolmogorov.h5"
# CUBE_FILE   = "ms01ma08.mhd_w.00300.vtk.h5"
LAMBDAS     = np.array([0.04, 0.06, 0.08, 0.11, 0.16, 0.21])  # meters
REF_BIN     = 1        # index into the radial bin array (0 would be R≈0)
NBINS       = 600       # radial bin count
DX_DEFAULT  = 1.0      # if the cube lacks coordinate grids
# ----------------------------------------------------------------------


def _axis_spacing(coord1d):
    uniq = np.unique(coord1d.ravel())
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    return float(np.median(d)) if d.size else DX_DEFAULT


def structure_function_2d(field, dx=1.0, nbins=60):
    """
    Isotropic second-order structure function via FFT
    (no SciPy dependency).
    """
    f   = field - field.mean()
    acf = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    acf = fftshift(acf)
    D   = 2.0 * f.var() - 2.0 * acf
    D[D < 0] = 0

    # radial distance array
    ny, nx = field.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(x, y) * dx

    r_min = dx * 1e-3
    r_max = R.max()*0.4          # stay away from the edges
    bins  = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)

    # weighted sum of D in each annulus
    sum_D, _ = np.histogram(R, bins=bins, weights=D)
    counts, _ = np.histogram(R, bins=bins)

    # avoid divide-by-zero
    D_R = sum_D / np.maximum(counts, 1)
    R_cent = 0.5 * (bins[1:] + bins[:-1])

    mask = counts > 0
    return R_cent[mask], D_R[mask]



# ----------------------------------------------------------------------
# 1. read cube & build Φ(X)
# ----------------------------------------------------------------------
cube = Path(CUBE_FILE)
with h5py.File(cube, "r") as f:
    ne = f["gas_density"][:]
    bz = f["k_mag_field"][:]
    dx = _axis_spacing(f["x_coor"][:, 0, 0]) if "x_coor" in f else DX_DEFAULT
    dz = _axis_spacing(f["z_coor"][0, 0, :]) if "z_coor" in f else DX_DEFAULT

Phi = (ne * bz).sum(axis=2) * dz
R, D_phi = structure_function_2d(Phi, dx=dx, nbins=NBINS)

# ----------------------------------------------------------------------
# 2. compute D_φ(R, λ) and record amplitude at R₀
# ----------------------------------------------------------------------
amp = []
for lam in LAMBDAS:
    D_varphi = 0.5 * (1.0 - np.exp(-2.0 * lam**4 * D_phi))
    amp.append(D_varphi[REF_BIN])

amp = np.array(amp)

# ----------------------------------------------------------------------
# 3. reference λ⁴ law through the first point
# ----------------------------------------------------------------------
norm = amp[0] / LAMBDAS[0]**4
lam4_ref = norm * LAMBDAS**4

# ----------------------------------------------------------------------
# 4. plot
# ----------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

# left: family of D_φ(R,λ) curves
for lam in LAMBDAS:
    D_varphi = 0.5 * (1.0 - np.exp(-2.0 * lam**4 * D_phi))
    ax_left.loglog(R, D_varphi, label=rf"λ={lam:.2f} m")
ax_left.set(xlabel="R  (same units as dx)",
            ylabel=r"$D_{\varphi}(R,\lambda)$",
            title="Angle structure functions")
ax_left.legend(fontsize=7, frameon=False)

# right: amplitude vs λ
ax_right.loglog(LAMBDAS, amp, "o-", label=r"$D_\varphi(R_0,\lambda)$")
ax_right.loglog(LAMBDAS, lam4_ref, "--", label=r"$\propto\lambda^{4}$")
ax_right.set(xlabel="λ  [m]", ylabel="amplitude at R₀",
             title=fr"Scaling at R bin {REF_BIN}  (R≈{R[REF_BIN]:.2f})")
ax_right.legend(frameon=False)

fig.suptitle("λ⁴ amplitude scaling test")
fig.tight_layout()
fig.savefig('fig4_lambda_scaling_synthetic_kolmogorov.pdf', bbox_inches='tight')
plt.show()
