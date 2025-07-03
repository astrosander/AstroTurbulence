#!/usr/bin/env python3
"""
wrap_saturation_fixed.py
------------------------

Demonstrate π–wrapping saturation of the angle structure function.

• Works for any cube with datasets 'gas_density', 'k_mag_field',
  optional 'x_coor', 'z_coor'.
• Plots Dφ(R,λ) for several long wavelengths.
• Shows the correct plateau π²/6.

Run:
    python wrap_saturation_fixed.py synthetic_tuned.h5 0.30 0.50 0.80
"""

import sys, math
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import rfftn, irfftn, fftshift

# ───────── utilities ─────────────────────────────────────────────────
DX_FALLBACK = 1.0
NBINS       = 70

def grid_spacing(arr):
    """Median positive spacing or fallback."""
    if arr is None or arr.size < 2:  # coord absent
        return DX_FALLBACK
    d = np.diff(np.sort(arr.astype(float)))
    d = d[d > 0]
    return float(np.median(d)) if d.size else DX_FALLBACK

def structure_function_2d(field, dx, nbins=60):
    f  = field - field.mean()
    ac = irfftn(np.abs(rfftn(f))**2, s=f.shape) / f.size
    sf = 2.0 * f.var() - 2.0 * fftshift(ac)
    sf[sf < 0] = 0.0               # numerical guard

    ny, nx = f.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(x, y) * dx

    rmin, rmax = dx*1e-3, R.max()*0.4
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    num, _ = np.histogram(R, bins=bins, weights=sf)
    cnt, _ = np.histogram(R, bins=bins)
    sfR = num / np.maximum(cnt, 1)
    Rcent = 0.5*(bins[1:] + bins[:-1])
    mask = cnt > 0
    return Rcent[mask], sfR[mask]

def wrap_pi(angle):
    """Wrap each angle into (-π/2, π/2]."""
    return (angle + np.pi/2.) % np.pi - np.pi/2.

# ───────── main routine ──────────────────────────────────────────────
def main():
    cube = Path("synthetic_tuned.h5")

    # lambdas = (0.30, 0.50, 0.80, 1.0, 1.5)

    lambdas = ()

    x=0.3
    while x<=0.8:
    	print(x)
    	lambdas += (round(x, 4),)
    	x+=0.05



    # --- load cube and build Φ map
    with h5py.File(cube, "r") as f:
        ne = f["gas_density"][:].astype(np.float32)
        bz = f["k_mag_field"][:].astype(np.float32)
        dx = grid_spacing(f["x_coor"][:,0,0]) if "x_coor" in f else DX_FALLBACK
        dz = grid_spacing(f["z_coor"][0,0,:]) if "z_coor" in f else DX_FALLBACK

    Phi = (ne * bz).sum(axis=2) * dz
    Phi_rms = Phi.std()
    print(f"Φ_rms = {Phi_rms:.3g}")

    sigma_Phi = Phi.std(ddof=0)          # RMS rotation measure
    lambda_crit = (np.pi**2 / (12*sigma_Phi**2))**0.25
    print("sigma_Phi =", sigma_Phi)
    print("λ_crit =", lambda_crit, "metres")

    plateau = math.pi**2 / 6.0      # correct mod-π saturation

    # --- plot
    fig, ax = plt.subplots(figsize=(6, 4.3))
    for lam in lambdas:
        ang = wrap_pi(lam**2 * Phi)
        R, Dphi = structure_function_2d(ang, dx, NBINS)
        ax.loglog(R, Dphi, label=rf"$\lambda={lam:.2f}\,$m")

    ax.axhline(plateau, lw=1.5, ls=":", color="k", label=r"$\pi^2/6$")
    ax.set_xlabel(r"$R$  (same units as $dx$)")
    ax.set_ylabel(r"$D_{\varphi}(R,\lambda)$")
    ax.set_title("Angle-SF saturation for wrapped field (mod π)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/wrap_saturation_fixed.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
