#!/usr/bin/env python3
"""
make_cube.py  –  build a Kolmogorov Bz + log-normal ne slab
Written 2025-04-18
Requires: turbustat, numpy, h5py
"""
import numpy as np, h5py
from turbustat.simulator import make_3dfield

# ---------- user knobs ------------------------------------------------
N         = 512          # grid points per axis
L_pc      = 100.0        # physical box size
B_rms     = 5.0          # μG (sets σ_RM)
ne_mean   = 0.10         # cm⁻³
ln_sigma  = 0.5          # dex dispersion of log-normal ne
seed_B, seed_ne = 42, 57
# ---------------------------------------------------------------------

dx = L_pc / N

print("Generating Kolmogorov B_z …")
Bz = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=seed_B)
Bz -= Bz.mean()
Bz *= B_rms / Bz.std()

print("Generating log-normal n_e …")
ln_ne = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=seed_ne)
ne    = np.exp(np.log(ne_mean) + ln_sigma * ln_ne/ln_ne.std())

with h5py.File("cube.h5", "w") as f:
    f["Bz"], f["ne"] = Bz.astype("f4"), ne.astype("f4")
    f.attrs["dx_pc"] = dx
    f.attrs["comment"] = (
        f"Kolmogorov fBm cube (B_rms={B_rms} μG, <ne>={ne_mean} cm⁻³)"
    )
print(f"cube.h5 saved  (N={N},  dx={dx:.3f} pc)")
