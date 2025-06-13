#!/usr/bin/env python
"""
make_cube.py  –  build a Kolmogorov B-field and a log-normal ne cube
--------------------------------------------------------------------
Output: cube.h5  with datasets  Bz  and  ne,  plus   dx_pc  attribute.
Requires: turbustat, numpy, h5py
"""
import numpy as np, h5py
from turbustat.simulator import make_3dfield   # robust fBm generator

# --------------------- parameters you may tweak --------------------
N       = 512          # grid points per side   (≤384 if RAM < 16 GB)
L_pc    = 100.0        # physical box length
B_rms   = 5.0          # μG  – stronger field → larger σ_RM
ne_mean = 0.10         # cm⁻³
ln_sigma= 0.5          # dex dispersion of log-normal ne
rng1, rng2 = 42, 57    # seeds for B and ne
# ------------------------------------------------------------------

dx = L_pc / N

print("Generating Kolmogorov Bz …")
Bz = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=rng1)
Bz -= Bz.mean()
Bz *= B_rms / Bz.std()               # rms = B_rms μG

print("Generating log-normal ne …")
ln_ne = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=rng2)
ne = np.exp(np.log(ne_mean) + ln_sigma * ln_ne / ln_ne.std())

with h5py.File("cube.h5", "w") as f:
    f["Bz"] = Bz.astype("f4")
    f["ne"] = ne.astype("f4")
    f.attrs["dx_pc"] = dx
    f.attrs["comment"] = (
        "fBm Kolmogorov cube (B_rms = %.1f μG, <ne>=%.2f cm^-3)" %
        (B_rms, ne_mean)
    )
print("cube.h5 written   (N = %d,  dx = %.3f pc)" % (N, dx))
