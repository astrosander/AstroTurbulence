#!/usr/bin/env python
"""
Generate a Kolmogorov 3-D cube with TurbuStat (no k=0 singularities),
plus a log-normal ne field that shares the same spectrum.
"""
import numpy as np, h5py
from turbustat.simulator import make_3dfield   # Kolmogorov fBm
N   = 512                  # gives ~2 dex of inertial range
Lpc = 100.0
dx  = Lpc / N

print("Making Bz, Kolmogorov P(k) ∝ k^-11/3 …")
# the simulator's power-law index is the 3-D spectral index
Bz = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=42)
Bz -= Bz.mean();  Bz /= Bz.std()            # 1 μG rms

print("Making ne, log-normal with same spectrum …")
lnne = make_3dfield(N, powerlaw=11/3, amp=1.0, randomseed=57)
ne   = np.exp(np.log(0.03) + 0.5*lnne/lnne.std())    # ⟨ne⟩≈0.03 cm⁻³

with h5py.File("cube.h5", "w") as f:
    f["Bz"], f["ne"] = Bz.astype("f4"), ne.astype("f4")
    f.attrs["dx_pc"] = dx
print("cube.h5 ready.")
