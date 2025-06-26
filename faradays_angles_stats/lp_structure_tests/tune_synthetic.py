#!/usr/bin/env python3
"""
tune_synthetic.py
-----------------

1.  Load a *reference* cube (e.g. Athena/PLUTO)   →  σΦ_ref
2.  Load a *synthetic* cube (power-law)           →  σΦ_syn
3.  Rescale either B_z *or* n_e in the synthetic cube so that
        σΦ_syn  →  σΦ_ref
4.  Save a new file  synthetic_tuned.h5  with the same dataset layout.

Edit the three PATHS just below and run:

    python tune_synthetic.py
"""

from pathlib import Path
import h5py
import numpy as np

# ───────────────────── PATHS (EDIT!) ────────────────────────────────
REF_CUBE   = Path("ms01ma08.mhd_w.00300.vtk.h5")   # Athena snapshot
SYN_CUBE   = Path("synthetic_powerbox.h5")         # raw synthetic file
OUT_CUBE   = Path("synthetic_tuned.h5")            # output filename
SCALE_WHAT = "ne"          # "bz"  or  "ne"
# ────────────────────────────────────────────────────────────────────


def _spacing(arr1d):
    uniq = np.unique(arr1d)
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def load_cube(path):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float32)
        bz = f["k_mag_field"][:].astype(np.float32)
        dx = _spacing(f["x_coor"][:, 0, 0]) if "x_coor" in f else 1.0
        dz = _spacing(f["z_coor"][0, 0, :]) if "z_coor" in f else 1.0
    return ne, bz, dx, dz


def rm_rms(ne, bz, dz):
    phi = (ne * bz).sum(axis=2) * dz
    return float(phi.std(ddof=0))


# ────────────────────────────────────────────────────────────────────
# 1. reference σΦ
ref_ne, ref_bz, _, ref_dz = load_cube(REF_CUBE)
sigma_ref = rm_rms(ref_ne, ref_bz, ref_dz)
print(f"σΦ_ref  = {sigma_ref:.3e}")

# 2. synthetic σΦ
syn_ne, syn_bz, _, syn_dz = load_cube(SYN_CUBE)
sigma_syn = rm_rms(syn_ne, syn_bz, syn_dz)
print(f"σΦ_syn  = {sigma_syn:.3e}")

print("Means:")
syn_bz+= ref_bz.mean()

print(ref_ne.mean(), ref_bz.mean())
print(syn_ne.mean(), syn_bz.mean())

# 3. scaling factor
factor = sigma_ref / sigma_syn

print(f"Scaling factor = {factor:.3f}  (will apply to {SCALE_WHAT})")

if SCALE_WHAT.lower() == "bz":
    syn_bz *= factor
elif SCALE_WHAT.lower() == "ne":
    syn_ne *= factor
else:
    raise ValueError("SCALE_WHAT must be 'bz' or 'ne'")

# 4. write tuned cube
print(f"writing {OUT_CUBE} …")
with h5py.File(OUT_CUBE, "w") as h5_in, h5py.File(SYN_CUBE, "r") as src:
    # copy coordinate grids verbatim
    for key in src:
        if key in ("gas_density", "k_mag_field"):
            continue
        h5_in.create_dataset(key, data=src[key], compression="gzip")

    h5_in.create_dataset("gas_density", data=syn_ne, compression="gzip")
    h5_in.create_dataset("k_mag_field", data=syn_bz, compression="gzip")

print("done ✔")