import h5py
import numpy as np
from pathlib import Path

cube = Path("synthetic_kolmogorov.h5")
# cube = Path("ms01ma08.mhd_w.00300.vtk.h5")

cube = cube.expanduser()

def _axis_spacing(coord1d, name="axis"):
    uniq = np.unique(coord1d.ravel())
    d    = np.diff(np.sort(uniq))
    d    = d[d > 0]
    if d.size:
        return float(np.median(d))
    return 1.0

ne_key = "gas_density"
bz_key = "k_mag_field"

with h5py.File(cube, "r") as f:
    ne = f[ne_key][:]
    bz = f[bz_key][:]

    dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0
    dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0

# RMS calculations
ne_rms = np.sqrt(np.mean(ne**2))
bz_rms = np.sqrt(np.mean(bz**2))

print(f"ne: mean={ne.mean()}, min={ne.min()}, max={ne.max()}, rms={ne_rms}")
print(f"bz: mean={bz.mean()}, min={bz.min()}, max={bz.max()}, rms={bz_rms}")



# ne: mean=1.0000004768371582, min=0.8572601079940796, max=1.1943234205245972, rms=1.000738501548767
# bz: mean=0.11952289938926697, min=-0.10909131914377213, max=0.3335864245891571, rms=0.12989802658557892
