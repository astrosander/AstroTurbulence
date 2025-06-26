
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


ne_key="gas_density"
bz_key="k_mag_field"

with h5py.File(cube, "r") as f:
    ne = f[ne_key][:]
    bz = f[bz_key][:]

    dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0
    dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0

print(ne.mean(), ne.min(), ne.max())
print(bz.mean(), bz.min(), bz.max())
