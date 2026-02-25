import numpy as np
import pyvista as pv

bcc_path = "synthetic_bcc.vtk"
w_path   = "synthetic_w.vtk"

dens_name = "dens"
bx_name, by_name, bz_name = "bcc3", "bcc1", "bcc2"

def to_cell(ds, name):
    if name in ds.cell_data: return ds
    if name in ds.point_data: return ds.point_data_to_cell_data()
    raise KeyError(f"{name} not found")

def cell_array(ds, name):
    if name in ds.cell_data: return ds.cell_data[name]
    if name in ds.point_data: return ds.point_data[name]
    raise KeyError(f"{name} not found")

def reshape_cell(ds, arr):
    nx, ny, nz = map(int, ds.dimensions)     # points dims
    return np.asarray(arr).reshape((nx-1, ny-1, nz-1), order="F")

bcc = pv.read(bcc_path)
w   = pv.read(w_path)

w = to_cell(w, dens_name)
dens = reshape_cell(w, cell_array(w, dens_name))

# ensure bcc fields are in cell_data
if any(n not in bcc.cell_data and n in bcc.point_data for n in (bx_name, by_name, bz_name)):
    bcc = bcc.point_data_to_cell_data()

bx = reshape_cell(bcc, cell_array(bcc, bx_name))
by = reshape_cell(bcc, cell_array(bcc, by_name))
bz = reshape_cell(bcc, cell_array(bcc, bz_name))

for name, arr in [("dens", dens), (bx_name, bx), (by_name, by), (bz_name, bz)]:
    print(f"{name}: mean={arr.mean():.6e}  std={arr.std():.6e}")
