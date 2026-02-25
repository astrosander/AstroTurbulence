import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

from main import isotropic_power_spectrum_2d, unit_polarization


#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 18

# =========================
# Helpers
# =========================
def _get_structured_array(mesh: pv.ImageData, name: str) -> np.ndarray:
    """
    Return a (nx, ny, nz) numpy array for scalar field `name` stored on VTK ImageData.
    Works for point or cell data; prefers point_data if both exist.
    Reshapes in Fortran order to match VTK memory layout.
    """
    nx, ny, nz = mesh.dimensions

    if name in mesh.point_data:
        return np.asarray(mesh.point_data[name]).reshape(
            (nx, ny, nz), order="F"
        ).astype(np.float64)

    if name in mesh.cell_data:
        carr = np.asarray(mesh.cell_data[name]).reshape(
            (nx - 1, ny - 1, nz - 1), order="F"
        )
        cx, cy, cz = carr.shape
        out = np.empty((cx + 1, cy + 1, cz + 1), dtype=np.float64)

        out[:cx, :cy, :cz] = carr
        out[cx, :cy, :cz] = carr[cx - 1]
        out[:cx, cy, :cz] = carr[:, cy - 1]
        out[:cx, :cy, cz] = carr[:, :, cz - 1]
        out[cx, cy, :cz] = carr[cx - 1, cy - 1]
        out[cx, :cy, cz] = carr[cx - 1, :, cz - 1]
        out[:cx, cy, cz] = carr[:, cy - 1, cz - 1]
        out[cx, cy, cz] = carr[cx - 1, cy - 1, cz - 1]
        return out

    raise KeyError(f"Field '{name}' not found.")


def _find_field(mesh: pv.ImageData, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in mesh.point_data or c in mesh.cell_data:
            return c
    # print(mesh.point_data.keys())
    all_fields = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
    # print(all_fields)
    for f in all_fields:
        fl = f.lower()
        for c in candidates:
            if c.lower() in fl or fl in c.lower():
                return f
    return None


# =========================
# USER SETTINGS
# =========================
nbins = 100

# List of VTK files to process
vtk_files = [
    Path("/home/amelnich/orcd/scratch/athena_runs/ma08_ms1/ms1ma12_256.mhd_w.00071.vtk"),
    Path("/home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_bcc.00001.vtk"),
]

# Create output directory
output_dir = Path("npz")
output_dir.mkdir(exist_ok=True)

# =========================
# MAIN
# =========================
for idx, vtk_b_path in enumerate(vtk_files, 1):
    # Generate output filename based on input filename
    output_name = f"directional_{vtk_b_path.stem}.npz"
    output_npz = output_dir / output_name

    print(f"\n{'='*60}")
    print(f"Processing file {idx}/{len(vtk_files)}: {vtk_b_path}")
    print(f"{'='*60}")

    if not vtk_b_path.exists():
        print(f"Warning: File {vtk_b_path} not found, skipping...")
        continue

    print("Reading VTK:", vtk_b_path)
    b_mesh = pv.read(str(vtk_b_path))

    print("\nAll VTK keys:")
    print(" point_data keys:", list(b_mesh.point_data.keys()))
    print(" cell_data keys :", list(b_mesh.cell_data.keys()))
    print()

    bx_candidates = ["bcc3"]#["bcc1"]
    by_candidates = ["bcc1"]#["bcc2"]
    bz_candidates = ["bcc2"]#["bcc3"]

    bx_name = _find_field(b_mesh, bx_candidates)
    by_name = _find_field(b_mesh, by_candidates)
    bz_name = _find_field(b_mesh, bz_candidates)

    if bx_name is None or by_name is None or bz_name is None:
        print("Available fields:")
        print(" point_data:", list(b_mesh.point_data.keys()))
        print(" cell_data :", list(b_mesh.cell_data.keys()))
        print(f"Error: Could not identify bx/by/bz fields for {vtk_b_path}, skipping...")
        continue

    print("Using fields:")
    print(" bx =", bx_name)
    print(" by =", by_name)
    print(" bz =", bz_name)

    bx = _get_structured_array(b_mesh, bx_name)
    by = _get_structured_array(b_mesh, by_name)
    bz = _get_structured_array(b_mesh, bz_name)

    # =========================
    # Statistics
    # =========================
    print("\nField statistics:")
    for name, arr in [("bx", bx), ("by", by), ("bz", bz)]:
        mean_val = np.mean(arr)
        rms_val = np.sqrt(np.mean(arr**2))
        print(f" {name}: mean = {mean_val:.6e}, RMS = {rms_val:.6e}")

    # =========================
    # Intrinsic polarization and unit polarization
    # =========================
    # P_i(x,y) = sum_z (Bx + i By)^2
    P_i = np.sum((bx + 1j * by) ** 2, axis=2)

    # u_i(x,y) = P_i / |P_i|
    u_i = unit_polarization(P_i)

    # Mean subtraction to suppress the k=0 spike (same idea as before)
    u_i -= u_i.mean()

    # =========================
    # Box size
    # =========================
    bounds = b_mesh.bounds
    L = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    print("Lxy =", L)

    # =========================
    # Spectrum
    # =========================
    k, Pk = isotropic_power_spectrum_2d(u_i, Lxy=L, nbins=nbins)

    # =========================
    # Save data
    # =========================
    np.savez(
        output_npz,
        k=k,
        Pk=Pk,
        Lxy=L,
        field="u_i",
        bx_field=bx_name,
        by_field=by_name,
        bz_field=bz_name,
        vtk=str(vtk_b_path),
    )

    print(f"Saved: {output_npz}")

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")
