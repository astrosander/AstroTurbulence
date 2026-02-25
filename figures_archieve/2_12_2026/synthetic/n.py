import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Input: single VTK file (w) -> density spectrum (2D isotropic)
# Only for:
#   /home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_w.00021.vtk
# ============================================================
w_path = "synthetic_w.vtk"
out_npz = "dens_spectrum_ms10ma08_512_w_00021.npz"
out_png = "dens_spectrum_ms10ma08_512_w_00021.png"

dens_name = "dens"
eps = 1e-30


def _as_cell_data(ds, name: str):
    """Ensure `name` is available as cell_data (convert from point_data if needed)."""
    if name in ds.cell_data:
        return ds
    if name in ds.point_data:
        return ds.point_data_to_cell_data()
    raise KeyError(f"{name} not found in cell_data or point_data")


def _reshape_cell(ds, arr):
    """Reshape flat cell array into (nx, ny, nz) with Fortran ordering."""
    dims = tuple(int(x) for x in ds.dimensions)  # point dims
    shape = (dims[0] - 1, dims[1] - 1, dims[2] - 1)  # cell dims
    return np.asarray(arr).reshape(shape, order="F")


# -------------------------
# Read density (w) file
# -------------------------
w = pv.read(w_path)
w = _as_cell_data(w, dens_name)
dens3d = _reshape_cell(w, w.cell_data[dens_name])  # (nx, ny, nz)

dx, dy, dz = (float(x) for x in w.spacing)
nx, ny, nz = dens3d.shape

# ------------------------------------------------------------
# 2D column density (LOS sum along z) then spectrum in (x,y)
# ------------------------------------------------------------
F = np.sum(dens3d * dz, axis=2)  # shape (nx, ny)

F_mean = float(np.mean(F))
F_std_pop = float(np.std(F, ddof=0))
F_std_samp = float(np.std(F, ddof=1))
print(f"[dens] F mean = {F_mean:.6e}, std(pop) = {F_std_pop:.6e}, std(sample) = {F_std_samp:.6e}")

# Remove mean before FFT (typical for spectra)
F0 = F - F_mean

# -------------------------
# k-space setup (2D)
# -------------------------
kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), indexing="ij")
K = np.sqrt(KX * KX + KY * KY)

kmax = float(K.max())
nbins = max(16, int(np.sqrt(nx * ny)))
k_edges = np.linspace(0.0, kmax, nbins + 1)
k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

# -------------------------
# Power + isotropic spectrum
# -------------------------
Fhat = np.fft.fft2(F0)
PF = np.abs(Fhat) ** 2
PF_s = np.fft.fftshift(PF)

kvals = K.ravel()
pvals = PF_s.ravel()

sums, _ = np.histogram(kvals, bins=k_edges, weights=pvals)
counts, _ = np.histogram(kvals, bins=k_edges)
E_k = sums / np.maximum(counts, 1)

# -------------------------
# Plot
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

axs[0].loglog(k_centers[1:], E_k[1:])
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"$E_{\Sigma\rho}(k)$")
axs[0].set_title("column density spectrum")

im = axs[1].imshow(
    np.log10(PF_s.T + eps),
    origin="lower",
    extent=[KX.min(), KX.max(), KY.min(), KY.max()],
    aspect="auto",
)
axs[1].set_xlabel(r"$k_x$")
axs[1].set_ylabel(r"$k_y$")
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label=r"$\log_{10} |\hat F|^2$")

fig.tight_layout()
fig.savefig(out_png)
plt.close(fig)

# -------------------------
# Save NPZ
# -------------------------
np.savez(
    out_npz,
    # field stats
    F_mean=F_mean,
    F_std_pop=F_std_pop,
    F_std_sample=F_std_samp,
    # spectrum products
    k_edges=k_edges,
    k_centers=k_centers,
    E_dens=E_k,
    PF_2d=PF_s,
    # k grids / spacings
    kx=np.fft.fftshift(kx),
    ky=np.fft.fftshift(ky),
    dx=dx,
    dy=dy,
    dz=dz,
    # provenance
    w_path=str(w_path),
    dens_name=str(dens_name),
)

print(f"[dens] wrote: {out_npz} and {out_png}")
