import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

# ===== Load Extracted Colormap =====
# Downloaded earlier as extracted_cmap_lut.npy
# lut = np.load("extracted_cmap_lut.npy")
# custom_cmap = ListedColormap(lut, name="extracted_cbmap")
custom_cmap="rainbow"

# ===== HDF5 Data Processing =====
BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

path = "synthetic_512.h5"
with h5py.File(path, "r") as f:
    bx_ds = f[BX_KEY]
    by_ds = f[BY_KEY]
    N = bx_ds.shape[0]
    z0 = N // 2

    # perpendicular magnetic field magnitude at midplane
    bperp_slice = np.sqrt(bx_ds[:, :, z0] ** 2 + by_ds[:, :, z0] ** 2)

    # Stokes parameters
    Q = np.zeros((N, N), dtype=np.float32)
    U = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        bxk = bx_ds[:, :, k]
        byk = by_ds[:, :, k]
        Q += bxk * bxk - byk * byk
        U += 2.0 * bxk * byk

Qn = Q / np.max(np.abs(Q))
Un = U / np.max(np.abs(U))

# ===== Visualization =====
fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

# Magnetic field slice
im0 = axs[0].imshow(bperp_slice, origin="lower", cmap=custom_cmap)
axs[0].set_title("Synthetic Magnetic Field")
fig.colorbar(im0, ax=axs[0])

# Stokes Q
im1 = axs[1].imshow(Qn, origin="lower", vmin=-1, vmax=1, cmap=custom_cmap)
axs[1].set_title("Stokes parameter Q")
fig.colorbar(im1, ax=axs[1])

# Stokes U
im2 = axs[2].imshow(Un, origin="lower", vmin=-1, vmax=1, cmap=custom_cmap)
axs[2].set_title("Stokes parameter U")
fig.colorbar(im2, ax=axs[2])

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("stokes.png", dpi=300)
plt.show()