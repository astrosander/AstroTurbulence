import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Path to your file
filename = 'ms01ma08.mhd_w.00300.vtk.h5'

# Sampling step
step = 4

with h5py.File(filename, 'r') as f:
    # Base shape from gas_density
    shape = f['gas_density'].shape
    idx = tuple(slice(0, dim, step) for dim in shape)

    # Pre-slice coordinate grids
    x = f['x_coor'][idx].ravel()
    y = f['y_coor'][idx].ravel()
    z = f['z_coor'][idx].ravel()

    keys = [
        'gas_density',
        'i_mag_field',
        'i_velocity',
        'j_mag_field',
        'j_velocity',
        'k_mag_field',
        'k_velocity',
        'x_coor',
        'y_coor',
        'z_coor'
    ]

    # Set up figure with a 3×4 grid (12 slots; 2 will remain empty)
    fig = plt.figure(figsize=(16, 12))
    n_plots = len(keys)
    ncols = 4
    nrows = int(np.ceil(n_plots / ncols))

    for i, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        C = f[key][idx].ravel()

        sc = ax.scatter(x, y, z,
                        c=C,
                        cmap='viridis',
                        marker='o',
                        s=4,
                        linewidth=0)

        ax.set_title(key, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.05)

    # Optionally hide any unused subplots
    for j in range(n_plots, nrows * ncols):
        fig.add_subplot(nrows, ncols, j+1).axis('off')

    plt.tight_layout()
    plt.show()
