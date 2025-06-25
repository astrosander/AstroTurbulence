import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

# parameters
filename  = "solenoidal.h5"
grid_size = 512
down      = 2   # optional downsampling factor to reduce points

# precompute the Y/Z coordinates once
y = np.arange(0, grid_size, down)
z = np.arange(0, grid_size, down)

def flatten_chunk(i0, i1):
    # every worker re-opens the file
    with h5py.File(filename, "r") as hf:
        # read only the slice you need, with downsampling
        block = hf["u"][i0:i1, :grid_size:down, :grid_size:down, 0]
        data = block[:]  # bring it into a plain numpy array

    # build X-coordinates for just this i0:i1 block
    x = np.arange(i0, i1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    return xx.ravel(), yy.ravel(), zz.ravel(), data.ravel()

# carve your grid into ~8–16 chunks
n_chunks = 16
slices   = np.linspace(0, grid_size, n_chunks + 1, dtype=int)
idxs     = list(zip(slices[:-1], slices[1:]))

# parallel flattening (processes)
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(flatten_chunk)(i0, i1) for i0, i1 in idxs
)

# stitch back together
x_flat = np.concatenate([r[0] for r in results])
y_flat = np.concatenate([r[1] for r in results])
z_flat = np.concatenate([r[2] for r in results])
u_flat = np.concatenate([r[3] for r in results])

# plotting
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")
sc  = ax.scatter(x_flat, y_flat, z_flat, c=u_flat, cmap="plasma",
                 alpha=0.6, s=1)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title(f"3D Velocity Field (x-comp), grid={grid_size}³, down={down}")
plt.colorbar(sc, label="Velocity")
plt.tight_layout()
plt.show()
