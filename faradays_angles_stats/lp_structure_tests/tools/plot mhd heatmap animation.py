import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def infer_edges_from_centers(centers):
    centers = np.asarray(centers).astype(float)
    d = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # extrapolate outer edges
    edges[0]  = centers[0]  - (edges[1]  - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges

def save_B_frames(
    filename,
    out_dir="frames",
    component="mag",       # "mag", "Bx", "By", "Bz"
    axis="z",              # slice direction: "z" → sweep k=0..nz-1
    dpi=150,
    cmap="viridis",
    interpolation="nearest"
):
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(filename, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2, 1, 0)  # (x,y,z)
        By = f["j_mag_field"][:].transpose(2, 1, 0)
        Bz = f["k_mag_field"][:].transpose(2, 1, 0)
        x_centers = np.squeeze(f["x_coor"][0, 0, :])

    nx, ny, nz = Bx.shape
    assert nx == ny == nz, "Assumes cubic grid."

    # choose volume to visualize
    if component.lower() == "mag":
        vol = np.sqrt(Bx**2 + By**2 + Bz**2)
        cbar_label = r"|B|"
    elif component.lower() == "bx":
        vol = Bx; cbar_label = "Bx"
    elif component.lower() == "by":
        vol = By; cbar_label = "By"
    elif component.lower() == "bz":
        vol = Bz; cbar_label = "Bz"
    else:
        raise ValueError('component must be "mag", "Bx", "By", or "Bz"')

    # fixed color scale across all frames (prevents flicker)
    vmin, vmax = np.nanmin(vol), np.nanmax(vol)

    # build physical extent (assume same spacing for x & y)
    x_edges = infer_edges_from_centers(x_centers) if x_centers.size == nx else x_centers
    extent_xy = [x_edges[0], x_edges[-1], x_edges[0], x_edges[-1]]

    # figure/axes once, update array each frame for speed
    fig = plt.figure(figsize=(6, 5), dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(vol[:, :, 0].T, origin="lower", extent=extent_xy,
                   aspect="equal", cmap=cmap, interpolation=interpolation,
                   vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    title = ax.set_title(f"{cbar_label} at z-index 0")

    # how many digits to zero-pad (256 -> 3)
    pad = max(3, len(str(nz - 1)))

    for k in range(nz):  # 0..255
        # slice through z; for other axes you could change here
        frame2d = vol[:, :, k]  # (x,y) at fixed z
        im.set_data(frame2d.T)
        title.set_text(f"{cbar_label} at z-index {k}")
        # save
        outfile = os.path.join(out_dir, f"frame_{k:0{pad}d}.png")
        plt.savefig(outfile, bbox_inches="tight")
        print(k)
    plt.close(fig)

save_B_frames(r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5", out_dir="frames", component="mag")

# --- usage ---
# save_B_frames("path/to/file.h5", out_dir="frames", component="mag")
