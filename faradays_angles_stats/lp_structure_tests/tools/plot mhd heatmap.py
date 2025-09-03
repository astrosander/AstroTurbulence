import h5py
import numpy as np
import matplotlib.pyplot as plt

def infer_edges_from_centers(centers):
    """Return cell-edge coordinates inferred from center coordinates."""
    centers = np.asarray(centers)
    d = np.diff(centers)
    # assume uniform spacing if diffs are ~constant, else use local half-steps
    if np.allclose(d, d[0], rtol=1e-4, atol=1e-12):
        dx = d[0]
        edges = np.empty(centers.size + 1, dtype=centers.dtype)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0]    = centers[0] - 0.5 * dx
        edges[-1]   = centers[-1] + 0.5 * dx
    else:
        # nonuniform: midpoint edges, extrapolate endpoints
        edges = np.empty(centers.size + 1, dtype=centers.dtype)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0]    = centers[0] - (edges[1] - centers[0])
        edges[-1]   = centers[-1] + (centers[-1] - edges[-2])
    return edges

def get_extent(x_like):
    """
    Build imshow extent [xmin,xmax,ymin,ymax] from a 1D coordinate array
    of centers or edges. Returns extent and inferred edges array.
    """
    x_like = np.asarray(x_like)
    if x_like.ndim != 1:
        # your file shows x_coor with shape (1,1,N); squeeze it first
        x_like = np.squeeze(x_like)
    # If we already have edges (N+1), use them. If centers (N), infer edges.
    if x_like.size >= 2 and np.all(np.diff(x_like) > 0):
        # accept as centers or edges; decide by length vs. data size later
        edges = x_like
        # If this is centers (will be handled by caller), they’ll pass centers
        # and we’ll convert. Here, just return min/max.
        return [edges[0], edges[-1], edges[0], edges[-1]], edges
    else:
        raise ValueError("Coordinate array must be 1D and strictly increasing.")

def plot_B_heatmap(filename, component="mag", k_slice="mid", show_colorbar=True):
    """
    Plot a 2D heatmap at a z-slice.
      component: "mag", "Bx", "By", or "Bz"
      k_slice: integer index or "mid"
    """
    with h5py.File(filename, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2, 1, 0)  # (x,y,z)
        By = f["j_mag_field"][:].transpose(2, 1, 0)
        Bz = f["k_mag_field"][:].transpose(2, 1, 0)
        x_raw = f["x_coor"][0, 0, :]  # 1D coords (likely centers)

    nx, ny, nz = Bx.shape
    if isinstance(k_slice, str) and k_slice.lower() == "mid":
        k = nz // 2
    else:
        k = int(k_slice)
    if not (0 <= k < nz):
        raise IndexError(f"k_slice {k} out of range 0..{nz-1}")

    # Choose field to plot
    if component.lower() == "mag":
        data3d = np.sqrt(Bx**2 + By**2 + Bz**2)
        title = r"|B| (mid-plane)"
    elif component in ("Bx", "bx"):
        data3d = Bx; title = "Bx"
    elif component in ("By", "by"):
        data3d = By; title = "By"
    elif component in ("Bz", "bz"):
        data3d = Bz; title = "Bz"
    else:
        raise ValueError('component must be "mag", "Bx", "By", or "Bz"')

    # 2D slice (x,y) at chosen z-index
    img = data3d[:, :, k]

    # Build plotting extent from coordinates. Your array is for x;
    # assume cubic grid so y shares the same spacing.
    x_centers = np.squeeze(x_raw)
    if x_centers.size == nx:
        x_edges = infer_edges_from_centers(x_centers)
    elif x_centers.size == nx + 1:
        x_edges = x_centers
    else:
        # fallback: infer from min/max if shape mismatch
        x_edges = infer_edges_from_centers(np.linspace(x_centers.min(),
                                                       x_centers.max(), nx))
    extent = [x_edges[0], x_edges[-1], x_edges[0], x_edges[-1]]

    # Plot
    # plt.figure(figsize=6, 5)
    m = plt.imshow(
        img.T,                # transpose so X→horizontal, Y→vertical
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest"
    )
    if show_colorbar:
        plt.colorbar(m, label=title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} at z-index {k}")
    plt.tight_layout()
    plt.show()

# plot_B_heatmap(r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5", component="mag", k_slice="mid")
# plot_B_heatmap(r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5", component="mag", k_slice=4)
plot_B_heatmap(r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\mhd_fields.h5", component="mag", k_slice=4)

# --- usage ---
# plot_B_heatmap("path/to/file.h5", component="mag", k_slice="mid")
# plot_B_heatmap("path/to/file.h5", component="Bz", k_slice=0)
