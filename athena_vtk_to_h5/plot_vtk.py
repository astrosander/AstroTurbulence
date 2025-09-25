VTK_W_PATH   = "../ms02ma20_256.mhd_w.00398.vtk"    # density etc.
VTK_BCC_PATH = "../ms02ma20_256.mhd_bcc.00398.vtk"  # magnetic field
OUT_FIG      = "density_B_projections_2x2.png"

import pyvista as pv

# --- Load meshes ---
w_mesh   = pv.read(VTK_W_PATH)
bcc_mesh = pv.read(VTK_BCC_PATH)

# --- Print all tags (arrays) ---
print("Density file arrays (tags):", w_mesh.array_names)
print("Magnetic field file arrays (tags):", bcc_mesh.array_names)

# --- What we want to show (name, mesh, pretty title) ---
# Use 'dens' for density (as reported in your output)
panes = [
    ("dens", w_mesh,   "Density (dens)"),
    ("bcc1", bcc_mesh, "B_x (bcc1)"),
    ("bcc2", bcc_mesh, "B_y (bcc2)"),
    ("bcc3", bcc_mesh, "B_z (bcc3)"),
]

# --- Prepare 2x2 plotter (off-screen for saving to file) ---
plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1400, 1400))
plotter.set_background("white")  # modern clean background

for idx, (arr_name, mesh, title) in enumerate(panes):
    r, c = divmod(idx, 2)
    plotter.subplot(r, c)

    if arr_name not in mesh.array_names:
        # Graceful fallback if array is missing
        # Show an empty scene with a note in the corner
        txt = f"{title}\n(NOT FOUND)"
        plotter.add_text(txt, position="upper_left", font_size=12)
        continue

    # Activate the scalar you want before slicing so the slice carries it
    mesh.set_active_scalars(arr_name)

    # Central Z slice through the cube (no borders/axes/shadows)
    slc = mesh.slice(normal="z", origin=mesh.center)

    plotter.add_mesh(
        slc,
        cmap="viridis",
        show_edges=False,   # no borders/edges
        lighting=False,     # no shading/shadows for a flat, modern look
        scalar_bar_args={"title": title, "vertical": False},  # title as legend
        show_scalar_bar=False,  # hide scalar bar for super clean layout
    )

    # Keep the view orthographic-like and minimal
    plotter.camera.position = (0, 0, 1)  # look along +z
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 1, 0)
    # Remove bounding box/axes for this subplot
    plotter.remove_bounds_axes()

# Save a single composite image of the 2x2 layout
plotter.screenshot(OUT_FIG)
plotter.close()

print(f"✅ Saved 2×2 central-slice projections to: {OUT_FIG}")
