#!/usr/bin/env python3
import numpy as np
import pyvista as pv

# -----------------------
# User config
# -----------------------
cases = {
    "synthetic": {
        "bcc_path": "../synthetic_bcc.vtk",
        "w_path":   "../synthetic_w.vtk",
        "out_npz":  "Pu_cache_synthetic.npz",
    }
}

dens_name = "dens"
bx_candidates = ["bcc3"]
by_candidates = ["bcc1"]
bz_candidates = ["bcc2"]

kappa = 1.0
gamma = 2.0

# z-slices
screen_frac = 0.25
emit_start_frac = 0.25
emit_end_frac = 1.0

# -----------------------
# Helpers
# -----------------------
def _find_array(ds, names):
    for n in names:
        if n in ds.cell_data:
            return ds.cell_data[n], "cell"
        if n in ds.point_data:
            return ds.point_data[n], "point"
    raise KeyError(f"None of {names} found in cell_data/point_data")

def _as_cell_data(ds, name):
    if name in ds.cell_data:
        return ds
    if name in ds.point_data:
        return ds.point_data_to_cell_data()
    raise KeyError(name)

def _reshape_cell(ds, arr):
    dims = tuple(int(x) for x in ds.dimensions)  # point dims
    shape = (dims[0] - 1, dims[1] - 1, dims[2] - 1)  # cell dims
    return np.asarray(arr).reshape(shape, order="F")

# -----------------------
# Main
# -----------------------
for tag, cfg in cases.items():
    bcc_path = cfg["bcc_path"]
    w_path   = cfg["w_path"]
    out_npz  = cfg["out_npz"]

    print(f"[{tag}] reading VTK...")
    bcc = pv.read(bcc_path)
    w   = pv.read(w_path)

    # density (cell)
    w = _as_cell_data(w, dens_name)
    dens = _reshape_cell(w, w.cell_data[dens_name])

    # B components
    bx_raw, bx_loc = _find_array(bcc, bx_candidates)
    by_raw, by_loc = _find_array(bcc, by_candidates)
    bz_raw, bz_loc = _find_array(bcc, bz_candidates)

    if bx_loc == "point" or by_loc == "point" or bz_loc == "point":
        bcc = bcc.point_data_to_cell_data()
        bx_raw, _ = _find_array(bcc, bx_candidates)
        by_raw, _ = _find_array(bcc, by_candidates)
        bz_raw, _ = _find_array(bcc, bz_candidates)

    bx = _reshape_cell(bcc, bx_raw)
    by = _reshape_cell(bcc, by_raw)
    bz = _reshape_cell(bcc, bz_raw)

    dx, dy, dz = (float(x) for x in bcc.spacing)
    nx, ny, nz = bx.shape

    # z ranges
    z_screen_end  = int(round(screen_frac * nz))
    z_emit_start  = int(round(emit_start_frac * nz))
    z_emit_end    = int(round(emit_end_frac   * nz))

    z_screen_end = max(1, min(z_screen_end, nz))
    z_emit_start = max(0, min(z_emit_start, nz))
    z_emit_end   = max(z_emit_start + 1, min(z_emit_end, nz))

    screen_sl = slice(0, z_screen_end)
    emit_sl   = slice(z_emit_start, z_emit_end)

    # RM stats and Phi_hat
    rm_los = kappa * np.sum(dens[:, :, screen_sl] * bz[:, :, screen_sl] * dz, axis=2)
    rm_mean = float(np.mean(rm_los))
    rm_std_pop = float(np.std(rm_los, ddof=0))
    rm_std_samp = float(np.std(rm_los, ddof=1))

    print(f"[{tag}] RM mean={rm_mean:.6e} std(pop)={rm_std_pop:.6e} std(sample)={rm_std_samp:.6e}")

    sigma_RM = rm_std_pop
    if not np.isfinite(sigma_RM) or sigma_RM <= 0.0:
        raise RuntimeError(f"[{tag}] sigma_RM is non-positive or non-finite: {sigma_RM}")

    Phi_hat = rm_los / sigma_RM

    # Pi and P_emit (eta-independent)
    bperp2 = bx * bx + by * by
    bperp  = np.sqrt(bperp2)

    phase = np.zeros_like(bx, dtype=np.complex128)
    m = bperp2 > 0
    phase[m] = (bx[m] + 1j * by[m]) ** 2 / bperp2[m]

    Pi = (bperp ** gamma) * phase
    P_emit = np.sum(Pi[:, :, emit_sl] * dz, axis=2)  # (nx, ny), complex

    # k-grid + binning precompute (eta-independent)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)

    kx_s = np.fft.fftshift(kx)
    ky_s = np.fft.fftshift(ky)

    KX, KY = np.meshgrid(kx_s, ky_s, indexing="ij")
    K = np.sqrt(KX * KX + KY * KY)
    kvals = K.ravel()

    kmax = float(K.max())
    nbins = max(16, int(np.sqrt(nx * ny)))
    k_edges = np.linspace(0.0, kmax, nbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    # Precompute fast binning: bin index per pixel + counts per bin
    bin_idx = np.digitize(kvals, k_edges) - 1  # 0..nbins-1, may spill
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx.astype(np.int32)

    counts = np.bincount(bin_idx[valid], minlength=nbins).astype(np.int64)

    # Save everything needed for fast plotting later
    np.savez_compressed(
        out_npz,
        case_tag=str(tag),
        bcc_path=str(bcc_path),
        w_path=str(w_path),

        # physics params
        kappa=float(kappa),
        gamma=float(gamma),

        # grids / spacing
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        kx=kx_s,
        ky=ky_s,

        # z windows
        screen_frac=float(screen_frac),
        emit_start_frac=float(emit_start_frac),
        emit_end_frac=float(emit_end_frac),
        z_screen_end=int(z_screen_end),
        z_emit_start=int(z_emit_start),
        z_emit_end=int(z_emit_end),

        # RM stats / normalized map
        sigma_RM=float(sigma_RM),
        rm_mean=float(rm_mean),
        rm_std_pop=float(rm_std_pop),
        rm_std_sample=float(rm_std_samp),
        Phi_hat=Phi_hat.astype(np.float32),

        # emission (eta independent)
        P_emit=P_emit.astype(np.complex64),

        # k-binning
        k_edges=k_edges.astype(np.float32),
        k_centers=k_centers.astype(np.float32),
        bin_idx=bin_idx,
        valid=valid,
        counts=counts,
        nbins=int(nbins),
    )

    print(f"[{tag}] wrote cache NPZ: {out_npz}")
