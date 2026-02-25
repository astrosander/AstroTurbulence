import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cases = {
    "ms_10": {
        "bcc_path": "/home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_bcc.00005.vtk",
        "w_path":   "/home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_w.00005.vtk",
        "out_npz":  "Pu_spectrum_ms10.npz",
        "out_png":  "Pu_spectrum_ms10.png",
    },
    "ms_1": {
        "bcc_path": "ms1ma12_256.mhd_bcc.00071.vtk",
        "w_path":   "ms1ma12_256.mhd_w.00071.vtk",
        "out_npz":  "Pu_spectrum_ms1.npz",
        "out_png":  "Pu_spectrum_ms1.png",
    },
    "ms_10+ms_1": {
        "bcc_path": "ms1ma12_256.mhd_bcc.00071.vtk",
        "w_path":   "/home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_w.00005.vtk",
        "out_npz":  "Pu_spectrum_ms10_plus_ms1.npz",
        "out_png":  "Pu_spectrum_ms10_plus_ms1.png",
    },
}

dens_name = "dens"
bx_candidates = ["bcc3"]
by_candidates = ["bcc1"]
bz_candidates = ["bcc2"]

kappa = 1.0
gamma = 2.0

eta_list = [10.0]
screen_frac = 0.25
emit_start_frac = 0.25
emit_end_frac = 1.0

eps = 1e-12

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
    dims = tuple(int(x) for x in ds.dimensions)
    shape = (dims[0] - 1, dims[1] - 1, dims[2] - 1)
    return np.asarray(arr).reshape(shape, order="F")

for tag, cfg in cases.items():
    bcc_path = cfg["bcc_path"]
    w_path = cfg["w_path"]
    out_npz = cfg["out_npz"]
    out_png = cfg["out_png"]

    bcc = pv.read(bcc_path)
    w = pv.read(w_path)

    w = _as_cell_data(w, dens_name)
    dens = _reshape_cell(w, w.cell_data[dens_name])

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

    z_screen_end = int(round(screen_frac * nz))
    z_emit_start = int(round(emit_start_frac * nz))
    z_emit_end = int(round(emit_end_frac * nz))

    z_screen_end = max(1, min(z_screen_end, nz))
    z_emit_start = max(0, min(z_emit_start, nz))
    z_emit_end = max(z_emit_start + 1, min(z_emit_end, nz))

    screen_sl = slice(0, z_screen_end)
    emit_sl = slice(z_emit_start, z_emit_end)

    rm_los = kappa * np.sum(dens[:, :, screen_sl] * bz[:, :, screen_sl] * dz, axis=2)
    rm_mean = float(np.mean(rm_los))
    rm_std_pop = float(np.std(rm_los, ddof=0))
    rm_std_samp = float(np.std(rm_los, ddof=1))

    print(f"[{tag}] RM mean = {rm_mean:.6e}, RM std(pop) = {rm_std_pop:.6e}, RM std(sample) = {rm_std_samp:.6e}")

    sigma_RM = rm_std_pop
    if not np.isfinite(sigma_RM) or sigma_RM <= 0.0:
        raise RuntimeError(f"[{tag}] sigma_RM is non-positive or non-finite: {sigma_RM}")

    bperp2 = bx * bx + by * by
    bperp = np.sqrt(bperp2)
    phase = np.zeros_like(bx, dtype=np.complex128)
    m = bperp2 > 0
    phase[m] = (bx[m] + 1j * by[m]) ** 2 / bperp2[m]
    Pi = (bperp ** gamma) * phase

    P_emit = np.sum(Pi[:, :, emit_sl] * dz, axis=2)
    Phi_hat = rm_los / sigma_RM

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), indexing="ij")
    K = np.sqrt(KX * KX + KY * KY)

    kmax = K.max()
    nbins = max(16, int(np.sqrt(nx * ny)))
    k_edges = np.linspace(0.0, kmax, nbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    Pu_list = []
    E_list = []

    kvals = K.ravel()
    eta_arr = np.array(eta_list, dtype=float)

    for eta in eta_arr:
        P = P_emit * np.exp(1j * eta * Phi_hat)

        Q = P.real
        U = P.imag
        amp = np.sqrt(Q * Q + U * U) + eps
        n1 = Q / amp
        n2 = U / amp

        n1hat = np.fft.fft2(n1)
        n2hat = np.fft.fft2(n2)

        Pu = (np.abs(n1hat) ** 2) + (np.abs(n2hat) ** 2)
        Pu_s = np.fft.fftshift(Pu)

        Pu_list.append(Pu_s)

        pvals = Pu_s.ravel()
        sums, _ = np.histogram(kvals, bins=k_edges, weights=pvals)
        counts, _ = np.histogram(kvals, bins=k_edges)
        Ek = sums / np.maximum(counts, 1)
        E_list.append(Ek)

    Pu_stack = np.stack(Pu_list, axis=0)
    E_stack = np.stack(E_list, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axs[0].loglog(k_centers[1:], E_stack[0, 1:])
    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$E_u(k)$")
    axs[0].set_title(rf"{tag}  (eta={eta_arr[0]:.3g})")

    im = axs[1].imshow(np.log10(Pu_stack[0].T + 1e-30), origin="lower",
                       extent=[KX.min(), KX.max(), KY.min(), KY.max()], aspect="auto")
    axs[1].set_xlabel(r"$k_x$")
    axs[1].set_ylabel(r"$k_y$")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label=r"$\log_{10} P_u$")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    np.savez(
        out_npz,
        eta=eta_arr,
        sigma_RM=float(sigma_RM),
        rm_mean=float(rm_mean),
        rm_std_pop=float(rm_std_pop),
        rm_std_sample=float(rm_std_samp),
        k_edges=k_edges,
        k_centers=k_centers,
        E_u=E_stack,
        Pu_2d=Pu_stack,
        kx=np.fft.fftshift(kx),
        ky=np.fft.fftshift(ky),
        dx=dx,
        dy=dy,
        dz=dz,
        kappa=kappa,
        gamma=gamma,
        case_tag=str(tag),
        bcc_path=str(bcc_path),
        w_path=str(w_path),
    )

    print(f"[{tag}] wrote: {out_npz} and {out_png}")

