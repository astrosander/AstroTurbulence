import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# Inputs: three NPZ outputs
# ==========================
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

# ==========================
# Physics / numerics
# ==========================
dens_name = "dens"
bx_candidates = ["bcc3"]
by_candidates = ["bcc1"]
bz_candidates = ["bcc2"]

kappa = 1.0
gamma = 2.0

# Instead of lam2_list, we will use eta_list where:
#   eta = 2 * sigma_RM * lambda^2
# and inside the exponent we use:
#   exp(i * eta * Phi_hat) where Phi_hat = Phi / sigma_RM
eta_list = [0.415]   # example: eta ≈ 0.415 (from your RM std and lambda^2 choice)

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

    # density as cell data
    w = _as_cell_data(w, dens_name)
    dens = _reshape_cell(w, w.cell_data[dens_name])

    # B components (bcc) as cell data
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

    # -------------------------
    # RM stats (LOS-integrated)
    # -------------------------
    rm_los = kappa * np.sum(dens * bz * dz, axis=2)  # shape (nx, ny)
    rm_mean = float(np.mean(rm_los))
    rm_std_pop = float(np.std(rm_los, ddof=0))
    rm_std_samp = float(np.std(rm_los, ddof=1))

    print(f"[{tag}] RM mean = {rm_mean:.6e}, RM std(pop) = {rm_std_pop:.6e}, RM std(sample) = {rm_std_samp:.6e}")

    # Guard: avoid divide-by-zero in normalization
    sigma_RM = rm_std_pop
    if not np.isfinite(sigma_RM) or sigma_RM <= 0.0:
        raise RuntimeError(f"[{tag}] sigma_RM is non-positive or non-finite: {sigma_RM}")

    # -------------------------
    # Intrinsic polarization Pi
    # -------------------------
    bperp2 = bx * bx + by * by
    bperp = np.sqrt(bperp2)
    phase = np.zeros_like(bx, dtype=np.complex128)
    m = bperp2 > 0
    phase[m] = (bx[m] + 1j * by[m]) ** 2 / bperp2[m]
    Pi = (bperp ** gamma) * phase

    # -------------------------
    # k-space setup (2D)
    # -------------------------
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), indexing="ij")
    K = np.sqrt(KX * KX + KY * KY)

    kmax = K.max()
    nbins = max(16, int(np.sqrt(nx * ny)))
    k_edges = np.linspace(0.0, kmax, nbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    # -------------------------
    # Build P(u) vs eta
    # -------------------------
    Pu_list = []
    E_list = []

    # Phi(z): cumulative Faraday depth along LOS
    Phi = np.cumsum(kappa * dens * bz * dz, axis=2)  # shape (nx, ny, nz)
    Phi_hat = Phi / sigma_RM

    for eta in eta_list:
        # Use eta in exponent (eta = 2*sigma_RM*lambda^2)
        P = np.sum(Pi * np.exp(1j * eta * Phi_hat) * dz, axis=2)  # shape (nx, ny)

        u = P / (np.abs(P) + eps)
        uhat = np.fft.fft2(u)
        Pu = np.abs(uhat) ** 2
        Pu_s = np.fft.fftshift(Pu)

        Pu_list.append(Pu_s)

        kvals = K.ravel()
        pvals = Pu_s.ravel()
        sums, _ = np.histogram(kvals, bins=k_edges, weights=pvals)
        counts, _ = np.histogram(kvals, bins=k_edges)
        Ek = sums / np.maximum(counts, 1)
        E_list.append(Ek)

    Pu_stack = np.stack(Pu_list, axis=0)  # (n_eta, nx, ny)
    E_stack = np.stack(E_list, axis=0)    # (n_eta, nbins)

    # -------------------------
    # Plot (x-axis is eta)
    # -------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axs[0].loglog(k_centers[1:], E_stack[0, 1:])
    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$E_u(k)$")
    axs[0].set_title(rf"{tag}  (eta={eta_list[0]:.3g})")

    im = axs[1].imshow(np.log10(Pu_stack[0].T + 1e-30), origin="lower",
                       extent=[KX.min(), KX.max(), KY.min(), KY.max()], aspect="auto")
    axs[1].set_xlabel(r"$k_x$")
    axs[1].set_ylabel(r"$k_y$")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label=r"$\log_{10} P_u$")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # -------------------------
    # Save NPZ (store eta, sigma_RM too)
    # -------------------------
    np.savez(
        out_npz,
        eta=np.array(eta_list, dtype=float),
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

