import os, json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv # <— new

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ───────────────────────────────────────────────────────────────────────
# VTK inputs
VTK_W_PATH   = Path("/home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_w.00006.vtk")    # contains dens, velx/y/z
VTK_BCC_PATH = Path("/home/amelnich/orcd/scratch/athena_runs/ma08_ms10/out_gpu/vtk/ms10ma08_512.mhd_bcc.00006.vtk")  # contains bcc1/2/3
# ───────────────────────────────────────────────────────────────────────

def _get_structured_array(mesh: pv.ImageData, name: str) -> np.ndarray:
    """
    Return a (nx, ny, nz) numpy array for scalar field `name` stored on VTK ImageData.
    Works for point or cell data; prefers point_data if both exist.
    Reshapes in Fortran order to match VTK memory layout.
    """
    nx, ny, nz = mesh.dimensions
    if name in mesh.point_data:
        arr = np.asarray(mesh.point_data[name]).reshape((nx, ny, nz), order="F")
        return arr.astype(np.float32)
    if name in mesh.cell_data:
        carr = np.asarray(mesh.cell_data[name]).reshape((nx-1, ny-1, nz-1), order="F")
        # pad nearest-neighbor on +faces to get point-like shape
        cx, cy, cz = carr.shape
        out = np.empty((cx+1, cy+1, cz+1), dtype=carr.dtype)
        out[:cx, :cy, :cz] = carr
        out[cx, :cy, :cz]  = carr[cx-1, :, :]
        out[:cx, cy, :cz]  = carr[:, cy-1, :]
        out[:cx, :cy, cz]  = carr[:, :, cz-1]
        out[cx, cy, :cz]   = carr[cx-1, cy-1, :]
        out[cx, :cy, cz]   = carr[cx-1, :, cz-1]
        out[:cx, cy, cz]   = carr[:, cy-1, cz-1]
        out[cx, cy, cz]    = carr[cx-1, cy-1, cz-1]
        return out.astype(np.float32)
    raise KeyError(f"'{name}' not found in VTK point_data or cell_data. "
                   f"Available: point={list(mesh.point_data.keys())}, cell={list(mesh.cell_data.keys())}")

def _infer_Lbox_from_imagedata(mesh: pv.ImageData):
    """
    Try to infer a single box-length L (assuming cubic domain). If dims/spacing
    are consistent across axes, returns L = nx*dx (== ny*dy == nz*dz). Otherwise returns None.
    """
    nx, ny, nz = mesh.dimensions
    dx, dy, dz = mesh.spacing
    Lx, Ly, Lz = nx*dx, ny*dy, nz*dz
    # Allow tiny numerical mismatches
    if np.allclose([Lx, Ly, Lz], Lx, rtol=1e-6, atol=1e-9):
        return float(Lx)
    return None

def _isotropic_k(shape, Lbox=None):
    nz, ny, nx = shape
    if Lbox is None:
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        kz = np.fft.fftfreq(nz) * nz
    else:
        dx = Lbox / nx; dy = Lbox / ny; dz = Lbox / nz
        kx = 2*np.pi*np.fft.fftfreq(nx, d=dx) * (Lbox/(2*np.pi))
        ky = 2*np.pi*np.fft.fftfreq(ny, d=dy) * (Lbox/(2*np.pi))
        kz = 2*np.pi*np.fft.fftfreq(nz, d=dz) * (Lbox/(2*np.pi))
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    Km = np.sqrt(KX**2 + KY**2 + KZ**2)
    return Km

def _vector_spectrum_sum(vx, vy, vz, Lbox=None, norm="total"):
    """
    Shell-summed 3D spectrum. Returns k, E(k), Nk.
    """
    N = vx.size
    Fvx = np.fft.fftn(vx); Fvy = np.fft.fftn(vy); Fvz = np.fft.fftn(vz)
    P = (np.abs(Fvx)**2 + np.abs(Fvy)**2 + np.abs(Fvz)**2) / N

    Km = _isotropic_k(vx.shape, Lbox=Lbox)
    kbin = np.floor(Km).astype(np.int64).ravel()
    E_sum = np.bincount(kbin, weights=P.ravel())
    Nk = np.bincount(kbin)
    k = np.arange(E_sum.size, dtype=float)
    mask = k > 0
    k, E, Nk = k[mask], E_sum[mask], Nk[mask]

    if norm == "total":
        E = E / (2.0 * N)
    elif norm == "energy_fraction":
        s = E.sum()
        if s > 0: E = E / s
    elif norm == "none":
        pass
    else:
        raise ValueError("norm must be one of ['total','energy_fraction','none']")
    return k, E, Nk

def _slope_guide(k, E, exponent=-5/3, frac_lo=0.07, frac_hi=0.35, scale=1.2):
    klo = max(1, int(len(k)*frac_lo))
    khi = max(klo+1, int(len(k)*frac_hi))
    kseg = k[klo:khi]
    kref = np.median(kseg) if len(kseg) else k[max(1, len(k)//3)]
    Eref = np.interp(kref, k, E)
    A = scale * Eref / (kref**exponent)
    return A * (k**exponent)

def _read_fields_from_vtk(vtk_w_path: Path, vtk_bcc_path: Path):
    """
    Returns vx, vy, vz, bx, by, bz (all float32, shaped as (nx,ny,nz)) and inferred Lbox (or None).
    """
    w_mesh = pv.read(str(vtk_w_path))
    b_mesh = pv.read(str(vtk_bcc_path))

    # velocity from *_w*
    vx = _get_structured_array(w_mesh, "velx")
    vy = _get_structured_array(w_mesh, "vely")
    vz = _get_structured_array(w_mesh, "velz")

    # magnetic from *_bcc*
    bx = _get_structured_array(b_mesh, "bcc1")
    by = _get_structured_array(b_mesh, "bcc2")
    bz = _get_structured_array(b_mesh, "bcc3")

    # sanity: shapes should match
    if vx.shape != bx.shape:
        raise ValueError(f"Velocity shape {vx.shape} != B-field shape {bx.shape}")

    # attempt Lbox
    Lw  = _infer_Lbox_from_imagedata(w_mesh)
    Lbc = _infer_Lbox_from_imagedata(b_mesh)
    Lbox = Lw if (Lw is not None and Lbc is not None and np.isclose(Lw, Lbc)) else None

    return vx, vy, vz, bx, by, bz, Lbox

def plot_from_vtk(vtk_w_path, vtk_bcc_path, out_prefix="spectrum", MA=2.0, kinj=2.0, norm="total"):
    vx, vy, vz, bx, by, bz, Lbox = _read_fields_from_vtk(vtk_w_path, vtk_bcc_path)

    # Spectra
    kv, Ev, Nv = _vector_spectrum_sum(vx, vy, vz, Lbox=Lbox, norm=norm)
    kb, Eb, Nb = _vector_spectrum_sum(bx, by, bz, Lbox=Lbox, norm=norm)

    # Reference slope (-5/3)
    gv = _slope_guide(kv, Ev, exponent=-5/3)
    gb = _slope_guide(kb, Eb, exponent=-5/3)

    # Transition wavenumber estimate
    kA = kinj * (MA**3)

    # Plot
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    ax1.loglog(kv, Ev, linewidth=1.5, label=r"$M_{\rm s}=1.0, M_{\rm A}=2.0$", color='blue')
    ax1.loglog(kv, gv, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    ax1.axvline(x=kA, linestyle='--', linewidth=1.2, label=rf"$k_A$", color='red')
    ax1.set_xlabel(r"$k\frac{L_{\rm box}}{2\pi}$" if Lbox is not None else r"$k$ (mode index)")
    ax1.set_ylabel(r"$E_v(k)$")
    ax1.set_ylim(1e-12, 1e0)
    ax1.legend(loc="lower left", fontsize=9)

    ax2.loglog(kb, Eb, linewidth=1.5, label=r"$M_{\rm s}=1.0, M_{\rm A}=2.0$", color='blue')
    ax2.loglog(kb, gb, linestyle="--", linewidth=1.4, label=r"slope: $-5/3$", color='black')
    ax2.axvline(x=kA, linestyle='--', linewidth=1.2, label=rf"$k_A$", color='red')
    ax2.set_xlabel(r"$k\frac{L_{\rm box}}{2\pi}$" if Lbox is not None else r"$k$ (mode index)")
    ax2.set_ylabel(r"$E_B(k)$")
    ax2.set_ylim(1e-12, 1e0)
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    out_path = f"{out_prefix}_both.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Save spectra + metadata for later recreation
    np.savez_compressed(f"{out_prefix}_spectra.npz", k_v=kv, E_v=Ev, k_b=kb, E_b=Eb)
    meta = {
        "kA": float(kA), "MA": float(MA), "kinj": float(kinj), "norm": norm,
        "png": out_path,
        "units": "k*Lbox/(2π)" if Lbox is not None else "mode_index",
        "Lbox": (float(Lbox) if Lbox is not None else None),
        "vtk_w": str(vtk_w_path), "vtk_bcc": str(vtk_bcc_path),
    }
    json.dump(meta, open(f"{out_prefix}_spectra.json","w"), indent=2)
    return out_path, meta

def main():
    out_prefix = "spectrum"
    MA = 2.0
    kinj = 2.0
    norm = "total"

    if not VTK_W_PATH.exists():
        raise FileNotFoundError(f"Could not find VTK file: {VTK_W_PATH}")
    if not VTK_BCC_PATH.exists():
        raise FileNotFoundError(f"Could not find VTK file: {VTK_BCC_PATH}")

    out_path, meta = plot_from_vtk(VTK_W_PATH, VTK_BCC_PATH, out_prefix=out_prefix, MA=MA, kinj=kinj, norm=norm)
    print("Saved:", out_path)
    print(json.dumps({"meta": meta}, indent=2))

if __name__ == "__main__":
    main()


