#!/usr/bin/env python3
#python compute_pfa_vs_chi_lp16.py ..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5  --geometry mixed --los-axis 2 --C 0.81 --chi-min 0.05 --chi-max 20 --n-chi 250  --save npz/curve_mixed_chi.npz --plot
import numpy as np
import h5py, os, argparse
import matplotlib.pyplot as plt

# ---------- I/O ----------
def load_fields(h5_path, bx="i_mag_field", by="j_mag_field", bz="k_mag_field", ne="gas_density"):
    with h5py.File(h5_path, "r") as f:
        Bx = f[bx][()]
        By = f[by][()]
        Bz = f[bz][()]
        ne = f[ne][()]
    return Bx, By, Bz, ne

def move_los(arr, los_axis):
    return np.moveaxis(arr, los_axis, 0)

# ---------- physics bits ----------
def polarized_emissivity_simple(Bx, By, gamma=2.0):
    # LP16 γ=2 -> (Bx + iBy)^2 (no extra amplitude factor)
    if gamma == 2.0:
        return (Bx + 1j*By)**2
    B2 = Bx**2 + By**2
    eps = np.finfo(B2.dtype).tiny
    amp = np.power(np.maximum(B2, eps), 0.5*(gamma - 2.0))
    return ((Bx + 1j*By)**2) * amp

def faraday_density(ne, Bpar, C=1.0):
    return C * ne * Bpar

def sigma_phi_from_phi(phi, los_axis):
    """σ_Φ from integrating φ along LOS with Δz=1/Nz."""
    phi_los = move_los(phi, los_axis)
    Nz = phi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_tot = np.sum(phi_los, axis=0) * dz
    return float(Phi_tot.std())

# ---------- P maps (LP16 defs) ----------
def P_map_mixed(Pi, phi, lam, los_axis):
    Pi_los  = move_los(Pi,  los_axis)
    phi_los = move_los(phi, los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_cum = np.cumsum(phi_los * dz, axis=0)
    phase = np.exp(2j * (lam**2) * Phi_cum)
    return np.sum(Pi_los * phase, axis=0) * dz

def P_map_separated(Pi, phi, lam, los_axis, emit_bounds=None, screen_bounds=None, detrend_emit=True):
    Pi_los  = move_los(Pi,  los_axis)
    phi_los = move_los(phi, los_axis)
    Nz = Pi_los.shape[0]
    dz = 1.0 / float(Nz)

    if emit_bounds is None:
        emit_bounds = (0, Nz)
    if screen_bounds is None:
        scr = max(1, int(0.1 * Nz))
        screen_bounds = (Nz - scr, Nz)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    P_emit = np.sum(Pi_los[z0e:z1e], axis=0) * dz
    if detrend_emit:
        P_emit = P_emit - P_emit.mean()
    Phi_screen = np.sum(phi_los[z0s:z1s], axis=0) * dz
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)

# ---------- PFA vs χ ----------
def compute_pfa_vs_chi(h5_path,
                       geometry="mixed",            # "mixed" or "separated"
                       los_axis=2,                  # 0,1,2
                       C=1.0,                       # Faraday constant in φ=C n_e B∥
                       gamma=2.0,
                       chi_min=0.05, chi_max=20.0, n_chi=60,
                       emit_bounds=None, screen_bounds=None,
                       subtract_mean_Bpar=False):
    """
    Returns: chi_grid, y(chi), meta dict with sigma_phi, label, lam2_grid.
    χ-grid is shared; λ²-grid is chi/(2σ_Φ).
    """
    Bx, By, Bz, ne = load_fields(h5_path)
    Pi = polarized_emissivity_simple(Bx, By, gamma=gamma)

    # LOS-parallel component: must match your PSA/directional setup
    if   los_axis == 0: Bpar = Bx
    elif los_axis == 1: Bpar = By
    else:               Bpar = Bz
    if subtract_mean_Bpar:
        Bpar = Bpar - Bpar.mean()

    phi = faraday_density(ne, Bpar, C=C)
    sigma_phi = sigma_phi_from_phi(phi, los_axis)
    if not np.isfinite(sigma_phi) or sigma_phi <= 0:
        raise RuntimeError("σ_Φ is zero or invalid; check inputs/LOS/C.")

    # Build shared χ-grid, then map to λ²-grid for evaluation
    chi_grid = np.geomspace(chi_min, chi_max, n_chi)
    lam2_grid = chi_grid / (2.0 * sigma_phi)
    lam_grid = np.sqrt(lam2_grid)

    # Evaluate ⟨|P|²⟩ on this λ-grid
    vals = []
    for lam in lam_grid:
        if geometry == "mixed":
            P = P_map_mixed(Pi, phi, lam, los_axis)
        else:
            P = P_map_separated(Pi, phi, lam, los_axis,
                                emit_bounds=emit_bounds, screen_bounds=screen_bounds,
                                detrend_emit=True)
        vals.append(np.mean(np.abs(P)**2))

    meta = {
        "sigma_phi": sigma_phi,
        "lam2_grid": lam2_grid,
        "label": f"{os.path.basename(h5_path)} | geom={geometry} | LOS={los_axis} | C={C:g}"
    }
    return chi_grid, np.array(vals), meta

# ---------- quick plot & save ----------
def save_curve_npz(outpath, chi, y, label, sigma_phi, lam2_grid):
    os.makedirs(os.path.dirname(outpath), exist_ok=True) if os.path.dirname(outpath) else None
    np.savez(outpath, chi=chi, y=y, label=np.array(label), sigma_phi=np.array(sigma_phi), lam2=lam2_grid)

def main():
    ap = argparse.ArgumentParser(description="Compute PFA vs χ with shared χ-grid (LP16-like).")
    ap.add_argument("h5", help="Path to HDF5 cube")
    ap.add_argument("--geometry", choices=["mixed","separated"], default="mixed")
    ap.add_argument("--los-axis", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--chi-min", type=float, default=0.05)
    ap.add_argument("--chi-max", type=float, default=20.0)
    ap.add_argument("--n-chi",  type=int,   default=60)
    ap.add_argument("--emit",   type=str,   default="")  # e.g. "0:230"
    ap.add_argument("--screen", type=str,   default="")  # e.g. "230:256"
    ap.add_argument("--randdom", action="store_true", help="subtract mean B∥ to force random-dominated")
    ap.add_argument("--save", type=str, default="", help="npz output path (e.g., npz/curve_mixed_chi.npz)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    emit_bounds = tuple(map(int, args.emit.split(":"))) if args.emit else None
    screen_bounds = tuple(map(int, args.screen.split(":"))) if args.screen else None

    chi, y, meta = compute_pfa_vs_chi(
        h5_path=args.h5,
        geometry=args.geometry,
        los_axis=args.los_axis,
        C=args.C,
        gamma=args.gamma,
        chi_min=args.chi_min, chi_max=args.chi_max, n_chi=args.n_chi,
        emit_bounds=emit_bounds, screen_bounds=screen_bounds,
        subtract_mean_Bpar=args.randdom
    )

    print(f"σ_Φ = {meta['sigma_phi']:.6g}")
    print(f"Saved label: {meta['label']}")

    if args.save:
        save_curve_npz(args.save, chi, y, meta["label"], meta["sigma_phi"], meta["lam2_grid"])
        print(f"Saved: {args.save}")

    if args.plot:
        plt.figure(figsize=(6.6, 4.8))
        plt.loglog(chi, y, "-o", ms=3, label=meta["label"])
        plt.axvspan(1.0, 3.0, color="grey", alpha=0.12, lw=0)
        plt.xlabel(r"$\chi = 2\sigma_\Phi \lambda^2$")
        plt.ylabel(r"$\langle |P|^2 \rangle$")
        plt.title(rf"PFA vs $\chi$  (σ$_\Phi$={meta['sigma_phi']:.3g})")
        plt.grid(True, which="both", ls=":", lw=0.6)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig("PFA.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    main()
