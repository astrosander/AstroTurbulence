# plot_fig2_zhang2016.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Sequence
from multiprocessing import Pool, freeze_support
import time
import os

BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

def load_fields(path):
    with h5py.File(path, "r") as f:
        bx = f[BX_KEY][...].astype(np.float64)
        by = f[BY_KEY][...].astype(np.float64)
        bz = f[BZ_KEY][...].astype(np.float64)
        ne = f[NE_KEY][...].astype(np.float64)
    return bx, by, bz, ne

def decorrelate_phi_along_z(phi):
    """Randomly roll φ(z,y,x) along z per (y,x). Preserves φ spectrum & r_i."""
    rng = np.random.default_rng(42)
    nz = phi.shape[2]
    out = np.empty_like(phi)
    for j in range(phi.shape[1]):
        for i in range(phi.shape[0]):
            s = int(rng.integers(0, nz))
            out[i, j, :] = np.roll(phi[i, j, :], s)
    return out

def pfa_variance_curve(path, lam2_grid, gamma=2.0, lam2_break_target=1.0,
                       decorrelate=False):
    """
    Mixed case PFA: Pmap(λ) = Σ_z P_i(z) exp[2i λ² Φ(z+1/2)] Δz, with φ = n_e B∥.
    Returns ⟨|P|²⟩ vs λ², normalized so that the small-λ² plateau is at ~0.1 (for styling).
    """
    bx, by, bz, ne = load_fields(path)
    nx, ny, nz = bx.shape
    dz = 1.0 / nz  # take physical LOS depth L=1 for all cubes (as in the paper)

    # γ=2 ⇒ P_i = (Bx+iBy)^2 exactly
    Pi = (bx + 1j*by)**2

    # constant n_e already (all ones); Faraday density (before scaling C):
    rm_base = ne * bz

    # cumulative Φ at cell centers: Φ(z+1/2) = Σ_0^z rm_base Δz + 0.5*rm_base Δz
    phi_cum = np.cumsum(rm_base, axis=2) * dz
    phi_half = phi_cum - 0.5 * rm_base * dz

    # choose Faraday scale so that λ^2 σ_Φ ≈ 1 at lam2_break_target
    Phi_tot = (rm_base * dz).sum(axis=2)  # total Φ per LOS for C=1
    sigma_Phi = float(Phi_tot.std())
    C = 1.0 / max(lam2_break_target * sigma_Phi, 1e-30)

    # Optionally remove residual Pi–Φ correlations (often tightens the -2)
    if decorrelate:
        phi_half = decorrelate_phi_along_z(phi_half)

    y = np.empty_like(lam2_grid, dtype=np.float64)
    for i, l2 in enumerate(lam2_grid):
        phase = np.exp(2j * (l2 * C) * phi_half)
        Pmap = (Pi * phase).sum(axis=2) * dz
        y[i] = np.mean(np.abs(Pmap)**2)

    # normalize small-λ² plateau to 0.1 (as in Zhang's styling)
    y *= (0.1 / y[0])
    return y

def process_wavelength_batch_worker(args):
    """Worker function for parallel wavelength processing"""
    path, lam2_batch, gamma, lam2_break_target, decorrelate = args
    
    with h5py.File(path, "r") as f:
        bx = f[BX_KEY][...].astype(np.float64)
        by = f[BY_KEY][...].astype(np.float64)
        bz = f[BZ_KEY][...].astype(np.float64)
        ne = f[NE_KEY][...].astype(np.float64)

    nx, ny, nz = bx.shape
    dz = 1.0 / nz

    Pi = (bx + 1j*by)**2
    rm_base = ne * bz
    phi_cum = np.cumsum(rm_base, axis=2) * dz
    phi_half = phi_cum - 0.5 * rm_base * dz

    Phi_tot = (rm_base * dz).sum(axis=2)
    sigma_Phi = float(Phi_tot.std())
    C = 1.0 / max(lam2_break_target * sigma_Phi, 1e-30)

    if decorrelate:
        phi_half = decorrelate_phi_along_z(phi_half)

    y = np.empty(lam2_batch.size, dtype=np.float64)
    for i, l2 in enumerate(lam2_batch):
        phase = np.exp(2j * (l2 * C) * phi_half)
        Pmap = (Pi * phase).sum(axis=2) * dz
        y[i] = np.mean(np.abs(Pmap)**2)

    return y

def pfa_variance_curve_parallel(path, lam2_grid, gamma=2.0, lam2_break_target=1.0,
                                decorrelate=False, n_processes=11):
    """Parallel version with multiprocessing"""
    batch_size = max(1, len(lam2_grid) // n_processes)
    lam2_batches = [lam2_grid[i:i+batch_size] for i in range(0, len(lam2_grid), batch_size)]
    
    args_list = [(path, lam2_batch, gamma, lam2_break_target, decorrelate) 
                 for lam2_batch in lam2_batches]
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_wavelength_batch_worker, args_list)
    
    y = np.concatenate(results)
    
    # Normalize small-λ² plateau to 0.1
    y *= (0.1 / y[0])
    return y

def main():
    freeze_support()
    
    # Create npz directory if it doesn't exist
    os.makedirs("npz", exist_ok=True)
    
    # supply whichever cubes you have; the full panel uses these five:
    paths = [
        "synthetic_128x128x256.h5",
        "synthetic_128x128x512.h5",
        "synthetic_128x128x1024.h5",
        "synthetic_128x128x2048.h5",
        "synthetic_128x128x4096.h5",
    ]
    labels = ["128×128×256","128×128×512","128×128×1024","128×128×2048","128×128×4096"]
    styles = [":","--","-.", (0,(3,1,1,1)),"-"]

    # λ² grid spanning 6 decades like Fig.2
    lam2 = np.logspace(-3, 3, 180)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    curves = []

    for p, lab, ls in zip(paths, labels, styles):
        try:
            print(f"\nProcessing {p} with 11 parallel processes...")
            start_time = time.time()
            if "256" in p:
                continue
                y = pfa_variance_curve_parallel(p, lam2, gamma=2.0, lam2_break_target=1.0, 
                                            decorrelate=True, n_processes=11)
            elif "512" in p:
                y = pfa_variance_curve_parallel(p, lam2, gamma=2.0, lam2_break_target=1.0, 
                                            decorrelate=True, n_processes=8)
            elif "1024" in p:
                y = pfa_variance_curve_parallel(p, lam2, gamma=2.0, lam2_break_target=1.0, 
                                            decorrelate=True, n_processes=4)
            elif "2048" in p:
                y = pfa_variance_curve_parallel(p, lam2, gamma=2.0, lam2_break_target=1.0, 
                                            decorrelate=True, n_processes=2)
            else:
                y = pfa_variance_curve_parallel(p, lam2, gamma=2.0, lam2_break_target=1.0, 
                                            decorrelate=True, n_processes=1) 
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds")
            
            curves.append(y)
            ax.plot(np.log10(lam2), np.log10(y), ls=ls, label=lab)
            
            # Save .npz file for this curve
            npz_filename = f"npz/curve_{lab.replace('×', 'x')}.npz"
            np.savez(npz_filename, lam2=lam2, y=y, label=lab)
            print(f"Saved to {npz_filename}")
            
        except FileNotFoundError:
            # skip missing cubes so you can start with just the 256 one
            continue

    # draw a λ^{-2} guide anchored near log10 λ² ≈ 0.8 (within the inertial segment)
    if curves:
        ref_i = min(len(curves)-1, 3)            # pick one of the longer boxes if present
        x0_log = 0.8
        i0 = int(np.argmin(np.abs(np.log10(lam2) - x0_log)))
        y0 = curves[ref_i][i0]
        ref = y0 * (lam2 / lam2[i0])**(-2.0)
        # Filter to x-axis range from 1 to 2 (log10)
        log10_lam2 = np.log10(lam2)
        mask = (log10_lam2 >= 1) & (log10_lam2 <= 2)
        ax.plot(log10_lam2[mask], np.log10(ref)[mask], color="k", linewidth=1.0, label=r"$\lambda^{-2}$")

    ax.set_xlabel(r"$\log_{10}[\lambda^2]$")
    ax.set_ylabel(r"$\log_{10}\langle |P|^2\rangle$")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("fig2_like.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
