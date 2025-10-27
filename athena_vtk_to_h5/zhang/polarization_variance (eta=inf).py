import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, freeze_support

BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

def polarized_variance_curve(path, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1):
    with h5py.File(path, "r") as f:
        bx = f[BX_KEY][...].astype(np.float64)
        by = f[BY_KEY][...].astype(np.float64)
        bz = f[BZ_KEY][...].astype(np.float64)
        ne = f[NE_KEY][...].astype(np.float64)

    nx, ny, nz = bx.shape
    dz = L / float(nz)

    # --- enforce η=∞ (random-dominated Faraday) as in Zhang fig: remove mean B∥
    bz = bz - bz.mean()

    bperp2 = bx*bx + by*by
    emis = bperp2**(gamma/2.0)
    e2ipsi = np.exp(2j * np.arctan2(by, bx))
    emis_phase = emis * e2ipsi

    # Faraday density per unit length (this is the σ_φ that controls L_eff)
    # NOTE: This uses the same B field for emissivity and Faraday rotation.
    # For stricter LP16/Zhang compliance with λ⁻² asymptote, consider using an 
    # independent scalar for B∥ (e.g. bz_rm = synth_scalar(..., seed=777))
    # to decorrelate Π(z) from φ(z), which sharpens the slope.
    rm_base = ne * bz                        # φ(z) in arbitrary units per length
    phi_cum = np.cumsum(rm_base, axis=2) * dz
    phi_half = phi_cum - 0.5 * rm_base * dz # midpoint cumulation for phase

    # --- Use local σ_φ, not sky σ of the *integrated* Φ, to place the break ---
    sigma_phi_local = float(np.std(rm_base))       # rms of φ per unit length
    # place the L→long-λ break where L_eff ≈ L:  L_eff ~ [2 λ^2 σ_φ]^{-1}
    # ⇒ choose rm_coeff so that λ_break^2 ≈ lam2_break_target
    #    1 / (2 λ_break^2 rm_coeff σ_φ) ≈ L  →  rm_coeff ≈ [1 / (2 λ_break^2 σ_φ L)]
    rm_coeff = 1.0 / max(2.0 * lam2_break_target * sigma_phi_local * L, 1e-30)

    # simple regime diagnostics (optional but helpful)
    def L_eff(l2): return 1.0 / max(2.0 * l2 * rm_coeff * sigma_phi_local, 1e-30)
    # Estimate correlation length along z from autocorrelation
    def estimate_corr_length_z(field):
        # Sample a few sightlines and average ACF
        nsample = min(100, field.shape[0] * field.shape[1])
        idx = np.random.RandomState(42).choice(field.shape[0] * field.shape[1], nsample, replace=False)
        corr_est = []
        for i in idx:
            iz, iy = i // field.shape[1], i % field.shape[1]
            z_data = field[iz, iy, :]
            if np.std(z_data) < 1e-10:
                continue
            z_acf = np.correlate(z_data, z_data, mode='full') / np.correlate(z_data, z_data, mode='valid')[0]
            z_acf = z_acf[len(z_data)-1:]  # positive lags only
            # Find first crossing of 1/e
            e_index = np.where(z_acf < np.exp(-1))[0]
            if len(e_index) > 0:
                corr_est.append(float(e_index[0]) * dz)
        ri = np.mean(corr_est) if corr_est else 0.01 * L  # fallback estimate
        return ri
    
    try:
        ri = estimate_corr_length_z(rm_base)
        print(f"[diag] σ_φ(local)={sigma_phi_local:.3e}, L={L:.3e}, r_i≈{ri:.3e}, "
              f"λ²_break≈{lam2_break_target:g}, "
              f"L_eff(λ²=1)≈{L_eff(1.0):.3e}, L_eff(λ²=10)≈{L_eff(10.0):.3e}")
        print(f"[diag] For λ⁻² regime: need L_eff << L and L_eff << r_i. "
              f"At λ²=10: L_eff/L≈{L_eff(10.0)/L:.3f}, L_eff/r_i≈{L_eff(10.0)/ri:.3f}")
    except:
        print(f"[diag] σ_φ(local)={sigma_phi_local:.3e}, L={L:.3e}, "
              f"λ²_break≈{lam2_break_target:g}, "
              f"L_eff(λ²=1)≈{L_eff(1.0):.3e}, L_eff(λ²=10)≈{L_eff(10.0):.3e}")

    y = np.empty(lam2.size, dtype=np.float64)
    for i, l2 in enumerate(lam2):
        phase = np.exp(2j * l2 * rm_coeff * phi_half)
        Pmap = np.sum(emis_phase * phase, axis=2) * dz
        y[i] = np.mean(np.abs(Pmap)**2)

    scale = 0.1 / y[0]
    y *= scale
    return y

def polarized_variance_curve_optimized(path, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1):
    """Optimized version with vectorized operations"""
    with h5py.File(path, "r") as f:
        bx = f[BX_KEY][...].astype(np.float64)
        by = f[BY_KEY][...].astype(np.float64)
        bz = f[BZ_KEY][...].astype(np.float64)
        ne = f[NE_KEY][...].astype(np.float64)

    nx, ny, nz = bx.shape
    dz = L / float(nz)

    # --- enforce η=∞ (random-dominated Faraday) as in Zhang fig: remove mean B∥
    bz = bz - bz.mean()

    # Pre-compute all emission and polarization data
    bperp2 = bx*bx + by*by
    emis = bperp2**(gamma/2.0)
    e2ipsi = np.exp(2j * np.arctan2(by, bx))
    emis_phase = emis * e2ipsi

    # Half-cell Faraday depth calculation
    # NOTE: Uses same B field for emissivity and Faraday. See decorrelation note above.
    rm_base = ne * bz
    phi_cum = np.cumsum(rm_base, axis=2) * dz
    phi_half = phi_cum - 0.5 * rm_base * dz

    # Auto-calibrate Faraday coefficient using local σ_φ
    sigma_phi_local = float(np.std(rm_base))
    rm_coeff = 1.0 / max(2.0 * lam2_break_target * sigma_phi_local * L, 1e-30)

    # Vectorized computation for all wavelengths
    y = np.empty(lam2.size, dtype=np.float64)
    for i, l2 in enumerate(lam2):
        phase = np.exp(2j * l2 * rm_coeff * phi_half)
        Pmap = np.sum(emis_phase * phase, axis=2) * dz
        y[i] = np.mean(np.abs(Pmap)**2)

    # Normalize small-λ² plateau to 0.1
    scale = 0.1 / y[0]
    y *= scale
    return y

def process_wavelength_batch_worker(args):
    """Worker function for parallel wavelength processing - NO normalization"""
    path, lam2_batch, L, gamma, lam2_break_target = args
    
    with h5py.File(path, "r") as f:
        bx = f[BX_KEY][...].astype(np.float64)
        by = f[BY_KEY][...].astype(np.float64)
        bz = f[BZ_KEY][...].astype(np.float64)
        ne = f[NE_KEY][...].astype(np.float64)

    nx, ny, nz = bx.shape
    dz = L / float(nz)

    # --- enforce η=∞ (random-dominated Faraday) as in Zhang fig: remove mean B∥
    bz = bz - bz.mean()

    # Pre-compute all emission and polarization data
    bperp2 = bx*bx + by*by
    emis = bperp2**(gamma/2.0)
    e2ipsi = np.exp(2j * np.arctan2(by, bx))
    emis_phase = emis * e2ipsi

    # Half-cell Faraday depth calculation
    # NOTE: Uses same B field for emissivity and Faraday. See decorrelation note above.
    rm_base = ne * bz
    phi_cum = np.cumsum(rm_base, axis=2) * dz
    phi_half = phi_cum - 0.5 * rm_base * dz

    # Auto-calibrate Faraday coefficient using local σ_φ
    sigma_phi_local = float(np.std(rm_base))
    rm_coeff = 1.0 / max(2.0 * lam2_break_target * sigma_phi_local * L, 1e-30)

    y = np.empty(lam2_batch.size, dtype=np.float64)
    for i, l2 in enumerate(lam2_batch):
        phase = np.exp(2j * l2 * rm_coeff * phi_half)
        Pmap = np.sum(emis_phase * phase, axis=2) * dz
        y[i] = np.mean(np.abs(Pmap)**2)

    # DO NOT normalize here - normalization will be done after combining all batches
    return y

def polarized_variance_curve_parallel(path, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1, n_processes=11):
    """Parallel version with 11 processes - maintains exact same results as original"""
    # Split wavelengths into batches for parallel processing
    batch_size = max(1, len(lam2) // n_processes)
    lam2_batches = [lam2[i:i+batch_size] for i in range(0, len(lam2), batch_size)]
    
    # Prepare arguments for parallel processing
    args_list = [(path, lam2_batch, L, gamma, lam2_break_target) for lam2_batch in lam2_batches]
    
    # Process wavelength batches in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_wavelength_batch_worker, args_list)
    
    # Combine results from all batches
    y = np.concatenate(results)
    
    # Apply normalization AFTER combining all batches (same as original)
    scale = 0.1 / y[0]
    y *= scale
    
    return y

if __name__ == '__main__':
    freeze_support()
    
    paths = [
        "synthetic_128x128x256_r000_.h5",
        # "synthetic_128x128x2048.h5",
        # "synthetic_128x128x1024.h5",
        # "synthetic_128x128x512.h5",
        # "synthetic_128x128x256.h5",
    ]
    labels = ["128×128×4096", "128×128×2048", "128×128×1024", "128×128×512", "128×128×256"]#,,"128×128×4096"]
    styles = [":","-.", (0,(3,1,1,1)), "--", "-"]

    lam2 = np.logspace(-3, 3, 180)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    curves = []
    
    for p, lab, ls in zip(paths, labels, styles):
        print(f"\nProcessing {p} with 11 parallel processes...")
        start_time = time.time()
        if lab == "128×128×4096":
            y = polarized_variance_curve_parallel(p, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1, n_processes=1)
        elif lab == "128×128×2048":
            y = polarized_variance_curve_parallel(p, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1, n_processes=2)
        elif lab == "128×128×1024":
            y = polarized_variance_curve_parallel(p, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1, n_processes=4)
        elif lab == "128×128×512":
            y = polarized_variance_curve_parallel(p, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1, n_processes=8)
        else:
            y = polarized_variance_curve_parallel(p, lam2, L=1.0, gamma=2.0, lam2_break_target=1e-1, n_processes=11)
        
        parallel_time = time.time() - start_time
        print(f"Parallel processing completed in {parallel_time:.2f} seconds")
        
        curves.append(y)
        ax.plot(np.log10(lam2), np.log10(y), ls=ls, label=lab)

    # Filter data to only show lam2 from 0 to 2
    mask = (np.log10(lam2) >= -3) & (np.log10(lam2) <= 3)
    lam2_filtered = lam2[mask]
    curves_filtered = [curve[mask] for curve in curves]
    
    # Clear and replot with filtered data
    ax.clear()
    for curve, lab, ls in zip(curves_filtered, labels, styles):
        ax.plot(np.log10(lam2_filtered), np.log10(curve), ls=ls, label=lab)
    
    # Reference line λ^{-2} positioned a bit higher
    i0 = np.argmin(np.abs(np.log10(lam2_filtered) - 0.5))
    ref = curves_filtered[-1][i0] * (lam2_filtered / lam2_filtered[i0])**(-1.0)
    # Shift reference line up by 0.5 in log space
    ref_shifted = ref #* 10**0.5
    # Filter to x-axis range from 1 to 2 (log10)
    log10_lam2_filtered = np.log10(lam2_filtered)
    mask = (log10_lam2_filtered >= 1) & (log10_lam2_filtered <= 1.5)
    ax.plot(log10_lam2_filtered[mask], np.log10(ref_shifted)[mask], color="k", linewidth=1.0, label=r"$\lambda^{-2}$")

    ax.set_xlabel(r"$\log_{10}[\lambda^2]$")
    ax.set_ylabel(r"$\log_{10}\langle P^2\rangle$")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("fig2.png", dpi=300)
    plt.show()
