import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count, shared_memory, freeze_support
import functools
import os

BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

def process_z_slice_parallel(args):
    """Process a single z-slice with all wavelengths in parallel"""
    k, bx_slice, by_slice, bz_slice, ne_slice, lam2, gamma, phi_prev = args
    
    # Compute emission and polarization angle for this slice
    b_perp_sq = bx_slice**2 + by_slice**2
    emis = b_perp_sq**(gamma/2.0)
    psi = np.arctan2(by_slice, bx_slice)
    e2ipsi = np.exp(2j * psi)
    
    # Update Faraday depth
    phi = phi_prev + (ne_slice * bz_slice).astype(np.float64)
    
    # Compute phase factor for all wavelengths at once
    phase_factor = np.exp(2j * lam2[:, None, None] * phi[None, :, :])
    # Compute polarization factor for this z-slice
    pol_factor = emis * e2ipsi
    # Return contribution for all wavelengths
    contribution = phase_factor * pol_factor[None, :, :]
    
    return contribution, phi

def process_wavelength_batch(args):
    """Process a batch of wavelengths with full vectorization"""
    lam2_batch, bx, by, bz, ne, gamma = args
    nx, ny, nz = bx.shape
    
    # Pre-compute all emission and polarization data
    b_perp_sq = bx**2 + by**2  # Shape: (nx, ny, nz)
    emis = b_perp_sq**(gamma/2.0)  # Shape: (nx, ny, nz)
    psi = np.arctan2(by, bx)  # Shape: (nx, ny, nz)
    e2ipsi = np.exp(2j * psi)  # Shape: (nx, ny, nz)
    
    # Vectorized computation of P for this wavelength batch
    lam2_batch = lam2_batch.astype(np.float64)
    P_batch = np.zeros((lam2_batch.size, nx, ny), dtype=np.complex128)
    phi = np.zeros((nx, ny), dtype=np.float64)
    
    # Process z-slices maintaining the original algorithm logic
    for k in range(nz):
        # Accumulate Faraday depth
        phi += (ne[:, :, k] * bz[:, :, k]).astype(np.float64)
        
        # Compute phase factor for all wavelengths in this batch at once
        phase_factor = np.exp(2j * lam2_batch[:, None, None] * phi[None, :, :])
        # Compute polarization factor for this z-slice
        pol_factor = emis[:, :, k] * e2ipsi[:, :, k]
        # Accumulate contribution
        P_batch += phase_factor * pol_factor[None, :, :]
    
    p2_batch = np.mean(np.abs(P_batch)**2, axis=(1, 2))
    return p2_batch

def process_wavelength_batch_chunked(args):
    """Memory-efficient worker function that processes data in chunks"""
    lam2_batch, path, gamma = args
    
    # Load data dimensions first
    with h5py.File(path, "r") as f:
        nx, ny, nz = f[BX_KEY].shape
    
    # Process data in smaller chunks to reduce memory usage
    chunk_size = min(64, nz)  # Process max 64 z-slices at a time
    lam2_batch = lam2_batch.astype(np.float64)
    P_batch = np.zeros((lam2_batch.size, nx, ny), dtype=np.complex128)
    phi = np.zeros((nx, ny), dtype=np.float64)
    
    # Process z-slices in chunks to reduce memory usage
    for start_k in range(0, nz, chunk_size):
        end_k = min(start_k + chunk_size, nz)
        
        # Load only the current chunk of data
        with h5py.File(path, "r") as f:
            bx_chunk = f[BX_KEY][:, :, start_k:end_k].astype(np.float64)
            by_chunk = f[BY_KEY][:, :, start_k:end_k].astype(np.float64)
            bz_chunk = f[BZ_KEY][:, :, start_k:end_k].astype(np.float64)
            ne_chunk = f[NE_KEY][:, :, start_k:end_k].astype(np.float64)
        
        # Process this chunk
        for k in range(end_k - start_k):
            # Accumulate Faraday depth
            phi += (ne_chunk[:, :, k] * bz_chunk[:, :, k]).astype(np.float64)
            
            # Compute emission and polarization for this slice
            b_perp_sq = bx_chunk[:, :, k]**2 + by_chunk[:, :, k]**2
            emis = b_perp_sq**(gamma/2.0)
            psi = np.arctan2(by_chunk[:, :, k], bx_chunk[:, :, k])
            e2ipsi = np.exp(2j * psi)
            
            # Compute phase factor for all wavelengths in this batch at once
            phase_factor = np.exp(2j * lam2_batch[:, None, None] * phi[None, :, :])
            # Compute polarization factor for this z-slice
            pol_factor = emis * e2ipsi
            # Accumulate contribution
            P_batch += phase_factor * pol_factor[None, :, :]
    
    p2_batch = np.mean(np.abs(P_batch)**2, axis=(1, 2))
    return p2_batch

def curve_parallel(path, lam2, gamma=2.0, n_processes=11):
    """Memory-efficient parallel version with chunked processing"""
    # Split wavelengths into batches for parallel processing
    batch_size = max(1, len(lam2) // n_processes)
    lam2_batches = [lam2[i:i+batch_size] for i in range(0, len(lam2), batch_size)]
    
    # Prepare arguments for parallel processing
    args_list = [(lam2_batch, path, gamma) for lam2_batch in lam2_batches]
    
    # Process wavelength batches in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_wavelength_batch_chunked, args_list)
    
    # Combine results from all batches
    p2 = np.concatenate(results)
    return p2

def curve(path, lam2, gamma=2.0):
    """Original sequential version for comparison"""
    with h5py.File(path, "r") as f:
        # Load all data at once for better memory access patterns
        bx = f[BX_KEY][...]  # Shape: (nx, ny, nz)
        by = f[BY_KEY][...]
        bz = f[BZ_KEY][...]
        ne = f[NE_KEY][...]
        nx, ny, nz = bx.shape
        
        # Pre-compute emission and polarization angle for all z-slices
        b_perp_sq = bx**2 + by**2  # Shape: (nx, ny, nz)
        emis = b_perp_sq**(gamma/2.0)  # Shape: (nx, ny, nz)
        psi = np.arctan2(by, bx)  # Shape: (nx, ny, nz)
        
        # Pre-compute complex exponentials to avoid repeated calculations
        e2ipsi = np.exp(2j * psi)  # Shape: (nx, ny, nz)
        
        # Vectorized computation of P
        lam2 = lam2.astype(np.float64)
        P = np.zeros((lam2.size, nx, ny), dtype=np.complex128)
        phi = np.zeros((nx, ny), dtype=np.float64)
        
        # Process z-slices maintaining the original algorithm logic
        for k in range(nz):
            # Accumulate Faraday depth (this is the key fix!)
            phi += (ne[:, :, k] * bz[:, :, k]).astype(np.float64)
            
            # Compute phase factor for this z-slice using accumulated phi
            phase_factor = np.exp(2j * lam2[:, None, None] * phi[None, :, :])
            # Compute polarization factor for this z-slice
            pol_factor = emis[:, :, k] * e2ipsi[:, :, k]
            # Accumulate contribution
            P += phase_factor * pol_factor[None, :, :]
        
        p2 = np.mean(np.abs(P)**2, axis=(1, 2))
        return p2

if __name__ == '__main__':
    freeze_support()
    
    paths = [
        "synthetic_128x128x256.h5",
        "synthetic_128x128x512.h5",
        "synthetic_128x128x1024.h5",
        "synthetic_128x128x2048.h5",
        "synthetic_128x128x4096.h5",
    ]
    labels = [
        "128×128×256",
        "128×128×512",
        "128×128×1024",
        "128×128×2048",
        "128×128×4096",
    ]
    styles = [":", "-.", "--", "-", (0, (5, 1, 1, 1))]

    lam2 = np.logspace(-3, 3, 180)
    fig, ax = plt.subplots(figsize=(7, 5))
    curves = []

    # Test both sequential and parallel versions for comparison
    for p, lab, ls in zip(paths, labels, styles):
        print(f"\nProcessing {p}...")
        
        # Sequential version
        # print("Running sequential version...")
        # start_time = time.time()
        # y_seq = curve(p, lam2, gamma=2.0)
        # seq_time = time.time() - start_time
        # print(f"Sequential completed in {seq_time:.2f} seconds")
        
        # Parallel version
        print("Running parallel version with 11 processes...")
        start_time = time.time()
        y_parallel = curve_parallel(p, lam2, gamma=2.0, n_processes=11)
        parallel_time = time.time() - start_time
        print(f"Parallel completed in {parallel_time:.2f} seconds")
        
        # Verify results are the same
        # if np.allclose(y_seq, y_parallel, rtol=1e-10):
        #     print("✓ Results match between sequential and parallel versions")
        # else:
        #     print("⚠ Warning: Results differ between sequential and parallel versions")
        
        # speedup = seq_time / parallel_time
        # print(f"Speedup: {speedup:.2f}x")
        
        curves.append(y_parallel)
        ax.plot(np.log10(lam2), np.log10(y_parallel), ls=ls, label=lab)

    ref_idx = np.argmin(np.abs(np.log10(lam2) - 1.0))
    y0 = curves[-1][ref_idx]
    ref = y0 * (lam2 / lam2[ref_idx])**(-2.0)
    ax.plot(np.log10(lam2), np.log10(ref), linewidth=1.0)

    ax.set_xlabel(r"$\log_{10}[\lambda^2]$")
    ax.set_ylabel(r"$\log_{10}\langle P^2\rangle$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("fig2.png", dpi=300)
    plt.show()
