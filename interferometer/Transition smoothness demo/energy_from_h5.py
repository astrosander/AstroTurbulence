#!/usr/bin/env python3
"""
D_phi(R, λ) from a 2D ENERGY spectrum E1D(k) (isotropic model).

- Loads a 2D map from .h5 (dataset "k_mag_field", shape [Nx,Ny,1]).
- Measures E1D(k) = 2π k <P2D>_ring from that map.
- Builds a radially isotropic P2D_model(k) = E1D(k)/(2π k) on the FFT grid.
- Normalizes so that C_Phi(0) = σ_Φ^2 measured from Φ = C_RM * map (assuming ne=1, dz=1).
- Computes D_Φ(R) and D_φ(R, λ) and plots for chosen λ.

Change H5_PATH to your file.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq

# --------------------------- Config ----------------------------
H5_PATH   = "h5/transition_smoothness/two_slope_2D_s3_r00.h5"  # <-- change if needed
OUTDIR    = "fig/dphi_from_energy_spectrum_2D"
C_RM      = 0.81
LAMBDAS   = (0.21, 0.50, 1.00)   # meters
NBINS_R   = 240
R_MIN_PIX = 1.0
R_MAX_FR  = 0.45
DPI       = 170

# ------------------------ Utilities ----------------------------
def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int = 60, kmin: float = 1e-3):
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx)
    kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.hypot(KY, KX)

    kmax = K.max()
    bins = np.logspace(np.log10(max(kmin, 1e-8)), np.log10(kmax), nbins + 1)
    k = K.ravel(); p = P2D.ravel()
    idx = np.digitize(k, bins) - 1

    good = (idx >= 0) & (idx < nbins) & np.isfinite(p)
    sums = np.bincount(idx[good], weights=p[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)

    prof = np.full(nbins, np.nan, float)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]

    kcent = 0.5 * (bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def radial_average_map(Map2D: np.ndarray, nbins: int, r_min_pix: float, r_max_frac: float):
    ny, nx = Map2D.shape
    yy = (np.arange(ny) - ny//2)[:, None]
    xx = (np.arange(nx) - nx//2)[None, :]
    R_pix = np.hypot(yy, xx)
    rmax = R_pix.max() * float(r_max_frac)

    eps = 1e-8
    bins = np.logspace(np.log10(max(r_min_pix, eps)), np.log10(rmax), nbins+1)
    idx  = np.digitize(R_pix.ravel(), bins) - 1
    vals = Map2D.ravel()

    nb = nbins
    good = (idx >= 0) & (idx < nb) & np.isfinite(vals)
    sums = np.bincount(idx[good], weights=vals[good], minlength=nb)
    cnts = np.bincount(idx[good], minlength=nb)

    prof = np.full(nb, np.nan)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]

    rcent_pix = 0.5*(bins[1:] + bins[:-1])
    sel = nz & (rcent_pix >= r_min_pix)
    return rcent_pix[sel], prof[sel]

# ------------------------- Core steps --------------------------
def load_map_2d(h5_path: str):
    with h5py.File(h5_path, "r") as h5:
        f2 = h5["k_mag_field"][:]      # (Nx, Ny, 1)
        f2 = f2[:, :, 0].T             # → (ny, nx)
    dx = 1.0
    return f2.astype(np.float64), dx

def energy_spectrum_1d_from_map(f2: np.ndarray, dx: float):
    Fk  = fft2(f2)
    P2D = (Fk * np.conj(Fk)).real
    k1d, Pk = ring_average_2d(P2D, dx, nbins=80, kmin=1e-3)
    E1D = 2.0 * np.pi * k1d * Pk
    return k1d, E1D

def build_p2d_model_from_E1D(k1d: np.ndarray, E1D: np.ndarray, nx: int, ny: int, dx: float):
    """Radially isotropic P2D_model(K)=E1D(K)/(2π K) on the FFT grid."""
    kx = fftfreq(nx, d=dx) * 2.0*np.pi
    ky = fftfreq(ny, d=dx) * 2.0*np.pi
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    Kabs = np.hypot(KX, KY)

    lk = np.log(np.maximum(k1d, 1e-12))
    lE = np.log(np.maximum(E1D, 1e-300))

    def E_of_k(kabs):
        ksafe = np.maximum(kabs, k1d[0])
        return np.exp(np.interp(np.log(ksafe), lk, lE, left=lE[0], right=lE[-1]))

    E_grid = E_of_k(Kabs)
    P2_model = E_grid / (2.0 * np.pi * np.maximum(Kabs, 1e-12))
    P2_model[0,0] = 0.0
    return P2_model

def dphi_from_p2d_model(P2_model: np.ndarray, phi_map: np.ndarray, lam: float,
                        nbins_R: int, r_min_pix: float, r_max_frac: float):
    """Normalize to σ_Φ^2 from phi_map, then compute D_Φ and D_φ."""
    ny, nx = P2_model.shape
    C_unnorm = ifft2(P2_model).real
    C0_model = C_unnorm[0,0] / (nx*ny)
    sigma_phi2 = phi_map.var(ddof=0)

    A = (sigma_phi2 / C0_model) if (np.isfinite(C0_model) and C0_model != 0.0) else 1.0
    C_map = np.fft.fftshift(A * C_unnorm / (nx*ny))

    R_pix, C_R = radial_average_map(C_map, nbins_R, r_min_pix, r_max_frac)
    D_Phi = np.maximum(2.0 * (sigma_phi2 - C_R), 0.0)
    Dphi  = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * D_Phi))
    return R_pix, Dphi, (R_pix, D_Phi, C_R, sigma_phi2)

# --------------------------- Main ------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load 2D map and treat Φ = C_RM * map (ne=1,dz=1)
    f2, dx = load_map_2d(H5_PATH)
    Phi = C_RM * f2
    ny, nx = Phi.shape
    print(f"[load] map shape: {Phi.shape}, dx={dx}")

    # Energy spectrum from the 2D map
    k1d, E1D = energy_spectrum_1d_from_map(f2, dx)

    # Build isotropic 2D model spectrum from E1D
    P2_model = build_p2d_model_from_E1D(k1d, E1D, nx, ny, dx)

    # Plot E1D(k)
    plt.figure(figsize=(6.6,4.6))
    m = np.isfinite(k1d) & np.isfinite(E1D) & (k1d>0) & (E1D>0)
    plt.loglog(k1d[m], E1D[m], lw=1.8, label="Energy spectrum")
    
    # Add reference lines: flat (0 slope) on left half, -5/3 slope on right half
    k_valid = k1d[m]
    E_valid = E1D[m]
    if len(k_valid) > 10:  # Ensure we have enough points
        # Define transition point (middle of k range in log space)
        k_mid = np.sqrt(k_valid[0] * k_valid[-1])
        
        # Left half: flat line (0 slope)
        k_left = k_valid[k_valid <= k_mid]
        if len(k_left) > 0:
            # Use median energy value in left segment for reference
            E_ref_left = np.median(E_valid[k_valid <= k_mid])
            plt.loglog(k_left, np.full_like(k_left, E_ref_left), '--', 
                      color='red', alpha=0.7, label='slope = 0')
        
        # Right half: -5/3 slope
        k_right = k_valid[k_valid >= k_mid]
        if len(k_right) > 0:
            # Find reference point at k_mid to normalize the line
            E_ref_right = np.interp(k_mid, k_valid, E_valid)
            E_line_right = E_ref_right * (k_right / k_mid)**(-5/3)
            plt.loglog(k_right, E_line_right, '--', 
                      color='blue', alpha=0.7, label='slope = -5/3')
    
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E_{1\mathrm{D}}(k)$")
    plt.title("Energy spectrum from loaded 2D field")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "E1D_from_map.png"), dpi=DPI)
    plt.savefig(os.path.join(OUTDIR, "E1D_from_map.pdf"))
    plt.close()

    # Angle structure function from the energy-spectrum model
    for i, lam in enumerate(LAMBDAS):
        R, Dphi, (R_D, D_Phi, C_R, s2) = dphi_from_p2d_model(
            P2_model=P2_model,
            phi_map=Phi,
            lam=lam,
            nbins_R=NBINS_R,
            r_min_pix=R_MIN_PIX,
            r_max_frac=R_MAX_FR
        )
        print(f"[λ={lam:.2f} m]  σ_Φ^2={s2:.3e}, max D_Φ={np.nanmax(D_Phi):.3e}, max D_φ={np.nanmax(Dphi):.3e}")

        plt.figure(figsize=(7.0,5.2))
        mR = np.isfinite(R) & np.isfinite(Dphi) & (R>0) & (Dphi>=0)
        plt.loglog(R[mR], np.maximum(Dphi[mR], 1e-300), lw=2.0, label=fr"$\lambda={lam:.2f}$ m")
        
        # Add R^5/3 reference line in central segment
        R_valid = R[mR]
        Dphi_valid = Dphi[mR]
        if len(R_valid) > 10:
            # Define central segment (middle third of R range in log space)
            R_min_central = np.percentile(R_valid, 33)  # Start at 33rd percentile
            R_max_central = np.percentile(R_valid, 67)  # End at 67th percentile
            
            # Create R^5/3 reference line in central segment
            R_central = R_valid[(R_valid >= R_min_central) & (R_valid <= R_max_central)]
            if len(R_central) > 0:
                # Find reference point to normalize the line (use median point)
                R_ref = np.median(R_central)
                # Find corresponding Dphi value at reference point
                Dphi_ref = np.interp(R_ref, R_valid, Dphi_valid)
                # Create R^5/3 line: Dphi ∝ R^(5/3)
                Dphi_line = Dphi_ref * (R_central / R_ref)**(1/3)
                plt.loglog(R_central, Dphi_line, '--', color='gray', alpha=0.8, 
                          label=r'$R^{5/3}$', linewidth=1.5)
        
        plt.xlabel(r"$R$ (pixels)")
        plt.ylabel(r"$D_\varphi(R)$")
        plt.title("Angle structure function from energy spectrum (isotropic)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        out = os.path.join(OUTDIR, f"Dphi_from_E1D_{i}")
        plt.savefig(out + ".png", dpi=DPI)
        plt.savefig(out + ".pdf")
        plt.close()

    print(f"Saved → {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
