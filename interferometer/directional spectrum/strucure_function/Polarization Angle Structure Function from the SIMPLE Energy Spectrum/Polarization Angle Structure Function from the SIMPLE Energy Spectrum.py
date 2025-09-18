#!/usr/bin/env python3
import os, h5py, numpy as np, matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fftn, ifft2, fft2, fftfreq

@dataclass
class Cfg:
    h5_path: str = "../two_slope_2D_s3_r00.h5"
    outdir:  str = "fig/dphi_no_shellavg"
    C_RM: float = 0.81
    lambdas = (0.21, 0.50, 1.0)
    nbins_R: int = 300
    R_min_pix: float = 1.0
    R_max_frac: float = 0.45
    dpi: int = 160

cfg = Cfg()

def load_cube(path):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = dz = 1.0
    return ne, bz, dx, dz

def radial_average_map(Map2D, nbins, r_min_pix, r_max_frac):
    ny, nx = Map2D.shape
    yy = (np.arange(ny) - ny//2)[:, None]
    xx = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(yy, xx)
    rmax = R.max() * float(r_max_frac)
    eps = 1e-8
    bins = np.logspace(np.log10(max(r_min_pix, eps)), np.log10(rmax), nbins+1)
    idx = np.digitize(R.ravel(), bins) - 1
    vals = Map2D.ravel()
    good = (idx >= 0) & (idx < nbins) & np.isfinite(vals)
    sums = np.bincount(idx[good], weights=vals[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)
    prof = np.full(nbins, np.nan)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    rcent = 0.5*(bins[1:] + bins[:-1])
    sel = nz & (rcent >= r_min_pix)
    return rcent[sel], prof[sel]

def build_phi(ne, bz, dz, C_RM):
    return C_RM * (ne * bz).sum(axis=2) * dz

def dphi_from_energy_kz0(ne, bz, dx, dz, C_RM, lam, nbins_R, R_min_pix, R_max_frac):
    """
    No shell-averaging:
      P2_model(kx,ky) := |FFT3[q](kx,ky,kz=0)|^2  with q = n_e B_z
      C = IFFT2(P2_model) / (Nx Ny), renorm so C(0)=σ_Φ^2
      D_Φ(R) = 2(σ_Φ^2 - C(R)),  D_φ(R) = 1/2 [1 - exp(-2 λ^4 D_Φ)]
    """
    q = ne * bz                                    # (nz, ny, nx)
    nz, ny, nx = q.shape

    # 3-D FFT and take the kz=0 slice —> 2-D model power on the map grid
    Qk = fftn(q)
    kz0 = 0  # assumes fftfreq ordering so index 0 is kz=0
    P2_model = (Qk[kz0, :, :] * np.conj(Qk[kz0, :, :])).real

    # Correlation from 2D power; note 1/(Nx Ny) for our FFT convention
    C_map_unnorm = ifft2(P2_model).real
    C0_model = C_map_unnorm[0, 0] / (nx * ny)

    # Measured Φ and its variance set the amplitude
    Phi = build_phi(ne, bz, dz, C_RM)
    sigma_phi2 = Phi.var(ddof=0)

    A = (sigma_phi2 / C0_model) if (np.isfinite(C0_model) and C0_model != 0.0) else 1.0
    C_map = np.fft.fftshift(A * C_map_unnorm / (nx * ny))

    # Radial average in REAL space (only place we bin)
    R_pix, C_R = radial_average_map(C_map, nbins_R, R_min_pix, R_max_frac)

    # Structure functions
    D_Phi = np.maximum(2.0 * (sigma_phi2 - C_R), 0.0)
    D_phi = 0.5 * (1.0 - np.exp(-2.0 * (lam**4) * D_Phi))
    return R_pix, D_phi, D_Phi, C_R, sigma_phi2

def main():
    os.makedirs(cfg.outdir, exist_ok=True)
    ne, bz, dx, dz = load_cube(cfg.h5_path)
    print(f"[load] ne,bz shapes: {ne.shape}, dx={dx}, dz={dz}")

    for i, lam in enumerate(cfg.lambdas):
        R, Dphi, DPhi, C_R, s2 = dphi_from_energy_kz0(
            ne, bz, dx, dz, cfg.C_RM, lam,
            nbins_R=cfg.nbins_R, R_min_pix=cfg.R_min_pix, R_max_frac=cfg.R_max_frac
        )
        print(f"[λ={lam:.2f} m]  max D_Φ={np.nanmax(DPhi):.6e}, max D_φ={np.nanmax(Dphi):.6e}")

        plt.figure(figsize=(7.0,5.2))
        m = np.isfinite(R) & np.isfinite(Dphi) & (R>0) & (Dphi>0)
        plt.loglog(R[m], Dphi[m], '-', lw=2.0, label=fr"$\lambda={lam:.2f}$ m")
        
        # Add power law reference lines
        if np.any(m):
            R_ref = R[m]
            ref_idx = len(R_ref) // 2  # middle point for normalization
            
            exponent=5/3
            norm = Dphi[m][0] / (R_ref[0]**exponent)
            power_line = norm * R_ref**exponent
            half = len(R_ref) // 20
            R_half = R_ref[:half]
            power_half = power_line[:half]

            # Plot only the first half
            plt.loglog(R_half, power_half, '--', color='gray', alpha=0.7, lw=1.5, label=r"$R^{5/3}$")
            # plt.loglog(R_ref[:len(R_ref) // 2], power_line, '--', color='gray', alpha=0.7, lw=1.5, label=r"$R^{5/3}$")
            # if ref_idx < len(R_ref):
            #     if i == 0:  # First wavelength: R^(5/3)
            #         exponent = 5/3
            #         norm = Dphi[m][ref_idx] / (R_ref[ref_idx]**exponent)
            #         power_line = norm * R_ref**exponent
            #         plt.loglog(R_ref, power_line, '--', color='gray', alpha=0.7, lw=1.5, label=r"$R^{5/3}$")
            #     elif i == 1:  # Second wavelength: R^(3/2)
            #         exponent = 3/2
            #         norm = Dphi[m][ref_idx] / (R_ref[ref_idx]**exponent)
            #         power_line = norm * R_ref**exponent
            #         plt.loglog(R_ref, power_line, '--', color='gray', alpha=0.7, lw=1.5, label=r"$R^{3/2}$")
        
        plt.xlabel(r"$R$ (pixels)")
        plt.ylabel(r"$D_\varphi(R)$")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        fn = os.path.join(cfg.outdir, f"Dphi_kz0_{i}")
        plt.savefig(fn + ".png", dpi=cfg.dpi); plt.savefig(fn + ".pdf"); plt.close()

    print(f"Saved → {os.path.abspath(cfg.outdir)}")

if __name__ == "__main__":
    main()
