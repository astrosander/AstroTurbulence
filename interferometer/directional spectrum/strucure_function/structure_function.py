#!/usr/bin/env python3
"""
Analyze a two-slope 3D cube (provided) and show what an observer sees in 2D
===========================================================================

Inputs (from your generator's HDF5 file):
  gas_density : n_e(x,y,z)
  k_mag_field : B_z(x,y,z)
  x_coor, z_coor (optional for dx,dz)

Outputs (fig/two_slope_observer/):
  - proj_Bz_map.{png,pdf}         : projected Bz map (sum over z)
  - proj_RM_map.{png,pdf}         : Faraday screen Φ=0.81∑ n_e B_z dz
  - proj_Bz_E1D.{png,pdf}         : E1D(k) of projected Bz with slope guides
  - proj_RM_E1D.{png,pdf}         : E1D(k) of Φ with slope guides
  - Pdir_ring_{i}.{png,pdf}       : directional spectrum from f=λ^2Φ (per λ)
  - S_of_R_{i}.{png,pdf}          : S(R)=IFFT{P_dir} (per λ)
  - Dphi_of_R_{i}.{png,pdf}       : Dφ(R)=½[1−S(R)] (per λ)

Notes
-----
• Your generator targets **3D shell slopes** α_low=+3/2, α_high=−5/3 in E_1D^(3D).
  For a simple LOS sum/projection to the sky (projection–slice theorem),
  the **observed 2D shell slope** is α^(2D)=α^(3D)−1. We plot guides at:
    low-k:  +1/2,  high-k:  −8/3,  anchored near k≈k_break.

• The directional part follows:
    A=cos(2f), B=sin(2f),  P_dir=|FFT(A)|^2+|FFT(B)|^2,
    S_map = IFFT2(P_dir)/Npix (Wiener–Khinchin, circular normalization),
    Dφ(R)=½[1−S(R)] after radial averaging.

Edit CONFIG below and run.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
from numpy.fft import fft2, ifft2, fftfreq

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # h5_path: str = "../../../faradays_angles_stats/lp_structure_tests/mhd_fields.h5"  # ← output of your generator
    h5_path: str = "../../../faradays_angles_stats/lp_structure_tests/ms01ma08.mhd_w.00300.vtk.h5"  # ← output of your generator
    outdir: str = "fig/two_slope_observer"

    # plotting & binning
    nbins_k: int = 240
    nbins_R: int = 240
    kmin: float = 1e-3
    kmax_frac: float = 1.0
    R_min: float = 1e-2
    R_max_frac: float = 0.45
    dpi: int = 160

    # physical
    C_RM: float = 0.81          # RM factor (rad m^-2 pc^-1 cm^3 μG^-1), scalar OK

    # wavelengths to analyze (meters); feel free to add/remove
    lambdas_m: Tuple[float, ...] = (0.06, 0.11, 0.21)

    # For slope guides on 2D spectra (projection–slice: α_2D = α_3D − 1)
    alpha3D_low: float = +1.5
    alpha3D_high: float = -5.0/3.0
    k_break_cyc: float = 0.06   # cycles per dx, same as in generator
    guide_span: float = 8.0     # span× / ÷ around k_break for the guide lines

C = Config()

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing_from_h5(arr, name: str, fallback: float) -> float:
    try:
        u = np.unique(arr.ravel())
        dif = np.diff(np.sort(u))
        dif = dif[dif > 0]
        if dif.size:
            return float(np.median(dif))
    except Exception:
        pass
    print(f"[!] {name}: using fallback dx={fallback}")
    return float(fallback)

def load_cube(path: str):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
        # dx, dz (units arbitrary but consistent)
        if "x_coor" in f:
            dx = _axis_spacing_from_h5(f["x_coor"][:,0,0], "x_coor", 1.0)
        else:
            dx = 1.0
        if "z_coor" in f:
            dz = _axis_spacing_from_h5(f["z_coor"][0,0,:], "z_coor", 1.0)
        else:
            dz = 1.0
        dx=1.0
        dz=1.0
    return ne, bz, dx, dz

def project_maps(ne: np.ndarray, bz: np.ndarray, dz: float, C_RM: float):
    """
    Observer-like maps on the sky:
      Bz_proj(x,y) = ∑_z B_z
      Φ(x,y)       = 0.81 ∑_z n_e B_z dz
    """
    Bz_proj = bz.sum(axis=2)
    Phi = C_RM * (ne * bz).sum(axis=2) * dz
    return Bz_proj, Phi

def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    kmax = float(kmax_frac) * K.max()
    bins = np.logspace(np.log10(max(kmin, 1e-8)), np.log10(kmax), nbins+1)
    idx = np.digitize(K.ravel(), bins) - 1
    p = P2D.ravel()
    nb = nbins
    good = (idx>=0) & (idx<nb) & np.isfinite(p)
    sums  = np.bincount(idx[good], weights=p[good], minlength=nb)
    cnts  = np.bincount(idx[good], minlength=nb)
    prof  = np.full(nb, np.nan, float)
    nz    = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    kcent = 0.5*(bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def radial_average_map(Map2D: np.ndarray, dx: float, nbins: int, r_min: float, r_max_frac: float):
    ny, nx = Map2D.shape
    y = (np.arange(ny) - ny//2)[:, None]
    x = (np.arange(nx) - nx//2)[None, :]
    R = np.hypot(y, x) * dx
    rmax = R.max() * float(r_max_frac)
    bins = np.logspace(np.log10(max(r_min, 1e-8)), np.log10(rmax), nbins+1)
    idx = np.digitize(R.ravel(), bins) - 1
    m = Map2D.ravel()
    nb = nbins
    good = (idx>=0) & (idx<nb) & np.isfinite(m)
    sums  = np.bincount(idx[good], weights=m[good], minlength=nb)
    cnts  = np.bincount(idx[good], minlength=nb)
    prof  = np.full(nb, np.nan, float)
    nz    = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    rcent = 0.5*(bins[1:] + bins[:-1])
    sel = nz & (rcent > r_min)
    return rcent[sel], prof[sel]

def E1D_from_map(Map2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    F = fft2(Map2D)
    P2 = (F * np.conj(F)).real
    k1d, Pk = ring_average_2d(P2, dx, nbins, kmin, kmax_frac)
    E1D = 2*np.pi * k1d * Pk
    return k1d, E1D, P2

def directional_spectrum_and_correlation(f: np.ndarray):
    A = np.cos(2.0 * f)
    B = np.sin(2.0 * f)
    FA = fft2(A); FB = fft2(B)
    # 2D power of the complex spin-2 unit field e^{i2f}
    P2_dir = (FA * np.conj(FA)).real + (FB * np.conj(FB)).real

    # Autocorrelation via Wiener–Khinchin, normalized so S(0)=1 exactly
    power0 = np.sum(A*A + B*B)                 # should be ~ Npix, but compute explicitly
    S_map  = np.fft.fftshift(ifft2(P2_dir).real) / power0

    return P2_dir, S_map, power0

# ──────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────

def _save_imshow(img, path, title, cmap="viridis"):
    plt.figure(figsize=(5.4,4.8))
    plt.imshow(img, origin="lower", cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path+".png", dpi=C.dpi); plt.savefig(path+".pdf")
    plt.close()

def _slope_guides(ax, k_break, yref, span, slopes_with_labels):
    kr = np.logspace(np.log10(k_break/span), np.log10(k_break*span), 200)
    for s, lab in slopes_with_labels:
        ax.loglog(kr, yref*(kr/k_break)**s, "--", lw=1.0, alpha=0.8, label=lab)

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    ne, bz, dx, dz = load_cube(C.h5_path)
    # print(dx*256)
    # dx=1.0
    print(f"[loaded] ne,bz shapes: {ne.shape}  dx={dx}  dz={dz}")

    # Project to sky plane
    Bz_proj, Phi = project_maps(ne, bz, dz, C.C_RM)

    # Save maps
    _save_imshow(Bz_proj, os.path.join(C.outdir, "proj_Bz_map"), "Projected $B_z$ (sum over z)")
    _save_imshow(Phi,     os.path.join(C.outdir, "proj_RM_map"), "Faraday screen $\\Phi=0.81\\sum n_e B_z\\,dz$")

    # Ring spectra (2D observer). Expect α_2D = α_3D − 1 → (+1/2, −8/3).
    kB = C.k_break_cyc
    slopes_2D = [(C.alpha3D_low-1.0, r"$+1/2$ guide"),
                 (C.alpha3D_high-1.0, r"$-8/3$ guide")]

    # E1D: projected Bz
    k_bz, E_bz, P2_bz = E1D_from_map(Bz_proj, dx, C.nbins_k, C.kmin, C.kmax_frac)
    plt.figure(figsize=(6.6,5.0))
    plt.loglog(k_bz, E_bz, lw=1.8, label=r"$E_{1\rm D}$ of $\sum_z B_z$")
    yref = np.interp(kB, k_bz, E_bz)
    _slope_guides(plt.gca(), kB, yref, C.guide_span, slopes_2D)
    plt.axvline(kB, color='k', ls=':', lw=1.0, label=r"$k_b$")
    plt.xlabel(r"$k$ (cycles/dx)"); plt.ylabel(r"$E_{1\rm D}(k)$")
    plt.title("Projected $B_z$ spectrum (2D)")
    plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "proj_Bz_E1D.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "proj_Bz_E1D.pdf")); plt.close()

    # E1D: RM (Phi)
    k_phi, E_phi, P2_phi = E1D_from_map(Phi, dx, C.nbins_k, C.kmin, C.kmax_frac)
    plt.figure(figsize=(6.6,5.0))
    plt.loglog(k_phi, E_phi, lw=1.8, label=r"$E_{1\rm D}$ of $\Phi$")
    yref = np.interp(kB, k_phi, E_phi)
    _slope_guides(plt.gca(), kB, yref, C.guide_span, slopes_2D)
    plt.axvline(kB, color='k', ls=':', lw=1.0, label=r"$k_b$")
    plt.xlabel(r"$k$ (cycles/dx)"); plt.ylabel(r"$E_{1\rm D}(k)$")
    plt.title("Faraday screen spectrum (2D)")
    plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "proj_RM_E1D.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir, "proj_RM_E1D.pdf")); plt.close()

    # Directional spectrum/correlation from f=λ^2 Φ
    for i, lam in enumerate(C.lambdas_m):
        f_map = (lam**2) * Phi

        # P2_dir, S_map = directional_spectrum_and_correlation(f_map)
        P2_dir, S_map, power0 = directional_spectrum_and_correlation(f_map)

        # P_dir ring spectrum
        kdir, Pdir_ring = ring_average_2d(P2_dir, dx, C.nbins_k, C.kmin, C.kmax_frac)
        # plt.figure(figsize=(6.6,5.0))
        # plt.loglog(kdir, Pdir_ring, lw=1.8, label=fr"$P_{{\rm dir}}(k)$, $\lambda={lam:.2f}$ m")
        # # plt.axvline(kB, color='k', ls=':', lw=1.0, label=r"$k_b$")
        # plt.xlabel(r"$k$"); plt.ylabel(r"$\langle |\widehat{\cos2f}|^2 + |\widehat{\sin2f}|^2\rangle$")
        # plt.title("Directional spectrum")
        # # plt.xlim(1,256/(3.1415/2))
        # plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        # plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"Pdir_ring_{i}.png"), dpi=C.dpi)
        # plt.savefig(os.path.join(C.outdir, f"Pdir_ring_{i}.pdf")); plt.close()
        plt.figure(figsize=(6.6,5.0))
        plt.loglog(kdir, Pdir_ring, lw=1.8, label=fr"$P_{{\rm dir}}(k)$, $\lambda={lam:.2f}$ m")

        # ── NEW: one -11/3 guide line across the plotted k-range, anchored to the spectrum
        valid = np.isfinite(kdir) & np.isfinite(Pdir_ring) & (kdir > 0)
        if np.count_nonzero(valid) > 5:
            s = -11.0/3.0
            # geometric-mean anchor within the valid range
            kref = np.exp(np.mean(np.log(kdir[valid])))
            yref = np.interp(kref, kdir[valid], Pdir_ring[valid])
            kg = np.logspace(np.log10(kdir[valid].min()), np.log10(kdir[valid].max()), 200)
            plt.loglog(kg, yref*(kg/kref)**s, "--", lw=1.0, alpha=0.9, label=r"$-11/3$")
        # if np.count_nonzero(valid) > 5:
        #     s = -5.0/3.0
        #     # geometric-mean anchor within the valid range
        #     kref = np.exp(np.mean(np.log(kdir[valid])))
        #     yref = np.interp(kref, kdir[valid], Pdir_ring[valid])
        #     kg = np.logspace(np.log10(kdir[valid].min()), np.log10(kdir[valid].max()), 200)
        #     plt.loglog(kg, yref*(kg/kref)**s, "--", lw=1.0, alpha=0.9, label=r"$-5/3$")
        # ── END NEW

        # plt.axvline(kB, color='k', ls=':', lw=1.0, label=r"$k_b$")
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\langle |\widehat{\cos2f}|^2 + |\widehat{\sin2f}|^2\rangle$")
        plt.title("Directional spectrum")
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(C.outdir, f"Pdir_ring_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Pdir_ring_{i}.pdf"))
        plt.close()

        # S(R) and D_phi(R)
        R1d, S_R = radial_average_map(S_map, dx, C.nbins_R, C.R_min, C.R_max_frac)
        Dphi_R = 0.5 * (1.0 - S_R)

        plt.figure(figsize=(6.6,5.0))
        plt.loglog(R1d, S_R, lw=1.8, label=fr"$S(R)$, $\lambda={lam:.2f}$ m")
        plt.xlabel(r"$R$ (dx)"); plt.ylabel(r"$S(R)$")
        plt.title("Directional correlation")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"S_of_R_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"S_of_R_{i}.pdf")); plt.close()

        plt.figure(figsize=(6.6,5.0))
        Dphi_R_scaled = Dphi_R / (lam**4)
        # Dphi_R=Dphi_R / (lam**4)
        # plt.loglog(R1d, Dphi_R_scaled, lw=1.8, label=fr"$D_\varphi/\lambda^4$, $\lambda={lam:.2f}$ m")


        plt.loglog(R1d, Dphi_R, lw=2.2, color="#1f77b4", label=fr"$D_\varphi(R)$")#.loglog(R1d, Dphi_R, lw=1.8, label=fr"$D_\varphi(R)$")#, $\lambda={lam:.2f}$ m
        # add ∝ R^{5/3} guide just as a visual reference (halo-like regime)
        if len(R1d) > 5:
            Rref, yref = R1d[2], Dphi_R[2]
            A = yref / (Rref**(5/3))
            Rg = np.logspace(np.log10(R1d[0]), np.log10(Rref*10), 200)
            plt.loglog(Rg, A*Rg**(5/3), "--", lw=1.6, color="#ff7f0e",
              alpha=0.9, label=r"$\propto R^{5/3}$")
        # if len(R1d) > 5:
        #     Rref = R1d[len(R1d)//8]
        #     yref = np.interp(Rref, R1d, Dphi_R_scaled)
        #     Rg = np.logspace(np.log10(Rref/6), np.log10(Rref*6), 200)
        #     plt.loglog(Rg, yref*(Rg/Rref)**(2), "--", lw=1.0, alpha=0.8, label=r"$\propto R^{2}$")
        plt.xlabel(r"$R$ (dx)"); plt.ylabel(r"$D_\varphi(R)$")
        plt.title("Polarization-angle structure function")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"Dphi_of_R_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Dphi_of_R_{i}.pdf")); plt.close()



        # --- RECONSTRUCT THE SPECTRUM FROM THE CORRELATION (S_map) ---
        # S_map = fftshift(ifft2(P2_dir).real) / Npix  ⇒  P2_dir ≈ fft2(ifftshift(S_map * Npix))
        Npix = f_map.size
        # P2_dir_rec = np.real(fft2(np.fft.ifftshift(S_map * Npix)))

        # minor numerical safety: enforce non-negativity
        # P2_dir_rec[P2_dir_rec < 0] = 0.0

        P2_dir_rec = np.real(fft2(np.fft.ifftshift(S_map * power0)))
        P2_dir_rec[P2_dir_rec < 0] = 0.0


        # Ring-average the reconstructed 2D spectrum
        kdir_rec, Pdir_ring_rec = ring_average_2d(P2_dir_rec, dx, C.nbins_k, C.kmin, C.kmax_frac)

        # --- PLOT: original vs reconstructed and their ratio ---
        fig = plt.figure(figsize=(6.6, 6.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

        # Top: spectra
        ax = fig.add_subplot(gs[0])
        ax.loglog(kdir, Pdir_ring, lw=1.8, label=fr"orig $P_{{\rm dir}}(k)$")
        ax.loglog(kdir_rec, Pdir_ring_rec, "--", lw=1.4, label="iFFT")
        ax.set_xticklabels([])  # hide x tick labels on top panel
        #ax.set_ylabel(r"$\langle |\widehat{\cos2f}|^2 + |\widehat{\sin2f}|^2\rangle$")
        ax.set_title("Directional spectrum: original vs reconstructed")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

        # Bottom: ratio (reconstructed / original) on shared x-range
        axr = fig.add_subplot(gs[1], sharex=ax)
        # interpolate reconstructed onto original k-grid (valid overlap only)
        valid = np.isfinite(kdir) & np.isfinite(Pdir_ring) & (kdir > 0)
        valid_rec = np.isfinite(kdir_rec) & np.isfinite(Pdir_ring_rec) & (kdir_rec > 0)
        if np.any(valid) and np.any(valid_rec):
            kmin_ratio = max(kdir[valid].min(), kdir_rec[valid_rec].min())
            kmax_ratio = min(kdir[valid].max(), kdir_rec[valid_rec].max())
            mask = valid & (kdir >= kmin_ratio) & (kdir <= kmax_ratio)
            if np.count_nonzero(mask) > 5:
                P_rec_interp = np.interp(kdir[mask], kdir_rec[valid_rec], Pdir_ring_rec[valid_rec])
                ratio = P_rec_interp / Pdir_ring[mask]
                axr.semilogx(kdir[mask], ratio, lw=1.3)
                axr.axhline(1.0, color="k", ls=":", lw=1.0)
                # report a compact scalar error in the console
                rel_err = np.nanmedian(np.abs(np.log10(ratio)))
                print(f"[recon λ={lam:.2f} m] median |log10(ratio)| = {rel_err:.3g}")
        axr.set_xlabel(r"$k$")
        axr.set_ylabel("recon/orig")
        axr.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(C.outdir, f"Pdir_recon_check_{i}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"Pdir_recon_check_{i}.pdf"))
        plt.close()


    print(f"Saved figures → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()