#!/usr/bin/env python3
"""
Athena vs synthetic_two_slope: underlying & directional spectra + analytics
===========================================================================

- Loads two .h5 cubes:
    1) Athena:      mhd_fields.h5
    2) Synthetic:   synthetic_two_slope.h5  (from your generator)

- Computes RM maps Φ(X,Y)=0.81 ∑_z n_e B_z Δz for each, then:
    • Underlying spectrum: P_2D=|FFT(Φ)|^2 → E_1D(k)=2πk P_ring(k)
    • Directional (angle) spectrum at chosen λ (via N_rms target):
         f=λ^2 Φ,  A=cos(2f), B=sin(2f),  P_dir=|Â|^2+| B̂|^2 (ring-avg)
    • Axis cuts P(kx,0) & P(0,ky) to show potential anisotropy

- Analytics (as in our earlier discussion):
    Fit a smooth two-slope model to the **synthetic** E_1D(k):
       α(k)=α_low t + α_high (1−t),  t=[1+(k/k_b)^s]^{-1}
       E_1D,model(k)=A0 k^{α(k)}  ⇒  P_2D,model=E_1D,model/(2πk)
  Build P_2D,model on the FFT grid → IFFT to correlation → D_Φ(R)=2[C(0)−C(R)]
  Then D_φ(R,λ)=½[1−exp(−2 λ^4 D_Φ(R))].
  Compare to D_φ from the **synthetic map** computed via the directional route.

Outputs (in fig/athena_vs_synth/):
  - E1D_athena_vs_synth.pdf/png
  - axis_cuts_{athena|synth}.pdf/png
  - Pdir_athena_vs_synth.pdf/png
  - Dphi_synth_analytic_vs_map.pdf/png
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from numpy.fft import fft2, ifft2, fftshift, fftfreq

# ──────────────────────────────────────────────────────────────────────
# Config (edit paths if needed)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    ATH_PATH: str = "mhd_fields.h5"         # Athena cube (.h5)
    SYN_PATH: str = "synthetic_two_slope.h5" # Prebuilt two-slope synthetic (.h5)

    ne_key: str = "gas_density"
    bz_key: str = "k_mag_field"
    x_key: str = "x_coor"     # optional (for dx)
    z_key: str = "z_coor"     # optional (for dz)

    outdir: str = "fig/athena_vs_synth"
    dpi: int = 170

    # ring-averaging
    nbins: int = 300
    kmin: float = 1e-3
    kmax_frac: float = 1.0

    # choose λ via N_rms = λ^2 σ_Φ /(2π) to avoid saturation
    N_rms_target: float = 0.4

    # smooth two-slope fit initial guesses (used for the **synthetic** E1D)
    alpha_low_guess: float = +1.5
    alpha_high_guess: float = -5.0/3.0
    s_guess: float = 8.0
    kb_guess: float = 0.06  # cycles/dx, in the inertial range

    # quick-slope bands to print rough slopes
    fit_low: Tuple[float, float] = (0.01, 0.04)
    fit_high: Tuple[float, float] = (0.08, 0.22)

C = Config()

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing_from_grid(arr: np.ndarray) -> float:
    v = np.asarray(arr)
    if v.ndim == 3:
        v = np.unique(v[:,0,0])
    dif = np.diff(np.sort(np.unique(v)))
    dif = dif[dif > 0]
    return float(np.median(dif)) if dif.size else 1.0

def load_h5_cube(path: str, ne_key: str, bz_key: str,
                 x_key: str=None, z_key: str=None) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Returns (ne,bz, dx, dz, z_axis_index). Tries to detect which axis is z if z_coor present.
    """
    with h5py.File(path, "r") as f:
        ne = np.asarray(f[ne_key][:], dtype=np.float64)
        bz = np.asarray(f[bz_key][:], dtype=np.float64)
        if ne.shape != bz.shape or ne.ndim != 3:
            raise ValueError(f"{path}: ne/bz must be 3D and same shape")
        Ny, Nx, Nz = ne.shape

        dx = 1.0; dz = 1.0
        if x_key and x_key in f:
            dx = _axis_spacing_from_grid(f[x_key][:])
        if z_key and z_key in f:
            zc = np.asarray(f[z_key][:])
            if zc.ndim == 3:
                z1d = zc[0,0,:]
            else:
                z1d = zc
            dz = float(np.median(np.diff(np.unique(z1d))))
            z_len = len(np.unique(z1d))
            # detect which axis matches z_len
            sizes = ne.shape
            z_axis = int(np.argmin([abs(s - z_len) for s in sizes]))
        else:
            z_axis = 2
        return ne, bz, dx, dz, z_axis

def rm_from_cube(ne: np.ndarray, bz: np.ndarray, dz: float, z_axis: int) -> np.ndarray:
    RM = 0.81 * np.sum(ne * bz, axis=z_axis) * dz
    RM = RM - RM.mean()
    return RM

def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int, kmin: float, kmax_frac: float):
    Ny, Nx = P2D.shape
    ky = fftfreq(Ny, d=dx); kx = fftfreq(Nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    kmax = K.max() * float(kmax_frac)
    bins = np.logspace(np.log10(max(kmin,1e-8)), np.log10(kmax), nbins+1)
    k = K.ravel(); p = P2D.ravel()
    idx = np.digitize(k, bins) - 1
    nb = len(bins)-1
    mask = (idx>=0) & (idx<nb) & np.isfinite(p)
    sums = np.bincount(idx[mask], weights=p[mask], minlength=nb)
    cnts = np.bincount(idx[mask], minlength=nb)
    prof = np.full(nb, np.nan, float)
    nz = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    kcent = 0.5*(bins[1:] + bins[:-1])
    m2 = np.isfinite(prof) & (kcent > kmin)
    return kcent[m2], prof[m2]

def fit_loglog(x, y, xmin, xmax):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0) & (x>=xmin) & (x<=xmax)
    if np.count_nonzero(m) < 6: return np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(np.exp(b))

def alpha_blend(k, kb, s, aL, aH):
    t = 1.0 / (1.0 + (k/float(kb))**float(s))
    return aL * t + aH * (1.0 - t)

def E1D_model(k, A0, kb, s, aL, aH):
    return A0 * np.power(k, alpha_blend(k, kb, s, aL, aH))

def fit_two_slope_smooth(k, E, kb_guess, s_guess, aL_guess, aH_guess):
    k = np.asarray(k); E = np.asarray(E)
    m = np.isfinite(k) & np.isfinite(E) & (k>0) & (E>0)
    lk, lE = np.log(k[m]), np.log(E[m])
    k = k[m]; E = E[m]

    kb_grid = kb_guess * np.array([0.6, 0.8, 1.0, 1.25, 1.6])
    s_grid  = s_guess  * np.array([0.5, 1.0, 2.0])

    a_lo_hint, _ = fit_loglog(k, E, C.fit_low[0], C.fit_low[1])
    a_hi_hint, _ = fit_loglog(k, E, C.fit_high[0], C.fit_high[1])
    if not np.isfinite(a_lo_hint): a_lo_hint = aL_guess
    if not np.isfinite(a_hi_hint): a_hi_hint = aH_guess

    best = None; best_params = None
    for kb in kb_grid:
        for s in s_grid:
            for aL in (a_lo_hint-0.3, a_lo_hint, a_lo_hint+0.3):
                for aH in (a_hi_hint-0.3, a_hi_hint, a_hi_hint+0.3):
                    alpha = alpha_blend(k, kb, s, aL, aH)
                    A0_log = np.mean(lE - alpha*np.log(k))
                    pred = A0_log + alpha*np.log(k)
                    err = np.mean((pred - lE)**2)
                    if (best is None) or (err < best):
                        best = err
                        best_params = (np.exp(A0_log), kb, s, aL, aH)
    return best_params

def axis_cuts(P2D: np.ndarray, dx: float):
    P = fftshift(P2D)
    Ny, Nx = P.shape
    cx, cy = Nx//2, Ny//2
    kx_axis = fftshift(fftfreq(Nx, d=dx))
    ky_axis = fftshift(fftfreq(Ny, d=dx))
    kx_cut = np.maximum(P[cy, :], 1e-30)
    ky_cut = np.maximum(P[:, cx], 1e-30)
    return np.abs(kx_axis), kx_cut, np.abs(ky_axis), ky_cut

def build_P2D_from_E1D_on_grid(A0, kb, s, aL, aH, dx: float, shape: Tuple[int,int]):
    Ny, Nx = shape
    ky = fftfreq(Ny, d=dx); kx = fftfreq(Nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    Ksafe = np.maximum(K, 1e-12)
    E1D = E1D_model(Ksafe, A0, kb, s, aL, aH)
    P2D = E1D / (2.0*np.pi*Ksafe)
    P2D[0,0] = 0.0
    return P2D

def correlation_from_P2D(P2D):
    Ny, Nx = P2D.shape
    Cmap = fftshift(ifft2(P2D).real) / (Nx*Ny)
    return Cmap

def radial_profile(Map2D, dx, nbins=240, r_min=1e-3, r_max_frac=0.45):
    Ny, Nx = Map2D.shape
    y = (np.arange(Ny) - Ny//2)[:,None]
    x = (np.arange(Nx) - Nx//2)[None,:]
    R = np.hypot(y, x) * dx
    r_max = R.max() * float(r_max_frac)
    bins = np.logspace(np.log10(max(r_min,1e-8)), np.log10(r_max), nbins+1)
    idx = np.digitize(R.ravel(), bins) - 1
    nb = len(bins)-1
    m = (idx>=0) & (idx<nb)
    sums = np.bincount(idx[m], weights=Map2D.ravel()[m], minlength=nb)
    cnts = np.bincount(idx[m], minlength=nb)
    prof = np.full(nb, np.nan, float); nz = cnts>0
    prof[nz] = sums[nz]/cnts[nz]
    rcent = 0.5*(bins[1:] + bins[:-1])
    m2 = np.isfinite(prof) & (rcent > r_min)
    return rcent[m2], prof[m2]

def directional_spectrum_from_Phi(Phi, dx, N_rms_target):
    sigmaPhi = float(Phi.std(ddof=0))
    lam = np.sqrt((2.0*np.pi*N_rms_target) / max(sigmaPhi, 1e-30))
    f = lam**2 * Phi
    A = np.cos(2.0*f); B = np.sin(2.0*f)
    FA = fft2(A); FB = fft2(B)
    Pdir2D = (FA*np.conj(FA) + FB*np.conj(FB)).real
    k, Pring = ring_average_2d(Pdir2D, dx, C.nbins, C.kmin, C.kmax_frac)
    return lam, k, Pring

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def process_one(path, label):
    ne, bz, dx, dz, z_axis = load_h5_cube(path, C.ne_key, C.bz_key, C.x_key, C.z_key)
    RM = rm_from_cube(ne, bz, dz, z_axis)
    F = fft2(RM)
    P2D = (F*np.conj(F)).real
    k, Pk = ring_average_2d(P2D, dx, C.nbins, C.kmin, C.kmax_frac)
    E1D = 2.0*np.pi * k * Pk
    lam, k_dir, Pdir = directional_spectrum_from_Phi(RM, dx, C.N_rms_target)
    kx_axis, kx_cut, ky_axis, ky_cut = axis_cuts(P2D, dx)
    return dict(label=label, dx=dx, RM=RM, P2D=P2D, k=k, E1D=E1D,
                lam=lam, k_dir=k_dir, Pdir=Pdir,
                kx_axis=kx_axis, kx_cut=kx_cut, ky_axis=ky_axis, ky_cut=ky_cut)

def main():
    os.makedirs(C.outdir, exist_ok=True)

    # Athena
    A = process_one(C.ATH_PATH, "Athena")
    # Synthetic two-slope
    S = process_one(C.SYN_PATH, "Two-slope synthetic")

    # Rough slopes
    a_lo_A, _ = fit_loglog(A["k"], A["E1D"], *C.fit_low)
    a_hi_A, _ = fit_loglog(A["k"], A["E1D"], *C.fit_high)
    a_lo_S, _ = fit_loglog(S["k"], S["E1D"], *C.fit_low)
    a_hi_S, _ = fit_loglog(S["k"], S["E1D"], *C.fit_high)
    print(f"[Athena]  α_low≈{a_lo_A:+.3f}, α_high≈{a_hi_A:+.3f}")
    print(f"[Synthetic] α_low≈{a_lo_S:+.3f}, α_high≈{a_hi_S:+.3f}")

    # Fit two-slope smooth to the **synthetic** E1D for analytics
    A0,kb,s,aL,aH = fit_two_slope_smooth(S["k"], S["E1D"],
                                         C.kb_guess, C.s_guess,
                                         C.alpha_low_guess, C.alpha_high_guess)
    print(f"[fit on synthetic] A0={A0:.3e}, kb={kb:.4f}, s={s:.2f}, a_low={aL:+.3f}, a_high={aH:+.3f}")

    # Analytics from fitted model (on synthetic grid)
    P2D_model = build_P2D_from_E1D_on_grid(A0, kb, s, aL, aH, S["dx"], S["RM"].shape)
    Cmap = correlation_from_P2D(P2D_model)
    r_R, C_R = radial_profile(Cmap, S["dx"], nbins=C.nbins, r_min=1e-3, r_max_frac=0.45)
    Dphi_RM = 2.0*(C_R[0] - C_R)  # D_Φ(R)
    # λ matched to N_rms target using synthetic σ_Φ (from model C(0) = Var(Φ))
    sigmaPhi_model = np.sqrt(max(C_R[0], 1e-30))
    lam_model = np.sqrt((2.0*np.pi*C.N_rms_target) / sigmaPhi_model)
    Dphi_angle_model = 0.5*(1.0 - np.exp(-2.0*(lam_model**4) * Dphi_RM))

    # For comparison: D_φ from the synthetic **map** via cos/sin route
    _, rS, Dphi_from_map = None, None, None
    # Build S(R) from map: IFFT of P_dir is exactly S_map (since A^2+B^2=1 pointwise)
    f = (S["lam"]**2) * S["RM"]
    Amap = np.cos(2.0*f); Bmap = np.sin(2.0*f)
    FA, FB = fft2(Amap), fft2(Bmap)
    Pdir2D = (FA*np.conj(FA) + FB*np.conj(FB)).real
    Smap = fftshift(ifft2(Pdir2D).real) / (S["RM"].size)  # Wiener–Khinchin
    rS, S_R = radial_profile(Smap, S["dx"], nbins=C.nbins, r_min=1e-3, r_max_frac=0.45)
    Dphi_from_map = 0.5*(1.0 - S_R)

    # ── Plots ─────────────────────────────────────────────────────────

    # 1) E1D: Athena vs Synthetic (+ model on synthetic)
    E1D_model_curve = E1D_model(S["k"], A0, kb, s, aL, aH)
    plt.figure(figsize=(7.2,5.1))
    plt.loglog(A["k"], A["E1D"], '-', lw=1.8, label="Athena RM")
    plt.loglog(S["k"], S["E1D"], '-', lw=1.8, label="Two-slope synthetic RM")
    plt.loglog(S["k"], E1D_model_curve, '--', lw=1.6, label="Two-slope analytic model (fit on synthetic)")
    # slope guides
    for slope, band, lab in [(+1.5, C.fit_low, r"$+3/2$"), (-5/3, C.fit_high, r"$-5/3$")]:
        x0 = np.sqrt(band[0]*band[1]); y0 = np.interp(x0, S["k"], S["E1D"])
        xs = np.logspace(np.log10(x0/4), np.log10(x0*4), 80)
        plt.loglog(xs, y0*(xs/x0)**slope, ':', lw=1.0, alpha=0.8, label=f"ref {lab}")
    plt.axvline(kb, color='k', ls=':', lw=1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E_{1\rm D}(k)$")
    plt.title("Underlying spectrum: Athena vs two-slope synthetic (+ model)")
    plt.legend(frameon=False)
    plt.grid(True, which='both', alpha=0.3)
    os.makedirs(C.outdir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir,"E1D_athena_vs_synth.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir,"E1D_athena_vs_synth.pdf")); plt.close()

    # 2) Axis cuts for each dataset
    for tag, D in [("athena", A), ("synth", S)]:
        kx_axis, kx_cut, ky_axis, ky_cut = D["kx_axis"], D["kx_cut"], D["ky_axis"], D["ky_cut"]
        plt.figure(figsize=(7.2,5.1))
        plt.loglog(kx_axis, kx_cut, '-', lw=1.6, label=r"$P(k_x, k_y{=}0)$")
        plt.loglog(ky_axis, ky_cut, '-', lw=1.6, label=r"$P(k_y, k_x{=}0)$")
        plt.xlabel(r"$|k|$")
        plt.ylabel(r"$P_{2\rm D}$")
        plt.ylim(10**(-8),10**9)
        plt.title(f"Axis cuts: {D['label']} RM")
        plt.legend(frameon=False)
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, f"axis_cuts_{tag}.png"), dpi=C.dpi)
        plt.savefig(os.path.join(C.outdir, f"axis_cuts_{tag}.pdf")); plt.close()

    # 3) Directional spectrum: Athena vs Synthetic
    plt.figure(figsize=(7.2,5.1))
    plt.loglog(A["k_dir"], A["Pdir"], '-', lw=1.8, label=fr"Athena $P_{{\rm dir}}(k)$, $N_{{\rm rms}}\!\approx\!{C.N_rms_target}$")
    plt.loglog(S["k_dir"], S["Pdir"], '--', lw=1.8, label=fr"Synthetic $P_{{\rm dir}}(k)$, $N_{{\rm rms}}\!\approx\!{C.N_rms_target}$")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$P_{\rm dir}(k)$")
    plt.title("Directional spectrum (cos/sin-angle estimator)")
    plt.legend(frameon=False)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir,"Pdir_athena_vs_synth.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir,"Pdir_athena_vs_synth.pdf")); plt.close()

    # 4) D_phi(R): synthetic analytic vs synthetic map
    plt.figure(figsize=(7.2,5.1))
    plt.loglog(r_R, Dphi_angle_model, '-', lw=1.8, label=r"analytic two-slope $D_\varphi(R)$ (from fitted $P_{2\rm D}$)")
    plt.loglog(rS,  Dphi_from_map,  '--', lw=1.8, label=r"map-based $D_\varphi(R)$ (directional corr.)")
    # Kolmogorov guide
    if len(r_R)>10:
        r0 = r_R[len(r_R)//6]; y0 = np.interp(r0, r_R, Dphi_angle_model)
        rr = np.logspace(np.log10(r0/4), np.log10(r0*4), 80)
        plt.loglog(rr, y0*(rr/r0)**(5.0/3.0), ':', lw=1.0, alpha=0.9, label=r"ref $R^{5/3}$")
    plt.xlabel(r"$R$ (dx)")
    plt.xlim(2)
    plt.ylabel(r"$D_\varphi(R)$")
    plt.title(r"Polarization-angle structure: analytic vs map (synthetic)")
    plt.legend(frameon=False)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir,"Dphi_synth_analytic_vs_map.png"), dpi=C.dpi)
    plt.savefig(os.path.join(C.outdir,"Dphi_synth_analytic_vs_map.pdf")); plt.close()

    print(f"\nSaved → {os.path.abspath(C.outdir)}")
    print(f"Athena:   λ={A['lam']:.3e} m (N_rms≈{C.N_rms_target})")
    print(f"Synthetic:λ={S['lam']:.3e} m (map directional),  analytic used λ={lam_model:.3e} m")

if __name__ == "__main__":
    main()
