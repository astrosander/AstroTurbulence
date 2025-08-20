#!/usr/bin/env python3
"""
Observe two-slope spectra: direct 2D vs 3D→2D projection
========================================================

What this does
--------------
• Load a generated 3D cube (HDF5 with 'gas_density', 'k_mag_field', coords).
• Build a *direct 2D* random field whose 1D ring spectrum E1D(k) has two slopes:
     +3/2 below k_break and -5/3 above k_break (smooth transition).
• Project the 3D cube to the sky plane in two ways:
     (a) Bz-projection  : Sky(x,y) = Σ_z Bz(x,y,z)
     (b) RM proxy       : Sky(x,y) = Σ_z n_e(x,y,z) * Bz(x,y,z) * dz
• For each 2D map, compute the observed 2D Fourier power and the ring-averaged
  spectra P2D(k) and E1D(k)=2πk P2D(k), fit slopes in user-chosen bands, and save plots.

Conventions
-----------
- k in "cycles per dx" (NumPy fftfreq units). Slope values are unit-invariant.
- For a 2D isotropic field:  E1D(k) = 2π k P2D(k).
- We synthesize the *2D spectral density* as P2D(k) ∝ k^(α−1) to target shell slope α.

Edit the CONFIG block below. No argparse.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable
from numpy.fft import fft2, ifft2, fftfreq

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Path to the previously generated 3D cube (HDF5)
    cube_path: str = "../../synthetic_two_slope.h5"   # <-- set to your file
    # What to project from the 3D cube:
    #   "Bz"  → sum_z Bz
    #   "RM"  → sum_z (n_e * Bz * dz)   (requires x_coor,z_coor or assumes dz=dx)
    projection_mode: str = "RM"

    # Direct 2D synthetic field size
    nx: int = 512
    ny: int = 512
    dx_2d: float = 1.0
    seed_2d: int = 123

    # Target shell slopes for the two-slope spectrum
    alpha_low: float  = +1.5         # +3/2 (rising)
    alpha_high: float = -5.0/3.0     # -5/3 (falling)
    k_break_cyc: float = 0.06        # break, pick within inertial range
    sharp: float = 8.0               # transition sharpness (higher = sharper)

    # Fit windows for slope measurements (both 2D direct and projections)
    kfit_low: Tuple[float, float]  = (0.01, 0.04)
    kfit_high: Tuple[float, float] = (0.08, 0.20)

    # Output
    outdir: str = "fig_observer_2d"
    dpi: int = 160

CFG = Config()

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _axis_spacing(arr, default=1.0) -> float:
    """Median positive spacing from a 1D coordinate array; fallback to default."""
    if arr is None:
        return float(default)
    a = np.asarray(arr).ravel()
    dif = np.diff(np.sort(np.unique(a)))
    dif = dif[dif > 0]
    return float(np.median(dif)) if dif.size else float(default)

def ring_average_2d(P2D: np.ndarray, dx: float, nbins: int = 260,
                    kmin: float = 1e-4, kmax_frac: float = 1.0):
    """Isotropic (ring) average of a 2D power map P2D(kx,ky) onto k=|k| bins."""
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    kmax = K.max() * float(kmax_frac)

    bins = np.logspace(np.log10(max(kmin,1e-8)), np.log10(kmax), nbins+1)
    k = K.ravel(); p = P2D.ravel()
    idx = np.digitize(k, bins) - 1
    good = (idx >= 0) & (idx < nbins) & np.isfinite(p)
    sums = np.bincount(idx[good], weights=p[good], minlength=nbins)
    cnts = np.bincount(idx[good], minlength=nbins)
    prof = np.full(nbins, np.nan, float)
    nz   = cnts > 0
    prof[nz] = sums[nz] / cnts[nz]
    kcent = 0.5*(bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcent > kmin)
    return kcent[m], prof[m]

def fit_loglog(x, y, xmin, xmax):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0) & (x>=xmin) & (x<=xmax)
    if np.count_nonzero(m) < 5:
        return np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(np.exp(b))

def slope_guide(ax, x, y, slope, label, style="--", alpha=0.8):
    """Draw a power-law guide through the pivot (x,y) with exponent 'slope'."""
    xr = np.logspace(np.log10(x/2), np.log10(x*2), 50)
    ax.loglog(xr, y*(xr/x)**slope, style, lw=1.0, alpha=alpha, label=f"{label}: $k^{{{slope:+.2f}}}$")

# ──────────────────────────────────────────────────────────────────────
# Direct 2D synthesis with target two-slope shell slopes
# ──────────────────────────────────────────────────────────────────────

def P2D_two_slope_factory(alpha_low, alpha_high, k_break, sharp):
    """
    Return a function P2D(K) that yields ring slope α_low at K<<k_break
    and α_high at K>>k_break. Since E1D=2πk P2D, we set P2D ∝ k^(α−1).
    """
    def P2D(K, eps=1e-12):
        K = np.asarray(K)
        Ksafe = np.maximum(K, eps)
        x = Ksafe / float(k_break)
        t = 1.0 / (1.0 + x**sharp)      # ~1 at low-k, ~0 at high-k
        g_low  = alpha_low  - 1.0       # density exponent at low k
        g_high = alpha_high - 1.0       # density exponent at high k
        return t * x**g_low + (1.0 - t) * x**g_high
    return P2D

def synthesize_2d(nx, ny, dx, P2D_func: Callable, seed=0):
    """Create a real 2D field shaped by P2D_func(|k|)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(ny, nx))
    Wk = fft2(w)
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    A = np.sqrt(np.maximum(P2D_func(K), 0.0))
    Fk = Wk * A
    f = ifft2(Fk).real
    f -= f.mean(); f /= (f.std(ddof=0) + 1e-12)
    return f

# ──────────────────────────────────────────────────────────────────────
# Load 3D cube and make sky maps
# ──────────────────────────────────────────────────────────────────────

def load_cube(path):
    """Return dict with ne, bz, coords (or None), and spacings dx, dz."""
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:] if "gas_density" in f else None
        bz = f["k_mag_field"][:] if "k_mag_field" in f else None
        x = f["x_coor"][:] if "x_coor" in f else None
        z = f["z_coor"][:] if "z_coor" in f else None
    if ne is None or bz is None:
        raise RuntimeError("Missing 'gas_density' or 'k_mag_field' in the HDF5 cube.")
    if ne.shape != bz.shape or ne.ndim != 3:
        raise RuntimeError("ne/bz shapes mismatch or not 3D.")
    dx = _axis_spacing(x[:,0,0], 1.0) if x is not None else 1.0
    dz = _axis_spacing(z[0,0,:], 1.0) if z is not None else dx
    return dict(ne=ne, bz=bz, dx=dx, dz=dz)

def project_sky_maps(ne3, bz3, dz, mode="RM"):
    """
    mode="Bz": Sky = Σ_z Bz
    mode="RM": Sky = Σ_z (ne * Bz * dz)
    """
    if mode.upper() == "BZ":
        sky = bz3.sum(axis=2)
    elif mode.upper() == "RM":
        sky = (ne3 * bz3).sum(axis=2) * float(dz)
    else:
        raise ValueError("projection_mode must be 'Bz' or 'RM'.")
    sky = sky.astype(np.float64)
    sky -= sky.mean(); sky /= (sky.std(ddof=0) + 1e-12)
    return sky

# ──────────────────────────────────────────────────────────────────────
# Analysis & plotting
# ──────────────────────────────────────────────────────────────────────

def analyze_map(Map2D, dx, label, kfit_low, kfit_high, outdir, dpi=160):
    os.makedirs(outdir, exist_ok=True)

    # Save the map (observer's image)
    plt.figure(figsize=(5.2,4.6))
    v = np.max(np.abs(Map2D))*0.98
    plt.imshow(Map2D.T, origin="lower", cmap="magma", vmin=-v, vmax=v)  # transpose for (x,y) display
    plt.colorbar(label=label)
    plt.title(f"{label} map"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_map.png"), dpi=dpi); plt.close()

    # Fourier power and ring spectra
    Fk = fft2(Map2D)
    P2D = (Fk * np.conj(Fk)).real
    k1d, Pk = ring_average_2d(P2D, dx=dx, nbins=260, kmin=1e-4, kmax_frac=1.0)
    E1D = 2*np.pi * k1d * Pk

    # Fit slopes in user windows
    aP_lo, _ = fit_loglog(k1d, Pk, *kfit_low)
    aP_hi, _ = fit_loglog(k1d, Pk, *kfit_high)
    aE_lo, _ = fit_loglog(k1d, E1D, *kfit_low)
    aE_hi, _ = fit_loglog(k1d, E1D, *kfit_high)

    print(f"[{label}] slopes:")
    print(f"  P2D(k): low ≈ {aP_lo:+.3f}, high ≈ {aP_hi:+.3f}   (expect α−1)")
    print(f"  E1D(k): low ≈ {aE_lo:+.3f}, high ≈ {aE_hi:+.3f}   (expect α)")

    # Plot P2D(k)
    plt.figure(figsize=(6.2,4.8))
    m = (k1d>0) & np.isfinite(Pk) & (Pk>0)
    plt.loglog(k1d[m], Pk[m], lw=1.6, label="measured")
    # place guides at bin center near each fit-range geometric mean
    for rng, slope, tag in [(kfit_low, aP_lo, "low-fit"), (kfit_high, aP_hi, "high-fit")]:
        if np.isfinite(slope):
            kref = np.sqrt(rng[0]*rng[1])
            pref = np.interp(kref, k1d[m], Pk[m])
            slope_guide(plt.gca(), kref, pref, slope, f"{tag}")
    plt.xlabel(r"$k$"); plt.ylabel(r"$P_{2D}(k)$")
    plt.title(f"{label}: ring-averaged 2D power")
    plt.grid(True, which="both", alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{label}_P2D.png"), dpi=dpi); plt.close()

    # Plot E1D(k)
    plt.figure(figsize=(6.2,4.8))
    m = (k1d>0) & np.isfinite(E1D) & (E1D>0)
    plt.loglog(k1d[m], E1D[m], lw=1.6, label="measured")
    for rng, slope, tag in [(kfit_low, aE_lo, "low-fit"), (kfit_high, aE_hi, "high-fit")]:
        if np.isfinite(slope):
            kref = np.sqrt(rng[0]*rng[1])
            pref = np.interp(kref, k1d[m], E1D[m])
            slope_guide(plt.gca(), kref, pref, slope, f"{tag}")
    plt.xlabel(r"$k$"); plt.ylabel(r"$E_{1D}(k)=2\pi k\,P_{2D}(k)$")
    plt.title(f"{label}: 1D (ring) spectrum")
    plt.grid(True, which="both", alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{label}_E1D.png"), dpi=dpi); plt.close()

    return dict(k=k1d, P2D=Pk, E1D=E1D,
                slopes=dict(P_low=aP_lo, P_high=aP_hi, E_low=aE_lo, E_high=aE_hi))

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(C=CFG):
    os.makedirs(C.outdir, exist_ok=True)

    # ========== A) DIRECT 2D FIELD WITH TWO-SLOPE SHELL SPECTRUM ==========
    print("\n[A] Direct 2D field with target two-slope shell spectrum …")
    P2D_func = P2D_two_slope_factory(C.alpha_low, C.alpha_high, C.k_break_cyc, C.sharp)
    f2 = synthesize_2d(C.nx, C.ny, C.dx_2d, P2D_func, seed=C.seed_2d)
    res2 = analyze_map(f2, C.dx_2d, "2D_direct", C.kfit_low, C.kfit_high, C.outdir, dpi=C.dpi)

    print("Expected (construction):")
    print(f"  E1D low slope ≈ {C.alpha_low:+.3f}, high slope ≈ {C.alpha_high:+.3f}")
    print(f"  P2D low slope ≈ {C.alpha_low-1:+.3f}, high slope ≈ {C.alpha_high-1:+.3f}")

    # ========== B) IMPORT 3D CUBE AND PROJECT TO SKY ==========
    print("\n[B] Load 3D cube and project to 2D sky …")
    cube = load_cube(C.cube_path)
    ne3, bz3, dx3, dz3 = cube["ne"], cube["bz"], cube["dx"], cube["dz"]

    # (B1) Bz-projection
    sky_bz = project_sky_maps(ne3, bz3, dz3, mode="Bz")
    res_bz = analyze_map(sky_bz, dx3, "proj_Bz", C.kfit_low, C.kfit_high, C.outdir, dpi=C.dpi)

    # (B2) RM-proxy projection
    sky_rm = project_sky_maps(ne3, bz3, dz3, mode=C.projection_mode)  # "RM" by default
    res_rm = analyze_map(sky_rm, dx3, f"proj_{C.projection_mode}", C.kfit_low, C.kfit_high, C.outdir, dpi=C.dpi)

    print("\nProjection note: for pure power laws, 3D→2D tends to preserve the 1D shell slope α.\n"
          "Compare measured slopes of proj_Bz / proj_RM to the 2D_direct target values.")

    print("\nSaved figures in:", os.path.abspath(C.outdir))

if __name__ == "__main__":
    main()
