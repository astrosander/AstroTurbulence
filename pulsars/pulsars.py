#!/usr/bin/env python3
"""
Pulsar-sampling demo: how many point sources are needed to recover the turbulence slope?
========================================================================================

Based on original of Chepurnov, A. & Lazarian, A. (2009). "Turbulence spectra from Doppler-broadened spectral lines: tests of the Velocity Channel Analysis and Velocity Coordinate Spectrum techniques." The Astrophysical Journal, 693(2), 1074–1083. doi:10.1088/0004-637X/693/2/1074

What this does
--------------
• Builds (or loads) a 2-D Faraday screen Φ(x,y) with a known power-law spectrum.
• Chooses random “pulsar” positions and samples either:
    (A) RM_i = Φ(x_i,y_i) + noise,              → uses RM pairs to estimate D_Φ(R)
    (B) χ_i(λ) = χ0_i + λ^2 Φ(x_i,y_i) + noise, → uses angle pairs to estimate S(R)
       and also supports Δχ_i = χ_i(λ2) - χ_i(λ1) to cancel intrinsic χ0_i.
• Bins all pairs by separation R, forms the estimator, fits log–log slope over an inertial band.
• Repeats for many realizations and a grid of pulsar counts N⋆ to measure bias/variance.
• Reports the minimum N⋆ yielding target slope precision and saves diagnostic plots.

Key estimators
--------------
RM structure function (recommended for point sources with 2 nearby bands):
    D_Φ(R) = ⟨ [ RM_i - RM_j ]^2 ⟩ for pairs with |r_i - r_j| ≈ R.

Directional (single-band) correlation (works only if intrinsic angles per source are aligned or calibrated):
    S(R) = ⟨ cos 2(χ_i - χ_j) ⟩ ,  D_φ(R) = 1/2 [1 - S(R)].

Two-frequency “Δχ” per-source cancels intrinsic angles:
    Δχ_i = χ_i(λ2) - χ_i(λ1) = (λ2^2 - λ1^2) Φ_i  (mod π issues avoided for small Δλ^2/weak rotation).

No argparse; tweak the CONFIG block below and run.
"""

import os
import h5py
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict
from numpy.fft import fft2, ifft2, fftfreq

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Map / screen
    nx: int = 1024
    ny: int = 1024
    dx: float = 1.0             # pixel scale (arbitrary units)
    make_phi: bool = True       # if False, load Φ from HDF5/NPZ below
    phi_path: Optional[str] = None   # if not None, try to load Φ map (dataset 'Phi' or top-level array)
    # 2D target spectrum for Φ: E1D(k) ∝ k^{alpha_E}, so P2D(k) ∝ k^{alpha_E - 1}
    alpha_E: float = -5.0/3.0   # Kolmogorov-like 2D ring-energy slope (seen by observer)
    kmin_frac: float = 1/256     # large-scale cutoff (in units of Nyquist)
    kmax_frac: float = 0.5       # small-scale cutoff (≤0.5 = Nyquist)

    # Pulsar sampling
    N_list: Tuple[int, ...] = (20, 40, 80, 160, 320, 640, 1280)
    n_real: int = 16            # realizations per N⋆
    r_bins: int = 24
    r_min: float = 2.0          # in map units (same as dx)
    r_max_frac: float = 0.45

    # Angle/RM observation setup
    use_rm_pairs: bool = True   # RM-pair estimator (recommended for point sources)
    use_angle_pairs: bool = True
    # wavelengths for angle estimator (meters)
    lambda1_m: float = 0.21
    lambda2_m: float = 0.24     # if equal to lambda1_m, acts like single-band
    intrinsic_mode: str = "constant"  # 'constant' or 'random' intrinsic χ0 per source
    # noise
    sigma_RM: float = 0.0       # RM measurement noise (rad m^-2)
    sigma_chi: float = 0.0      # angle noise per band (radians)

    # Fit inertial range (in map units)
    fit_rmin: float = 4.0
    fit_rmax: float = 64.0

    # Decision threshold: how close the recovered slope must be to “truth”
    slope_tolerance: float = 0.10  # |alpha_est - alpha_true| ≤ this (1σ target)

    # Output
    outdir: str = "fig/pulsar_sampling"
    dpi: int = 160

C = Config()

# ──────────────────────────────────────────────────────────────────────
# Utilities: k/R grids, spectra, structure function on full grids
# ──────────────────────────────────────────────────────────────────────

def _K_grid(ny, nx, dx):
    ky = fftfreq(ny, d=dx)
    kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    return np.hypot(KY, KX)

def make_phi_powerlaw_2d(ny, nx, dx, alpha_E, kmin_frac, kmax_frac, seed=0):
    """
    Build Φ(x,y) whose *ring energy* E1D(k) ∝ k^{alpha_E}. Then P2D(k) ∝ k^{alpha_E - 1}.
    We shape white noise in Fourier domain by sqrt(P2D(k)), apply band-pass, and IFFT.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(ny, nx)) + 1j*rng.normal(size=(ny, nx))
    K = _K_grid(ny, nx, dx)
    k_nyq = 0.5/dx
    kmin = kmin_frac * k_nyq
    kmax = kmax_frac * k_nyq

    # 2D spectrum (ring): P2D ∝ k^{alpha_E-1}; band-pass window
    with np.errstate(divide='ignore'):
        P2 = np.where((K>=kmin) & (K<=kmax), np.power(np.maximum(K, 1e-12), alpha_E-1.0), 0.0)

    A = np.sqrt(P2)
    F = W * A
    phi = ifft2(F).real
    phi -= phi.mean()
    phi_std = phi.std(ddof=0)
    if phi_std > 0: phi /= phi_std
    return phi.astype(np.float32)

def ring_average_2d(P2D: np.ndarray, dx: float, nbins=240, kmin=1e-3, kmax_frac=1.0):
    ny, nx = P2D.shape
    ky = fftfreq(ny, d=dx); kx = fftfreq(nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KY, KX)
    kmax = float(kmax_frac) * K.max()
    bins = np.logspace(np.log10(max(kmin, 1e-8)), np.log10(kmax), nbins+1)
    idx = np.digitize(K.ravel(), bins) - 1
    p = P2D.ravel()
    good = (idx>=0) & (idx<nbins) & np.isfinite(p)
    sums  = np.bincount(idx[good], weights=p[good], minlength=nbins)
    cnts  = np.bincount(idx[good], minlength=nbins)
    prof  = np.full(nbins, np.nan, float)
    nz    = cnts>0
    prof[nz] = sums[nz]/cnts[nz]
    kcen = 0.5*(bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (kcen > kmin)
    return kcen[m], prof[m]

def structure_function_fullgrid(field: np.ndarray, dx: float, nbins=240, rmin=1.0, rmax_frac=0.45):
    """
    Isotropic 2nd-order structure function via FFT (Wiener–Khinchin), then radial average.
    D(R) = 2 Var - 2 AC(R).
    """
    f = field - field.mean()
    F = fft2(f)
    ac = ifft2(F*np.conj(F)).real / (field.size)
    ac = np.fft.fftshift(ac)
    var = f.var(ddof=0)
    D = 2*var - 2*ac
    ny, nx = field.shape
    y = (np.arange(ny)-ny//2)[:,None]
    x = (np.arange(nx)-nx//2)[None,:]
    R = np.hypot(y, x) * dx
    rmax = R.max()*float(rmax_frac)
    bins = np.logspace(np.log10(max(rmin, 1e-8)), np.log10(rmax), nbins+1)
    idx = np.digitize(R.ravel(), bins) - 1
    d   = D.ravel()
    good= (idx>=0) & (idx<nbins) & np.isfinite(d)
    sums  = np.bincount(idx[good], weights=d[good], minlength=nbins)
    cnts  = np.bincount(idx[good], minlength=nbins)
    prof  = np.full(nbins, np.nan, float)
    nz    = cnts>0
    prof[nz] = sums[nz]/cnts[nz]
    rcen = 0.5*(bins[1:] + bins[:-1])
    m = np.isfinite(prof) & (rcen > rmin)
    return rcen[m], prof[m]

def fit_loglog(x, y, xmin, xmax):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0) & (x>=xmin) & (x<=xmax)
    if np.count_nonzero(m) < 5: return np.nan, np.nan
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(np.exp(b))

# ──────────────────────────────────────────────────────────────────────
# Sampling: pulsar positions, interpolation, pair binning
# ──────────────────────────────────────────────────────────────────────

def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Sample img[y,x] at fractional coords (x,y) with bilinear interpolation; coords in [0, nx-1]/[0, ny-1]."""
    ny, nx = img.shape
    x0 = np.clip(np.floor(x).astype(int), 0, nx-1)
    y0 = np.clip(np.floor(y).astype(int), 0, ny-1)
    x1 = np.clip(x0+1, 0, nx-1)
    y1 = np.clip(y0+1, 0, ny-1)
    dx = np.clip(x - x0, 0.0, 1.0)
    dy = np.clip(y - y0, 0.0, 1.0)
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    I = (Ia*(1-dx)*(1-dy) + Ib*dx*(1-dy) + Ic*(1-dx)*dy + Id*dx*dy)
    return I

def random_pulsars(n: int, nx: int, ny: int, rng: np.random.Generator):
    x = rng.uniform(0, nx-1, size=n)
    y = rng.uniform(0, ny-1, size=n)
    return x, y

def pairwise_binned_stat(x: np.ndarray,
                         y: np.ndarray,
                         val_pairs: np.ndarray,
                         r_edges: np.ndarray,
                         dx: float = 1.0):
    """
    Bin a per-pair statistic over separation R with robust edges handling.

    Parameters
    ----------
    x, y : arrays of source positions in *pixel* units.
    val_pairs : array of length M = n(n-1)/2 with the statistic for all i<j pairs,
                computed using the SAME np.triu_indices ordering as here.
    r_edges : bin edges in *physical* units (same units you want on the x-axis).
    dx : pixel scale that converts (x,y) pixel distances to physical units.

    Returns
    -------
    r_cent_sel : bin centers (only bins that received pairs)
    prof_sel   : mean value in those bins
    counts_sel : number of pairs per returned bin
    """
    n = x.size
    if val_pairs.size != n*(n-1)//2:
        raise ValueError("val_pairs length must be n(n-1)/2 in i<j order")

    # All unordered pairs
    ii, jj = np.triu_indices(n, k=1)

    # Pair separations in PHYSICAL units (match r_edges units)
    rx = (x[ii] - x[jj]) * dx
    ry = (y[ii] - y[jj]) * dx
    r  = np.hypot(rx, ry)

    nb = len(r_edges) - 1
    if nb <= 0:
        raise ValueError("r_edges must have at least two values")

    # Digitize → indices in [0..nb-1]; discard out-of-range pairs
    inds = np.digitize(r, r_edges, right=False) - 1
    in_range = (inds >= 0) & (inds < nb) & np.isfinite(val_pairs)

    if not np.any(in_range):
        # No pairs landed in bins — return empty arrays with the right dtype
        return 0.5*(r_edges[1:] + r_edges[:-1]), np.full(nb, np.nan), np.zeros(nb, dtype=int)

    inds = inds[in_range]
    vals = val_pairs[in_range]

    # Bin safely with minlength = nb (length fixed to nb)
    counts = np.bincount(inds, minlength=nb)
    sums   = np.bincount(inds, weights=vals, minlength=nb)

    prof = np.full(nb, np.nan, float)
    nz   = counts > 0
    prof[nz] = sums[nz] / counts[nz]

    r_cent = 0.5*(r_edges[1:] + r_edges[:-1])

    # Return only bins that actually have pairs
    return r_cent[nz], prof[nz], counts[nz]

# ──────────────────────────────────────────────────────────────────────
# Estimators on sparse point sets
# ──────────────────────────────────────────────────────────────────────

def rm_pairs_estimator(phi_map, xs, ys, dx_map, sigma_RM=0.0):
    """Return pairwise values (RM_i - RM_j)^2 for all pairs (i<j)."""
    RM = bilinear_sample(phi_map, xs, ys)
    if sigma_RM>0:
        RM = RM + np.random.default_rng().normal(scale=sigma_RM, size=RM.size)
    n = RM.size
    i,j = np.triu_indices(n, k=1)
    d = (RM[i] - RM[j])**2
    return d

def angle_pairs_estimator(phi_map, xs, ys, lam1, lam2=None, sigma_chi=0.0, intrinsic_mode="constant"):
    """
    Return pairwise values cos(2(χ_i - χ_j)) for all pairs.
    Modes:
      - single-band: lam2 is None or == lam1; χ_i = χ0_i + lam1^2 Φ_i + noise
      - two-band Δχ: if lam2!=lam1, use Δχ_i = χ_i(lam2)-χ_i(lam1) = (lam2^2 - lam1^2) Φ_i (+ noise combo)
    Intrinsic chi0_i:
      - 'constant': same χ0 for all sources (cancels in the difference)
      - 'random': independent Uniform[0,π) per source (kills signal unless lam2!=lam1 used)
    """
    rng = np.random.default_rng()
    Phi = bilinear_sample(phi_map, xs, ys)
    n = Phi.size

    if (lam2 is not None) and (abs(lam2 - lam1) > 0):
        # Δχ per source
        scale = (lam2**2 - lam1**2)
        dchi = scale * Phi
        if sigma_chi>0:
            # noise from two bands adds in quadrature
            dchi = dchi + rng.normal(scale=math.sqrt(2)*sigma_chi, size=n)
        # treat Δχ like an “angle” (small-angle recommended)
        chi = dchi
        # intrinsic cancels by construction
    else:
        # single band
        if intrinsic_mode == "constant":
            chi0 = rng.uniform(0, np.pi)  # same for all
            chi = chi0 + (lam1**2) * Phi
        else:
            chi0 = rng.uniform(0, np.pi, size=n)  # random per source
            chi = chi0 + (lam1**2) * Phi
        if sigma_chi>0:
            chi = chi + rng.normal(scale=sigma_chi, size=n)

    # pairwise cos(2(χ_i - χ_j))
    i,j = np.triu_indices(n, k=1)
    c = np.cos(2.0*(chi[i] - chi[j]))
    return c

# ──────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────

def main(C=C):
    os.makedirs(C.outdir, exist_ok=True)

    # 1) Obtain Φ map (truth)
    if (not C.make_phi) and C.phi_path:
        # load Φ from file
        path = os.path.expanduser(C.phi_path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".npz", ".npy"):
            arr = np.load(path)
            Phi = arr["Phi"] if isinstance(arr, np.lib.npyio.NpzFile) else arr
        else:
            with h5py.File(path, "r") as f:
                Phi = f["Phi"][:] if "Phi" in f else f[list(f.keys())[0]][:]
        Phi = np.array(Phi, dtype=np.float64)
        ny, nx = Phi.shape
        dx = C.dx
    else:
        ny, nx, dx = C.ny, C.nx, C.dx
        Phi = make_phi_powerlaw_2d(ny, nx, dx, C.alpha_E, C.kmin_frac, C.kmax_frac, seed=2025)

    # 2) Truth: E1D(k) and D_Φ(R) slopes
    F = fft2(Phi)
    P2 = (F * np.conj(F)).real
    k1d, Pk = ring_average_2d(P2, dx, nbins=320, kmin=1e-3, kmax_frac=1.0)
    E1D = 2*np.pi * k1d * Pk
    r_truth, Dphi_truth = structure_function_fullgrid(Phi, dx, nbins=320, rmin=C.r_min, rmax_frac=C.r_max_frac)

    aE, _ = fit_loglog(k1d, E1D, xmin=max(k1d.min()*1.5, 1e-3), xmax=k1d.max()/2)
    aD, _ = fit_loglog(r_truth, Dphi_truth, xmin=C.fit_rmin, xmax=C.fit_rmax)

    # Save truth plots
    plt.figure(figsize=(6,4))
    plt.loglog(k1d, E1D, lw=1.6)
    plt.xlabel("k"); plt.ylabel("E1D(k)")
    plt.title(f"Truth E1D slope ≈ {aE:.2f}")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "truth_E1D.png"), dpi=C.dpi); plt.close()

    plt.figure(figsize=(6,4))
    plt.loglog(r_truth, Dphi_truth, lw=1.6)
    plt.xlabel("R"); plt.ylabel("D$_\\Phi$(R)")
    plt.title(f"Truth D_Φ slope ≈ {aD:.2f}")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "truth_Dphi.png"), dpi=C.dpi); plt.close()

    # 3) Prepare radial bins for pair estimators
    rmax = min(C.r_max_frac*max(nx,ny)*dx, r_truth.max())
    r_edges = np.logspace(np.log10(C.r_min), np.log10(rmax), C.r_bins+1)
    r_cent = 0.5*(r_edges[1:] + r_edges[:-1])

    rng = np.random.default_rng(1234)
    report = {
        "config": asdict(C),
        "truth": {"alpha_E": float(aE), "alpha_Dphi": float(aD)}
    }
    results_rows = []

    # 4) Loop over pulsar counts and realizations
    for Nstar in C.N_list:
        slopes_rm = []
        slopes_ang = []
        for rep in range(C.n_real):
            xs, ys = random_pulsars(Nstar, nx, ny, rng)

            # RM pairs
            if C.use_rm_pairs:
                val_pairs = rm_pairs_estimator(Phi, xs, ys, dx, sigma_RM=C.sigma_RM)
                r_rm, Dhat, cnts = pairwise_binned_stat(xs, ys, val_pairs, r_edges, dx)
                # r_rm, Dhat, cnts = pairwise_binned_stat(xs, ys, val_pairs, r_edges)
                a_hat, _ = fit_loglog(r_rm*dx, Dhat, xmin=C.fit_rmin, xmax=C.fit_rmax)
                slopes_rm.append(a_hat)

            # Angle pairs
            if C.use_angle_pairs:
                val_pairs = angle_pairs_estimator(Phi, xs, ys,
                                                  lam1=C.lambda1_m,
                                                  lam2=C.lambda2_m if abs(C.lambda2_m-C.lambda1_m)>0 else None,
                                                  sigma_chi=C.sigma_chi,
                                                  intrinsic_mode=C.intrinsic_mode)
                r_ang, Shat, cnts = pairwise_binned_stat(xs, ys, val_pairs, r_edges, dx)
                # Convert to D_phi(R) = 1/2[1 - S(R)] for slope fit (moderate-rotation regime)
                Dphi_hat = 0.5*(1.0 - Shat)
                a_hat, _ = fit_loglog(r_ang*dx, np.maximum(Dphi_hat, 1e-12), xmin=C.fit_rmin, xmax=C.fit_rmax)
                slopes_ang.append(a_hat)

        # summarize
        row = {"Nstar": int(Nstar)}
        if slopes_rm:
            s = np.array(slopes_rm, float)
            row.update({
                "rm_mean": float(np.nanmean(s)),
                "rm_std":  float(np.nanstd(s, ddof=1)),
                "rm_bias": float(np.nanmean(s) - aD),
            })
        if slopes_ang:
            s = np.array(slopes_ang, float)
            row.update({
                "ang_mean": float(np.nanmean(s)),
                "ang_std":  float(np.nanstd(s, ddof=1)),
                "ang_bias": float(np.nanmean(s) - aD),
            })
        results_rows.append(row)
        print(f"N⋆={Nstar:4d}  "
              + (f"RM: α={row.get('rm_mean',np.nan):+.3f}±{row.get('rm_std',np.nan):.3f}  " if slopes_rm else "")
              + (f"ANG: α={row.get('ang_mean',np.nan):+.3f}±{row.get('ang_std',np.nan):.3f}  " if slopes_ang else "")
              + f"(truth α_DΦ≈{aD:.3f})")

    # 5) Decide “minimum N⋆” that meets tolerance
    def find_min_N(key_mean, key_std):
        for row in results_rows:
            mu = row.get(key_mean, np.nan)
            sd = row.get(key_std,  np.nan)
            if np.isfinite(mu) and np.isfinite(sd):
                if abs(mu - aD) <= C.slope_tolerance and sd <= C.slope_tolerance:
                    return row["Nstar"]
        return None

    minN_rm  = find_min_N("rm_mean", "rm_std") if C.use_rm_pairs else None
    minN_ang = find_min_N("ang_mean", "ang_std") if C.use_angle_pairs else None

    # 6) Plots: slope vs N⋆
    Nvals = [r["Nstar"] for r in results_rows]
    if C.use_rm_pairs:
        mu = [r.get("rm_mean", np.nan) for r in results_rows]
        sd = [r.get("rm_std",  np.nan) for r in results_rows]
        plt.figure(figsize=(6.4,4.4))
        plt.semilogx(Nvals, mu, marker='o', lw=1.6, label="RM pairs")
        plt.fill_between(Nvals, np.array(mu)-np.array(sd), np.array(mu)+np.array(sd), alpha=0.2)
        plt.axhline(aD, color='k', ls='--', lw=1.0, label="truth")
        if minN_rm: plt.axvline(minN_rm, color='C0', ls=':', lw=1.0, label=f"min pulsars {minN_rm}")
        plt.xlabel("Number of pulsars")
        plt.ylabel("Fit of $D_\\Phi(R)$")
        plt.title("Slope recovery vs $N_\\star$ (RM pairs)")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "slope_vs_N_rm.png"), dpi=C.dpi); plt.close()

    if C.use_angle_pairs:
        mu = [r.get("ang_mean", np.nan) for r in results_rows]
        sd = [r.get("ang_std",  np.nan) for r in results_rows]
        plt.figure(figsize=(6.4,4.4))
        plt.semilogx(Nvals, mu, marker='o', lw=1.6, label="Angle pairs")
        plt.fill_between(Nvals, np.array(mu)-np.array(sd), np.array(mu)+np.array(sd), alpha=0.2)
        plt.axhline(aD, color='k', ls='--', lw=1.0, label="truth")
        if minN_ang: plt.axvline(minN_ang, color='C1', ls=':', lw=1.0, label=f"min pulsars {minN_ang}")
        plt.xlabel("Number of pulsars")
        plt.ylabel("Fit of $D_\\varphi(R)=\\frac{{1}}{{2}}[1-S(R)]$")
        mode = "Δχ two-band" if abs(C.lambda2_m - C.lambda1_m)>0 else f"single-band ({C.intrinsic_mode})"
        plt.title(f"Slope recovery vs $N_\\star$ (angles, {mode})")
        plt.grid(True, which='both', alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(C.outdir, "slope_vs_N_angle.png"), dpi=C.dpi); plt.close()

    # 7) Save JSON summary
    report["results"] = results_rows
    report["minN_rm"]  = minN_rm
    report["minN_ang"] = minN_ang
    with open(os.path.join(C.outdir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 8) Print concise recommendation
    print("\n=== Recommendation (data-driven) ===")
    if minN_rm:
        print(f"RM pairs: need ≈ {minN_rm} pulsars to achieve |bias|, 1σ ≤ {C.slope_tolerance:.2f}")
    else:
        print("RM pairs: increase N⋆ or realizations; target precision not reached within N_list.")
    if C.use_angle_pairs:
        if minN_ang:
            print(f"Angle pairs: need ≈ {minN_ang} pulsars under current angle setup.")
        else:
            print("Angle pairs: with current setup (single-band random χ0 or noise), slope not robust—use Δχ (two bands) or calibrate intrinsic angles.")

    print(f"Truth slopes:  E1D ≈ {aE:.3f},  D_Φ ≈ {aD:.3f}")
    print(f"Saved outputs → {os.path.abspath(C.outdir)}")

if __name__ == "__main__":
    main()
