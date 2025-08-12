#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
White-noise simulations for LP16 validation
-------------------------------------------

This script:
  • Builds P from your 3D MHD cube for multiple LOS angles θ.
  • Applies optional Gaussian beam (FWHM in px).
  • Adds white Gaussian thermal noise to Q/U at target per-pixel SNR levels.
  • Uses *split halves* (P1, P2 with independent noise) and computes cross-correlators:
      C⋆12 = <P1 P2*> ,   C12 = <P1 P2>
    → unbiased by thermal noise at all lags, including zero-lag.
  • Compares ring azimuth profiles to LP16 (m=2, m=4, m=2+4) via *orthogonal projection*
    (no fits), both in the global frame and an auto-σ local frame (from smoothed Pavg).
  • Logs weighted correlations (r) and meta (coverage, σ, etc.) to JSONL.

Outputs:
  img/lp16_noisegrid_<tag>.jsonl

Tune the parameter grids near the top (θ, R, SNR, FWHM, N_mc).
"""

import json, math, time, pathlib
from collections import defaultdict

import numpy as np
import h5py
import scipy.ndimage as ndi
import matplotlib.pyplot as plt  # (not used for plotting here; left for quick debugging)

# ===================== USER SETTINGS ===================== #
# --- Data cube ---
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

# --- Geometry / LOS integration ---
N        = 256     # output image (N x N)
nsamp    = 192     # LOS samples
p        = 3.0     # synchrotron electron index (only affects amplitude; p=3 → |B_perp|^0)
theta_list = [90]  # LOS tilt relative to <B>
R_fracs    = [0.15, 0.22, 0.30, 0.40]     # ring radii as fraction of (N/2)
ring_width = 4.0                          # px half-thickness |R - R0| < ring_width
nbins_phi  = 256

# --- Faraday (off by default here) ---
include_faraday = False
ne0, lambda_m, rm_coeff = 1.0, 0.21, 0.812

# --- Local frame (α from smoothed Pavg) ---
sigma_factors = [0.15, 0.22, 0.30, 0.40, 0.55]  # σ = factor * R_pix
min_bins_used = 20

# --- Noise & beam grids ---
snr_list = [np.inf, 20, 10, 7, 5, 3, 2]       # per-pixel polarization SNR
fwhm_list = [0, 1, 2, 4, 8]                   # px
orders = ["beam_then_noise", "noise_then_beam"]
N_mc   = 30                                   # Monte Carlo trials per grid point (set 100–200 for final)

# --- Robustness & numerics ---
min_bin_frac = 0.001          # min fraction of ring pixels per φ-bin to accept the bin
eps_amp      = 1e-20          # tiny guard to avoid divide-by-zero

# --- Output ---
out_dir = pathlib.Path("img")
tag     = f"N{N}_ns{nsamp}_{'faraday' if include_faraday else 'nofaraday'}"
log_path = out_dir / f"lp16_noisegrid_{tag}_90.jsonl"
rng_seed_base = 42
# ========================================================= #


# -------------------- helpers (from your robust validator) -------------------- #
def load_cube(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2,1,0).astype(np.float32)
        By = f["j_mag_field"][:].transpose(2,1,0).astype(np.float32)
        Bz = f["k_mag_field"][:].transpose(2,1,0).astype(np.float32)
        x_edges = f["x_coor"][0,0,:]
        L = float(x_edges[-1] - x_edges[0])
    nx = Bx.shape[0]
    Brms  = float(np.sqrt(np.mean(Bx**2 + By**2 + Bz**2)))
    Bmean = np.array([Bx.mean(), By.mean(), Bz.mean()], dtype=np.float64)
    print(f"Mean field from cube: <B>={Bmean}  |<B>|={np.linalg.norm(Bmean):.3g}")
    return (Bx,By,Bz), L, nx, Brms, Bmean

def unit(v):
    n = np.linalg.norm(v);  return v/n if n>0 else v

def basis_from_Bmean(Bmean, theta_deg):
    bm = unit(Bmean.astype(np.float64))
    helper = np.array([0.0,0.0,1.0]) if abs(np.dot(bm,[0,0,1]))<0.98 else np.array([1.0,0.0,0.0])
    rot_axis = unit(np.cross(bm, helper))
    th = np.radians(theta_deg); ct, st = np.cos(th), np.sin(th)
    n_hat = ct*bm + st*np.cross(rot_axis,bm) + (1-ct)*np.dot(rot_axis,bm)*rot_axis
    n_hat = unit(n_hat.astype(np.float32))
    b_perp = Bmean - np.dot(Bmean,n_hat)*n_hat
    e1_hat = unit(b_perp).astype(np.float32) if np.linalg.norm(b_perp)>1e-12 else unit(np.cross(n_hat, rot_axis)).astype(np.float32)
    e2_hat = unit(np.cross(n_hat, e1_hat)).astype(np.float32)
    return n_hat, e1_hat, e2_hat

def make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat):
    i_idx, j_idx = np.indices((N,N), dtype=np.float32)
    x0 = ((i_idx-N/2)/N)[...,None]*L*e1_hat + ((j_idx-N/2)/N)[...,None]*L*e2_hat
    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32); ds = float(s[1]-s[0])
    los = (s[:,None]*n_hat[None,:])[None,None,:,:]
    pos_phys = x0[...,None,:] + los
    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = (pos_frac * (nx - 1)).reshape(-1,3).T.astype(np.float32)
    return pos_idx, ds

def interp(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N,N,nsamp)

def build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, e1_hat, e2_hat, p, ds,
            include_faraday=False, ne0=1.0, lambda_m=0.21, rm_coeff=0.812):
    Bx_l, By_l, Bz_l = (interp(Bx,pos_idx,N,nsamp),
                         interp(By,pos_idx,N,nsamp),
                         interp(Bz,pos_idx,N,nsamp))
    B = np.stack((Bx_l,By_l,Bz_l), axis=-1)               # (N,N,nsamp,3)
    B_par  = np.tensordot(B, n_hat, axes=([-1],[0]))      # (N,N,nsamp)
    Bp1 = B[...,0]*e1_hat[0] + B[...,1]*e1_hat[1] + B[...,2]*e1_hat[2]
    Bp2 = B[...,0]*e2_hat[0] + B[...,1]*e2_hat[1] + B[...,2]*e2_hat[2]
    Bperp = np.sqrt(Bp1**2 + Bp2**2) + eps_amp

    jP = (Bp1 + 1j*Bp2)**2 * (Bperp ** ((p - 3.0)/2.0))  # p=3 → amplitude 1

    if include_faraday:
        RM_tot   = rm_coeff * ne0 * np.cumsum(B_par, axis=-1) * ds
        RM_froms = RM_tot[..., -1][..., None] - RM_tot
        phase = np.exp(2j * (lambda_m**2) * RM_froms)
        P = np.sum(jP * phase, axis=-1) * ds
    else:
        P = np.sum(jP, axis=-1) * ds
    return P

def hann2d(N):
    w1 = np.hanning(N); W = np.outer(w1,w1)
    return (W / (W.mean() + eps_amp)).astype(np.float32)

def correlate_cross(P1, P2, apodize=True, demean=True):
    """Cross-correlators:
        C⋆12 = ifft2( F1 * conj(F2) )
        C12  = ifft2( fftshift(F1) * fftshift(F2)[::-1,::-1] )
      Both normalized by C⋆12(0).
    """
    A1 = P1 - np.mean(P1) if demean else P1
    A2 = P2 - np.mean(P2) if demean else P2
    W  = hann2d(P1.shape[0]) if apodize else 1.0
    A1 = A1 * W; A2 = A2 * W

    F1 = np.fft.fft2(A1)
    F2 = np.fft.fft2(A2)

    Cstar = np.fft.ifft2(F1 * np.conj(F2))
    Cstar = np.fft.fftshift(Cstar)

    F1c = np.fft.fftshift(F1); F2c = np.fft.fftshift(F2)
    Spec = F1c * F2c[::-1, ::-1]
    C = np.fft.ifft2(np.fft.ifftshift(Spec))
    C = np.fft.fftshift(C)

    C0 = Cstar[Cstar.shape[0]//2, Cstar.shape[1]//2].real
    if C0 != 0:
        Cstar = Cstar / C0
        C     = C / C0
    return Cstar, C

def ring_mask(N, R_pixels, ring_width):
    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x,x, indexing='ij')
    R = np.hypot(X,Y)
    return np.abs(R - R_pixels) < ring_width

def ring_azimuth(arr, R_pixels, ring_width, nbins, min_bin_frac):
    """Return (phi centers, counts, mean value per bin, valid mask, ring_pixels)."""
    N = arr.shape[0]
    mask = ring_mask(N, R_pixels, ring_width)
    ring_pixels = int(np.sum(mask))
    bins = np.linspace(-np.pi, np.pi, nbins+1)
    ctrs = 0.5*(bins[:-1]+bins[1:])
    if ring_pixels == 0:
        return ctrs, np.zeros(nbins, int), np.full(nbins, np.nan), np.zeros(nbins, bool), 0

    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x,x, indexing='ij')
    phi = np.arctan2(Y,X)

    counts, _ = np.histogram(phi[mask], bins=bins)
    vals,   _ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    with np.errstate(invalid='ignore'):
        y = np.divide(vals, np.maximum(counts,1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)

    mean_per_bin = ring_pixels / nbins
    min_count = max(1, int(max(min_bin_frac * ring_pixels, 0.1 * mean_per_bin)))
    valid = counts >= min_count
    if not np.any(valid) and np.any(counts > 0):
        valid = counts >= 1
    return ctrs, counts, y, valid, ring_pixels

def weighted_projection(phi, y, counts, modes, valid):
    w = counts.astype(np.float64)
    w = np.where(valid, w, 0.0)
    Wsum = np.sum(w)
    if Wsum <= 0 or y is None or np.all(~np.isfinite(y)):
        return np.nan, {}, np.full_like(y, np.nan, dtype=np.float64)
    y0 = (w@y)/Wsum
    yzm = y - y0
    y_th = np.full_like(y, y0, dtype=np.float64)
    coeffs = {}
    for m in modes:
        c = np.cos(m*phi); s = np.sin(m*phi)
        cc = np.sum(w*c*c)+eps_amp; ss = np.sum(w*s*s)+eps_amp
        ac = np.sum(w*yzm*c)/cc
        as_ = np.sum(w*yzm*s)/ss
        y_th += ac*c + as_*s
        coeffs[m] = (ac, as_)
    return y0, coeffs, y_th

def weighted_corr(y, yhat, w, valid):
    w_eff = np.where(valid, w.astype(np.float64), 0.0)
    Wsum  = np.sum(w_eff)
    if Wsum <= 0 or y is None or yhat is None:
        return np.nan
    ybar  = (w_eff@y)/Wsum; yhbar = (w_eff@yhat)/Wsum
    num   = np.sum(w_eff*(y-ybar)*(yhat-yhbar))
    den   = math.sqrt((np.sum(w_eff*(y-ybar)**2))*(np.sum(w_eff*(yhat-yhbar)**2)) + eps_amp)
    return float(num/den) if den>0 else np.nan

def local_align_from_Pavg(Pavg, sigma_pix, P1, P2):
    """Compute alpha from smoothed average map, rotate both halves by -2i*alpha."""
    if sigma_pix <= 0: 
        return P1, P2
    Ps = ndi.gaussian_filter(Pavg, sigma=sigma_pix, mode="wrap")
    alpha = 0.5 * np.angle(Ps)
    phase = np.exp(-2j*alpha)
    return P1*phase, P2*phase

def gaussian_smooth_complex(P, sigma_pix):
    if sigma_pix <= 0: 
        return P
    Pr = ndi.gaussian_filter(P.real, sigma=sigma_pix, mode="wrap")
    Pi = ndi.gaussian_filter(P.imag, sigma=sigma_pix, mode="wrap")
    return Pr + 1j*Pi

# -------------------- noise & beam utilities -------------------- #
def fwhm_to_sigma(fwhm_px):
    return float(fwhm_px) / (2.0 * math.sqrt(2.0 * math.log(2.0))) if fwhm_px>0 else 0.0

def add_white_noise_split(P_clean, snr_px, rng, sigma_post=None, order="beam_then_noise",
                          sigma_beam=0.0):
    """
    Create P1, P2 with independent white noise to achieve target per-pixel SNR after beam.
    If order=="beam_then_noise": sigma_post is used directly.
    If order=="noise_then_beam": we scale pre-beam noise so that after gaussian smoothing
      with sigma_beam the post-beam std equals sigma_post using the continuous factor:
          var_post = var_pre * (1 / (4π σ_b^2))  in 2D for normalized Gaussian.
    """
    if not np.isfinite(snr_px) or snr_px<=0:
        # No noise requested
        return P_clean, P_clean

    if sigma_post is None:
        raise ValueError("sigma_post must be provided (target Q/U std after beam).")

    if order == "beam_then_noise" or sigma_beam<=0:
        sigma_Q = sigma_U = float(sigma_post)
        eta1_Q = rng.normal(0.0, sigma_Q, size=P_clean.shape)
        eta1_U = rng.normal(0.0, sigma_U, size=P_clean.shape)
        eta2_Q = rng.normal(0.0, sigma_Q, size=P_clean.shape)
        eta2_U = rng.normal(0.0, sigma_U, size=P_clean.shape)
        P1 = P_clean + (eta1_Q + 1j*eta1_U)
        P2 = P_clean + (eta2_Q + 1j*eta2_U)
        return P1, P2
    else:
        # noise_then_beam: choose pre-beam noise so that post-beam std == sigma_post
        # var_post = var_pre * (1/(4π σ_b^2))  →  std_pre = std_post * 2*sqrt(pi)*σ_b
        std_pre = float(sigma_post) * 2.0 * math.sqrt(math.pi) * float(sigma_beam)
        eta1_Q = rng.normal(0.0, std_pre, size=P_clean.shape)
        eta1_U = rng.normal(0.0, std_pre, size=P_clean.shape)
        eta2_Q = rng.normal(0.0, std_pre, size=P_clean.shape)
        eta2_U = rng.normal(0.0, std_pre, size=P_clean.shape)
        P1 = gaussian_smooth_complex(P_clean + (eta1_Q + 1j*eta1_U), sigma_beam)
        P2 = gaussian_smooth_complex(P_clean + (eta2_Q + 1j*eta2_U), sigma_beam)
        return P1, P2

# ------------------------ MAIN ------------------------ #
def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    # truncate/create log
    open(log_path, "w", encoding="utf-8").close()

    (Bx,By,Bz), L, nx, Brms, Bmean = load_cube(filename)
    print(f"Loaded cube {nx}^3; L={L:g}; Brms={Brms:.6f}")

    for theta_deg in theta_list:
        n_hat, e1_hat, e2_hat = basis_from_Bmean(Bmean, theta_deg)
        pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
        # Build noiseless P (no beam yet)
        P0 = build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, e1_hat, e2_hat, p, ds,
                     include_faraday=include_faraday, ne0=ne0, lambda_m=lambda_m, rm_coeff=rm_coeff)

        for fwhm_px in fwhm_list:
            sigma_b = fwhm_to_sigma(fwhm_px)
            # Apply beam to the clean signal
            P_beam = gaussian_smooth_complex(P0, sigma_b)
            # rms(|P|) after beam
            rmsP = float(np.sqrt(np.mean(np.abs(P_beam)**2)))

            for order in orders:
                for snr_px in snr_list:
                    for imc in range(N_mc):
                        rng = np.random.default_rng(rng_seed_base + 100000*theta_deg + 1000*fwhm_px + 100*imc + (0 if order=="beam_then_noise" else 1))

                        # Target post-beam Q/U std to achieve SNR per pixel:
                        # SNR_px = rms(|P_beam|) / (sqrt(2) * sigma_post)
                        sigma_post = None if not np.isfinite(snr_px) else (rmsP / (math.sqrt(2.0) * float(snr_px)))

                        if order == "beam_then_noise":
                            # Add noise AFTER beam; both halves independent
                            P1, P2 = add_white_noise_split(P_beam, snr_px, rng, sigma_post=sigma_post,
                                                           order=order, sigma_beam=sigma_b)
                        else:
                            # Add noise BEFORE beam, then smooth (std scaled to hit target post-beam)
                            P1, P2 = add_white_noise_split(P0, snr_px, rng, sigma_post=sigma_post,
                                                           order=order, sigma_beam=sigma_b)

                        # -------- GLOBAL frame: cross-correlators --------
                        Cstar_g, C_g = correlate_cross(P1, P2, apodize=True, demean=True)

                        recs_to_write = []

                        for R_frac in R_fracs:
                            R_pix = R_frac * (N/2.0)

                            # Ring azimuth (GLOBAL)
                            phi, counts_g, y_re_star_g, valid_g, ring_pix = ring_azimuth(Cstar_g.real, R_pix, ring_width, nbins_phi, min_bin_frac)
                            _,   _,       y_re_pp_g,   _,       _       = ring_azimuth(C_g.real,      R_pix, ring_width, nbins_phi, min_bin_frac)
                            _,   _,       y_im_pp_g,   _,       _       = ring_azimuth(C_g.imag,      R_pix, ring_width, nbins_phi, min_bin_frac)

                            # LP16 projections (GLOBAL)
                            _, _, y_star_th_g = weighted_projection(phi, y_re_star_g, counts_g, modes=[2],   valid=valid_g)
                            _, _, y_pp4_g     = weighted_projection(phi, y_re_pp_g,   counts_g, modes=[4],   valid=valid_g)
                            _, _, y_pp24_g    = weighted_projection(phi, y_re_pp_g,   counts_g, modes=[2,4], valid=valid_g)
                            _, _, y_ip4_g     = weighted_projection(phi, y_im_pp_g,   counts_g, modes=[4],   valid=valid_g)
                            _, _, y_ip24_g    = weighted_projection(phi, y_im_pp_g,   counts_g, modes=[2,4], valid=valid_g)

                            r_star_g = weighted_corr(y_re_star_g, y_star_th_g, counts_g, valid_g)
                            r_pp4_g  = weighted_corr(y_re_pp_g,   y_pp4_g,    counts_g, valid_g)
                            r_pp24_g = weighted_corr(y_re_pp_g,   y_pp24_g,   counts_g, valid_g)
                            r_ip4_g  = weighted_corr(y_im_pp_g,   y_ip4_g,    counts_g, valid_g)
                            r_ip24_g = weighted_corr(y_im_pp_g,   y_ip24_g,   counts_g, valid_g)

                            # -------- LOCAL frame: α from smoothed Pavg --------
                            best = dict(score=-np.inf, sigma=0.0, eff_bins=0,
                                        r_pp4=np.nan, r_pp24=np.nan, r_ip24=np.nan)
                            Pavg = 0.5*(P1 + P2)

                            for k in sigma_factors:
                                sigma_loc = float(k * R_pix)
                                P1r, P2r  = local_align_from_Pavg(Pavg, sigma_loc, P1, P2)
                                Cstar_l, C_l = correlate_cross(P1r, P2r, apodize=True, demean=True)

                                _, _, y_re_star_l, valid_l, _ = ring_azimuth(Cstar_l.real, R_pix, ring_width, nbins_phi, min_bin_frac)
                                _, _, y_re_pp_l,   _,       _ = ring_azimuth(C_l.real,     R_pix, ring_width, nbins_phi, min_bin_frac)
                                _, _, y_im_pp_l,   _,       _ = ring_azimuth(C_l.imag,     R_pix, ring_width, nbins_phi, min_bin_frac)

                                _, _, y_pp4_l   = weighted_projection(phi, y_re_pp_l, counts_g, modes=[4],   valid=valid_l)
                                _, _, y_pp24_l  = weighted_projection(phi, y_re_pp_l, counts_g, modes=[2,4], valid=valid_l)
                                r_pp4_l  = weighted_corr(y_re_pp_l,  y_pp4_l,  counts_g, valid_l)
                                r_pp24_l = weighted_corr(y_re_pp_l,  y_pp24_l, counts_g, valid_l)
                                _, _, y_ip24_l  = weighted_projection(phi, y_im_pp_l, counts_g, modes=[2,4], valid=valid_l)
                                r_ip24_l = weighted_corr(y_im_pp_l,  y_ip24_l, counts_g, valid_l)

                                eff_bins_l = int(np.sum(valid_l))
                                if eff_bins_l < min_bins_used:
                                    continue
                                score = (r_pp24_l if np.isfinite(r_pp24_l) else -np.inf) + 0.3*(r_ip24_l if np.isfinite(r_ip24_l) else 0.0)
                                if score > best["score"]:
                                    best.update(dict(score=score, sigma=sigma_loc, eff_bins=eff_bins_l,
                                                     r_pp4=r_pp4_l, r_pp24=r_pp24_l, r_ip24=r_ip24_l))

                            # ----- record -----
                            rec = dict(
                                theta_deg=int(theta_deg),
                                R_frac=float(R_frac),
                                R_pix=float(R_pix),
                                fwhm_px=int(fwhm_px),
                                order=str(order),
                                snr_px=(None if not np.isfinite(snr_px) else float(snr_px)),
                                trial=int(imc),
                                ring_pixels=int(ring_pix),
                                global_eff_bins=int(np.sum(valid_g)),
                                global_r_RePPstar_m2=(None if not np.isfinite(r_star_g) else float(r_star_g)),
                                global_r_RePP_m4=(None if not np.isfinite(r_pp4_g) else float(r_pp4_g)),
                                global_r_RePP_m24=(None if not np.isfinite(r_pp24_g) else float(r_pp24_g)),
                                global_r_ImPP_m4=(None if not np.isfinite(r_ip4_g) else float(r_ip4_g)),
                                global_r_ImPP_m24=(None if not np.isfinite(r_ip24_g) else float(r_ip24_g)),
                                local_sigma=(None if best["sigma"]==0 else float(best["sigma"])),
                                local_eff_bins=int(best["eff_bins"]),
                                local_r_RePP_m4=(None if not np.isfinite(best["r_pp4"]) else float(best["r_pp4"])),
                                local_r_RePP_m24=(None if not np.isfinite(best["r_pp24"]) else float(best["r_pp24"])),
                                local_r_ImPP_m24=(None if not np.isfinite(best["r_ip24"]) else float(best["r_ip24"])),
                                timestamp=time.time(),
                            )
                            recs_to_write.append(rec)

                        # write records for all R in this (θ, fwhm, order, SNR, trial)
                        if recs_to_write:
                            with open(log_path, "a", encoding="utf-8") as fh:
                                for rr in recs_to_write:
                                    fh.write(json.dumps(rr) + "\n")

                print(f"θ={theta_deg:>2}°, FWHM={fwhm_px:>2}px, order={order}: done SNR grid.")

        print(f"θ={theta_deg:>2}° complete.")

    print(f"Saved logs → {log_path}")

if __name__ == "__main__":
    main()
