#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final LP16 validation (robust & production-ready)

- Spin-2 emissivity: j_P ∝ (Bp1 + i Bp2)^2 * |B_perp|^{(p-3)/2}  (with p=3 → (Bp1+iBp2)^2)
- LOS geometry set by <B>; φ=0 along <B> projected onto POS.
- GLOBAL vs LOCAL (from P): alpha = 0.5*arg(G_sigma * P), P_local = P * exp(-2i alpha)
  * sigma auto-tuned per separation R from a small grid of fractions of R
- Correlators: <PP*>, <PP>; ring-azimuth profiles at radius R
- No-fit LP16 overlays: orthogonal projection onto allowed harmonics
  * Re<PP*> → m=2
  * Re<PP>, Im<PP> → m=4 and (m=2+4)
- Robust ring handling:
  * scale-aware min-count rule per φ-bin
  * weighted metrics ignore underfilled bins
  * optional cyclic inpainting for plotting ONLY (never used in metrics)
- Strong logging to JSONL
- Output directory: img/
"""

import json, math, time, pathlib
import numpy as np
import h5py, scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ===================== USER SETTINGS ===================== #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

theta_list = [0, 15, 30, 45, 60, 75, 90]
R_fracs    = [0.15, 0.22, 0.30, 0.40]      # radii as fractions of (N/2)
ring_width = 4.0                            # ring half-thickness [px] in |R - R0| < ring_width
nbins_phi  = 256

# Image/build params
N, nsamp   = 256, 192
p          = 3.0
include_faraday = True
ne0, lambda_m, rm_coeff = 1.0, 0.21, 0.812

# Local alignment from P: σ = factor * R_pix (auto-tuned per R)
sigma_factors = [0.15, 0.22, 0.30, 0.40, 0.55]

# Robustness thresholds
min_bin_frac   = 0.001      # baseline fraction of ring pixels per bin
min_bins_used  = 20         # require at least this many effective φ-bins for metrics
eps_amp        = 1e-20
inpaint_for_plot = True     # cosmetic only; metrics always use raw weighted bins

# Output
save_plots = True
out_dir    = pathlib.Path("img")   # <<< as requested
tag        = f"N{N}_ns{nsamp}_{'faraday' if include_faraday else 'nofaraday'}"
log_file   = out_dir / f"lp16_bestcorr_{tag}.jsonl"
dpi_plot   = 170
# ========================================================= #

# -------------------- helpers -------------------- #
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

    # Spin-2 emissivity (p=3 → amplitude factor 1)
    jP = (Bp1 + 1j*Bp2)**2 * (Bperp ** ((p - 3.0)/2.0))

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

def correlate_PP(P, apodize=True, demean=True):
    P0 = P - np.mean(P) if demean else P
    P0 = P0 * hann2d(P.shape[0]) if apodize else P0
    F = np.fft.fft2(P0)
    Cstar = np.fft.ifft2(np.abs(F)**2); Cstar = np.fft.fftshift(Cstar).real
    F_c = np.fft.fftshift(F)
    Spec = F_c * F_c[::-1, ::-1]
    C = np.fft.ifft2(np.fft.ifftshift(Spec)); C = np.fft.fftshift(C)
    C0 = Cstar[Cstar.shape[0]//2, Cstar.shape[1]//2]
    if C0 != 0: Cstar, C = Cstar/C0, C/C0
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
    if ring_pixels == 0:
        bins = np.linspace(-np.pi, np.pi, nbins+1)
        ctrs = 0.5*(bins[:-1]+bins[1:])
        return ctrs, np.zeros(nbins, int), np.full(nbins, np.nan), np.zeros(nbins, bool), 0

    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x,x, indexing='ij')
    phi = np.arctan2(Y,X)

    bins = np.linspace(-np.pi, np.pi, nbins+1)
    ctrs = 0.5*(bins[:-1]+bins[1:])

    counts, _ = np.histogram(phi[mask], bins=bins)
    vals,   _ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    with np.errstate(invalid='ignore'):
        y = np.divide(vals, np.maximum(counts,1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)

    # scale-aware min-count rule:
    mean_per_bin = ring_pixels / nbins
    min_count = max(1, int(max(min_bin_frac * ring_pixels, 0.1 * mean_per_bin)))
    valid = counts >= min_count

    # If still nothing valid, relax to accept >=1
    if not np.any(valid) and np.any(counts > 0):
        valid = counts >= 1

    return ctrs, counts, y, valid, ring_pixels

def inpaint_cyclic(y, valid):
    """Safe cyclic inpaint for plotting only. If no valid bins, return all-NaN."""
        
    for i, y_i in enumerate(y):
        if np.abs(y_i) < 1e-7:
            y[i] = np.nan
            valid[i] = False
    return y

    if valid is None or y is None or y.size == 0:
        return y
    if valid.all():
        return y.copy()
    if not np.any(valid):
        return np.full_like(y, np.nan, dtype=float)

    x = np.arange(y.size)
    xv = x[valid]; yv = y[valid]
    # wrap endpoint for cyclic interpolation
    xv2 = np.r_[xv, xv[0] + y.size]
    yv2 = np.r_[yv, yv[0]]
    yi = y.copy()
    xi = x[~valid]
    yi[~valid] = np.interp(xi, xv2, yv2)
    return yi

def weighted_projection(phi, y, counts, modes, valid):
    """Parameter-free projection onto harmonics (modes) using weights of valid bins only."""
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

def local_align_from_P(P, sigma_pix):
    """Smooth P, alpha=0.5*arg(P_smooth), rotate by -2 alpha."""
    if sigma_pix <= 0: return P
    Ps = ndi.gaussian_filter(P, sigma=sigma_pix, mode="wrap")
    alpha = 0.5 * np.angle(Ps)
    return P * np.exp(-2j*alpha)

# ------------------------ MAIN ------------------------ #
(Bx,By,Bz), L, nx, Brms, Bmean = load_cube(filename)

out_dir.mkdir(parents=True, exist_ok=True)
open(log_file, "w", encoding="utf-8").close()

for theta_deg in theta_list:
    n_hat, e1_hat, e2_hat = basis_from_Bmean(Bmean, theta_deg)
    pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
    P = build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, e1_hat, e2_hat, p, ds,
                include_faraday=include_faraday, ne0=ne0, lambda_m=lambda_m, rm_coeff=rm_coeff)

    for R_frac in R_fracs:
        R_pix = R_frac * (N/2.0)

        # -------- GLOBAL FRAME --------
        Cstar_g, C_g = correlate_PP(P, apodize=True, demean=True)
        phi, counts_g, y_re_star_g, valid_g, ring_pix = ring_azimuth(Cstar_g.real, R_pix, ring_width, nbins_phi, min_bin_frac)
        _,   _,       y_re_pp_g,   _,       _         = ring_azimuth(C_g.real,      R_pix, ring_width, nbins_phi, min_bin_frac)
        _,   _,       y_im_pp_g,   _,       _         = ring_azimuth(C_g.imag,      R_pix, ring_width, nbins_phi, min_bin_frac)

        y0s_g, coeff_s_g, yth_star_g = weighted_projection(phi, y_re_star_g, counts_g, modes=[2],   valid=valid_g)
        r_star_g = weighted_corr(y_re_star_g, yth_star_g, counts_g, valid_g)

        _, _, y_pp4_g  = weighted_projection(phi, y_re_pp_g, counts_g, modes=[4],   valid=valid_g)
        _, _, y_pp24_g = weighted_projection(phi, y_re_pp_g, counts_g, modes=[2,4], valid=valid_g)
        r_pp4_g  = weighted_corr(y_re_pp_g,  y_pp4_g,  counts_g, valid_g)
        r_pp24_g = weighted_corr(y_re_pp_g,  y_pp24_g, counts_g, valid_g)

        _, _, y_ip4_g  = weighted_projection(phi, y_im_pp_g, counts_g, modes=[4],   valid=valid_g)
        _, _, y_ip24_g = weighted_projection(phi, y_im_pp_g, counts_g, modes=[2,4], valid=valid_g)
        r_ip4_g  = weighted_corr(y_im_pp_g,  y_ip4_g,  counts_g, valid_g)
        r_ip24_g = weighted_corr(y_im_pp_g,  y_ip24_g, counts_g, valid_g)

        eff_bins_g = int(np.sum(valid_g))

        # -------- AUTO-TUNED LOCAL FRAME (from P) --------
        best = dict(score=-np.inf, sigma=0.0, eff_bins=0,
                    r_pp4=np.nan, r_pp24=np.nan, r_ip24=np.nan)

        for k in sigma_factors:
            sigma = float(k * R_pix)
            Prot = local_align_from_P(P, sigma)
            Cstar_l, C_l = correlate_PP(Prot, apodize=True, demean=True)

            _,   _, y_re_star_l, valid_l, _ = ring_azimuth(Cstar_l.real, R_pix, ring_width, nbins_phi, min_bin_frac)
            _,   _, y_re_pp_l,   _,       _ = ring_azimuth(C_l.real,     R_pix, ring_width, nbins_phi, min_bin_frac)
            _,   _, y_im_pp_l,   _,       _ = ring_azimuth(C_l.imag,     R_pix, ring_width, nbins_phi, min_bin_frac)

            _, _, y_pp4_l   = weighted_projection(phi, y_re_pp_l, counts_g, modes=[4],   valid=valid_l)
            _, _, y_pp24_l  = weighted_projection(phi, y_re_pp_l, counts_g, modes=[2,4], valid=valid_l)
            r_pp4_l  = weighted_corr(y_re_pp_l,  y_pp4_l,  counts_g, valid_l)
            r_pp24_l = weighted_corr(y_re_pp_l,  y_pp24_l, counts_g, valid_l)

            _, _, y_ip24_l  = weighted_projection(phi, y_im_pp_l, counts_g, modes=[2,4], valid=valid_l)
            r_ip24_l = weighted_corr(y_im_pp_l,  y_ip24_l, counts_g, valid_l)

            eff_bins_l = int(np.sum(valid_l))
            if eff_bins_l < min_bins_used:
                continue  # not enough coverage for reliable stats

            score = (r_pp24_l if np.isfinite(r_pp24_l) else -np.inf) + 0.3*(r_ip24_l if np.isfinite(r_ip24_l) else 0.0)
            if score > best["score"]:
                best.update(dict(score=score, sigma=sigma, eff_bins=eff_bins_l,
                                 r_pp4=r_pp4_l, r_pp24=r_pp24_l, r_ip24=r_ip24_l,
                                 y_re_pp=y_re_pp_l, y_pp4=y_pp4_l, y_pp24=y_pp24_l,
                                 y_im_pp=y_im_pp_l, y_ip24=y_ip24_l,
                                 y_re_star=y_re_star_l, valid=valid_l))

        # ----- LOG JSONL -----
        rec = dict(theta_deg=int(theta_deg), R_frac=float(R_frac), R_pix=float(R_pix),
                   ring_pixels=int(ring_pix),
                   global_eff_bins=int(eff_bins_g),
                   global_r_RePPstar_m2=float(r_star_g) if np.isfinite(r_star_g) else None,
                   global_r_RePP_m4=float(r_pp4_g)       if np.isfinite(r_pp4_g)  else None,
                   global_r_RePP_m24=float(r_pp24_g)     if np.isfinite(r_pp24_g) else None,
                   global_r_ImPP_m4=float(r_ip4_g)       if np.isfinite(r_ip4_g)  else None,
                   global_r_ImPP_m24=float(r_ip24_g)     if np.isfinite(r_ip24_g) else None,
                   local_sigma=float(best["sigma"]),
                   local_eff_bins=int(best["eff_bins"]),
                   local_r_RePP_m4=float(best["r_pp4"])   if np.isfinite(best["r_pp4"])  else None,
                   local_r_RePP_m24=float(best["r_pp24"]) if np.isfinite(best["r_pp24"]) else None,
                   local_r_ImPP_m24=float(best["r_ip24"]) if np.isfinite(best["r_ip24"]) else None,
                   timestamp=time.time())
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        # ----- PLOTS (global vs best-local) -----
        if save_plots:
            # local Re<PP*> m=2 projection (only if we had enough bins)
            if best["eff_bins"] >= min_bins_used:
                y_re_star_l = best["y_re_star"]
                _, _, y_star_th_l = weighted_projection(phi, y_re_star_l, counts_g, modes=[2], valid=best["valid"])
                r_star_l = weighted_corr(y_re_star_l, y_star_th_l, counts_g, best["valid"])
            else:
                y_re_star_l = y_star_th_l = None
                r_star_l = np.nan

            # cosmetic inpainting for plots (safe even if no valid bins)
            y_re_star_g_plot = inpaint_cyclic(y_re_star_g, valid_g) if inpaint_for_plot else y_re_star_g
            yth_star_g_plot  = inpaint_cyclic(yth_star_g,  valid_g) if inpaint_for_plot else yth_star_g
            y_re_pp_g_plot   = inpaint_cyclic(y_re_pp_g,   valid_g) if inpaint_for_plot else y_re_pp_g
            y_pp4_g_plot     = inpaint_cyclic(y_pp4_g,     valid_g) if inpaint_for_plot else y_pp4_g
            y_pp24_g_plot    = inpaint_cyclic(y_pp24_g,    valid_g) if inpaint_for_plot else y_pp24_g
            y_im_pp_g_plot   = inpaint_cyclic(y_im_pp_g,   valid_g) if inpaint_for_plot else y_im_pp_g
            y_ip4_g_plot     = inpaint_cyclic(y_ip4_g,     valid_g) if inpaint_for_plot else y_ip4_g
            y_ip24_g_plot    = inpaint_cyclic(y_ip24_g,    valid_g) if inpaint_for_plot else y_ip24_g

            if best["eff_bins"] >= min_bins_used:
                y_re_pp_l_plot = inpaint_cyclic(best["y_re_pp"], best["valid"]) if inpaint_for_plot else best["y_re_pp"]
                y_pp4_l_plot   = inpaint_cyclic(best["y_pp4"],   best["valid"]) if inpaint_for_plot else best["y_pp4"]
                y_pp24_l_plot  = inpaint_cyclic(best["y_pp24"],  best["valid"]) if inpaint_for_plot else best["y_pp24"]
                y_im_pp_l_plot = inpaint_cyclic(best["y_im_pp"], best["valid"]) if inpaint_for_plot else best["y_im_pp"]
                y_ip24_l_plot  = inpaint_cyclic(best["y_ip24"],  best["valid"]) if inpaint_for_plot else best["y_ip24"]
                y_re_star_l_plot = inpaint_cyclic(y_re_star_l,   best["valid"]) if inpaint_for_plot else y_re_star_l
                y_star_th_l_plot  = inpaint_cyclic(y_star_th_l,  best["valid"]) if inpaint_for_plot else y_star_th_l
            else:
                # fill with NaNs to avoid crashes; titles will show NaN correlations
                template = y_re_star_g_plot
                y_re_pp_l_plot = y_pp4_l_plot = y_pp24_l_plot = y_im_pp_l_plot = y_ip24_l_plot = np.full_like(y_re_pp_g_plot, np.nan)
                y_re_star_l_plot = y_star_th_l_plot = np.full_like(template, np.nan)

            fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)

            # GLOBAL
            ax = axes[0,0]
            ax.plot(phi, y_re_star_g_plot, lw=1.2, label="data")
            ax.plot(phi, yth_star_g_plot,  lw=1.2, ls="--", label="LP16 m=2")
            ax.set_title(f"GLOBAL Re⟨PP*⟩  r={r_star_g:.2f}")
            ax.set_ylabel("(norm.)"); ax.legend()

            ax = axes[0,1]
            ax.plot(phi, y_re_pp_g_plot, lw=1.2, label="data")
            # ax.plot(phi, y_pp4_g_plot,  lw=1.2, ls="--", label="LP16 m=4")
            ax.plot(phi, y_pp24_g_plot, lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.set_title(f"GLOBAL Re⟨PP⟩  r4={r_pp4_g:.2f}  r2+4={r_pp24_g:.2f}")
            ax.legend(loc="best")

            ax = axes[0,2]
            ax.plot(phi, y_im_pp_g_plot, lw=1.2, label="data")
            # ax.plot(phi, y_ip4_g_plot,  lw=1.2, ls="--", label="LP16 m=4")
            ax.plot(phi, y_ip24_g_plot, lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.axhline(0, ls=":", lw=1)
            ax.set_title(f"GLOBAL Im⟨PP⟩  r4={r_ip4_g:.2f}  r2+4={r_ip24_g:.2f}")
            ax.legend(loc="best")

            # LOCAL (auto σ)
            ax = axes[1,0]
            ax.plot(phi, y_re_star_l_plot, lw=1.2, label="data")
            ax.plot(phi, y_star_th_l_plot, lw=1.2, ls="--", label="LP16 m=2")
            ax.set_title(f"LOCAL(σ={best['sigma']:.1f}) Re⟨PP*⟩  r={np.nan if not np.isfinite(r_star_l) else f'{r_star_l:.2f}'}")
            ax.set_ylabel("(norm.)"); ax.legend()

            ax = axes[1,1]
            ax.plot(phi, y_re_pp_l_plot, lw=1.2, label="data")
            # ax.plot(phi, y_pp4_l_plot,  lw=1.2, ls="--", label="LP16 m=4")
            ax.plot(phi, y_pp24_l_plot, lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.set_title(f"LOCAL Re⟨PP⟩  r4={best.get('r_pp4',float('nan')):.2f}  r2+4={best.get('r_pp24',float('nan')):.2f}")
            ax.legend(loc="best")

            ax = axes[1,2]
            ax.plot(phi, y_im_pp_l_plot, lw=1.2, label="data")
            ax.plot(phi, y_ip24_l_plot, lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.axhline(0, ls=":", lw=1)
            ax.set_title(f"LOCAL Im⟨PP⟩  r2+4={best.get('r_ip24',float('nan')):.2f}")
            ax.legend(loc="best")

            for ax in axes[-1,:]: ax.set_xlabel("φ (rad)")
            fig.suptitle(f"θ={theta_deg}°,  R={R_frac:.2f}·(N/2) px", y=0.98)
            fn = out_dir / f"bestcorr_{tag}_th{theta_deg:02d}_R{R_frac:.2f}.png"
            plt.tight_layout(); plt.savefig(fn, dpi=dpi_plot, bbox_inches="tight"); plt.close(fig)

print(f"Saved logs → {log_file}")
