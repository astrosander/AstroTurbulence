#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LP16 validation with parameter-free overlays and strong logging.

- Builds P(X) with correct spin-2 emissivity:
    j_P ∝ (B_perp,1 + i B_perp,2)^2 * |B_perp|^{(p-3)/2}
- LOS geometry set relative to the cube's <B>; φ=0 along <B>_POS.
- Compares GLOBAL frame vs LOCAL field-aligned frame, where:
    alpha(x) = 0.5 * arg( G_sigma * P ),  P_local = P * exp(-2i alpha)
  (sigma auto-tuned per R from a small grid of k * R)
- Computes <PP*> and <PP>; extracts ring-averaged azimuthal profiles.
- Overlays LP16 "theory" without fitting by projecting onto the
  allowed harmonic subspaces (m=2 for Re<PP*>, m=4 / m=2+4 for <PP>).
- Logs correlations to JSONL and saves comparison plots.

Tested with Python 3.9 / NumPy 1.26 / SciPy 1.10 / Matplotlib 3.7
"""

import json, math, time, pathlib
import numpy as np
import h5py, scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ===================== USER SETTINGS ===================== #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

theta_list = [0, 15, 30, 45, 60, 75, 90]   # LOS angle wrt <B>
R_fracs    = [0.15, 0.22, 0.30, 0.40]      # radii as fractions of (N/2)
ring_width = 3.0                            # ring thickness [pixels]
nbins_phi  = 256

# Image/build params
N, nsamp   = 256, 192
p          = 3.0
include_faraday = False
ne0, lambda_m, rm_coeff = 1.0, 0.21, 0.812

# Local alignment from P: σ = factor * R_pix (auto-tuned per R)
sigma_factors = [0.15, 0.22, 0.30, 0.40, 0.55]

# Output
save_plots = True
out_dir    = pathlib.Path("img")
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
    Bperp = np.sqrt(Bp1**2 + Bp2**2) + 1e-30

    # Correct spin-2 emissivity
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
    return (W / W.mean()).astype(np.float32)

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

def ring_azimuth(arr, R_pixels, ring_width, nbins):
    N = arr.shape[0]
    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x,x, indexing='ij')
    R = np.hypot(X,Y); phi = np.arctan2(Y,X)
    mask = np.abs(R - R_pixels) < ring_width
    bins = np.linspace(-np.pi, np.pi, nbins+1)
    ctrs = 0.5*(bins[:-1]+bins[1:])
    counts, _ = np.histogram(phi[mask], bins=bins)
    vals,   _ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    y = np.divide(vals, np.maximum(counts,1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)
    return ctrs, counts, y

def weighted_projection(phi, y, counts, modes):
    """Parameter-free orthogonal projection onto harmonics in 'modes'."""
    w = counts.astype(np.float64)
    y0 = (w@y)/np.sum(w) if np.sum(w)>0 else np.mean(y)
    yzm = y - y0
    y_th = np.full_like(y, y0, dtype=np.float64)
    coeffs = {}
    for m in modes:
        c = np.cos(m*phi); s = np.sin(m*phi)
        cc = np.sum(w*c*c)+1e-30; ss = np.sum(w*s*s)+1e-30
        ac = np.sum(w*yzm*c)/cc
        as_ = np.sum(w*yzm*s)/ss
        y_th += ac*c + as_*s
        coeffs[m] = (ac, as_)
    return y0, coeffs, y_th

def weighted_corr(y, yhat, w):
    ybar  = (w@y)/np.sum(w); yhbar = (w@yhat)/np.sum(w)
    num   = np.sum(w*(y-ybar)*(yhat-yhbar))
    den   = math.sqrt((np.sum(w*(y-ybar)**2))*(np.sum(w*(yhat-yhbar)**2)) + 1e-30)
    return float(num/den) if den>0 else np.nan

def local_align_from_P(P, sigma_pix):
    """Smooth P, take alpha=0.5*arg(P_smooth), rotate by -2 alpha."""
    if sigma_pix <= 0: return P
    Ps = ndi.gaussian_filter(P, sigma=sigma_pix, mode="wrap")
    alpha = 0.5 * np.angle(Ps)
    return P * np.exp(-2j*alpha)

# ------------------------ MAIN ------------------------ #
(Bx,By,Bz), L, nx, Brms, Bmean = load_cube(filename)

# ensure output dir
out_dir.mkdir(parents=True, exist_ok=True)
# reset log file
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
        phi, counts, y_re_star_g = ring_azimuth(Cstar_g.real, R_pix, ring_width, nbins_phi)
        _,   _,     y_re_pp_g    = ring_azimuth(C_g.real,      R_pix, ring_width, nbins_phi)
        _,   _,     y_im_pp_g    = ring_azimuth(C_g.imag,      R_pix, ring_width, nbins_phi)

        _, coeff_s_g, yth_star_g = weighted_projection(phi, y_re_star_g, counts, modes=[2])
        r_star_g = weighted_corr(y_re_star_g, yth_star_g, counts)

        _, _, y_pp4_g   = weighted_projection(phi, y_re_pp_g, counts, modes=[4])
        _, _, y_pp24_g  = weighted_projection(phi, y_re_pp_g, counts, modes=[2,4])
        r_pp4_g  = weighted_corr(y_re_pp_g,  y_pp4_g,  counts)
        r_pp24_g = weighted_corr(y_re_pp_g,  y_pp24_g, counts)

        _, _, y_ip4_g   = weighted_projection(phi, y_im_pp_g, counts, modes=[4])
        _, _, y_ip24_g  = weighted_projection(phi, y_im_pp_g, counts, modes=[2,4])
        r_ip4_g  = weighted_corr(y_im_pp_g,  y_ip4_g,  counts)
        r_ip24_g = weighted_corr(y_im_pp_g,  y_ip24_g, counts)

        # -------- AUTO-TUNED LOCAL FRAME (from P) --------
        best = dict(score=-1, sigma=0.0)
        for k in sigma_factors:
            sigma = k * R_pix
            Prot = local_align_from_P(P, sigma)
            Cstar_l, C_l = correlate_PP(Prot, apodize=True, demean=True)

            _,   _, y_re_star = ring_azimuth(Cstar_l.real, R_pix, ring_width, nbins_phi)
            _,   _, y_re_pp   = ring_azimuth(C_l.real,     R_pix, ring_width, nbins_phi)
            _,   _, y_im_pp   = ring_azimuth(C_l.imag,     R_pix, ring_width, nbins_phi)

            _, _, y_pp4   = weighted_projection(phi, y_re_pp, counts, modes=[4])
            _, _, y_pp24  = weighted_projection(phi, y_re_pp, counts, modes=[2,4])
            r_pp4   = weighted_corr(y_re_pp,  y_pp4,  counts)
            r_pp24  = weighted_corr(y_re_pp,  y_pp24, counts)

            _, _, y_ip24  = weighted_projection(phi, y_im_pp, counts, modes=[2,4])
            r_ip24  = weighted_corr(y_im_pp,  y_ip24, counts)

            score = r_pp24 + 0.3*r_ip24   # objective for choosing σ
            if score > best["score"]:
                best.update(dict(score=score, sigma=sigma,
                                 r_pp4=r_pp4, r_pp24=r_pp24, r_ip24=r_ip24,
                                 y_re_pp=y_re_pp, y_pp4=y_pp4, y_pp24=y_pp24,
                                 y_im_pp=y_im_pp, y_ip24=y_ip24,
                                 y_re_star=y_re_star))

        # ----- LOG JSONL -----
        rec = dict(theta_deg=int(theta_deg), R_frac=float(R_frac), R_pix=float(R_pix),
                   global_r_RePPstar_m2=float(r_star_g),
                   global_r_RePP_m4=float(r_pp4_g), global_r_RePP_m24=float(r_pp24_g),
                   global_r_ImPP_m4=float(r_ip4_g), global_r_ImPP_m24=float(r_ip24_g),
                   local_sigma=float(best["sigma"]),
                   local_r_RePP_m4=float(best["r_pp4"]), local_r_RePP_m24=float(best["r_pp24"]),
                   local_r_ImPP_m24=float(best["r_ip24"]),
                   timestamp=time.time())
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        # ----- PLOTS (global vs best-local) -----
        if save_plots:
            # local frame again for star curve & overlays
            Prot = local_align_from_P(P, best["sigma"])
            Cstar_l, C_l = correlate_PP(Prot, apodize=True, demean=True)
            _, _, y_re_star_l = ring_azimuth(Cstar_l.real, R_pix, ring_width, nbins_phi)
            # local m=2 projection for Re<PP*>
            _, coeff_s_l, y_star_th_l = weighted_projection(phi, y_re_star_l, counts, modes=[2])
            r_star_l = weighted_corr(y_re_star_l, y_star_th_l, counts)

            # build plots
            fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)

            # GLOBAL
            ax = axes[0,0]
            ax.plot(phi, y_re_star_g, lw=1.2, label="data")
            ax.plot(phi, yth_star_g,  lw=1.2, ls="--", label="LP16 m=2")
            ax.set_title(f"GLOBAL Re⟨PP*⟩  r={r_star_g:.2f}")
            ax.set_ylabel("(norm.)"); ax.legend()

            ax = axes[0,1]
            ax.plot(phi, y_re_pp_g, lw=1.2, label="data")
            ax.plot(phi, y_pp4_g,  lw=1.2, ls="--", label="LP16 m=4")
            ax.plot(phi, y_pp24_g, lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.set_title(f"GLOBAL Re⟨PP⟩  r4={r_pp4_g:.2f}  r2+4={r_pp24_g:.2f}")
            ax.legend(loc="best")

            ax = axes[0,2]
            ax.plot(phi, y_im_pp_g, lw=1.2, label="data")
            ax.plot(phi, y_ip4_g,  lw=1.2, ls="--", label="LP16 m=4")
            ax.plot(phi, y_ip24_g, lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.axhline(0, ls=":", lw=1)
            ax.set_title(f"GLOBAL Im⟨PP⟩  r4={r_ip4_g:.2f}  r2+4={r_ip24_g:.2f}")
            ax.legend(loc="best")

            # LOCAL (auto σ)
            ax = axes[1,0]
            ax.plot(phi, y_re_star_l, lw=1.2, label="data")
            ax.plot(phi, y_star_th_l, lw=1.2, ls="--", label="LP16 m=2")
            ax.set_title(f"LOCAL(σ={best['sigma']:.1f}) Re⟨PP*⟩  r={r_star_l:.2f}")
            ax.set_ylabel("(norm.)"); ax.legend()

            ax = axes[1,1]
            ax.plot(phi, best["y_re_pp"], lw=1.2, label="data")
            ax.plot(phi, best["y_pp4"],  lw=1.2, ls="--", label="LP16 m=4")
            ax.plot(phi, best["y_pp24"], lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.set_title(f"LOCAL Re⟨PP⟩  r4={best['r_pp4']:.2f}  r2+4={best['r_pp24']:.2f}")
            ax.legend(loc="best")

            ax = axes[1,2]
            ax.plot(phi, best["y_im_pp"], lw=1.2, label="data")
            ax.plot(phi, best["y_ip24"], lw=1.2, ls=":",  label="LP16 m=2+4")
            ax.axhline(0, ls=":", lw=1)
            ax.set_title(f"LOCAL Im⟨PP⟩  r2+4={best['r_ip24']:.2f}")
            ax.legend(loc="best")

            for ax in axes[-1,:]: ax.set_xlabel("φ (rad)")
            fig.suptitle(f"θ={theta_deg}°,  R={R_frac:.2f}·(N/2) px", y=0.98)
            fn = out_dir / f"bestcorr_{tag}_th{theta_deg:02d}_R{R_frac:.2f}.png"
            plt.tight_layout(); plt.savefig(fn, dpi=dpi_plot, bbox_inches="tight"); plt.close(fig)

print(f"Saved logs → {log_file}")
