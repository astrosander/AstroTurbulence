#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Slope(D_phi vs R) as a function of m (robust).
- RM synthesis with P2D(k) ~ k^{-(m+2)} -> D_RM(R) ~ R^m
- External screen: D_phi(R,λ) = (2 λ^2)^2 D_RM(R)  => slope(D_phi) = m
- Unbiased autocorr via apodization + zero-padding + window normalization
- Fit slope on best power-law window (sliding log–log regression with max R^2)

Output: img/slope_vs_m_Dphi.png
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Controls ---------------- #
out_dir = pathlib.Path("img"); out_dir.mkdir(parents=True, exist_ok=True)

N              = 512                   # base map size
pad_factor     = 2                     # zero-padding factor (2 => 1024x1024 FFT)
seed           = 12345
m_vals         = np.linspace(0.3, 1.5, 121)   # dense m grid
n_realizations = 3                     # avg over a few seeds per m
kmin_pix       = 2.0                   # injection (outer-scale) cutoff in cycles/box
kmax_pix       = N/2.0                 # Nyquist
nbins_R        = 280
phase_factor   = 2.0                   # φ = 2 λ^2 RM; slope unaffected by λ
lam_fixed      = 0.2                   # any positive value (only amplitude)
min_counts     = 60                    # min pixels per annulus
apod_alpha     = 0.2                   # cosine-taper strength (0..1); 0.2 is mild
smooth_win     = 5                     # odd; 0/1 disables (light smoothing of radial D)
# Physically motivated inertial range in R (pixels)
# R_low ~ few pixels above Nyquist scale; R_high ~ fraction of outer scale
R_low_pix_fac  = 2.5                   # R_low ≈ R_low_pix_fac * (1/kmax)
R_high_pix_fac = 0.35                  # R_high ≈ R_high_pix_fac * (1/kmin)
# Sliding window for best-fit region in log-log
win_bins       = 28                    # bins per regression window (power-law patch)

dpi = 170

# ---------------- Utilities ---------------- #
rng_master = np.random.default_rng(seed)

def rng_for(m, r):
    # deterministic per (m, realization)
    return np.random.default_rng(abs(hash((float(m), int(r)))) % (2**32 - 1))

def hann1d(n):
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/max(n-1,1))

def tukey1d(n, alpha=0.2):
    if alpha <= 0: return np.ones(n)
    if alpha >= 1: return hann1d(n)
    w = np.ones(n)
    edge = int(alpha*(n-1)/2)
    if edge>0:
        t = np.linspace(0, np.pi, edge+1)
        w[:edge+1] = 0.5*(1-np.cos(t))
        w[-edge-1:] = w[:edge+1][::-1]
    return w

def make_rm_map(N, m, rng, kmin_pix=2.0, kmax_pix=None):
    if kmax_pix is None: kmax_pix = N/2.0
    fx = np.fft.fftfreq(N) * N
    fy = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(fx, fy, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)
    beta = m + 2.0
    k_clamped = np.clip(k, kmin_pix, kmax_pix)
    amp = k_clamped**(-beta/2.0)
    amp[0,0] = 0.0
    z = (rng.normal(size=(N,N)) + 1j*rng.normal(size=(N,N))) * amp
    # Hermitian symmetry
    N0, M0 = z.shape
    for i in range(N0):
        for j in range(M0):
            i2 = (-i) % N0; j2 = (-j) % M0
            if (i > N0//2) or (i == N0//2 and j > M0//2):
                z[i,j] = np.conj(z[i2,j2])
    rm = np.fft.ifft2(z).real
    rm -= rm.mean()
    s = rm.std()
    if s>0: rm /= s
    return rm

def autocorr2d_unbiased(field, apod_alpha=0.2, pad_factor=2):
    """
    Linear (unbiased) autocorr via:
      C = IFFT(|FFT(padded(w*field))|^2) / IFFT(|FFT(padded(w))|^2)
    with Tukey/Hann apodization w to reduce ringing; then normalize C(0)=1.
    """
    N = field.shape[0]
    M = pad_factor*N
    w1 = tukey1d(N, apod_alpha)
    w2 = np.outer(w1, w1)
    fwin = field * w2

    F = np.fft.fft2(fwin, s=(M, M))
    W = np.fft.fft2(w2,   s=(M, M))
    num = np.fft.ifft2(np.abs(F)**2).real
    den = np.fft.ifft2(np.abs(W)**2).real
    den = np.maximum(den, 1e-12)  # avoid divide-by-zero
    C = num / den
    C = np.fft.fftshift(C)
    # normalize to C(0)=1
    c0 = C[M//2, M//2]
    if c0 != 0: C /= c0
    return C

def radial_profile(img, nbins, Rmin, Rmax, min_counts=50):
    N = img.shape[0]
    y, x = np.indices((N, N))
    cx = cy = (N-1)/2.0
    R = np.hypot(x - cx, y - cy)

    bins = np.linspace(Rmin, Rmax, nbins+1)
    Rcent = 0.5*(bins[:-1] + bins[1:])
    which = np.digitize(R.ravel(), bins) - 1
    in_range = (which >= 0) & (which < nbins)
    idx = which[in_range]
    vals = img.ravel()[in_range]

    counts = np.bincount(idx, minlength=nbins)
    sums   = np.bincount(idx, weights=vals, minlength=nbins)
    with np.errstate(invalid="ignore"):
        prof = np.divide(sums, np.maximum(counts,1), out=np.zeros_like(sums, dtype=np.float64), where=counts>0)

    valid = counts >= min_counts
    return Rcent, prof, valid

def structure_function_from_corr(C):
    return 2.0*(1.0 - C)

def smooth_1d(y, win):
    if win is None or win <= 1: return y.copy()
    win = int(win) + (1 - int(win) % 2)
    pad = win//2
    ypad = np.pad(y, pad, mode='reflect')
    ker = np.ones(win, dtype=float) / win
    ys = np.convolve(ypad, ker, mode='same')[pad:-pad]
    return ys

def best_powerlaw_window(logR, logD, win_bins):
    """
    Slide a window of length win_bins across (logR, logD).
    Return slope with the highest R^2 (linear model).
    """
    n = logR.size
    if n < max(8, win_bins):
        # fall back to whole range
        A = np.vstack([logR, np.ones_like(logR)]).T
        slope, intercept = np.linalg.lstsq(A, logD, rcond=None)[0]
        # R^2
        yhat = A @ np.array([slope, intercept])
        SS_res = np.sum((logD - yhat)**2)
        SS_tot = np.sum((logD - logD.mean())**2) + 1e-12
        R2 = 1 - SS_res/SS_tot
        return float(slope), (0, n), float(R2)
    best = (-np.inf, None, None)  # R2, (i0,i1), slope
    for i0 in range(0, n - win_bins + 1):
        i1 = i0 + win_bins
        xr = logR[i0:i1]; yr = logD[i0:i1]
        A = np.vstack([xr, np.ones_like(xr)]).T
        slope, intercept = np.linalg.lstsq(A, yr, rcond=None)[0]
        yhat = A @ np.array([slope, intercept])
        SS_res = np.sum((yr - yhat)**2)
        SS_tot = np.sum((yr - yr.mean())**2) + 1e-12
        R2 = 1 - SS_res/SS_tot
        if R2 > best[0]:
            best = (R2, (i0, i1), slope)
    R2, (i0, i1), slope = best
    return float(slope), (i0, i1), float(R2)

def measure_slope_for_m(m, n_real=3):
    slopes = []
    for r in range(n_real):
        rng = rng_for(m, r)
        rm  = make_rm_map(N, m, rng, kmin_pix=kmin_pix, kmax_pix=kmax_pix)
        C   = autocorr2d_unbiased(rm, apod_alpha=apod_alpha, pad_factor=pad_factor)
        # Radial profile on the padded correlation
        M   = C.shape[0]
        Rmin = max(2.0, R_low_pix_fac * (1.0 / kmax_pix) * N)  # ~ few pixels
        Rmax = min(0.45*M, R_high_pix_fac * (1.0 / kmin_pix) * N * pad_factor)  # below outer scale image
        R, C_R, valid = radial_profile(C, nbins=nbins_R, Rmin=Rmin, Rmax=Rmax, min_counts=min_counts)
        Rv = R[valid]
        if Rv.size < 20:
            continue
        D = structure_function_from_corr(C_R)[valid]
        D = smooth_1d(D, smooth_win)
        # positive values only (for log)
        mask = (Rv>0) & (D>0)
        Rv, D = Rv[mask], D[mask]
        if Rv.size < max(20, win_bins):
            continue
        logR, logD = np.log(Rv), np.log(D)  # natural log ok (just scales intercept)
        slope, (i0, i1), R2 = best_powerlaw_window(logR, logD, win_bins=win_bins)
        slopes.append(slope)
    if len(slopes)==0:
        return np.nan
    return float(np.median(slopes))

# ---------------- Run survey ---------------- #
measured = []
for m in m_vals:
    s = measure_slope_for_m(m, n_real=n_realizations)
    measured.append(s)
    print(f"m={m:5.3f}  →  slope(D_phi vs R) = {s:6.3f}")

measured = np.array(measured, float)

# ---------------- Plot ---------------- #
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.plot(m_vals, measured, "o", ms=3.5, label="measured slope of $D_\\phi(R)$")
# running median for clarity
from numpy.lib.stride_tricks import sliding_window_view
def runmed(y, w=9):
    w = int(w) if int(w)%2==1 else int(w)+1
    if y.size < w: return y
    sw = sliding_window_view(y, w)
    med = np.median(sw, axis=1)
    padL = (w-1)//2
    padR = y.size - med.size - padL
    return np.r_[np.repeat(med[0], padL), med, np.repeat(med[-1], padR)]
ax.plot(m_vals, runmed(measured, w=11), "-", lw=1.6, label="running median")

# theory line y = m
mline = np.linspace(m_vals.min()*0.95, m_vals.max()*1.05, 400)
ax.plot(mline, mline, "--", lw=1.5, label="LP16 theory: slope = m")

ax.set_xlabel("m  (target slope in $D_{RM}\\propto R^m$)")
ax.set_ylabel("Measured slope of $D_\\phi(R)$")
ax.set_title("Slope of $D_\\phi(R)$ vs m (apodized, unbiased, best-window fit)")
ax.grid(True, alpha=0.25)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(out_dir / "slope_vs_m_Dphi.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {(out_dir / 'slope_vs_m_Dphi.png').resolve()}")
