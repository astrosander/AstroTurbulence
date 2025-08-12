#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Slope of D_phi(R) vs m — robust & adaptive.

- RM ~ Gaussian, P2D(k) ∝ k^{-(m+2)} within [kmin,kmax] (band-pass).
- Unbiased autocorr via windowed FFT and division by window autocorr.
- D(R) = 2 [C(0) - C_R], using un-normalized C (correct zero-lag).
- Adaptive radial profile: widen rings / lower min_counts until enough bins.
- Slope by plateau derivative in log-log; sliding-fit fallback.
- Averaging across realizations per m.

Output: img/slope_vs_m_Dphi.png
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- User controls ---------------- #
out_dir = pathlib.Path("img"); out_dir.mkdir(parents=True, exist_ok=True)

N                = 512                 # base map size
pad_factor       = 2                   # zero-padding factor
seed             = 12345
m_vals           = np.linspace(0.30, 1.50, 121)
n_realizations   = 4                   # few realizations per m

# Spectral band (cycles per box in FFT units)
kmin_pix         = 2.0
kmax_pix         = N/2.0

# Apodization and radial stats (initial guesses; will adapt if needed)
apod_alpha       = 0.2                 # Tukey window parameter
nbins_R_init     = 160                 # initial radial bin count on padded grid
min_counts_init  = 20                  # initial min pixels per annulus

# Radial fit range tied to k-band (pixels on padded grid)
R_low_pix_fac    = 2.5                 # lower R ≈ few pixels above Nyquist
R_high_pix_fac   = 0.40                # upper R ≈ fraction of outer scale

# Smoothing & fitting
smooth_win_bins  = 5                   # odd; running-mean smoothing of D(R)
plateau_width_bins = 19                # for local derivative smoothing
max_curv           = 0.12              # |2nd deriv| threshold in log-log
deriv_band_frac    = 0.25              # keep where dlogD/dlogR within ±25% of median
win_bins_fallback  = 28                # sliding-fit window length (bins)

dpi = 170

# Choose one of: "frac_R", "frac_D", "Rmax"
fit_domain_mode = "frac_R"
fit_frac_R      = 0.5     # keep first 50% of the available R-range
fit_frac_D      = 0.5     # keep where D(R) <= 50% of max(D)
fit_Rmax_pix    = None    # absolute R cutoff (padded pixels); e.g., 80.0

ne_target_mean   = 1.0
ne_target_rms    = 1.000738263130188
bz_target_mean   = 0.11952292174100876
bz_target_rms    = 0.12989801168441772

# ---------------- Helpers ---------------- #
def rng_for(m, r):
    return np.random.default_rng(abs(hash((float(m), int(r), 911))) % (2**32 - 1))

def tukey1d(n, alpha=0.2):
    if alpha <= 0: return np.ones(n)
    if alpha >= 1:
        return 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/max(n-1,1))
    w = np.ones(n)
    edge = int(alpha*(n-1)/2)
    if edge>0:
        t = np.linspace(0, np.pi, edge+1)
        ramp = 0.5*(1-np.cos(t))
        w[:edge+1] = ramp
        w[-edge-1:] = ramp[::-1]
    return w

def make_rm_map(N, m, rng, kmin_pix=2.0, kmax_pix=None):
    if kmax_pix is None: kmax_pix = N/2.0
    fx = np.fft.fftfreq(N) * N
    fy = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(fx, fy, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)

    beta = m + 2.0
    band = (k >= kmin_pix) & (k <= kmax_pix)
    amp = np.zeros_like(k, dtype=float)
    with np.errstate(divide="ignore"):
        amp[band] = k[band]**(-beta/2.0)

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

def rescale_to_mean_rms(x, target_mean, target_rms):
    """
    Rescale a zero-mean-ish field x to have the requested mean and RMS.
    Works even if x isn't exactly zero-mean or unit-std.
    """
    x = x - np.mean(x)
    std0 = float(np.std(x))
    if not np.isfinite(std0) or std0 == 0.0:
        # Degenerate input; fall back to a constant field at the mean.
        return np.full_like(x, target_mean, dtype=float)
    xz = x / std0  # zero-mean, unit-variance
    # Compute desired std from mean & rms: sigma^2 = rms^2 - mean^2
    var_des = max(target_rms*target_rms - target_mean*target_mean, 0.0)
    sigma = np.sqrt(var_des)
    return sigma * xz + target_mean

def field_stats(name, arr):
    rms = float(np.sqrt(np.mean(arr*arr)))
    print(f"{name}: mean={np.mean(arr)}, min={np.min(arr)}, max={np.max(arr)}, rms={rms}")
 

def autocorr2d_unbiased(field, apod_alpha=0.2, pad_factor=2):
    """Unbiased autocorr: IFFT(|FFT(w*f)|^2) / IFFT(|FFT(w)|^2), zero-padded & shifted."""
    N = field.shape[0]
    M = pad_factor*N
    w1 = tukey1d(N, apod_alpha)
    w2 = np.outer(w1, w1)
    fwin = field * w2

    F = np.fft.fft2(fwin, s=(M, M))
    W = np.fft.fft2(w2,   s=(M, M))
    num = np.fft.ifft2(np.abs(F)**2).real
    den = np.fft.ifft2(np.abs(W)**2).real
    den = np.maximum(den, 1e-12)
    C = num / den
    C = np.fft.fftshift(C)
    return C

def radial_profile(img, nbins, Rmin, Rmax, min_counts=20):
    """Standard ring-mean radial profile."""
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
    return Rcent, prof, valid, counts

def radial_profile_adaptive(C, Rmin, Rmax,
                            nbins_init=160, min_counts_init=20,
                            needed_bins=30, max_tries=5):
    """Widen rings / relax counts until we have enough valid bins for fitting."""
    nbins = nbins_init
    min_counts = min_counts_init
    for t in range(max_tries):
        R, C_R, valid, counts = radial_profile(C, nbins, Rmin, Rmax, min_counts)
        if valid.sum() >= needed_bins:
            return R[valid], C_R[valid]
        # adapt: coarsen bins and relax counts
        nbins = max(40, int(nbins*0.7))
        min_counts = max(5, int(min_counts*0.7))
    # final attempt: accept whatever we have, even if small
    R, C_R, valid, counts = radial_profile(C, nbins, Rmin, Rmax, max(5, min_counts))
    return R[valid], C_R[valid]

def structure_function_from_corr_unscaled(C_center, C_R):
    return 2.0*(C_center - C_R)

def pick_fit_mask(R, D, mode="frac_R", frac_R=0.5, frac_D=0.5, Rmax_pix=None, min_pts=12):
    """Return a boolean mask selecting the early part of the curve for slope fits."""
    R = np.asarray(R); D = np.asarray(D)
    n = R.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(n, dtype=bool)
    if mode == "frac_R":
        frac = float(np.clip(frac_R, 0.05, 1.0))
        Rcut = R.min() + frac*(R.max() - R.min())
        mask = R <= Rcut
    elif mode == "frac_D":
        frac = float(np.clip(frac_D, 0.05, 1.0))
        Dcut = frac * float(np.nanmax(D))
        mask = D <= Dcut
    elif mode == "Rmax" and (Rmax_pix is not None):
        mask = R <= float(Rmax_pix)
    # ensure monotonic early segment has enough points; otherwise fall back
    if mask.sum() < min_pts:
        return np.ones(n, dtype=bool)
    return mask

def smooth_running_mean(y, win):
    if win is None or win <= 1: return y.copy()
    win = int(win) + (1 - int(win) % 2)  # force odd
    pad = win//2
    ypad = np.pad(y, pad, mode="reflect")
    ker = np.ones(win, dtype=float)/win
    return np.convolve(ypad, ker, mode="same")[pad:-pad]

def local_derivative_plateau(logR, logD, width_bins=19, max_curv=0.12, band_frac=0.25):
    n = logR.size
    if n < max(12, width_bins):
        return None
    # smooth logD slightly (boxcar in log-space)
    w = width_bins if width_bins % 2 == 1 else width_bins + 1
    pad = w//2
    ld_pad = np.pad(logD, pad, mode="edge")
    ker = np.ones(w)/w
    ld_s = np.convolve(ld_pad, ker, mode="same")[pad:-pad]

    d1 = np.gradient(ld_s, logR)
    d2 = np.gradient(d1,   logR)

    curv_mask = np.abs(d2) <= max_curv
    mid = slice(n//4, 3*n//4)
    d1_med = np.median(d1[mid])
    band = (d1 >= (1-band_frac)*d1_med) & (d1 <= (1+band_frac)*d1_med)
    mask = curv_mask & band
    if mask.sum() < max(6, w//2):
        return None
    return float(np.median(d1[mask]))

def best_window_regression(logR, logD, win_bins=28):
    n = logR.size
    if n < max(8, win_bins):
        A = np.vstack([logR, np.ones_like(logR)]).T
        slope, intercept = np.linalg.lstsq(A, logD, rcond=None)[0]
        return float(slope)
    best_R2 = -np.inf; best_slope = None
    for i0 in range(0, n - win_bins + 1):
        i1 = i0 + win_bins
        xr = logR[i0:i1]; yr = logD[i0:i1]
        A = np.vstack([xr, np.ones_like(xr)]).T
        slope, intercept = np.linalg.lstsq(A, yr, rcond=None)[0]
        yhat = A @ np.array([slope, intercept])
        SS_res = np.sum((yr - yhat)**2)
        SS_tot = np.sum((yr - yr.mean())**2) + 1e-12
        R2 = 1 - SS_res/SS_tot
        if R2 > best_R2:
            best_R2 = R2; best_slope = float(slope)
    return best_slope

def measure_slope_for_m(m, n_real=4):
    slopes = []
    for r in range(n_real):
        rng = rng_for(m, r)
        rm  = make_rm_map(N, m, rng, kmin_pix=kmin_pix, kmax_pix=kmax_pix)
        C   = autocorr2d_unbiased(rm, apod_alpha=apod_alpha, pad_factor=pad_factor)
        M   = C.shape[0]
        c0  = C[M//2, M//2]

        # R-range from spectral cuts, in padded pixels
        R_low  = max(2.0, R_low_pix_fac  * (N / kmax_pix) / pad_factor)
        R_high = min(0.48*M, R_high_pix_fac * (N / kmin_pix) / pad_factor)

        # Adaptive radial profile to ensure enough bins
        Rv, C_Rv = radial_profile_adaptive(
            C, R_low, R_high,
            nbins_init=nbins_R_init,
            min_counts_init=min_counts_init,
            needed_bins=30, max_tries=5
        )
        if Rv.size < 20:
            continue

        D = structure_function_from_corr_unscaled(c0, C_Rv)
        # enforce positivity with tiny floor (for log safety) without biasing slope
        eps = 1e-12 * (np.nanmax(D) if np.isfinite(np.nanmax(D)) else 1.0)
        D = np.maximum(D, eps)

        # slight smoothing
        D = smooth_running_mean(D, smooth_win_bins)

        # --- restrict to the initial part of the curve before fitting ---
        fit_mask = pick_fit_mask(
            Rv, D,
            mode=fit_domain_mode,
            frac_R=fit_frac_R,
            frac_D=fit_frac_D,
            Rmax_pix=fit_Rmax_pix,
            min_pts=12
        )
        R_fit = Rv[fit_mask]; D_fit = D[fit_mask]
        logR = np.log(R_fit);  logD = np.log(D_fit)

        slope = local_derivative_plateau(
            logR, logD,
            width_bins=plateau_width_bins,
            max_curv=max_curv,
            band_frac=deriv_band_frac
        )
        if slope is None:
            slope = best_window_regression(logR, logD, win_bins=win_bins_fallback)

        slopes.append(float(slope))

    if not slopes:
        return np.nan
    return float(np.median(slopes))

# ---------------- Survey ---------------- #
measured = []
for m in m_vals:
    s = measure_slope_for_m(m, n_real=n_realizations)
    measured.append(s)
    print(f"m={m:5.3f}  →  slope(D_phi vs R) = {s:6.3f}")

measured = np.array(measured, float)

# ---------------- Example: synthesize ne and Bz with prescribed stats ---------------- #
# This block is independent of (and after) the survey above. It just demonstrates how to
# generate two *independent* Gaussian fields with the same P(k) slope (choose m_demo),
# then rescale them to the requested mean/RMS and print stats.
m_demo = 5.0/3.0
rng_ne = np.random.default_rng(123456)   # separate seeds so ne and Bz are independent
rng_bz = np.random.default_rng(987654)
ne_raw = make_rm_map(N, m_demo, rng_ne, kmin_pix=kmin_pix, kmax_pix=kmax_pix)
bz_raw = make_rm_map(N, m_demo, rng_bz, kmin_pix=kmin_pix, kmax_pix=kmax_pix)
ne_map = rescale_to_mean_rms(ne_raw, ne_target_mean, ne_target_rms)
bz_map = rescale_to_mean_rms(bz_raw, bz_target_mean, bz_target_rms)
field_stats("ne", ne_map)
field_stats("bz", bz_map)

# ---------------- Plot ---------------- #
def running_median(y, w=9):
    """Odd-window running median with edge hold."""
    w = int(w)
    if w < 1: return y
    if w % 2 == 0: w += 1
    if y.size < w: return y
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(y, w)
    med = np.median(sw, axis=1)
    pad = (w-1)//2
    return np.r_[np.repeat(med[0], pad), med, np.repeat(med[-1], pad)]

fig, ax = plt.subplots(figsize=(6.8, 4.8))
ax.plot(m_vals, measured, "o", ms=3.2, label="measured slope of $D_\\phi(R)$")
ax.plot(m_vals, running_median(measured, w=13), "-", lw=1.6, label="running median")
m_line = np.linspace(m_vals.min()*0.95, m_vals.max()*1.05, 400)
ax.plot(m_line, m_line, "--", lw=1.5, label="LP16: slope = m")
ax.set_xlabel("m  (in $D_{RM}\\propto R^m$)")
ax.set_ylabel("Measured slope of $D_\\phi(R)$")
ax.set_title("Slope of $D_\\phi(R)$ vs m — early-R fit")
ax.grid(True, alpha=0.28)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(out_dir / "slope_vs_m_Dphi.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {(out_dir / 'slope_vs_m_Dphi.png').resolve()}")
