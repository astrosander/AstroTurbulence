#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ===================== USER PARAMS ===================== #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

theta_list = [30, 45, 60]        # LOS tilt angles (deg) relative to +z
N          = 256                  # pixels per side in the POS map
nsamp      = 192                  # samples along each ray
p          = 3.0                  # CR electron index (~3 → emissivity ∝ |B_perp|^2)
R0_frac    = 0.30                 # ring radius as a fraction of half-image
ring_width_pix = 1.0              # ring thickness (pixels)
nbins_phi  = 181                  # azimuth bins

# Mean field to boost anisotropy (optional)
add_uniform_B0   = True
B0_amp_factor    = 0.5            # × Brms, direction = +z

# Normalization choice
normalize_by_C0  = True           # divide correlation by C(0)

# Optional internal Faraday rotation (set to True to get non-zero Im signal)
include_faraday  = False
ne0              = 1.0            # cm^-3 (uniform electron density)
lambda_m         = 0.21           # observing wavelength in meters (e.g., 1.4 GHz → 0.21 m)
rm_coeff         = 0.812          # rad m^-2 per (cm^-3 μG pc); here we treat L in arbitrary units → relative test

savefig    = True
out_prefix = "anisotropy_compare_phasefit"
# ====================================================== #

def load_cube(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2, 1, 0).astype(np.float32)
        By = f["j_mag_field"][:].transpose(2, 1, 0).astype(np.float32)
        Bz = f["k_mag_field"][:].transpose(2, 1, 0).astype(np.float32)
        x_edges = f["x_coor"][0, 0, :]
        L = float(x_edges[-1] - x_edges[0])
    nx = Bx.shape[0]
    Brms = np.sqrt(np.mean(Bx**2 + By**2 + Bz**2)).item()
    print(f"Loaded cube {nx}^3; L={L:g}; Brms={Brms:g}")
    return (Bx, By, Bz), L, nx, Brms

def geometry(theta_deg):
    th = np.radians(theta_deg).astype(np.float32)
    n_hat  = np.array([np.sin(th), 0.0, np.cos(th)], dtype=np.float32)   # LOS
    e1_hat = np.array([np.cos(th), 0.0, -np.sin(th)], dtype=np.float32)  # POS x̂
    e2_hat = np.array([0.0, 1.0, 0.0], dtype=np.float32)                 # POS ŷ
    return n_hat, e1_hat, e2_hat

def make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat):
    i_idx, j_idx = np.indices((N, N), dtype=np.float32)
    x0 = ((i_idx - N/2) / N)[..., None] * L * e1_hat + \
         ((j_idx - N/2) / N)[..., None] * L * e2_hat     # (N,N,3)

    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32) # (nsamp,)
    ds = float(s[1] - s[0])

    los_offsets = (s[:, None] * n_hat[None, :])[None, None, :, :]        # (1,1,nsamp,3)
    pos_phys = x0[..., None, :] + los_offsets                            # (N,N,nsamp,3)

    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = pos_frac * (nx - 1)
    pos_idx  = pos_idx.reshape(-1, 3).T.astype(np.float32)               # (3, N*N*nsamp)
    return pos_idx, ds

def interp_field(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N, N, nsamp)

def synth_P(Bx, By, Bz, pos_idx, N, nsamp, n_hat, p, ds, B0_amp=None,
            include_faraday=False, ne0=1.0, lambda_m=0.21, rm_coeff=0.812):
    # Interpolate B along rays
    Bx_l = interp_field(Bx, pos_idx, N, nsamp)
    By_l = interp_field(By, pos_idx, N, nsamp)
    Bz_l = interp_field(Bz, pos_idx, N, nsamp)
    B = np.stack((Bx_l, By_l, Bz_l), axis=-1)           # (N,N,nsamp,3)

    # Optional uniform mean field (along +z)
    if B0_amp is not None and B0_amp > 0:
        B = B + np.array([0.0, 0.0, B0_amp], dtype=B.dtype)[None, None, None, :]

    # Decompose relative to LOS
    B_par  = np.tensordot(B, n_hat, axes=([-1], [0]))      # (N,N,nsamp)
    B_perp = B - B_par[..., None] * n_hat                   # (N,N,nsamp,3)

    # Emissivity and intrinsic polarisation angle
    Bperp_mag = np.linalg.norm(B_perp, axis=-1)
    eps = Bperp_mag ** ((p + 1) / 2.0)
    psi = 0.5 * np.arctan2(B_perp[..., 1], B_perp[..., 0])  # radians

    # Internal Faraday rotation (optional)
    if include_faraday:
        # crude cumulative RM along s: Φ_k = 0.812 ∑ ne * B_par * Δs
        # Units here are relative; choose ne0 and lambda_m to tune signal
        Phi = rm_coeff * ne0 * np.cumsum(B_par, axis=-1) * ds   # (N,N,nsamp)
        phase = 2.0 * (psi + (lambda_m**2) * Phi)
    else:
        phase = 2.0 * psi

    P = np.sum(eps * np.exp(1j * phase), axis=-1) * ds        # (N,N)
    return P

def correlation(P, normalize_by_C0=True):
    F  = np.fft.fft2(P)
    C  = np.fft.ifft2(F * F.conj())
    C  = np.fft.fftshift(C)
    if normalize_by_C0:
        C0 = C[C.shape[0]//2, C.shape[1]//2].real
        if C0 != 0:
            C = C / C0
    return C.real.astype(np.float64), C.imag.astype(np.float64)

def ring_azimuth(C_re, C_im, R0_frac, ring_width_pix, nbins):
    N = C_re.shape[0]
    x = (np.arange(N) - N/2)
    y = (np.arange(N) - N/2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R   = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)                                   # [-π, π]

    R0 = R0_frac * (N/2.0)
    mask = np.abs(R - R0) < ring_width_pix

    bins = np.linspace(-np.pi, np.pi, nbins, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(phi[mask], bins=bins)

    def az_avg(arr):
        vals, _ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
        return np.divide(vals, np.maximum(counts, 1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)

    re_phi = az_avg(C_re)
    im_phi = az_avg(C_im)
    return centers, counts, re_phi, im_phi

def fit_phaseaware(phi, y, counts, basis="cos+sin"):
    """
    Weighted LS with both cos and sin bases:
      y ≈ a0 + a_c cos(2φ) + a_s sin(2φ)
    Returns: model, (a0, a_c, a_s), R^2, amplitude A, phase φ0
      where A = sqrt(a_c^2 + a_s^2), φ0 = 0.5*atan2(a_s, a_c)
    """
    w = counts.astype(np.float64)
    Xcols = [np.ones_like(phi, dtype=np.float64)]
    if "cos" in basis:
        Xcols.append(np.cos(2.0 * phi))
    if "sin" in basis:
        Xcols.append(np.sin(2.0 * phi))
    X = np.stack(Xcols, axis=1)  # shape (n, 1+Nbasis)

    WX   = w[:, None] * X
    XtWX = X.T @ WX
    XtWy = X.T @ (w * y)
    coeffs = np.linalg.pinv(XtWX, rcond=1e-12) @ XtWy
    y_model = X @ coeffs

    y_mean_w = (w @ y) / np.sum(w) if np.sum(w) > 0 else np.mean(y)
    ss_res = np.sum(w * (y - y_model)**2)
    ss_tot = np.sum(w * (y - y_mean_w)**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if X.shape[1] == 3:
        a0, a_c, a_s = coeffs
        A = np.hypot(a_c, a_s)
        phi0 = 0.5 * np.arctan2(a_s, a_c)
    else:
        a0, A, phi0 = coeffs[0], 0.0, 0.0
    return y_model, coeffs, r2, A, phi0

# ===================== MAIN ===================== #
(Bx, By, Bz), L, nx, Brms = load_cube(filename)
B0_amp = B0_amp_factor * Brms if add_uniform_B0 else None
print(f"Uniform B0 amplitude: {B0_amp if B0_amp else 0:.5g} (same units as cube)")
print(f"Internal Faraday: {include_faraday} (λ={lambda_m} m, ne0={ne0})\n")

nrows = len(theta_list)
fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.3 * nrows))
if nrows == 1:
    axes = np.array([axes])

for r, theta_deg in enumerate(theta_list):
    print(f"=== Processing θ = {theta_deg}° ===")
    n_hat, e1_hat, e2_hat = geometry(theta_deg)
    pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
    P = synth_P(Bx, By, Bz, pos_idx, N, nsamp, n_hat, p, ds, B0_amp=B0_amp,
                include_faraday=include_faraday, ne0=ne0, lambda_m=lambda_m, rm_coeff=rm_coeff)

    C_re, C_im = correlation(P, normalize_by_C0=normalize_by_C0)
    phi_cent, counts, re_phi, im_phi = ring_azimuth(C_re, C_im, R0_frac, ring_width_pix, nbins_phi)

    # Phase-aware fits (cos+sin)
    re_model, (a0, ac, as_), r2_re, A_re, phi0_re = fit_phaseaware(phi_cent, re_phi, counts, basis="cos+sin")
    im_model, (b0, bc, bs),  r2_im, A_im, phi0_im = fit_phaseaware(phi_cent, im_phi, counts, basis="cos+sin")

    # Print compact diagnostics
    print(f"Re: a0={a0:.3e}, A={A_re:.3e}, phi0={phi0_re:.3f} rad, R^2={r2_re:.3f}")
    print(f"Im: b0={b0:.3e}, A={A_im:.3e}, phi0={phi0_im:.3f} rad, R^2={r2_im:.3f}")

    # --------- Plot (Re) --------- #
    ax_re = axes[r, 0]
    ax_re.plot(phi_cent, re_phi, lw=1.4, label="data")
    ax_re.plot(phi_cent, re_model, lw=1.2, ls="--",
               label=r"fit: $a_0+a_c\cos 2\varphi+a_s\sin 2\varphi$")
    ax_re.set_xlabel(r"$\varphi$ (rad)")
    ax_re.set_ylabel(r"Re$\{C\}$" + ("" if not normalize_by_C0 else "  (norm.)"))
    ax_re.set_title(fr"θ={theta_deg}°,  A={A_re:.2e},  φ0={phi0_re:.2f} rad,  $R^2$={r2_re:.2f}")
    ax_re.legend(loc="best")

    # --------- Plot (Im) --------- #
    ax_im = axes[r, 1]
    ax_im.plot(phi_cent, im_phi, lw=1.4, label="data")
    ax_im.plot(phi_cent, im_model, lw=1.2, ls="--",
               label=r"fit: $b_0+b_c\cos 2\varphi+b_s\sin 2\varphi$")
    ax_im.set_xlabel(r"$\varphi$ (rad)")
    ax_im.set_ylabel(r"Im$\{C\}$" + ("" if not normalize_by_C0 else "  (norm.)"))
    ax_im.set_title(fr"θ={theta_deg}°,  A={A_im:.2e},  φ0={phi0_im:.2f} rad,  $R^2$={r2_im:.2f}")
    ax_im.legend(loc="best")

plt.tight_layout()
if savefig:
    out = f"{out_prefix}_R{R0_frac:.2f}_N{N}_ns{nsamp}{'_faraday' if include_faraday else ''}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved figure → {out}")
plt.show()
