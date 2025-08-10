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
p          = 3.0                  # CR electron spectral index
R0_frac    = 0.30                 # ring radius as a fraction of half-image
ring_width_pix = 1.0              # ring thickness (pixels)
nbins_phi  = 181                  # azimuth bins
add_uniform_B0   = True           # add mean field to boost anisotropy
B0_amp_factor    = 0.5            # × Brms
savefig    = True
out_prefix = "anisotropy_compare"
# ====================================================== #

# --------------- Helpers ---------------- #
def load_cube(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2, 1, 0).astype(np.float32)  # (x,y,z)
        By = f["j_mag_field"][:].transpose(2, 1, 0).astype(np.float32)
        Bz = f["k_mag_field"][:].transpose(2, 1, 0).astype(np.float32)
        x_edges = f["x_coor"][0, 0, :]
        L = float(x_edges[-1] - x_edges[0])
    nx = Bx.shape[0]
    Brms = np.sqrt(np.mean(Bx**2 + By**2 + Bz**2)).item()
    print(f"Loaded cube {nx}^3; L={L:g}; Brms={Brms:g}")
    return (Bx, By, Bz), L, nx, Brms

def geometry(theta_deg):
    th = np.radians(theta_deg)
    n_hat  = np.array([np.sin(th), 0.0, np.cos(th)], dtype=np.float32)   # LOS
    e1_hat = np.array([np.cos(th), 0.0, -np.sin(th)], dtype=np.float32)  # POS x̂
    e2_hat = np.array([0.0, 1.0, 0.0], dtype=np.float32)                 # POS ŷ
    return n_hat, e1_hat, e2_hat

def make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat):
    # POS pixel centers in physical units
    i_idx, j_idx = np.indices((N, N), dtype=np.float32)
    x0 = ((i_idx - N/2) / N)[..., None] * L * e1_hat + \
         ((j_idx - N/2) / N)[..., None] * L * e2_hat     # (N,N,3)

    # LOS sample coordinates
    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32) # (nsamp,)
    ds = float(s[1] - s[0])

    # --- FIX: build LOS offsets with explicit dims for broadcasting ---
    # los_offsets: (nsamp,3) → add new axes to become (1,1,nsamp,3)
    los_offsets = (s[:, None] * n_hat[None, :])[None, None, :, :]        # (1,1,nsamp,3)
    pos_phys = x0[..., None, :] + los_offsets                            # (N,N,nsamp,3)
    # ------------------------------------------------------------------

    # Map to array index space [0, nx-1] with periodic wrap
    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = pos_frac * (nx - 1)
    pos_idx  = pos_idx.reshape(-1, 3).T.astype(np.float32)  # (3, N*N*nsamp) for map_coordinates
    return pos_idx, ds

def interp_field(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N, N, nsamp)

def synth_P(Bx, By, Bz, pos_idx, N, nsamp, n_hat, p, ds, B0_amp=None):
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

    # Pure emitter (no Faraday rotation)
    P = np.sum(eps * np.exp(2j * psi), axis=-1) * ds        # (N,N)
    return P

def correlation(P):
    F  = np.fft.fft2(P)
    C  = np.fft.ifft2(F * F.conj())
    C  = np.fft.fftshift(C) / P.size
    return C.real, C.imag

def ring_azimuth(C_re, C_im, R0_frac, ring_width_pix, nbins):
    N = C_re.shape[0]
    x = (np.arange(N) - N/2)
    y = (np.arange(N) - N/2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R   = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)                                   # [-π, π]

    R0 = R0_frac * (N/2.0)
    mask = np.abs(R - R0) < ring_width_pix

    bins = np.linspace(-np.pi, np.pi, nbins, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(phi[mask], bins=bins)

    def az_avg(arr):
        vals, _ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
        return np.divide(vals, np.maximum(counts, 1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)

    re_phi = az_avg(C_re)
    im_phi = az_avg(C_im)
    return centers, counts, re_phi, im_phi

def fit_component(phi, y, counts, mode="cos"):
    """Weighted LS fit: re ≈ a0 + a1 cos(2φ), im ≈ b0 + b1 sin(2φ)."""
    basis = np.cos(2.0 * phi) if mode == "cos" else np.sin(2.0 * phi)
    w = counts.astype(np.float64)
    X0 = np.ones_like(phi, dtype=np.float64)
    # Solve (X^T W X) β = X^T W y
    X = np.stack([X0, basis], axis=1)
    WX = w[:, None] * X
    XtWX = X.T @ WX
    XtWy = X.T @ (w * y)
    coeffs = np.linalg.pinv(XtWX, rcond=1e-10) @ XtWy
    y_model = X @ coeffs
    # Weighted R^2
    y_mean_w = (w @ y) / np.sum(w) if np.sum(w) > 0 else np.mean(y)
    ss_res = np.sum(w * (y - y_model)**2)
    ss_tot = np.sum(w * (y - y_mean_w)**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return y_model, coeffs, r2
# -------------------------------------- #

# ============== MAIN ============== #
(Bx, By, Bz), L, nx, Brms = load_cube(filename)
B0_amp = B0_amp_factor * Brms if add_uniform_B0 else None
print(f"Uniform B0 amplitude: {B0_amp if B0_amp else 0:.4g} (same units as cube)")

# Prepare figure with 2 columns (Re, Im) and rows = number of angles
nrows = len(theta_list)
fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.2 * nrows))
if nrows == 1:
    axes = np.array([axes])  # ensure 2D array

for r, theta_deg in enumerate(theta_list):
    print(f"\n=== Processing θ = {theta_deg}° ===")
    n_hat, e1_hat, e2_hat = geometry(theta_deg)
    pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
    P = synth_P(Bx, By, Bz, pos_idx, N, nsamp, n_hat, p, ds, B0_amp=B0_amp)
    C_re, C_im = correlation(P)
    phi_cent, counts, re_phi, im_phi = ring_azimuth(C_re, C_im, R0_frac, ring_width_pix, nbins_phi)

    # Fit "theory" shapes
    re_model, (a0, a1), r2_re = fit_component(phi_cent, re_phi, counts, mode="cos")
    im_model, (b0, b1), r2_im = fit_component(phi_cent, im_phi, counts, mode="sin")

    # --------- Plot (Re) --------- #
    ax_re = axes[r, 0]
    ax_re.plot(phi_cent, re_phi, lw=1.4, label="data")
    ax_re.plot(phi_cent, re_model, lw=1.2, ls="--", label=r"theory: $a_0+a_1\cos 2\varphi$")
    ax_re.set_xlabel(r"$\varphi$ (rad)")
    ax_re.set_ylabel(r"Re$\{C\}$")
    ax_re.set_title(fr"θ={theta_deg}°,  fit: $a_0={a0:.3g},\,a_1={a1:.3g},\,R^2={r2_re:.3f}$")

    print(fr"θ={theta_deg}°,  fit: $a_0={a0:.3g},\,a_1={a1:.3g},\,R^2={r2_re:.3f}$")

    ax_re.legend(loc="best")

    # --------- Plot (Im) --------- #
    ax_im = axes[r, 1]
    ax_im.plot(phi_cent, im_phi, lw=1.4, label="data")
    ax_im.plot(phi_cent, im_model, lw=1.2, ls="--", label=r"theory: $b_0+b_1\sin 2\varphi$")
    ax_im.set_xlabel(r"$\varphi$ (rad)")
    ax_im.set_ylabel(r"Im$\{C\}$")
    ax_im.set_title(fr"θ={theta_deg}°,  fit: $b_0={b0:.3g},\,b_1={b1:.3g},\,R^2={r2_im:.3f}$")


    print(fr"θ={theta_deg}°,  fit: $b_0={b0:.3g},\,b_1={b1:.3g},\,R^2={r2_im:.3f}$")
    
    ax_im.legend(loc="best")

plt.tight_layout()
if savefig:
    out = f"{out_prefix}_R{R0_frac:.2f}_N{N}_ns{nsamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure → {out}")
plt.show()
