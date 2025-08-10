import numpy as np
import h5py, scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ------------------ USER SETTINGS ------------------ #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
theta_list = [30, 45, 60]

N          = 256       # POS pixels (use 512 after testing)
nsamp      = 192       # samples along LOS
p          = 3.0       # CR electron index (~3)
R0_frac    = 0.30      # ring radius fraction of half-image
ring_width = 1.0       # ring thickness [pix]
nbins_phi  = 181       # azimuth bins
m_orders   = (2, 4)    # fit harmonics for <PP> (spin-2 & spin-4)

# Mean field boost (optional)
add_uniform_B0 = False
B0_amp_factor  = 0.5   # × Brms along +z

# Internal Faraday rotation (turn ON to get nonzero Im for <PP>)
include_faraday = False
ne0             = 1.0       # arbitrary units (uniform ne)
lambda_m        = 0.21      # wavelength [m]
rm_coeff        = 0.812     # rad m^-2 per (cm^-3 μG pc); treated relatively here

savefig   = True
out_name  = "lp16_compare_PP_PPc_R{:.2f}_N{}_ns{}_{}".format(R0_frac, N, nsamp,
                     "faraday" if include_faraday else "nofaraday")
# --------------------------------------------------- #

def load_cube(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2,1,0).astype(np.float32)
        By = f["j_mag_field"][:].transpose(2,1,0).astype(np.float32)
        Bz = f["k_mag_field"][:].transpose(2,1,0).astype(np.float32)
        x_edges = f["x_coor"][0,0,:]
        L = float(x_edges[-1] - x_edges[0])
    nx = Bx.shape[0]
    Brms = float(np.sqrt(np.mean(Bx**2 + By**2 + Bz**2)))
    print(f"Loaded cube {nx}^3; L={L:g}; Brms={Brms:g}")
    return (Bx,By,Bz), L, nx, Brms

def geom(theta_deg):
    th = np.radians(theta_deg).astype(np.float32)
    n  = np.array([np.sin(th),0.0,np.cos(th)], np.float32)
    e1 = np.array([np.cos(th),0.0,-np.sin(th)], np.float32)
    e2 = np.array([0.0,1.0,0.0], np.float32)
    return n,e1,e2

def make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat):
    i_idx, j_idx = np.indices((N,N), dtype=np.float32)
    x0 = ((i_idx-N/2)/N)[...,None]*L*e1_hat + ((j_idx-N/2)/N)[...,None]*L*e2_hat  # (N,N,3)
    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32)                          # (nsamp,)
    ds = float(s[1]-s[0])
    los_offsets = (s[:,None]*n_hat[None,:])[None,None,:,:]                         # (1,1,nsamp,3)
    pos_phys = x0[...,None,:] + los_offsets                                       # (N,N,nsamp,3)
    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = (pos_frac * (nx - 1)).reshape(-1,3).T.astype(np.float32)           # (3, N*N*nsamp)
    return pos_idx, ds

def interp(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N,N,nsamp)

def build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds, B0_amp=None,
            include_faraday=False, ne0=1.0, lambda_m=0.21, rm_coeff=0.812):
    Bx_l, By_l, Bz_l = (interp(Bx,pos_idx,N,nsamp),
                         interp(By,pos_idx,N,nsamp),
                         interp(Bz,pos_idx,N,nsamp))
    B = np.stack((Bx_l,By_l,Bz_l), axis=-1)                                    # (N,N,nsamp,3)
    if B0_amp is not None and B0_amp>0:
        B += np.array([0,0,B0_amp], dtype=B.dtype)[None,None,None,:]

    B_par  = np.tensordot(B, n_hat, axes=([-1],[0]))                           # (N,N,nsamp)
    B_perp = B - B_par[...,None]*n_hat                                          # (N,N,nsamp,3)
    Bp_mag = np.linalg.norm(B_perp, axis=-1)
    eps    = Bp_mag ** ((p+1)/2.0)
    psi    = 0.5*np.arctan2(B_perp[...,1], B_perp[...,0])

    if include_faraday:
        Phi = rm_coeff * ne0 * np.cumsum(B_par, axis=-1) * ds                  # relative units
        phase = 2.0*(psi + (lambda_m**2)*Phi)
    else:
        phase = 2.0*psi

    P = np.sum(eps*np.exp(1j*phase), axis=-1) * ds                             # (N,N)
    return P

def hann2d(N):
    w1 = np.hanning(N); W = np.outer(w1,w1)
    # normalize so mean power not dramatically changed
    return (W / W.mean()).astype(np.float32)

def correlate_PP(P, apodize=True, demean=True):
    """Return both correlators on the same apodized/demeaned map."""
    if demean:
        P = P - np.mean(P)
    if apodize:
        W = hann2d(P.shape[0])
        P = P * W

    F = np.fft.fft2(P)

    # 1) C_* = <P P*>  (MUST be real)
    power = (np.abs(F)**2).astype(np.float64)               # strictly real
    Cstar = np.fft.ifft2(power)                             # numerical C_* (complex dtype)
    Cstar = np.fft.fftshift(Cstar).real                     # force real

    # 2) C = <P P>  (complex; carries parity info)
    Sspec = F * F                                           # NOT Hermitian
    C = np.fft.ifft2(Sspec)
    C = np.fft.fftshift(C)                                  # complex

    # Optional normalization by zero-lag of C_*
    C0 = Cstar[Cstar.shape[0]//2, Cstar.shape[1]//2]
    if C0 != 0:
        Cstar = Cstar / C0
        C     = C / C0

    return Cstar, C

def ring_azimuth(arr, R0_frac, ring_width, nbins):
    """Azimuthal average on a ring; returns (phi_centers, counts, avg_values)."""
    N = arr.shape[0]
    x = np.arange(N) - N/2
    X,Y = np.meshgrid(x,x, indexing='ij')
    R   = np.hypot(X,Y)
    phi = np.arctan2(Y,X)

    R0   = R0_frac*(N/2.0)
    mask = np.abs(R - R0) < ring_width

    bins   = np.linspace(-np.pi, np.pi, nbins)
    centers= 0.5*(bins[:-1] + bins[1:])
    counts,_ = np.histogram(phi[mask], bins=bins)

    vals,_ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.nan_to_num(vals / np.maximum(counts,1))
    return centers, counts, avg

def fit_harmonics(phi, y, counts, m):
    """Weighted fit: y ≈ a0 + ac cos(m phi) + as sin(m phi)."""
    w = counts.astype(np.float64)
    X = np.stack([np.ones_like(phi),
                  np.cos(m*phi), np.sin(m*phi)], axis=1).astype(np.float64)
    WX  = w[:,None]*X
    beta= np.linalg.pinv(X.T @ WX, rcond=1e-12) @ (X.T @ (w*y))
    yfit= X @ beta
    # weighted R^2
    ybar = (w@y)/np.sum(w) if np.sum(w)>0 else np.mean(y)
    ssr  = np.sum(w*(y - yfit)**2)
    sst  = np.sum(w*(y - ybar)**2)
    r2   = 1.0 - ssr/sst if sst>0 else np.nan
    a0,ac,as_ = beta
    A   = float(np.hypot(ac,as_))
    ph0 = float(np.arctan2(as_,ac)/m)   # phase of the m-mode, divided by m
    return yfit, (a0,ac,as_), A, ph0, r2

# ------------------ MAIN ------------------ #
(Bx,By,Bz), L, nx, Brms = load_cube(filename)
B0_amp = B0_amp_factor*Brms if add_uniform_B0 else 0.0
print(f"Uniform B0 amplitude: {B0_amp:.6g}")
print(f"Internal Faraday: {include_faraday} (λ={lambda_m} m, ne0={ne0})\n")

rows = len(theta_list)
fig, axes = plt.subplots(rows, 4, figsize=(16, 3.6*rows))  # [Re C_* | Im C_* | Re C | Im C]

for r, theta_deg in enumerate(theta_list):
    print(f"=== θ = {theta_deg}° ===")
    n_hat, e1_hat, e2_hat = geom(theta_deg)
    pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
    P  = build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds,
                 B0_amp=B0_amp, include_faraday=include_faraday,
                 ne0=ne0, lambda_m=lambda_m, rm_coeff=rm_coeff)

    Cstar, C = correlate_PP(P, apodize=True, demean=True)
    # --- sanity: Im(C_*) leakage should be ~0 everywhere ---
    Cstar_im_leak = np.abs(np.imag(Cstar)).max()
    print(f"Leak check Im<C P*> max = {Cstar_im_leak:.3e} (should be ~ machine-0)")

    # Ring averages
    phi, nbin, re_star = ring_azimuth(Cstar.real, R0_frac, ring_width, nbins_phi)
    _,    _, im_star   = ring_azimuth(Cstar.imag, R0_frac, ring_width, nbins_phi)  # just to plot leak (should be ~0)

    phi2, _, re_pp = ring_azimuth(C.real, R0_frac, ring_width, nbins_phi)
    _,    _, im_pp = ring_azimuth(C.imag, R0_frac, ring_width, nbins_phi)

    # --- Fit m=2 on Re<C_*>, this is the LP16 quadrupole for PP* ---
    yfit2, pars2, A2, ph2, r2 = fit_harmonics(phi, re_star, nbin, m=2)
    print(f"Re<C P*>: A_m=2={A2:.3e}, phase={ph2:.2f} rad, R^2={r2:.2f}")
    # Also report any m=1 leakage (should be tiny, tests recentering/apodization)
    _, _, A1_leak, _, r2_1 = fit_harmonics(phi, re_star, nbin, m=1)
    print(f"  m=1 leakage (should be small): A={A1_leak:.3e}, R^2={r2_1:.2f}")

    # --- For <PP> (complex), try both m=2 and m=4 and report the better one ---
    best = None
    for m in m_orders:
        _, _, A_re, ph_re, r2_re = fit_harmonics(phi2, re_pp, nbin, m=m)
        _, _, A_im, ph_im, r2_im = fit_harmonics(phi2, im_pp, nbin, m=m)
        score = r2_re + r2_im
        if (best is None) or (score > best[0]):
            best = (score, m, A_re, ph_re, r2_re, A_im, ph_im, r2_im)
    score,mopt,A_re,ph_re,r2_re,A_im,ph_im,r2_im = best
    print(f"<P P>: best m={mopt} | Re: A={A_re:.3e}, φ0={ph_re:.2f}, R^2={r2_re:.2f} | "
          f"Im: A={A_im:.3e}, φ0={ph_im:.2f}, R^2={r2_im:.2f}")

    # ---------- PLOTS ----------
    # Re <PP*>
    ax = axes[r,0]
    ax.plot(phi, re_star, lw=1.5, label="data")
    ax.plot(phi, yfit2,  lw=1.2, ls="--", label="fit m=2")
    ax.set_title(fr"θ={theta_deg}°  Re$\langle PP^*\rangle$  A$_2$={A2:.2e}, $R^2$={r2:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    # Im <PP*> (should be ~0; shows only leakage)
    ax = axes[r,1]
    ax.plot(phi, im_star, lw=1.2)
    ax.axhline(0, ls="--", lw=1)
    ax.set_title(fr"Im$\langle PP^*\rangle$ (leak check)")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)")

    # Re <PP> (complex)
    yfit_re, *_ = fit_harmonics(phi2, re_pp, nbin, m=mopt)
    ax = axes[r,2]
    ax.plot(phi2, re_pp, lw=1.5, label="data")
    ax.plot(phi2, yfit_re, lw=1.2, ls="--", label=f"fit m={mopt}")
    ax.set_title(fr"Re$\langle PP\rangle$  (m={mopt})  A={A_re:.2e}, $R^2$={r2_re:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    # Im <PP>
    yfit_im, *_ = fit_harmonics(phi2, im_pp, nbin, m=mopt)
    ax = axes[r,3]
    ax.plot(phi2, im_pp, lw=1.5, label="data")
    ax.plot(phi2, yfit_im, lw=1.2, ls="--", label=f"fit m={mopt}")
    ax.axhline(0, ls=":", lw=1)
    ax.set_title(fr"Im$\langle PP\rangle$  (m={mopt})  A={A_im:.2e}, $R^2$={r2_im:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

plt.tight_layout()
if savefig:
    plt.savefig(out_name + ".png", dpi=180, bbox_inches="tight")
    print("Saved →", out_name + ".png")
plt.show()
