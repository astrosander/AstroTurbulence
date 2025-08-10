import numpy as np
import h5py, scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ------------------ USER SETTINGS ------------------ #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
theta_list = [0, 15, 30, 45, 60, 75, 90]

N          = 256
nsamp      = 192
p          = 3.0
R0_frac    = 0.30
ring_width = 2.5          # wider ring → better S/N
nbins_phi  = 181

add_uniform_B0 = True
B0_amp_factor  = 0.5       # × Brms along +z

# Faraday: OFF by default (turn ON to get physical Im in <PP>)
include_faraday = False
ne0             = 1.0
lambda_m        = 0.21
rm_coeff        = 0.812

savefig   = True
out_name  = f"lp16_fix_PP_PPc_R{R0_frac:.2f}_N{N}_ns{nsamp}_" + ("faraday" if include_faraday else "nofaraday")
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
    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32)
    ds = float(s[1]-s[0])
    los_offsets = (s[:,None]*n_hat[None,:])[None,None,:,:]                         # (1,1,nsamp,3)
    pos_phys = x0[...,None,:] + los_offsets                                       # (N,N,nsamp,3)
    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = (pos_frac * (nx - 1)).reshape(-1,3).T.astype(np.float32)           # (3, N*N*nsamp)
    return pos_idx, ds

def interp(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N,N,nsamp)

def build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds, B0_amp=0.0,
            include_faraday=False, ne0=1.0, lambda_m=0.21, rm_coeff=0.812):
    Bx_l, By_l, Bz_l = (interp(Bx,pos_idx,N,nsamp),
                         interp(By,pos_idx,N,nsamp),
                         interp(Bz,pos_idx,N,nsamp))
    B = np.stack((Bx_l,By_l,Bz_l), axis=-1)                                    # (N,N,nsamp,3)
    if B0_amp>0:
        B += np.array([0,0,B0_amp], dtype=B.dtype)[None,None,None,:]

    B_par  = np.tensordot(B, n_hat, axes=([-1],[0]))                           # (N,N,nsamp)
    B_perp = B - B_par[...,None]*n_hat                                          # (N,N,nsamp,3)
    Bp_mag = np.linalg.norm(B_perp, axis=-1)
    eps    = Bp_mag ** ((p+1)/2.0)
    psi    = 0.5*np.arctan2(B_perp[...,1], B_perp[...,0])

    if include_faraday:
        Phi = rm_coeff * ne0 * np.cumsum(B_par, axis=-1) * ds
        phase = 2.0*(psi + (lambda_m**2)*Phi)
    else:
        phase = 2.0*psi

    P = np.sum(eps*np.exp(1j*phase), axis=-1) * ds                             # (N,N)
    return P

def hann2d(N):
    w1 = np.hanning(N); W = np.outer(w1,w1)
    return (W / W.mean()).astype(np.float32)

def correlate_both(P, apodize=True, demean=True):
    """Return C_* = <PP*> (real) and C = <PP> (complex)."""
    if demean:
        P = P - np.mean(P)
    if apodize:
        P = P * hann2d(P.shape[0])

    F = np.fft.fft2(P)

    # 1) C_* = <P P*>  ⇒ IFFT(|F|^2)  (strictly real)
    Cstar = np.fft.ifft2(np.abs(F)**2)
    Cstar = np.fft.fftshift(Cstar).real

    # 2) C   = <P P>    ⇒ IFFT( F(k) * F(-k) )
    F_c   = np.fft.fftshift(F)
    F_neg = F_c[::-1, ::-1]                      # values at -k
    Spec  = F_c * F_neg
    C     = np.fft.ifft2(np.fft.ifftshift(Spec))
    C     = np.fft.fftshift(C)

    # Normalize by zero-lag of C_*
    C0 = Cstar[Cstar.shape[0]//2, Cstar.shape[1]//2]
    if C0 != 0:
        Cstar = Cstar / C0
        C     = C / C0
    return Cstar, C

def ring_azimuth(arr, R0_frac, ring_width, nbins):
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
    vals,_   = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    avg = np.divide(vals, np.maximum(counts,1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)
    return centers, counts, avg

def fit_multi(phi, y, counts, m_list):
    """Weighted LS fit: y ≈ a0 + Σ_m [a_c^m cos(mφ) + a_s^m sin(mφ)]."""
    w = counts.astype(np.float64)
    cols = [np.ones_like(phi, dtype=np.float64)]
    for m in m_list:
        cols += [np.cos(m*phi), np.sin(m*phi)]
    X = np.stack(cols, axis=1)
    WX = w[:,None]*X
    beta = np.linalg.pinv(X.T @ WX, rcond=1e-12) @ (X.T @ (w*y))
    yfit = X @ beta
    ybar = (w@y)/np.sum(w) if np.sum(w)>0 else np.mean(y)
    ssr  = np.sum(w*(y - yfit)**2); sst = np.sum(w*(y - ybar)**2)
    r2   = 1.0 - ssr/sst if sst>0 else np.nan

    # amplitudes & phases per m
    amps, phases = {}, {}
    idx = 1
    for m in m_list:
        ac, as_ = beta[idx], beta[idx+1]
        amps[m]   = float(np.hypot(ac, as_))
        phases[m] = float(np.arctan2(as_, ac)/m)
        idx += 2
    return yfit, beta, r2, amps, phases

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

    Cstar, C = correlate_both(P, apodize=True, demean=True)
    leak = np.abs(np.imag(Cstar)).max()
    print(f"Im<PP*> leak = {leak:.3e} (should be ~0)")

    # --- Ring averages
    phi, counts, re_star = ring_azimuth(Cstar.real, R0_frac, ring_width, nbins_phi)
    _,   _,     im_star  = ring_azimuth(Cstar.imag, R0_frac, ring_width, nbins_phi)

    phi2, counts2, re_pp = ring_azimuth(C.real,  R0_frac, ring_width, nbins_phi)
    _,    _,      im_pp  = ring_azimuth(C.imag,  R0_frac, ring_width, nbins_phi)

    # --- Fits
    # Re<PP*>: only m=2 expected
    yfit_star, _, r2_star, amps_star, ph_star = fit_multi(phi, re_star, counts, m_list=[2])
    print(f"Re<PP*>: A2={amps_star[2]:.3e}, phase={ph_star[2]:.2f}, R^2={r2_star:.2f}")

    # <PP>: fit m=2 AND m=4 simultaneously
    yfit_re, _, r2_re, amps_re, ph_re = fit_multi(phi2, re_pp, counts2, m_list=[2,4])
    yfit_im, _, r2_im, amps_im, ph_im = fit_multi(phi2, im_pp, counts2, m_list=[2,4])
    print(f"<PP> Re: A2={amps_re[2]:.3e}, A4={amps_re[4]:.3e}, R^2={r2_re:.2f}")
    print(f"<PP> Im: A2={amps_im[2]:.3e}, A4={amps_im[4]:.3e}, R^2={r2_im:.2f}")

    # ---------- PLOTS ----------
    # Re <PP*>
    ax = axes[r,0]
    ax.plot(phi, re_star, lw=1.5, label="data")
    ax.plot(phi, yfit_star,  lw=1.2, ls="--", label="fit m=2")
    ax.set_title(fr"θ={theta_deg}°  Re⟨PP*⟩  A2={amps_star[2]:.2e}, $R^2$={r2_star:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    # Im <PP*> (leak check)
    ax = axes[r,1]
    ax.plot(phi, im_star, lw=1.2); ax.axhline(0, ls="--", lw=1)
    ax.set_title("Im⟨PP*⟩ (≈0)"); ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)")

    # Re <PP>
    ax = axes[r,2]
    ax.plot(phi2, re_pp, lw=1.5, label="data")
    ax.plot(phi2, yfit_re, lw=1.2, ls="--", label="fit m=2+4")
    ax.set_title(fr"Re⟨PP⟩  A2={amps_re[2]:.2e}, A4={amps_re[4]:.2e}, $R^2$={r2_re:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    # Im <PP>
    ax = axes[r,3]
    ax.plot(phi2, im_pp, lw=1.5, label="data")
    ax.plot(phi2, yfit_im, lw=1.2, ls="--", label="fit m=2+4")
    ax.axhline(0, ls=":", lw=1)
    ax.set_title(fr"Im⟨PP⟩  A2={amps_im[2]:.2e}, A4={amps_im[4]:.2e}, $R^2$={r2_im:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

plt.tight_layout()
if savefig:
    plt.savefig(out_name + ".png", dpi=180, bbox_inches="tight")
    print("Saved →", out_name + ".png")
plt.show()
