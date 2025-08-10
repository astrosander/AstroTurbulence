import numpy as np
import h5py, scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ---------------- #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

# Angles measured RELATIVE TO the cube's mean magnetic field ⟨B⟩
theta_list = [0, 15, 30, 45, 60, 75, 90]  # degrees

N          = 256         # POS pixels (use 512 when happy)
nsamp      = 192         # samples along LOS
p          = 3.0         # CR electron index (~3 → ε ∝ |B⊥|^2)
R0_frac    = 0.30        # ring radius as fraction of half-image
ring_width = 2.5         # ring thickness [pixels] → better S/N
nbins_phi  = 181         # azimuth bins
m_list_pp  = [2, 4, 6]   # harmonics to fit for ⟨PP⟩ (spin-4 dominates; 2,6 are useful)
                         # for ⟨PP*⟩ we fit only m=2

# Do NOT add an artificial B0; we use the simulation's own mean field
use_cube_mean_only = True

# Internal Faraday rotation: keep OFF to match the LP16 pure-emitter regime
include_faraday = False
ne0             = 1.0
lambda_m        = 0.21
rm_coeff        = 0.812

savefig   = True
out_name  = f"lp16_angles_vs_Bmean_R{R0_frac:.2f}_N{N}_ns{nsamp}_" + ("faraday" if include_faraday else "nofaraday")
# ------------------------------------------------ #

def load_cube(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["i_mag_field"][:].transpose(2,1,0).astype(np.float32)
        By = f["j_mag_field"][:].transpose(2,1,0).astype(np.float32)
        Bz = f["k_mag_field"][:].transpose(2,1,0).astype(np.float32)
        x_edges = f["x_coor"][0,0,:]
        L = float(x_edges[-1] - x_edges[0])
    nx = Bx.shape[0]
    Brms = float(np.sqrt(np.mean(Bx**2 + By**2 + Bz**2)))
    Bmean = np.array([Bx.mean(), By.mean(), Bz.mean()], dtype=np.float64)
    return (Bx,By,Bz), L, nx, Brms, Bmean

def unit(v):
    n = np.linalg.norm(v)
    return v/n if n>0 else v

def basis_from_Bmean(Bmean, theta_deg):
    """
    Build (n_hat, e1_hat, e2_hat) so that:
      - |angle(Bmean, n_hat)| = theta_deg
      - e1_hat is along the POS projection of Bmean (sets φ=0 to ⟨B⟩⊥)
      - e2_hat completes a right-handed triad
    """
    bm = unit(Bmean.astype(np.float64))
    # Choose any axis perpendicular to bm as rotation axis (stable choice)
    # If bm is too close to z, pick x as helper and re-orthonormalize.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(bm, helper)) > 0.98:
        helper = np.array([1.0, 0.0, 0.0])
    rot_axis = unit(np.cross(bm, helper))
    # n_hat = bm rotated by +theta around rot_axis:
    th = np.radians(theta_deg)
    ct, st = np.cos(th), np.sin(th)
    # Rodrigues' rotation
    n_hat = ct*bm + st*np.cross(rot_axis, bm) + (1-ct)*np.dot(rot_axis,bm)*rot_axis
    n_hat = unit(n_hat.astype(np.float32))

    # Plane-of-sky basis: e1 along Bmean projection
    b_perp = Bmean - np.dot(Bmean, n_hat)*n_hat
    if np.linalg.norm(b_perp) < 1e-10:
        # if projection vanishes (θ≈0), pick any orthogonal e1
        e1_hat = unit(np.cross(n_hat, rot_axis)).astype(np.float32)
    else:
        e1_hat = unit(b_perp).astype(np.float32)
    e2_hat = unit(np.cross(n_hat, e1_hat)).astype(np.float32)
    return n_hat.astype(np.float32), e1_hat, e2_hat

def make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat):
    i_idx, j_idx = np.indices((N,N), dtype=np.float32)
    x0 = ((i_idx-N/2)/N)[...,None]*L*e1_hat + ((j_idx-N/2)/N)[...,None]*L*e2_hat
    s  = np.linspace(-L/2, L/2, nsamp, dtype=np.float32); ds = float(s[1]-s[0])
    los = (s[:,None]*n_hat[None,:])[None,None,:,:]  # (1,1,nsamp,3)
    pos_phys = x0[...,None,:] + los
    pos_frac = (pos_phys / L) % 1.0
    pos_idx  = (pos_frac * (nx - 1)).reshape(-1,3).T.astype(np.float32)
    return pos_idx, ds

def interp(field, pos_idx, N, nsamp):
    return ndi.map_coordinates(field, pos_idx, order=1, mode="wrap").reshape(N,N,nsamp)

def build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds,
            include_faraday=False, ne0=1.0, lambda_m=0.21, rm_coeff=0.812):
    Bx_l, By_l, Bz_l = (interp(Bx,pos_idx,N,nsamp),
                         interp(By,pos_idx,N,nsamp),
                         interp(Bz,pos_idx,N,nsamp))
    B = np.stack((Bx_l,By_l,Bz_l), axis=-1)    # (N,N,nsamp,3)
    B_par  = np.tensordot(B, n_hat, axes=([-1],[0]))    # (N,N,nsamp)
    B_perp = B - B_par[...,None]*n_hat                 # (N,N,nsamp,3)

    # ε ∝ |B⊥|^{(p+1)/2}, ψ = 1/2 atan2(B⊥_y, B⊥_x) in (e1,e2) basis
    # Project B_perp onto (e1,e2) implicitly via components already in cube basis,
    # BUT ψ must be computed in the POS basis (e1,e2). Build those components:
    # Convert B_perp vectors into the (e1, e2) components:
    e1, e2 = e1_hat, e2_hat
    Bp_e1 = B_perp[...,0]*e1[0] + B_perp[...,1]*e1[1] + B_perp[...,2]*e1[2]
    Bp_e2 = B_perp[...,0]*e2[0] + B_perp[...,1]*e2[1] + B_perp[...,2]*e2[2]

    Bp_mag = np.sqrt(Bp_e1**2 + Bp_e2**2)
    eps    = Bp_mag ** ((p+1)/2.0)
    psi    = 0.5*np.arctan2(Bp_e2, Bp_e1)

    if include_faraday:
        Phi = rm_coeff * ne0 * np.cumsum(B_par, axis=-1) * ds
        phase = 2.0*(psi + (lambda_m**2)*Phi)
    else:
        phase = 2.0*psi

    P = np.sum(eps*np.exp(1j*phase), axis=-1) * ds   # (N,N)
    return P

def hann2d(N):
    w1 = np.hanning(N); W = np.outer(w1,w1)
    return (W / W.mean()).astype(np.float32)

def correlate_PP(P, apodize=True, demean=True):
    """Return C_* = <PP*> (real) and C = <PP> (complex), normalized by C_*(0)."""
    if demean:
        P = P - np.mean(P)
    if apodize:
        P = P * hann2d(P.shape[0])
    F = np.fft.fft2(P)

    # <PP*>
    Cstar = np.fft.ifft2(np.abs(F)**2)
    Cstar = np.fft.fftshift(Cstar).real

    # <PP>  → IFFT( F(k) * F(-k) )
    F_c   = np.fft.fftshift(F)
    Spec  = F_c * F_c[::-1, ::-1]
    C     = np.fft.ifft2(np.fft.ifftshift(Spec))
    C     = np.fft.fftshift(C)

    C0 = Cstar[Cstar.shape[0]//2, Cstar.shape[1]//2]
    if C0 != 0:
        Cstar = Cstar / C0
        C     = C / C0
    return Cstar, C

def ring_azimuth(arr, R0_frac, ring_width, nbins, e1_hat):
    """Azimuthal average on a ring; φ measured from e1_hat direction."""
    N = arr.shape[0]
    # Build φ referenced to e1_hat; since our POS axes already use e1,e2 as pixel axes,
    # the angle on the grid is simply atan2(Y, X) with X || e1.
    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x, x, indexing='ij')
    phi = np.arctan2(Y, X)  # e1 along +X by construction
    R   = np.hypot(X, Y)

    R0   = R0_frac*(N/2.0)
    mask = np.abs(R - R0) < ring_width

    bins    = np.linspace(-np.pi, np.pi, nbins)
    centers = 0.5*(bins[:-1] + bins[1:])
    counts, _ = np.histogram(phi[mask], bins=bins)
    vals, _   = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    avg = np.divide(vals, np.maximum(counts,1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)
    return centers, counts, avg

def fit_multi(phi, y, counts, m_list):
    """Weighted LS: y ≈ a0 + Σ_m [a_c^m cos(mφ) + a_s^m sin(mφ)]."""
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
    amps, phases = {}, {}
    idx = 1
    for m in m_list:
        ac, as_ = beta[idx], beta[idx+1]
        amps[m]   = float(np.hypot(ac, as_))
        phases[m] = float(np.arctan2(as_, ac)/m)
        idx += 2
    return yfit, r2, amps, phases

def xi_minus(P):
    """
    LP16 / CMB-style spin-4 demodulated correlator:
      ξ_-(R) = ⟨ P(X) P(X+R) e^{-4iφ_R} ⟩
    Should be ~real (Im≈0) for a pure emitter with mirror symmetry.
    """
    # Do it in real space by rotating the separation phase
    # via a convolution with a spin-4 kernel in Fourier space.
    F = np.fft.fft2(P)
    kx = np.fft.fftfreq(P.shape[0]); ky = np.fft.fftfreq(P.shape[1])
    KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), indexing='ij')
    phi_k = np.arctan2(KY, KX)
    # e^{-4iφ_R} in real space ↔ e^{+4iφ_k} in Fourier space
    Spec = (np.fft.fftshift(F) * np.fft.fftshift(F)) * np.exp(4j*phi_k)
    xi   = np.fft.ifft2(np.fft.ifftshift(Spec))
    xi   = np.fft.fftshift(xi)
    return xi

# ----------------- MAIN ----------------- #
(Bx,By,Bz), L, nx, Brms, Bmean = load_cube(filename)
print(f"Mean field from cube: <B> = {Bmean}  (|<B>|={np.linalg.norm(Bmean):.3g})")

rows = len(theta_list)
fig, axes = plt.subplots(rows, 3, figsize=(12, 3.6*rows))  # [Re<PP*> | Re<PP> | Im<PP>]

for r, theta_deg in enumerate(theta_list):
    n_hat, e1_hat, e2_hat = basis_from_Bmean(Bmean, theta_deg)
    print(f"\nθ={theta_deg}°  (angle between LOS and <B>)")
    pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
    P = build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, p, ds,
                include_faraday=include_faraday, ne0=ne0, lambda_m=lambda_m, rm_coeff=rm_coeff)

    # Correlators
    Cstar, C = correlate_PP(P, apodize=True, demean=True)

    # Ring-averaged azimuthal curves at chosen radius
    phi, counts, re_star = ring_azimuth(Cstar.real, R0_frac, ring_width, nbins_phi, e1_hat)
    phi2, counts2, re_pp = ring_azimuth(C.real,  R0_frac, ring_width, nbins_phi, e1_hat)
    _,    _,      im_pp  = ring_azimuth(C.imag,  R0_frac, ring_width, nbins_phi, e1_hat)

    # Fits
    yfit_star, r2_star, amps_star, ph_star = fit_multi(phi, re_star, counts, m_list=[2])
    yfit_re,   r2_re,   amps_re,   ph_re   = fit_multi(phi2, re_pp, counts2, m_list=m_list_pp)
    yfit_im,   r2_im,   amps_im,   ph_im   = fit_multi(phi2, im_pp, counts2, m_list=m_list_pp)

    print(f"Re<PP*>: A2={amps_star[2]:.3e}, phase={ph_star[2]:.2f}, R^2={r2_star:.2f}")
    print(f"<PP> Re : " + " ".join([f"A{m}={amps_re[m]:.3e}" for m in m_list_pp]) + f", R^2={r2_re:.2f}")
    print(f"<PP> Im : " + " ".join([f"A{m}={amps_im[m]:.3e}" for m in m_list_pp]) + f", R^2={r2_im:.2f}")

    # --- LP16 sanity: ξ_-(R) should be ~ real (Im ~ 0)
    xi = xi_minus(P)
    xi_phi, xi_counts, xi_re = ring_azimuth(xi.real, R0_frac, ring_width, nbins_phi, e1_hat)
    _,      _,         xi_im = ring_azimuth(xi.imag, R0_frac, ring_width, nbins_phi, e1_hat)
    im_xi_max = np.max(np.abs(xi_im))
    print(f"ξ_-(R) imaginary (should be ~0): max|Im|={im_xi_max:.2e}")

    # -------- PLOTS --------
    ax = axes[r,0]  # Re<PP*>
    ax.plot(phi, re_star, lw=1.5, label="data")
    ax.plot(phi, yfit_star, lw=1.2, ls="--", label="fit m=2")
    ax.set_title(fr"θ={theta_deg}°  Re⟨PP*⟩  A2={amps_star[2]:.2e}, $R^2$={r2_star:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    ax = axes[r,1]  # Re<PP>
    ax.plot(phi2, re_pp, lw=1.5, label="data")
    ax.plot(phi2, yfit_re, lw=1.2, ls="--", label="fit m=2+4+6")
    ax.set_title(fr"Re⟨PP⟩  " + " ".join([f"A{m}={amps_re[m]:.2e}" for m in m_list_pp]) + f", $R^2$={r2_re:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    ax = axes[r,2]  # Im<PP>
    ax.plot(phi2, im_pp, lw=1.5, label="data")
    ax.plot(phi2, yfit_im, lw=1.2, ls="--", label="fit m=2+4+6")
    ax.axhline(0, ls=":", lw=1)
    ax.set_title(fr"Im⟨PP⟩  " + " ".join([f"A{m}={amps_im[m]:.2e}" for m in m_list_pp]) + f", $R^2$={r2_im:.2f}")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend(loc="best")

plt.tight_layout()
if savefig:
    plt.savefig(out_name + ".png", dpi=180, bbox_inches="tight")
    print("Saved →", out_name + ".png")
plt.show()
