import numpy as np
import h5py, scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ================= USER SETTINGS ================= #
filename   = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
theta_list = [0, 15, 30, 45, 60, 75, 90]   # angle between LOS and <B>
N, nsamp   = 256, 192
p          = 3.0
R0_frac    = 0.30
ring_width = 3.0
nbins_phi  = 256

include_faraday = False       # keep False for LP16 pure-emitter checks
ne0, lambda_m, rm_coeff = 1.0, 0.21, 0.812

savefig = True
out_png = f"lp16_theory_overlays_R{R0_frac:.2f}_N{N}_ns{nsamp}_{'faraday' if include_faraday else 'nofaraday'}.png"
# ================================================= #

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
    # components in POS basis:
    Bp1 = B[...,0]*e1_hat[0] + B[...,1]*e1_hat[1] + B[...,2]*e1_hat[2]
    Bp2 = B[...,0]*e2_hat[0] + B[...,1]*e2_hat[1] + B[...,2]*e2_hat[2]
    Bperp = np.sqrt(Bp1**2 + Bp2**2) + 1e-30

    # Correct spin-2 emissivity: j_P ∝ (Bp1 + i Bp2)^2 * Bperp^{(p-3)/2}
    jP   = (Bp1 + 1j*Bp2)**2 * (Bperp ** ((p - 3.0)/2.0))

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
    if demean:  P = P - np.mean(P)
    if apodize: P = P * hann2d(P.shape[0])
    F = np.fft.fft2(P)
    # <PP*>
    Cstar = np.fft.ifft2(np.abs(F)**2)
    Cstar = np.fft.fftshift(Cstar).real
    # <PP> = IFFT(F(k)F(-k))
    F_c = np.fft.fftshift(F)
    Spec = F_c * F_c[::-1, ::-1]
    C = np.fft.ifft2(np.fft.ifftshift(Spec))
    C = np.fft.fftshift(C)
    # normalize by zero-lag of C_*
    C0 = Cstar[Cstar.shape[0]//2, Cstar.shape[1]//2]
    if C0 != 0:
        Cstar = Cstar / C0
        C     = C / C0
    return Cstar, C

def ring_azimuth(arr, R0_frac, ring_width, nbins):
    N = arr.shape[0]
    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x,x, indexing='ij')
    R = np.hypot(X,Y); phi = np.arctan2(Y,X)
    R0 = R0_frac*(N/2.0)
    mask = np.abs(R - R0) < ring_width
    bins = np.linspace(-np.pi, np.pi, nbins+1)
    ctrs = 0.5*(bins[:-1]+bins[1:])
    counts, _ = np.histogram(phi[mask], bins=bins)
    vals,   _ = np.histogram(phi[mask], bins=bins, weights=arr[mask])
    y = np.divide(vals, np.maximum(counts,1), out=np.zeros_like(vals, dtype=np.float64), where=counts>0)
    return ctrs, counts, y

# ---------- NO-FIT "THEORY" PROJECTIONS ----------
def weighted_projection(phi, y, counts, modes):
    """
    Parameter-free projection: y_th(φ) = y0 + Σ_m [α_c^m cos(mφ) + α_s^m sin(mφ)],
    with α = <y, u>_w / <u,u>_w (inner product weighted by counts).
    Returns y0, dict of {m: (αc, αs)}, and the reconstructed curve y_th.
    """
    w = counts.astype(np.float64)
    y0 = (w@y)/np.sum(w) if np.sum(w)>0 else np.mean(y)
    yzm = y - y0
    y_th = np.full_like(y, y0, dtype=np.float64)
    coeffs = {}
    for m in modes:
        c = np.cos(m*phi); s = np.sin(m*phi)
        cc = np.sum(w * c * c) + 1e-30
        ss = np.sum(w * s * s) + 1e-30
        ac = np.sum(w * yzm * c) / cc
        as_ = np.sum(w * yzm * s) / ss
        y_th += ac*c + as_*s
        coeffs[m] = (ac, as_)
    return y0, coeffs, y_th

def weighted_corr(y, yhat, w):
    """Weighted Pearson correlation r_w between data and theory curve."""
    ybar = (w@y)/np.sum(w); yhbar = (w@yhat)/np.sum(w)
    num  = np.sum(w * (y - ybar) * (yhat - yhbar))
    den  = np.sqrt(np.sum(w*(y-ybar)**2) * np.sum(w*(yhat-yhbar)**2) + 1e-30)
    return float(num/den) if den>0 else np.nan

# ===================== MAIN ===================== #
(Bx,By,Bz), L, nx, Brms, Bmean = load_cube(filename)

rows = len(theta_list)
fig, axes = plt.subplots(rows, 3, figsize=(12, 3.6*rows))  # [Re<PP*> | Re<PP> | Im<PP>]

for r, theta_deg in enumerate(theta_list):
    n_hat, e1_hat, e2_hat = basis_from_Bmean(Bmean, theta_deg)
    print(f"\nθ={theta_deg}°  (angle LOS–<B>)")
    pos_idx, ds = make_pos_idx(N, nsamp, L, nx, n_hat, e1_hat, e2_hat)
    P = build_P(Bx,By,Bz, pos_idx, N, nsamp, n_hat, e1_hat, e2_hat, p, ds,
                include_faraday=include_faraday, ne0=ne0, lambda_m=lambda_m, rm_coeff=rm_coeff)

    Cstar, C = correlate_PP(P, apodize=True, demean=True)

    # Ring data
    phi, counts, y_re_star = ring_azimuth(Cstar.real, R0_frac, ring_width, nbins_phi)
    _,   _,    y_re_pp     = ring_azimuth(C.real,        R0_frac, ring_width, nbins_phi)
    _,   _,    y_im_pp     = ring_azimuth(C.imag,        R0_frac, ring_width, nbins_phi)

    # ===== LP16 theory overlays (no fit) =====
    # Re<PP*> → m=2 only
    y0_s, coeff_s, yth_star = weighted_projection(phi, y_re_star, counts, modes=[2])
    r_star = weighted_corr(y_re_star, yth_star, counts)
    print(f"Re<PP*>  projection m=2: r_w={r_star:.3f}  coeffs={coeff_s[2]}")

    # Re<PP> → m=4 only (baseline) and m=2+4 (anisotropy allowed)
    _, _, yth_pp_m4      = weighted_projection(phi, y_re_pp, counts, modes=[4])
    _, _, yth_pp_m24     = weighted_projection(phi, y_re_pp, counts, modes=[2,4])
    r_pp_m4  = weighted_corr(y_re_pp, yth_pp_m4,  counts)
    r_pp_m24 = weighted_corr(y_re_pp, yth_pp_m24, counts)
    print(f"Re<PP>   proj m=4:  r_w={r_pp_m4:.3f};   proj m=2+4: r_w={r_pp_m24:.3f}")

    # Im<PP> → expect ≈0 in ensemble; still compare to m=4 & m=2+4 projections
    _, _, yth_ip_m4      = weighted_projection(phi, y_im_pp, counts, modes=[4])
    _, _, yth_ip_m24     = weighted_projection(phi, y_im_pp, counts, modes=[2,4])
    r_ip_m4  = weighted_corr(y_im_pp, yth_ip_m4,  counts)
    r_ip_m24 = weighted_corr(y_im_pp, yth_ip_m24, counts)
    print(f"Im<PP>   proj m=4:  r_w={r_ip_m4:.3f};   proj m=2+4: r_w={r_ip_m24:.3f}")

    # --------------- PLOTS ---------------
    # Re<PP*>
    ax = axes[r,0]
    ax.plot(phi, y_re_star, lw=1.2, label="data")
    ax.plot(phi, yth_star,  lw=1.4, ls="--", label="LP16 m=2 projection")
    ax.set_title(fr"θ={theta_deg}°  Re⟨PP*⟩  (r = {r_star:.2f})")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend()

    # Re<PP>
    ax = axes[r,1]
    ax.plot(phi, y_re_pp,  lw=1.2, label="data")
    ax.plot(phi, yth_pp_m4,  lw=1.2, ls="--", label="LP16 m=4")
    ax.plot(phi, yth_pp_m24, lw=1.2, ls=":",  label="LP16 m=2+4")
    ax.set_title(fr"Re⟨PP⟩  (r4={r_pp_m4:.2f}, r2+4={r_pp_m24:.2f})")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend(loc="best")

    # Im<PP>
    ax = axes[r,2]
    ax.plot(phi, y_im_pp,  lw=1.2, label="data")
    ax.plot(phi, yth_ip_m4,  lw=1.2, ls="--", label="LP16 m=4")
    ax.plot(phi, yth_ip_m24, lw=1.2, ls=":",  label="LP16 m=2+4")
    ax.axhline(0, ls=":", lw=1)
    ax.set_title(fr"Im⟨PP⟩  (r4={r_ip_m4:.2f}, r2+4={r_ip_m24:.2f})")
    ax.set_xlabel("φ (rad)"); ax.set_ylabel("(norm.)"); ax.legend(loc="best")

plt.tight_layout()
if savefig:
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print("Saved →", out_png)
