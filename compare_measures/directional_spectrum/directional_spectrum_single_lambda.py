import numpy as np
import h5py, matplotlib.pyplot as plt

h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
los_axis = 2
C = 1.0
emit_frac = (0.15, 1.00)
screen_frac = (0.00, 0.10)
ring_bins = 48*2
kfit_bounds = (4,25)


import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# kfit_bounds = (75,96)

def load_fields(p):
    with h5py.File(p,"r") as f:
        Bx = f["i_mag_field"][()]
        By = f["j_mag_field"][()]
        Bz = f["k_mag_field"][()]
        ne = f["gas_density"][()]
    return Bx,By,Bz,ne

def polarized_emissivity_simple(Bx,By,gamma=2.0):
    if gamma==2.0:
        return (Bx + 1j*By)**2
    Bp2 = Bx**2 + By**2
    amp = np.power(np.maximum(Bp2, np.finfo(Bp2.dtype).eps), 0.5*(gamma-2.0))
    return amp*(Bx+1j*By)**2

def faraday_density(ne,Bpar,C=1.0):
    return C*ne*Bpar

def move_los(a,axis):
    return np.moveaxis(a,axis,0)

def separated_P_map(Pi,phi,lam,los_axis,emit_frac,screen_frac):
    Pi_l = move_los(Pi,los_axis)
    ph_l = move_los(phi,los_axis)
    Nz = Pi_l.shape[0]
    e0,e1 = int(emit_frac[0]*Nz), int(emit_frac[1]*Nz)
    s0,s1 = int(screen_frac[0]*Nz), int(screen_frac[1]*Nz)
    P_emit = Pi_l[e0:e1].sum(0)
    P_emit = P_emit - P_emit.mean()
    Phi = ph_l[s0:s1].sum(0)
    sigma_RM = Phi.std()              
    # print(sigma_RM)
    return P_emit*np.exp(2j*(lam**2)*Phi), sigma_RM

def hann2d(ny,nx):
    wy = 0.5*(1-np.cos(2*np.pi*np.arange(ny)/(ny-1)))
    wx = 0.5*(1-np.cos(2*np.pi*np.arange(nx)/(nx-1)))
    W = wy[:,None]*wx[None,:]
    return W/np.sqrt((W**2).mean())

def centered_indices(ny,nx):
    iy = np.arange(-ny//2, ny - ny//2)
    ix = np.arange(-nx//2, nx - nx//2)
    return np.meshgrid(ix,iy)

def ring_average(field2d, ring_bins=48, k_min=3.0, k_max=None, apod=True, energy_like=False):
    F = field2d
    ny,nx = F.shape
    if apod:
        F = F*hann2d(ny,nx)
    Fk = np.fft.fftshift(np.fft.fft2(F))
    S = (Fk*np.conj(Fk)).real/(ny*nx)**2
    kx,ky = centered_indices(ny,nx)
    k = np.sqrt(kx**2+ky**2)
    if k_max is None:
        k_max = min(kx.max(),ky.max())
    edges = np.linspace(k_min,k_max,ring_bins+1)
    kc = 0.5*(edges[1:]+edges[:-1])
    Pk = np.zeros_like(kc)
    cnt = np.zeros_like(kc,int)
    for i in range(ring_bins):
        m = (k>=edges[i])&(k<edges[i+1])
        cnt[i] = m.sum()
        Pk[i] = S[m].mean() if cnt[i]>0 else np.nan
    good = (cnt>10)&np.isfinite(Pk)
    kc,Pk = kc[good],Pk[good]
    if energy_like:
        Pk = 2*np.pi*kc*Pk
    return kc,Pk,S,kx,ky,edges

def ring_average_realspace(A, ring_bins=64, r_min=0.5, r_max=None):
    ny,nx = A.shape
    A = np.fft.fftshift(A)
    y,x = centered_indices(ny,nx)
    r = np.sqrt(x**2+y**2)
    if r_max is None:
        r_max = min(x.max(),y.max())
    edges = np.linspace(r_min,r_max,ring_bins+1)
    rc = 0.5*(edges[1:]+edges[:-1])
    Sr = np.zeros_like(rc)
    cnt = np.zeros_like(rc,int)
    for i in range(ring_bins):
        m = (r>=edges[i])&(r<edges[i+1])
        cnt[i] = m.sum()
        Sr[i] = A[m].mean() if cnt[i]>0 else np.nan
    good = (cnt>20)&np.isfinite(Sr)
    return rc[good], Sr[good]

def extract_emit_and_phi(Pi, phi, los_axis, emit_frac, screen_frac):
    Pi_l = move_los(Pi, los_axis)
    ph_l = move_los(phi, los_axis)
    Nz = Pi_l.shape[0]
    e0,e1 = int(emit_frac[0]*Nz), int(emit_frac[1]*Nz)
    s0,s1 = int(screen_frac[0]*Nz), int(screen_frac[1]*Nz)
    P_emit = Pi_l[e0:e1].sum(0)
    P_emit = P_emit - P_emit.mean()
    Phi = ph_l[s0:s1].sum(0)
    return P_emit, Phi

def radial_corr_length_unbiased(field2d, bins=200, method="efold"):
    F = np.asarray(field2d)
    if np.iscomplexobj(F):
        F = np.abs(F)
    F = np.nan_to_num(F - np.nanmean(F))
    ny, nx = F.shape
    py, px = 2*ny, 2*nx
    Z = np.zeros((py, px), dtype=float)
    Z[:ny, :nx] = F
    G = np.fft.fft2(Z)
    C = np.fft.ifft2(np.abs(G)**2).real
    C = np.fft.fftshift(C)
    cy, cx = py//2, px//2
    C = C[cy-(ny-1):cy+ny, cx-(nx-1):cx+nx]  # linear ACF support
    C /= C.max() if C.max() != 0 else 1.0

    y = np.arange(-(ny-1), ny)
    x = np.arange(-(nx-1), nx)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    r = R.ravel()
    c = C.ravel()

    edges = np.linspace(0, r.max(), bins+1)
    rc = 0.5*(edges[1:]+edges[:-1])
    Cr = np.empty_like(rc); Cr[:] = np.nan
    for i in range(bins):
        m = (r >= edges[i]) & (r < edges[i+1])
        if np.any(m):
            Cr[i] = np.nanmean(c[m])

    good = np.isfinite(Cr)
    rc, Cr = rc[good], Cr[good]
    target = np.exp(-1.0) if method == "efold" else 0.5
    idx = np.where(Cr <= target)[0]
    if idx.size:
        j = idx[0]
        rlen = rc[j] if j == 0 else np.interp(target, [Cr[j-1], Cr[j]], [rc[j-1], rc[j]])
    else:
        rlen = np.nan
    return rlen, rc, Cr

def find_break_by_rss(kc, Pdir, kfit_bounds=(4,25), min_pts=6):
    m = np.isfinite(Pdir) & (Pdir > 0)
    if kfit_bounds is not None:
        m &= (kc >= kfit_bounds[0]) & (kc <= kfit_bounds[1])
    x = np.log(kc[m])
    y = np.log(Pdir[m])
    best_i, best_sse = None, np.inf
    for i in range(min_pts, len(x) - min_pts):
        c1 = np.polyfit(x[:i], y[:i], 1)
        c2 = np.polyfit(x[i:], y[i:], 1)
        sse = ((y[:i] - (c1[0]*x[:i] + c1[1]))**2).sum() + ((y[i:] - (c2[0]*x[i:] + c2[1]))**2).sum()
        if sse < best_sse:
            best_sse, best_i, c1b, c2b = sse, i, c1, c2
    k_break = np.exp(x[best_i])
    frac = (k_break - kc.min())/(kc.max() - kc.min())
    return frac, (c1b[1], c1b[0]), (c2b[1], c2b[0]), k_break

def plot_fit(ax, kc, Pdir, frac_min, frac_max, color, isPlot=True, linestyle='-', label_prefix='Fit'):
    k_min = kc.min()
    k_max = kc.max()
    k_range = k_max - k_min
    k_min_fit = k_min + frac_min * k_range
    k_max_fit = k_min + frac_max * k_range
    
    fit_mask = (kc >= k_min_fit) & (kc <= k_max_fit)
    if fit_mask.sum() > 2:
        kc_fit = kc[fit_mask]
        Pdir_fit = Pdir[fit_mask]
        log_k = np.log(kc_fit)
        log_P = np.log(Pdir_fit)
        coeffs = np.polyfit(log_k, log_P, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        k_fit_line = np.logspace(np.log10(kc_fit.min()), np.log10(kc_fit.max()), 100)
        P_fit_line = np.exp(intercept) * k_fit_line**slope
        
        if isPlot:
            ax.loglog(k_fit_line, P_fit_line, linestyle=linestyle, lw=3.5, color=color,
                       label=f'slope = {slope:.2f}', alpha=0.9)
        return intercept, slope
    return None

def _sse(x, y, a, b):
    return np.square(y - (a*x + b)).sum()

BREAK_STATE = {"prev_frac": None, "prev_slope": None, "active": False, "prev_out": None, "last_order_key": None}

def _whole_slope(kc, Pdir):
    m = np.isfinite(Pdir) & (Pdir > 0)
    x, y = np.log(kc[m]), np.log(Pdir[m])
    a, b = np.polyfit(x, y, 1)
    return a, b

def _two_segment_candidates(kc, Pdir, min_pts=6, cont_eps=0.08):
    m = np.isfinite(Pdir) & (Pdir > 0)
    x, y = np.log(kc[m]), np.log(Pdir[m])
    n = x.size
    if n < 2*min_pts+1:
        return [], None
    a1, b1 = np.polyfit(x, y, 1)
    rss1 = np.sum((y - (a1*x + b1))**2)
    bic1 = n*np.log(rss1/n) + 2*np.log(n)

    cands = []
    xmin, xmax = x.min(), x.max()
    for i in range(min_pts, n-min_pts):
        aL, bL = np.polyfit(x[:i], y[:i], 1)
        aR, bR = np.polyfit(x[i:], y[i:], 1)
        rss2 = np.sum((y[:i] - (aL*x[:i] + bL))**2) + np.sum((y[i:] - (aR*x[i:] + bR))**2)
        bic2 = n*np.log(rss2/n) + 4*np.log(n)
        dBIC = bic1 - bic2
        kbr = np.exp(x[i])
        yL = np.exp(bL)*kbr**aL
        yR = np.exp(bR)*kbr**aR
        if abs(np.log(yL) - np.log(yR)) > cont_eps:
            continue
        frac = (x[i] - xmin) / (xmax - xmin)
        cands.append((dBIC, frac, kbr, (bL, aL), (bR, aR)))
    cands.sort(key=lambda t: (t[1], -t[0]))
    return cands, (a1, b1)

def _select_break_stable(cands, s_whole, *, state, tau_on=6.0, tau_off=3.0, lam_frac=3.0, lam_s=1.2, monotonic=True):
    prev_f = state.get("prev_frac")
    prev_s = state.get("prev_slope")
    active = state.get("active", False)

    thresh = tau_off if active else tau_on
    cands = [c for c in cands if c[0] >= thresh]
    if not cands:
        state.update(prev_slope=s_whole, active=False)
        return s_whole, None

    if monotonic and (prev_f is not None):
        cands = [c for c in cands if c[1] >= prev_f - 1e-3]
        if not cands:
            state.update(prev_slope=s_whole, active=active)
            return s_whole, None

    def cost(c):
        dBIC, frac, _, (_, aL), _ = c
        pf = 0.0 if prev_f is None else (frac - prev_f)**2
        ps = 0.0 if prev_s is None else (aL - prev_s)**2
        return lam_frac*pf + lam_s*ps + 0.05*frac - dBIC/5.0

    best = min(cands, key=cost)
    sL = best[3][1]
    state.update(prev_frac=best[1], prev_slope=sL, active=True)
    return sL, best

def _ema(s, state, alpha=0.3):
    p = state.get("prev_out")
    out = s if (p is None) else (alpha*s + (1.0-alpha)*p)
    state["prev_out"] = out
    return out

def two_segment_break_bic(kc, Pdir, min_pts=6, cont_eps=0.08):
    m = np.isfinite(Pdir) & (Pdir > 0)
    x = np.log(kc[m]); y = np.log(Pdir[m])
    n = x.size
    if n < 2*min_pts+1: return None
    a1, b1 = np.polyfit(x, y, 1)
    rss1 = np.sum((y - (a1*x + b1))**2)
    bic1 = n*np.log(rss1/n) + 2*np.log(n)
    best = None
    for i in range(min_pts, n-min_pts):
        aL, bL = np.polyfit(x[:i], y[:i], 1)
        aR, bR = np.polyfit(x[i:], y[i:], 1)
        rss2 = np.sum((y[:i]-(aL*x[:i]+bL))**2) + np.sum((y[i:]-(aR*x[i:]+bR))**2)
        bic2 = n*np.log(rss2/n) + 4*np.log(n)
        dBIC = bic1 - bic2
        k_break = np.exp(x[i])
        yL = np.exp(bL)*k_break**aL
        yR = np.exp(bR)*k_break**aR
        if abs(np.log(yL) - np.log(yR)) > cont_eps:
            continue
        frac = (x[i] - x.min())/(x.max() - x.min())
        if (best is None) or (dBIC > best[0]):
            best = (dBIC, k_break, (bL, aL), (bR, aR), frac)
    return best

def smooth_select(s_whole, s_left, dBIC, frac, tau=6.0, width=1.5, fmin=0.04):
    if (dBIC is None) or (frac is None) or (frac < fmin):
        return s_whole
    w = 0.5*(1.0 + np.tanh((dBIC - tau)/width))
    return (1.0 - w)*s_whole + w*s_left

def plot_fit_kbounds(ax, kc, Pdir, kmin, kmax, color, isPlot=True, linestyle='-', label_prefix='Fit'):
    m = (kc>=kmin)&(kc<=kmax)&np.isfinite(Pdir)&(Pdir>0)
    if m.sum()<3: return None
    x, y = np.log(kc[m]), np.log(Pdir[m])
    a, b = np.polyfit(x, y, 1)
    if isPlot and ax is not None:
        kk = np.logspace(np.log10(kmin), np.log10(kmax), 100)
        yy = np.exp(b)*kk**a
        ax.loglog(kk, yy, linestyle=linestyle, lw=3.5, color=color, label=f'slope = {a:.2f}', alpha=0.9)
    return b, a

def break_by_running_slope(kc, Pdir, kfit_bounds, win_pts=9, dev=0.20, min_pts=6):
    kmin, kmax = kfit_bounds
    m = (kc>=kmin)&(kc<=kmax)&np.isfinite(Pdir)&(Pdir>0)
    x, y = np.log(kc[m]), np.log(Pdir[m])
    n = x.size
    if n < max(win_pts, 2*min_pts)+1: return None
    a0, b0 = np.polyfit(x, y, 1)
    half = win_pts//2
    slopes = np.full(n, np.nan)
    for i in range(half, n-half):
        aa, bb = np.polyfit(x[i-half:i+half+1], y[i-half:i+half+1], 1)
        slopes[i] = aa
    cand = np.where(np.abs(slopes - a0) >= dev)[0]
    for i in cand:
        if i < max(min_pts, half) or (n - i - 1) < max(min_pts, half): 
            continue
        xl = x[:i+1]; yl = y[:i+1]
        xr = x[i:];   yr = y[i:]
        if xl.size < min_pts or xr.size < min_pts: 
            continue
        a1, b1 = np.polyfit(xl, yl, 1)
        a2, b2 = np.polyfit(xr, yr, 1)
        return np.exp(x[i]), (b1, a1), (b2, a2)
    return None

def main(lam, save_path=None, show_plots=True):
    Bx,By,Bz,ne = load_fields(h5_path)
    Pi = polarized_emissivity_simple(Bx,By,2.0)
    Bpar = Bz
    phi = faraday_density(ne,Bpar,C)

    P, sigma_RM = separated_P_map(Pi,phi,lam,los_axis,emit_frac,screen_frac)
    Q = P.real
    U = P.imag
    chi = 0.5*np.arctan2(U,Q)

    c2 = np.cos(2*chi)
    s2 = np.sin(2*chi)
    kc,Pk,S2D,kx,ky,edges = ring_average(c2, ring_bins, 3.0, None, True, False)
    kc2,Pk2,_,_,_,_ = ring_average(s2, ring_bins, 3.0, None, True, False)
    Pdir = Pk + Pk2

    # === estimate r_phi and r_i from YOUR maps, then overlay K=1/r ===
    P_emit_map, Phi_map = extract_emit_and_phi(Pi, phi, los_axis, emit_frac, screen_frac)
    r_i,  rc_i,  Ci  = radial_corr_length_unbiased(P_emit_map, bins=256, method="efold")
    r_phi, rc_ph, Cph = radial_corr_length_unbiased(Phi_map,      bins=256, method="efold")

    K_i   = (256.0/r_i)   if (np.isfinite(r_i)   and r_i > 0) else None
    K_phi = (256.0/r_phi) if (np.isfinite(r_phi) and r_phi > 0) else None

    c4 = np.cos(4*chi)
    s4 = np.sin(4*chi)
    Fc = np.fft.fft2(c4)
    Fs = np.fft.fft2(s4)
    corr = np.fft.ifft2(np.abs(Fc)**2 + np.abs(Fs)**2).real
    corr /= corr[0,0]
    r, Csum = ring_average_realspace(corr, ring_bins=64, r_min=0.5)
    S_padc = 0.5 - 0.5*Csum

    plt.figure(figsize=(12,5.5))
    ax1 = plt.subplot(1,2,1)
    im = ax1.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(c2)))**2 + np.abs(np.fft.fftshift(np.fft.fft2(s2)))**2 + 1e-30), origin="lower", cmap="plasma", vmin=-0.5, vmax=8.5)
    theta = np.linspace(0,2*np.pi,512)
    for r0 in edges:
        ax1.plot(r0*np.cos(theta), r0*np.sin(theta), color='w', alpha=0.2, lw=0.8)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.ax.tick_params(labelsize=16)
    ax1.set_title("Directional Spectrum 2D", fontsize=24, fontweight='bold', pad=15)
    ax1.set_xlabel("$k_x$", fontsize=22)
    ax1.set_ylabel("$k_y$", fontsize=22)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    ax2 = plt.subplot(1,2,2)
    ax2.loglog(kc, Pdir, '-', color='#2C3E50', ms=5, lw=2.5, label='Data', alpha=0.8)

    # Add vertical lines for correlation length break wavenumbers
    if K_phi is not None:
        ax2.axvline(K_phi, color="#9B59B6", lw=2.0, ls="--", alpha=0.9,
                    label=fr"$K_\phi=1/r_\phi$ = {K_phi:.3f}")
    if K_i is not None:
        ax2.axvline(K_i, color="#16A085", lw=2.0, ls="--", alpha=0.9,
                    label=fr"$K_i=1/r_i$ = {K_i:.3f}")

    if K_phi is not None:
        print(f"r_phi = {r_phi:.3f} px  ->  K_phi = {K_phi:.3f} px^-1")
    else:
        print(f"r_phi = {r_phi:.3f} px  ->  K_phi = None")
    if K_i is not None:
        print(f"r_i   = {r_i:.3f} px  ->  K_i   = {K_i:.3f} px^-1")
    else:
        print(f"r_i   = {r_i:.3f} px  ->  K_i   = None")

    alphas = []
    slopes  = []

    for i in np.linspace(0.01, 1.00, 1000):
        result = plot_fit(ax2, kc, Pdir, 0, i, "red", isPlot=False)
        alphas.append(i)
        if result is not None and isinstance(result, tuple) and len(result) == 2:
            intercept, slope = result
            slopes.append(slope)
        else:
            slopes.append(None)

    # Find slope closest to zero from the right (coarse search)
    best_coarse = None
    min_abs_slope = float('inf')
    for i, slope in reversed(list(zip(alphas, slopes))):
        if slope is not None:
            abs_slope = abs(slope)
            if abs_slope <= min_abs_slope:
                min_abs_slope = abs_slope
                best_coarse = i

    # Refined search around the candidate for higher accuracy
    best = best_coarse
    if best_coarse is not None:
        # Determine search range around candidate (adaptive based on coarse grid spacing)
        coarse_spacing = (alphas[-1] - alphas[0]) / (len(alphas) - 1) if len(alphas) > 1 else 0.01
        search_range = max(0.01, 2 * coarse_spacing)  # Search ±2 grid spacings around candidate
        i_min = max(0.01, best_coarse - search_range)
        i_max = min(1.00, best_coarse + search_range)
        
        # Fine search with 1000 points in the refined range
        fine_alphas = np.linspace(i_min, i_max, 1000)
        min_abs_slope_fine = float('inf')
        for i_fine in reversed(fine_alphas):
            result = plot_fit(ax2, kc, Pdir, 0, i_fine, "red", isPlot=False)
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                intercept_fine, slope_fine = result
                abs_slope_fine = abs(slope_fine)
                if abs_slope_fine <= min_abs_slope_fine:
                    min_abs_slope_fine = abs_slope_fine
                    best = i_fine

    # Plot slope vs i
    # fig, ax = plt.subplots()
    # ax.plot(alphas, slopes, lw=1)
    # ax.set_xlabel("i")
    # ax.set_ylabel("slope")
    # ax.set_title("Slope vs i")
    # ax.grid(True)
    # plt.show()

    chi = 2*lam**2*sigma_RM
    if chi < 4:
        best = 0.0

    plot_fit(ax2, kc, Pdir, 0, best, "#27AE60")
    cands, whole = _two_segment_candidates(kc, Pdir, min_pts=6, cont_eps=0.08)
    s_whole = whole[0] if whole is not None else np.nan
    sL, best = _select_break_stable(cands, s_whole, state=BREAK_STATE, tau_on=6.0, tau_off=3.0, lam_frac=3.0, lam_s=1.2, monotonic=True)
    if best is None:
        kk = np.logspace(np.log10(kc.min()), np.log10(kc.max()), 200)
        ax2.loglog(kk, np.exp(whole[1])*kk**s_whole, lw=3.5, color="#3498DB", label=f'slope = {s_whole:.2f}')
        main_slope = (whole[1], s_whole)
    else:
        _, _, kbr, (bL, aL), (bR, aR) = best
        kkL = np.logspace(np.log10(kc.min()), np.log10(kbr), 120)
        kkR = np.logspace(np.log10(kbr), np.log10(kc.max()), 120)
        ax2.loglog(kkL, np.exp(bL)*kkL**aL, lw=3.5, color="#E74C3C", label=f'slope = {aL:.2f}')
        ax2.loglog(kkR, np.exp(bR)*kkR**aR, lw=3.5, color="#3498DB", label=f'slope = {aR:.2f}')
        main_slope = (bL, aL)
    print(chi, '\t', main_slope[1])
    # plt.figure(figsize=(6,4))
    # plt.loglog(xv, yl, 'r-', label='left-end')
    # plt.loglog(xv, yr, 'b-', label='right-start')
    # plt.loglog([xs[j]], [ys[j]], 'ko', ms=6, label='intersection')
    # plt.xlabel("x1"); plt.ylabel("Y at junction k")
    # plt.xlim(0,1)
    # plt.legend(); plt.grid(True, which='both', alpha=0.3)
    # plt.tight_layout(); #plt.show()

    # plot_fit(ax2, kc, Pdir, 0.3, 1.0, "green")
    # plot_fit(ax2, kc, Pdir, 0.17, 1.0)
    # plot_fit(ax2, kc, Pdir, 0.6, 0.8)
    # # plot_fit(ax2, kc, Pdir, 0.6, 0.8)
    # plot_fit(ax2, kc, Pdir, 0.4, 1.0)

    ax2.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    ax2.set_xlabel("$k$", fontsize=22)
    ax2.set_ylabel("$P_{dir}(k)$", fontsize=22)
    ax2.set_title("Directional Power Spectrum", fontsize=24, fontweight='bold', pad=15)
    ax2.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.8)
    
    fig = plt.gcf()

    if chi < -0.2999 and chi > -0.001:
        fig.text(0.5, 1.0, f'Synchrotron-dominated $\chi = {chi:.2f}$', ha='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.8))
    elif chi > 0.2999 and chi < 3.0:
        fig.text(0.5, 1.0, f'Transitional $\chi = {chi:.2f}$', ha='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFD93D', alpha=0.8))
    else:
        fig.text(0.5, 1.0, f'Faraday-dominated $\chi = {chi:.2f}$', ha='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#6BCB77', alpha=0.8))

    plt.ylim(10**-8, 10**-2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig("directional.png", dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Optional: visualize the ACFs used to pick r (helps sanity-check that the 1/e crossing isn't window-limited)
    fig_acf, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(rc_ph, Cph, '-', lw=2); ax[0].axhline(np.exp(-1), ls='--', color='k', alpha=0.6)
    ax[0].axvline(r_phi, ls='--', color='#9B59B6'); ax[0].set_title(r"$C_\phi(r)$"); ax[0].set_xlabel("r [px]"); ax[0].set_ylabel("norm. ACF")
    ax[1].plot(rc_i,  Ci,  '-', lw=2); ax[1].axhline(np.exp(-1), ls='--', color='k', alpha=0.6)
    ax[1].axvline(r_i,   ls='--', color='#16A085'); ax[1].set_title(r"$C_{|P_{\rm emit}|}(r)$"); ax[1].set_xlabel("r [px]")
    fig_acf.tight_layout()
    if not show_plots:
        plt.close(fig_acf)

    plt.figure(figsize=(5.5,4.5))
    plt.plot(r, S_padc, '-', color="blue", ms=4)
    plt.xlabel("R (pixels)")
    plt.ylabel(r"$S(R)=\langle \sin^2[2(\chi(X)-\chi(X+R))]\rangle$")
    plt.title("PADC from one map")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not show_plots:
        plt.close()

    mean_val = np.mean(S_padc)
    std_val = np.std(S_padc)

    # Print in console
    # print(f"lambda={lam}, chi = {chi:.2f}, <S_padc> = {mean_val:.5f}; sigma_S_padc = {std_val:.5f}")
    
    return sigma_RM

if __name__ == "__main__":
    lam = 50.0  # 2.1
    main(lam)