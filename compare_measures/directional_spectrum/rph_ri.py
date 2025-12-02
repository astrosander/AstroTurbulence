import numpy as np
import h5py, matplotlib.pyplot as plt
from pathlib import Path
import csv
import os

h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
lam = 0.0
ring_bins = 96
REGIME = "r_phi_lt_ri"
los_axis = 2
C = 1.0
# emit_frac = (0.15, 1.00)
# screen_frac = (0.00, 0.10)

emit_frac   = (0.00, 0.46)
screen_frac = (0.69, 0.70)

if REGIME == "r_phi_lt_ri":
    screen_frac = (0.5, 0.504)
    emit_frac = (0.1, 0.6)
    gamma = 2.0
    auto_los = True
    los_perpendicular = True
    
elif REGIME == "r_phi_gt_ri":
    screen_frac = (0.00, 0.15)
    emit_frac = (0.15, 0.30)
    gamma = 2.5
    auto_los = True
    los_perpendicular = False
    
elif REGIME == "auto_los":
    auto_los = True
    los_perpendicular = True
    
else:
    auto_los = False
    los_perpendicular = True


emit_frac   = (0.00, 0.46)   # thicker emitter -> larger r_i
screen_frac = (0.69, 0.70)   # thinner screen -> smaller r_phi


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

def load_fields(p):
    with h5py.File(p,"r") as f:
        Bx = f["i_mag_field"][()]
        By = f["j_mag_field"][()]
        Bz = f["k_mag_field"][()]
        ne = f["gas_density"][()]
    return Bx,By,Bz,ne

def auto_select_los_axis(Bx, By, Bz, perpendicular=True):
    Bmean = np.array([Bx.mean(), By.mean(), Bz.mean()])
    Bmean_abs = np.abs(Bmean)
    
    if perpendicular:
        los_axis = int(np.argmin(Bmean_abs))
    else:
        los_axis = int(np.argmax(Bmean_abs))
    
    return los_axis

def get_field_components_for_los(Bx, By, Bz, los_axis):
    if los_axis == 0:
        B_perp1, B_perp2, B_parallel = By, Bz, Bx
    elif los_axis == 1:
        B_perp1, B_perp2, B_parallel = Bx, Bz, By
    else:
        B_perp1, B_perp2, B_parallel = Bx, By, Bz
    
    return B_perp1, B_perp2, B_parallel

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
    return P_emit*np.exp(2j*(lam**2)*Phi), sigma_RM, P_emit, Phi

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

def radial_corr_length_unbiased(field2d, bins=256, method="efold"):
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
    C = C[cy-(ny-1):cy+ny, cx-(nx-1):cx+nx]
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

def theory_Pdir_piecewise(k_ref, lam, sigma_RM,
                          r_i, r_phi, Lz,
                          m_i=2.0/3.0, m_phi=2.0/3.0,
                          mtilde_phi=1.0,  # effective index in LP16 Eq. 154
                          K_i=None, K_phi=None,
                          norm_pivot=None, P_pivot=None):
    """
    Return theoretical P_dir(k, lambda) based on r_dir(λ) decorrelation scale.
    
    Uses the λ-dependent decorrelation radius r_dir(λ) (analogous to LP16's R_s)
    to determine spectral slopes:
    - For k << min(1/r_i, k_dir(λ)): flat (slope ≈ 0)
    - For 1/r_i << k << k_dir(λ): mixed regime (slope ≈ -11/3)
    - For k >> k_dir(λ): Faraday-randomized (slope ≈ -8/3)

    Parameters
    ----------
    k_ref : 1D array
        k values (your 'kc' from ring_average).
    lam : float
        Wavelength (same units you use elsewhere).
    sigma_RM : float
        RMS of the Faraday depth Phi_map (same as in your code).
    r_i, r_phi : float
        Correlation lengths of emission and Faraday depth in real-space pixels.
    Lz : float
        Thickness of the Faraday slab along the LOS, in the same units as r_i, r_phi
        (for a cube with unit cells, Lz ~ Nz along LOS).
    m_i, m_phi : float
        Structure-function exponents for emissivity and RM.
    mtilde_phi : float
        Effective exponent as in LP16 Eq. (154); for Kolmogorov RM you can
        take ~1.
    K_i, K_phi : float, optional
        Positions of the spectral breaks; if None, defaults to 1/r_i, 1/r_phi.
    norm_pivot : float, optional
        k at which to force P_pred(k) = P_pivot. If None, no renormalization.
    P_pivot : float, optional
        Target value of P_pred(norm_pivot).

    Returns
    -------
    P_pred : 1D array
        Theoretical directional spectrum at k_ref.
    slopes : dict
        Dictionary with sP, sM, sH, chi, r_dir, k_dir.
    """

    # 1. Chi parameter
    chi = 2.0 * sigma_RM * (lam ** 2)

    # 2. Asymptotic slopes
    s_mix = -(m_i + m_phi + 2.0)  # -11/3 for Kolmogorov
    s_em  = -(m_i + 2.0)           # -8/3 for Kolmogorov
    s_flat = 0.0

    # 3. Calculate r_dir(λ) using LP16 R_s formula
    # r_dir(λ) ≈ r_φ (L_σ_φ / √(L r_φ))^(2/(1+m̃_φ))
    # where L_σ_φ = (√2 λ² σ_φ)⁻¹ = (√2 / χ)
    if lam > 0 and sigma_RM > 0 and Lz > 0 and r_phi > 0:
        L_sigma_phi = 1.0 / (np.sqrt(2.0) * (lam ** 2) * sigma_RM)  # = √2 / χ
        sqrt_L_rphi = np.sqrt(Lz * r_phi)
        
        if L_sigma_phi < sqrt_L_rphi:  # Thick-screen, strong-rotation regime
            ratio = L_sigma_phi / sqrt_L_rphi
            exponent = 2.0 / (1.0 + mtilde_phi)
            r_dir = r_phi * (ratio ** exponent)
        else:
            # For thin-screen or weak-rotation, use simpler form
            # r_dir(χ) ≈ r_φ [4/(C_D χ²)]^(1/m_φ)
            # Using order-unity constant C_D ≈ 2κ ≈ 1
            C_D = 1.0  # Order-unity constant
            if chi > 0:
                r_dir = r_phi * (4.0 / (C_D * chi ** 2)) ** (1.0 / m_phi)
            else:
                r_dir = np.inf  # No Faraday decorrelation at λ=0
    else:
        r_dir = np.inf  # No Faraday decorrelation

    # 4. Convert to k-space: k_dir(λ) = 1/r_dir(λ)
    if np.isfinite(r_dir) and r_dir > 0:
        k_dir = 1.0 / r_dir
    else:
        k_dir = 0.0  # Infinite r_dir means k_dir = 0 (no decorrelation)

    # 5. Break positions in k-space
    if K_i is None:
        K_i = 1.0 / r_i if r_i > 0 else None
    if K_phi is None:
        K_phi = 1.0 / r_phi if r_phi > 0 else None

    # 6. Build piecewise spectrum based on k_dir(λ)
    # Logic from theory:
    # - For k << 1/r_i AND k << k_dir(λ): flat (slope = 0)
    # - For 1/r_i << k << k_dir(λ): mixed regime (slope = -11/3)
    # - For k >> k_dir(λ): Faraday-randomized (slope = -8/3)
    
    P_pred = np.zeros_like(k_ref, dtype=float)
    
    # Normalize at a reference point (use K_i if available, otherwise first k)
    if K_i is not None:
        k_norm = K_i
    elif k_dir > 0:
        k_norm = k_dir
    else:
        k_norm = k_ref[0] if len(k_ref) > 0 else 1.0
    P_norm = 1.0
    
    # Build spectrum with proper slopes in each regime
    for idx, k in enumerate(k_ref):
        # Determine slope based on position relative to K_i and k_dir
        if K_i is not None and k <= K_i:
            # Low-k range: k < K_i
            # Flat only if also well below k_dir
            if k_dir > 0:
                # Use flat if k << k_dir, otherwise transition
                if k < 0.1 * k_dir:  # Well below k_dir
                    slope = s_flat
                elif k < k_dir:  # Approaching k_dir
                    # Smooth transition: interpolate between flat and mixed
                    frac = (k - 0.1 * k_dir) / (0.9 * k_dir)
                    slope = s_flat + frac * (s_mix - s_flat)
                else:  # k > k_dir (shouldn't happen if k <= K_i, but handle it)
                    slope = s_em
            else:
                # No k_dir (λ=0 case): use flat
                slope = s_flat
        elif k_dir > 0 and k <= k_dir:
            # Mid-k range: K_i < k < k_dir (mixed regime)
            slope = s_mix
        else:
            # High-k range: k > k_dir (Faraday-randomized)
            slope = s_em
        
        # Calculate P(k) ensuring continuity
        if idx == 0:
            P_pred[idx] = P_norm * (k / k_norm) ** slope
        else:
            # Continue from previous point to ensure continuity
            P_pred[idx] = P_pred[idx-1] * (k / k_ref[idx-1]) ** slope

    # 8. Optional renormalization to match measured spectrum at a pivot k
    if (norm_pivot is not None) and (P_pivot is not None):
        # Find nearest k_ref to the pivot
        j = np.argmin(np.abs(k_ref - norm_pivot))
        if P_pred[j] > 0:
            P_pred *= (P_pivot / P_pred[j])

    # 9. Calculate effective slopes for reporting
    # These represent the slopes in the three traditional ranges (k<K_i, K_i<k<K_phi, k>K_phi)
    # but adjusted based on where k_dir(λ) falls
    if k_dir > 0:
        if K_i is not None and k_dir > K_i:
            # k_dir is above K_i: low-k can be flat, mid-k is mixed, high-k is emission
            sP_eff = s_flat
            sM_eff = s_mix
            sH_eff = s_em
        else:
            # k_dir is below or at K_i: all ranges above k_dir use emission slope
            sP_eff = s_flat if (K_i is not None) else s_em
            sM_eff = s_em
            sH_eff = s_em
    else:
        # No Faraday decorrelation (λ=0): all slopes are mixed
        sP_eff = s_flat if (K_i is not None) else s_mix
        sM_eff = s_mix
        sH_eff = s_mix

    slopes = {
        "sP": sP_eff,
        "sM": sM_eff,
        "sH": sH_eff,
        "chi": chi,
        "r_dir": r_dir,
        "k_dir": k_dir
    }
    print(P_pred, slopes)
    return P_pred, slopes

def _measure_rphi_ri(Bx, By, Bz, ne, los_axis, emit_frac, screen_frac, gamma=2.0):
    B_perp1, B_perp2, B_parallel = get_field_components_for_los(Bx, By, Bz, los_axis)
    
    Pi = polarized_emissivity_simple(B_perp1, B_perp2, gamma)
    phi = faraday_density(ne, B_parallel, C)
    _, _, P_emit_map, Phi_map = separated_P_map(Pi, phi, 0.0, los_axis, emit_frac, screen_frac)
    r_i, _, _ = radial_corr_length_unbiased(P_emit_map, bins=256, method="efold")
    r_phi, _, _ = radial_corr_length_unbiased(Phi_map, bins=256, method="efold")
    print("r_i=", r_i)
    print("r_phi=", r_phi)
    
    return r_phi, r_i

def find_params_swap(target="rphi<ri", h5_path_override=None):
    path = h5_path_override if h5_path_override is not None else h5_path
    Bx, By, Bz, ne = load_fields(path)
    
    los_choices = [0, 1, 2]
    screen_th = [0.01, 0.02, 0.03, 0.05, 0.10]
    emit_th = [0.25, 0.50, 0.75, 1.00]
    gamma_choices = [2.0, 2.4, 3.0]
    
    best = None
    print(f"Searching for parameters to achieve: {target}")
    print(f"Testing {len(los_choices)} LOS axes × {len(screen_th)} screen thicknesses × "
          f"{len(emit_th)} emitter thicknesses × {len(gamma_choices)} gamma values = "
          f"{len(los_choices) * len(screen_th) * len(emit_th) * len(gamma_choices)} combinations...")
    
    for la in los_choices:
        for s in screen_th:
            screen = (0.00, s)
            for e in emit_th:
                emit = (0.00, e)
                for g in gamma_choices:
                    try:

                        print("LOLOL")
                        rphi, ri = _measure_rphi_ri(Bx, By, Bz, ne, la, emit, screen, gamma=g)
                        if not (np.isfinite(rphi) and np.isfinite(ri) and rphi > 0 and ri > 0):
                            continue
                        ratio = rphi / ri
                        if target == "rphi<ri":
                            cond, key = (ratio < 1.0), ratio
                        else:
                            cond, key = (ratio > 1.0), -ratio
                        if cond:
                            cand = (key, la, screen, emit, g, rphi, ri)
                            if (best is None) or (cand[0] < best[0]):
                                best = cand
                    except Exception:
                        continue
    
    if best is None:
        print("  No combination found matching target condition. Searching for best approximation...")
        alt = None
        for la in los_choices:
            for s in screen_th:
                screen = (0.00, s)
                for e in emit_th:
                    emit = (0.00, e)
                    for g in gamma_choices:
                        try:
                            print("LOLOL")
                            rphi, ri = _measure_rphi_ri(Bx, By, Bz, ne, la, emit, screen, gamma=g)
                            if not (np.isfinite(rphi) and np.isfinite(ri) and rphi > 0 and ri > 0):
                                continue
                            ratio = rphi / ri
                            key = ratio if target == "rphi<ri" else -ratio
                            cand = (key, la, screen, emit, g, rphi, ri)
                            if (alt is None) or (cand[0] < alt[0]):
                                alt = cand
                        except Exception:
                            continue
        best = alt
    
    if best is None:
        raise ValueError("Could not find any valid parameter combination. Check data file.")
    
    _, la, screen, emit, g, rphi, ri = best
    result = {
        "los_axis": la,
        "screen_frac": screen,
        "emit_frac": emit,
        "gamma": g,
        "r_phi": rphi,
        "r_i": ri,
        "ratio": rphi / ri
    }
    
    print(f"  Best found: r_phi={rphi:.3f}, r_i={ri:.3f}, ratio={result['ratio']:.3f}")
    print(f"    los_axis={la}, screen_frac={screen}, emit_frac={emit}, gamma={g}")
    
    return result

def fit_segment(ax, kc, Pdir, kmin, kmax, color, label):
    # Use small tolerance to handle floating point errors at boundaries
    # Use relative tolerance based on the range
    eps = max(np.finfo(kc.dtype).eps * max(abs(kmin), abs(kmax)), 1e-10)
    # Include points on the border (accounting for floating point errors)
    m = (kc >= kmin - eps)&(kc <= kmax + eps)&np.isfinite(Pdir)&(Pdir>0)
    if m.sum() == 0: return None
    # Use only the actual range where points exist
    k_actual = kc[m]
    k_actual_min = k_actual.min()
    k_actual_max = k_actual.max()
    # Least squares fit using only points in the range
    x, y = np.log(k_actual), np.log(Pdir[m])
    a, b = np.polyfit(x, y, 1)
    # Plot only over the range where points exist
    kk = np.logspace(np.log10(k_actual_min), np.log10(k_actual_max), 160)
    ax.loglog(kk, np.exp(b)*kk**a, lw=3.5, color=color, alpha=0.95, label=f"{label} {a:.2f}")
    return a

def save_slopes_to_csv(lam, sigma_RM, sP, sM, sH, csv_path=None):
    """Save slope values to CSV file, appending for different chi values.
    
    chi = 2 * sigma_RM * lambda^2
    """
    if csv_path is None:
        script_dir = Path(__file__).parent
        csv_path = script_dir / "slopes_vs_lambda.csv"
    
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()
    
    # Calculate chi = 2 * sigma_RM * lambda^2
    chi = 2.0 * sigma_RM * (lam ** 2)
    
    # Write or append to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['chi', 'slope_k_lt_K_i', 'slope_K_i_lt_k_lt_K_phi', 'slope_k_gt_K_phi'])
        # Write data row
        writer.writerow([
            chi,
            sP if sP is not None else '',
            sM if sM is not None else '',
            sH if sH is not None else ''
        ])

def plot_spectrum(lam, save_path=None, show_plots=True, 
                  los_axis_override=None, emit_frac_override=None, 
                  screen_frac_override=None, gamma_override=None, csv_path=None):
    use_los_axis = los_axis_override if los_axis_override is not None else los_axis
    use_emit_frac = emit_frac_override if emit_frac_override is not None else emit_frac
    use_screen_frac = screen_frac_override if screen_frac_override is not None else screen_frac
    use_gamma = gamma_override if gamma_override is not None else gamma
    
    Bx,By,Bz,ne = load_fields(h5_path)
    
    if 'auto_los' in globals() and auto_los:
        perp = los_perpendicular if 'los_perpendicular' in globals() else True
        use_los_axis = auto_select_los_axis(Bx, By, Bz, perpendicular=perp)
    
    B_perp1, B_perp2, B_parallel = get_field_components_for_los(Bx, By, Bz, use_los_axis)
    
    Pi = polarized_emissivity_simple(B_perp1, B_perp2, use_gamma)
    phi = faraday_density(ne, B_parallel, C)
    P, sigma_RM, P_emit_map, Phi_map = separated_P_map(
        Pi, phi, lam, use_los_axis, use_emit_frac, use_screen_frac)
    Q, U = P.real, P.imag
    chi_angle = 0.5*np.arctan2(U,Q)
    c2, s2 = np.cos(2*chi_angle), np.sin(2*chi_angle)
    kc,Pk,_,_,_,_ = ring_average(c2, ring_bins, 3.0, None, True, False)
    _,Pk2,_,_,_,_ = ring_average(s2, ring_bins, 3.0, None, True, False)
    Pdir = Pk + Pk2

    r_i,  _, _ = radial_corr_length_unbiased(P_emit_map, bins=256, method="efold")
    r_phi, _, _ = radial_corr_length_unbiased(Phi_map,      bins=256, method="efold")
    Ny, Nx = P_emit_map.shape
    Kphi_idx = (1.0/r_phi)*Nx if (np.isfinite(r_phi) and r_phi>0) else None
    Ki_idx   = (1.0/r_i)*Nx   if (np.isfinite(r_i)   and r_i>0)   else None

    fig = plt.figure(figsize=(12, 5.5))
    ax = plt.subplot(1,1,1)
    
    # Modern color scheme matching previous simulation
    ax.loglog(kc, Pdir, 'o-', color='#2C3E50', lw=2.5, alpha=0.8, label='Data', ms=5)
    
    if Kphi_idx is not None: 
        ax.axvline(Kphi_idx, color="#9B59B6", lw=2.0, ls="--", alpha=0.9,
                   label=fr"$K_\phi=1/r_\phi$ = {Kphi_idx:.3f}")
    if Ki_idx is not None: 
        ax.axvline(Ki_idx, color="#16A085", lw=2.0, ls="--", alpha=0.9,
                   label=fr"$K_i=1/r_i$ = {Ki_idx:.3f}")

    kmin, kmax = kc.min(), kc.max()
    
    # Calculate Lz (screen thickness in grid units along LOS)
    Nz = Bx.shape[use_los_axis]
    Lz = (use_screen_frac[1] - use_screen_frac[0]) * Nz
    
    # Create smooth k_ref for plotting
    k_ref = np.logspace(np.log10(kmin), np.log10(kmax), 200)
    
    # Compute λ-dependent theoretical prediction based on R_s picture
    if np.isfinite(r_i) and r_i > 0 and np.isfinite(r_phi) and r_phi > 0 and Lz > 0:
        # Prepare parameters for theory function
        # K_i and K_phi are in same units as kc (wavenumber space)
        # Pass them if available, otherwise function will calculate from r_i, r_phi
        theory_kwargs = {
            'k_ref': k_ref,
            'lam': lam,
            'sigma_RM': sigma_RM,
            'r_i': r_i,
            'r_phi': r_phi,
            'Lz': Lz,
            'm_i': 2.0/3.0,
            'm_phi': 2.0/3.0,
            'mtilde_phi': 1.0,
        }
        
        # Add K_i and K_phi if available (they're in same units as k_ref)
        if Ki_idx is not None:
            theory_kwargs['K_i'] = Ki_idx
        if Kphi_idx is not None:
            theory_kwargs['K_phi'] = Kphi_idx
        
        # Normalize to match data at mid-range k
        mid_idx = len(kc) // 2
        if mid_idx < len(kc) and mid_idx < len(Pdir):
            if np.isfinite(Pdir[mid_idx]) and Pdir[mid_idx] > 0:
                theory_kwargs['norm_pivot'] = kc[mid_idx]
                theory_kwargs['P_pivot'] = Pdir[mid_idx]
        
        P_pred_piecewise, slopes_th = theory_Pdir_piecewise(**theory_kwargs)

        # Plot theoretical prediction
        ax.loglog(k_ref, P_pred_piecewise, color='blue', lw=2.0, alpha=0.7, 
                  label=rf'Theory ($r_{{\rm dir}}(\lambda)$), $\chi={slopes_th["chi"]:.2f}$')
        
        # Add vertical line for k_dir(λ) if it's in the plot range
        k_dir = slopes_th.get("k_dir", 0.0)
        if k_dir > 0 and k_dir >= kmin and k_dir <= kmax:
            ax.axvline(k_dir, color="#F39C12", lw=2.0, ls=":", alpha=0.9,
                      label=fr"$k_{{\rm dir}}(\lambda)=1/r_{{\rm dir}}$ = {k_dir:.3f}")
    else:
        # Fallback: simple prediction if parameters are invalid
        mid_idx = len(Pdir) // 2
        if mid_idx < len(Pdir) and np.isfinite(Pdir[mid_idx]) and Pdir[mid_idx] > 0:
            P_ref = Pdir[mid_idx]
            k_ref_mid = kc[mid_idx]
        else:
            valid_mask = np.isfinite(Pdir) & (Pdir > 0)
            P_ref = np.median(Pdir[valid_mask]) if valid_mask.sum() > 0 else 1.0
            k_ref_mid = np.median(kc[valid_mask]) if valid_mask.sum() > 0 else kmin
        P_pred_piecewise = P_ref * (k_ref / k_ref_mid)**(-11.0/3.0)
        ax.loglog(k_ref, P_pred_piecewise, color='blue', lw=2.0, alpha=0.7, 
                  label=r'Theory (fallback)')
    
    # Three separate, non-overlapping slope measurements:
    sP = None
    sM = None
    sH = None
    
    # 1. k < K_i (low range) - fit to the same binned data that's plotted
    if Ki_idx is not None:
        # Clamp Ki_idx to valid range, but use original for comparison
        k_i_upper = min(max(Ki_idx, kmin), kmax)  # Clamp between kmin and kmax
        if k_i_upper > kmin and Ki_idx > kmin:  # Only if Ki_idx is above kmin
            sP = fit_segment(ax, kc, Pdir, kmin, k_i_upper, "#7F8C8D", "$k<K_i$")
    
    # 2. K_i < k < K_phi (mid range) - fit to the same binned data that's plotted
    if (Ki_idx is not None) and (Kphi_idx is not None) and (Kphi_idx > Ki_idx):
        # Clamp both to valid range
        k_i_lower = max(Ki_idx, kmin)
        k_phi_upper = min(Kphi_idx, kmax)
        if k_phi_upper > k_i_lower:
            sM = fit_segment(ax, kc, Pdir, k_i_lower, k_phi_upper, "#E67E22", "$K_i<k<K_\phi$")
    
    # 3. k > K_phi (high range)
    if Kphi_idx is not None:
        # Clamp Kphi_idx to valid range
        k_phi_lower = max(Kphi_idx, kmin)
        if k_phi_lower < kmax:  # Only if Kphi_idx is below kmax
            if lam==0.0:
                # Use small tolerance to handle floating point errors at boundaries
                eps = max(np.finfo(kc.dtype).eps * max(abs(k_phi_lower), abs(kmax)), 1e-10)
                # Include points on the border (accounting for floating point errors)
                m = (kc >= k_phi_lower - eps)&(kc <= kmax + eps)&np.isfinite(Pdir)&(Pdir>0)
                if m.sum() > 0:
                    k_actual = kc[m]
                    k_actual_min = k_actual.min()
                    k_actual_max = k_actual.max()
                    # Least squares fit using only points in the range
                    xh, yh = np.log(k_actual), np.log(Pdir[m])
                    s_fixed = -11.0/3.0
                    b_fixed = np.mean(yh - s_fixed*xh)
                    # Plot only over the range where points exist
                    kk = np.logspace(np.log10(k_actual_min), np.log10(k_actual_max), 160)
                    ax.loglog(kk, np.exp(b_fixed)*kk**s_fixed, lw=3.5, color="#E74C3C", 
                             alpha=0.95, label=f"$k>K_\phi$ {s_fixed:.2f}")
                    sH = s_fixed
            else:
                sH = fit_segment(ax, kc, Pdir, k_phi_lower, kmax, "#E74C3C", "$k>K_\phi$")

    chi = 2*(lam**2)*sigma_RM
    
    # Add regime indicator with modern colors (matching previous simulation)
    if chi < 1.0:
        regime = "Synchrotron-dominated"
        regime_color = '#FF6B6B'  # Red
    elif chi < 3.0:
        regime = "Transitional"
        regime_color = '#FFD93D'  # Yellow/Gold
    else:
        regime = "Faraday-dominated"
        regime_color = '#6BCB77'  # Green
    
    fig.text(0.5, 1.0, f'{regime} $\\chi = {chi:.3f}$', ha='center', 
             fontsize=20, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=regime_color, alpha=0.8))
    
    ax.set_title(rf"$r_\phi={r_phi:.2f}$,  $r_i={r_i:.2f}$", fontsize=24, fontweight='bold', pad=15)
    ax.set_xlabel("$k$", fontsize=22)
    ax.set_ylabel("$P_{dir}(k)$", fontsize=22)
    ax.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save slopes to CSV (using chi = 2*sigma_RM*lambda^2)
    save_slopes_to_csv(lam, sigma_RM, sP, sM, sH, csv_path)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return sigma_RM, sP, sM, sH

def generate_chi_animation(chi_min=0.0, chi_max=5.0, n_frames=50, 
                          frames_dir=None, show_progress=True):
    """Generate animation frames for varying chi values."""
    if frames_dir is None:
        script_dir = Path(__file__).parent
        frames_dir = script_dir / "rph_ri_animation_frames1"
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    if show_progress:
        print(f"Frames will be saved to: {frames_dir}")
        print(f"Generating {n_frames} frames for chi from {chi_min:.2f} to {chi_max:.2f}")
    
    # Compute sigma_RM once
    if show_progress:
        print("\nComputing sigma_RM from data...")
    Bx, By, Bz, ne = load_fields(h5_path)
    
    use_los_axis = los_axis
    if 'auto_los' in globals() and auto_los:
        perp = los_perpendicular if 'los_perpendicular' in globals() else True
        use_los_axis = auto_select_los_axis(Bx, By, Bz, perpendicular=perp)
        orientation = "perpendicular" if perp else "parallel"
        if show_progress:
            print(f"Auto-selected LOS axis: {use_los_axis} ({orientation} to mean B)")
    
    B_perp1, B_perp2, B_parallel = get_field_components_for_los(Bx, By, Bz, use_los_axis)
    
    Pi = polarized_emissivity_simple(B_perp1, B_perp2, gamma)
    phi = faraday_density(ne, B_parallel, C)
    _, sigma_RM, _, _ = separated_P_map(Pi, phi, 1.0, use_los_axis, emit_frac, screen_frac)
    if show_progress:
        print(f"sigma_RM = {sigma_RM:.6f}")
    
    chi_values = np.linspace(chi_min, chi_max, n_frames)
    
    if show_progress:
        print(f"\nGenerating frames...")
    
    for i, chi_target in enumerate(chi_values):
        if chi_target <= 0: 
            lam = 0.0
        else:
            lam = np.sqrt(chi_target / (2.0 * sigma_RM))
        
        frame_filename = frames_dir / f"{i:04d}.png"
        
        if show_progress and (i % 10 == 0 or i == 0 or i == len(chi_values) - 1):
            progress_pct = 100 * (i + 1) / n_frames
            print(f"  Frame {i+1}/{n_frames} ({progress_pct:.1f}%): chi={chi_target:.3f}, lam={lam:.6f}")
        
        try:
            plot_spectrum(lam, save_path=str(frame_filename), show_plots=False)
        except Exception as e:
            if show_progress:
                print(f"    Error: {e}")
            continue
    
    if show_progress:
        print(f"\n Animation frames saved to: {frames_dir}")
        print(f"   Total frames: {len(list(frames_dir.glob('*.png')))}")
    
    return frames_dir

if __name__ == "__main__":
    print("=" * 70)
    print(f"Configuration: REGIME = '{REGIME}'")
    print(f"  emit_frac = {emit_frac}")
    print(f"  screen_frac = {screen_frac}")
    print(f"  gamma = {gamma}")
    if 'auto_los' in globals() and auto_los:
        orientation = "perpendicular" if ('los_perpendicular' in globals() and los_perpendicular) else "parallel"
        print(f"  LOS axis: AUTO (will select {orientation} to mean B)")
    else:
        print(f"  LOS axis: {los_axis} (manual)")
    print("=" * 70)
    print()
    
    generate_chi_animation(chi_min=0.0, chi_max=5.0, n_frames=50, show_progress=True)
