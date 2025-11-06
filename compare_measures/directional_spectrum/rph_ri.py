import numpy as np
import h5py, matplotlib.pyplot as plt
from pathlib import Path

h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
lam = 0.0
los_axis = 2
C = 1.0
# emit_frac = (0.15, 1.00)
# screen_frac = (0.00, 0.10)

emit_frac   = (0.00, 1.00)   # thicker emitter -> larger r_i
screen_frac = (0.00, 0.03)   # thinner screen -> smaller r_phi

ring_bins = 96

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

def fit_segment(ax, kc, Pdir, kmin, kmax, color, label):
    m = (kc>=kmin)&(kc<=kmax)&np.isfinite(Pdir)&(Pdir>0)
    if m.sum()<6: return None
    x, y = np.log(kc[m]), np.log(Pdir[m])
    a, b = np.polyfit(x, y, 1)
    kk = np.logspace(np.log10(kmin), np.log10(kmax), 160)
    ax.loglog(kk, np.exp(b)*kk**a, lw=3.5, color=color, alpha=0.95, label=f"{label} {a:.2f}")
    return a

def plot_spectrum(lam, save_path=None, show_plots=True):
    """Plot directional spectrum for given lambda value."""
    Bx,By,Bz,ne = load_fields(h5_path)
    Pi = polarized_emissivity_simple(Bx,By,2.0)
    phi = faraday_density(ne,Bz,C)
    P, sigma_RM, P_emit_map, Phi_map = separated_P_map(Pi,phi,lam,los_axis,emit_frac,screen_frac)
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
    ax.loglog(kc, Pdir, '-', color='#2C3E50', lw=2.5, alpha=0.8, label='Data', ms=5)
    
    if Kphi_idx is not None: 
        ax.axvline(Kphi_idx, color="#9B59B6", lw=2.0, ls="--", alpha=0.9,
                   label=fr"$K_\phi=1/r_\phi$ = {Kphi_idx:.3f}")
    if Ki_idx is not None: 
        ax.axvline(Ki_idx, color="#16A085", lw=2.0, ls="--", alpha=0.9,
                   label=fr"$K_i=1/r_i$ = {Ki_idx:.3f}")

    kmin, kmax = kc.min(), kc.max()
    if (Kphi_idx is not None) and (Kphi_idx>kmin):
        sP = fit_segment(ax, kc, Pdir, kmin, Kphi_idx, "#7F8C8D", "plateau")
    if (Kphi_idx is not None) and (Ki_idx is not None) and (Ki_idx>Kphi_idx):
        sM = fit_segment(ax, kc, Pdir, Kphi_idx, Ki_idx, "#E67E22", "mid")
    if (Ki_idx is not None) and (Ki_idx<kmax):
        m = (kc>=Ki_idx)&(kc<=kmax)&np.isfinite(Pdir)&(Pdir>0)
        if m.sum()>=6:
            if lam==0.0:
                xh, yh = np.log(kc[m]), np.log(Pdir[m])
                s_fixed = -11.0/3.0
                b_fixed = np.mean(yh - s_fixed*xh)
                kk = np.logspace(np.log10(Ki_idx), np.log10(kmax), 160)
                ax.loglog(kk, np.exp(b_fixed)*kk**s_fixed, lw=3.5, color="#E74C3C", 
                         alpha=0.95, label=f"high {s_fixed:.2f}")
            else:
                sH = fit_segment(ax, kc, Pdir, Ki_idx, kmax, "#E74C3C", "high")

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
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return sigma_RM

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
    Pi = polarized_emissivity_simple(Bx, By, 2.0)
    phi = faraday_density(ne, Bz, C)
    _, sigma_RM, _, _ = separated_P_map(Pi, phi, 1.0, los_axis, emit_frac, screen_frac)
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
        print(f"\n✅ Animation frames saved to: {frames_dir}")
        print(f"   Total frames: {len(list(frames_dir.glob('*.png')))}")
    
    return frames_dir

if __name__ == "__main__":
    # For single plot
    # plot_spectrum(lam, show_plots=True)
    
    # Uncomment to generate animation:
    generate_chi_animation(chi_min=0.0, chi_max=8.0, n_frames=50, show_progress=True)
