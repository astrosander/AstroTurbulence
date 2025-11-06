import numpy as np
import h5py, matplotlib.pyplot as plt

h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
los_axis = 2
C = 1.0
lam = 2.1
emit_frac = (0.15, 1.00)
screen_frac = (0.00, 0.10)
ring_bins = 48*2
kfit_bounds = (4,25)

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
    return P_emit*np.exp(2j*(lam**2)*Phi)

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

Bx,By,Bz,ne = load_fields(h5_path)
Pi = polarized_emissivity_simple(Bx,By,2.0)
Bpar = Bz
phi = faraday_density(ne,Bpar,C)

P = separated_P_map(Pi,phi,lam,los_axis,emit_frac,screen_frac)
Q = P.real
U = P.imag
chi = 0.5*np.arctan2(U,Q)

c2 = np.cos(2*chi)
s2 = np.sin(2*chi)
kc,Pk,S2D,kx,ky,edges = ring_average(c2, ring_bins, 3.0, None, True, False)
kc2,Pk2,_,_,_,_ = ring_average(s2, ring_bins, 3.0, None, True, False)
Pdir = Pk + Pk2

c4 = np.cos(4*chi)
s4 = np.sin(4*chi)
Fc = np.fft.fft2(c4)
Fs = np.fft.fft2(s4)
corr = np.fft.ifft2(np.abs(Fc)**2 + np.abs(Fs)**2).real
corr /= corr[0,0]
r, Csum = ring_average_realspace(corr, ring_bins=64, r_min=0.5)
S_padc = 0.5 - 0.5*Csum

plt.figure(figsize=(10,4.8))
ax1 = plt.subplot(1,2,1)
im = ax1.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(c2)))**2 + np.abs(np.fft.fftshift(np.fft.fft2(s2)))**2 + 1e-30), origin="lower", cmap="magma")
theta = np.linspace(0,2*np.pi,512)
for r0 in edges:
    ax1.plot(r0*np.cos(theta), r0*np.sin(theta), color='w', alpha=0.15, lw=0.7)
plt.colorbar(im, ax=ax1)
ax1.set_title("Directional spectrum 2D")
ax1.set_xlabel("$k_x$")
ax1.set_ylabel("$k_y$")
plt.xlim(left=0)
plt.ylim(bottom=0)

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
            ax.loglog(k_fit_line, P_fit_line, linestyle=linestyle, lw=2, color=color,
                       label=f'{label_prefix} [{frac_min:.1f}-{frac_max:.1f}] (slope = {slope:.2f})')
        return intercept, slope
    return None

ax2 = plt.subplot(1,2,2)
ax2.loglog(kc, Pdir, '-', color='black', ms=4, lw=1.5, label='Data')


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

print("best:", best)

# Plot slope vs i
# fig, ax = plt.subplots()
# ax.plot(alphas, slopes, lw=1)
# ax.set_xlabel("i")
# ax.set_ylabel("slope")
# ax.set_title("Slope vs i")
# ax.grid(True)
# plt.show()


x1=0.3
plot_fit(ax2, kc, Pdir, 0, best, "green")
plot_fit(ax2, kc, Pdir, best, x1, "red")
plot_fit(ax2, kc, Pdir, x1, 1.0, "blue")
# plot_fit(ax2, kc, Pdir, 0.3, 1.0, "green")
# plot_fit(ax2, kc, Pdir, 0.17, 1.0)
# plot_fit(ax2, kc, Pdir, 0.6, 0.8)
# # plot_fit(ax2, kc, Pdir, 0.6, 0.8)
# plot_fit(ax2, kc, Pdir, 0.4, 1.0)


ax2.legend()

ax2.set_xlabel("$k$")
ax2.set_ylabel("$P_{dir}(k)$")
ax2.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("directional.png", dpi=300)
plt.show()

plt.figure(figsize=(5.5,4.5))
plt.plot(r, S_padc, '-', color="blue", ms=4)
plt.xlabel("R (pixels)")
plt.ylabel(r"$S(R)=\langle \sin^2[2(\chi(X)-\chi(X+R))]\rangle$")
plt.title("PADC from one map")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()