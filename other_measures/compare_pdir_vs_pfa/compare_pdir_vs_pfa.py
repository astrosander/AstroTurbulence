#!/usr/bin/env python3
import numpy as np, h5py, matplotlib.pyplot as plt, os

def ensure3d(a):
    if a.ndim == 3: return a
    if a.ndim == 2: return a[:,None,:]
    return a[:,None,None]

def cumulative_to_observer(arr, axis=0):
    return np.flip(np.cumsum(np.flip(arr, axis=axis), axis=axis), axis=axis)

def polarization_map_mixed(ne, bz, dz, lam, psi, K=1.0, zaxis=0, B0_par=0.0, emissivity="density", gamma_emit=2.0):
    """
    Compute $P(\lambda) = \sum_z \epsilon_i e^{2i\psi_i} e^{2i\lambda^2\phi_i}$
    where $\phi_i = K \sum_{j=z}^{z_{\rm obs}} n_e B_z \Delta z$
    """
    if emissivity == "density":
        p_i = ne.astype(np.float64)  # $\epsilon \propto n_e$
    elif emissivity == "constant":
        p_i = np.ones_like(ne, dtype=np.float64)  # $\epsilon =$ const
    elif emissivity == "Bperp":
        bx, by = psi
        Bperp = np.sqrt(bx**2 + by**2)
        p_i = (Bperp**gamma_emit).astype(np.float64)  # $\epsilon \propto B_\perp^{\gamma}$
        psi = np.arctan2(by, bx)  # $\psi = \arctan(B_y/B_x)$
    else:
        raise ValueError("Unknown emissivity")

    if isinstance(psi, tuple):  # guard in case psi carried Bx,By
        psi = np.zeros_like(ne, dtype=np.float64)

    bz_tot = bz + B0_par
    phi_cells = K * ne * bz_tot * dz  # $\phi_i = K n_e B_z \Delta z$
    phi_to_obs = cumulative_to_observer(phi_cells, axis=zaxis)  # $\sum_{j=z}^{z_{\rm obs}} \phi_j$
    phase = np.exp(2j * (lam**2) * phi_to_obs)  # $e^{2i\lambda^2\phi}$
    P = np.sum(p_i * np.exp(2j * psi) * phase, axis=zaxis)  # $P = \sum_z \epsilon e^{2i\psi} e^{2i\lambda^2\phi}$
    return P

def directional_spectrum(P):
    """
    Compute $P_{\rm dir}(k)$ from $\cos(2\chi) = Q/|P|$, $\sin(2\chi) = U/|P|$
    where $P_{\rm dir}(k) = |\mathcal{F}[\cos(2\chi)]|^2 + |\mathcal{F}[\sin(2\chi)]|^2$
    """
    Q = np.real(P); U = np.imag(P)
    amp = np.sqrt(Q**2 + U**2) + 1e-30
    cos2 = Q/amp; sin2 = U/amp  # $\cos(2\chi)$, $\sin(2\chi)$
    F1 = np.fft.fftshift(np.fft.fft2(cos2))  # $\mathcal{F}[\cos(2\chi)]$
    F2 = np.fft.fftshift(np.fft.fft2(sin2))  # $\mathcal{F}[\sin(2\chi)]$
    P2 = (np.abs(F1)**2 + np.abs(F2)**2) / P.size  # $P_{\rm dir}(k_x,k_y)$
    ny, nx = P2.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny)); kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2).ravel(); p = P2.ravel()  # $k = \sqrt{k_x^2 + k_y^2}$
    idx = np.argsort(kr); kr = kr[idx]; p = p[idx]
    n_bins = max(24, int(np.sqrt(nx*ny)))
    edges = np.linspace(kr.min(), kr.max(), n_bins+1)
    centers = 0.5*(edges[1:]+edges[:-1])  # $k$ bin centers
    Pk = np.zeros_like(centers); cnt = np.zeros_like(centers)
    ind = np.digitize(kr, edges) - 1
    v = (ind>=0) & (ind<n_bins)
    for i,val in zip(ind[v], p[v]): Pk[i]+=val; cnt[i]+=1  # azimuthal average
    cnt[cnt==0]=1; Pk/=cnt
    return centers, Pk

def try_get(f, keys):
    for k in keys:
        if k in f: return np.array(f[k])
    raise KeyError(f"None of {keys} found in H5 file.")

def slope_guide(x, y_anchor_x, y_anchor_y, slope):
    """Reference line: $y = y_0 (x/x_0)^{\alpha}$"""
    return y_anchor_y * (x / y_anchor_x)**slope

# ======= user edit: path to your cube =======
H5_PATH = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\synthetic_kolmogorov_normal.h5"#"../mhd_fields.h5"
# ============================================

with h5py.File(H5_PATH, "r") as f:
    # Load $n_e$, $B_z$ (required), $B_x$, $B_y$ (optional)
    ne = try_get(f, ["gas_density", "ne", "electron_density"]).astype(np.float64)
    bz = try_get(f, ["k_mag_field_z", "k_mag_field", "bz", "j_mag_field", "k_mag_field_z"]).astype(np.float64)
    try:
        bx = try_get(f, ["k_mag_field_x", "bx", "i_mag_field"]).astype(np.float64)
        by = try_get(f, ["k_mag_field_y", "by", "j_mag_field"]).astype(np.float64)
        have_bperp = True  # $\epsilon \propto B_\perp^{\gamma}$ available
    except Exception:
        bx = by = None
        have_bperp = False  # fallback to $\epsilon \propto n_e$

ne = ensure3d(ne); bz = ensure3d(bz)
nz, ny, nx = ne.shape
dz = 1.0  # $\Delta z$
K = 1.0   # rotation measure constant
zaxis = 0
B0_par = 0.0  # mean field

# $\lambda$ grid: weak → strong rotation
lams = np.arange(0.1, 3.30, 0.01)# np.array([0.07, 0.10, 0.15, 0.22, 0.33, 0.50, 0.75, 1.10, 1.60, 2.30, 3.30], dtype=np.float64)

# emissivity: $\epsilon \propto n_e$ or $\epsilon \propto B_\perp^{\gamma}$
if have_bperp:
    emissivity = "Bperp"
    psi_input = (ensure3d(bx), ensure3d(by))
else:
    emissivity = "density"
    psi_input = np.zeros_like(ne, dtype=np.float64)

# compute $P(\lambda)$ and $\langle|P|^2\rangle$
PFA_vals = []
P_maps = []
for lam in lams:
    P = polarization_map_mixed(ne, bz, dz, lam, psi_input, K=K, zaxis=zaxis, B0_par=B0_par, emissivity=emissivity)
    P_maps.append(P)
    PFA_vals.append(np.mean(np.abs(P)**2))
PFA_vals = np.array(PFA_vals)

# $P_{\rm dir}(k)$ at three $\lambda$
idxs = [1, len(lams)//2, len(lams)-2]
PDIR_curves = []
for i in idxs:
    k, Pk = directional_spectrum(P_maps[i])
    PDIR_curves.append((lams[i], k, Pk))

# ---- figure ----
os.makedirs("fig_compare", exist_ok=True)
plt.figure(figsize=(11,4.2))

# Define popular custom colors for academic papers
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#F72585', '#4CC9F0']
line_styles = ['-', '--', '-.', ':']

# Left: PFA (⟨|P|^2⟩ vs λ)
ax1 = plt.subplot(1,2,1)
ax1.loglog(lams, PFA_vals, 'o-', color=colors[0], lw=2.0, ms=6, 
           label=r"$\langle|P(\lambda)|^2\rangle$")
# slope guides: λ^{-2} and λ^{-(2+m)} with m=2/3 (Kolmogorov)
x0 = lams[len(lams)//2]; y0 = np.interp(x0, lams, PFA_vals)
ax1.loglog(lams, slope_guide(lams, x0, y0, slope=-2.0), 
           color=colors[1], linestyle='--', lw=1.5, 
           label=r"$\propto \lambda^{-2}$")
ax1.loglog(lams, slope_guide(lams, x0, y0, slope=-(2.0+2.0/3.0)), 
           color=colors[2], linestyle=':', lw=1.8, 
           label=r"$\propto \lambda^{-(2+2/3)}$")
ax1.set_xlabel(r"$\lambda$", fontsize=12)
ax1.set_ylabel(r"$\langle|P|^2\rangle$", fontsize=12)
ax1.set_title(r"$\langle|P(\lambda)|^2\rangle$ vs $\lambda$", fontsize=12)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle=':')

# Right: directional spectrum P_dir(k) at three λ
ax2 = plt.subplot(1,2,2)
for i, (lam, k, Pk) in enumerate(PDIR_curves):
    m = (k>0) & (Pk>0)
    ax2.loglog(k[m], Pk[m], color=colors[i], lw=2.0, 
               label=fr"$\lambda={lam:.2f}$")

# Add -5/3 reference line for Kolmogorov turbulence
if len(PDIR_curves) > 0:
    # Use the middle curve for reference
    lam_ref, k_ref, Pk_ref = PDIR_curves[len(PDIR_curves)//2]
    m_ref = (k_ref > 0) & (Pk_ref > 0)
    if np.any(m_ref):
        k0 = k_ref[m_ref][len(k_ref[m_ref])//2]  # middle k point
        Pk0 = Pk_ref[m_ref][len(Pk_ref[m_ref])//2]  # corresponding Pk
        k_guide = np.logspace(np.log10(k_ref[m_ref].min()), np.log10(k_ref[m_ref].max()), 50)
        Pk_guide = slope_guide(k_guide, k0, Pk0, slope=-11/3)
        ax2.loglog(k_guide, Pk_guide, color=colors[3], linestyle='--', lw=1.5,
                   label=r"$\propto k^{-11/3}$")

ax2.set_xlabel(r"$k$", fontsize=12)
ax2.set_ylabel(r"$P_{\rm dir}(k)$", fontsize=12)
ax2.set_title(r"$P_{\rm dir}(k)$", 
              fontsize=12)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig("fig_compare/compare_pdir_vs_pfa.pdf", dpi=300, bbox_inches='tight')
plt.savefig("fig_compare/compare_pdir_vs_pfa.png", dpi=200, bbox_inches='tight')
print("Saved: fig_compare/compare_pdir_vs_pfa.[pdf|png]")
