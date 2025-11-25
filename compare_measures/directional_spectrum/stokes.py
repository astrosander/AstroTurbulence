import numpy as np
import h5py
import matplotlib.pyplot as plt

h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"

def load_fields(p):
    with h5py.File(p,"r") as f:
        Bx = f["i_mag_field"][()].astype(np.float64)
        By = f["j_mag_field"][()].astype(np.float64)
        Bz = f["k_mag_field"][()].astype(np.float64)
        ne = f["gas_density"][()].astype(np.float64)
    return Bx,By,Bz,ne

def stokes_from_B_n(Bx,By,Bz,n,los_axis=0,eps=1e-30):
    theta = np.arctan2(By,Bx)
    cos2t = np.cos(2*theta)
    sin2t = np.sin(2*theta)
    b2 = Bx*Bx+By*By+Bz*Bz+eps
    sin2g = (Bx*Bx+By*By)/b2
    I = np.sum(n,axis=los_axis)
    Q = np.sum(n*cos2t*sin2g,axis=los_axis)
    U = np.sum(n*sin2t*sin2g,axis=los_axis)
    return I,Q,U

def angle_from_QU(Q,U):
    return 0.5*np.arctan2(U,Q)

def directional_spectrum_2d(chi):
    c2 = np.cos(2*chi)
    s2 = np.sin(2*chi)
    Fc = np.fft.fft2(c2)
    Fs = np.fft.fft2(s2)
    P = np.abs(Fc)**2 + np.abs(Fs)**2
    return np.fft.fftshift(P)

def radial_average_2d(P):
    ny,nx = P.shape
    y,x = np.indices((ny,nx))
    cy,cx = ny//2,nx//2
    r = np.sqrt((y-cy)**2+(x-cx)**2)
    r_i = r.astype(np.int64)
    tbin = np.bincount(r_i.ravel(), P.ravel())
    nr = np.bincount(r_i.ravel())
    prof = tbin/np.maximum(nr,1)
    k = np.arange(len(prof),dtype=float)
    return k[1:], prof[1:]

def directional_sectors(P, n_sectors=8):
    ny,nx = P.shape
    y,x = np.indices((ny,nx))
    cy,cx = ny//2,nx//2
    ang = np.arctan2(y-cy,x-cx)
    out = []
    for i in range(n_sectors):
        a0 = -np.pi + i*(2*np.pi/n_sectors)
        a1 = -np.pi + (i+1)*(2*np.pi/n_sectors)
        mask = (ang>=a0)&(ang<a1)
        k, pk = radial_average_2d(P*mask)
        out.append((0.5*(a0+a1), k, pk))
    return out

def fit_loglog(k, pk, kmin=None, kmax=None):
    m = np.isfinite(pk) & (pk>0)
    k = k[m]; pk = pk[m]
    if kmin is None: kmin = max(4, int(0.02*len(k)))
    if kmax is None: kmax = int(0.25*len(k))
    sel = (np.arange(len(k))>=kmin)&(np.arange(len(k))<=kmax)
    x = np.log10(k[sel]); y = np.log10(pk[sel])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept, (k[sel][0], k[sel][-1])

Bx,By,Bz,ne = load_fields(h5_path)
I,Q,U = stokes_from_B_n(Bx,By,Bz,ne,los_axis=0)
chi = angle_from_QU(Q,U)
P2d = directional_spectrum_2d(chi)
k, Pk = radial_average_2d(P2d)
slope, intercept, krange = fit_loglog(k, Pk)

plt.figure(figsize=(5,4))
plt.imshow(np.log10(P2d+1e-30), origin="lower")
plt.title("log10 $P_{dir}(k_x,k_y)$")
plt.colorbar()
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.loglog(k, Pk, label="Radial avg")
km = np.logspace(np.log10(krange[0]), np.log10(krange[1]), 50)
plt.loglog(km, 10**(intercept)*km**slope, linestyle="--", label=f"slope={slope:.2f}")
plt.xlabel("k (pix$^{-1}$)")
plt.ylabel("$P_{dir}(k)$")
plt.legend()
plt.tight_layout()

sectors = directional_sectors(P2d, n_sectors=8)
plt.figure(figsize=(6,4))
for ang, ks, ps in sectors:
    plt.loglog(ks, ps, alpha=0.6)
plt.xlabel("k (pix$^{-1}$)")
plt.ylabel("$P_{dir}(k)$ in sectors")
plt.tight_layout()

plt.show()
