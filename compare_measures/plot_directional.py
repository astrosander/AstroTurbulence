import numpy as np
import h5py
import matplotlib.pyplot as plt

BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

def pmap_from_cube(h5path, lam, L=1.0, gamma=2.0, rm_coeff=1.0):
    lam2 = float(lam) ** 2
    with h5py.File(h5path, "r") as f:
        bx = f[BX_KEY]; by = f[BY_KEY]; bz = f[BZ_KEY]; ne = f[NE_KEY]
        nx, ny, nz = bx.shape
        dz = L / float(nz)
        P = np.zeros((nx, ny), dtype=np.complex128)
        phi = np.zeros((nx, ny), dtype=np.float64)
        for k in range(nz):
            bxk = bx[:, :, k][...]
            byk = by[:, :, k][...]
            bzk = bz[:, :, k][...]
            nek = ne[:, :, k][...]
            rm_cell = rm_coeff * (nek * bzk) * dz
            phi_half = phi + 0.5 * rm_cell
            emis = (bxk * bxk + byk * byk) ** (gamma / 2.0)
            e2ipsi = np.exp(2j * np.arctan2(byk, bxk))
            P += (emis * e2ipsi) * np.exp(2j * lam2 * phi_half) * dz
            phi += rm_cell
    Q = P.real
    U = P.imag
    PI = np.maximum(np.abs(P), 1e-30)
    A = Q / PI
    B = U / PI
    return A, B

def ring_average_power(A, B):
    A = A - A.mean()
    B = B - B.mean()
    Ah = np.fft.fft2(A)
    Bh = np.fft.fft2(B)
    P2 = (np.abs(Ah)**2 + np.abs(Bh)**2) / (A.size)  # normalization
    nx, ny = A.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KR = np.sqrt(KX**2 + KY**2)
    kb = KR.astype(np.int32)
    kmax = kb.max()
    num = np.bincount(kb.ravel(), weights=P2.ravel(), minlength=kmax+1)
    den = np.bincount(kb.ravel(), minlength=kmax+1)
    spec = num / np.maximum(den, 1)
    k = np.arange(kmax + 1, dtype=np.float64)
    return k[1:], spec[1:]  # drop DC

def plot_directional_spectrum(h5path, lam, label=None, rm_coeff=1.0, gamma=2.0):
    A, B = pmap_from_cube(h5path, lam, L=1.0, gamma=gamma, rm_coeff=rm_coeff)
    k, P1D = ring_average_power(A, B)
    plt.loglog(k, P1D, label=label if label else f"{lam:.3g} m")
    kref = k[(k > 5) & (k < k.max()/3)]
    if kref.size:
        ref = P1D[np.argmin(np.abs(k - kref[len(kref)//2]))] * (kref / kref[len(kref)//2])**(-11/3)
        plt.loglog(kref, ref, color="k", label=f"slope=-11/3", linewidth=1.0)

if __name__ == "__main__":
    h5path = r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"   # change to any of your cubes
    lam = 10                             # wavelength in meters (example)
    plot_directional_spectrum(h5path, lam, label=f"{lam} m")
    plt.xlabel("k")
    plt.ylabel(r"$P_{\mathrm{dir}}(k)=|\widehat{\cos2\chi}|^2+|\widehat{\sin2\chi}|^2$")
    plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.show()
