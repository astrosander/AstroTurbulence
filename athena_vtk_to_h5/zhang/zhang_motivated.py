import numpy as np
import h5py
from dataclasses import dataclass
from typing import Optional, Tuple

BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

@dataclass
class BoxSpec:
    nx: int
    ny: int
    nz: int
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    # sampling: keep only this many sky pixels (LP16: 128×128), while nz can be large
    save_nx: int = 128
    save_ny: int = 128

def kgrid_hermitian(box: BoxSpec):
    """
    Physical wavenumbers with 2π factors and physical lengths (isotropic definition):
      kx = 2π * fftfreq(nx, d=Lx/nx), etc.
    Using rfftfreq along z to exploit Hermitian symmetry in ifft.
    Returns KX,KY,KZ with shapes (nx,ny,nz//2+1).
    """
    dx, dy, dz = box.Lx/box.nx, box.Ly/box.ny, box.Lz/box.nz
    kx = (2*np.pi)*np.fft.fftfreq(box.nx, d=dx).astype(np.float32)
    ky = (2*np.pi)*np.fft.fftfreq(box.ny, d=dy).astype(np.float32)
    kz = (2*np.pi)*np.fft.rfftfreq(box.nz, d=dz).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return KX, KY, KZ

def complex_gaussian(shape, rng):
    a = rng.standard_normal(shape, dtype=np.float32)
    b = rng.standard_normal(shape, dtype=np.float32)
    return (a + 1j * b).astype(np.complex64)

def _soft_window(k: np.ndarray, kmin: float, kmax: float, edge: float = 0.0) -> np.ndarray:
    """
    Optional smooth cosine tapers near kmin/kmax to reduce ringing.
    edge=0 disables and returns a sharp top-hat.
    """
    win = (k >= kmin) & (k <= kmax)
    if edge <= 0:
        return win.astype(np.float32)
    w = np.zeros_like(k, dtype=np.float32)
    # inner taper [kmin, kmin*(1+edge)]
    ki1, ki2 = kmin, kmin*(1+edge)
    ko1, ko2 = kmax*(1-edge), kmax
    # central plateau
    mid = (k >= ki2) & (k <= ko1)
    w[mid] = 1.0
    # low-k rise
    low = (k >= ki1) & (k < ki2)
    if np.any(low):
        x = (k[low]-ki1)/(ki2-ki1)
        w[low] = 0.5*(1 - np.cos(np.pi*x))
    # high-k fall
    high = (k > ko1) & (k <= ko2)
    if np.any(high):
        x = (ko2-k[high])/(ko2-ko1)
        w[high] = 0.5*(1 - np.cos(np.pi*x))
    return w

def synth_divfree_vector(box: BoxSpec,
                         beta: float = 11/3,
                         Linj: Optional[float] = None,
                         kmax: Optional[float] = None,
                         rms: float = 1.0,
                         seed: int = 7,
                         taper_edge: float = 0.0):
    rng = np.random.default_rng(seed)
    KX, KY, KZ = kgrid_hermitian(box)
    K2 = (KX*KX + KY*KY + KZ*KZ).astype(np.float32)
    K = np.sqrt(K2, dtype=np.float32)
    # isotropic injection and dissipation scales (spherical in k)
    if Linj is None:
        Linj = max(box.Lx, box.Ly, box.Lz)   # inject at box scale (LP16-like)
    kmin = 2*np.pi/float(Linj)
    # Nyquist (smallest resolved wavelengths among axes)
    kN_x = np.pi/(box.Lx/box.nx)
    kN_y = np.pi/(box.Ly/box.ny)
    kN_z = np.pi/(box.Lz/box.nz)
    kN = float(min(kN_x, kN_y, kN_z))
    if kmax is None:
        kmax = 0.95*kN
    # amplitude envelope (top-hat w/ optional soft edges)
    win = _soft_window(K, kmin, kmax, edge=taper_edge)
    amp = np.where(win>0, (K + (K==0)).astype(np.float32)**(-beta/2.0)*win, 0.0).astype(np.float32)
    Xx = complex_gaussian(K.shape, rng) * amp
    Xy = complex_gaussian(K.shape, rng) * amp
    Xz = complex_gaussian(K.shape, rng) * amp
    denom = K2.copy()
    denom[denom == 0] = 1.0
    dot = (Xx*KX + Xy*KY + Xz*KZ) / denom
    Bxk = (Xx - dot*KX).astype(np.complex64)
    Byk = (Xy - dot*KY).astype(np.complex64)
    Bzk = (Xz - dot*KZ).astype(np.complex64)
    Bxk[0,0,0] = 0
    Byk[0,0,0] = 0
    Bzk[0,0,0] = 0
    Bx = np.fft.irfftn(Bxk, s=(box.nx, box.ny, box.nz)).astype(np.float32)
    By = np.fft.irfftn(Byk, s=(box.nx, box.ny, box.nz)).astype(np.float32)
    Bz = np.fft.irfftn(Bzk, s=(box.nx, box.ny, box.nz)).astype(np.float32)
    for A in (Bx, By, Bz):
        A -= A.mean()
    sx = float(np.sqrt((Bx*Bx).mean()))
    sy = float(np.sqrt((By*By).mean()))
    sz = float(np.sqrt((Bz*Bz).mean()))
    Bx *= (rms / (sx + 1e-12))
    By *= (rms / (sy + 1e-12))
    Bz *= (rms / (sz + 1e-12))
    Bz -= Bz.mean()
    return Bx, By, Bz

def synth_scalar(box: BoxSpec,
                 beta: float = 11/3,
                 Linj: Optional[float] = None,
                 kmax: Optional[float] = None,
                 rms: float = 1.0,
                 seed: int = 13,
                 taper_edge: float = 0.0):
    rng = np.random.default_rng(seed)
    KX, KY, KZ = kgrid_hermitian(box)
    K2 = (KX*KX + KY*KY + KZ*KZ).astype(np.float32)
    K = np.sqrt(K2, dtype=np.float32)
    if Linj is None:
        Linj = max(box.Lx, box.Ly, box.Lz)
    kmin = 2*np.pi/float(Linj)
    kN_x = np.pi/(box.Lx/box.nx)
    kN_y = np.pi/(box.Ly/box.ny)
    kN_z = np.pi/(box.Lz/box.nz)
    kN = float(min(kN_x, kN_y, kN_z))
    if kmax is None:
        kmax = 0.95*kN
    win = _soft_window(K, kmin, kmax, edge=taper_edge)
    amp = np.where(win>0, (K + (K==0)).astype(np.float32)**(-beta/2.0)*win, 0.0).astype(np.float32)
    Nk = complex_gaussian(K.shape, rng) * amp
    Nk[0,0,0] = 0
    n = np.fft.irfftn(Nk, s=(box.nx, box.ny, box.nz)).astype(np.float32)
    n -= n.mean()
    sn = float(np.sqrt((n*n).mean()))
    n *= (rms / (sn + 1e-12))
    return n

def _downsample_xy(arr: np.ndarray, save_nx: int, save_ny: int, method: str = "stride") -> np.ndarray:
    """
    Keep only save_nx × save_ny sightlines. Two options:
      - 'stride': pick evenly spaced indices (closest to LP16's 'keep 128^2 lines')
      - 'block':  average non-overlapping blocks (anti-alias, changes small-scale stats)
    """
    nx, ny, nz = arr.shape
    if (save_nx, save_ny) == (nx, ny):
        return arr
    if method == "block":
        fx = nx // save_nx
        fy = ny // save_ny
        arr = arr[:fx*save_nx, :fy*save_ny, :]
        arr = arr.reshape(save_nx, fx, save_ny, fy, nz).mean(axis=(1,3))
        return arr
    # stride
    ix = np.linspace(0, nx-1, save_nx, dtype=int)
    iy = np.linspace(0, ny-1, save_ny, dtype=int)
    return arr[np.ix_(ix, iy, np.arange(nz))]

def long_wave_diagnostic(field_z: np.ndarray, box: BoxSpec) -> dict:
    """
    Report whether the very first few kz modes carry non-zero power.
    Answers questions like: is P(k=2π/1024) == 0?
    Uses full-resolution cube (before downsampling sky).
    """
    # FFT along z only, average over kx,ky
    kz = (2*np.pi)*np.fft.rfftfreq(box.nz, d=box.Lz/box.nz).astype(np.float64)
    Fz = np.fft.rfftn(field_z, s=(box.nx, box.ny, box.nz), axes=(0,1,2))
    # power spectrum density vs kz, averaged over kx,ky
    Pkz = np.mean(np.abs(Fz)**2, axis=(0,1))
    def at_wavenumber(index: int) -> float:
        if index < len(Pkz):
            return float(Pkz[index])
        return 0.0
    diag = {
        "kz_0": float(kz[0]),
        "kz_1": float(kz[1]) if len(kz)>1 else np.nan,
        "P_at_kz1": at_wavenumber(1),   # power at 2π/Lz
        "P_at_kz2": at_wavenumber(2),   # power at 4π/Lz
    }
    return diag

def main(n_realizations: int = 1,
         seeds: Optional[Tuple[int,int]] = None,
         Linj: Optional[float] = None,
         Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
         nx: int = 128, ny: int = 128, nz: int = 4096,
         save_nx: int = 128, save_ny: int = 128,
         beta: float = 11/3, taper_edge: float = 0.0):
    box = BoxSpec(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, save_nx=save_nx, save_ny=save_ny)
    if seeds is None:
        seeds = (2025, 1234)  # (magnetic, density)
    bseed, nseed = seeds
    # ensemble loop
    for r in range(n_realizations):
        bx, by, bz = synth_divfree_vector(box, beta=beta, Linj=Linj, rms=1.0, seed=bseed+r, taper_edge=taper_edge)
        ne        = synth_scalar     (box, beta=beta, Linj=Linj, rms=1.0, seed=nseed+r, taper_edge=taper_edge)
        # Store full cube or only 128×128 sightlines? LP16 kept only 128×128 LOS maps.
        bx_s = _downsample_xy(bx, save_nx, save_ny, method="stride")
        by_s = _downsample_xy(by, save_nx, save_ny, method="stride")
        bz_s = _downsample_xy(bz, save_nx, save_ny, method="stride")
        ne_s = _downsample_xy(ne, save_nx, save_ny, method="stride")
        
        mz = float(bz_s.mean()); sz = float(bz_s.std())
        eta = np.inf if abs(mz) < 1e-30 else (sz / abs(mz))
        tag = f"r{r:03d}_"
        with h5py.File(f"synthetic_{save_nx}x{save_ny}x{nz}_{tag}.h5", "w") as f:
            f.create_dataset(BX_KEY, data=bx_s, dtype="f4", compression="gzip", compression_opts=4)
            f.create_dataset(BY_KEY, data=by_s, dtype="f4", compression="gzip", compression_opts=4)
            f.create_dataset(BZ_KEY, data=bz_s, dtype="f4", compression="gzip", compression_opts=4)
            f.create_dataset(NE_KEY, data=ne_s, dtype="f4", compression="gzip", compression_opts=4)
            # metadata / diagnostics
            f.attrs["box_nxyz"] = (box.nx, box.ny, box.nz)
            f.attrs["box_Lxyz"] = (box.Lx, box.Ly, box.Lz)
            f.attrs["saved_xy"] = (save_nx, save_ny)
            f.attrs["beta"] = beta
            f.attrs["Linj"] = float(Linj if Linj is not None else max(box.Lx, box.Ly, box.Lz))
            # η ~ σ(Bz)/|⟨Bz⟩|
            f.attrs["eta_definition"] = "sigma(Bz)/abs(mean(Bz)) on saved sightlines"
            f.attrs["eta"] = eta
            # long-wave diagnostic (see Patch 3)
            kdiag = long_wave_diagnostic(bz, box)
            for k, val in kdiag.items():
                f.attrs[k] = val

if __name__ == "__main__":
    # LP16-like: box-scale injection, isotropic, keep only 128×128 sightlines, deep LOS
    main(n_realizations=1, Linj=None, Lx=1.0, Ly=1.0, Lz=1.0,
         nx=128, ny=128, nz=256, save_nx=128, save_ny=128,
         beta=11/3, taper_edge=0.0)
