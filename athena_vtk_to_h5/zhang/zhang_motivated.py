import numpy as np
import h5py

BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"
NE_KEY = "gas_density"

def kgrid_hermitian(nx, ny, nz):
    kx = (np.fft.fftfreq(nx) * nx).astype(np.float32)
    ky = (np.fft.fftfreq(ny) * ny).astype(np.float32)
    kz = (np.fft.rfftfreq(nz) * nz).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return KX, KY, KZ

def complex_gaussian(shape, rng):
    a = rng.standard_normal(shape, dtype=np.float32)
    b = rng.standard_normal(shape, dtype=np.float32)
    return (a + 1j * b).astype(np.complex64)

def synth_divfree_vector(nx, ny, nz, beta=11/3, kmin=1.0, kmax=None, rms=1.0, seed=7):
    rng = np.random.default_rng(seed)
    if kmax is None:
        kmax = min(nx, ny, nz) / 2
    KX, KY, KZ = kgrid_hermitian(nx, ny, nz)
    K2 = (KX*KX + KY*KY + KZ*KZ).astype(np.float32)
    K = np.sqrt(K2, dtype=np.float32)
    mask = (K >= kmin) & (K <= kmax)
    amp = np.zeros_like(K, dtype=np.float32)
    amp[mask] = K[mask] ** (-beta/2.0)
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
    Bx = np.fft.irfftn(Bxk, s=(nx, ny, nz)).astype(np.float32)
    By = np.fft.irfftn(Byk, s=(nx, ny, nz)).astype(np.float32)
    Bz = np.fft.irfftn(Bzk, s=(nx, ny, nz)).astype(np.float32)
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

def synth_scalar(nx, ny, nz, beta=11/3, kmin=1.0, kmax=None, rms=1.0, seed=13):
    rng = np.random.default_rng(seed)
    if kmax is None:
        kmax = min(nx, ny, nz) / 2
    KX, KY, KZ = kgrid_hermitian(nx, ny, nz)
    K2 = (KX*KX + KY*KY + KZ*KZ).astype(np.float32)
    K = np.sqrt(K2, dtype=np.float32)
    mask = (K >= kmin) & (K <= kmax)
    amp = np.zeros_like(K, dtype=np.float32)
    amp[mask] = K[mask] ** (-beta/2.0)
    Nk = complex_gaussian(K.shape, rng) * amp
    Nk[0,0,0] = 0
    n = np.fft.irfftn(Nk, s=(nx, ny, nz)).astype(np.float32)
    n -= n.mean()
    sn = float(np.sqrt((n*n).mean()))
    n *= (rms / (sn + 1e-12))
    return n

def main():
    nx, ny, nz = 128, 128, 4096
    bx, by, bz = synth_divfree_vector(nx, ny, nz, beta=11/3, kmin=1.0, kmax=min(nx,ny,nz)/2, rms=1.0, seed=2025)
    ne = synth_scalar(nx, ny, nz, beta=11/3, kmin=1.0, kmax=min(nx,ny,nz)/2, rms=1.0, seed=1234)
    mz = float(bz.mean())
    sz = float(bz.std())
    eta = np.inf if abs(mz) < 1e-30 else (sz / abs(mz))
    with h5py.File(f"synthetic_{nx}x{ny}x{nz}.h5", "w") as f:
        f.create_dataset(BX_KEY, data=bx, dtype="f4", compression="gzip", compression_opts=4)
        f.create_dataset(BY_KEY, data=by, dtype="f4", compression="gzip", compression_opts=4)
        f.create_dataset(BZ_KEY, data=bz, dtype="f4", compression="gzip", compression_opts=4)
        f.create_dataset(NE_KEY, data=ne, dtype="f4", compression="gzip", compression_opts=4)
        f.attrs["eta_definition"] = "sigma(Bz) / mean(Bz)"
        f.attrs["eta"] = eta

if __name__ == "__main__":
    main()
