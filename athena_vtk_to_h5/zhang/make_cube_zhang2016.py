# make_cube_zhang2016.py
import numpy as np
import h5py
import argparse

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
    return (a + 1j*b).astype(np.complex64)

def synth_divfree_vector(nx, ny, nz, beta=11/3, kmin=1.0, kmax=None, rms=1.0, seed=2025):
    """
    Isotropic, divergence-free 3D vector field with power spectrum ~ k^{-beta}.
    Hermitian spectral construction; irfftn in z.
    """
    rng = np.random.default_rng(seed)
    if kmax is None:
        kmax = min(nx, ny, nz) / 2.0

    KX, KY, KZ = kgrid_hermitian(nx, ny, nz)
    K2 = (KX*KX + KY*KY + KZ*KZ).astype(np.float32)
    K  = np.sqrt(K2, dtype=np.float32)

    # isotropic shell amplitude; remove DC and outside band
    amp = np.zeros_like(K, dtype=np.float32)
    mask = (K >= float(kmin)) & (K <= float(kmax))
    amp[mask] = K[mask]**(-beta/2.0)

    # random complex vector field in k-space
    Xx = complex_gaussian(K.shape, rng) * amp
    Xy = complex_gaussian(K.shape, rng) * amp
    Xz = complex_gaussian(K.shape, rng) * amp

    # project out divergence: Bk = X - (k k·X)/k^2
    denom = K2.copy(); denom[denom == 0] = 1.0
    dot = (Xx*KX + Xy*KY + Xz*KZ) / denom
    Bxk = (Xx - dot*KX).astype(np.complex64)
    Byk = (Xy - dot*KY).astype(np.complex64)
    Bzk = (Xz - dot*KZ).astype(np.complex64)

    # enforce zero DC
    Bxk[0,0,0] = 0; Byk[0,0,0] = 0; Bzk[0,0,0] = 0

    # back to x-space
    Bx = np.fft.irfftn(Bxk, s=(nx, ny, nz)).astype(np.float32)
    By = np.fft.irfftn(Byk, s=(nx, ny, nz)).astype(np.float32)
    Bz = np.fft.irfftn(Bzk, s=(nx, ny, nz)).astype(np.float32)

    # zero-mean each component; scale each to same rms
    for A in (Bx, By, Bz):
        A -= A.mean()
        A *= (rms / (np.sqrt((A*A).mean()) + 1e-20))

    # η = ∞ → strictly zero mean B∥ (we will use LOS = z)
    Bz -= Bz.mean()

    return Bx, By, Bz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--beta", type=float, default=11/3)
    ap.add_argument("--seedB", type=int, default=2025)
    ap.add_argument("--outfile", type=str, default=None)
    args = ap.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    Bx, By, Bz = synth_divfree_vector(nx, ny, nz, beta=args.beta, kmin=1.0, kmax=min(nx,ny,nz)/2,
                                      rms=1.0, seed=args.seedB)

    # Constant electron density (n_e = 1) as in the cleanest LP16/Zhang test
    ne = np.ones((nx, ny, nz), dtype=np.float32)

    of = args.outfile or f"synthetic_{nx}x{ny}x{nz}.h5"
    with h5py.File(of, "w") as f:
        f.create_dataset(BX_KEY, data=Bx, dtype="f4", compression="gzip", compression_opts=4)
        f.create_dataset(BY_KEY, data=By, dtype="f4", compression="gzip", compression_opts=4)
        f.create_dataset(BZ_KEY, data=Bz, dtype="f4", compression="gzip", compression_opts=4)
        f.create_dataset(NE_KEY, data=ne, dtype="f4", compression="gzip", compression_opts=4)
        f.attrs["beta"] = float(args.beta)
        f.attrs["kmin"] = 1.0
        f.attrs["kmax"] = float(min(nx,ny,nz)/2)
        f.attrs["ne_type"] = "constant_1"
        f.attrs["eta"] = "infinity (mean Bz removed)"
    print(f"Saved {of}")

if __name__ == "__main__":
    main()
