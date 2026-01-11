#!/usr/bin/env python3

import json
import numpy as np

Nx = 512*4
Ny = 512*4
Lx = 10.0
Ly = 10.0
m_psi = 4#2/3
m_phi = 2/3#5/3
R0_psi = 1.0
r_phi = 0.1
inner_psi = 0.0
inner_phi = 0.0
rms_psi = 1.0
rms_phi = 1.0
p0 = 1.0
chi = "0.01"
seed = 0
out = "screens_cube.npz"


def von_karman_psd_2d(k, m, k0, k_inner=None):
    beta = m + 2.0
    psd = (k * k + k0 * k0) ** (-0.5 * beta)
    if k_inner is not None and k_inner > 0:
        psd *= np.exp(-(k / k_inner) ** 2)
    psd = psd.astype(np.float64)
    psd[k == 0] = 0.0
    return psd


def gaussian_real_field_2d_from_psd(Ny, Nx, Ly, Lx, psd_func, rng):
    dy = Ly / Ny
    dx = Lx / Nx

    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)[:, None]
    kx = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)[None, :]
    k = np.sqrt(kx * kx + ky * ky)

    psd = psd_func(k)
    noise = (rng.normal(size=psd.shape) + 1j * rng.normal(size=psd.shape)) / np.sqrt(2.0)

    fk = noise * np.sqrt(psd)
    field = np.fft.irfft2(fk, s=(Ny, Nx))
    return field.astype(np.float64)


def rescale_to_rms(field, target_rms):
    field = field - np.mean(field)
    rms = np.sqrt(np.mean(field * field))
    if rms == 0:
        return field
    return field * (target_rms / rms)


def main():
    rng = np.random.default_rng(seed)

    chi_list = np.array([float(x) for x in chi.split(",")], dtype=np.float64)
    print(chi_list)
    # chi_list = [1]
    Nchi = chi_list.size

    k0_psi = 2.0 * np.pi / R0_psi
    k0_phi = 2.0 * np.pi / r_phi

    k_inner_psi = (2.0 * np.pi / inner_psi) if inner_psi and inner_psi > 0 else None
    k_inner_phi = (2.0 * np.pi / inner_phi) if inner_phi and inner_phi > 0 else None

    def psd_psi(k):
        return von_karman_psd_2d(k, m=m_psi, k0=k0_psi, k_inner=k_inner_psi)

    def psd_phi(k):
        return von_karman_psd_2d(k, m=m_phi, k0=k0_phi, k_inner=k_inner_phi)

    psi = gaussian_real_field_2d_from_psd(Ny, Nx, Ly, Lx, psd_psi, rng)
    phi = gaussian_real_field_2d_from_psd(Ny, Nx, Ly, Lx, psd_phi, rng)

    psi = rescale_to_rms(psi, rms_psi)
    phi = rescale_to_rms(phi, rms_phi)

    P0 = p0 * np.exp(2j * psi)

    Q = np.empty((Nchi, Ny, Nx), dtype=np.float64)
    U = np.empty((Nchi, Ny, Nx), dtype=np.float64)

    for i, chi_val in enumerate(chi_list):
        P = P0 * np.exp(2j * chi_val * phi)
        Q[i] = P.real
        U[i] = P.imag

    meta = dict(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        m_psi=m_psi, m_phi=m_phi,
        R0_psi=R0_psi, r_phi=r_phi,
        inner_psi=inner_psi, inner_phi=inner_phi,
        rms_psi=rms_psi, rms_phi=rms_phi,
        p0=p0, chi=chi_list.tolist(),
        seed=seed,
        model="Separated screens: intrinsic synchrotron region + foreground Faraday screen",
    )

    np.savez_compressed(
        out,
        psi=psi,
        phi=phi,
        chi=chi_list,
        Q=Q,
        U=U,
        meta=json.dumps(meta),
    )
    print(f"Saved: {out}")
    print("meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
