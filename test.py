import numpy as np
from numpy.fft import fftn, ifftn

def generate_kolmogorov_field(N, L):
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    kz = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0,0,0] = 1e-10  # avoid division by zero

    # Random phases
    phase = np.exp(2j * np.pi * np.random.rand(N, N, N))

    # Kolmogorov amplitude ∝ k^(-11/6)
    amplitude = K**(-11/6)
    amplitude[K == 0] = 0

    spectrum = amplitude * phase
    v_field = np.real(ifftn(spectrum))

    return v_field



field = generate_kolmogorov_field(256, 2*np.pi)