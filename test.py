import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform

# Parameters
N = 512  # cube size
np.random.seed(0)

# Generate synthetic velocity fields with smooth turbulent structure
vx = gaussian_filter(np.random.normal(size=(N, N, N)), sigma=3)
vy = gaussian_filter(np.random.normal(size=(N, N, N)), sigma=3)

# Project along z-axis to get Q and U maps for synchrotron
Q_syn = np.sum(vx**2 - vy**2, axis=2)
U_syn = np.sum(2 * vx * vy, axis=2)

# For dust: normalize by the total transverse velocity magnitude squared
v_perp2 = vx**2 + vy**2
Q_dust = np.sum((vx**2 - vy**2) / (v_perp2 + 1e-8), axis=2)  # prevent division by zero
U_dust = np.sum(2 * vx * vy / (v_perp2 + 1e-8), axis=2)

# Compute polarization angles
phi_syn = 0.5 * np.arctan2(U_syn, Q_syn)
phi_dust = 0.5 * np.arctan2(U_dust, Q_dust)

# Function to compute angular structure function D_phi(R)
def compute_structure_function(phi_map, max_R=50):
    ny, nx = phi_map.shape
    Y, X = np.indices((ny, nx))
    coords = np.column_stack((X.ravel(), Y.ravel()))
    dists = squareform(pdist(coords))
    delta_phi_sq = squareform(pdist(phi_map.ravel().reshape(-1, 1)))**2

    R_vals = np.arange(1, max_R)
    D_phi = np.zeros_like(R_vals, dtype=float)

    for i, R in enumerate(R_vals):
        mask = (dists >= R - 0.5) & (dists < R + 0.5)
        D_phi[i] = np.mean(delta_phi_sq[mask])

    return R_vals, D_phi





# Redefine helper functions due to cell scope reset
from numpy.fft import fft2, ifft2, fftshift

def structure_function_fft(phi):
    phi = phi - np.mean(phi)
    phi2 = phi**2
    corr = fftshift(ifft2(fft2(phi) * np.conj(fft2(phi))).real)
    norm = fftshift(ifft2(fft2(np.ones_like(phi)) * np.conj(fft2(np.ones_like(phi)))).real)
    corr /= norm
    phi2_corr = fftshift(ifft2(fft2(phi2) * np.conj(fft2(np.ones_like(phi)))).real) / norm
    Dphi_map = phi2_corr - corr**2
    return Dphi_map

def radial_profile(data):
    y, x = np.indices(data.shape)
    center = np.array(data.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

# Re-run computations
Dphi_map_syn = structure_function_fft(phi_syn)
Dphi_map_dust = structure_function_fft(phi_dust)
R_vals = np.arange(1, N//2)

Dphi_syn = radial_profile(Dphi_map_syn)[1:N//2]
Dphi_dust = radial_profile(Dphi_map_dust)[1:N//2]

# Plot again
plt.figure(figsize=(8, 6))
plt.loglog(R_vals, Dphi_syn, 'r-', label='Velocity (synchrotron)')
plt.loglog(R_vals, Dphi_dust, 'b--', label='Velocity (dust)')
plt.xlabel("R [pixels]")
plt.ylabel(r"$D_\phi(R)$")
plt.title("Structure Function of Polarization Angle from Velocity Field (N=256, Nz=64)")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.show()
