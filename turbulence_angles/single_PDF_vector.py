import h5py
import numpy as np
import matplotlib.pyplot as plt

# Parameters
filename = "baseline_512.h5"

# Open the HDF5 file and read the velocity components
with h5py.File(filename, "r") as hf:
    vx = np.array(hf["u"][:,:,:,0])  # x-component of velocity
    vy = np.array(hf["u"][:,:,:,1])  # y-component of velocity
    vz = np.array(hf["u"][:,:,:,2])  # z-component of velocity
    u = np.array(hf["u"])

vx = vx.flatten()
vy = vy.flatten()
vz = vz.flatten()

vel_mag = np.sqrt(vx**2+vy**2+vz**2)

phi = np.arctan2(vy, vx)  # Azimuthal angle
theta = np.arctan2(np.sqrt(vx**2 + vy**2), vz)  # Polar angle



def los_integrate(u: np.ndarray, axis: int = 2) -> np.ndarray:
	return np.sum(u, axis=axis)
u_int = los_integrate(u, axis=2)


# STAT_PAIRS = 10**5

theta = np.arctan2(vy, vx)

nbins = 1800

bins = np.linspace(-np.pi, np.pi, nbins + 1)
counts, _ = np.histogram(theta.ravel(), bins=bins, density=False)
centers = 0.5 * (bins[:-1] + bins[1:])
# sinθ factor for isotropic surface element
weights = np.sin(np.abs(centers))
pdf = counts / (weights * counts.sum())


name = "vector_mine"
plt.figure()

plt.semilogy(centers, pdf, label=f"empirical {name}")

plt.xlabel(r"$\theta$")
plt.ylabel(r"$P(\theta)$")
plt.title(f"Single‑point angle PDF – {name}")
plt.legend()
plt.tight_layout()
plt.show()