"""
Ploting distributions of the velocities spectrum of 512^3 cube of both v_x, v_y, v_z components and velocity field magnitude
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "baseline_512.h5"

with h5py.File(filename, "r") as hf:
    vx = np.array(hf["u"][:,:,:,0])
    vy = np.array(hf["u"][:,:,:,1])
    vz = np.array(hf["u"][:,:,:,2])

vx = vx.flatten()
vy = vy.flatten()
vz = vz.flatten()

vel_mag = np.sqrt(vx**2+vy**2+vz**2)


print("vx shape:", vx.shape)
print("vy shape:", vy.shape)
print("vz shape:", vz.shape)


plt.figure(figsize=(8, 5), dpi=120)

plt.hist(vx, bins=300, color='coral', alpha=0.6, label='$v_x$')
plt.hist(vy, bins=300, color='mediumseagreen', alpha=0.6, label='$v_y$')
plt.hist(vz, bins=300, color='cornflowerblue', alpha=0.6, label='$v_z$')
plt.hist(vel_mag, bins=300, color='gold', alpha=0.5, label='$v_{mag}$')

plt.title("Velocity Component Distribution", fontsize=18, fontweight='bold', color='darkslategray')
plt.xlabel("Velocity", fontsize=14, color='gray')
plt.ylabel("Frequency", fontsize=14, color='gray')

plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.legend(loc='upper right', fontsize=12, frameon=False)

plt.show()