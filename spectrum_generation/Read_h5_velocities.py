import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "simulation.h5"

with h5py.File(filename, "r") as hf:
    vx = np.array(hf["u"][:,:,:,0])
    vy = np.array(hf["u"][:,:,:,1])
    vz = np.array(hf["u"][:,:,:,2])
    u = np.array(hf["u"])

vx = vx.flatten()
vy = vy.flatten()
vz = vz.flatten()

vel_mag = np.sqrt(vx**2+vy**2+vz**2)

phi = np.arctan2(vy, vx)  
theta = np.arctan2(np.sqrt(vx**2 + vy**2), vz)  