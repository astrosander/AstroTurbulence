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

# --------------------------------------------------------------------
# 3‑D angular diagnostics
# --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def polar_angle(u):
    """
    θ(x,y,z) = arctan(√(ux²+uy²) / uz)   ∈ [0, π]
    """
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    return np.arctan2(np.sqrt(ux**2 + uy**2), uz)

def angle_between(v1, v2):
    """
    Smallest 3‑D angle between vectors v1 and v2 (any shape [...,3]).
    """
    dot = (v1 * v2).sum(axis=-1)
    n1  = np.linalg.norm(v1, axis=-1)
    n2  = np.linalg.norm(v2, axis=-1)
    cos = np.clip(dot / (n1 * n2 + 1e-30), -1.0, 1.0)
    return np.arccos(cos)

def pairwise_3d_angle(u, R, max_pairs=5_000_000_000_000_000_000, rng=None):
    """
    Return Δθ catalogue where Δθ is the 3‑D angle between velocity
    increments separated by |R|.
    """
    Nx, Ny, Nz, _ = u.shape
    rng = rng or np.random.default_rng()
    dirs = [(R,0,0),(-R,0,0),(0,R,0),(0,-R,0),(0,0,R),(0,0,-R)]
    angles = []
    for dx,dy,dz in dirs:
        n_pick = int(np.cbrt(max_pairs // len(dirs)))
        xs = rng.integers(0, Nx, size=n_pick)
        ys = rng.integers(0, Ny, size=n_pick)
        zs = rng.integers(0, Nz, size=n_pick)
        v1 = u[xs, ys, zs]
        v2 = u[(xs+dx)%Nx, (ys+dy)%Ny, (zs+dz)%Nz]
        angles.append(angle_between(v1, v2))
    return np.concatenate(angles)

def plot_3d_polar(u, R_values, outdir="figures"):
    """
    Single‑point PDF P1(θ) and two‑point PDF PΔ(Δθ;R) for polar angle.
    """
    θ = polar_angle(u)
    # ------ single‑point PDF ------
    θ_flat = θ.ravel()
    bins = np.linspace(0, np.pi, 91)
    counts, _ = np.histogram(θ_flat, bins=bins, density=False)
    centres = 0.5*(bins[:-1]+bins[1:])
    pdf = counts / (np.sin(centres) * counts.sum() * (bins[1]-bins[0]))
    plt.figure()
    plt.semilogy(centres, pdf, '.')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P_{1}(\theta)$')
    plt.title('Single‑point polar‑angle PDF')
    plt.tight_layout()
    plt.savefig(f"{outdir}/pdf_theta_single.png", dpi=200)
    plt.close()

    # ------ two‑point PDFs ------
    plt.figure()
    for R in R_values:
        Δθ = pairwise_3d_angle(u, R)
        bins = np.linspace(0, np.pi, 91)
        c, _ = np.histogram(Δθ, bins=bins, density=False)
        centre = 0.5*(bins[:-1]+bins[1:])
        PΔ = c / (np.sin(centre) * c.sum() * (bins[1]-bins[0]))
        plt.semilogy(centre, PΔ, label=f'R={R}')
    plt.xlabel(r'$\Delta\theta$')
    plt.ylabel(r'$P_{\Delta}(\Delta\theta;R)$')
    plt.title('Two‑point polar‑angle PDFs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/pdf_theta_pairs.png", dpi=200)
    plt.close()


plot_3d_polar(u, R_values=[2,4,8,16,32])