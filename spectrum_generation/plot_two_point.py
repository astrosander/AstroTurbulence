import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

filename = "baseline_512.h5"

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


def los_integrate(u: np.ndarray, axis: int = 2) -> np.ndarray:
	return np.sum(u, axis=axis)
u_int = los_integrate(u, axis=2)


# STAT_PAIRS = 10**5

theta = np.arctan2(vy, vx)

def angular_difference(phi1, phi2):
    return (phi2 - phi1 + np.pi) % (2*np.pi) - np.pi


def collect_pairs(theta_map, R, max_pairs=2_000_000, rng=None):
   	Nx, Ny = theta_map.shape
    rng = rng or np.random.default_rng()

    dirs = [(R, 0), (-R, 0), (0, R), (0, -R),
            (R, R), (-R, -R), (R, -R), (-R, R)]
    deltas = []
    for dx, dy in dirs:
        n_pick = int(np.sqrt(max_pairs // len(dirs)))
        xs = rng.integers(0, Nx, size=n_pick)
        ys = rng.integers(0, Ny, size=n_pick)
        theta1 = theta_map[xs, ys]
        theta2 = theta_map[(xs+dx) % Nx, (ys+dy) % Ny]
        deltas.append(angular_difference(theta1, theta2))
    return np.concatenate(deltas)


def two_point_pdf(theta_map, R, bins=181, **kwargs):
    dth = collect_pairs(theta_map, R, **kwargs)
    edges = np.linspace(-np.pi, np.pi, bins)
    counts, _ = np.histogram(dth, edges, density=False)
    centres = 0.5*(edges[:-1] + edges[1:])
    pdf = counts / counts.sum() / (edges[1]-edges[0])
    return centres, pdf, dth


def angle_structure_function(theta_map, R_values, **kwargs):
    D = []
    for R in R_values:
        dth = collect_pairs(theta_map, R, **kwargs)
        D.append(0.5 * (1 - np.cos(2*dth).mean()))
    return np.asarray(R_values), np.asarray(D)


def plot_two_point(theta_map, R_list, label, outdir="figures",
                   max_pairs=1000_000_000_000_000):

    plt.figure()
    for R in R_list:
        x, pdf, _ = two_point_pdf(theta_map, R, max_pairs=max_pairs)
        plt.semilogy(x, pdf, label=f"R={R}")
    plt.xlabel(r"$\Delta\varphi$")
    plt.ylabel(r"$P(\Delta\varphi;R)$")
    plt.title(f"Two‑point PDFs – {label}")
    plt.legend(loc="lower center", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/pdf_pairs_{label}.png", dpi=200)
    plt.close()


    R_vals = np.unique(np.logspace(0, np.log10(theta_map.shape[0]//3), 25).astype(int))
    R, D = angle_structure_function(theta_map, R_vals, max_pairs=max_pairs)

    plt.figure()
    plt.loglog(R, D, "o", ms=3, label="simulation")
    expected_slope = 2/3 if "vector" in label else 5/3
    plt.loglog(R, 0.1*R**expected_slope, "--",
               label=fr"$\propto R^{{{expected_slope:.2f}}}$")
    plt.xlabel(r"$R$ [pixels]")
    plt.ylabel(r"$D_{\varphi}(R)$")
    plt.title(f"Structure function – {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/structure_{label}.png", dpi=200)
    plt.close()

def angle_maps_vector(u_int: np.ndarray) -> np.ndarray:
    ux, uy = u_int[..., 0], u_int[..., 1]
    return np.arctan2(uy, ux)

@njit(parallel=True, fastmath=True)
def _stokes_accumulate(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Nx, Ny, Nz, _ = u.shape
    Q = np.zeros((Nx, Ny), dtype=np.float32)
    U = np.zeros((Nx, Ny), dtype=np.float32)
    for i in prange(Nx):
        for j in prange(Ny):
            q, u_local = 0.0, 0.0
            for k in range(Nz):
                ux, uy = u[i, j, k, 0], u[i, j, k, 1]
                theta = np.arctan2(uy, ux)
                q += np.cos(2.0 * theta)
                u_local += np.sin(2.0 * theta)
            Q[i, j] = q
            U[i, j] = u_local
    return Q, U

def angle_maps_stokes(u: np.ndarray) -> np.ndarray:
    Q, U = _stokes_accumulate(u)
    return 0.5 * np.arctan2(U, Q)


theta_v = angle_maps_vector(u_int)
theta_s = angle_maps_stokes(u)

print("Two–point statistics …")
plot_two_point(theta_v, R_list=[2,4,8,16,32], label="vector")
plot_two_point(theta_s, R_list=[2,4,8,16,32], label="stokes")