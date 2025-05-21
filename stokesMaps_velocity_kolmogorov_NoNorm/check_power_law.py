import struct
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def read_fortran_2d_array(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        f.read(4)                              
        ndim = struct.unpack("i", f.read(4))[0]
        f.read(4)                              
        f.read(4)
        if ndim == 2:
            nx, ny = struct.unpack("ii", f.read(8))
        else:
            dims = struct.unpack("iiii", f.read(16))
            nx, ny = dims[:2]
        f.read(4)

        f.read(4)
        data = np.fromfile(f, dtype=np.float32, count=nx * ny)
        f.read(4)

    return data.reshape((nx, ny), order="F")


def angular_difference(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return (phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi


def sample_pairs_2d(field: np.ndarray, R: int, max_pairs: int = 300_000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    Nx, Ny = field.shape
    rng = rng or np.random.default_rng()
    dirs: list[Tuple[int, int]] = [
        (R, 0), (-R, 0), (0, R), (0, -R),
        (R, R), (-R, -R), (R, -R), (-R, R),
    ]
    out: list[np.ndarray] = []
    for dx, dy in dirs:
        n = int(np.sqrt(max_pairs // len(dirs))) * 1000 
        xs = rng.integers(0, Nx, size=n)
        ys = rng.integers(0, Ny, size=n)
        p1 = field[xs, ys]
        p2 = field[(xs + dx) % Nx, (ys + dy) % Ny]
        out.append(angular_difference(p1, p2))
    return np.concatenate(out)


def structure_function_2d(field: np.ndarray, Rs: np.ndarray,
                          max_pairs: int = 300_000) -> Tuple[np.ndarray, np.ndarray]:
    D = []
    for R in Rs:
        dphi = sample_pairs_2d(field, int(R), max_pairs=max_pairs)
        harmonic = 2      
        D.append( 0.5 * (1 - np.cos(harmonic * dphi).mean()) )
        # D.append(0.5 * (1 - np.cos(dphi).mean()))
    return Rs, np.asarray(D)

def plot_structure(field: np.ndarray, Rs: np.ndarray, title: str, outfile: Path,
                   expected_slope: float | None = None) -> float:
    R_arr, D = structure_function_2d(field, Rs)

    half = len(R_arr) // 2
    slope, intercept, *_ = linregress(np.log(R_arr[:half]), np.log(D[:half]))

    plt.figure(figsize=(4.8, 3.6))  
    plt.loglog(R_arr, D, "o", label=f"simulation")

    ref_slope = slope if expected_slope is None else expected_slope
    C = D[1] / (R_arr[1] ** ref_slope)
    plt.loglog(R_arr, C * R_arr ** ref_slope, "--", color="black",
               label=f"$\\propto R^{{{ref_slope:.2f}}}$")

    plt.xlabel("R [pixels]")
    plt.ylabel("D(R)")
    plt.title(title, fontsize=13)
    plt.legend(frameon=False)
    plt.ylim(1e-3, 1e0)
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return slope

def main():
    outdir = Path("figures")
    outdir.mkdir(exist_ok=True)

    I = read_fortran_2d_array("input/test_Kolm_L512V_L512_I")
    Q = read_fortran_2d_array("input/test_Kolm_L512V_L512_Q")
    U = read_fortran_2d_array("input/test_Kolm_L512V_L512_U")
    ang = read_fortran_2d_array("input/test_Kolm_L512V_L512_ang")

    if np.max(np.abs(ang)) > np.pi:
        ang_vec = np.radians(ang)
    else:
        ang_vec = ang

    ang_stokes = 0.5 * np.arctan2(U, Q)

    N = I.shape[0]
    Rs = np.unique(np.logspace(0, np.log10(N // 3), 20).astype(int))

    slope_stokes = plot_structure(
        ang_stokes,
        Rs,
        title="Structure Function – Stokes Azimuth",
        outfile=outdir / "2d_structure_stokes.png",
        expected_slope=5 / 3,
    )

    slope_vector = plot_structure(
        ang_vec,
        Rs,
        title="Structure Function – Vector Azimuth",
        outfile=outdir / "2d_structure_vector.png",
        expected_slope=5 / 3,
    )

    print(f"Stokes azimuth : {slope_stokes:.3f}")
    print(f"Vector azimuth : {slope_vector:.3f}\n")

if __name__ == "__main__":
    main()
