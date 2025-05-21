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

def plot_structure_dual(Rs: np.ndarray, D1: np.ndarray, D2: np.ndarray, 
                        labels: Tuple[str, str], title: str, outfile: Path,
                        expected_slope: float | None = None) -> Tuple[float, float]:
    half = len(Rs) // 2
    slope1, *_ = linregress(np.log(Rs[:half]), np.log(D1[:half]))
    slope2, *_ = linregress(np.log(Rs[:half]), np.log(D2[:half]))

    plt.figure(figsize=(5.5, 4))

    plt.loglog(Rs, D1, "-", label=f"{labels[0]} (slope={slope1:.2f})")
    plt.loglog(Rs, D2, "-", label=f"{labels[1]} (slope={slope2:.2f})")

    # Add a reference line with the expected slope
    if expected_slope is not None:
        C = D1[1] / (Rs[1] ** expected_slope)
        plt.loglog(Rs, C * Rs ** expected_slope, "--", color="black",
                   label=f"$\\propto R^{{5/3}}$")

    plt.xlabel("R [pixels]")
    plt.ylabel("D(R)")
    plt.title(title, fontsize=13)
    plt.legend(frameon=False)
    plt.ylim(1e-3, 1e0)
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return slope1, slope2

def main():
    outdir = Path("figures")
    outdir.mkdir(exist_ok=True)

    I = read_fortran_2d_array("stokesMaps_velocity_kolmogorov_NoNorm/input/synchrotron/test_Kolm_L512V_L512_I")
    Q = read_fortran_2d_array("stokesMaps_velocity_kolmogorov_NoNorm/input/synchrotron/test_Kolm_L512V_L512_Q")
    U = read_fortran_2d_array("stokesMaps_velocity_kolmogorov_NoNorm/input/synchrotron/test_Kolm_L512V_L512_U")
    ang = read_fortran_2d_array("stokesMaps_velocity_kolmogorov_NoNorm/input/synchrotron/test_Kolm_L512V_L512_ang")

    if np.max(np.abs(ang)) > np.pi:
        ang_vec = np.radians(ang)
    else:
        ang_vec = ang

    ang_stokes = 0.5 * np.arctan2(U, Q)

    N = I.shape[0]
    Rs = np.unique(np.logspace(0, np.log10(N // 3), 20).astype(int))

    Rs1, D1 = structure_function_2d(ang_stokes, Rs)
    Rs2, D2 = structure_function_2d(ang_vec, Rs)

    slope_stokes, slope_vector = plot_structure_dual(
        Rs, D1, D2,
        labels=("Stokes Azimuth", "Vector Azimuth"),
        title="Structure Function – Azimuth Comparison",
        outfile=outdir / "2d_structure_comparison.png",
        expected_slope=5 / 3,
    )

    print(f"Stokes azimuth : {slope_stokes:.3f}")
    print(f"Vector azimuth : {slope_vector:.3f}\n")


if __name__ == "__main__":
    main()
