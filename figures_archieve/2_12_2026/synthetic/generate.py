from __future__ import annotations
import numpy as np
import pyvista as pv

# ============================================================
# Settings
# ============================================================
N = 256
L = 1.0
SEED = 12345

# Target 3D spectral slopes (power spectra)
RHO_SLOPE = -2.26
B_SLOPE   = -(11.0 / 3.0)

# Output files
W_FILE   = "synthetic_w.vtk"
BCC_FILE = "synthetic_bcc.vtk"

# Required Athena stats (must match)
TARGET = {
    "dens": {"mean": 1.000000e+00, "std": 1.972260e+00},
    "bcc3": {"mean": 1.250005e+01, "std": 2.164334e+00},
    "bcc1": {"mean": 4.648027e-07, "std": 3.886881e+00},
    "bcc2": {"mean": -9.316136e-08, "std": 3.751195e+00},
}

# ============================================================
# Helpers
# ============================================================
def kgrid(N: int, L: float):
    """Return kx,ky,kz,|k| with k in rad/unit (2π/L convention)."""
    dx = L / N
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kk = np.sqrt(kx*kx + ky*ky + kz*kz)
    return kx, ky, kz, kk

def amp_filter_from_power_slope(kk: np.ndarray, slope: float):
    """
    Want P(k) ~ k^{slope}. For a Gaussian field, |F(k)|^2 ~ P(k),
    so amplitude multiplier is ~ k^{slope/2}.
    """
    filt = np.zeros_like(kk, dtype=np.float64)
    m = kk > 0
    filt[m] = kk[m] ** (0.5 * slope)
    return filt

def gaussian_field_with_spectrum(N: int, L: float, slope: float, rng: np.random.Generator):
    """Zero-mean Gaussian field with approximate isotropic P(k) ~ k^{slope}."""
    noise = rng.normal(size=(N, N, N)).astype(np.float64)
    F = np.fft.fftn(noise)

    _, _, _, kk = kgrid(N, L)
    filt = amp_filter_from_power_slope(kk, slope)

    F *= filt
    field = np.fft.ifftn(F).real
    # enforce exact zero mean (k=0 mode can drift slightly numerically)
    field -= field.mean()
    return field

def rescale_to_mean_std(x: np.ndarray, mean_t: float, std_t: float):
    """Affine rescale to exact target mean and (population) std."""
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        raise ValueError("Field std is zero; cannot rescale.")
    y = (x - m) * (std_t / s) + mean_t

    # one more tiny correction pass to kill floating drift
    y = y - y.mean() + mean_t
    y = (y - mean_t) * (std_t / y.std(ddof=0)) + mean_t
    return y

def lognormal_from_gaussian(g: np.ndarray, mean_t: float, std_t: float):
    """
    If g ~ N(0,1) then rho = exp(mu + sigma*g) is lognormal.
    Choose mu,sigma so that E[rho]=mean_t and Std[rho]=std_t.
    """
    g = np.asarray(g, dtype=np.float64)
    # normalize g to exactly mean 0 std 1 (population)
    g = g - g.mean()
    g = g / g.std(ddof=0)

    cv2 = (std_t / mean_t) ** 2
    sigma2 = np.log(1.0 + cv2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean_t) - 0.5 * sigma2

    rho = np.exp(mu + sigma * g)

    # finite-sample correction to match exactly
    rho = rescale_to_mean_std(rho, mean_t, std_t)

    # keep strictly positive after tiny affine corrections (should already be)
    # If numerical correction ever makes a negative value (very unlikely), shift minimally:
    rmin = rho.min()
    if rmin <= 0:
        rho = rho + (1e-12 - rmin)
        rho = rescale_to_mean_std(rho, mean_t, std_t)

    return rho

def flatten_cell_F(arr3: np.ndarray) -> np.ndarray:
    """VTK cell_data expects flat arrays; Fortran order matches your reader."""
    return np.asarray(arr3, dtype=np.float64).ravel(order="F")

def make_grid(N: int, L: float) -> pv.ImageData:
    dx = L / N
    g = pv.ImageData()
    g.dimensions = (N + 1, N + 1, N + 1)  # point dims for N^3 cells
    g.origin = (0.0, 0.0, 0.0)
    g.spacing = (dx, dx, dx)
    return g

def print_stats(name: str, arr: np.ndarray):
    print(f"{name}: mean={arr.mean():.6e}  std={arr.std(ddof=0):.6e}")

# ============================================================
# Main
# ============================================================
def main():
    rng = np.random.default_rng(SEED)

    # -------------------------
    # Density: slope -2.26, then lognormal -> match Athena stats
    # -------------------------
    g_rho = gaussian_field_with_spectrum(N, L, RHO_SLOPE, rng)
    dens = lognormal_from_gaussian(
        g_rho,
        mean_t=TARGET["dens"]["mean"],
        std_t=TARGET["dens"]["std"],
    )

    # -------------------------
    # Magnetic components: each with slope -5/3, then match Athena stats
    # -------------------------
    bx0 = gaussian_field_with_spectrum(N, L, B_SLOPE, rng)
    by0 = gaussian_field_with_spectrum(N, L, B_SLOPE, rng)
    bz0 = gaussian_field_with_spectrum(N, L, B_SLOPE, rng)

    # Athena naming (as you used earlier): bcc1=By, bcc2=Bz, bcc3=Bx
    bcc3 = rescale_to_mean_std(bx0, TARGET["bcc3"]["mean"], TARGET["bcc3"]["std"])
    bcc1 = rescale_to_mean_std(by0, TARGET["bcc1"]["mean"], TARGET["bcc1"]["std"])
    bcc2 = rescale_to_mean_std(bz0, TARGET["bcc2"]["mean"], TARGET["bcc2"]["std"])

    # -------------------------
    # Print stats (should match)
    # -------------------------
    print_stats("dens", dens)
    print_stats("bcc3", bcc3)
    print_stats("bcc1", bcc1)
    print_stats("bcc2", bcc2)

    # # -------------------------
    # # Write synthetic_w.vtk (dens)
    # # -------------------------
    # w_grid = make_grid(N, L)
    # w_grid.cell_data["dens"] = flatten_cell_F(dens)
    # w_grid.save(W_FILE)

    # # -------------------------
    # # Write synthetic_bcc.vtk (bcc1,bcc2,bcc3)
    # # -------------------------
    # bcc_grid = make_grid(N, L)
    # bcc_grid.cell_data["bcc1"] = flatten_cell_F(bcc1)
    # bcc_grid.cell_data["bcc2"] = flatten_cell_F(bcc2)
    # bcc_grid.cell_data["bcc3"] = flatten_cell_F(bcc3)
    # bcc_grid.save(BCC_FILE)

    # print("Wrote:")
    # print(f"  {W_FILE}")
    # print(f"  {BCC_FILE}")

if __name__ == "__main__":
    main()
