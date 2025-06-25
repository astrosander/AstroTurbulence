#!/usr/bin/env python3
"""
Faraday-screen angle statistics – real-cube version (robust spacing)
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from numpy.fft import rfftn, irfftn, fftshift


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _axis_spacing(coord_array, name=""):
    """Return median positive spacing or 1.0 if not possible."""
    c = np.unique(coord_array.ravel())
    diffs = np.diff(np.sort(c))
    diffs = diffs[diffs > 0]
    if diffs.size:
        return float(np.median(diffs))
    print(f" [!] {name}: could not determine spacing – using 1")
    return 1.0


def structure_function_2d(field, nbins=100, r_max=None, dx=1.0):
    f = field - field.mean()
    power = np.abs(rfftn(f)) ** 2
    ac = irfftn(power, s=f.shape) / f.size
    ac = fftshift(ac)
    D = 2.0 * f.var() - 2.0 * ac
    D[D < 0] = 0

    iy, ix = np.indices(f.shape)
    iy -= f.shape[0] // 2
    ix -= f.shape[1] // 2
    R = np.sqrt(ix**2 + iy**2) * dx
    if r_max is None:
        r_max = R.max() / 2

    bins = np.linspace(0, r_max, nbins + 1)
    D_R, _, _ = binned_statistic(R.ravel(), D.ravel(),
                                 statistic="mean", bins=bins)
    R_cent = 0.5 * (bins[1:] + bins[:-1])
    return R_cent, D_R


def angle_structure_function(D_phi, lam):
    return 0.5 * (1.0 - np.exp(-2.0 * lam**4 * D_phi))


# ----------------------------------------------------------------------
def main(cube_file="ms01ma08.mhd_w.00300.vtk.h5",
         ne_key="gas_density", bz_key="k_mag_field",
         lam_list=(0.06, 0.11, 0.21)):
    cube_file = Path(cube_file)
    with h5py.File(cube_file, "r") as f:
        ne = f[ne_key][:]
        bz = f[bz_key][:]

        dx = _axis_spacing(f["x_coor"][:, 0, 0], "x_coor") if "x_coor" in f else 1.0
        dz = _axis_spacing(f["z_coor"][0, 0, :], "z_coor") if "z_coor" in f else 1.0

    print("   cube shape :", ne.shape)
    print("   dx, dz     :", dx, dz)

    Phi = (ne * bz).sum(axis=2) * dz
    sigma_phi = Phi.std()
    if sigma_phi == 0:
        print(" [!] Warning: Φ has zero variance – check B-field component!")

    R, Dphi = structure_function_2d(Phi, dx=dx)

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(R, Dphi, label="simulation")
    ax[0].loglog(R, Dphi[1]*(R/R[1])**(5/3), "--", lw=0.8,
                 label=r"$\propto R^{5/3}$")
    ax[0].set(xlabel="R", ylabel=r"$D_\Phi$", title="RM structure")
    ax[0].legend()

    for lam in lam_list:
        ax[1].loglog(R, angle_structure_function(Dphi, lam),
                     label=rf"$\lambda={lam:.2f}$ m")
    ax[1].set(xlabel="R", ylabel=r"$D_\varphi$", title="Angle structure")
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
