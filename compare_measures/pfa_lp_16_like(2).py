"""
PFA (Polarization Frequency Analysis) following LP16 (Lazarian & Pogosyan 2016)
from a sub-Alfvénic turbulence cube stored in an HDF5 file.

Two geometries at a grid of wavelengths λ (user may choose λ=1.0 or a range):
1) External Faraday screen (separated regions): emission and Faraday rotation occur
   in different volumes (LP16 Appendix C). Resulting |P(λ)|^2 is independent of λ.
2) Mixed volume: emission and Faraday rotation occur within the same volume
   (LP16 main text); here |P(λ)|^2 generally decreases with λ and exhibits
   asymptotic power laws described in LP16/PFA tests.

This script computes ⟨|P(λ)|^2⟩ vs λ^2 (the PFA “spectrum”) and styles the plot
as in LP16: log–log axes, slope fit, and a marker at λ=1.0.

Author: (you)
"""

from __future__ import annotations
import h5py
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Iterable
import argparse, os

# -----------------------------
# Data I/O and field primitives
# -----------------------------

@dataclass
class FieldKeys:
    bx: str = "i_mag_field"
    by: str = "j_mag_field"
    bz: str = "k_mag_field"
    ne: str = "gas_density"

@dataclass
class PFAConfig:
    faraday_const: float = 1.0   # φ = C * n_e * B_parallel
    lam_grid: Iterable[float] = tuple(np.geomspace(0.05, 5.0, 40))
    lam_mark: float = 1.0        # highlight this λ on the curve
    gamma: float = 2.0           # emissivity exponent (|B_perp|^γ); γ=2 ⇒ P_i = (Bx+iBy)^2
    los_axis: int = 0            # which array axis is LOS integration axis
    voxel_depth: float = 1.0     # Δz in arbitrary units
    detrend_emit: bool = True    # subtract mean of P_emit in separated case before forming variance


def load_fields(h5_path: str, keys: FieldKeys) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        Bx = f[keys.bx][()]
        By = f[keys.by][()]
        Bz = f[keys.bz][()]
        ne = f[keys.ne][()]
    return Bx, By, Bz, ne


# -----------------------------------------
# Synchrotron emissivity & Faraday quantities
# -----------------------------------------

def polarized_emissivity(Bx: np.ndarray, By: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    B_perp2 = Bx**2 + By**2
    eps = np.finfo(B_perp2.dtype).eps
    phase_term = (Bx + 1j * By)**2
    if gamma == 2.0:
        return phase_term
    amp = np.power(np.maximum(B_perp2, eps), 0.5 * (gamma - 2.0))
    return amp * phase_term


def faraday_density(ne: np.ndarray, B_parallel: np.ndarray, C: float = 1.0) -> np.ndarray:
    return C * ne * B_parallel


# -------------------------------------
# P(λ) maps for two geometries
# -------------------------------------

def _move_los(arr: np.ndarray, los_axis: int) -> np.ndarray:
    return np.moveaxis(arr, los_axis, 0)


def P_map_separated(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                    emit_bounds: Optional[Tuple[int, int]] = None,
                    screen_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """External Faraday screen: P = e^{2 i λ^2 Φ_screen(X)} * ∫ Pi dz (no internal rotation).
    Note |P| is independent of λ; we retain λ for completeness and consistency.
    """
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz, Ny, Nx = Pi_los.shape

    if emit_bounds is None:
        emit_bounds = (0, Nz)
    if screen_bounds is None:
        scr_N = max(1, int(0.1 * Nz))
        screen_bounds = (0, scr_N)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    P_emit = np.sum(Pi_los[z0e:z1e, :, :], axis=0) * cfg.voxel_depth
    if cfg.detrend_emit:
        P_emit = P_emit - P_emit.mean()

    Phi_screen = np.sum(phi_los[z0s:z1s, :, :], axis=0) * cfg.voxel_depth
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)


def P_map_mixed(Pi: np.ndarray, phi: np.ndarray, lam: float, cfg: PFAConfig,
                bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Mixed emission & Faraday rotation in same volume: P = ∫ Pi e^{2 i λ^2 Φ(z)} dz."""
    Pi_los = _move_los(Pi, cfg.los_axis)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz, Ny, Nx = Pi_los.shape
    if bounds is None:
        bounds = (0, Nz)
    z0, z1 = bounds

    Phi_cum = np.cumsum(phi_los[z0:z1, :, :] * cfg.voxel_depth, axis=0)
    phase = np.exp(2j * (lam**2) * Phi_cum)
    contrib = Pi_los[z0:z1, :, :] * phase
    P_map = np.sum(contrib, axis=0) * cfg.voxel_depth
    return P_map


# -----------------------------
# PFA: variance vs λ^2
# -----------------------------

def pfa_curve_separated(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                        emit_bounds: Optional[Tuple[int, int]] = None,
                        screen_bounds: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        P = P_map_separated(Pi, phi, lam, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
        var.append(np.mean(np.abs(P)**2))  # ⟨|P|^2⟩ over map (average many LOS)
    return lam_grid**2, np.array(var)


def pfa_curve_mixed(Pi: np.ndarray, phi: np.ndarray, cfg: PFAConfig,
                    bounds: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    lam_grid = np.array(tuple(cfg.lam_grid), dtype=float)
    var = []
    for lam in lam_grid:
        P = P_map_mixed(Pi, phi, lam, cfg, bounds=bounds)
        var.append(np.mean(np.abs(P)**2))
    return lam_grid**2, np.array(var)


def fit_log_slope(x: np.ndarray, y: np.ndarray, frac: Tuple[float, float] = (0.3, 0.8)) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 10:
        return np.nan, np.nan
    i0 = int(frac[0] * xv.size)
    i1 = max(i0 + 10, int(frac[1] * xv.size))
    xv = xv[i0:i1]
    yv = yv[i0:i1]
    if xv.size < 4:
        return np.nan, np.nan
    m, a = np.polyfit(np.log10(xv), np.log10(yv), 1)
    return m, a


# --------------
# High-level driver
# --------------

def compute_pfa_from_file(h5_path: str,
                          keys: FieldKeys = FieldKeys(),
                          cfg: PFAConfig = PFAConfig(),
                          separated: bool = False,
                          emit_bounds: Optional[Tuple[int, int]] = None,
                          screen_bounds: Optional[Tuple[int, int]] = None,
                          mixed_bounds: Optional[Tuple[int, int]] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Load data, build Pi and φ, and compute PFA curve (λ^2 vs ⟨|P|^2⟩)."""
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)

    # Choose B_parallel for Faraday density according to LOS axis
    if cfg.los_axis == 0:
        Bpar = Bz
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bx
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    if separated:
        return pfa_curve_separated(Pi, phi, cfg, emit_bounds=emit_bounds, screen_bounds=screen_bounds)
    else:
        return pfa_curve_mixed(Pi, phi, cfg, bounds=mixed_bounds)


# --------------
# Plotting helper
# --------------

def plot_pfa(lam2: np.ndarray, var_list: list, labels: list, cfg: PFAConfig,
             title: str = "PFA (variance) vs $\\lambda^2$"):
    plt.figure(figsize=(6.5, 5.0))
    for var, lab in zip(var_list, labels):
        plt.loglog(lam2, var, label=lab)

    # Slope guide for the first curve
    if var_list:
        m, a = fit_log_slope(lam2, var_list[0])
        if np.isfinite(m):
            mid = 10**np.nanmedian(np.log10(lam2))
            ymid = 10**(a + m * np.log10(mid))
            xline = np.array([mid/5, mid*5])
            yline = ymid * (xline/mid)**m
            plt.loglog(xline, yline, '--', linewidth=1, label=f"fit slope ≈ {m:.2f}")

    # Mark λ = lam_mark
    lam_mark2 = cfg.lam_mark**2
    for var, lab in zip(var_list, labels):
        # interpolate value at λ^2
        vmark = np.interp(lam_mark2, lam2, var)
        plt.scatter([lam_mark2], [vmark], marker='o')

    plt.xlabel(r"$\lambda^2$ (arb. units)")
    plt.ylabel(r"$\langle |P|^2 \rangle$ (arb. units)")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.2)
    plt.legend()
    plt.tight_layout()


def demo(h5_path: str):
    keys = FieldKeys()
    cfg = PFAConfig(lam_grid=np.geomspace(0.05, 5.0, 50), lam_mark=1.0, los_axis=0)

    # Build Pi and φ once
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)
    Bpar = Bz  # assuming LOS along axis-0 uses Bz as parallel component
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg)
    lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg)

    assert np.allclose(lam2_sep, lam2_mix)
    plot_pfa(lam2_sep, [var_sep, var_mix], ["Separated (screen)", "Mixed (in-situ)"], cfg,
             title="PFA at multiple $\lambda$ (marker at $\lambda=1$)")
    plt.show()



def main():
    """Minimal CLI to plot PFA (variance) versus either λ² or χ≡2σ_Φλ².
    Single external dependency: the HDF5 cube path.
    """
    parser = argparse.ArgumentParser(description="Plot PFA (⟨|P|^2⟩) for mixed and/or separated geometries.")
    parser.add_argument("h5", nargs="?", default=r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
                        help="Path to HDF5 cube with fields: i_mag_field, j_mag_field, k_mag_field, gas_density")
    parser.add_argument("--geometry", choices=["mixed", "separated", "both"], default="both",
                        help="Which geometry to plot")
    parser.add_argument("--los-axis", type=int, default=0, help="LOS integration axis (0,1,2)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Emissivity exponent; γ=2 → (Bx+iBy)^2")
    parser.add_argument("--C", type=float, default=1.0, help="Faraday constant C in φ=C n_e B∥")
    parser.add_argument("--lam2-min", type=float, default=2.5e-3, help="Min λ² for grid")
    parser.add_argument("--lam2-max", type=float, default=25.0, help="Max λ² for grid")
    parser.add_argument("--nlam", type=int, default=50, help="Number of λ points (uniform in λ²)")
    parser.add_argument("--x", choices=["chi", "lam2"], default="chi", help="x-axis units: χ=2σ_Φλ² or λ²")
    parser.add_argument("--chi-max", type=float, default=20.0, help="Clip χ to (0, χ_max)")
    parser.add_argument("--mark-lam", type=float, default=1.0, help="Mark this λ on the curve")
    parser.add_argument("--save", type=str, default="", help="Optional output image path (PNG)")
    args = parser.parse_args()

    # Build config and load cube
    cfg = PFAConfig(
        faraday_const=args.C,
        lam_grid=tuple(np.sqrt(np.linspace(args.lam2_min, args.lam2_max, args.nlam))),
        lam_mark=args.mark_lam,
        gamma=args.gamma,
        los_axis=args.los_axis,
    )
    keys = FieldKeys()

    Bx, By, Bz, ne = load_fields(args.h5, keys)
    Pi = polarized_emissivity(Bx, By, gamma=cfg.gamma)

    # Choose B_parallel consistent with LOS axis
    if cfg.los_axis == 0:
        Bpar = Bx
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)

    # Compute σ_Φ from integrated φ along LOS (Δz = 1/Nz)
    phi_los = _move_los(phi, cfg.los_axis)
    Nz = phi_los.shape[0]
    dz = 1.0 / float(Nz)
    Phi_tot = np.sum(phi_los, axis=0) * dz
    sigma_phi = float(Phi_tot.std()) if Phi_tot.size else 0.0

    # Compute PFA curves according to geometry selection
    lam2 = np.array(cfg.lam_grid) ** 2
    curves = []
    labels = []

    if args.geometry in ("separated", "both"):
        lam2_sep, var_sep = pfa_curve_separated(Pi, phi, cfg)
        curves.append((lam2_sep, var_sep))
        labels.append("Separated (screen)")

    if args.geometry in ("mixed", "both"):
        lam2_mix, var_mix = pfa_curve_mixed(Pi, phi, cfg)
        curves.append((lam2_mix, var_mix))
        labels.append("Mixed (in-situ)")

    # Choose x-axis
    def to_x(lam2_arr: np.ndarray) -> np.ndarray:
        if args.x == "chi":
            return 2.0 * sigma_phi * lam2_arr
        return lam2_arr

    x_label = r"$\chi = 2\sigma_\Phi \lambda^2$" if args.x == "chi" else r"$\lambda^2$"

    # Plot
    plt.figure(figsize=(7.0, 5.2))
    first_x = None
    first_y = None

    for (lam2_arr, var_arr), lab in zip(curves, labels):
        x = to_x(lam2_arr)
        y = var_arr
        if args.x == "chi":
            m = (x > 0) & (x < args.chi_max)
            x, y = x[m], y[m]
        plt.loglog(x, y, label=lab)
        if first_x is None:
            first_x, first_y = x, y

    # Slope guide on the first curve
    if first_x is not None and first_y is not None and first_x.size > 6:
        m_fit, a_fit = fit_log_slope(first_x, first_y, frac=(0.3, 0.8))
        if np.isfinite(m_fit):
            mid = 10 ** (np.nanmedian(np.log10(first_x)))
            ymid = 10 ** (a_fit + m_fit * np.log10(mid))
            xline = np.array([mid / 5, mid * 5])
            yline = ymid * (xline / mid) ** m_fit
            plt.loglog(xline, yline, "--", lw=1.2, label=f"fit slope ≈ {m_fit:.2f}")

    # Mark λ=lam_mark and shade χ∈[1,3]
    if args.x == "chi" and sigma_phi > 0:
        chi_mark = 2.0 * sigma_phi * (cfg.lam_mark ** 2)
        if first_x is not None and (first_x.min() < chi_mark < first_x.max()):
            ymark = np.interp(chi_mark, first_x, first_y)
            plt.scatter([chi_mark], [ymark], s=22, zorder=5)
        plt.axvspan(1.0, 3.0, color="grey", alpha=0.12, lw=0)

    plt.xlabel(x_label)
    plt.ylabel(r"$\langle |P|^2 \rangle$")
    title = "PFA variance vs $\chi$" if args.x == "chi" else "PFA variance vs $\lambda^2$"
    if sigma_phi > 0:
        title += f"   ($\sigma_\Phi$={sigma_phi:.3g})"
    plt.title(title)
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if args.save:
        outdir = os.path.dirname(args.save)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()
