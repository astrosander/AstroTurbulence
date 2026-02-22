#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce LP16 (Lazarian & Pogosyan 2016) Figure 5 using exact paper equations
in the limit of negligible Faraday rotation.

Uses:
  - Eq. (30): intrinsic polarization correlation model xi_i
  - Eq. (44): observed polarization correlation (with Faraday term neglected)
  - Eq. (45): small-R asymptotic structure-function slope
  - Eq. (46): projected correlation length R_P

No surrogate curve fits. No interpolation of the plotted functions.
Only exact algebraic reduction + numerical quadrature of the paper equations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Avoid PySide6 / Qt backend issues
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import textwrap

# NumPy compatibility (np.trapezoid in newer NumPy, np.trapz in older)
TRAPZ = getattr(np, "trapezoid", np.trapz)


# ============================================================
# Exact paper equations
# ============================================================

def xi_i_eq30(R, dz, r_i, m, sigma_i=1.0, Pbar_i=0.0):
    r"""
    Eq. (30), saturated isotropic power-law model for intrinsic polarization correlation:
        xi_i(R, dz) = |Pbar_i|^2 + sigma_i^2 * r_i^m / [ r_i^m + (R^2 + dz^2)^{m/2} ]

    For Figure 5 turbulence fluctuations, we use Pbar_i = 0 (default).
    """
    return (abs(Pbar_i) ** 2
            + sigma_i**2 * (r_i ** m) / (r_i ** m + (R * R + dz * dz) ** (m / 2.0)))


def corr_P_eq44_negligible_faraday(R_values, r_i, m, L, sigma_i=1.0, Pbar_i=0.0, z_grid=None):
    r"""
    Eq. (44) in the negligible-Faraday limit:
        <P(X1) P*(X2)> = \int_0^L dΔz (L-Δz) xi_i(R, Δz)

    This is the exact weak/no-Faraday expression used for Figure 5.
    """
    if z_grid is None:
        z_grid = make_z_grid(L)

    z = z_grid
    w = (L - z)[None, :]              # shape (1, nz)
    R = np.asarray(R_values, float)[:, None]  # shape (nR, 1)
    zz = z[None, :]                   # shape (1, nz)

    xi = xi_i_eq30(R, zz, r_i=r_i, m=m, sigma_i=sigma_i, Pbar_i=Pbar_i)
    return TRAPZ(w * xi, z, axis=1)


def D_P_from_corr(R_values, r_i, m, L, sigma_i=1.0, Pbar_i=0.0, z_grid=None):
    r"""
    Structure function of complex polarization:
        D_P(R) = < |P1 - P2|^2 > = 2 [ C_P(0) - C_P(R) ]
    where C_P(R) = <P(X1)P*(X2)>.

    (This is the standard exact identity for a complex field.)
    """
    if z_grid is None:
        z_grid = make_z_grid(L)

    C_R = corr_P_eq44_negligible_faraday(R_values, r_i, m, L, sigma_i, Pbar_i, z_grid)
    C_0 = corr_P_eq44_negligible_faraday(np.array([0.0]), r_i, m, L, sigma_i, Pbar_i, z_grid)[0]
    D_R = 2.0 * (C_0 - C_R)
    D_inf = 2.0 * C_0  # for Pbar_i=0, C(R→∞)→0
    return D_R, C_0, D_inf


def D_P_asymptotic_eq45(R_values, r_i, m, L, sigma_i=1.0):
    r"""
    Eq. (45), m<1 form:
        D_P(R) ~ sigma_i^2 * L * R * (R/r_i)^m
    (equivalent to the first expression in Eq. 45 with R_P from Eq. 46)
    """
    mbar = min(float(m), 1.0)
    return sigma_i**2 * L * R_values * (R_values / r_i) ** mbar


def R_P_eq46(r_i, m, L):
    r"""
    Eq. (46), observed/projection correlation length (for m<1):
        R_P ~ r_i (L/r_i)^((1-mbar)/(1+mbar)),  mbar=min(m,1)
    """
    mbar = min(float(m), 1.0)
    return r_i * (L / r_i) ** ((1.0 - mbar) / (1.0 + mbar))


# ============================================================
# Numerics (quadrature grid only; no interpolation)
# ============================================================

def make_z_grid(L,
                n_log1=2200, n_log2=2600, n_lin=3200,
                zmin=1e-12, zmid1=1e-4, zmid2=1.0):
    """
    Nonuniform quadrature grid on Δz in [0, L], dense near zero (for small-R behavior),
    then linear to L. This is still direct quadrature, not interpolation.
    """
    z1 = np.geomspace(zmin, zmid1, n_log1)
    z2 = np.geomspace(zmid1, zmid2, n_log2)
    z3 = np.linspace(zmid2, L, n_lin)
    z = np.unique(np.concatenate(([0.0], z1, z2, z3)))
    return z


# ============================================================
# Figure 5 recreation
# ============================================================

def reproduce_figure5_exact(
    out_png="Figure5_exact_equations.png",
    out_pdf="Figure5_exact_equations.svg",
    L_over_ri=100.0,
    r_i=1.0,
    sigma_i=1.0,
    Pbar_i=0.0,
    left_norm_factor=2.0,      # plot left panel as D_P / (left_norm_factor * sigma_i^2 * L^2)
    left_asymptote_offset=3.0, # caption says asymptote is offset for clarity
    nR_left=220,
    nR_right=260
):
    """
    Reproduces Figure 5 with exact Eq. (30)+(44), asymptote Eq. (45), and R_P markers Eq. (46).

    Notes on normalization:
      - Left panel in the paper does not explicitly annotate units on the axis label.
      - Using left_norm_factor=2 gives a visual match to the scanned figure amplitude.
      - Set left_norm_factor=1 if you prefer D_P/(sigma_i^2 L^2).
    """
    L = L_over_ri * r_i
    z = make_z_grid(L)

    # Common x-axis variable shown in the figure: x = R/r_i
    x_left = np.logspace(-4, 2, nR_left)
    R_left = x_left * r_i

    x_right = np.logspace(-4, 2, nR_right)
    R_right = x_right * r_i

    # -----------------
    # Left panel: m = 2/3
    # -----------------
    m_left = 2.0 / 3.0
    D_left, C0_left, Dinf_left = D_P_from_corr(R_left, r_i=r_i, m=m_left, L=L,
                                               sigma_i=sigma_i, Pbar_i=Pbar_i, z_grid=z)

    # Exact Eq. (45) asymptotic (offset for clarity as in caption)
    D_asym_left = D_P_asymptotic_eq45(R_left, r_i=r_i, m=m_left, L=L, sigma_i=sigma_i)

    # Left-panel normalization (to visually match the published figure)
    left_denom = left_norm_factor * sigma_i**2 * L**2
    y_left = D_left / left_denom
    y_asym_left = (left_asymptote_offset * D_asym_left) / left_denom

    # -----------------
    # Right panel: m = 1/3, 1/2, 4/5 (normalized by D_P(infty))
    # -----------------
    m_list = [1/3, 1/2, 4/5]  # longest to shortest R_P (blue, orange, green in the scan)
    colors = ["#5B7DB1", "#D9901A", "#88A92A"]  # blue, orange, green
    curves_right = []
    Rp_markers = []

    for m in m_list:
        D, C0, Dinf = D_P_from_corr(R_right, r_i=r_i, m=m, L=L,
                                    sigma_i=sigma_i, Pbar_i=Pbar_i, z_grid=z)
        curves_right.append(D / Dinf)             # exact normalization by asymptotic value
        Rp_markers.append(R_P_eq46(r_i, m, L) / r_i)

    # ========================================================
    # Plot styling to match the paper
    # ========================================================
    fig = plt.figure(figsize=(13.8, 5.8))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.90, bottom=0.28, wspace=0.28)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    for ax in (axL, axR):
        # ax.set_facecolor("#efefef")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1e2)
        ax.set_ylim(1e-8, 2)
        ax.tick_params(which="both", direction="in", labelsize=14, color="0.35")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # Left panel curves
    axL.loglog(x_left, y_left, color="black", lw=2.2, zorder=5)

    # Asymptotic dotted slope (offset for clarity), shown only in the small-R regime
    mask_small = x_left <= 0.1
    # Use marker-only dotted style to mimic the scanned figure look
    idx = np.where(mask_small)[0][::4]
    axL.plot(x_left[idx], y_asym_left[idx], linestyle="None", marker="o",
             ms=5.0, color="#5F84B8", zorder=4)

    # Left panel annotations
    axL.text(-0.08, 1.05, r"$D_P(\mathrm{R})$", transform=axL.transAxes, fontsize=28)
    axL.text(1.03, -0.02, r"$\mathrm{R}/r_i$", transform=axL.transAxes, fontsize=28)
    axL.text(0.11, 0.55, r"$\sim \mathbf{R}^{1+\bar m_i}$", transform=axL.transAxes, fontsize=24)

    # Right panel curves and R_P markers
    for y, c in zip(curves_right, colors):
        axR.loglog(x_right, y, color=c, lw=2.2, zorder=4)

    # y = 1/2 line (definition of observed correlation length R_P in text)
    axR.hlines(0.5, 1.0, 100.0, colors="black", lw=2.2, zorder=3)

    # Vertical markers at R_P / r_i from Eq. (46)
    for xrp, c in zip(Rp_markers, colors):
        axR.vlines(xrp, 1e-8, 1e-5, colors=c, lw=2.0, zorder=2)

    # Right panel annotations
    axR.text(-0.17, 1.05, r"$D_P(\mathrm{R})/D_P(\infty)$", transform=axR.transAxes, fontsize=28)
    axR.text(1.03, -0.02, r"$\mathrm{R}/r_i$", transform=axR.transAxes, fontsize=28)
    axR.text(0.06, 0.45, r"$\sim \mathbf{R}^{1+\bar m_i}$", transform=axR.transAxes, fontsize=24)
    axR.text(0.63, 0.47, r"$r_P/r_i$", transform=axR.transAxes, fontsize=22)


    # Save
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Diagnostics / sanity checks
    x_intersect_pred = r_i / L
    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print(f"Predicted normalized-curve intersection from Eq.(45): R/r_i ≈ r_i/L = {x_intersect_pred:.6g}")
    for m, xrp in zip(m_list, Rp_markers):
        print(f"Eq.(46): m={m:.6g} -> R_P/r_i = {xrp:.6g}")


if __name__ == "__main__":
    reproduce_figure5_exact(
        out_png="Figure5_exact_equations.png",
        out_pdf="Figure5_exact_equations.svg",
        L_over_ri=100.0,
        r_i=1.0,
        sigma_i=1.0,
        Pbar_i=0.0,            # turbulent fluctuations only (Figure 5 use case)
        left_norm_factor=2.0,  # visually matches the scanned figure; set to 1.0 if desired
        left_asymptote_offset=3.0
    )