#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lazarian & Pogosyan (2016) Figure 9 recreation using EXACT paper equations
(Eq. 14, Eq. 30, Eq. 90, Eq. 93), with fast vectorized numerics.

No surrogate/fitted curves. No heuristic shape functions.
This is an exact-analytics + numerical quadrature implementation.

IMPORTANT:
- Uses a non-interactive Matplotlib backend ("Agg") to avoid PySide6/Shiboken issues.
- Eq. (93) (as printed) corresponds to C(0)-C(R); the standard structure-function identity is
      D(R) = 2 [C(0) - C(R)].
  The code supports both conventions (see FULL_MODE).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # <- avoids Qt/PySide backend errors
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, FixedLocator
import textwrap


# -----------------------------
# NumPy compatibility
# -----------------------------
TRAPZ = getattr(np, "trapezoid", np.trapz)


# -----------------------------
# Fast cumulative trapezoid (vectorized along axis)
# -----------------------------
def cumtrapz0_axis(y, x, axis=-1):
    """
    Cumulative trapezoid with zero initial value.
    Returns array with same shape as y, where out[...,0]=0 along the integration axis.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    dx = np.diff(x)
    shp = [1] * y.ndim
    shp[axis] = dx.size
    dx = dx.reshape(shp)

    sl_lo = [slice(None)] * y.ndim
    sl_hi = [slice(None)] * y.ndim
    sl_lo[axis] = slice(None, -1)
    sl_hi[axis] = slice(1, None)

    mids = 0.5 * (y[tuple(sl_lo)] + y[tuple(sl_hi)]) * dx
    c = np.cumsum(mids, axis=axis)

    out = np.zeros_like(y, dtype=float)
    sl_out = [slice(None)] * y.ndim
    sl_out[axis] = slice(1, None)
    out[tuple(sl_out)] = c
    return out


# -----------------------------
# Eq. (14), Eq. (30) model kernels
# -----------------------------
def xhat_saturated_powerlaw(R, dz, r, m):
    r"""
    Normalized correlation model (Eq. 14 / Eq. 30 fluctuating part):
        \hat{\xi}(R,\Delta z) = r^m / [ r^m + (R^2 + \Delta z^2)^{m/2} ]
    """
    p = (R * R + dz * dz) ** (m / 2.0)
    return (r ** m) / (r ** m + p)


def Dhat_from_xhat(xhat):
    r"""
    \hat D = D/(2\sigma^2) = 1 - \hat\xi
    """
    return 1.0 - xhat


# -----------------------------
# Exact algebraic reductions (no approximations)
# -----------------------------
def T_even_batch(f, z):
    r"""
    For even kernel f(|u-v|):
        T_f(z) = \int_0^z du \int_0^z dv f(|u-v|)
               = 2 \int_0^z (z-s) f(s)\,ds
    Batch version: f shape (nR, nz), returns T_f shape (nR, nz).
    """
    H = cumtrapz0_axis(f, z, axis=1)                # H(z) = ∫_0^z f(s) ds
    J = cumtrapz0_axis(f * z[None, :], z, axis=1)   # J(z) = ∫_0^z s f(s) ds
    return 2.0 * (z[None, :] * H - J)


def mirror_interp_rows(Y, z, L):
    """
    Row-wise interpolation of Y(z) evaluated at (L - z), for nonuniform z.
    Y shape = (nR, nz)
    Returns shape = (nR, nz)
    """
    xq = L - z
    out = np.empty_like(Y)
    for i in range(Y.shape[0]):
        out[i] = np.interp(xq, z, Y[i])
    return out


def Q_nested_batch(g, f, L, z):
    r"""
    Exact reduction of the Eq. (90)-type nested integral:

        Q[g,f] = ∫_0^L dz1 ∫_0^L dz2 g(|z1-z2|)
                 ∫_0^{z1} du ∫_0^{z2} dv f(|u-v|)

    using:
        J_f(z1,z2) = 1/2 [ T_f(z1) + T_f(z2) - T_f(|z1-z2|) ].

    Batch version for arrays g,f of shape (nR, nz), where the nz axis is z-grid.
    """
    T_f = T_even_batch(f, z)              # (nR, nz)
    H_g = cumtrapz0_axis(g, z, axis=1)    # H_g(z)=∫_0^z g(s) ds
    S_g = H_g + mirror_interp_rows(H_g, z, L)  # H_g(z) + H_g(L-z)

    integrand = T_f * (S_g - (L - z)[None, :] * g)
    return TRAPZ(integrand, z, axis=1)


def intrinsic_linear_weight(L, dz):
    r"""
    Exact weight for Eq. (93) intrinsic linear term:
        ∫_0^L dz1 ∫_0^L dz2 z1 z2 A(|z1-z2|)
      = ∫_0^L dΔ A(Δ) [ 2L^3/3 - L^2Δ + Δ^3/3 ].
    """
    return (2.0 * L**3) / 3.0 - (L**2) * dz + (dz**3) / 3.0


# -----------------------------
# Figure 9 exact quantities (Eq. 90 + Eq. 93)
# -----------------------------
def figure9_terms_exact_batch(ri, m_i, rphi, m_phi, L, R_values, z):
    """
    Compute the exact weak-Faraday (Eq. 90 / Eq. 93) curves in batch over R.

    Returns dict with:
      - C_eq90               : correlation from Eq. (90)
      - D_full_eq90_std      : 2*(C(0)-C(R))  [standard structure-function identity]
      - D_full_eq93_printed  : Eq. (93) printed RHS convention = C(0)-C(R)
      - D_intr_linear_eq93   : intrinsic linear contribution from Eq. (93)
      - D_far_linear_eq93    : Faraday linear contribution from Eq. (93)
      - D_quad_eq93          : quadratic correction term from Eq. (93)
    """
    R = np.asarray(R_values, dtype=float)[:, None]   # (nR,1)
    zz = z[None, :]                                  # (1,nz)

    # Correlation kernels from Eq. (14), Eq. (30) (fluctuating part)
    x_i = xhat_saturated_powerlaw(R, zz, ri, m_i)
    x_f = xhat_saturated_powerlaw(R, zz, rphi, m_phi)
    D_i = Dhat_from_xhat(x_i)
    D_f = Dhat_from_xhat(x_f)

    # R = 0 kernels
    x0_i = xhat_saturated_powerlaw(np.array([[0.0]]), zz, ri, m_i)[0]
    x0_f = xhat_saturated_powerlaw(np.array([[0.0]]), zz, rphi, m_phi)[0]
    D0_i = Dhat_from_xhat(x0_i)
    D0_f = Dhat_from_xhat(x0_f)

    # Eq. (90): correlation
    C_R = Q_nested_batch(x_i, x_f, L, z)
    C_0 = Q_nested_batch(x0_i[None, :], x0_f[None, :], L, z)[0]

    # Standard complex structure function identity:
    #   <|A1-A2|^2> = 2 [C(0)-C(R)]
    D_std = 2.0 * (C_0 - C_R)

    # Eq. (93): printed decomposition (RHS convention = C(0)-C(R))
    A_i = D_i - D0_i[None, :]
    A_f = D_f - D0_f[None, :]

    # Intrinsic linear term
    W_intr = intrinsic_linear_weight(L, z)
    D_intr = TRAPZ(A_i * W_intr[None, :], z, axis=1)

    # Faraday linear term
    T_Af = T_even_batch(A_f, z)
    D_far = TRAPZ(z[None, :] * T_Af, z, axis=1)

    # Quadratic term
    q_R = Q_nested_batch(D_i, D_f, L, z)
    q_0 = Q_nested_batch(D0_i[None, :], D0_f[None, :], L, z)[0]
    D_quad = q_R - q_0

    # Printed Eq. (93) RHS
    D_eq93 = D_intr + D_far - D_quad

    return {
        "C_eq90": C_R,
        "D_full_eq90_std": D_std,
        "D_full_eq93_printed": D_eq93,
        "D_intr_linear_eq93": D_intr,
        "D_far_linear_eq93": D_far,
        "D_quad_eq93": D_quad,
        "C0_eq90": C_0,
    }


# -----------------------------
# Utility functions for panel setup
# -----------------------------
def projected_correlation_length(r, m, L):
    """
    Eq. (45)-style projected correlation scale used for vertical markers:
        R_P ~ r (L/r)^((1-\bar m)/(1+\bar m)),   \bar m = min(m,1)
    """
    mbar = min(float(m), 1.0)
    return r * (L / r) ** ((1.0 - mbar) / (1.0 + mbar))


def classify_lengths(ri, mi, rf, mf):
    """
    Paper notation:
      r_m = min(ri,rf), r_M = max(ri,rf),
      m_m and m_M are the slopes corresponding to r_m and r_M.
    """
    if ri < rf:
        return ri, mi, rf, mf
    if rf < ri:
        return rf, mf, ri, mi
    # Degenerate lengths: keep exponent ordering consistent for annotation
    if mi <= mf:
        return ri, mi, rf, mf
    return rf, mf, ri, mi


def make_nonuniform_z_grid(L,
                           n_log1=2500, n_log2=2500, n_lin=3500,
                           zmin=1e-10, zmid1=1e-4, zmid2=0.5):
    """
    Nonuniform z-grid (dense near 0 to resolve cusp behavior at small R,
    then linear to L). This dramatically improves convergence and speed.
    """
    z1 = np.geomspace(zmin, zmid1, n_log1)
    z2 = np.geomspace(zmid1, zmid2, n_log2)
    z3 = np.linspace(zmid2, L, n_lin)
    z = np.unique(np.concatenate(([0.0], z1, z2, z3)))
    return z


# -----------------------------
# Plotting (Figure 9 style)
# -----------------------------
def draw_figure9_exact(output_png="Figure9_exact_equations_fast.png",
                       output_pdf="Figure9_exact_equations_fast.svg",
                       nR=180,
                       full_mode="eq93_printed",
                       do_consistency_check=True):
    """
    full_mode:
      - "eq93_printed" : plots Eq. (93) printed convention (C0 - C(R))
      - "eq90_std"     : plots standard structure function 2[C0 - C(R)]
    """
    # Example parameters from caption
    L = 100.0

    # Exact quadrature grid for Δz∈[0,L]
    z = make_nonuniform_z_grid(L)

    # Plot x-axis in units of R/r_M
    x_plot = np.logspace(-5, 2, nR)

    # Panel definitions (exactly the 4 cases in Figure 9)
    panels = [
        # Top-left: m_phi=1/3, m=4/5, r_i=r_M, r_phi=r_m=0.1 r_i
        dict(name="TL", ri=1.0, mi=4/5, rf=0.1, mf=1/3, col="left"),
        # Top-right: m_phi=4/5, m=1/3
        dict(name="TR", ri=1.0, mi=1/3, rf=0.1, mf=4/5, col="right"),
        # Bottom-left: degenerate slopes m_phi=m=2/3
        dict(name="BL", ri=1.0, mi=2/3, rf=0.1, mf=2/3, col="left"),
        # Bottom-right: degenerate lengths r_i=r_phi, m_phi=4/5, m=1/3
        dict(name="BR", ri=1.0, mi=1/3, rf=1.0, mf=4/5, col="right"),
    ]

    # Style
    c_intr = "#E69500"   # orange dotted (intrinsic)
    c_far = "#6B8EC7"    # blue dashed  (Faraday)
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
    axs = axs.ravel()

    # In-panel labels (placed to visually match the paper figure)
    panel_text = {
        "TL": [
            (r"$\sim \mathbf{R}^{1+\bar m_m}$", (0.02, 0.40), 24),
            (r"$\sim \mathbf{R}^{1+\bar m_M}$", (0.34, 0.80), 24),
            (r"$r_m<r_M$",                       (0.44, 0.24), 20),
            (r"$m_M>m_m$",                       (0.42, 0.12), 20),
            (r"$R_P/r_M$",                       (0.72, 0.40), 22),
        ],
        "TR": [
            (r"$\sim \mathbf{R}^{1+\bar m_M}$", (0.03, 0.10), 24),
            (r"$\sim \mathbf{R}^{1+\bar m_m}$", (0.14, 0.60), 24),
            (r"$r_m<r_M$",                       (0.44, 0.24), 20),
            (r"$m_M<m_m$",                       (0.42, 0.12), 20),
            (r"$R_P/r_M$",                       (0.68, 0.40), 22),
        ],
        "BL": [
            (r"$\sim \mathbf{R}^{1+\bar m_m}$", (0.02, 0.34), 24),
            (r"$r_m<r_M$",                       (0.44, 0.24), 20),
            (r"$m_M=m_m$",                       (0.42, 0.12), 20),
            (r"$R_P/r_M$",                       (0.66, 0.40), 22),
        ],
        "BR": [
            (r"$\sim \mathbf{R}^{1+\bar m_M}$", (0.03, 0.08), 24),
            (r"$\sim \mathbf{R}^{1+\bar m_m}$", (0.24, 0.58), 24),
            (r"$r_m=r_M$",                       (0.44, 0.24), 20),
            (r"$m_M<m_m$",                       (0.42, 0.12), 20),
            (r"$R_P/r_M$",                       (0.66, 0.40), 22),
        ],
    }

    consistency_report = []

    for ax, p in zip(axs, panels):
        # Paper notation r_m/r_M and corresponding slopes
        r_m, m_m, r_M, m_M = classify_lengths(p["ri"], p["mi"], p["rf"], p["mf"])

        # Convert plotted x = R/r_M to physical R
        R_vals = x_plot * r_M

        # Exact Eq. 90 / Eq. 93 values
        vals = figure9_terms_exact_batch(
            ri=p["ri"], m_i=p["mi"],
            rphi=p["rf"], m_phi=p["mf"],
            L=L, R_values=R_vals, z=z
        )

        # Choose "full result" convention
        if full_mode == "eq90_std":
            y_full = vals["D_full_eq90_std"].copy()
        elif full_mode == "eq93_printed":
            y_full = vals["D_full_eq93_printed"].copy()
        else:
            raise ValueError("full_mode must be 'eq93_printed' or 'eq90_std'")

        # Individual contributions shown in the paper are the Eq. (93) linear terms
        y_intr = vals["D_intr_linear_eq93"].copy()
        y_far = vals["D_far_linear_eq93"].copy()

        # Convert to caption units: divide by σ_i^2 σ_φ^2 L^3  (σ factors already factored out)
        y_full /= L**3
        y_intr /= L**3
        y_far  /= L**3

        # Column scaling exactly as caption states
        if p["col"] == "left":
            col_scale = (L / r_M) ** m_M
        else:
            col_scale = (L / r_m) ** m_m

        y_full *= col_scale
        y_intr *= col_scale
        y_far  *= col_scale

        # Plot curves
        ax.loglog(x_plot, y_full, color="black", lw=4.0, solid_capstyle="round", zorder=5)
        ax.loglog(x_plot, y_intr, color=c_intr, lw=1.6, ls=(0, (1.5, 2.0)), zorder=3)  # dotted
        ax.loglog(x_plot, y_far,  color=c_far,  lw=1.6, ls="--", zorder=2)              # dashed

        # Projected correlation-length markers
        Rp_i = projected_correlation_length(p["ri"], p["mi"], L) / r_M
        Rp_f = projected_correlation_length(p["rf"], p["mf"], L) / r_M
        ax.vlines(Rp_i, 1e-8, 7e-6, color=c_intr, lw=1.6, linestyles=(0, (1.5, 2.0)))
        ax.vlines(Rp_f, 1e-8, 4e-6, color=c_far,  lw=1.6, linestyles="--")

        # Axes style to match Figure 9
        ax.set_xlim(1e-5, 1e2)
        ax.set_ylim(1e-9, 1e3)#4)
        ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 1e2]))
        ax.yaxis.set_major_locator(FixedLocator([1, 1e-3, 1e-6]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.tick_params(which="both", direction="in", labelsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Paper-like axis labels as free text
        ax.text(-0.10, 1.04, r"$D_{\mathrm{dP}}(R)$", transform=ax.transAxes, fontsize=22)
        ax.text(1.03, -0.02, r"$\mathrm{R}/r_M$", transform=ax.transAxes, fontsize=22)

        # In-panel annotations
        for txt, (tx, ty), fs in panel_text[p["name"]]:
            ax.text(tx, ty, txt, transform=ax.transAxes, fontsize=fs)

        # Consistency check: Eq. (93) printed convention should equal 0.5 * standard Eq. (90) SF
        if do_consistency_check:
            d93 = vals["D_full_eq93_printed"]
            d90_half = 0.5 * vals["D_full_eq90_std"]
            rel = np.max(np.abs(d93 - d90_half) / np.maximum(1.0, np.abs(d93)))
            consistency_report.append((p["name"], rel))

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.23, wspace=0.24, hspace=0.25)

    # Caption (compact but complete enough)
    caption = (
        r"$\bf{Figure\ 9.}$ Structure function of the derivative of the polarization with respect to $\lambda^2$, "
        r"$D_{dP}\equiv\left\langle\left|\frac{dP(X_1)}{d\lambda^2}-\frac{dP(X_2)}{d\lambda^2}\right|^2\right\rangle$, "
        r"in units of $\sigma_i^2\sigma_\phi^2L^3$. Example parameters are $r_i=r_M$, $r_\phi=r_m=0.1\,r_i$, "
        r"$L=100\,r_i$ (except bottom right where $r_i=r_\phi$). Bold solid line is the full result from the exact "
        r"Eq. (90)/Eq. (93) evaluation; dotted and dashed lines are the intrinsic and Faraday linear terms from Eq. (93). "
        r"Vertical lines mark the projected correlation lengths $R_P$."
    )
    # fig.text(0.015, 0.04, "\n".join(textwrap.wrap(caption, width=175)),
    #          ha="left", va="bottom", fontsize=10)

    # Save files
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    if do_consistency_check:
        print("Eq.(93) vs 0.5*Eq.(90) consistency (max relative diff per panel):")
        for name, rel in consistency_report:
            print(f"  {name}: {rel:.3e}")

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    # Choose convention for the solid curve:
    #   "eq93_printed"  -> plots C(0)-C(R) (paper Eq. 93 RHS convention)
    #   "eq90_std"      -> plots 2[C(0)-C(R)] (standard structure-function identity)
    FULL_MODE = "eq93_printed"

    draw_figure9_exact(
        output_png="Figure9_exact_equations_fast.png",
        output_pdf="Figure9_exact_equations_fast.svg",
        nR=180,
        full_mode=FULL_MODE,
        do_consistency_check=True
    )