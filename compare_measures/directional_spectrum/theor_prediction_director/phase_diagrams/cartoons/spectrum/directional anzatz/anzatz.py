#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
from matplotlib.colors import LinearSegmentedColormap

# Enable LaTeX rendering for publication quality
# Set LaTeX parameters - matplotlib will handle errors gracefully during plotting
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 32,
    'axes.titlesize': 36,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.titlesize': 40,
    'text.latex.preamble': r'\usepackage{amsmath}'
})

TRAPZ = getattr(np, "trapezoid", np.trapz)

def create_blue_to_red_colormap(n_colors):
    """
    Create a sophisticated blue-to-red colormap for eta values.
    Blue represents small eta, red represents large eta.
    Optimized for ApJ publication quality.
    """
    base_colors = [
        '#1a237e',  # Deep indigo blue (smallest eta)
        '#283593',  # Rich blue
        '#3949ab',  # Medium blue
        '#5c6bc0',  # Soft blue
        '#7986cb',  # Light blue
        '#9fa8da',  # Pale blue
        '#64b5f6',  # Sky blue (transition)
        '#ffc107',  # Amber yellow (transition)
        '#ff9800',  # Orange (warm transition)
        '#ff5722',  # Orange-red (transition to red)
        '#d32f2f',  # Strong red
        '#b71c1c',  # Dark red (largest eta)
    ]
    cmap = LinearSegmentedColormap.from_list('blue_to_red_elite', base_colors, N=512)
    positions = np.linspace(0, 1, n_colors)
    return [cmap(pos) for pos in positions]

def eta_from_lambda(lambda_vals, sigma_phi):
    lam = np.asarray(lambda_vals, float)
    return 2.0 * sigma_phi * lam * lam

def xi_i_director_regularized(R, r_i, m_i):
    t = (np.asarray(R, float) / float(r_i)) ** float(m_i)
    return 1.0 / (1.0 + t)

def Dphi_hat_regularized(R, r_phi, m_phi):
    t = (np.asarray(R, float) / float(r_phi)) ** float(m_phi)
    return t / (1.0 + t)

def xi_u_factorized(R, r_i, m_i, r_phi, m_phi, eta):
    xi_i = xi_i_director_regularized(R, r_i=r_i, m_i=m_i)
    dphi_hat = Dphi_hat_regularized(R, r_phi=r_phi, m_phi=m_phi)
    return xi_i * np.exp(-(float(eta) ** 2) * dphi_hat)

def directional_structure_measures(R, r_i, m_i, r_phi, m_phi, eta):
    xi_u = xi_u_factorized(R, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi, eta=eta)
    D_eta = 0.5 * (1.0 - xi_u)
    D_u = 2.0 * (1.0 - xi_u)
    D_u_half = 1.0 - xi_u
    return xi_u, D_eta, D_u, D_u_half


def local_log_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log(x[m])
    ly = np.log(y[m])
    s = np.gradient(ly, lx)
    return x[m], s

def crossover_R_from_smallR(eta, r_i, m_i, r_phi, m_phi):
    eta = float(eta)
    if eta <= 0.0:
        return np.nan
    dm = float(m_phi) - float(m_i)
    if abs(dm) < 1e-14:
        return np.nan
    val = (float(r_phi) ** float(m_phi)) / (eta * eta * (float(r_i) ** float(m_i)))
    if val <= 0:
        return np.nan
    return val ** (1.0 / dm)

def plot_directional_structure_and_proxy(
    out_png="directional_structure_and_proxy.png",
    out_svg="directional_structure_and_proxy.svg",
    r_i=1.0,
    r_phi=0.1,
    m_i=4.0/5.0,
    m_phi=1.0/3.0,
    eta_list=(0.0, 0.05, 0.09, 0.2, 0.3, 0.5),
    x_min=1e-7,
    x_max=1e2,
    nR=500
):
    x = np.logspace(np.log10(x_min), np.log10(x_max), int(nR))
    R = x * float(r_i)

    # Use blue-to-red colormap
    colors = create_blue_to_red_colormap(len(eta_list))

    # ApJ publication quality figure - large size for maximum impact
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])  # Structure function panel (left)
    ax2 = fig.add_subplot(gs[0, 1])  # Log derivative panel (right)

    curves = []
    for eta, c in zip(eta_list, colors):
        xi_u, D_eta, D_u, D_u_half = directional_structure_measures(
            R=R, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi, eta=eta
        )
        curves.append((eta, D_u_half, c))

    # Plot structure function on left panel
    for eta, D_u_half, c in curves:
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax1.loglog(x, D_u_half, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)

    # Plot log derivative on right panel
    for eta, D_u_half, c in curves:
        xs, sl = local_log_slope(x, D_u_half)
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        ax2.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)

    # Left panel: structure function - reference lines spanning full R range
    xg = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Full range
    yg_i = 3e-9 * (xg / x_min) ** m_i
    yg_f = 8e-8 * (xg / x_min) ** m_phi
    ax1.loglog(xg, yg_i, "-", color="#2D5BFF", lw=3.0, zorder=5, alpha=0.8)
    ax1.loglog(xg, yg_f, "-.", color="#E45756", lw=3.0, zorder=5, alpha=0.8)
    ax1.text(2.2e-2, 1.5e-3, rf"$\propto R^{{m_\Phi}}$", color="#E45756", fontsize=24, rotation=16)
    ax1.text(3.0e-2, 1.5e-2, rf"$\propto R^{{m_i}}$", color="#2D5BFF", fontsize=24, rotation=33)

    eta_star = [e for e in eta_list if e > 0]
    if len(eta_star) > 0:
        eta_ref = eta_star[len(eta_star)//2]
        Rx = crossover_R_from_smallR(eta_ref, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi)
        if np.isfinite(Rx) and Rx > 0:
            xr = Rx / r_i
            ax1.plot([xr], [np.interp(np.log10(xr), np.log10(x), np.log10(curves[len(curves)//2][1]))], alpha=0)
            # ax1.axvline(xr, color="0.2", lw=1.2, ls=":")
            # ax1.text(xr*1.05, 1.7e-5, r"$R_\times$", fontsize=16)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax1.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(2.5)
    ax1.spines["bottom"].set_linewidth(2.5)
    ax1.xaxis.set_major_formatter(LogFormatterMathtext())
    ax1.yaxis.set_major_formatter(LogFormatterMathtext())

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(1e-13, 1e1)
    ax1.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax1.yaxis.set_major_locator(FixedLocator([1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]))
    ax1.text(-0.10, 1.05, r"$D_u(R)/2=1-\xi_u$", transform=ax1.transAxes, fontsize=36, weight='bold')
    ax1.text(1.03, -0.02, r"$R/r_i$", transform=ax1.transAxes, fontsize=32, weight='bold')

    ax1.text(0.03, 0.92, rf"$m_i={m_i:.3g},\ m_\Phi={m_phi:.3g}$", transform=ax1.transAxes, fontsize=20)
    ax1.text(0.03, 0.84, rf"$r_\phi/r_i={r_phi/r_i:.3g}$", transform=ax1.transAxes, fontsize=20)
    ax1.text(0.03, 0.76, r"$\xi_u=\xi_i\,e^{-\eta^2\hat D_\Phi}$", transform=ax1.transAxes, fontsize=20)
    ax1.legend(frameon=False, fontsize=20, loc="lower right", handlelength=1.5)

    # Right panel: log derivative
    ax2.axhline(m_i, color="black", lw=3.0, zorder=5)
    ax2.axhline(m_phi, color="black", lw=3.0, ls="--", zorder=5)
    ax2.text(1.4e-4, 0.1, rf"$m_i={m_i:.3g}$", fontsize=20)
    ax2.text(1.4e-4, 0.5, rf"$m_\Phi={m_phi:.3g}$", fontsize=20)
    
    ax2.set_xscale("log")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 2)
    ax2.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax2.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(2.5)
    ax2.spines["bottom"].set_linewidth(2.5)
    ax2.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext())
    ax2.text(-0.02, 1.05, r"$d\ln D_u/d\ln R$", transform=ax2.transAxes, fontsize=32, weight='bold')
    ax2.text(1.03, -0.02, r"$R/r_i$", transform=ax2.transAxes, fontsize=32, weight='bold')

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print(out_png)
    print(out_svg)
    print("eta_list =", list(eta_list))
    print(f"m_i={m_i}, m_phi={m_phi}, r_phi/r_i={r_phi/r_i}")

def run():
    plot_directional_structure_and_proxy(
        out_png="directional_structure_and_proxy.png",
        out_svg="directional_structure_and_proxy.svg",
        r_i=1.0,
        r_phi=1.0,  # r_f_over_ri = 1.0, so r_phi = 1.0 * r_i = 1.0
        m_i=1.67,    # same as fig9.py
        m_phi=1.1,  # same as fig9.py (0.25)
        eta_list=np.concatenate([[0.0], np.geomspace(5e-3, 1e0, 10)]),  # same as fig9.py
        x_min=1e-7,  # same as fig9.py
        x_max=1e2,   # same as fig9.py
        nR=500
    )

if __name__ == "__main__":
    run()