#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

TRAPZ = getattr(np, "trapezoid", np.trapz)

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
    x_min=1e-4,
    x_max=1e1,
    nR=500
):
    x = np.logspace(np.log10(x_min), np.log10(x_max), int(nR))
    R = x * float(r_i)

    cmap = plt.cm.plasma
    colors = [cmap(v) for v in np.linspace(0.08, 0.92, len(eta_list))]

    fig = plt.figure(figsize=(13.2, 5.0))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.92, bottom=0.20, wspace=0.24)
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
        ax1.loglog(x, D_u_half, lw=2.2, color=c, label=rf"$\eta={eta:g}$")

    # Plot log derivative on right panel
    for eta, D_u_half, c in curves:
        xs, sl = local_log_slope(x, D_u_half)
        ax2.semilogx(xs, sl, lw=2.0, color=c)

    # Left panel: structure function
    xg = np.array([1e-3, 8e-2])
    yg_i = 3e-5 * (xg / xg[0]) ** m_i
    yg_f = 8e-4 * (xg / xg[0]) ** m_phi
    ax1.loglog(xg, yg_i, "--", color="#2D5BFF", lw=1.8)
    ax1.loglog(xg, yg_f, "--", color="#E45756", lw=1.8)
    ax1.text(2.2e-2, 1.5e-3, rf"$\propto R^{{m_\phi}}$", color="#E45756", fontsize=18, rotation=16)
    ax1.text(3.0e-2, 1.5e-2, rf"$\propto R^{{m_i}}$", color="#2D5BFF", fontsize=18, rotation=33)

    eta_star = [e for e in eta_list if e > 0]
    if len(eta_star) > 0:
        eta_ref = eta_star[len(eta_star)//2]
        Rx = crossover_R_from_smallR(eta_ref, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi)
        if np.isfinite(Rx) and Rx > 0:
            xr = Rx / r_i
            ax1.plot([xr], [np.interp(np.log10(xr), np.log10(x), np.log10(curves[len(curves)//2][1]))], alpha=0)
            ax1.axvline(xr, color="0.2", lw=1.2, ls=":")
            ax1.text(xr*1.05, 1.7e-5, r"$R_\times$", fontsize=16)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.tick_params(which="both", direction="in", labelsize=13)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_major_formatter(LogFormatterMathtext())
    ax1.yaxis.set_major_formatter(LogFormatterMathtext())

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(1e-6, 1.2)
    ax1.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10]))
    ax1.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
    ax1.text(-0.10, 1.05, r"$D_u(R)/2=1-\xi_u$", transform=ax1.transAxes, fontsize=24)
    ax1.text(1.03, -0.02, r"$R/r_i$", transform=ax1.transAxes, fontsize=22)

    ax1.text(0.03, 0.92, rf"$m_i={m_i:.3g},\ m_\phi={m_phi:.3g}$", transform=ax1.transAxes, fontsize=15)
    ax1.text(0.03, 0.84, rf"$r_\phi/r_i={r_phi/r_i:.3g}$", transform=ax1.transAxes, fontsize=15)
    ax1.text(0.03, 0.76, r"$\xi_u=\xi_i\,e^{-\eta^2\hat D_\Phi}$", transform=ax1.transAxes, fontsize=15)
    ax1.legend(frameon=False, fontsize=12, loc="lower right")

    # Right panel: log derivative
    ax2.axhline(m_i, color="black", lw=1.8)
    ax2.axhline(m_phi, color="black", lw=1.8, ls="--")
    ax2.text(1.4e-4, m_i + 0.03, rf"$m_i={m_i:.3g}$", fontsize=14)
    ax2.text(1.4e-4, m_phi + 0.03, rf"$m_\phi={m_phi:.3g}$", fontsize=14)
    
    ax2.set_xscale("log")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(which="both", direction="in", labelsize=13)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10]))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext())
    ax2.text(-0.02, 1.05, r"$d\ln D_u/d\ln R$", transform=ax2.transAxes, fontsize=24)
    ax2.text(1.03, -0.02, r"$R/r_i$", transform=ax2.transAxes, fontsize=22)

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
        r_phi=0.1,
        m_i=4.0/5.0,
        m_phi=1.0/3.0,
        eta_list=(0.0, 0.05, 0.09, 0.2, 0.3, 0.5),
        x_min=1e-4,
        x_max=1e1,
        nR=500
    )

if __name__ == "__main__":
    run()