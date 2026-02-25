#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
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

def xi_i_eq30(R, dz, r_i, m, sigma_i=1.0, Pbar_i=0.0):
    rr = (R * R + dz * dz) ** (m / 2.0)
    return abs(Pbar_i) ** 2 + sigma_i**2 * (r_i**m) / (r_i**m + rr)

def make_z_grid(L, n_log1=240, n_log2=220, n_lin=160, zmin=1e-12, zmid1=1e-4, zmid2=1.0):
    z1 = np.geomspace(zmin, zmid1, n_log1)
    z2 = np.geomspace(zmid1, zmid2, n_log2)
    z3 = np.linspace(zmid2, L, n_lin)
    z = np.unique(np.concatenate(([0.0], z1, z2, z3)))
    return z

def make_s_grid_for_rm(L, n_log1=150, n_log2=130, n_lin=100, zmin=1e-12, zmid1=1e-4, zmid2=1.0):
    z1 = np.geomspace(zmin, zmid1, n_log1)
    z2 = np.geomspace(zmid1, zmid2, n_log2)
    z3 = np.linspace(zmid2, L, n_lin)
    z = np.unique(np.concatenate(([0.0], z1, z2, z3)))
    return z

def cumtrapz_1d(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out

def interp1(x, y, xq):
    return np.interp(np.asarray(xq, float), np.asarray(x, float), np.asarray(y, float))

def xi_phi_hat(R, dz, r_f, m_f):
    rr = (R * R + dz * dz) ** (m_f / 2.0)
    return (r_f**m_f) / (r_f**m_f + rr)

def build_A0A1_hat(R, s_grid, r_f, m_f):
    xi = xi_phi_hat(R, s_grid, r_f=r_f, m_f=m_f)
    A0 = cumtrapz_1d(xi, s_grid)
    A1 = cumtrapz_1d(s_grid * xi, s_grid)
    return A0, A1

def Vhat_from_A(A0_0, A1_0, s_grid, zq):
    zq = np.asarray(zq, float)
    A0z = interp1(s_grid, A0_0, zq)
    A1z = interp1(s_grid, A1_0, zq)
    return 2.0 * (zq * A0z - A1z)

def C_hat_bDelta(A0_R, A1_R, s_grid, b, delta, L):
    b = np.asarray(b, float)
    delta = np.asarray(delta, float)
    nb = b.size
    nd = delta.size
    A0_b = interp1(s_grid, A0_R, b)
    A1_b = interp1(s_grid, A1_R, b)
    A0_d = interp1(s_grid, A0_R, delta)
    A1_d = interp1(s_grid, A1_R, delta)
    bplus = b[:, None] + delta[None, :]
    valid = (bplus <= L)
    bplus_safe = np.where(valid, bplus, L)
    A0_bp = interp1(s_grid, A0_R, bplus_safe.ravel()).reshape(nb, nd)
    A1_bp = interp1(s_grid, A1_R, bplus_safe.ravel()).reshape(nb, nd)
    Chat = (
        b[:, None] * A0_b[:, None] - A1_b[:, None]
        - delta[None, :] * A0_d[None, :]
        + bplus_safe * A0_bp - A1_bp
        + A1_d[None, :]
    )
    Chat = np.where(valid, Chat, 0.0)
    return Chat, valid

def inner_integral_IF(delta, eta, b_grid, Dhat_matrix):
    b = np.asarray(b_grid, float)
    delta = np.asarray(delta, float)
    db = b[1] - b[0]
    F = np.exp(-(eta * eta) * Dhat_matrix)
    Cum = np.zeros_like(F)
    Cum[1:, :] = np.cumsum(0.5 * (F[1:, :] + F[:-1, :]) * db, axis=0)
    L = b[-1]
    b_target = L - delta
    t = b_target / db
    idx = np.floor(t).astype(int)
    idx = np.clip(idx, 0, b.size - 2)
    frac = t - idx
    cols = np.arange(delta.size)
    IF = Cum[idx, cols] + frac * (Cum[idx + 1, cols] - Cum[idx, cols])
    return IF

class LP16EtaKernelCache:
    def __init__(self, L, delta_grid, s_grid, r_f, m_f, eta, beta=0.0, nb_z2=160):
        self.L = float(L)
        self.delta = np.asarray(delta_grid, float)
        self.s_grid = np.asarray(s_grid, float)
        self.r_f = float(r_f)
        self.m_f = float(m_f)
        self.eta = float(eta)
        self.beta = float(beta)
        self.b = np.linspace(0.0, self.L, int(nb_z2))
        self.A0_0, self.A1_0 = build_A0A1_hat(0.0, self.s_grid, r_f=self.r_f, m_f=self.m_f)
        self.bplus = self.b[:, None] + self.delta[None, :]
        self.valid = (self.bplus <= self.L)
        bplus_clip = np.minimum(self.bplus, self.L)
        self.Vhat_b = Vhat_from_A(self.A0_0, self.A1_0, self.s_grid, self.b)[:, None]
        self.Vhat_bp = Vhat_from_A(self.A0_0, self.A1_0, self.s_grid, bplus_clip.ravel()).reshape(self.b.size, self.delta.size)
        if (self.eta != 0.0) and (self.beta != 0.0):
            self.phase = np.exp(1j * self.eta * self.beta * self.delta)
        else:
            self.phase = np.ones_like(self.delta, dtype=np.complex128)

    def build_kernel(self, R_values):
        R_values = np.asarray(R_values, float)
        K = np.empty((R_values.size, self.delta.size), dtype=np.complex128)
        if self.eta == 0.0:
            IF0 = (self.L - self.delta).astype(float)
            K[:] = self.phase[None, :] * IF0[None, :]
            return K
        for i, R in enumerate(R_values):
            A0_R, A1_R = build_A0A1_hat(R, self.s_grid, r_f=self.r_f, m_f=self.m_f)
            Chat, _ = C_hat_bDelta(A0_R, A1_R, self.s_grid, b=self.b, delta=self.delta, L=self.L)
            Dhat = 0.5 * (self.Vhat_b + self.Vhat_bp - 2.0 * Chat)
            np.maximum(Dhat, 0.0, out=Dhat)
            Dhat = np.where(self.valid, Dhat, 0.0)
            IF = inner_integral_IF(self.delta, self.eta, self.b, Dhat)
            K[i, :] = self.phase * IF
        return K

def corr_from_kernel(R_values, kernel, delta_grid, r_i, m_i, sigma_i=1.0, Pbar_i=0.0):
    R_values = np.asarray(R_values, float)
    delta = np.asarray(delta_grid, float)
    K = np.asarray(kernel)
    xi = xi_i_eq30(R_values[:, None], delta[None, :], r_i=r_i, m=m_i, sigma_i=sigma_i, Pbar_i=Pbar_i)
    return TRAPZ(xi * K, delta, axis=1)

def D_from_kernel(R_values, kernel_R, kernel_0, delta_grid, r_i, m_i, sigma_i=1.0, Pbar_i=0.0):
    C_R = corr_from_kernel(R_values, kernel_R, delta_grid, r_i, m_i, sigma_i, Pbar_i)
    C_0 = corr_from_kernel(np.array([0.0]), kernel_0, delta_grid, r_i, m_i, sigma_i, Pbar_i)[0]
    D_R = 2.0 * (np.real(C_0) - np.real(C_R))
    D_inf = 2.0 * np.real(C_0)
    return D_R, float(np.real(C_0)), float(D_inf)

def local_log_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log(x[m])
    ly = np.log(y[m])
    s = np.gradient(ly, lx)
    return x[m], s

def D_P_asymptotic_eq45(R_values, r_i, m, L, sigma_i=1.0):
    mbar = min(float(m), 1.0)
    return sigma_i**2 * L * R_values * (R_values / r_i) ** mbar

def run():
    out_png = "lp16_eta_slopebreak_corrected.png"
    out_svg = "lp16_eta_slopebreak_corrected.svg"

    r_i = 1.0
    L_over_ri = 100.0
    L = L_over_ri * r_i
    sigma_i = 1.0
    Pbar_i = 0.0

    m_i = 0.67#0.8+1
    m_phi = 0.1
    r_f_over_ri = 1.0#0.1#1.0#0.3
    r_f = r_f_over_ri * r_i

    beta = 0.0
    eta_list = np.concatenate([[0.0], np.geomspace(5e-3, 1e0, 10)])#[0.0, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    nR = 100
    x = np.logspace(-8, 8, nR)
    R = x * r_i

    delta = make_z_grid(L, n_log1=240, n_log2=220, n_lin=160)
    s_grid = make_s_grid_for_rm(L, n_log1=150, n_log2=130, n_lin=100)

    t0 = time.time()
    curves = []
    for eta in eta_list:
        cache = LP16EtaKernelCache(L=L, delta_grid=delta, s_grid=s_grid, r_f=r_f, m_f=m_phi, eta=eta, beta=beta, nb_z2=160)
        K_R = cache.build_kernel(R)
        K_0 = cache.build_kernel(np.array([0.0]))
        D, C0, Dinf = D_from_kernel(R_values=R, kernel_R=K_R, kernel_0=K_0, delta_grid=delta, r_i=r_i, m_i=m_i, sigma_i=sigma_i, Pbar_i=Pbar_i)
        curves.append((eta, D, Dinf))

    # ApJ publication quality figure - match anzatz.py layout
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])  # Center panel: D_P(R)/D_P(inf)
    ax2 = fig.add_subplot(gs[0, 1])  # Right panel: log derivative

    # Left panel: normalized structure function
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-7, 1e2)
    ax1.set_ylim(1e-13, 10)
    ax1.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax1.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(2.5)
    ax1.spines["bottom"].set_linewidth(2.5)
    # ax1.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    # ax1.xaxis.set_major_formatter(LogFormatterMathtext())
    # ax1.yaxis.set_major_locator(FixedLocator([1e0, 1e-2, 1e-4, 1e-6]))
    # ax1.yaxis.set_major_formatter(LogFormatterMathtext())
    ax1.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax1.yaxis.set_major_locator(FixedLocator([1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]))
    # Right panel: log derivative
    ax2.set_xscale("log")
    ax2.set_xlim(1e-7, 1e2)
    ax2.set_ylim(0, 2)
    ax2.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax2.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(2.5)
    ax2.spines["bottom"].set_linewidth(2.5)
    ax2.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext())

    # Use blue-to-red colormap
    colors = create_blue_to_red_colormap(len(curves))

    for (eta, D, Dinf), c in zip(curves, colors):
        y2 = D / Dinf
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax1.loglog(x, y2, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        xs, sl = local_log_slope(x, y2)
        ax2.semilogx(xs, sl, lw=3.0, color=c, zorder=zorder, alpha=0.95)

    # Reference lines on left panel: power-law asymptotes spanning full R range
    xg = np.logspace(-7, 2, 200)  # Full range from 1e-7 to 1e2
    yg0 = 1e-8
    yg_intr = yg0 * (xg / xg[0]) ** (1.0 + m_i)
    yg_far = (1.5e-8) * (xg / xg[0]) ** (1.0 + m_phi)
    ax1.loglog(xg, yg_intr, color="#2D5BFF", lw=3.0, ls="-", zorder=5, alpha=0.8)
    ax1.text(1.25e-3, 5e-7, rf"$\propto R^{{1+m_i}}$", fontsize=24, color="#2D5BFF")
    ax1.loglog(xg, yg_far, color="#E45756", lw=3.0, ls="-.", zorder=5, alpha=0.8)
    ax1.text(2.0e-3, 8e-6, rf"$\propto R^{{1+m_\phi}}$", fontsize=24, color="#E45756")

    # Reference lines on right panel: horizontal lines at slopes
    ax2.axhline(1.0 + m_i, color="#2D5BFF", lw=3.0, ls="-", zorder=5, alpha=0.8)
    ax2.axhline(1.0 + m_phi, color="#E45756", lw=3.0, ls="-.", zorder=5, alpha=0.8)
    ax2.text(1.4e-4, 1.0 + m_i + 0.05, rf"$1+m_i={1+m_i:.3f}$", fontsize=20, color="#2D5BFF")
    ax2.text(1.4e-4, 1.0 + m_phi + 0.05, rf"$1+m_\phi={1+m_phi:.3f}$", fontsize=20, color="#E45756")

    # Large, clear axis labels for maximum citation potential
    ax1.text(-0.16, 1.05, r"$D_P(\mathrm{R})/D_P(\infty)$", transform=ax1.transAxes, fontsize=36, weight='bold')
    ax2.text(-0.02, 1.05, r"$d\ln D_P/d\ln R$", transform=ax2.transAxes, fontsize=32, weight='bold')

    for ax in (ax1, ax2):
        ax.text(1.03, -0.02, r"$R/r_i$", transform=ax.transAxes, fontsize=32, weight='bold')

    ax1.legend(frameon=False, fontsize=20, loc="lower right", ncol=1, handlelength=1.5)
    ax1.text(0.05, 0.90, rf"$m_i={m_i:.3g},\ m_\phi={m_phi:.3g}$", transform=ax1.transAxes, fontsize=20)
    ax1.text(0.05, 0.82, rf"$L/r_i={L_over_ri:.3g},\ r_\phi/r_i={r_f_over_ri:.3g}$", transform=ax1.transAxes, fontsize=20)
    ax1.text(0.05, 0.74, rf"$\beta=\bar\phi/\sigma_\phi={beta:.3g}$", transform=ax1.transAxes, fontsize=20)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    dt = time.time() - t0
    print(out_png)
    print(out_svg)
    print(f"runtime_s={dt:.3f}")
    print(f"m_i={m_i} m_phi={m_phi} L_over_ri={L_over_ri} r_phi_over_ri={r_f_over_ri}")
    print("eta_list=", eta_list)

if __name__ == "__main__":
    run()