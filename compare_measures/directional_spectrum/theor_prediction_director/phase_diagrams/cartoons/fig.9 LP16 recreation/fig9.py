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
    return np.unique(np.concatenate(([0.0], z1, z2, z3)))

def make_s_grid_for_rm(L, n_log1=150, n_log2=130, n_lin=100, zmin=1e-12, zmid1=1e-4, zmid2=1.0):
    z1 = np.geomspace(zmin, zmid1, n_log1)
    z2 = np.geomspace(zmid1, zmid2, n_log2)
    z3 = np.linspace(zmid2, L, n_lin)
    return np.unique(np.concatenate(([0.0], z1, z2, z3)))

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
    A0_b = interp1(s_grid, A0_R, b)
    A1_b = interp1(s_grid, A1_R, b)
    A0_d = interp1(s_grid, A0_R, delta)
    A1_d = interp1(s_grid, A1_R, delta)
    bplus = b[:, None] + delta[None, :]
    valid = (bplus <= L)
    bplus_safe = np.where(valid, bplus, L)
    A0_bp = interp1(s_grid, A0_R, bplus_safe.ravel()).reshape(b.size, delta.size)
    A1_bp = interp1(s_grid, A1_R, bplus_safe.ravel()).reshape(b.size, delta.size)
    Chat = (
        b[:, None] * A0_b[:, None] - A1_b[:, None]
        - delta[None, :] * A0_d[None, :]
        + bplus_safe * A0_bp - A1_bp
        + A1_d[None, :]
    )
    Chat = np.where(valid, Chat, 0.0)
    return Chat, valid

def inner_integral_cols(delta, b_grid, M):
    b = np.asarray(b_grid, float)
    delta = np.asarray(delta, float)
    db = b[1] - b[0]
    Cum = np.zeros_like(M, dtype=np.complex128)
    Cum[1:, :] = np.cumsum(0.5 * (M[1:, :] + M[:-1, :]) * db, axis=0)
    L = b[-1]
    b_target = L - delta
    t = b_target / db
    idx = np.floor(t).astype(int)
    idx = np.clip(idx, 0, b.size - 2)
    frac = t - idx
    cols = np.arange(delta.size)
    return Cum[idx, cols] + frac * (Cum[idx + 1, cols] - Cum[idx, cols])

class LP16EtaDerivativeKernelCache:
    def __init__(self, L, delta_grid, s_grid, r_f, m_f, eta, beta=0.0, nb_z2=160):
        self.L = float(L)
        self.delta = np.asarray(delta_grid, float)
        self.s_grid = np.asarray(s_grid, float)
        self.r_f = float(r_f)
        self.m_f = float(m_f)
        self.eta = float(eta)
        self.beta = float(beta)
        self.b = np.linspace(0.0, self.L, int(nb_z2))
        self.z2 = self.b[:, None]
        self.bplus = self.b[:, None] + self.delta[None, :]
        self.valid = (self.bplus <= self.L)
        self.z1 = np.minimum(self.bplus, self.L)
        self.A0_0, self.A1_0 = build_A0A1_hat(0.0, self.s_grid, r_f=self.r_f, m_f=self.m_f)
        self.Vhat_b = Vhat_from_A(self.A0_0, self.A1_0, self.s_grid, self.b)[:, None]
        self.Vhat_bp = Vhat_from_A(self.A0_0, self.A1_0, self.s_grid, self.z1.ravel()).reshape(self.b.size, self.delta.size)
        if (self.eta != 0.0) and (self.beta != 0.0):
            self.phase = np.exp(1j * self.eta * self.beta * self.delta)
        else:
            self.phase = np.ones_like(self.delta, dtype=np.complex128)

    def build_derivative_kernel(self, R_values):
        R_values = np.asarray(R_values, float)
        Kd = np.empty((R_values.size, self.delta.size), dtype=np.complex128)
        eta = self.eta
        beta = self.beta
        for i, R in enumerate(R_values):
            A0_R, A1_R = build_A0A1_hat(R, self.s_grid, r_f=self.r_f, m_f=self.m_f)
            Chat, _ = C_hat_bDelta(A0_R, A1_R, self.s_grid, b=self.b, delta=self.delta, L=self.L)
            Dhat = 0.5 * (self.Vhat_b + self.Vhat_bp - 2.0 * Chat)
            np.maximum(Dhat, 0.0, out=Dhat)
            Dhat = np.where(self.valid, Dhat, 0.0)
            if eta == 0.0:
                E = np.ones_like(Dhat)
            else:
                E = np.exp(-(eta * eta) * Dhat)
            if beta == 0.0:
                A = eta * (Chat - self.Vhat_bp)
                B = eta * (Chat - self.Vhat_b)
            else:
                A = 1j * beta * self.z1 + eta * (Chat - self.Vhat_bp)
                B = -1j * beta * self.z2 + eta * (Chat - self.Vhat_b)
            F = A * B + Chat
            M = np.where(self.valid, E * F, 0.0)
            IFd = inner_integral_cols(self.delta, self.b, M)
            Kd[i, :] = 2.0 * self.phase * IFd
        return Kd

def corr_from_derivative_kernel(R_values, Kd, delta_grid, r_i, m_i, sigma_i=1.0, Pbar_i=0.0):
    R_values = np.asarray(R_values, float)
    delta = np.asarray(delta_grid, float)
    xi = xi_i_eq30(R_values[:, None], delta[None, :], r_i=r_i, m=m_i, sigma_i=sigma_i, Pbar_i=Pbar_i)
    return TRAPZ(xi * Kd, delta, axis=1)

def DdP_from_derivative_kernel(R_values, Kd_R, Kd_0, delta_grid, r_i, m_i, sigma_i=1.0, Pbar_i=0.0, full_mode="eq93_printed"):
    C_R = corr_from_derivative_kernel(R_values, Kd_R, delta_grid, r_i, m_i, sigma_i, Pbar_i)
    C_0 = corr_from_derivative_kernel(np.array([0.0]), Kd_0, delta_grid, r_i, m_i, sigma_i, Pbar_i)[0]
    D_std = 2.0 * (np.real(C_0) - np.real(C_R))
    if full_mode == "eq90_std":
        D = D_std
    elif full_mode == "eq93_printed":
        D = 0.5 * D_std
    else:
        raise ValueError("full_mode must be 'eq90_std' or 'eq93_printed'")
    return D, float(np.real(C_0))

def projected_correlation_length(r, m, L):
    mbar = min(float(m), 1.0)
    return r * (L / r) ** ((1.0 - mbar) / (1.0 + mbar))

def classify_lengths(ri, mi, rf, mf):
    if ri < rf:
        return ri, mi, rf, mf
    if rf < ri:
        return rf, mf, ri, mi
    if mi <= mf:
        return ri, mi, rf, mf
    return rf, mf, ri, mi

def local_log_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log(x[m])
    ly = np.log(y[m])
    s = np.gradient(ly, lx)
    return x[m], s

def eta_from_lambda(lambda_val, sigma_phi):
    lam = np.asarray(lambda_val, float)
    return 2.0 * sigma_phi * lam * lam

def draw_figure9_multiple_eta(output_png="Figure9_multiple_eta_full.png",
                              output_svg="Figure9_multiple_eta_full.svg",
                              full_mode="eq93_printed",
                              beta=0.0,
                              eta_list=(0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
                              nR=100,
                              nb_z2=160):
    # Use same geometry as fig5.py
    r_i = 1.0
    L_over_ri = 100.0
    L = L_over_ri * r_i
    sigma_i = 1.0
    Pbar_i = 0.0
    
    m_i = 0.7
    m_phi = 1.0 / 4.0
    r_f_over_ri = 1.0
    r_f = r_f_over_ri * r_i
    
    delta = make_z_grid(L, n_log1=240, n_log2=220, n_lin=160)
    s_grid = make_s_grid_for_rm(L, n_log1=150, n_log2=130, n_lin=100)
    x_plot = np.logspace(-8, 8, nR)
    R_vals = x_plot * r_i

    # ApJ publication quality figure - match anzatz.py layout
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])  # Center panel: normalized structure function
    ax2 = fig.add_subplot(gs[0, 1])  # Right panel: log derivative

    # Use blue-to-red colormap
    colors = create_blue_to_red_colormap(len(eta_list))

    t0 = time.time()
    curves = []

    for eta, c in zip(eta_list, colors):
        cache = LP16EtaDerivativeKernelCache(
            L=L,
            delta_grid=delta,
            s_grid=s_grid,
            r_f=r_f,
            m_f=m_phi,
            eta=eta,
            beta=beta,
            nb_z2=nb_z2
        )
        Kd_R = cache.build_derivative_kernel(R_vals)
        Kd_0 = cache.build_derivative_kernel(np.array([0.0]))
        D, C0 = DdP_from_derivative_kernel(
            R_vals,
            Kd_R,
            Kd_0,
            delta,
            r_i=r_i,
            m_i=m_i,
            sigma_i=sigma_i,
            Pbar_i=Pbar_i,
            full_mode=full_mode
        )
        curves.append((eta, D, C0))
        
        # Normalize by max for the normalized panel
        y2 = D / (D.max() if D.max() > 0 else 1.0)
        
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax1.loglog(x_plot, y2, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(x_plot, y2)
        ax2.semilogx(xs, sl, lw=3.0, color=c, zorder=zorder, alpha=0.95)

    # Left panel: normalized structure function
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-7, 1e2)
    ax1.set_ylim(1e-10, 2)
    ax1.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax1.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(2.5)
    ax1.spines["bottom"].set_linewidth(2.5)
    ax1.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext())
    ax1.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
    ax1.yaxis.set_major_formatter(LogFormatterMathtext())

    # Right panel: log derivative
    ax2.set_xscale("log")
    ax2.set_xlim(1e-7, 1e2)
    ax2.set_ylim(0, 2.1)
    ax2.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax2.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(2.5)
    ax2.spines["bottom"].set_linewidth(2.5)
    ax2.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext())

    # Large, clear axis labels for maximum citation potential
    ax1.text(-0.16, 1.05, r"$D_{\mathrm{dP}}(\mathrm{R})/D_{\mathrm{dP}}(\max)$", transform=ax1.transAxes, fontsize=36, weight='bold')
    ax2.text(-0.02, 1.05, r"$d\ln D_{\mathrm{dP}}/d\ln R$", transform=ax2.transAxes, fontsize=32, weight='bold')

    for ax in (ax1, ax2):
        ax.text(1.03, -0.02, r"$R/r_i$", transform=ax.transAxes, fontsize=32, weight='bold')

    ax1.legend(frameon=False, fontsize=20, loc="lower right", ncol=1, handlelength=1.5)
    ax1.text(0.05, 0.90, rf"$m_i={m_i:.3g},\ m_\phi={m_phi:.3g}$", transform=ax1.transAxes, fontsize=20)
    ax1.text(0.05, 0.82, rf"$L/r_i={L_over_ri:.3g},\ r_\phi/r_i={r_f_over_ri:.3g}$", transform=ax1.transAxes, fontsize=20)
    ax1.text(0.05, 0.74, rf"$\beta=\bar\phi/\sigma_\phi={beta:.3g}$", transform=ax1.transAxes, fontsize=20)

    ax2.axhline(1.0 + m_i, color="black", lw=3.0, zorder=5)
    ax2.axhline(1.0 + m_phi, color="black", lw=3.0, ls="--", zorder=5)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    dt = time.time() - t0
    print(output_png)
    print(output_svg)
    print(f"runtime_s={dt:.3f}")
    print("eta_list=", list(eta_list))
    print("full_mode=", full_mode)
    print("beta=", beta)
    print("nb_z2=", nb_z2)
    print("nR=", nR)

def run():
    draw_figure9_multiple_eta(
        output_png="Figure9_multiple_eta_full.png",
        output_svg="Figure9_multiple_eta_full.svg",
        full_mode="eq93_printed",
        beta=0.0,
        eta_list=np.concatenate([[0.0], np.geomspace(5e-3, 1e0, 10)]),
        nR=100,
        nb_z2=160*2
    )

if __name__ == "__main__":
    run()