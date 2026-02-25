#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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

TRAPZ = getattr(np, "trapezoid", np.trapz)

# LP16 kernel functions from fig5.py for computing P(R) exactly
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

# LP16 derivative kernel functions from fig9.py for computing dP/d(lambda^2)
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

def xi_i_director_regularized(R, r_i, m_i):
    """Exact match to anzatz.py: xi_i = 1/(1 + (R/r_\phi)^m_i)"""
    t = (np.asarray(R, float) / float(r_i)) ** float(m_i)
    return 1.0 / (1.0 + t)

def Dphi_hat_regularized(R, r_phi, m_phi):
    """Exact match to anzatz.py: dphi_hat = (R/r_phi)^m_phi / (1 + (R/r_phi)^m_phi)"""
    t = (np.asarray(R, float) / float(r_phi)) ** float(m_phi)
    return t / (1.0 + t)

def xi_u_factorized(R, r_i, m_i, r_phi, m_phi, eta):
    """Exact match to anzatz.py: xi_u = xi_i * exp(-eta^2 * dphi_hat)"""
    xi_i = xi_i_director_regularized(R, r_i=r_i, m_i=m_i)
    dphi_hat = Dphi_hat_regularized(R, r_phi=r_phi, m_phi=m_phi)
    return xi_i * np.exp(-(float(eta) ** 2) * dphi_hat)

def directional_structure_measures(R, r_i, m_i, r_phi, m_phi, eta):
    """Exact match to anzatz.py: D_u/2 = 1 - xi_u"""
    xi_u = xi_u_factorized(R, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi, eta=eta)
    D_u_half = 1.0 - xi_u
    return D_u_half

# Keep old functions for backward compatibility with P and dP/d(lambda^2)
def Xi_i(R, A_P=1.0, R0=1.0, m_psi=2/3):
    return A_P / (1.0 + (R / R0) ** m_psi)

def f_saturating_powerlaw(R, Rb, m):
    x = (R / Rb) ** m
    return x / (1.0 + x)

def chi2_from_lambda(lam, sigma_phi2, chi2_factor=2.0):
    return chi2_factor * (lam ** 4) * sigma_phi2

def xi_P(R, A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2, chi2_factor=2.0):
    Xi = Xi_i(R, A_P=A_P, R0=R0, m_psi=m_psi)
    fphi = f_saturating_powerlaw(R, r_phi, m_phi)
    chi2 = chi2_from_lambda(lam, sigma_phi2, chi2_factor=chi2_factor)
    return Xi * np.exp(-chi2 * fphi)

def dP_dlambda2(R, A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2, chi2_factor=2.0):
    """
    Compute d(xi_P)/d(lambda^2).
    Since chi2 = chi2_factor * lam^4 * sigma_phi2,
    d(chi2)/d(lambda^2) = 2 * chi2_factor * lam^2 * sigma_phi2
    and d(xi_P)/d(chi2) = -xi_P * fphi
    """
    xi = xi_P(R, A_P=A_P, R0=R0, m_psi=m_psi,
        r_phi=r_phi, m_phi=m_phi,
        lam=lam, sigma_phi2=sigma_phi2,
              chi2_factor=chi2_factor)
    fphi = f_saturating_powerlaw(R, r_phi, m_phi)
    dchi2_dlambda2 = 2.0 * chi2_factor * (lam ** 2) * sigma_phi2
    return -xi * fphi * dchi2_dlambda2

def sf_spectrum_proxy(k, R, Du2, sigma_ln=0.35):
    lnR = np.log(R)
    dlnR = np.diff(lnR)

    ln_kR = np.log(k[:, None]) + lnR[None, :]

    W = np.exp(-(ln_kR ** 2) / (2.0 * sigma_ln ** 2)) / (math.sqrt(2.0 * math.pi) * sigma_ln)

    integrand = Du2[None, :] * W
    return np.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dlnR[None, :], axis=1)

def local_log_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        # Not enough valid points for gradient
        return np.array([]), np.array([])
    lx = np.log(x[m])
    ly = np.log(y[m])
    s = np.gradient(ly, lx)
    return x[m], s

def _setup_panel(ax, x_min, x_max, ylabel, xlabel, ylim, legend=False, params_text=False, m_psi=None, m_phi=None, r_phi=None, R0=None):
    """Helper function to set up a main plot panel."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    # Move x-axis ticks down
    ax.tick_params(axis="x", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4,1e5,1e6,1e7]))
    if ylim[0] < 1e-10:
        ax.yaxis.set_major_locator(FixedLocator([1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]))
    ax.text(-0.10, 1.05, ylabel, transform=ax.transAxes, fontsize=36, weight='bold')
    ax.text(1.03, -0.02, xlabel, transform=ax.transAxes, fontsize=32, weight='bold')
    if legend:
        ax.legend(frameon=False, fontsize=20, loc="lower right", handlelength=1.5)
    if params_text and m_psi is not None and m_phi is not None:
        ax.text(0.03, 0.92, rf"$m_i={m_psi:.3g},\ m_\Phi={m_phi:.3g}$", transform=ax.transAxes, fontsize=20)
        if r_phi is not None and R0 is not None:
            ax.text(0.03, 0.84, rf"$r_\phi/R_0={r_phi/R0:.3g}$", transform=ax.transAxes, fontsize=20)

def _setup_derivative_panel(ax, x_min, x_max, ylabel, xlabel, ylim, m_psi=None, m_phi=None):
    """Helper function to set up a derivative panel."""
    if m_psi is not None:
        ax.axhline(m_psi, color="black", lw=3.0, zorder=5)
        offset = 0.1 if m_psi >= 0 else -0.1
        ax.text(1.4e-4, m_psi + offset, rf"${m_psi:.3g}$", fontsize=20)
    if m_phi is not None:
        ax.axhline(m_phi, color="black", lw=3.0, ls="--", zorder=5)
        offset = 0.1 if m_phi >= 0 else -0.1
        ax.text(1.4e-4, m_phi + offset, rf"${m_phi:.3g}$", fontsize=20)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ylim)
    ax.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    # Move x-axis ticks down
    ax.tick_params(axis="x", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4,1e5,1e6,1e7]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.text(-0.02, 1.05, ylabel, transform=ax.transAxes, fontsize=32, weight='bold')
    ax.text(1.03, -0.02, xlabel, transform=ax.transAxes, fontsize=32, weight='bold')

def run_and_plot(params,
                 Rmin=1e-5, Rmax=1e3, NR=1400,
                 Nk=900,
                 sigma_ln=0.35,
                 out_prefix="sf_run",
                 sigma_phi2_list=None):
    A_P = params["A_P"]
    R0 = params["R0"]
    m_psi = params["m_psi"]
    r_phi = params["r_phi"]
    m_phi = params["m_phi"]
    lam = params["lam"]
    sigma_phi2_default = params["sigma_phi2"]
    chi2_factor = params.get("chi2_factor", 2.0)

    # If sigma_phi2_list is provided, use it; otherwise use single value
    if sigma_phi2_list is None:
        sigma_phi2_list = [sigma_phi2_default]

    # Use normalized R axis: x = R/r_\phi (matching reference files)
    # Plot limits
    x_min = 1e-7
    x_max = 1e2
    # Use same x_plot as fig9.py: np.logspace(-8, 8, nR) with nR=100
    nR = 100
    x_plot = np.logspace(-8, 8, nR)
    R_plot = x_plot * R0  # R = x * r_i, where r_i = R0
    # For plotting, use the original range
    x = np.logspace(np.log10(x_min), np.log10(x_max), nR)
    R = x * R0
    
    kmin = 1.0 / (x_max * R0)
    kmax = 1.0 / (x_min * R0)
    # Reduce Nk for speed
    Nk_actual = min(Nk, 400)  # Cap at 400 for speed
    k = np.logspace(np.log10(kmin), np.log10(kmax), Nk_actual)

    # Set sigma_phi = 1 (matching anzatz.py)
    sigma_phi = 1.0
    
    # If sigma_phi2_list is provided, convert to eta_list using eta = 2 * sigma_phi * lambda^2
    # Otherwise, use default eta_list
    if sigma_phi2_list is not None and len(sigma_phi2_list) > 0:
        # For backward compatibility: if sigma_phi2_list is provided, 
        # we'll generate eta_list directly (same as main function)
        eta_list = np.concatenate([[0.0], np.geomspace(3e-3, 1e0, 10)])
    else:
        # Default: use eta_list from main
        eta_list = np.concatenate([[0.0], np.geomspace(3e-3, 1e0, 10)])
    
    # Use blue-to-red colormap matching reference files
    colors = create_blue_to_red_colormap(len(eta_list))
    
    # Compute structure functions, correlation functions, derivatives, and spectra
    # Use exact formulas from anzatz.py for D_u/2
    all_Du2 = []
    all_xi_P = []
    all_dP_dlambda2 = []
    all_M = []
    
    # Map parameters: R0 -> r_i, m_psi -> m_i
    r_i = R0
    m_i = m_psi
    
    # Set up LP16 parameters matching fig5.py exactly
    L_over_ri = 100.0
    L = L_over_ri * r_i
    sigma_i = 1.0
    Pbar_i = 0.0
    r_f_over_ri = 1.0
    r_f = r_f_over_ri * r_i
    beta = 0.0
    
    # Create grids for LP16 kernel computation (matching fig9.py exactly)
    delta = make_z_grid(L, n_log1=240, n_log2=220, n_lin=160)
    s_grid = make_s_grid_for_rm(L, n_log1=150, n_log2=130, n_lin=100)
    
    for idx, eta in enumerate(eta_list):
        # Use exact anzatz.py formula for D_u/2 with eta = 2 * sigma_phi * lambda^2
        # Compute directly on R (plot domain) for accuracy - no interpolation needed
        Du2 = directional_structure_measures(R, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi, eta=eta)
        # Also compute on R_plot for spectrum calculation
        Du2_plot = directional_structure_measures(R_plot, r_i=r_i, m_i=m_i, r_phi=r_phi, m_phi=m_phi, eta=eta)
        
        # For P(R), use LP16 kernel method exactly as in fig5.py
        # P(R) = D_P(R)/D_P(inf) where D_P is computed using LP16 kernel
        # Use m_phi - 1 and m_i - 1 as inputs (but labels still show m_phi and m_i)
        m_f_input = m_phi - 1.0
        m_i_input = m_i - 1.0

        # Compute directly on R (plot domain) for accuracy
        cache = LP16EtaKernelCache(L=L, delta_grid=delta, s_grid=s_grid, r_f=r_f, m_f=m_f_input, eta=eta, beta=beta, nb_z2=160)
        K_R = cache.build_kernel(R)
        K_0 = cache.build_kernel(np.array([0.0]))
        D_P, C0, Dinf = D_from_kernel(R_values=R, kernel_R=K_R, kernel_0=K_0, delta_grid=delta, r_i=r_i, m_i=m_i_input, sigma_i=sigma_i, Pbar_i=Pbar_i)
        P_R = D_P / Dinf  # Normalized structure function matching fig5.py
        
        # For dP/d(lambda^2), use LP16 derivative kernel method from fig9.py exactly
        # Match fig9.py exactly: use m_phi and m_i (not m_phi - 1 and m_i - 1)
        cache_dP = LP16EtaDerivativeKernelCache(
            L=L,
            delta_grid=delta,
            s_grid=s_grid,
            r_f=r_f,
            m_f=m_phi-1,  # Use m_phi directly (matching fig9.py)
            eta=eta,
            beta=beta,
            nb_z2=160
        )
        # Compute directly on R (plot domain) for accuracy
        Kd_R = cache_dP.build_derivative_kernel(R)
        Kd_0 = cache_dP.build_derivative_kernel(np.array([0.0]))
        D_dP, C0_dP = DdP_from_derivative_kernel(
            R_values=R,
            Kd_R=Kd_R,
            Kd_0=Kd_0,
            delta_grid=delta,
            r_i=r_i,
            m_i=m_i-1,  # Use m_i directly (matching fig9.py)
            sigma_i=sigma_i,
            Pbar_i=Pbar_i,
            full_mode="eq93_printed"
        )
        # Store unnormalized D_dP (normalize in plotting section like fig9.py)
        dP_dl2 = D_dP
        
        all_Du2.append(Du2)
        all_xi_P.append(P_R)  # Now using LP16 kernel result
        all_dP_dlambda2.append(dP_dl2)

        # Calculate spectrum using extended domain (R_plot) for better accuracy
        M = sf_spectrum_proxy(k, R_plot, Du2_plot, sigma_ln=sigma_ln)
        all_M.append(M)

    # ============================================
    # Figure 1: Structure Functions (3 rows: D_u/2, P, dP/d(lambda^2))
    # ============================================
    fig1 = plt.figure(figsize=(16, 18))
    fig1.patch.set_facecolor('white')
    gs1 = fig1.add_gridspec(3, 2, left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.25, wspace=0.28)
    
    # Row 1: D_u/2
    ax1_du = fig1.add_subplot(gs1[0, 0])
    ax1_du_der = fig1.add_subplot(gs1[0, 1])
    
    # Row 2: P (xi_P)
    ax1_p = fig1.add_subplot(gs1[1, 0])
    ax1_p_der = fig1.add_subplot(gs1[1, 1])
    
    # Row 3: dP/d(lambda^2)
    ax1_dp = fig1.add_subplot(gs1[2, 0])
    ax1_dp_der = fig1.add_subplot(gs1[2, 1])

    xg = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    # Row 1: D_u/2 - reference lines
    yg_psi = 3e-9 * (xg / x_min) ** 5/3
    yg_phi = 8e-8 * (xg / x_min) ** 1.1
    ax1_du.loglog(xg, yg_psi, "-", color="blue", lw=3.0, zorder=5, alpha=0.8)
    ax1_du.loglog(xg, yg_phi, "-.", color="red", lw=3.0, zorder=5, alpha=0.8)
    ax1_du.text(2.2e-2, 1.5e-3, rf"$\propto R^{{m_\Phi}}$", color="red", fontsize=24, rotation=16)
    ax1_du.text(3.0e-2, 1.5e-2, rf"$\propto R^{{m_i}}$", color="blue", fontsize=24, rotation=33)
    
    for idx, (Du2, eta, c) in enumerate(zip(all_Du2, eta_list, colors)):
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax1_du.loglog(x, Du2, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(x, Du2)
        if len(xs) > 0:
            ax1_du_der.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)
    
    _setup_panel(ax1_du, x_min, x_max, r"$D_u(R)/2$", r"$R/r_\phi$", 
                 ylim=(1e-13, 1e1), legend=True, params_text=True,
                 m_psi=m_psi, m_phi=m_phi, r_phi=r_phi, R0=R0)
    ax1_du_der.axhline(1.1, color="red", lw=3.0, ls="-.", zorder=5)
    ax1_du_der.axhline(5/3, color="blue", lw=3.0, ls="-", zorder=5)
    _setup_derivative_panel(ax1_du_der, x_min, x_max, r"$d\ln D_u/d\ln R$", r"$R/r_\phi$",
                           ylim=(0, 2))

    # Row 2: P (D_P(R)/D_P(inf)) - matching fig5.py exactly
    # Reference lines for P(R) matching fig5.py
    xg_p = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    yg0_p = 1e-8
    yg_intr_p = yg0_p * (xg_p / x_min) ** (5/3)
    yg_far_p = (1.5e-8) * (xg_p / x_min) ** (1.1)
    ax1_p.loglog(xg_p, yg_intr_p, color="blue", lw=3.0, ls="-", zorder=5, alpha=0.8)
    ax1_p.text(1.25e-3, 5e-7, rf"$\propto R^{{m_i}}$", fontsize=24, color="blue")
    ax1_p.loglog(xg_p, yg_far_p, color="red", lw=3.0, ls="-.", zorder=5, alpha=0.8)
    ax1_p.text(2.0e-3, 8e-6, rf"$\propto R^{{m_\Phi}}$", fontsize=24, color="red")
    
    for idx, (xi, eta, c) in enumerate(zip(all_xi_P, eta_list, colors)):
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax1_p.loglog(x, xi, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(x, xi)
        if len(xs) > 0:
            ax1_p_der.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)
    
    _setup_panel(ax1_p, x_min, x_max, r"$D_P(R)/D_P(\infty)$", r"$R/r_\phi$",
                 ylim=(1e-13, 10), legend=True)
    ax1_p.yaxis.set_major_locator(FixedLocator([1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]))
    
    # Reference lines on right panel for P derivative
    ax1_p_der.axhline(1.1, color="red", lw=3.0, ls="-.", zorder=5)
    ax1_p_der.axhline(5/3, color="blue", lw=3.0, ls="-", zorder=5)
    _setup_derivative_panel(ax1_p_der, x_min, x_max, r"$d\ln D_P/d\ln R$", r"$R/r_\phi$",
                           ylim=(0, 2))

    # Row 3: dP/d(lambda^2) - using fig9.py method exactly
    # Reference lines matching fig9.py
    xg_dp = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    yg0_dp = 1e-8
    yg_intr_dp = yg0_dp * (xg_dp / x_min) ** (5/3)
    yg_far_dp = (1.5e-8) * (xg_dp / x_min) ** (1.1)
    ax1_dp.loglog(xg_dp, yg_intr_dp, color="blue", lw=3.0, ls="-", zorder=5, alpha=0.8)
    ax1_dp.text(1.25e-3, 5e-7, rf"$\propto R^{{m_i}}$", fontsize=24, color="blue")
    ax1_dp.loglog(xg_dp, yg_far_dp, color="red", lw=3.0, ls="-.", zorder=5, alpha=0.8)
    ax1_dp.text(2.0e-3, 8e-6, rf"$\propto R^{{m_\Phi}}$", fontsize=24, color="red")
    
    # Plot dP (normalize by max in plotting section, matching fig9.py exactly)
    for idx, (dP, eta, c) in enumerate(zip(all_dP_dlambda2, eta_list, colors)):
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        # Normalize by max (matching fig9.py exactly)
        y2 = dP / (dP.max() if dP.max() > 0 else 1.0)
        ax1_dp.loglog(x, y2, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(x, y2)
        if len(xs) > 0:
            ax1_dp_der.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)
    
    _setup_panel(ax1_dp, x_min, x_max, r"$dP/d\lambda^2$", r"$R/r_\phi$",
                 ylim=(1e-13, 10), legend=True)
    ax1_dp.yaxis.set_major_locator(FixedLocator([1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]))
    
    # Reference lines on right panel for dP derivative
    ax1_dp_der.axhline(1.1, color="red", lw=3.0, ls="-.", zorder=5)
    ax1_dp_der.axhline(5/3, color="blue", lw=3.0, ls="-", zorder=5)
    _setup_derivative_panel(ax1_dp_der, x_min, x_max, r"$d\ln |dP/d\lambda^2|/d\ln R$", r"$R/r_\phi$",
                           ylim=(0, 2))

    fig1.savefig(f"{out_prefix}_structure_functions.png", dpi=300, bbox_inches="tight")
    fig1.savefig(f"{out_prefix}_structure_functions.svg", bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {out_prefix}_structure_functions.png and .svg")

    # ============================================
    # Figure 2: Spectrum (3 rows: spectrum of D_u/2, P, dP/d(lambda^2))
    # ============================================
    fig2 = plt.figure(figsize=(16, 18))
    fig2.patch.set_facecolor('white')
    gs2 = fig2.add_gridspec(3, 2, left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.25, wspace=0.28)
    
    # Row 1: Spectrum of D_u/2
    ax2_du = fig2.add_subplot(gs2[0, 0])
    ax2_du_der = fig2.add_subplot(gs2[0, 1])
    
    # Row 2: Spectrum of P
    ax2_p = fig2.add_subplot(gs2[1, 0])
    ax2_p_der = fig2.add_subplot(gs2[1, 1])
    
    # Row 3: Spectrum of dP/d(lambda^2)
    ax2_dp = fig2.add_subplot(gs2[2, 0])
    ax2_dp_der = fig2.add_subplot(gs2[2, 1])

    # Compute spectra for P and dP/d(lambda^2) over extended domain (R_plot)
    all_M_P = []
    all_M_dP = []
    for idx, eta in enumerate(eta_list):
        # Recompute P and dP over extended domain for spectrum calculation
        m_f_input = m_phi - 1.0
        m_i_input = m_i - 1.0
        
        # P over extended domain (R_plot)
        cache_P = LP16EtaKernelCache(L=L, delta_grid=delta, s_grid=s_grid, r_f=r_f, m_f=m_f_input, eta=eta, beta=beta, nb_z2=160)
        K_R_P_plot = cache_P.build_kernel(R_plot)
        K_0_P = cache_P.build_kernel(np.array([0.0]))
        D_P_plot, C0_P, Dinf_P = D_from_kernel(R_values=R_plot, kernel_R=K_R_P_plot, kernel_0=K_0_P, delta_grid=delta, r_i=r_i, m_i=m_i_input, sigma_i=sigma_i, Pbar_i=Pbar_i)
        P_R_plot = D_P_plot / Dinf_P
        
        # dP over extended domain (R_plot)
        cache_dP = LP16EtaDerivativeKernelCache(
            L=L, delta_grid=delta, s_grid=s_grid, r_f=r_f, m_f=m_phi-1, eta=eta, beta=beta, nb_z2=160
        )
        Kd_R_plot = cache_dP.build_derivative_kernel(R_plot)
        Kd_0 = cache_dP.build_derivative_kernel(np.array([0.0]))
        D_dP_plot, C0_dP = DdP_from_derivative_kernel(
            R_values=R_plot, Kd_R=Kd_R_plot, Kd_0=Kd_0, delta_grid=delta, r_i=r_i, m_i=m_i-1, 
            sigma_i=sigma_i, Pbar_i=Pbar_i, full_mode="eq93_printed"
        )
        dP_dl2_plot = D_dP_plot / (D_dP_plot.max() if D_dP_plot.max() > 0 else 1.0)
        
        # Calculate spectra over extended domain
        M_P = sf_spectrum_proxy(k, R_plot, P_R_plot, sigma_ln=sigma_ln)
        M_dP = sf_spectrum_proxy(k, R_plot, np.abs(dP_dl2_plot), sigma_ln=sigma_ln)
        
        all_M_P.append(M_P)
        all_M_dP.append(M_dP)

    # Use fixed y-range for all spectra to keep panels comparable
    y_min_spectrum = 1e-12
    y_max_spectrum = 1e0

    # Reference lines for spectrum
    k_anchor = math.sqrt(kmin * kmax)
    kg = np.logspace(np.log10(kmin), np.log10(kmax), 200)
    
    # Row 1: Spectrum of D_u/2 (P_u)
    y_anchor_ref = all_M[0][np.argmin(np.abs(k - k_anchor))]
    yg_psi_k = y_anchor_ref * (kg / k_anchor) ** (-5/3)
    yg_phi_k = y_anchor_ref * (kg / k_anchor) ** (-1.1)
    ax2_du.loglog(kg, yg_psi_k, "-", color="blue", lw=3.0, zorder=5, alpha=0.8)
    ax2_du.loglog(kg, yg_phi_k, "-.", color="red", lw=3.0, zorder=5, alpha=0.8)
    
    for idx, (M, eta, c) in enumerate(zip(all_M, eta_list, colors)):
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax2_du.loglog(k, M, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(k, M)
        if len(xs) > 0:
            ax2_du_der.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)
    
    # Use same tick style as structure functions
    ax2_du.set_xscale("log")
    ax2_du.set_yscale("log")
    ax2_du.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax2_du.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    # Move x-axis ticks down
    ax2_du.tick_params(axis="x", pad=8)
    ax2_du.spines["top"].set_visible(False)
    ax2_du.spines["right"].set_visible(False)
    ax2_du.spines["left"].set_linewidth(2.5)
    ax2_du.spines["bottom"].set_linewidth(2.5)
    ax2_du.xaxis.set_major_formatter(LogFormatterMathtext())
    ax2_du.yaxis.set_major_formatter(LogFormatterMathtext())
    ax2_du.set_xlim(kmin, kmax)
    ax2_du.set_ylim(1e-12, 1e0)
    ax2_du.xaxis.set_major_locator(FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4,1e5,1e6,1e7]))
    # Fixed y-ticks: 1e0, 1e-2, ..., 1e-12
    ax2_du.yaxis.set_major_locator(
        FixedLocator([1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    )
    ax2_du.text(-0.10, 1.05, r"$P_u(k)$", transform=ax2_du.transAxes, fontsize=36, weight='bold')
    ax2_du.text(1.03, -0.02, r"$k$", transform=ax2_du.transAxes, fontsize=32, weight='bold')
    ax2_du.text(0.03, 0.92, rf"$m_i={m_psi:.3g},\ m_\Phi={m_phi:.3g}$", transform=ax2_du.transAxes, fontsize=20)
    ax2_du.text(0.03, 0.84, rf"$r_\phi/R_0={r_phi/R0:.3g}$", transform=ax2_du.transAxes, fontsize=20)
    ax2_du.legend(frameon=False, fontsize=20, loc="lower right", handlelength=1.5)
    
    ax2_du_der.axhline(-1.1, color="blue", lw=3.0, zorder=5)
    ax2_du_der.axhline(-5/3, color="red", lw=3.0, ls="-.", zorder=5)
    _setup_derivative_panel(ax2_du_der, kmin, kmax, r"$d\ln P_u/d\ln k$", r"$k$",
                           ylim=(-2, 0.1))

    # Row 2: Spectrum of P
    y_anchor_p = all_M_P[0][np.argmin(np.abs(k - k_anchor))]
    yg_psi_p = y_anchor_p * (kg / k_anchor) ** (-5/3)
    yg_phi_p = y_anchor_p * (kg / k_anchor) ** (-1.1)
    ax2_p.loglog(kg, yg_psi_p, "-", color="blue", lw=3.0, zorder=5, alpha=0.8)
    ax2_p.loglog(kg, yg_phi_p, "-.", color="red", lw=3.0, zorder=5, alpha=0.8)
    
    for idx, (M_P, eta, c) in enumerate(zip(all_M_P, eta_list, colors)):
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax2_p.loglog(k, M_P, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(k, M_P)
        if len(xs) > 0:
            ax2_p_der.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)
    
    # Use same tick style as structure functions
    ax2_p.set_xscale("log")
    ax2_p.set_yscale("log")
    ax2_p.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax2_p.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    # Move x-axis ticks down
    ax2_p.tick_params(axis="x", pad=8)
    ax2_p.spines["top"].set_visible(False)
    ax2_p.spines["right"].set_visible(False)
    ax2_p.spines["left"].set_linewidth(2.5)
    ax2_p.spines["bottom"].set_linewidth(2.5)
    ax2_p.xaxis.set_major_formatter(LogFormatterMathtext())
    ax2_p.yaxis.set_major_formatter(LogFormatterMathtext())
    ax2_p.set_xlim(kmin, kmax)
    ax2_p.set_ylim(1e-12, 1e0)
    ax2_p.xaxis.set_major_locator(
        FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
    )
    ax2_p.yaxis.set_major_locator(
        FixedLocator([1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    )
    ax2_p.text(-0.10, 1.05, r"$P(k)$", transform=ax2_p.transAxes, fontsize=36, weight='bold')
    ax2_p.text(1.03, -0.02, r"$k$", transform=ax2_p.transAxes, fontsize=32, weight='bold')
    ax2_p.legend(frameon=False, fontsize=20, loc="lower right", handlelength=1.5)
    
    ax2_p_der.axhline(-1.1, color="blue", lw=3.0, zorder=5)
    ax2_p_der.axhline(-5/3, color="red", lw=3.0, ls="-.", zorder=5)
    _setup_derivative_panel(ax2_p_der, kmin, kmax, r"$d\ln P/d\ln k$", r"$k$",
                           ylim=(-2, 0.1))

    # Row 3: Spectrum of dP/d(lambda^2)
    y_anchor_dp = all_M_dP[0][np.argmin(np.abs(k - k_anchor))]
    yg_psi_dp = y_anchor_dp * (kg / k_anchor) ** (-5/3)
    yg_phi_dp = y_anchor_dp * (kg / k_anchor) ** (-1.1)
    ax2_dp.loglog(kg, yg_psi_dp, "-", color="blue", lw=3.0, zorder=5, alpha=0.8)
    ax2_dp.loglog(kg, yg_phi_dp, "-.", color="red", lw=3.0, zorder=5, alpha=0.8)
    
    for idx, (M_dP, eta, c) in enumerate(zip(all_M_dP, eta_list, colors)):
        lw = 3.5 if eta == 0.0 else 3.0
        zorder = 100 if eta == 0.0 else 50
        label = r"$\eta = 0$" if eta == 0.0 else rf"$\eta = {eta:.3g}$"
        ax2_dp.loglog(k, M_dP, lw=lw, color=c, label=label, zorder=zorder, alpha=0.95)
        
        xs, sl = local_log_slope(k, M_dP)
        if len(xs) > 0:
            ax2_dp_der.semilogx(xs, sl, lw=lw, color=c, zorder=zorder, alpha=0.95)
    
    # Use same tick style as structure functions
    ax2_dp.set_xscale("log")
    ax2_dp.set_yscale("log")
    ax2_dp.tick_params(which="both", direction="in", labelsize=22, width=2.5, length=8)
    ax2_dp.tick_params(which="minor", direction="in", labelsize=18, width=1.5, length=5)
    # Move x-axis ticks down
    ax2_dp.tick_params(axis="x", pad=8)
    ax2_dp.spines["top"].set_visible(False)
    ax2_dp.spines["right"].set_visible(False)
    ax2_dp.spines["left"].set_linewidth(2.5)
    ax2_dp.spines["bottom"].set_linewidth(2.5)
    ax2_dp.xaxis.set_major_formatter(LogFormatterMathtext())
    ax2_dp.yaxis.set_major_formatter(LogFormatterMathtext())
    ax2_dp.set_xlim(kmin, kmax)
    ax2_dp.set_ylim(1e-12, 1e0)
    ax2_dp.xaxis.set_major_locator(
        FixedLocator([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
    )
    ax2_dp.yaxis.set_major_locator(
        FixedLocator([1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    )
    ax2_dp.text(-0.10, 1.05, r"$dP/d\lambda^2(k)$", transform=ax2_dp.transAxes, fontsize=36, weight='bold')
    ax2_dp.text(1.03, -0.02, r"$k$", transform=ax2_dp.transAxes, fontsize=32, weight='bold')
    ax2_dp.legend(frameon=False, fontsize=20, loc="lower right", handlelength=1.5)
    
    ax2_dp_der.axhline(-1.1, color="blue", lw=3.0, zorder=5)
    ax2_dp_der.axhline(-5/3, color="red", lw=3.0, ls="-.", zorder=5)
    _setup_derivative_panel(ax2_dp_der, kmin, kmax, r"$d\ln [dP/d\lambda^2]/d\ln k$", r"$k$",
                           ylim=(-2, 0.1))

    fig2.savefig(f"{out_prefix}_spectrum.png", dpi=300, bbox_inches="tight")
    fig2.savefig(f"{out_prefix}_spectrum.svg", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out_prefix}_spectrum.png and .svg")

    return all_Du2, all_xi_P, all_dP_dlambda2, all_M


if __name__ == "__main__":
    # Set sigma_phi = 1 (matching anzatz.py)
    sigma_phi = 1.0
    
    params = dict(
        A_P=1.0,
        R0=1.0,
        m_psi=5/3,
        r_phi=1.0,
        m_phi=1.1,
        lam=1,  # This will be computed from eta for each curve
        sigma_phi2=1.0,  # sigma_phi = 1, so sigma_phi2 = 1
        chi2_factor=4.0,
    )

    # Use the same eta_list as fig5.py: from 0 to 1
    eta_list = np.concatenate([[0.0], np.geomspace(3e-3, 1e0, 10)])
    
    # Compute corresponding lambda values for each eta
    # eta = 2 * sigma_phi * lambda^2, so lambda = sqrt(eta / (2 * sigma_phi))
    lambda_list = [np.sqrt(eta / (2.0 * sigma_phi)) if eta > 0 else 0.0 for eta in eta_list]
    
    # For backward compatibility, create sigma_phi2_list (all 1.0 since sigma_phi = 1)
    sigma_phi2_list = [1.0] * len(eta_list)
    
    print(f"Using eta_list (same as fig5.py): {[f'{eta:.3g}' for eta in eta_list]}")
    print(f"sigma_phi = {sigma_phi}")
    print(f"Corresponding lambda values: {[f'{lam:.6f}' for lam in lambda_list]}")
    print(f"Verification: eta = 2 * sigma_phi * lambda^2")
    for eta, lam in zip(eta_list, lambda_list):
        if eta > 0:
            eta_check = 2.0 * sigma_phi * lam * lam
            print(f"  eta={eta:.3g}, lambda={lam:.6f}, check={eta_check:.6f}")

    all_Du2, all_xi_P, all_dP_dlambda2, all_M = run_and_plot(
        params,
        Rmin=1e-5, Rmax=1e2, NR=300,  # Reduced from 1400 for speed
        Nk=400,  # Reduced from 900 for speed
        sigma_ln=0.35,
        out_prefix="sf_demo",
        sigma_phi2_list=sigma_phi2_list
    )

    print("\nFigures generated successfully!")
