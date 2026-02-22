#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

TRAPZ = getattr(np, "trapezoid", np.trapz)

def xi_i_eq30(R, dz, r_i, m, sigma_i=1.0, Pbar_i=0.0):
    rr = (R * R + dz * dz) ** (m / 2.0)
    return abs(Pbar_i) ** 2 + sigma_i**2 * (r_i**m) / (r_i**m + rr)

def make_z_grid(L, n_log1=220, n_log2=200, n_lin=140, zmin=1e-12, zmid1=1e-4, zmid2=1.0):
    z1 = np.geomspace(zmin, zmid1, n_log1)
    z2 = np.geomspace(zmid1, zmid2, n_log2)
    z3 = np.linspace(zmid2, L, n_lin)
    return np.unique(np.concatenate(([0.0], z1, z2, z3)))

def make_s_grid_for_rm(L, n_log1=150, n_log2=120, n_lin=90, zmin=1e-12, zmid1=1e-4, zmid2=1.0):
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
                              nR=130,
                              nb_z2=160):
    L = 100.0
    sigma_i = 1.0
    Pbar_i = 0.0
    delta = make_z_grid(L)
    s_grid = make_s_grid_for_rm(L)
    x_plot = np.logspace(-5, 2, nR)

    panels = [
        dict(name="TL", ri=1.0, mi=4/5, rf=0.1, mf=1/3, col="left"),
        dict(name="TR", ri=1.0, mi=1/3, rf=0.1, mf=4/5, col="right"),
        dict(name="BL", ri=1.0, mi=2/3, rf=0.1, mf=2/3, col="left"),
        dict(name="BR", ri=1.0, mi=1/3, rf=1.0, mf=4/5, col="right"),
    ]

    fig = plt.figure(figsize=(14.5, 10.2))
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.985, top=0.93, bottom=0.12, wspace=0.24, hspace=0.24, height_ratios=[1.0, 1.0, 0.72])
    axs = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]
    axs_s = [fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]

    cmap = plt.cm.viridis
    colors = [cmap(v) for v in np.linspace(0.08, 0.95, len(eta_list))]

    panel_titles = {
        "TL": r"$m_\phi=1/3,\ m=4/5,\ r_\phi/r_i=0.1$",
        "TR": r"$m_\phi=4/5,\ m=1/3,\ r_\phi/r_i=0.1$",
        "BL": r"$m_\phi=2/3,\ m=2/3,\ r_\phi/r_i=0.1$",
        "BR": r"$m_\phi=4/5,\ m=1/3,\ r_\phi/r_i=1$",
    }

    t0 = time.time()
    all_curves = {}

    for ax, p in zip(axs, panels):
        r_m, m_m, r_M, m_M = classify_lengths(p["ri"], p["mi"], p["rf"], p["mf"])
        R_vals = x_plot * r_M

        if p["col"] == "left":
            col_scale = (L / r_M) ** m_M
        else:
            col_scale = (L / r_m) ** m_m

        panel_key = p["name"]
        all_curves[panel_key] = []

        for eta, c in zip(eta_list, colors):
            cache = LP16EtaDerivativeKernelCache(
                L=L,
                delta_grid=delta,
                s_grid=s_grid,
                r_f=p["rf"],
                m_f=p["mf"],
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
                r_i=p["ri"],
                m_i=p["mi"],
                sigma_i=sigma_i,
                Pbar_i=Pbar_i,
                full_mode=full_mode
            )
            y = (D / (L**3)) * col_scale
            all_curves[panel_key].append((eta, x_plot.copy(), y.copy()))
            ax.loglog(x_plot, y, color=c, lw=2.2)

        Rp_i = projected_correlation_length(p["ri"], p["mi"], L) / r_M
        Rp_f = projected_correlation_length(p["rf"], p["mf"], L) / r_M
        ax.vlines(Rp_i, 1e-10, 1e-6, color="#E69500", lw=1.8, linestyles=(0, (1.5, 2.0)))
        ax.vlines(Rp_f, 1e-10, 1e-6, color="#6B8EC7", lw=1.8, linestyles="--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-5, 1e2)
        ax.set_ylim(1e-10, 1e3)
        ax.tick_params(which="both", direction="in", labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 1e2]))
        ax.yaxis.set_major_locator(FixedLocator([1e2, 1e-1, 1e-4, 1e-7, 1e-10]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

        ax.text(-0.10, 1.04, r"$D_{\mathrm{dP}}(R)$", transform=ax.transAxes, fontsize=20)
        ax.text(1.03, -0.02, r"$R/r_M$", transform=ax.transAxes, fontsize=20)
        ax.text(0.03, 0.90, panel_titles[panel_key], transform=ax.transAxes, fontsize=15)
        ax.text(0.03, 0.82, rf"$\beta=\bar\phi/\sigma_\phi={beta:g}$", transform=ax.transAxes, fontsize=13)

        if p["col"] == "left":
            ax.text(0.03, 0.74, r"scaled by $(L/r_M)^{m_M}$", transform=ax.transAxes, fontsize=13)
        else:
            ax.text(0.03, 0.74, r"scaled by $(L/r_m)^{m_m}$", transform=ax.transAxes, fontsize=13)

        ax.text(0.66, 0.10, r"$R_{P,i}$", color="#E69500", transform=ax.transAxes, fontsize=13)
        ax.text(0.80, 0.10, r"$R_{P,\phi}$", color="#6B8EC7", transform=ax.transAxes, fontsize=13)

    slope_panel_map = [("TL", axs_s[0]), ("TR", axs_s[1])]
    for key, ax in slope_panel_map:
        for (eta, x, y), c in zip(all_curves[key], colors):
            xs, s = local_log_slope(x, y)
            ax.semilogx(xs, s, color=c, lw=1.9)
        p = [pp for pp in panels if pp["name"] == key][0]
        r_m, m_m, r_M, m_M = classify_lengths(p["ri"], p["mi"], p["rf"], p["mf"])
        ax.axhline(1.0 + m_m, color="black", lw=1.4, ls="--")
        ax.axhline(1.0 + m_M, color="black", lw=1.4)
        ax.text(1.4e-5, 1.0 + m_M + 0.03, rf"$1+m_M={1+m_M:.3g}$", fontsize=12)
        ax.text(1.4e-5, 1.0 + m_m + 0.03, rf"$1+m_m={1+m_m:.3g}$", fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1e2)
        ax.set_ylim(0.8, 2.05)
        ax.tick_params(which="both", direction="in", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 1e2]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.text(-0.02, 1.04, r"$d\ln D_{\mathrm{dP}}/d\ln R$", transform=ax.transAxes, fontsize=18)
        ax.text(1.03, -0.02, r"$R/r_M$", transform=ax.transAxes, fontsize=18)

    handles = [plt.Line2D([0], [0], color=c, lw=2.5) for c in colors]
    labels = [rf"$\eta={eta:g}$" for eta in eta_list]
    fig.legend(handles, labels, ncol=min(4, len(labels)), frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.52, 0.995))

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
        eta_list=np.concatenate([[0.0], np.geomspace(1e-4, 1e0, 20)]),#(0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
        nR=130,
        nb_z2=160
    )

if __name__ == "__main__":
    run()