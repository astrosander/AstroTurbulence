import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, FixedLocator
from matplotlib.colors import LinearSegmentedColormap, LogNorm

TRAPZ = getattr(np, "trapezoid", np.trapz)

def cumtrapz0(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    out = np.empty_like(y, dtype=float)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out

def t_even_from_f(f, z):
    h = cumtrapz0(f, z)
    j = cumtrapz0(f * z, z)
    return 2.0 * (z * h - j)

def xhat_kernel(R, dz, r, m):
    return (r ** m) / (r ** m + (R * R + dz * dz) ** (m / 2.0))

def make_uniform_grid(L, nz):
    return np.linspace(0.0, L, nz)

def trap_weights_uniform(z):
    w = np.ones_like(z)
    w[0] = 0.5
    w[-1] = 0.5
    w *= (z[-1] - z[0]) / (len(z) - 1)
    return w

def prepare_difference_index(nz):
    i = np.arange(nz)
    return np.abs(i[:, None] - i[None, :])

def structure_functions_case(ri, mi, rphi, mphi, L, etas, xvals, nz=360, sigma_i=1.0, sigma_phi=1.0):
    z = make_uniform_grid(L, nz)
    w = trap_weights_uniform(z)
    W = w[:, None] * w[None, :]
    didx = prepare_difference_index(nz)
    dzdiff = z[didx]
    Rall = np.concatenate(([0.0], xvals * ri))
    xi_i_diff = np.array([sigma_i**2 * xhat_kernel(R, z, ri, mi) for R in Rall])
    xi_phi_diff = np.array([sigma_phi**2 * xhat_kernel(R, z, rphi, mphi) for R in Rall])
    T_phi = np.array([t_even_from_f(f, z) for f in xi_phi_diff])
    V = T_phi[0]
    sigma_RM = np.sqrt(V[-1])
    tvals = np.array(etas, float) / (2.0 * sigma_RM)
    n_eta = len(etas)
    nR = len(Rall)
    C_P = np.empty((n_eta, nR), dtype=float)
    C_dP = np.empty((n_eta, nR), dtype=float)
    V1 = V[:, None]
    V2 = V[None, :]
    for k in range(nR):
        T = T_phi[k]
        J = 0.5 * (T[:, None] + T[None, :] - T[didx])
        C12 = J
        xi_i = xi_i_diff[k][didx]
        baseV = V1 + V2
        for q, t in enumerate(tvals):
            tt = t * t
            F = -2.0 * tt * baseV + 4.0 * tt * C12
            E = np.exp(F)
            Kp = xi_i * E
            C_P[q, k] = np.sum(W * Kp)
            d1 = 4.0 * t * (C12 - V1)
            d2 = 4.0 * t * (C12 - V2)
            Kd = xi_i * E * (d1 * d2 + 4.0 * C12)
            C_dP[q, k] = np.sum(W * Kd)
    D_P = 2.0 * (C_P[:, [0]] - C_P[:, 1:])
    D_dP = 2.0 * (C_dP[:, [0]] - C_dP[:, 1:])
    D_P_inf = 2.0 * C_P[:, 0]
    D_dP_inf = 2.0 * C_dP[:, 0]
    return {
        "x": xvals,
        "z": z,
        "sigma_RM": sigma_RM,
        "etas": np.array(etas, float),
        "lambda2": tvals,
        "D_P": D_P,
        "D_dP": D_dP,
        "D_P_norm": D_P / D_P_inf[:, None],
        "D_dP_norm": D_dP / D_dP_inf[:, None],
        "D_P_inf": D_P_inf,
        "D_dP_inf": D_dP_inf,
        "DdP_fig9_units": D_dP / (sigma_i**2 * sigma_phi**2 * L**3),
        "DP_fig5_units": D_P / (sigma_i**2 * L**2)
    }

def local_slope(x, y):
    lx = np.log10(x)
    ly = np.log10(np.maximum(y, 1e-300))
    return np.gradient(ly, lx, axis=-1)

def projected_scale(r, m, L):
    mbar = min(float(m), 1.0)
    return r * (L / r) ** ((1.0 - mbar) / (1.0 + mbar))

ri = 1.0
L = 100.0 * ri
# Use same eta values as p.py
num_etas = 8
etas = np.concatenate(([0], np.geomspace(0.1, 10, num_etas-1)))
xvals = np.logspace(-4, 2, 140)

# Cases for new figure layout: m_i > m_phi (using m_i=4/5, m_phi=1/3)
cases_fig1 = [
    {"name": r"$r_\phi=r_i=1,\ m_i=4/5,\ m_\phi=1/3$", "mi": 4/5, "mphi": 1/3, "ri": ri, "rphi": ri},
    {"name": r"$r_\phi=0.1,\ r_i=1,\ m_i=4/5,\ m_\phi=1/3$", "mi": 4/5, "mphi": 1/3, "ri": ri, "rphi": 0.1 * ri}
]

# Cases for log derivative plot
cases_fig2 = [
    {"name": r"$r_\phi=r_i=1,\ m_i=4/5,\ m_\phi=1/3$", "mi": 4/5, "mphi": 1/3, "ri": ri, "rphi": ri},
    {"name": r"$r_\phi=0.1,\ r_i=1,\ m_i=4/5,\ m_\phi=1/3$", "mi": 4/5, "mphi": 1/3, "ri": ri, "rphi": 0.1 * ri}
]

results_fig1 = [structure_functions_case(c["ri"], c["mi"], c["rphi"], c["mphi"], L, etas, xvals, nz=340) for c in cases_fig1]
results_fig2 = [structure_functions_case(c["ri"], c["mi"], c["rphi"], c["mphi"], L, etas, xvals, nz=340) for c in cases_fig2]

# Create colormap from blue to red (matching p.py)
cmap = LinearSegmentedColormap.from_list('blue_to_red', ['#0000FF', '#FF0000'])
# Normalize eta values logarithmically (handle eta=0 by using small epsilon)
eta_min = min([e for e in etas if e > 0]) if any(e > 0 for e in etas) else 1e-6
norm = LogNorm(vmin=eta_min, vmax=max(etas))

# Figure 1: Two columns (P on left, dP on right), two rows
fig1, axs1 = plt.subplots(2, 2, figsize=(13.5, 9.5))

# Left column: P plots
for row, (c, res) in enumerate(zip(cases_fig1, results_fig1)):
    ax = axs1[row, 0]
    for i, eta in enumerate(res["etas"]):
        if eta == 0:
            color = cmap(0.0)  # Use blue for eta=0
        else:
            color = cmap(norm(eta))
        ax.loglog(res["x"], np.maximum(res["D_P_norm"][i], 1e-12), color=color, lw=2.0, label=rf"$\eta={eta:.3f}$")
    rp_i = projected_scale(c["ri"], c["mi"], L) / c["ri"]
    # When r_i = r_phi, normalize by r_phi so lines are together; otherwise normalize by r_i
    if abs(c["ri"] - c["rphi"]) < 1e-10:
        rp_f = projected_scale(c["rphi"], c["mphi"], L) / c["rphi"]
    else:
        rp_f = projected_scale(c["rphi"], c["mphi"], L) / c["ri"]
    ax.vlines(rp_i, 1e-5, 1e-3, colors="crimson", linestyles=":", lw=2.2, alpha=0.8, label=r"$r_{p,i}$")
    ax.vlines(rp_f, 1e-5, 1e-3, colors="darkblue", linestyles="--", lw=2.2, alpha=0.8, label=r"$r_{p,\phi}$")
    x0 = np.array([1e-4, 3e-2])
    y0 = 2e-4 * (x0 / x0[0]) ** min(c["mi"], 1.0)
    y0_phi = 2e-4 * (x0 / x0[0]) ** min(c["mphi"], 1.0)
    ax.loglog(x0, y0, color="darkgreen", lw=2.4, alpha=0.8, label=r"$R^{m_i}$")
    ax.loglog(x0, y0_phi, color="darkmagenta", lw=2.4, alpha=0.8, ls="--", label=r"$R^{m_\phi}$")
    ax.set_xlim(1e-4, 1e2)
    ax.set_ylim(1e-5, 2)
    ax.set_title(c["name"], fontsize=13)
    ax.tick_params(which="both", direction="in", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

# Right column: dP plots
for row, (c, res) in enumerate(zip(cases_fig1, results_fig1)):
    ax = axs1[row, 1]
    for i, eta in enumerate(res["etas"]):
        if eta == 0:
            color = cmap(0.0)  # Use blue for eta=0
        else:
            color = cmap(norm(eta))
        ax.loglog(res["x"], np.maximum(res["DdP_fig9_units"][i], 1e-12), color=color, lw=2.0, label=rf"$\eta={eta:.3f}$")
    x0 = np.array([1e-4, 3e-2])
    y1 = 3e-5 * (x0 / x0[0]) ** min(c["mphi"], 1.0)
    y2 = 2e-3 * (x0 / x0[0]) ** min(c["mi"], 1.0)
    ax.loglog(x0, y1, color="darkmagenta", lw=2.4, ls="--", alpha=0.8, label=r"$R^{m_\phi}$")
    ax.loglog(x0, y2, color="darkorange", lw=2.4, ls=":", alpha=0.8, label=r"$R^{m_i}$")
    rp_i = projected_scale(c["ri"], c["mi"], L) / c["ri"]
    # When r_i = r_phi, normalize by r_phi so lines are together; otherwise normalize by r_i
    if abs(c["ri"] - c["rphi"]) < 1e-10:
        rp_f = projected_scale(c["rphi"], c["mphi"], L) / c["rphi"]
    else:
        rp_f = projected_scale(c["rphi"], c["mphi"], L) / c["ri"]
    ax.vlines(rp_i, 1e-3, 1e-1, colors="crimson", linestyles=":", lw=2.2, alpha=0.8, label=r"$r_{p,i}$")
    ax.vlines(rp_f, 1e-3, 1e-1, colors="darkblue", linestyles="--", lw=2.2, alpha=0.8, label=r"$r_{p,\phi}$")
    ax.set_xlim(1e-4, 1e2)
    ax.set_ylim(1e-3, 10)
    ax.set_title(c["name"], fontsize=13)
    ax.tick_params(which="both", direction="in", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.yaxis.set_major_locator(FixedLocator([1, 1e-3, 1e-6, 1e-9, 1e-12]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

axs1[0, 0].set_ylabel(r"$D_P(R)/D_P(\infty)$", fontsize=14)
axs1[1, 0].set_ylabel(r"$D_P(R)/D_P(\infty)$", fontsize=14)
axs1[0, 1].set_ylabel(r"$D_{dP}(R)/(\sigma_i^2\sigma_\phi^2 L^3)$", fontsize=14)
axs1[1, 1].set_ylabel(r"$D_{dP}(R)/(\sigma_i^2\sigma_\phi^2 L^3)$", fontsize=14)
axs1[1, 0].set_xlabel(r"$R/r_i$", fontsize=14)
axs1[1, 1].set_xlabel(r"$R/r_i$", fontsize=14)
for ax in axs1.ravel():
    ax.legend(loc="lower right", fontsize=9, frameon=False, ncol=1)
fig1.tight_layout()
fig1.savefig("LP16_compare_DP_multiple_eta.png", dpi=300, bbox_inches="tight")
fig1.savefig("LP16_compare_DP_multiple_eta.svg", bbox_inches="tight")
plt.close(fig1)

# Figure 2: Log derivative plot with 2 cases
fig2, axs2 = plt.subplots(1, 2, figsize=(13.5, 4.5))

for col, (c, res) in enumerate(zip(cases_fig2, results_fig2)):
    ax = axs2[col]
    for i, eta in enumerate(res["etas"]):
        if eta == 0:
            color = cmap(0.0)  # Use blue for eta=0
        else:
            color = cmap(norm(eta))
        sp = local_slope(res["x"], res["D_P_norm"][i][None, :])[0]
        sd = local_slope(res["x"], np.maximum(res["DdP_fig9_units"][i], 1e-300)[None, :])[0]
        ax.semilogx(res["x"], sp, color=color, lw=2.0)
        ax.semilogx(res["x"], sd, color=color, lw=1.5, ls="--")
    mi_slope = min(c["mi"], 1.0)
    mphi_slope = min(c["mphi"], 1.0)
    ax.axhline(mi_slope, color="crimson", lw=2.2, ls=":", alpha=0.8, label=fr"$m_i={mi_slope:.2f}$")
    ax.axhline(mphi_slope, color="darkblue", lw=2.2, ls="--", alpha=0.8, label=fr"$m_\phi={mphi_slope:.2f}$")
    ax.set_xlim(1e-4, 1e2)
    ax.set_ylim(0, 1.2)
    ax.set_title(c["name"], fontsize=13)
    ax.tick_params(which="both", direction="in", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())

axs2[0].set_ylabel(r"local slope $d\log D / d\log R$", fontsize=14)
axs2[0].set_xlabel(r"$R/r_i$", fontsize=14)
axs2[1].set_xlabel(r"$R/r_i$", fontsize=14)
for ax in axs2:
    ax.legend(loc="upper right", fontsize=9, frameon=False)
fig2.tight_layout()
fig2.savefig("LP16_compare_slopes_P_vs_dP_multiple_eta.png", dpi=300, bbox_inches="tight")
fig2.savefig("LP16_compare_slopes_P_vs_dP_multiple_eta.svg", bbox_inches="tight")
plt.close(fig2)

for c, res in zip(cases_fig1, results_fig1):
    print(c["name"])
    print("sigma_RM =", res["sigma_RM"])
    for eta, lam2, dp0, dd0 in zip(res["etas"], res["lambda2"], res["D_P_inf"], res["D_dP_inf"]):
        print(f"  eta={eta:g}  lambda^2={lam2:.6e}  D_P(inf)={dp0:.6e}  D_dP(inf)={dd0:.6e}")