#!/usr/bin/env python3

import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Tuple, Dict

@dataclass
class Params:
    nx: int = 256
    ny: int = 256
    nz_screen: int = 128
    nz_halo: int = 1
    dx_pc: float = 2.0

    beta_Bz_screen: float = 11/3
    beta_ne_screen: float = 11/3
    beta_Bx_halo: float = 11/3
    beta_By_halo: float = 11/3

    amp_Bz_screen: float = 1.0
    amp_ne_screen: float = 1.0
    amp_Bx_halo: float = 1.0
    amp_By_halo: float = 1.0

    mean_ne_screen: float = 0.2
    mean_Bz_screen: float = 0.0

    use_sqB_for_Pi: bool = True
    normalize_Pi_amplitude: bool = True

    C_RM: float = 0.81

    freqs_MHz: Tuple[float, ...] = (100, 2000, 2400, 3000,  3600, 4000, 4800, 6000, 10000, 100000, 10**10)

    r_min_pix: int = 2
    r_max_pix: int = 80
    n_r_bins: int = 40

    fit_r_min_pix: int = 4
    fit_r_max_pix: int = 32

    seed_Bz: int = 1
    seed_ne: int = 2
    seed_Bx: int = 3
    seed_By: int = 4

    out_prefix: str = "img/freq_regime_separation"
    make_plots: bool = True
    dpi: int = 140

PRM = Params()

def _k_grid(nx: int, ny: int, nz: int, dx: float):
    kx = 2.0*np.pi*nfft.fftfreq(nx, d=dx)
    ky = 2.0*np.pi*nfft.fftfreq(ny, d=dx)
    kz = 2.0*np.pi*nfft.fftfreq(nz, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    return KX, KY, KZ, K

def generate_powerlaw_3d(shape, beta: float, amp: float=1.0, seed: int=0, dx: float=1.0) -> np.ndarray:
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)
    _, _, _, K = _k_grid(nx, ny, nz, dx)
    phase = rng.uniform(0, 2*np.pi, size=(nx, ny, nz))
    with np.errstate(divide='ignore'):
        Pk = np.where(K > 0, K**(-beta), 0.0)
    Ak = np.sqrt(Pk)
    fk = Ak * (np.cos(phase) + 1j*np.sin(phase))
    fk[0, 0, 0] = 0.0
    field = np.real(nfft.ifftn(fk))
    field -= np.mean(field)
    std = np.std(field)
    if std > 0:
        field = field / std
    return amp * field

def make_faraday_screen(PRM: Params) -> Dict[str, np.ndarray]:
    nx, ny, nzs = PRM.nx, PRM.ny, PRM.nz_screen
    dx = PRM.dx_pc
    ne = generate_powerlaw_3d((nx, ny, nzs), PRM.beta_ne_screen, PRM.amp_ne_screen, PRM.seed_ne, dx)
    Bz = generate_powerlaw_3d((nx, ny, nzs), PRM.beta_Bz_screen, PRM.amp_Bz_screen, PRM.seed_Bz, dx)
    ne = ne + PRM.mean_ne_screen
    Bz = Bz + PRM.mean_Bz_screen
    RM = PRM.C_RM * np.sum(ne * Bz, axis=2) * PRM.dx_pc
    return {"ne": ne, "Bz": Bz, "RM": RM}

def make_halo_polarization(PRM: Params) -> Dict[str, np.ndarray]:
    nx, ny, nzh = PRM.nx, PRM.ny, PRM.nz_halo
    dx = PRM.dx_pc
    Bx = generate_powerlaw_3d((nx, ny, nzh), PRM.beta_Bx_halo, PRM.amp_Bx_halo, PRM.seed_Bx, dx)
    By = generate_powerlaw_3d((nx, ny, nzh), PRM.beta_By_halo, PRM.amp_By_halo, PRM.seed_By, dx)
    if nzh > 1:
        bx = np.sum(Bx, axis=2) / nzh
        by = np.sum(By, axis=2) / nzh
    else:
        bx = Bx[..., 0]
        by = By[..., 0]
    Bc = bx + 1j*by
    if PRM.use_sqB_for_Pi:
        Pi = Bc**2
    else:
        Pi = np.exp(2j*np.angle(Bc))
    if PRM.normalize_Pi_amplitude:
        amp = np.abs(Pi)
        amp[amp == 0] = 1.0
        Pi = Pi / amp
    return {"Bx": Bx, "By": By, "Pi": Pi}

def radial_profile_from_map(Map: np.ndarray, r_edges: np.ndarray):
    ny, nx = Map.shape
    y = np.fft.fftfreq(ny) * ny
    x = np.fft.fftfreq(nx) * nx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    r = R.ravel()
    m = Map.ravel()
    inds = np.digitize(r, r_edges) - 1
    nb = len(r_edges) - 1
    prof = np.zeros(nb, dtype=float)
    for i in range(nb):
        mask = (inds == i)
        prof[i] = np.nan if not np.any(mask) else m[mask].mean()
    r_centers = 0.5*(r_edges[:-1] + r_edges[1:])
    return r_centers, prof

def corr2d_periodic(field: np.ndarray) -> np.ndarray:
    F = nfft.fft2(field)
    corr = np.real(nfft.ifft2(F * np.conj(F)))
    corr = np.fft.fftshift(corr)
    corr /= np.max(corr)
    return corr

def structure_function_from_map(field: np.ndarray, r_edges: np.ndarray):
    f = field - np.mean(field)
    var = np.var(f)
    corr_map = corr2d_periodic(f)
    D_map = 2 * (var - corr_map*var)
    return radial_profile_from_map(D_map, r_edges)

def angle_structure_function_from_P(P: np.ndarray, r_edges: np.ndarray):
    eps = 1e-12
    Pn = P / np.maximum(np.abs(P), eps)
    S_map = np.real(corr2d_periodic(Pn))
    Dphi_map = 0.5 * (1.0 - S_map)
    return radial_profile_from_map(Dphi_map, r_edges)

def fit_loglog_slope(r: np.ndarray, y: np.ndarray, rmin: float, rmax: float):
    mask = (r >= rmin) & (r <= rmax) & np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan
    X = np.log(r[mask]); Y = np.log(y[mask])
    A = np.vstack([X, np.ones_like(X)]).T
    alpha, logC = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(alpha), float(np.exp(logC))

def simulateAnd_analyze(PRM: Params):
    scr = make_faraday_screen(PRM)
    halo = make_halo_polarization(PRM)
    RM = scr["RM"]; Pi = halo["Pi"]

    r_edges = np.linspace(PRM.r_min_pix, PRM.r_max_pix, PRM.n_r_bins+1)
    r_centers = 0.5*(r_edges[:-1] + r_edges[1:])

    r_RM, Dphi_RM = structure_function_from_map(RM, r_edges)
    sigma_RM2 = float(np.var(RM))

    c = 299_792_458.0
    freqs_Hz = np.array(PRM.freqs_MHz, float) * 1e6
    lambdas_m = c / freqs_Hz

    Dphi_lambda = []
    slopes = []
    amps = []
    for lam in lambdas_m:
        phase = 2.0 * (lam**2) * RM
        P = Pi * np.exp(1j * phase)
        r_phi, Dphi = angle_structure_function_from_P(P, r_edges)
        Dphi_lambda.append(Dphi)
        a, C = fit_loglog_slope(r_phi, Dphi, PRM.fit_r_min_pix, PRM.fit_r_max_pix)
        slopes.append(a); amps.append(C)
    Dphi_lambda = np.array(Dphi_lambda); slopes = np.array(slopes); amps = np.array(amps)

    with np.errstate(invalid="ignore"):
         Dphi_lambda_avg = np.nanmean(Dphi_lambda, axis=1)

    S_lambda = 1.0 - 2.0 * Dphi_lambda_avg
    valid = np.isfinite(S_lambda) & (S_lambda > 0) & np.isfinite(lambdas_m) & (lambdas_m > 0)
    if np.count_nonzero(valid) >= 3:
         X = np.log(lambdas_m[valid])
         Y = np.log(S_lambda[valid])
         A = np.vstack([X, np.ones_like(X)]).T
         lin_slope, lin_intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
         lin_slope = float(lin_slope); lin_intercept = float(lin_intercept)
    else:
         lin_slope, lin_intercept = np.nan, np.nan

    moderate = valid & (S_lambda > 0.1)
    if np.count_nonzero(moderate) >= 3:
         X2 = np.log(lambdas_m[moderate]); Y2 = np.log(S_lambda[moderate])
         A2 = np.vstack([X2, np.ones_like(X2)]).T
         lin_slope_mod, lin_intercept_mod = np.linalg.lstsq(A2, Y2, rcond=None)[0]
         lin_slope_mod = float(lin_slope_mod); lin_intercept_mod = float(lin_intercept_mod)
    else:
         lin_slope_mod, lin_intercept_mod = np.nan, np.nan

    R0 = max(PRM.fit_r_min_pix, PRM.r_min_pix)
    idx0 = np.argmin(np.abs(r_RM - R0))
    DPhi_R0 = float(Dphi_RM[idx0]) if np.isfinite(Dphi_RM[idx0]) else np.nan
    eps = 0.1
    lam_high_m = (eps / (2.0 * DPhi_R0))**0.25 if (np.isfinite(DPhi_R0) and DPhi_R0 > 0) else np.nan
    lam_low_m  = (10.0 / (4.0 * sigma_RM2))**0.25 if (sigma_RM2 > 0) else np.nan
    freq_high_MHz = (c / lam_high_m) / 1e6 if (np.isfinite(lam_high_m) and lam_high_m > 0) else np.nan
    freq_low_MHz  = (c / lam_low_m)  / 1e6 if (np.isfinite(lam_low_m)  and lam_low_m  > 0) else np.nan

    return {
        "params": asdict(PRM),
        "r_centers_pix": r_centers,
        "RM_Dphi": Dphi_RM,
        "lambdas_m": lambdas_m,
        "freqs_MHz": np.array(PRM.freqs_MHz, float),
        "Dphi_lambda": Dphi_lambda,
        "slopes": slopes,
        "amps": amps,
        "sigma_RM2": sigma_RM2,
        "Dphi_lambda_avg": Dphi_lambda_avg,
        "S_lambda": S_lambda,
        "lin_slope_all": lin_slope,
        "lin_intercept_all": lin_intercept,
        "lin_slope_moderate": lin_slope_mod,
        "lin_intercept_moderate": lin_intercept_mod,

        "DPhi_R0": DPhi_R0,
        "freq_high_MHz": freq_high_MHz,
        "freq_low_MHz": freq_low_MHz,
    }

def plot_results(res, PRM: Params) -> None:
    r = res["r_centers_pix"]
    plt.figure(figsize=(6,4))
    plt.loglog(r, res["RM_Dphi"], marker='o', linestyle='-')
    plt.xlabel("R (pixels)")
    plt.ylabel("$D_\\Phi(R)$")
    plt.title("RM structure function")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PRM.out_prefix}_RM_structure.pdf", dpi=PRM.dpi);plt.savefig(f"{PRM.out_prefix}_RM_structure.png", dpi=PRM.dpi)

    plt.figure(figsize=(6,4))
    for i, fMHz in enumerate(res["freqs_MHz"]):
        plt.loglog(r, res["Dphi_lambda"][i], marker='o', markersize=2,linestyle='-', label=f"{int(fMHz)} MHz")
    plt.xlabel("R (pixels)")
    plt.ylabel("$D_\\phi(R, \\lambda)$")
    plt.title("Polarization-angle structure vs R at different ν")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PRM.out_prefix}_Dphi_vs_R.pdf", dpi=PRM.dpi);plt.savefig(f"{PRM.out_prefix}_Dphi_vs_R.png", dpi=PRM.dpi)

    plt.figure(figsize=(6,4))
    for i, fMHz in enumerate(res["freqs_MHz"]):
        lam = res["lambdas_m"][i]
        plt.loglog(r, res["Dphi_lambda"][i]/(lam**4 + 1e-30), marker='o', linestyle='-', label=f"{int(fMHz)} MHz")
    plt.xlabel("R (pixels)")
    plt.ylabel("$D_\\phi(R, \\lambda) / \\lambda^4$")
    plt.title("Check $λ^4$ scaling (Faraday dominated)")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PRM.out_prefix}_Dphi_over_lambda4.pdf", dpi=PRM.dpi);plt.savefig(f"{PRM.out_prefix}_Dphi_over_lambda4.png", dpi=PRM.dpi)

    plt.figure(figsize=(6,4))
    plt.semilogx(res["freqs_MHz"], res["slopes"], marker='o', linestyle='-')
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Slope $\\alpha$ ($D_\\phi \\propto R^\\alpha$)")
    plt.title("Measured inertial-range slope vs $\\nu$")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PRM.out_prefix}_slope_vs_freq.pdf", dpi=PRM.dpi);plt.savefig(f"{PRM.out_prefix}_slope_vs_freq.png", dpi=PRM.dpi)

    plt.figure(figsize=(6,4))
    lam = res["lambdas_m"]
    Dbar = res["Dphi_lambda_avg"]
    plt.loglog(lam, np.maximum(Dbar, 1e-30), marker='o', linestyle='-')
    plt.xlabel("Wavelength $\\lambda$")
    plt.ylabel(r"$\overline{D_\varphi}(\lambda)$  (avg over R)")
    plt.title(r"Angle structure (avg over $R$) vs $\lambda$")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PRM.out_prefix}_Dphi_avg_vs_lambda.pdf", dpi=PRM.dpi);plt.savefig(f"{PRM.out_prefix}_Dphi_avg_vs_lambda.png", dpi=PRM.dpi)

    S = np.maximum(res["S_lambda"], 1e-300)
    mask = (S > 0) & np.isfinite(S) & (lam > 0)
    X = np.log(lam[mask]); Y = np.log(S[mask])

    plt.figure(figsize=(6,4))
    plt.plot(X, Y, marker='o', linestyle='None')
    m = res["lin_slope_all"]; b = res["lin_intercept_all"]
    if np.isfinite(m) and np.isfinite(b):
         xline = np.linspace(np.min(X), np.max(X), 200)
         yline = m * xline + b
         plt.plot(xline, yline, linestyle='-', label=f"fit: slope={m:.2f}")
    m2 = res.get("lin_slope_moderate", np.nan); b2 = res.get("lin_intercept_moderate", np.nan)
    if np.isfinite(m2) and np.isfinite(b2):
         yline2 = m2 * xline + b2
         plt.plot(xline, yline2, linestyle='--', label=f"moderate-only slope={m2:.2f}")
    plt.xlabel(r"$\log \lambda$")
    plt.title(r"Linearization: expect $\propto \lambda^4$ (slope $\approx 4$)")
    plt.grid(True, alpha=0.3)
    if np.isfinite(m) or np.isfinite(m2):
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PRM.out_prefix}_linearization_log1m2Dphi_vs_lambda.pdf", dpi=PRM.dpi);plt.savefig(f"{PRM.out_prefix}_linearization_log1m2Dphi_vs_lambda.png", dpi=PRM.dpi)

def save_npz(res, PRM: Params) -> str:
    path = f"{PRM.out_prefix}_results.npz"
    np.savez_compressed(path, **res)
    return path

def main():
    res = simulateAnd_analyze(PRM)
    print("=== Frequency Regime Separation (LP16-based) ===")
    print(f"Grid: {PRM.nx}x{PRM.ny}, nz_screen={PRM.nz_screen}, dx={PRM.dx_pc} pc")
    print(f"β_3D screen (Bz, n_e) = ({PRM.beta_Bz_screen:.3f}, {PRM.beta_ne_screen:.3f}); "
        f"β_3D halo (Bx, By) = ({PRM.beta_Bx_halo:.3f}, {PRM.beta_By_halo:.3f})")
    print(f"σ_Φ^2 = {res['sigma_RM2']:.4e} [rad^2 m^-4]")
    R0 = max(PRM.fit_r_min_pix, PRM.r_min_pix)
    print(f"D_Φ(R0={R0} px) ≈ {res['DPhi_R0']:.4e}")
    print("Slope α(ν) from D_φ ∝ R^α:")
    for fMHz, a in zip(res["freqs_MHz"], res["slopes"]):
        print(f"  {int(fMHz):4d} MHz : α = {a:.3f}")
    print("Heuristic regime boundaries (LP16-style):")
    print(f"  High-ν / halo-dominated if ν ≳ {res['freq_high_MHz']:.1f} MHz  (2 λ^4 D_Φ(R0) ≲ 0.1)")
    print(f"  Low-ν  / saturated        if ν ≲ {res['freq_low_MHz']:.1f} MHz  (4 λ^4 σ_Φ^2 ≳ 10)")

    print("Linearization from ⟨R⟩-averaged angle structure:")
    print("  Fit of log(1 - 2 * D̄_φ(λ)) vs log λ (all valid points): "
         f"slope = {res['lin_slope_all']:.3f}")
    print("  Fit restricted to moderate rotation (S>0.1): "
         f"slope = {res['lin_slope_moderate']:.3f}")

    npz = save_npz(res, PRM)
    print(f"Saved results: {npz}")
    if PRM.make_plots:
        plot_results(res, PRM)
        print(f"Saved figures: {PRM.out_prefix}_*.pdf")

if __name__ == "__main__":
    main()
