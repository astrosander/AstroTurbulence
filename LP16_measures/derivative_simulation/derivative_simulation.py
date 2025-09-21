#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json

def load_density_and_field(path: str):
    with h5py.File(path, "r") as f:
        ne = f["gas_density"][:].astype(np.float64)
        bz = f["k_mag_field"][:].astype(np.float64)
    dx = dz = 1.0
    return ne, bz, dx, dz

def cumulative_faraday_depth(ne, bz, dz, K=1.0, axis=0):
    phi_per_cell = K * ne * bz * dz
    return np.cumsum(phi_per_cell, axis=axis)

def polarization_cube(ne, bz, dz, lam_grid, K=1.0, emissivity="density", zaxis=0):
    Φ = cumulative_faraday_depth(ne, bz, dz, K=K, axis=zaxis)
    if emissivity == "constant":
        p_i = np.ones_like(ne, dtype=np.float64)
    elif emissivity == "density":
        p_i = ne.astype(np.float64)
    else:
        raise ValueError("emissivity must be 'constant' or 'density'")
    Pλ = []
    for lam in lam_grid:
        phase = np.exp(2j * (lam**2) * Φ)
        Pmap = np.sum(p_i * phase, axis=zaxis)
        Pλ.append(Pmap)
    return np.array(Pλ)

def finite_diff_derivative(Pλ, lam2):
    nlam = len(lam2)
    dP = np.empty_like(Pλ, dtype=np.complex128)
    for i in range(nlam):
        if i == 0:
            dP[i] = (Pλ[i+1] - Pλ[i]) / (lam2[i+1] - lam2[i])
        elif i == nlam-1:
            dP[i] = (Pλ[i] - Pλ[i-1]) / (lam2[i] - lam2[i-1])
        else:
            dP[i] = (Pλ[i+1] - Pλ[i-1]) / (lam2[i+1] - lam2[i-1])
    return dP

def var_vs_lambda2_of_derivative(dP):
    dP_flat = dP.reshape(dP.shape[0], -1)
    mean = np.mean(dP_flat, axis=1)
    mean2 = np.mean(np.abs(dP_flat)**2, axis=1)
    var = mean2 - np.abs(mean)**2
    return var

def fit_loglog(x, y):
    m = (x>0) & (y>0)
    xlog = np.log10(x[m]); ylog = np.log10(y[m]+1e-30)
    A = np.vstack([xlog, np.ones_like(xlog)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, ylog, rcond=None)
    slope, intercept = coeff[0], coeff[1]
    ypred = A @ coeff
    ssr = np.sum((ylog - ypred)**2)
    sst = np.sum((ylog - ylog.mean())**2) + 1e-30
    r2 = 1 - ssr/sst
    return {"slope": slope, "intercept": intercept, "r2": r2}

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
    lam_min = 0.05
    lam_max = 1.0
    nlam = 64
    kfaraday = 1.0
    emissivity = "density"
    zaxis = 0
    out_prefix = "DER_out"

    ne, bz, dx, dz = load_density_and_field(h5_path)
    lam_grid = np.linspace(lam_min, lam_max, nlam)
    lam2 = lam_grid**2
    Pλ = polarization_cube(ne, bz, dz, lam_grid, K=kfaraday, emissivity=emissivity, zaxis=zaxis)
    dP = finite_diff_derivative(Pλ, lam2)
    var_dP = var_vs_lambda2_of_derivative(dP)
    fit = fit_loglog(lam2, var_dP)

    np.savetxt(f"{out_prefix}_var_dP.csv", np.c_[lam2, var_dP], delimiter=",", header="lambda2,var(|dP/dlambda2|)", comments="")
    with open(f"{out_prefix}_fit.json", "w") as f:
        json.dump(fit, f, indent=2)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.loglog(lam2[lam2>0], var_dP[lam2>0], marker=".", linestyle="none")
    if np.isfinite(fit["slope"]):
        xx = np.linspace(lam2[lam2>0].min(), lam2.max(), 200)
        yy = 10**(fit["intercept"]) * xx**(fit["slope"])
        plt.loglog(xx, yy, label=f"slope={fit['slope']:.2f}, R^2={fit['r2']:.3f}")
        plt.legend()
    plt.xlabel(r"$\lambda^2$ (m$^2$)")
    plt.ylabel(r"Var$[\partial P/\partial(\lambda^2)]$")
    plt.title("Derivative-based polarization statistic")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_var_dP.png", dpi=150)
    print(f"[DERIV] Saved: {out_prefix}_var_dP.csv, {out_prefix}_fit.json, {out_prefix}_var_dP.png")

if __name__ == "__main__":
    main()
