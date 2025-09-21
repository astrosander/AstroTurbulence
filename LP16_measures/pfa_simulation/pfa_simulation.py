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

def variance_vs_lambda2(Pλ):
    Pλ_flat = Pλ.reshape(Pλ.shape[0], -1)
    meanP = np.mean(Pλ_flat, axis=1)
    meanP2 = np.mean(np.abs(Pλ_flat)**2, axis=1)
    varP = meanP2 - np.abs(meanP)**2
    return varP

def fit_piecewise_loglog(x, y):
    xm = np.median(x)
    left = x <= xm
    right = x > xm
    def fit(x_, y_):
        m = (x_>0) & (y_>0)
        xlog = np.log10(x_[m]); ylog = np.log10(y_[m]+1e-30)
        A = np.vstack([xlog, np.ones_like(xlog)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, ylog, rcond=None)
        slope, intercept = coeff[0], coeff[1]
        ypred = A @ coeff
        ssr = np.sum((ylog - ypred)**2)
        return slope, intercept, ssr, m.sum()
    s1, b1, ssr1, n1 = fit(x[left], y[left])
    s2, b2, ssr2, n2 = fit(x[right], y[right])
    s_all, b_all, ssr_all, n_all = fit(x, y)
    k1, k2 = 2, 4
    aic1 = n_all*np.log(ssr_all/n_all + 1e-30) + 2*k1
    aic2 = (n1*np.log(ssr1/max(n1,1) + 1e-30) + n2*np.log(ssr2/max(n2,1) + 1e-30) + 2*k2)
    return {"slope_all": s_all, "intercept_all": b_all, "AIC1": aic1,
            "left_slope": s1, "right_slope": s2, "AIC2": aic2, "split_x": xm}

def main():
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"
    lam_min = 0.05
    lam_max = 1.0
    nlam = 64
    kfaraday = 1.0
    emissivity = "density"
    zaxis = 0
    out_prefix = "PFA_out"

    ne, bz, dx, dz = load_density_and_field(h5_path)
    lam_grid = np.linspace(lam_min, lam_max, nlam)
    Pλ = polarization_cube(ne, bz, dz, lam_grid, K=kfaraday, emissivity=emissivity, zaxis=zaxis)
    lam2 = lam_grid**2
    varP = variance_vs_lambda2(Pλ)

    np.savetxt(f"{out_prefix}_var.csv", np.c_[lam2, varP], delimiter=",", header="lambda2,varP", comments="")
    fits = fit_piecewise_loglog(lam2, varP)
    with open(f"{out_prefix}_fit.json", "w") as f:
        json.dump(fits, f, indent=2)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.loglog(lam2[lam2>0], varP[lam2>0], marker=".", linestyle="none")
    if np.isfinite(fits["slope_all"]):
        xx = np.linspace(lam2[lam2>0].min(), lam2.max(), 200)
        yy = 10**(fits["intercept_all"]) * xx**(fits["slope_all"])
        plt.loglog(xx, yy, label=f"all slope={fits['slope_all']:.2f}")
    plt.xlabel(r"$\lambda^2$ (m$^2$)")
    plt.ylabel(r"Var$[P(\lambda)]$")
    plt.title("PFA/PVA: Variance vs $\\lambda^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_var.png", dpi=150)
    print(f"[PFA/PVA] Saved: {out_prefix}_var.csv, {out_prefix}_fit.json, {out_prefix}_var.png")

if __name__ == "__main__":
    main()
