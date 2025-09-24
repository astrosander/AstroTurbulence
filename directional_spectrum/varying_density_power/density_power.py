import numpy as np, h5py, matplotlib.pyplot as plt, json, csv
from typing import Tuple, Dict, Optional

# --------------------------
# I/O and small utilities
# --------------------------

def load_density_and_field(path: str) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Load ne and Bz (LoS component). Returns (ne, bz, dx, dy, dz).
    If spacings are not stored, defaults to 1.0.
    """
    with h5py.File(path, "r") as f:
        ne = np.array(f["gas_density"], dtype=np.float64)
        bz = np.array(f["k_mag_field"], dtype=np.float64)
        dx = float(f.attrs.get("dx", 1.0))
        dy = float(f.attrs.get("dy", 1.0))
        dz = float(f.attrs.get("dz", 1.0))

        dx=1.0
        dy=1.0
        dz=1.0
    return ne, bz, dx, dy, dz

def ensure3d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3: return a
    if a.ndim == 2: return a[:, None, :]
    if a.ndim == 1: return a[:, None, None]
    raise ValueError("Input must be 1D/2D/3D.")

def fftfreq_grid(nz: int, ny: int, nx: int, dz: float, dy: float, dx: float):
    kz = np.fft.fftfreq(nz, d=dz)
    ky = np.fft.fftfreq(ny, d=dy)
    kx = np.fft.fftfreq(nx, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
    return KX, KY, KZ

# --------------------------
# Isotropized spectra (3D, 2D)
# --------------------------

def shell_average_3d(F: np.ndarray, KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray,
                     n_bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a 3D complex Fourier cube F and k-grid, return isotropized shell centers, power per shell, counts.
    Power uses |F|^2 normalized by total number of voxels for convenience (relative shapes matter).
    """
    nz, ny, nx = F.shape
    KR = np.sqrt(KX**2 + KY**2 + KZ**2)
    power = np.abs(F)**2 / (nz * ny * nx)
    kr = KR.ravel(); p = power.ravel()
    m = kr > 0  # drop zero-mode from fitting/shaping
    kr = kr[m]; p = p[m]
    idx = np.argsort(kr)
    kr = kr[idx]; p = p[idx]
    if n_bins is None:
        n_bins = max(24, int(round((nz*ny*nx)**(1/3))))
    edges = np.linspace(kr.min(), kr.max(), n_bins + 1)
    centers = 0.5*(edges[1:] + edges[:-1])
    cnt = np.zeros_like(centers)
    Pk = np.zeros_like(centers)
    ind = np.digitize(kr, edges) - 1
    v = (ind >= 0) & (ind < n_bins)
    np.add.at(Pk, ind[v], p[v])
    np.add.at(cnt, ind[v], 1)
    cnt[cnt == 0] = 1
    Pk /= cnt
    return centers, Pk, cnt

def ring_average_2d(arr: np.ndarray, dx: float, dy: float,
                    n_bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D isotropized power spectrum of real map `arr`.
    """
    ny, nx = arr.shape
    F = np.fft.fftshift(np.fft.fft2(arr))
    P2 = (np.abs(F)**2) / (nx * ny)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX**2 + KY**2).ravel()
    p = P2.ravel()
    idx = np.argsort(KR); KR = KR[idx]; p = p[idx]
    if n_bins is None:
        n_bins = max(16, int(np.sqrt(nx*ny)))
    edges = np.linspace(KR.min(), KR.max(), n_bins + 1)
    centers = 0.5*(edges[1:] + edges[:-1])
    cnt = np.zeros_like(centers); Pk = np.zeros_like(centers)
    ind = np.digitize(KR, edges) - 1
    v = (ind>=0) & (ind<n_bins)
    np.add.at(Pk, ind[v], p[v]); np.add.at(cnt, ind[v], 1)
    cnt[cnt==0] = 1; Pk /= cnt
    return centers, Pk

# --------------------------
# Fit slope on log–log
# --------------------------

def fit_loglog(k: np.ndarray, pk: np.ndarray, kmin_frac=0.05, kmax_frac=0.35) -> Dict[str, float]:
    m = (k > 0) & (pk > 0)
    k = k[m]; pk = pk[m]
    if k.size < 12:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "xmin": np.nan, "xmax": np.nan}
    xmin = kmin_frac * k.max(); xmax = kmax_frac * k.max()
    w = (k >= xmin) & (k <= xmax)
    if w.sum() < 12:
        w = (k > 0)
    X = np.log10(k[w]); Y = np.log10(pk[w])
    A = np.vstack([X, np.ones_like(X)]).T
    c, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    Yp = A @ c
    ssr = float(np.sum((Y - Yp)**2)); sst = float(np.sum((Y - Y.mean())**2) + 1e-30)
    return {"slope": float(c[0]), "intercept": float(c[1]), "r2": 1 - ssr/sst,
            "xmin": float(xmin), "xmax": float(xmax)}

# --------------------------
# Spectral shaping (exact shell-by-shell)
# --------------------------

def _target_power_powerlaw(kc: np.ndarray, alpha3d: float) -> np.ndarray:
    # 3D spectral density P3(k) ~ k^{-alpha3d}; shape only (norm set later)
    eps = 1e-30
    return (kc + eps)**(-alpha3d)

def _target_power_broken(kc: np.ndarray, alpha_low: float, alpha_high: float,
                         kb: float, sharpness: float = 8.0) -> np.ndarray:
    # Smooth transition like in your draft (alpha(k) blend) – we directly shape P3(k) ~ k^{-alpha(k)}.
    t = 1.0 / (1.0 + (kc / (kb + 1e-30))**sharpness)
    alpha_k = alpha_low * t + alpha_high * (1 - t)
    eps = 1e-30
    return (kc + eps)**(-alpha_k)

def impose_power_spectrum_3d(field: np.ndarray, dx: float, dy: float, dz: float,
                              target: Dict, keep_mean=True, keep_rms=True,
                              n_bins: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Enforce a target isotropic 3D spectrum on 'field' by reweighting Fourier shells.

    target examples:
      {"kind":"powerlaw", "alpha3d": 11/3}            # P3 ~ k^-alpha3d
      {"kind":"broken", "alpha_low": 1.5, "alpha_high": 11/3, "kb": 0.1, "sharpness": 8}

    Returns (new_field, meta) where meta has measured slope and info.
    """
    f = np.array(field, dtype=np.float64, copy=True)
    mu = f.mean() if keep_mean else 0.0
    f = f - f.mean()

    nz, ny, nx = f.shape
    F = np.fft.fftn(f)
    KX, KY, KZ = fftfreq_grid(nz, ny, nx, dz, dy, dx)

    # Build shell stats
    kc, Pk_cur, cnt = shell_average_3d(F, KX, KY, KZ, n_bins=n_bins)

    # Target (shape only)
    if target["kind"] == "powerlaw":
        Pk_tar_shape = _target_power_powerlaw(kc, float(target["alpha3d"]))
    elif target["kind"] == "broken":
        Pk_tar_shape = _target_power_broken(
            kc, float(target["alpha_low"]), float(target["alpha_high"]),
            float(target["kb"]), float(target.get("sharpness", 8.0))
        )
    else:
        raise ValueError("target['kind'] must be 'powerlaw' or 'broken'.")

    # Choose normalization so total power matches (excluding zero bin)
    # Total power ~ sum over shells (count * shell_power).
    eps = 1e-30
    S_cur = np.sum(cnt * Pk_cur)
    S_tar_unnorm = np.sum(cnt * Pk_tar_shape)
    norm = S_cur / (S_tar_unnorm + eps)
    Pk_tar = norm * Pk_tar_shape

    # Compute per-shell weights: sqrt(P_target / P_current)
    w_shell = np.sqrt(Pk_tar / (Pk_cur + eps))

    # Apply weights to F shell-by-shell
    KR = np.sqrt(KX**2 + KY**2 + KZ**2)
    # Digitize KR against edges from kc
    edges = np.concatenate([[kc[0] - (kc[1]-kc[0])], 0.5*(kc[1:] + kc[:-1]), [kc[-1] + (kc[-1]-kc[-2])]])
    ind = np.digitize(KR, edges) - 1
    mask_valid = (ind >= 0) & (ind < kc.size)
    W = np.ones_like(F, dtype=np.float64)
    W[mask_valid] = w_shell[ind[mask_valid]]
    # Keep DC exactly (don’t explode it)
    W[0, 0, 0] = 0.0
    F_new = F * W

    # Back to real space
    g = np.fft.ifftn(F_new).real
    if keep_rms:
        s0 = field.std()
        sg = g.std() if g.std() != 0 else 1.0
        g = g * (s0 / sg)
    if keep_mean:
        g = g + mu

    # Measure resulting 3D slope on (g - mean)
    G = g - g.mean()
    FG = np.fft.fftn(G)
    kc_m, Pk_m, _ = shell_average_3d(FG, KX, KY, KZ, n_bins=n_bins)
    fit = fit_loglog(kc_m, Pk_m)

    return g, {"k": kc_m.tolist(), "Pk": Pk_m.tolist(), "fit": fit}

# --------------------------
# Polarization & directional spectrum
# --------------------------

def faraday_depth(ne: np.ndarray, bz: np.ndarray, dz: float, K: float = 0.812) -> np.ndarray:
    """
    Phi(X) = \int ne * Bz dz  (units assume ne[cm^-3], Bz[μG], dz[pc], K≈0.812 rad m^-2).
    For arbitrary code units, K is an overall scale factor.
    """
    return K * np.sum(ne * bz * dz, axis=0)

def directional_spectrum(
    ne: np.ndarray, bz: np.ndarray, dx: float, dy: float, dz: float,
    lam: float, geometry: str = "screen", emissivity: Optional[np.ndarray] = None,
    K: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute P_dir(k) from cos2χ and sin2χ using your S(R) definition.

    geometry='screen': background uniform emitter -> χ(X)=λ^2 Φ(X)
    geometry='mixed' : internal emission (Burn slab). If emissivity is None, use ε=1 per cell.
    """
    assert geometry in ("screen", "mixed")
    if geometry == "screen":
        Phi = faraday_depth(ne, bz, dz, K=K)                     # shape (ny, nx)
        chi = (lam**2) * Phi
        Q = np.cos(2.0 * chi); U = np.sin(2.0 * chi)
        A = np.sqrt(Q**2 + U**2) + 1e-30
        c2 = Q / A; s2 = U / A
    else:
        nz, ny, nx = ne.shape
        eps = emissivity if emissivity is not None else np.ones_like(ne)
        # cumulative Phi_to_obs from far side (z = nz-1) to observer (z=0)
        phi_cells = K * ne * bz * dz
        phi_to_obs = np.cumsum(phi_cells[::-1, :, :], axis=0)[::-1, :, :]
        P = np.sum(eps * np.exp(2j * (lam**2) * phi_to_obs), axis=0)
        Q = P.real; U = P.imag
        A = np.sqrt(Q**2 + U**2) + 1e-30
        c2 = Q / A; s2 = U / A

    # Directional spectrum = ring power of cos2χ and sin2χ summed
    k1, p1 = ring_average_2d(c2, dx, dy)
    k2, p2 = ring_average_2d(s2, dx, dy)
    if k1.shape == k2.shape and np.allclose(k1, k2):
        k = k1; Pk = p1 + p2
    else:
        # interpolate to common k
        if k1.size <= k2.size:
            k = k1; Pk = p1 + np.interp(k, k2, p2)
        else:
            k = k2; Pk = np.interp(k, k1, p1) + p2
    return k, Pk

# --------------------------
# Weak-rotation sanity check (LP16)
# --------------------------

def estimate_rotation_dispersion(ne: np.ndarray, bz: np.ndarray, dz: float, K: float = 1.0) -> float:
    """Rough σ_Φ from slab (variance across the map of total Φ)."""
    Phi = faraday_depth(ne, bz, dz, K=K)
    return float(np.std(Phi))

def main():
    # ---- Paths & params
    h5_path = r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\mhd_fields.h5"
    lam = 0.5           # wavelength units consistent with K (if K is arbitrary, lam is in the same arbitrary system)
    geometry = "screen" # or "mixed"
    K = 1.0             # overall Faraday scaling in your code units

    # ---- Load & basic stats
    ne, bz, dx, dy, dz = load_density_and_field(h5_path)
    ne = ensure3d(ne); bz = ensure3d(bz)  # (nz, ny, nx)
    mu_n = float(np.mean(ne)); rms_n = float(np.std(ne))
    with open("n_stats.json", "w") as f:
        json.dump({"mean": mu_n, "rms": rms_n}, f, indent=2)

    # ---- Measure original 3D n-spectrum
    nz, ny, nx = ne.shape
    F_ne = np.fft.fftn(ne - ne.mean())
    KX, KY, KZ = fftfreq_grid(nz, ny, nx, dz, dy, dx)
    k0, Pn0, _ = shell_average_3d(F_ne, KX, KY, KZ)
    fit0 = fit_loglog(k0, Pn0)
    with open("n_slope_original.json", "w") as f:
        json.dump({"slope_logPk_vs_logk": float(fit0["slope"]),
                   "alpha3d": float(-fit0["slope"]),
                   "r2": float(fit0["r2"])}, f, indent=2)

    # ---- Plot original n-spectrum
    plt.figure()
    m0 = (k0 > 0) & (Pn0 > 0)
    plt.loglog(k0[m0], Pn0[m0], label="original n")
    if np.isfinite(fit0["slope"]):
        xx = np.geomspace(k0[m0].min(), k0[m0].max(), 256)
        yy = 10**fit0["intercept"] * xx**fit0["slope"]
        plt.loglog(xx, yy, "--", lw=1)
    plt.xlabel("$k$"); plt.ylabel("$P3_n(k)$")
    plt.title("3D n-spectrum")
    plt.legend(); plt.tight_layout()
    plt.savefig("n_power_original.png", dpi=200)
    plt.savefig("n_power_original.pdf", dpi=200)

    # ---- Targets: single-slope and one broken example
    targets = [
        {"label": "alpha=2.7",  "spec": {"kind": "powerlaw", "alpha3d": 2.7}},
        {"label": "alpha=3.0",  "spec": {"kind": "powerlaw", "alpha3d": 3.0}},
        {"label": "alpha=11/3", "spec": {"kind": "powerlaw", "alpha3d": 11/3}},
        {"label": "alpha=3.9",  "spec": {"kind": "powerlaw", "alpha3d": 3.9}},
        # {"label": "broken",     "spec": {"kind": "broken", "alpha_low": 1.5, "alpha_high": 11/3, "kb": 0.1, "sharpness": 8}},
    ]

    # ---- Shape density to each target & collect spectra
    out_curves = {}
    plt.figure()
    for t in targets:
        g, meta = impose_power_spectrum_3d(ne, dx, dy, dz, target=t["spec"], keep_mean=True, keep_rms=True)
        k, Pk = np.array(meta["k"]), np.array(meta["Pk"])
        fit = meta["fit"]
        out_curves[t["label"]] = {
            "target": t["spec"],
            "measured_slope": float(fit["slope"]),
            "measured_alpha3d": float(-fit["slope"]),
            "r2": float(fit["r2"]),
        }
        m = (k > 0) & (Pk > 0)
        plt.loglog(k[m], Pk[m], label=f"{t['label']}; $\\alpha={-fit['slope']:.2f}$")
        if np.isfinite(fit["slope"]):
            xx = np.geomspace(k[m].min(), k[m].max(), 200)
            yy = 10**fit["intercept"] * xx**fit["slope"]
            plt.loglog(xx, yy, "--", lw=1)
        # Save CSV
        with open(f"n_power_{t['label'].replace('/','_')}.csv","w",newline="") as fcsv:
            w = csv.writer(fcsv); w.writerow(["k","P3_n(k)"]); w.writerows(zip(k.tolist(), Pk.tolist()))
        # Replace ne for directional test
        ne_target = g

    plt.xlabel("$k$"); plt.ylabel("$P3_n(k)$")
    plt.title("3D n-spectra after spectral shaping")
    plt.legend(); plt.tight_layout()
    plt.savefig("n_power_targets.png", dpi=200)
    plt.savefig("n_power_targets.pdf", dpi=200)

    # ---- Directional spectrum vs α_n
    plt.figure()
    k_dir0, Pdir0 = directional_spectrum(ne, bz, dx, dy, dz, lam, geometry=geometry, K=K)
    m = (k_dir0 > 0) & (Pdir0 > 0)
    plt.loglog(k_dir0[m], Pdir0[m], label="original n")
    for t in targets:
        g, _ = impose_power_spectrum_3d(ne, dx, dy, dz, target=t["spec"], keep_mean=True, keep_rms=True)
        k_dir, Pdir = directional_spectrum(g, bz, dx, dy, dz, lam, geometry=geometry, K=K)
        md = (k_dir > 0) & (Pdir > 0)
        plt.loglog(k_dir[md], Pdir[md], label=t["label"])
    plt.xlabel("k"); plt.ylabel(r"$P_{\rm dir}(k)$")
    plt.title(f"Directional spectra from n-slope")
    plt.legend(); plt.tight_layout()
    plt.savefig("Pdir_vs_n_slope.png", dpi=200)
    plt.savefig("Pdir_vs_n_slope.pdf", dpi=200)

    # ---- Weak-rotation check (optional)
    sigma_Phi = estimate_rotation_dispersion(ne, bz, dz, K=K)
    with open("rotation_dispersion.json","w") as f:
        json.dump({"sigma_Phi": sigma_Phi, "lambda": lam,
                   "lambda4_sigmaPhi2": (lam**4) * (sigma_Phi**2)}, f, indent=2)

    # ---- Save summary
    with open("n_power_targets_summary.json", "w") as f:
        json.dump({"original_alpha3d": float(-fit0["slope"]), "targets": out_curves}, f, indent=2)

if __name__ == "__main__":
    main()
