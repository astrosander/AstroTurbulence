
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

# ======================
# USER SETTINGS
# ======================

# --- Paths & dataset keys ---
H5_PATH = r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"  # <-- edit to your local file
# Set these to YOUR dataset names:
BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"     # EDIT if needed
NE_KEY = "gas_density"                                                   # EDIT if needed; else None for constant n_e
CR_KEY: Optional[str] = None

# --- Geometry ---
# Internal vs separated
USE_EXTERNAL_SCREEN = True     # True -> separated-geometry calculations (Appendix C)
# If no separate screen datasets exist, use z-slice ranges from the same cube to build a screen & remote emitter:
SEPARATED_FROM_SAME_CUBE = True
SCREEN_Z_RANGE = (0, 80)       # indices [z0, z1) used for Faraday screen (foreground), e.g. near the observer
EMIT_Z_RANGE   = (120, 256)    # indices [z0, z1) used for emission (remote region)

# --- Units ---
# Convert your cube units to physical:
B_UNIT_TO_uG   = 1.0           # multiply B fields by this to get μG
NE_UNIT_TO_CM3 = 1.0           # multiply 'ne' by this to get cm^-3
DELTA_Z_PC     = 1.0           # voxel size along z in pc

# --- Synchrotron emissivity exponent ---
A_EXP = 2.0

# --- Wavelength grid (meters) ---
LAMBDA_LIST_M = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18,
                 0.21, 0.24, 0.27, 0.30, 0.33, 0.36,
                 0.39, 0.42, 0.45, 0.48, 0.51, 0.54,
                 0.57, 0.60, 0.63, 0.66, 0.69, 0.72,
                 0.75, 0.78, 0.81, 0.84, 0.87, 0.90,
                 0.93, 0.96, 0.99]

# --- Plots ---
MAKE_PLOTS = True
SAVE_PNG = True
OUTPUT_DIR = "lp2016_outputs"

# ======================

K_RM = 0.81  # rad m^-2 per (cm^-3 μG pc)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def orient_to_znynx(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    nz, ny, nx = arr.shape
    if (nz >= ny) and (nz >= nx):
        return arr
    if nx >= nz and nx >= ny:
        return np.transpose(arr, (2, 0, 1))
    if ny >= nz and ny >= nx:
        return np.transpose(arr, (1, 2, 0))
    return arr

def load_h5_fields(h5_path: str,
                   bx_key: str, by_key: str, bz_key: str,
                   ne_key: Optional[str], cr_key: Optional[str]) -> Dict[str, np.ndarray]:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        bx = orient_to_znynx(np.array(f[bx_key], dtype=np.float32))
        by = orient_to_znynx(np.array(f[by_key], dtype=np.float32))
        bz = orient_to_znynx(np.array(f[bz_key], dtype=np.float32))
        data = {"bx": bx, "by": by, "bz": bz}
        if ne_key is not None and ne_key in f:
            ne = orient_to_znynx(np.array(f[ne_key], dtype=np.float32))
            data["ne"] = ne
        if cr_key is not None and cr_key in f:
            data["cr"] = orient_to_znynx(np.array(f[cr_key], dtype=np.float32))
    return data

def intrinsic_polarization(bx: np.ndarray, by: np.ndarray, a_exp: float, cr: Optional[np.ndarray]) -> np.ndarray:
    bx2, by2 = bx*bx, by*by
    if a_exp == 2.0:
        Pi = (bx + 1j*by)**2
    else:
        amp = np.power(np.maximum(bx2 + by2, 1e-30), 0.5*a_exp)
        psi = np.arctan2(by, bx)
        Pi = amp * np.exp(2j*psi)
    if cr is not None:
        Pi = Pi * (cr / (np.mean(cr) + 1e-30))
    return Pi

def faraday_depth_increment(ne: np.ndarray, bz: np.ndarray, delta_z_pc: float) -> np.ndarray:
    return K_RM * ne * bz * delta_z_pc

def cumulative_phi(phi_inc: np.ndarray, observer_at_z0: bool = True) -> np.ndarray:
    if observer_at_z0:
        return np.cumsum(phi_inc, axis=0)
    else:
        return np.flip(np.cumsum(np.flip(phi_inc, axis=0), axis=0), axis=0)

def integrate_polarization(Pi: np.ndarray, phi_cum: np.ndarray, lam_m: float, los_mask: Optional[np.ndarray]) -> np.ndarray:
    phase = np.exp(2j * (lam_m**2) * phi_cum)
    if los_mask is not None:
        integrand = np.where(los_mask, Pi * phase, 0.0)
    else:
        integrand = Pi * phase
    return np.sum(integrand, axis=0)

def isotropic_power_slope(field2d: np.ndarray, kmin_frac: float=0.05, kmax_frac: float=0.4) -> Tuple[float, np.ndarray]:
    ny, nx = field2d.shape
    F = np.fft.rfft2(field2d)
    P2 = np.abs(F)**2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.rfftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX**2 + KY**2)
    kmax = 0.5 * min(nx, ny)
    nbins = int(np.sqrt(nx*ny) // 2)
    k_edges = np.linspace(0.0, kmax, nbins+1)
    idx = np.digitize(KR.ravel(), k_edges) - 1
    # Clip bin indices to valid range [0, nbins-1]
    idx = np.clip(idx, 0, nbins-1)
    counts = np.bincount(idx, minlength=nbins)
    sums = np.bincount(idx, weights=P2.ravel(), minlength=nbins)
    ps1d = np.zeros(nbins, dtype=float)
    valid = counts > 0
    ps1d[valid] = sums[valid] / counts[valid]
    k_centers = 0.5*(k_edges[:-1] + k_edges[1:])
    mask = (k_centers > kmin_frac*kmax) & (k_centers < kmax_frac*kmax) & (ps1d > 0)
    if np.sum(mask) < 5:
        return np.nan, k_centers
    x = np.log(k_centers[mask] + 1e-12)
    y = np.log(ps1d[mask] + 1e-30)
    slope, _ = np.polyfit(x, y, 1)
    return slope, k_centers

def compute_measures(P_maps: Dict[float, np.ndarray], lam_list: List[float]) -> Dict[str, Dict]:
    psa_slopes = {}
    kbins = None
    for lam in lam_list:
        slope, kb = isotropic_power_slope(np.abs(P_maps[lam]))
        psa_slopes[lam] = slope
        if kbins is None:
            kbins = kb
    lam_arr = np.array(lam_list, dtype=float)
    var_absP = np.array([np.var(np.abs(P_maps[lam])) for lam in lam_list], dtype=float)
    l2 = lam_arr**2
    order = np.argsort(l2)
    l2s = l2[order]
    lam_sorted = lam_arr[order]
    dPdl2_vars = []
    l2_mid = []
    for i in range(len(l2s)-1):
        dP = (P_maps[lam_sorted[i+1]] - P_maps[lam_sorted[i]]) / (l2s[i+1]-l2s[i])
        dPdl2_vars.append(np.var(np.abs(dP)))
        l2_mid.append(0.5*(l2s[i] + l2s[i+1]))
    return {
        "psa": {"slope_by_lambda": psa_slopes, "k_bins": kbins},
        "pva": {"lambda": lam_arr, "var_absP": var_absP},
        "dPdl2": {"lambda_mid": np.sqrt(np.array(l2_mid)), "var_abs_dPdl2": np.array(dPdl2_vars)}
    }

def classify_regime(phi_total_map: np.ndarray, lam_m: float) -> str:
    sigma_phi = np.std(phi_total_map)
    rot = 2.0 * (lam_m**2) * sigma_phi
    if rot < 1.0:
        return "weak"
    elif rot > 3.0:
        return "strong"
    else:
        return "transition"

def recommend_measure(regime: str) -> str:
    if regime == "weak":
        return "PSA for magnetic statistics; add dP/dλ² to highlight RM fluctuations."
    if regime == "strong":
        return "PVA (variance vs λ) dominates; PSA suppressed by Faraday rotation."
    return "Use both PSA and PVA; derivative measure helps isolate RM."

def main():
    ensure_dir(OUTPUT_DIR)
    data = load_h5_fields(H5_PATH, BX_KEY, BY_KEY, BZ_KEY, NE_KEY, CR_KEY)
    bx, by, bz = data["bx"]*B_UNIT_TO_uG, data["by"]*B_UNIT_TO_uG, data["bz"]*B_UNIT_TO_uG
    Nz, Ny, Nx = bx.shape
    print(f"Loaded fields: Bx,By,Bz with shape (Nz,Ny,Nx) = {bx.shape}")
    if "ne" in data:
        ne = data["ne"] * NE_UNIT_TO_CM3
        print("Using electron density dataset:", NE_KEY)
    else:
        ne = np.full_like(bx, 0.03*NE_UNIT_TO_CM3, dtype=np.float32)
        print("No electron density dataset found; using constant n_e = 0.03 cm^-3.")
    cr = data.get("cr", None)

    # Intrinsic polarization
    Pi = intrinsic_polarization(bx, by, a_exp=A_EXP, cr=cr)

    # Build φ'(z) increments
    phi_inc_full = faraday_depth_increment(ne, bz, DELTA_Z_PC)

    # Geometry selection
    if USE_EXTERNAL_SCREEN:
        if SEPARATED_FROM_SAME_CUBE:
            z0, z1 = SCREEN_Z_RANGE
            # Screen: only accumulate φ in [z0:z1)
            phi_inc = np.zeros_like(phi_inc_full)
            z0c = max(0, int(z0)); z1c = min(int(z1), phi_inc.shape[0])
            phi_inc[z0c:z1c] = phi_inc_full[z0c:z1c]
            # Emission only in EMIT_Z_RANGE
            e0, e1 = EMIT_Z_RANGE
            e0c = max(0, int(e0)); e1c = min(int(e1), Pi.shape[0])
            los_mask = np.zeros_like(Pi, dtype=bool)
            los_mask[e0c:e1c, :, :] = True
            print(f"Separated geometry (same cube): screen z∈[{z0c},{z1c}), emitter z∈[{e0c},{e1c})")
        else:
            # If you had separate screen datasets, put them here
            phi_inc = phi_inc_full
            los_mask = None
            print("USE_EXTERNAL_SCREEN=True but SEPARATED_FROM_SAME_CUBE=False; using full cube for φ and emission.")
    else:
        # Coincident: full internal rotation & emission everywhere
        phi_inc = phi_inc_full
        los_mask = None
        print("Coincident geometry (internal Faraday rotation).")

    # Cumulative φ(z)
    phi_cum = cumulative_phi(phi_inc, observer_at_z0=True)
    phi_total_map = phi_cum[-1]  # to far boundary

    # Build P maps across λ
    P_maps = {}
    for lam in LAMBDA_LIST_M:
        P_maps[lam] = integrate_polarization(Pi, phi_cum, lam, los_mask=los_mask)
        regime = classify_regime(phi_total_map, lam)
        print(f"λ = {lam:.3f} m -> regime: {regime} | {recommend_measure(regime)}")

    # Measures
    measures = compute_measures(P_maps, LAMBDA_LIST_M)

    # Print summaries
    print("\nPSA power-spectrum slopes of |P|:")
    for lam in LAMBDA_LIST_M:
        print(f"  λ = {lam:.3f} m : slope ≈ {measures['psa']['slope_by_lambda'][lam]:.3f}")

    print("\nPVA (variance of |P|) vs λ:")
    for lam, v in zip(measures["pva"]["lambda"], measures["pva"]["var_absP"]):
        print(f"  λ = {lam:.3f} m : Var(|P|) = {v:.6e}")

    print("\nDerivative measure (variance of |dP/d(λ^2)|) vs λ_mid:")
    for lam_m, v in zip(measures["dPdl2"]["lambda_mid"], measures["dPdl2"]["var_abs_dPdl2"]):
        print(f"  λ_mid ≈ {lam_m:.3f} m : Var(|dP/d(λ^2)|) = {v:.6e}")

    # Plots
    if MAKE_PLOTS:
        ensure_dir(OUTPUT_DIR)
        
        # Individual polarization maps
        lam_min, lam_max = LAMBDA_LIST_M[0], LAMBDA_LIST_M[-1]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"$|P|(\\lambda={lam_min:.3f}$ m)")
        plt.imshow(np.abs(P_maps[lam_min]))
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title(f"$|P|(\\lambda={lam_max:.3f}$ m)")
        plt.imshow(np.abs(P_maps[lam_max]))
        plt.colorbar()
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "P_maps_comparison.png"), dpi=160, bbox_inches="tight")
        
        # Combined measures plot - all on one figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # PSA slopes (normalized to 0-1 range)
        slopes = [measures["psa"]["slope_by_lambda"][l] for l in LAMBDA_LIST_M]
        slopes_norm = np.array(slopes)
        slopes_norm = (slopes_norm - np.min(slopes_norm)) / (np.max(slopes_norm) - np.min(slopes_norm) + 1e-10)
        ax.plot(LAMBDA_LIST_M, slopes_norm, 'bo-', linewidth=3, markersize=8, label='PSA slope (normalized)')
        
        # PVA variance (normalized to 0-1 range)
        lam_arr = measures["pva"]["lambda"]
        var_absP = measures["pva"]["var_absP"]
        var_absP_norm = (var_absP - np.min(var_absP)) / (np.max(var_absP) - np.min(var_absP) + 1e-10)
        ax.plot(lam_arr, var_absP_norm, 'ro-', linewidth=3, markersize=8, label='PVA variance (normalized)')
        
        # Derivative measure (normalized to 0-1 range)
        lam_mid = measures["dPdl2"]["lambda_mid"]
        var_dPdl2 = measures["dPdl2"]["var_abs_dPdl2"]
        var_dPdl2_norm = (var_dPdl2 - np.min(var_dPdl2)) / (np.max(var_dPdl2) - np.min(var_dPdl2) + 1e-10)
        ax.plot(lam_mid, var_dPdl2_norm, 'go-', linewidth=3, markersize=8, label='Derivative variance (normalized)')
        
        ax.set_title("LP16 Measures Comparison: PSA, PVA, and Derivative vs $\\lambda$", fontsize=14, fontweight='bold')
        ax.set_xlabel("$\\lambda$ (m)", fontsize=12)
        ax.set_ylabel("Normalized measure value", fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # Add regime background colors
        regime_colors = {'weak': 'lightblue', 'transition': 'yellow', 'strong': 'lightcoral'}
        
        # Group wavelengths by regime
        regime_wavelengths = {'weak': [], 'transition': [], 'strong': []}
        for lam in LAMBDA_LIST_M:
            regime = classify_regime(phi_total_map, lam)
            regime_wavelengths[regime].append(lam)
        
        # Fill background regions for each regime
        y_min, y_max = -0.1, 1.1
        for regime, wavelengths in regime_wavelengths.items():
            if wavelengths:
                # Find consecutive wavelength ranges for this regime
                wavelengths = sorted(wavelengths)
                start_idx = 0
                for i in range(1, len(wavelengths)):
                    if wavelengths[i] - wavelengths[i-1] > 0.05:  # Gap larger than typical spacing
                        # Fill the previous range
                        if i > start_idx:
                            ax.axvspan(wavelengths[start_idx], wavelengths[i-1], 
                                      color=regime_colors[regime], alpha=0.3, zorder=0)
                        start_idx = i
                # Fill the last range
                if start_idx < len(wavelengths):
                    ax.axvspan(wavelengths[start_idx], wavelengths[-1], 
                              color=regime_colors[regime], alpha=0.3, zorder=0)
        
        # Add regime legend
        from matplotlib.patches import Patch
        regime_patches = [Patch(facecolor=color, alpha=0.3, label=f'{regime.capitalize()} regime') 
                         for regime, color in regime_colors.items()]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + regime_patches, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "all_measures_combined.png"), dpi=160, bbox_inches="tight")
        
        # Faraday depth histogram
        plt.figure(figsize=(8, 6))
        plt.title("Histogram of total Faraday depth $\\Phi$ across map")
        plt.hist(phi_total_map.ravel(), bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel("$\\Phi$ (rad m$^{-2}$)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "phi_hist.png"), dpi=160, bbox_inches="tight")
        
        plt.show()

if __name__ == "__main__":
    main()
