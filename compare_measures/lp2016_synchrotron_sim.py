
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

"""
Lazarian & Pogosyan (2016)-inspired Synchrotron Polarization + Faraday Rotation Simulation

This script:
1) Loads 3D MHD fields (Bx, By, Bz; optional n_e, rho, velocities) from an HDF5 file.
2) Builds intrinsic complex synchrotron polarization P_i(x, y, z).
3) Integrates along the line-of-sight z to form observed P(x, y; λ) including Faraday rotation.
4) Computes three measures:
   - PSA: spatial correlations at fixed λ (here via 2D power-spectrum slope of |P|)
   - PVA: variance of |P| vs λ
   - Derivative measure: spatial variance/correlation of dP/d(λ^2)
5) Classifies which measure is best at each λ based on Faraday strength (weak/transition/strong).

USAGE
-----
- No argparse: edit the "USER SETTINGS" below and run.

USER SETTINGS
-------------
"""
# === USER SETTINGS ===
H5_PATH = r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5"  # <-- edit to your local file
# What datasets to read
BX_KEY, BY_KEY, BZ_KEY = "i_mag_field", "j_mag_field", "k_mag_field"   # magnetic field components
NE_KEY = "gas_density"                            # optional electron density (cm^-3) dataset in the H5
# If NE_KEY is absent, use constant n_e:
NE_FALLBACK_CM3 = 0.03

# Voxel size along z (parsec). If your grid has different units, convert accordingly.
DELTA_Z_PC = 1.0

# Synchrotron emissivity exponent ("a") for polarization amplitude.
# a=2 -> P_i ∝ (Bx + i By)^2 (common convenient case); for general a, P_i ∝ |B_perp|^a * exp(2i ψ).
A_EXP = 2.0

# Optional: weight intrinsic emissivity by cosmic-ray density if present (dataset name), else constant.
CR_KEY: Optional[str] = None  # e.g., "cr" or None

# Treat z-axis as line-of-sight (observer at z=0).
OBSERVER_AT_Z0 = True

# External Faraday screen option: if True, Faraday rotation uses a *separate* set of Bz/ne cubes (foreground screen).
USE_EXTERNAL_SCREEN = True#False
EXT_BZ_KEY, EXT_NE_KEY = "bz_screen", "ne_screen"  # keys for external screen if present

# Wavelengths to analyze (meters)
LAMBDA_LIST_M = np.arange(0.03, 1.0, 0.03)#[0.03, 0.06, 0.1, 0.2, 0.5, 1.0]

# Plot diagnostics?
MAKE_PLOTS = True
SAVE_PNG = True
OUTPUT_DIR = "lp2016_outputs"
# ======================


# --- Constants ---
K_RM = 0.81  # rad m^-2 per (cm^-3 μG pc)

def load_h5_fields(h5_path: str,
                   bx_key: str,
                   by_key: str,
                   bz_key: str,
                   ne_key: Optional[str] = None,
                   cr_key: Optional[str] = None,
                   screen_bz_key: Optional[str] = None,
                   screen_ne_key: Optional[str] = None
                   ) -> Dict[str, np.ndarray]:
    """
    Load arrays from an HDF5 file.
    Expected shapes: (Nz, Ny, Nx) or (Ny, Nx, Nz) (we will auto-orient to (Nz, Ny, Nx)).
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        print(f.keys())
        bx = np.array(f[bx_key], dtype=np.float32)
        by = np.array(f[by_key], dtype=np.float32)
        bz = np.array(f[bz_key], dtype=np.float32)

        # Auto-orient to (Nz, Ny, Nx)
        bx, by, bz = orient_to_znynx(bx), orient_to_znynx(by), orient_to_znynx(bz)

        data = {"bx": bx, "by": by, "bz": bz}

        if ne_key is not None and ne_key in f:
            ne = np.array(f[ne_key], dtype=np.float32)
            ne = orient_to_znynx(ne)
            data["ne"] = ne

        if cr_key is not None and cr_key in f:
            cr = np.array(f[cr_key], dtype=np.float32)
            data["cr"] = orient_to_znynx(cr)

        if screen_bz_key is not None and screen_bz_key in f:
            bz_scr = np.array(f[screen_bz_key], dtype=np.float32)
            data["bz_screen"] = orient_to_znynx(bz_scr)
        if screen_ne_key is not None and screen_ne_key in f:
            ne_scr = np.array(f[screen_ne_key], dtype=np.float32)
            data["ne_screen"] = orient_to_znynx(ne_scr)

    return data


def orient_to_znynx(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is shaped (Nz, Ny, Nx). If not, try to permute axes intelligently.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    # Heuristic: choose the axis order that sets the largest dimension as Nz if observer at z=0 is assumed,
    # but generally we can't know. Default to as-is. Users can permute here if needed.
    # We'll prefer the input as (Nz, Ny, Nx) if it "looks" like it (first dim differs).
    # Else, try common alternatives.
    nz, ny, nx = arr.shape
    if (nz >= ny) and (nz >= nx):
        return arr  # likely already Nz, Ny, Nx
    # Try (Ny, Nx, Nz) -> (Nz, Ny, Nx)
    if nx >= nz and nx >= ny:
        return np.transpose(arr, (2, 0, 1))
    # Try (Nx, Ny, Nz) -> (Nz, Ny, Nx)
    if ny >= nz and ny >= nx:
        return np.transpose(arr, (1, 2, 0))
    # Fallback: leave as-is
    return arr


def intrinsic_polarization(bx: np.ndarray, by: np.ndarray, a_exp: float,
                           cr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute intrinsic complex polarization P_i(x, y, z).
    For a_exp == 2: use algebraic form P_i ∝ (Bx + i By)^2 (LP12 convenient case).
    Otherwise: P_i ∝ |B_perp|^a * exp(2i ψ), with ψ = atan2(By, Bx).
    Optionally weighted by cosmic ray density 'cr' if provided.
    """
    bperp2 = bx**2 + by**2
    if a_exp == 2.0:
        Pi = (bx + 1j*by)**2
    else:
        amp = np.power(np.maximum(bperp2, 1e-30), 0.5 * a_exp)
        psi = np.arctan2(by, bx)
        Pi = amp * np.exp(2j * psi)

    if cr is not None:
        # Weight by CR density (normalized to unit mean to keep scale reasonable)
        cr_norm = cr / (np.mean(cr) + 1e-30)
        Pi = Pi * cr_norm

    return Pi


def faraday_depth_density(ne: np.ndarray,
                          bz: np.ndarray,
                          delta_z_pc: float) -> np.ndarray:
    """
    Faraday depth *density* per layer: φ'(z) = 0.81 n_e (cm^-3) * B_parallel (μG) [rad m^-2 pc^-1] * Δz_pc
    Returns φ_increment(z) = φ'(z) * Δz_pc, so cumulative φ(z) is the cumsum of this along z.
    """
    return K_RM * ne * bz * delta_z_pc


def cumulative_phi(phi_inc: np.ndarray, observer_at_z0: bool = True) -> np.ndarray:
    """
    Compute cumulative φ(z) needed for P(λ^2) = ∑ P_i(z) exp(2i λ^2 φ(z)),
    where φ(z) is the integral from observer to emission point.
    If observer_at_z0=True, we integrate from z'=0 up to current z.
    If observer_at_z0=False, integrate from far end down to current z.
    """
    if observer_at_z0:
        # φ(z_k) = sum_{j=0}^{k} φ_inc(j)
        return np.cumsum(phi_inc, axis=0)
    else:
        # observer at far end (z=Nz-1): φ(z_k) = sum_{j=k}^{Nz-1} φ_inc(j)
        return np.flip(np.cumsum(np.flip(phi_inc, axis=0), axis=0), axis=0)


def integrate_polarization(Pi: np.ndarray,
                           phi_cum: np.ndarray,
                           lam_m: float,
                           los_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Integrate P(x, y; λ) = ∑_z P_i(x, y, z) * exp(2i λ^2 φ(z)).
    Optionally use a boolean los_mask of shape (Nz, Ny, Nx) to restrict emission to a subset (e.g., external screen geometry).
    Returns 2D complex array P(x, y).
    """
    phase = np.exp(2j * (lam_m**2) * phi_cum)  # exp(2i λ^2 φ(z))
    if los_mask is not None:
        Pi_eff = np.where(los_mask, Pi, 0.0)
        phase_eff = np.where(los_mask, phase, 1.0)
        integrand = Pi_eff * phase_eff
    else:
        integrand = Pi * phase
    # Sum along z (axis=0), result shape (Ny, Nx)
    return np.sum(integrand, axis=0)


def compute_measures(P_maps: Dict[float, np.ndarray],
                      lam_list: List[float]) -> Dict[str, Dict]:
    """
    Compute PSA slope, PVA variance vs λ, and derivative measure for dP/d(λ^2).

    Returns a dict with:
      - 'psa': {'slope_by_lambda': {λ: slope}, 'k_bins': k_edges, 'note': str}
      - 'pva': {'lambda': array, 'var_absP': array}
      - 'dPdl2': {'lambda_mid': array, 'var_abs_dPdl2': array}
    """
    # PSA: power-spectrum slope of |P| for each λ
    psa_slopes = {}
    example_kbins = None
    for lam in lam_list:
        P = P_maps[lam]
        slope, kbins = isotropic_power_slope(np.abs(P))
        psa_slopes[lam] = slope
        if example_kbins is None:
            example_kbins = kbins

    # PVA: variance of |P| vs λ
    lam_arr = np.array(lam_list, dtype=float)
    var_absP = np.array([np.var(np.abs(P_maps[lam])) for lam in lam_list], dtype=float)

    # Derivative measure: finite difference in λ^2
    l2 = lam_arr**2
    # Sort by increasing λ^2 just in case user didn't
    order = np.argsort(l2)
    l2s = l2[order]
    lam_sorted = lam_arr[order]
    dPdl2_vars = []
    l2_mid = []
    for i in range(len(l2s)-1):
        lam_a, lam_b = lam_sorted[i], lam_sorted[i+1]
        dP = (P_maps[lam_b] - P_maps[lam_a]) / (l2s[i+1] - l2s[i])
        dPdl2_vars.append(np.var(np.abs(dP)))
        l2_mid.append(0.5*(l2s[i] + l2s[i+1]))
    dPdl2_vars = np.array(dPdl2_vars, dtype=float)
    l2_mid = np.array(l2_mid, dtype=float)

    return {
        "psa": {"slope_by_lambda": psa_slopes, "k_bins": example_kbins, "note": "PSA slope from 2D power of |P|"},
        "pva": {"lambda": lam_arr, "var_absP": var_absP},
        "dPdl2": {"lambda_mid": np.sqrt(l2_mid), "var_abs_dPdl2": dPdl2_vars},
    }


def isotropic_power_slope(field2d: np.ndarray,
                          kmin_frac: float = 0.05,
                          kmax_frac: float = 0.4) -> Tuple[float, np.ndarray]:
    """
    Estimate 2D isotropic power-spectrum slope for a scalar field (|P| here).
    Returns (slope, k_bin_centers). We FFT, radially average, and fit log power vs log k in [kmin_frac, kmax_frac] of Nyquist.
    """
    ny, nx = field2d.shape
    # FFT
    F = np.fft.rfft2(field2d)
    power2d = np.abs(F)**2
    # Build k arrays
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.rfftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX**2 + KY**2)
    kmax = 0.5 * min(nx, ny)
    # Radial bins
    nbins = int(np.sqrt(nx*ny) // 2)
    k_edges = np.linspace(0.0, kmax, nbins+1)
    bin_idx = np.digitize(KR.ravel(), k_edges) - 1
    # Clip bin indices to valid range [0, nbins-1]
    bin_idx = np.clip(bin_idx, 0, nbins-1)
    ps1d = np.zeros(nbins, dtype=float)
    counts = np.bincount(bin_idx, minlength=nbins)
    sums = np.bincount(bin_idx, weights=power2d.ravel(), minlength=nbins)
    valid = counts > 0
    ps1d[valid] = sums[valid] / counts[valid]
    k_centers = 0.5*(k_edges[:-1] + k_edges[1:])
    # Fit slope on a central k-range to avoid windowing/aliasing
    kmin = kmin_frac * kmax
    kmax_fit = kmax_frac * kmax
    fit_mask = (k_centers > kmin) & (k_centers < kmax_fit) & (ps1d > 0)
    if np.sum(fit_mask) < 5:
        return np.nan, k_centers
    x = np.log(k_centers[fit_mask] + 1e-12)
    y = np.log(ps1d[fit_mask] + 1e-30)
    slope, intercept = np.polyfit(x, y, 1)
    return slope, k_centers


def classify_regime(phi_total_map: np.ndarray,
                    lam_m: float) -> str:
    """
    Classify Faraday strength regime at a given λ using RMS Faraday depth across the map:
      R = 2 λ^2 σ_Φ  [radians]
    If R < 0.5 -> 'weak'; 0.5 <= R <= 1.5 -> 'transition'; R > 1.5 -> 'strong'.
    This heuristic captures LP16's idea that behavior changes when ~1 rad rotation occurs across typical paths.
    """
    sigma_phi = np.std(phi_total_map)  # RMS across (x,y) of total φ through the box
    rot = 2.0 * (lam_m**2) * sigma_phi
    if rot < 0.5:
        return "weak"
    elif rot > 1.5:
        return "strong"
    else:
        return "transition"


def recommend_measure(regime: str) -> str:
    """
    Recommendation per λ based on regime (LP16 qualitative guidance):
      - weak: PSA (spatial), plus derivative dP/dλ^2 if RM info is needed
      - transition: use both PSA and PVA
      - strong: PVA (variance vs λ) dominates
    """
    if regime == "weak":
        return "Use PSA for magnetic statistics; add dP/dλ² if you need RM fluctuations."
    if regime == "strong":
        return "Use PVA (variance vs λ); Faraday dominates, PSA is suppressed."
    return "Use both PSA and PVA; derivative measure can help isolate RM fluctuations."


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(OUTPUT_DIR)
    # Load fields
    try:
        data = load_h5_fields(H5_PATH, BX_KEY, BY_KEY, BZ_KEY,
                              ne_key=NE_KEY, cr_key=CR_KEY,
                              screen_bz_key=(EXT_BZ_KEY if USE_EXTERNAL_SCREEN else None),
                              screen_ne_key=(EXT_NE_KEY if USE_EXTERNAL_SCREEN else None))
    except Exception as e:
        print("Error loading H5 file:", e)
        print("Edit H5_PATH and dataset keys at top of this script.")
        return

    bx, by, bz = data["bx"], data["by"], data["bz"]
    Nz, Ny, Nx = bx.shape
    print(f"Loaded fields: Bx,By,Bz with shape (Nz,Ny,Nx) = {bx.shape}")

    ne = data.get("ne", None)
    if ne is None:
        print(f"No electron density dataset '{NE_KEY}' found. Using constant n_e = {NE_FALLBACK_CM3} cm^-3.")
        ne = np.full_like(bx, fill_value=NE_FALLBACK_CM3, dtype=np.float32)

    cr = data.get("cr", None)

    # Build intrinsic polarization Pi(x,y,z)
    Pi = intrinsic_polarization(bx, by, a_exp=A_EXP, cr=cr)

    # Choose Faraday screen: internal (use bz, ne) or external (bz_screen, ne_screen)
    if USE_EXTERNAL_SCREEN:
        if ("bz_screen" not in data) or ("ne_screen" not in data):
            print("External screen keys not found; falling back to internal Faraday rotation.")
            phi_inc = faraday_depth_density(ne, bz, DELTA_Z_PC)
        else:
            phi_inc = faraday_depth_density(data["ne_screen"], data["bz_screen"], DELTA_Z_PC)
            print("Using EXTERNAL Faraday screen for rotation.")
    else:
        phi_inc = faraday_depth_density(ne, bz, DELTA_Z_PC)

    # Cumulative φ(z)
    phi_cum = cumulative_phi(phi_inc, observer_at_z0=OBSERVER_AT_Z0)
    # Total φ to far boundary (for regime classification maps): φ_total(x,y) = φ(z=Nz-1) if observer_at_z0 else φ(z=0)
    phi_total_map = phi_cum[-1] if OBSERVER_AT_Z0 else phi_cum[0]

    # Optional: restrict emissivity region (for external-emitter case).
    los_mask = None
    if USE_EXTERNAL_SCREEN:
        # Example: emission only beyond some depth Ls..Lf (indices). Set by hand if you want to mimic Appendix C.
        # By default we emit from the whole cube.
        los_mask = None

    # Build P maps for all λ
    P_maps = {}
    for lam in LAMBDA_LIST_M:
        P_map = integrate_polarization(Pi, phi_cum, lam, los_mask=los_mask)
        P_maps[lam] = P_map
        regime = classify_regime(phi_total_map, lam)
        print(f"λ = {lam:.3f} m -> regime: {regime}  | recommendation: {recommend_measure(regime)}")

    # Compute measures
    measures = compute_measures(P_maps, LAMBDA_LIST_M)

    # Report PSA slopes
    print("\nPSA power-spectrum slopes of |P| (log P(k) vs log k):")
    for lam, slope in measures["psa"]["slope_by_lambda"].items():
        print(f"  λ = {lam:.3f} m : slope ≈ {slope:.3f}")

    # Report PVA
    lam_arr = measures["pva"]["lambda"]
    var_absP = measures["pva"]["var_absP"]
    print("\nPVA (variance of |P| vs λ):")
    for lam, v in zip(lam_arr, var_absP):
        print(f"  λ = {lam:.3f} m : Var(|P|) = {v:.6e}")

    # Derivative measure
    lam_mid = measures["dPdl2"]["lambda_mid"]
    var_dPdl2 = measures["dPdl2"]["var_abs_dPdl2"]
    print("\nDerivative measure (variance of |dP/d(λ^2)| vs λ_mid):")
    for lam_m, v in zip(lam_mid, var_dPdl2):
        print(f"  λ_mid ≈ {lam_m:.3f} m : Var(|dP/d(λ^2)|) = {v:.6e}")

    # Optional diagnostics & plots
    if MAKE_PLOTS:
        # 1) Example P maps at first/last λ
        lam_min, lam_max = LAMBDA_LIST_M[0], LAMBDA_LIST_M[-1]
        plt.figure()
        plt.title(f"$|P|(\\lambda={lam_min:.3f}$ m)")
        plt.imshow(np.abs(P_maps[lam_min]))
        plt.colorbar()
        if SAVE_PNG:
            plt.savefig(os.path.join(OUTPUT_DIR, f"P_abs_lam_{lam_min:.3f}.png"), dpi=160, bbox_inches="tight")

        plt.figure()
        plt.title(f"$|P|(\\lambda={lam_max:.3f}$ m)")
        plt.imshow(np.abs(P_maps[lam_max]))
        plt.colorbar()
        if SAVE_PNG:
            plt.savefig(os.path.join(OUTPUT_DIR, f"P_abs_lam_{lam_max:.3f}.png"), dpi=160, bbox_inches="tight")

        # 2) PVA: Var(|P|) vs λ
        plt.figure()
        plt.title("PVA: $\\mathrm{Var}(|P|)$ vs $\\lambda$")
        plt.plot(lam_arr, var_absP, marker="o")
        plt.xlabel("$\\lambda$ (m)")
        plt.ylabel("$\\mathrm{Var}(|P|)$")
        if SAVE_PNG:
            plt.savefig(os.path.join(OUTPUT_DIR, "PVA_var_vs_lambda.png"), dpi=160, bbox_inches="tight")

        # 3) Derivative measure: Var(|dP/d(λ^2)|) vs λ_mid
        plt.figure()
        plt.title("Derivative measure: $\\mathrm{Var}(|dP/d(\\lambda^2)|)$ vs $\\lambda_{\\mathrm{mid}}$")
        plt.plot(lam_mid, var_dPdl2, marker="o")
        plt.xlabel("$\\lambda_{\\mathrm{mid}}$ (m)")
        plt.ylabel("$\\mathrm{Var}(|dP/d(\\lambda^2)|)$")
        if SAVE_PNG:
            plt.savefig(os.path.join(OUTPUT_DIR, "derivative_var_vs_lambda.png"), dpi=160, bbox_inches="tight")

        # 4) PSA slopes vs λ
        slopes = [measures["psa"]["slope_by_lambda"][lam] for lam in LAMBDA_LIST_M]
        plt.figure()
        plt.title("PSA: Power-spectrum slope of $|P|$ vs $\\lambda$")
        plt.plot(LAMBDA_LIST_M, slopes, marker="o")
        plt.xlabel("$\\lambda$ (m)")
        plt.ylabel("slope")
        if SAVE_PNG:
            plt.savefig(os.path.join(OUTPUT_DIR, "PSA_slope_vs_lambda.png"), dpi=160, bbox_inches="tight")

        # 5) Histogram of total φ across map (for regime intuition)
        plt.figure()
        plt.title("Histogram of total Faraday depth $\\Phi$ across map")
        plt.hist(phi_total_map.ravel(), bins=100)
        plt.xlabel("$\\Phi$ (rad m$^{-2}$)")
        if SAVE_PNG:
            plt.savefig(os.path.join(OUTPUT_DIR, "phi_total_hist.png"), dpi=160, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    main()
