
import os
import h5py
import numpy as np
import numpy.fft as nfft
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
LAMBDA_LIST_M = np.arange(0.01, 1.5, 0.01)

# [0.03, 0.06, 0.09, 0.12, 0.15, 0.18,
#                  0.21, 0.24, 0.27, 0.30, 0.33, 0.36,
#                  0.39, 0.42, 0.45, 0.48, 0.51, 0.54,
#                  0.57, 0.60, 0.63, 0.66, 0.69, 0.72,
#                  0.75, 0.78, 0.81, 0.84, 0.87, 0.90,
#                  0.93, 0.96, 0.99]

# --- Plots ---
MAKE_PLOTS = True
SAVE_PNG = True
OUTPUT_DIR = "lp2016_outputs"

# ======================

K_RM = 0.81  # rad m^-2 per (cm^-3 μG pc)

def _hann2d(ny: int, nx: int) -> np.ndarray:
    """2D Hann window for apodization."""
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return np.outer(wy, wx)

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

def isotropic_power_slope(P_map: np.ndarray, kmin_frac: float=0.1, kmax_frac: float=0.3) -> Tuple[float, np.ndarray]:
    """Compute PSA spectrum using the exact same algorithm as psa_lp_16_like.py
    
    This function operates on the COMPLEX polarization map P, not its magnitude.
    The power spectrum is computed as |FFT(P)|^2, where P is complex.
    """
    # Remove mean (detrend) - exactly like psa_lp_16_like.py, works with complex fields
    P = P_map - np.mean(P_map)
    
    # Apply Hann window to reduce ringing/leakage - exactly like psa_lp_16_like.py
    win = _hann2d(P.shape[0], P.shape[1])
    P = P * win
    
    # 2D FFT and power - exactly like psa_lp_16_like.py
    # This computes |FFT(complex P)|^2, NOT FFT(|P|)^2
    F = nfft.fftshift(nfft.fft2(P))
    P2D = (F * np.conj(F)).real  # |F|^2
    
    ny, nx = P.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny  # dimensionless spatial frequency index
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY)
    
    # Radial bins - exactly like psa_lp_16_like.py with fixed ring_bins=64
    kmax = KR.max()
    ring_bins = 64  # Fixed value like in psa_lp_16_like.py
    bins = np.linspace(0.0, kmax, ring_bins + 1)
    which_bin = np.digitize(KR.ravel(), bins) - 1
    
    Ek = np.zeros(ring_bins, dtype=float)
    counts = np.zeros(ring_bins, dtype=int)
    
    flatP = P2D.ravel()
    for b in range(ring_bins):
        mask = which_bin == b
        if np.any(mask):
            Ek[b] = flatP[mask].mean()
            counts[b] = mask.sum()
        else:
            Ek[b] = np.nan
    
    # Bin centers - exactly like psa_lp_16_like.py
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Fit slope over the specified range
    kmin = kmin_frac * kmax
    kmax_fit = kmax_frac * kmax
    mask = (k_centers > kmin) & (k_centers < kmax_fit) & (Ek > 0) & np.isfinite(Ek)
    if np.sum(mask) < 5:
        return np.nan, k_centers
    
    x = np.log(k_centers[mask] + 1e-12)
    y = np.log(Ek[mask] + 1e-30)
    slope, _ = np.polyfit(x, y, 1)
    return slope, k_centers

# === NEW: helpers for the directional (angle-based) measure ===

def map_angles(P: np.ndarray) -> np.ndarray:
    """Polarization angle χ = 0.5 * arg(P)."""
    return 0.5 * np.angle(P)

def ring_average_power2d(P_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (k_centers, P1D(k)) from isotropic ring-average of |FFT(P_map)|^2.
    
    This function operates on the COMPLEX polarization map P, not its magnitude.
    The power spectrum is computed as |FFT(P)|^2, where P is complex.
    """
    # Remove mean (detrend) - exactly like psa_lp_16_like.py, works with complex fields
    P = P_map - np.mean(P_map)
    
    # Apply Hann window to reduce ringing/leakage - exactly like psa_lp_16_like.py
    win = _hann2d(P.shape[0], P.shape[1])
    P = P * win
    
    # 2D FFT and power - exactly like psa_lp_16_like.py
    # This computes |FFT(complex P)|^2, NOT FFT(|P|)^2
    F = nfft.fftshift(nfft.fft2(P))
    P2D = (F * np.conj(F)).real  # |F|^2
    
    ny, nx = P.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny  # dimensionless spatial frequency index
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY)
    
    # Radial bins - exactly like psa_lp_16_like.py with fixed ring_bins=64
    kmax = KR.max()
    ring_bins = 64  # Fixed value like in psa_lp_16_like.py
    bins = np.linspace(0.0, kmax, ring_bins + 1)
    which_bin = np.digitize(KR.ravel(), bins) - 1
    
    P1D = np.zeros(ring_bins, dtype=float)
    counts = np.zeros(ring_bins, dtype=int)
    
    flatP = P2D.ravel()
    for b in range(ring_bins):
        mask = which_bin == b
        if np.any(mask):
            P1D[b] = flatP[mask].mean()
            counts[b] = mask.sum()
        else:
            P1D[b] = np.nan
    
    # Bin centers - exactly like psa_lp_16_like.py
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    return k_centers, P1D

def directional_spectrum_from_angles(chi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Directional spectrum P_dir(k) = |FFT(cos 2χ)|^2 + |FFT(sin 2χ)|^2, ring-averaged to 1D."""
    A = np.cos(2.0 * chi)
    B = np.sin(2.0 * chi)
    kA, PA = ring_average_power2d(A)
    kB, PB = ring_average_power2d(B)
    # kA and kB share the same bins by construction
    return kA, PA + PB

def directional_slope(P_map: np.ndarray, kmin_frac: float=0.1, kmax_frac: float=0.3) -> float:
    """Slope of the directional spectrum P_dir(k) over a mid-k range."""
    chi = map_angles(P_map)
    k, Pdir = directional_spectrum_from_angles(chi)
    if np.all(Pdir <= 0):
        return np.nan
    kmax = np.max(k)
    mask = (k > kmin_frac*kmax) & (k < kmax_frac*kmax) & (Pdir > 0)
    if np.sum(mask) < 5:
        return np.nan
    x = np.log(k[mask] + 1e-12)
    y = np.log(Pdir[mask] + 1e-30)
    slope, _ = np.polyfit(x, y, 1)
    return slope

# === NEW: PFA and derivative spectrum measures (from pfa_lp_16_like.py) ===

def compute_pfa_variance(P_map: np.ndarray) -> float:
    """Compute ⟨|P|²⟩ for a single P map (PFA measure at one λ)."""
    return np.mean(np.abs(P_map)**2)

def compute_derivative_variance(dP_map: np.ndarray) -> float:
    """Compute ⟨|dP/dλ²|²⟩ for a single dP/dλ² map."""
    return np.mean(np.abs(dP_map)**2)

def dP_map_mixed_geometry(Pi: np.ndarray, phi_inc: np.ndarray, lam_m: float, voxel_depth: float = 1.0) -> np.ndarray:
    """Compute dP/dλ² map for mixed geometry: 2i ∫ Pi(z) Φ(z) e^{2i λ² Φ(z)} dz.
    
    This follows the exact algorithm from pfa_lp_16_like.py dP_map_mixed().
    """
    # Cumulative Faraday depth Φ(z) = ∫ φ(z') dz'
    Phi_cum = np.cumsum(phi_inc * voxel_depth, axis=0)
    
    # Phase factor exp(2i λ² Φ(z))
    phase = np.exp(2j * (lam_m**2) * Phi_cum)
    
    # Derivative integrand: 2i Pi(z) Φ(z) exp(2i λ² Φ(z))
    contrib = 2j * (Pi * Phi_cum) * phase
    
    # Integrate along LOS
    return np.sum(contrib, axis=0) * voxel_depth

def P_map_mixed_geometry(Pi: np.ndarray, phi_inc: np.ndarray, lam_m: float, voxel_depth: float = 1.0) -> np.ndarray:
    """Compute P map for mixed geometry: ∫ Pi(z) e^{2i λ² Φ(z)} dz.
    
    This follows the exact algorithm from pfa_lp_16_like.py P_map_mixed().
    """
    # Cumulative Faraday depth Φ(z) = ∫ φ(z') dz'
    Phi_cum = np.cumsum(phi_inc * voxel_depth, axis=0)
    
    # Phase factor exp(2i λ² Φ(z))
    phase = np.exp(2j * (lam_m**2) * Phi_cum)
    
    # Integrand: Pi(z) exp(2i λ² Φ(z))
    contrib = Pi * phase
    
    # Integrate along LOS
    return np.sum(contrib, axis=0) * voxel_depth

def dP_map_from_phi(Pi: np.ndarray, phi_cum: np.ndarray, lam_m: float, los_mask: Optional[np.ndarray]) -> np.ndarray:
    """Compute dP/dλ² map for separated geometry: 2i ∫ Pi(z) Φ(z) e^{2i λ² Φ(z)} dz."""
    phase = np.exp(2j * (lam_m**2) * phi_cum)
    if los_mask is not None:
        integrand = np.where(los_mask, 2j * Pi * phi_cum * phase, 0.0)
    else:
        integrand = 2j * Pi * phi_cum * phase
    return np.sum(integrand, axis=0)

def compute_measures(P_maps: Dict[float, np.ndarray], lam_list: List[float], 
                     dP_maps: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, Dict]:
    psa_slopes = {}
    pdir_slopes = {}   # NEW
    pfa_variances = {}  # NEW: PFA variance ⟨|P|²⟩
    derivative_variances = {}  # NEW: Derivative spectrum variance ⟨|dP/dλ²|²⟩
    kbins = None
    for lam in lam_list:
        # PSA on COMPLEX P (not |P|) - this is the correct way per LP16
        # PSA computes power spectrum as |FFT(P)|^2 where P is complex
        slope_psa, kb = isotropic_power_slope(P_maps[lam])
        psa_slopes[lam] = slope_psa
        if kbins is None:
            kbins = kb
        # NEW: directional spectrum slope from polarization angles
        pdir_slopes[lam] = directional_slope(P_maps[lam])
        # NEW: PFA variance
        pfa_variances[lam] = compute_pfa_variance(P_maps[lam])
        # NEW: Derivative variance (if dP_maps provided)
        if dP_maps is not None and lam in dP_maps:
            derivative_variances[lam] = compute_derivative_variance(dP_maps[lam])

    lam_arr = np.array(lam_list, dtype=float)
    # Removed PVA variance calculation as it drops to zero

    # Derivative vs λ^2 (original finite-difference method)
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
        "psa":  {"slope_by_lambda": psa_slopes,  "k_bins": kbins},
        "dPdl2":{"lambda_mid": np.sqrt(np.array(l2_mid)), "var_abs_dPdl2": np.array(dPdl2_vars)},
        "pdir": {"slope_by_lambda": pdir_slopes},  # NEW
        "pfa":  {"variance_by_lambda": pfa_variances},  # NEW: PFA spectrum
        "derivative_spectrum": {"variance_by_lambda": derivative_variances}  # NEW: Derivative spectrum
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

    # Build P maps and dP/dλ² maps across λ
    P_maps = {}
    dP_maps = {}
    for lam in LAMBDA_LIST_M:
        if USE_EXTERNAL_SCREEN and SEPARATED_FROM_SAME_CUBE:
            # Separated geometry: use existing method
            P_maps[lam] = integrate_polarization(Pi, phi_cum, lam, los_mask=los_mask)
            dP_maps[lam] = dP_map_from_phi(Pi, phi_cum, lam, los_mask=los_mask)
        else:
            # Mixed geometry: use exact algorithms from pfa_lp_16_like.py
            P_maps[lam] = P_map_mixed_geometry(Pi, phi_inc, lam, DELTA_Z_PC)
            dP_maps[lam] = dP_map_mixed_geometry(Pi, phi_inc, lam, DELTA_Z_PC)
        
        regime = classify_regime(phi_total_map, lam)
        print(f"λ = {lam:.3f} m -> regime: {regime} | {recommend_measure(regime)}")

    # Measures
    measures = compute_measures(P_maps, LAMBDA_LIST_M, dP_maps=dP_maps)
    
    # Compute PFA and derivative spectrum curves (exact algorithms from reference files)
    print("\nComputing PFA and derivative spectrum curves...")
    
    # PFA curve: ⟨|P|²⟩ vs λ²
    pfa_lambda_squared = []
    pfa_variances = []
    for lam in LAMBDA_LIST_M:
        pfa_lambda_squared.append(lam**2)
        pfa_variances.append(measures['pfa']['variance_by_lambda'][lam])
    
    # Derivative spectrum curve: ⟨|dP/dλ²|²⟩ vs λ²  
    derivative_lambda_squared = []
    derivative_spectrum_variances = []
    for lam in LAMBDA_LIST_M:
        derivative_lambda_squared.append(lam**2)
        if lam in measures['derivative_spectrum']['variance_by_lambda']:
            derivative_spectrum_variances.append(measures['derivative_spectrum']['variance_by_lambda'][lam])
        else:
            derivative_spectrum_variances.append(np.nan)
    
    # Add to measures for plotting
    measures['pfa_curve'] = {
        'lambda_squared': np.array(pfa_lambda_squared),
        'variances': np.array(pfa_variances)
    }
    measures['derivative_spectrum_curve'] = {
        'lambda_squared': np.array(derivative_lambda_squared), 
        'variances': np.array(derivative_spectrum_variances)
    }

    # Print summaries
    print("\nPSA power-spectrum slopes of |P|:")
    for lam in LAMBDA_LIST_M:
        print(f"  λ = {lam:.3f} m : slope ≈ {measures['psa']['slope_by_lambda'][lam]:.3f}")

    print("\nDerivative measure (variance of |dP/d(λ^2)|) vs λ_mid:")
    for lam_m, v in zip(measures["dPdl2"]["lambda_mid"], measures["dPdl2"]["var_abs_dPdl2"]):
        print(f"  λ_mid ≈ {lam_m:.3f} m : Var(|dP/d(λ^2)|) = {v:.6e}")

    print("\nDirectional spectrum slope (new measure) vs λ:")
    for lam in LAMBDA_LIST_M:
        print(f"  λ = {lam:.3f} m : slope ≈ {measures['pdir']['slope_by_lambda'][lam]:.3f}")
    
    print("\nPFA variance ⟨|P|²⟩ vs λ:")
    for lam in LAMBDA_LIST_M:
        print(f"  λ = {lam:.3f} m : ⟨|P|²⟩ = {measures['pfa']['variance_by_lambda'][lam]:.6e}")
    
    print("\nDerivative spectrum ⟨|dP/dλ²|²⟩ vs λ:")
    for lam in LAMBDA_LIST_M:
        if lam in measures['derivative_spectrum']['variance_by_lambda']:
            print(f"  λ = {lam:.3f} m : ⟨|dP/dλ²|²⟩ = {measures['derivative_spectrum']['variance_by_lambda'][lam]:.6e}")

    # Compute full power spectra for all wavelengths
    print(f"\nComputing full power spectra for all wavelengths...")
    
    # PSA power spectra for all wavelengths (on COMPLEX P, not magnitude)
    psa_spectra = {}
    for lam in LAMBDA_LIST_M:
        k_centers, ps1d = ring_average_power2d(P_maps[lam])  # Pass complex P_map
        psa_spectra[lam] = {'k': k_centers, 'P': ps1d}
    
    # Directional power spectra for all wavelengths
    directional_spectra = {}
    for lam in LAMBDA_LIST_M:
        chi = map_angles(P_maps[lam])
        k_dir, P_dir = directional_spectrum_from_angles(chi)
        directional_spectra[lam] = {'k': k_dir, 'P': P_dir}

    # Save all data to NPZ file for later replotting
    print(f"\nSaving all measures and full spectra to NPZ file...")
    
    # Prepare data for saving
    save_data = {
        'lambda_list': np.array(LAMBDA_LIST_M),
        'psa_slopes': np.array([measures['psa']['slope_by_lambda'][lam] for lam in LAMBDA_LIST_M]),
        'psa_k_bins': measures['psa']['k_bins'],
        'derivative_lambda_mid': measures['dPdl2']['lambda_mid'],
        'derivative_variance': measures['dPdl2']['var_abs_dPdl2'],
        'directional_slopes': np.array([measures['pdir']['slope_by_lambda'][lam] for lam in LAMBDA_LIST_M]),
        'pfa_variances': np.array([measures['pfa']['variance_by_lambda'][lam] for lam in LAMBDA_LIST_M]),
        'derivative_spectrum_variances': np.array([measures['derivative_spectrum']['variance_by_lambda'].get(lam, np.nan) for lam in LAMBDA_LIST_M]),
        'pfa_curve_lambda_squared': measures['pfa_curve']['lambda_squared'],
        'pfa_curve_variances': measures['pfa_curve']['variances'],
        'derivative_spectrum_curve_lambda_squared': measures['derivative_spectrum_curve']['lambda_squared'],
        'derivative_spectrum_curve_variances': measures['derivative_spectrum_curve']['variances'],
        'phi_total_map': phi_total_map,
        'regime_classification': np.array([classify_regime(phi_total_map, lam) for lam in LAMBDA_LIST_M])
    }
    
    # Add full power spectra for all wavelengths
    for lam in LAMBDA_LIST_M:
        # PSA spectra
        save_data[f'psa_k_lambda_{lam:.3f}'] = psa_spectra[lam]['k']
        save_data[f'psa_P_lambda_{lam:.3f}'] = psa_spectra[lam]['P']
        
        # Directional spectra
        save_data[f'dir_k_lambda_{lam:.3f}'] = directional_spectra[lam]['k']
        save_data[f'dir_P_lambda_{lam:.3f}'] = directional_spectra[lam]['P']
    
    # Add polarization maps for representative wavelengths
    representative_lambdas = [LAMBDA_LIST_M[0], LAMBDA_LIST_M[len(LAMBDA_LIST_M)//2], LAMBDA_LIST_M[-1]]
    for i, lam in enumerate(representative_lambdas):
        save_data[f'P_map_lambda_{lam:.3f}'] = P_maps[lam]
    
    # Save to NPZ file
    npz_filename = os.path.join(OUTPUT_DIR, "measures_data.npz")
    np.savez(npz_filename, **save_data)
    print(f"Data saved to: {npz_filename}")
    print(f"Saved full power spectra for {len(LAMBDA_LIST_M)} wavelengths")

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
        ax.plot(LAMBDA_LIST_M, slopes_norm, 'b.-', linewidth=1.5, markersize=4, label='PSA slope')
        
        # Derivative measure (normalized to 0-1 range)
        lam_mid = measures["dPdl2"]["lambda_mid"]
        var_dPdl2 = measures["dPdl2"]["var_abs_dPdl2"]
        var_dPdl2_norm = (var_dPdl2 - np.min(var_dPdl2)) / (np.max(var_dPdl2) - np.min(var_dPdl2) + 1e-10)
        ax.plot(lam_mid, var_dPdl2_norm, 'g.-', linewidth=1.5, markersize=4, label='Derivative variance')
        
        # NEW: Directional-spectrum slope (normalized)
        pdir_slopes = [measures["pdir"]["slope_by_lambda"][l] for l in LAMBDA_LIST_M]
        pdir_norm = np.array(pdir_slopes, dtype=float)
        pdir_norm = (pdir_norm - np.nanmin(pdir_norm)) / (np.nanmax(pdir_norm) - np.nanmin(pdir_norm) + 1e-10)
        ax.plot(LAMBDA_LIST_M, pdir_norm, 'm.-', linewidth=1.5, markersize=4, label='Directional spectrum slope')
        
        # NEW: PFA variance (normalized)
        pfa_vars = [measures["pfa"]["variance_by_lambda"][l] for l in LAMBDA_LIST_M]
        pfa_norm = np.array(pfa_vars, dtype=float)
        pfa_norm = (pfa_norm - np.nanmin(pfa_norm)) / (np.nanmax(pfa_norm) - np.nanmin(pfa_norm) + 1e-10)
        ax.plot(LAMBDA_LIST_M, pfa_norm, 'r.-', linewidth=1.5, markersize=4, label='PFA variance')
        
        # NEW: Derivative spectrum variance (normalized)
        deriv_spec_vars = [measures["derivative_spectrum"]["variance_by_lambda"].get(l, np.nan) for l in LAMBDA_LIST_M]
        deriv_spec_norm = np.array(deriv_spec_vars, dtype=float)
        deriv_spec_norm = (deriv_spec_norm - np.nanmin(deriv_spec_norm)) / (np.nanmax(deriv_spec_norm) - np.nanmin(deriv_spec_norm) + 1e-10)
        ax.plot(LAMBDA_LIST_M, deriv_spec_norm, 'c.-', linewidth=1.5, markersize=4, label='Derivative spectrum variance')
        
        # ax.set_title("LP16 Measures Comparison: PSA (Inertial Range), Derivative, and Directional vs $\\lambda$", fontsize=14, fontweight='bold')
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
                            # Fill between curves and x-axis
                            ax.fill_between([wavelengths[start_idx], wavelengths[i-1]], 
                                          y_min, y_max, 
                                          color=regime_colors[regime], alpha=0.2, zorder=0)
                        start_idx = i
                # Fill the last range
                if start_idx < len(wavelengths):
                    ax.axvspan(wavelengths[start_idx], wavelengths[-1], 
                              color=regime_colors[regime], alpha=0.3, zorder=0)
                    # Fill between curves and x-axis
                    ax.fill_between([wavelengths[start_idx], wavelengths[-1]], 
                                  y_min, y_max, 
                                  color=regime_colors[regime], alpha=0.2, zorder=0)
        
        # Add regime legend
        from matplotlib.patches import Patch
        regime_patches = [
            Patch(facecolor='lightblue',  alpha=0.25, label=r'$2\lambda^{2}\sigma_{\Phi} < 1$'),
            Patch(facecolor='yellow',     alpha=0.25, label=r'$1 < 2\lambda^{2}\sigma_{\Phi} < 3$'),
            Patch(facecolor='lightcoral', alpha=0.25, label=r'$2\lambda^{2}\sigma_{\Phi} > 3$')
        ]
        # Primary curve legend + equation-based patches
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + regime_patches, labels + [p.get_label() for p in regime_patches],
                  loc='upper right', fontsize=10)
        
        plt.tight_layout()
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "all_measures_combined.png"), dpi=160, bbox_inches="tight")
        
        # PSA power spectrum for representative wavelengths
        plt.figure(figsize=(10, 8))
        
        # Select representative wavelengths (low, mid, high)
        lam_indices = [0, len(LAMBDA_LIST_M)//2, -1]
        colors = ['blue', 'red', 'green']
        
        for i, lam_idx in enumerate(lam_indices):
            lam = LAMBDA_LIST_M[lam_idx]
            P_map = P_maps[lam]
            
            # Calculate power spectrum using the same algorithm as psa_lp_16_like.py
            # Use complex P_map, not its magnitude
            k_centers, ps1d = ring_average_power2d(P_map)
            
            # Plot only valid points
            mask = ps1d > 0
            plt.loglog(k_centers[mask], ps1d[mask], '.-', color=colors[i], 
                      linewidth=1.5, markersize=4, label=f'$\\lambda = {lam:.3f}$ m')
        
        plt.xlabel("$k$ (pixel$^{-1}$)", fontsize=12)
        plt.ylabel("$P_{|P|}(k)$ (arbitrary units)", fontsize=12)
        plt.title("PSA Power Spectrum: $P_{|P|}(k)$ for Representative Wavelengths", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "psa_spectrum.png"), dpi=160, bbox_inches="tight")
        
        # PFA curve: ⟨|P|²⟩ vs λ² (exact style from pfa_lp_16_like.py)
        plt.figure(figsize=(8, 6))
        pfa_lam2 = measures['pfa_curve']['lambda_squared']
        pfa_vars = measures['pfa_curve']['variances']
        plt.loglog(pfa_lam2, pfa_vars, 'b.-', linewidth=1.5, markersize=4, label='PFA mixed geometry')
        plt.xlabel("$\\lambda^2$ (m$^2$)", fontsize=12)
        plt.ylabel("$\\langle |P|^2 \\rangle$ (arbitrary units)", fontsize=12)
        plt.title("PFA: $\\langle|P|^2\\rangle$ vs $\\lambda^2$ (Mixed Geometry)", fontsize=14, fontweight='bold')
        plt.grid(True, which='both', alpha=0.2)
        plt.legend(fontsize=11)
        plt.tight_layout()
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "pfa_spectra.png"), dpi=160, bbox_inches="tight")
        
        # Derivative spectrum curve: ⟨|dP/dλ²|²⟩ vs λ² (exact style from pfa_and_derivative_lp_16_like.py)
        plt.figure(figsize=(8, 6))
        deriv_lam2 = measures['derivative_spectrum_curve']['lambda_squared']
        deriv_vars = measures['derivative_spectrum_curve']['variances']
        # Filter out NaN values
        valid_mask = ~np.isnan(deriv_vars)
        if np.any(valid_mask):
            plt.loglog(deriv_lam2[valid_mask], deriv_vars[valid_mask], 'r.-', linewidth=1.5, markersize=4, label='Derivative spectrum mixed geometry')
        plt.xlabel("$\\lambda^2$ (m$^2$)", fontsize=12)
        plt.ylabel("$\\langle |dP/d\\lambda^2|^2 \\rangle$ (arbitrary units)", fontsize=12)
        plt.title("Derivative Spectrum: $\\langle|dP/d\\lambda^2|^2\\rangle$ vs $\\lambda^2$ (Mixed Geometry)", fontsize=14, fontweight='bold')
        plt.grid(True, which='both', alpha=0.2)
        plt.legend(fontsize=11)
        plt.tight_layout()
        if SAVE_PNG: plt.savefig(os.path.join(OUTPUT_DIR, "derivative_spectra.png"), dpi=160, bbox_inches="tight")
        
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
