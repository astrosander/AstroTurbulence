
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List

"""
Separated-Geometry Synchrotron + Faraday Screen Simulator
----------------------------------------------------------
Implements the LP16 formalism for an *external* Faraday screen
(observer -> screen of thickness L -> vacuum -> remote synchrotron-emitting halo between Ls and Lf),
plus the single-frequency directional measure S(R) and spectrum Pdir(k) from the newer work.

Geometry: Lf > Ls > L  (screen stops at z < L; emission is from Ls <= z < Lf)

Outputs (exactly two figures):
  1) directional_spectrum_vs_k.png  (Pdir(k) overlays for a few λ showing transition)
  2) nrms_vs_lambda_scorecard.png   (Nrms(λ) curve with shaded regimes + which measure to use)

Measures computed:
  - PSA: spatial power spectrum slope of |P| at fixed λ
  - PVA: Var(|P|) vs λ
  - Derivative: Var(|dP/d(λ²)|) vs λ (finite diff in λ²)
  - NEW: S(R) ≡ <cos(2[χ(x) - χ(x+R)])>; and Pdir(k) = |FFT(cos 2χ)|² + |FFT(sin 2χ)|²

Edit the USER SETTINGS block below to point to your data.
"""

# ==============================
# USER SETTINGS
# ==============================
H5_PATH = r"..\faradays_angles_stats\lp_structure_tests\mhd_fields.h5"  # <-- edit to your local file
# What datasets to read
BX_KEY, BY_KEY, BZ_KEY = "bcc1", "bcc2", "bcc3"   # magnetic field components
NE_KEY = "ne" 
CR_KEY = None               # optional cosmic-ray density (weights emissivity); set to dataset name or None

# Units (convert your HDF5 units to μG, cm^-3, pc):
B_UNIT_TO_uG   = 1.0
NE_UNIT_TO_CM3 = 1.0
DELTA_Z_PC     = 1.0

# Separated geometry indices (integers; will be clamped into [0, Nz])
L_index  = 80     # screen thickness: screen spans z ∈ [0, L_index)
Ls_index = 120    # emission start index (in the halo)
Lf_index = 256    # emission end index (exclusive); requires Lf > Ls > L

# Wavelengths (meters); keep a modest count for speed. You can extend this list.
LAMBDA_LIST_M = [0.05, 0.10, 0.21, 0.60, 1.00]

# Intrinsic synchrotron emissivity exponent: a=2 => P_i ∝ (Bx + i By)^2
A_EXP = 2.0

# Output
OUTDIR = "faraday_screen_outputs"
SAVE_FIGS = True
SHOW_FIGS = False  # set True to see windows when running locally

# LP16 / screen diagnostic thresholds (heuristic):
# Nrms = λ² σ_Φ / (2π). Weak if < 0.3; Transition if 0.3–1; Strong if > 1.
WEAK_MAX = 0.3
STRONG_MIN = 1.0

# ==============================

K_RM = 0.81  # rad m^-2 per (cm^-3 μG pc)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def orient_to_znynx(arr: np.ndarray) -> np.ndarray:
    """Heuristic: make array shape (Nz, Ny, Nx)."""
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

def load_h5(H5_PATH: str,
            BX_KEY: str, BY_KEY: str, BZ_KEY: str,
            NE_KEY: Optional[str], CR_KEY: Optional[str]) -> Dict[str, np.ndarray]:
    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"HDF5 file not found: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        bx = orient_to_znynx(np.array(f[BX_KEY], dtype=np.float32))
        by = orient_to_znynx(np.array(f[BY_KEY], dtype=np.float32))
        bz = orient_to_znynx(np.array(f[BZ_KEY], dtype=np.float32))
        data = {"bx": bx, "by": by, "bz": bz}
        if NE_KEY is not None and NE_KEY in f:
            data["ne"] = orient_to_znynx(np.array(f[NE_KEY], dtype=np.float32))
        if CR_KEY is not None and CR_KEY in f:
            data["cr"] = orient_to_znynx(np.array(f[CR_KEY], dtype=np.float32))
    return data

def intrinsic_polarization(bx: np.ndarray, by: np.ndarray, a_exp: float, cr: Optional[np.ndarray]) -> np.ndarray:
    """Complex intrinsic polarization P_i(x,y,z). For a=2 use (Bx + i By)^2."""
    if a_exp == 2.0:
        Pi = (bx + 1j*by)**2
    else:
        amp = np.power(np.maximum(bx*bx + by*by, 1e-30), 0.5*a_exp)
        psi = np.arctan2(by, bx)
        Pi = amp * np.exp(2j*psi)
    if cr is not None:
        Pi = Pi * (cr / (np.mean(cr) + 1e-30))
    return Pi

def faraday_increment(ne: np.ndarray, bz: np.ndarray, delta_z_pc: float) -> np.ndarray:
    """φ increment per layer in rad m^-2: K n_e Bz Δz."""
    return K_RM * ne * bz * delta_z_pc

def cumulative_phi_screen(phi_inc: np.ndarray, L_index: int) -> np.ndarray:
    """Build a φ(z) that accumulates ONLY up to the screen thickness L_index.
       For z >= L_index, φ(z) is constant (equal to total screen φ)."""
    Nz = phi_inc.shape[0]
    L = max(0, min(L_index, Nz))
    # Only keep increments inside the screen
    phi_inc_scr = np.zeros_like(phi_inc)
    phi_inc_scr[:L] = phi_inc[:L]
    phi_cum = np.cumsum(phi_inc_scr, axis=0)
    # For z >= L, cumulative stays equal to phi_cum[L-1] (or 0 if L=0)
    if L < Nz:
        phi_cum[L:] = phi_cum[L-1] if L > 0 else 0.0
    return phi_cum

def integrate_p(Pi: np.ndarray, phi_cum: np.ndarray, lam_m: float, Ls: int, Lf: int) -> np.ndarray:
    """Observed P(x,y;λ) = sum over emitter slab [Ls:Lf) of P_i * e^{2i λ² φ_screen(x,y)}."""
    Nz = Pi.shape[0]
    s = max(0, min(Ls, Nz))
    f = max(0, min(Lf, Nz))
    if f <= s:
        raise ValueError("Require Lf > Ls for emission region.")
    # e^{2i λ² φ(z)}; for z >= L, φ is constant across z as constructed
    phase = np.exp(2j * (lam_m**2) * phi_cum)
    integrand = Pi * phase
    return np.sum(integrand[s:f], axis=0)

def map_angles(P: np.ndarray) -> np.ndarray:
    """Polarization angle χ = 0.5 * arg(P)."""
    return 0.5 * np.angle(P)

def ring_average_power2d(field2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (k_centers, P1D(k)) from isotropic ring-average of |FFT(field2d)|^2."""
    ny, nx = field2d.shape
    F = np.fft.rfft2(field2d)
    P2 = np.abs(F)**2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.rfftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX**2 + KY**2)
    nbins = int(np.sqrt(nx*ny) // 2)
    kmax = 0.5 * min(nx, ny)
    k_edges = np.linspace(0.0, kmax, nbins+1)
    idx = np.digitize(KR.ravel(), k_edges) - 1
    # Clip bin indices to valid range [0, nbins-1]
    idx = np.clip(idx, 0, nbins-1)
    counts = np.bincount(idx, minlength=nbins)
    sums = np.bincount(idx, weights=P2.ravel(), minlength=nbins)
    P1D = np.zeros(nbins, dtype=float)
    valid = counts > 0
    P1D[valid] = sums[valid] / counts[valid]
    k_centers = 0.5*(k_edges[:-1] + k_edges[1:])
    return k_centers, P1D

def psa_slope_absP(P_map: np.ndarray) -> float:
    """Slope of |P| power spectrum in a mid-k fit window."""
    k, P1D = ring_average_power2d(np.abs(P_map))
    if np.all(P1D <= 0):
        return np.nan
    kmax = np.max(k)
    mask = (k > 0.05*kmax) & (k < 0.4*kmax) & (P1D > 0)
    if np.sum(mask) < 5:
        return np.nan
    x = np.log(k[mask] + 1e-12)
    y = np.log(P1D[mask] + 1e-30)
    slope, _ = np.polyfit(x, y, 1)
    return slope

def directional_spectrum_from_angles(chi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pdir(k) = |FFT(cos 2χ)|² + |FFT(sin 2χ)|² and ring-average to 1D."""
    A = np.cos(2.0*chi)
    B = np.sin(2.0*chi)
    k1, PA = ring_average_power2d(A)
    k2, PB = ring_average_power2d(B)
    # Both ring-averages share same bins by construction
    return k1, PA + PB

def dPdl2_variance(P_by_lambda: Dict[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Finite difference in λ² for Var(|dP/d(λ²)|). Returns (λ_mid, var_array)."""
    lam = np.array(sorted(P_by_lambda.keys()), dtype=float)
    l2 = lam**2
    var_list = []
    lam_mid = []
    for i in range(len(l2)-1):
        dP = (P_by_lambda[lam[i+1]] - P_by_lambda[lam[i]]) / (l2[i+1] - l2[i])
        var_list.append(np.var(np.abs(dP)))
        lam_mid.append(np.sqrt(0.5*(l2[i+1] + l2[i])))
    return np.array(lam_mid), np.array(var_list)

def main():
    ensure_dir(OUTDIR)
    data = load_h5(H5_PATH, BX_KEY, BY_KEY, BZ_KEY, NE_KEY, CR_KEY)
    bx = data["bx"] * B_UNIT_TO_uG
    by = data["by"] * B_UNIT_TO_uG
    bz = data["bz"] * B_UNIT_TO_uG
    Nz, Ny, Nx = bx.shape
    print(f"Loaded fields: (Nz,Ny,Nx) = {bx.shape}")
    if "ne" in data:
        ne = data["ne"] * NE_UNIT_TO_CM3
        print("Using electron density from dataset:", NE_KEY)
    else:
        ne = np.full_like(bx, 0.03*NE_UNIT_TO_CM3, dtype=np.float32)
        print("No electron density dataset found; using constant n_e = 0.03 cm^-3.")

    # Clamp geometry indices & validate Lf > Ls > L
    L = max(0, min(L_index, Nz))
    Ls = max(0, min(Ls_index, Nz))
    Lf = max(0, min(Lf_index, Nz))
    if not (Lf > Ls > L):
        raise ValueError(f"Require Lf > Ls > L, got L={L}, Ls={Ls}, Lf={Lf}")

    # Construct intrinsic polarization ONLY in the emitter slab [Ls:Lf)
    Pi_full = intrinsic_polarization(bx, by, A_EXP, data.get("cr", None))
    Pi = np.zeros_like(Pi_full, dtype=np.complex64)
    Pi[Ls:Lf] = Pi_full[Ls:Lf]

    # Faraday screen increments from z ∈ [0, L)
    phi_inc_all = faraday_increment(ne, bz, DELTA_Z_PC)
    phi_cum = cumulative_phi_screen(phi_inc_all, L)

    # Total screen φ map and its RMS for regime classification & Nrms
    phi_total_map = phi_cum[L-1] if L > 0 else np.zeros((Ny, Nx), dtype=np.float32)
    sigma_phi = float(np.std(phi_total_map))
    print(f"σ_Φ (screen) = {sigma_phi:.3e} rad m^-2")

    # Build P(x,y; λ) maps
    P_by_lambda: Dict[float, np.ndarray] = {}
    psa_by_lambda: Dict[float, float] = {}
    var_absP_by_lambda: Dict[float, float] = {}
    nrms_by_lambda: Dict[float, float] = {}

    for lam in LAMBDA_LIST_M:
        P_map = integrate_p(Pi, phi_cum, lam, Ls, Lf)
        P_by_lambda[lam] = P_map
        psa_by_lambda[lam] = psa_slope_absP(P_map)
        var_absP_by_lambda[lam] = float(np.var(np.abs(P_map)))
        nrms = (lam**2) * sigma_phi / (2.0*np.pi)
        nrms_by_lambda[lam] = nrms
        # Print regime recommendation
        if nrms < WEAK_MAX:
            rec = "Weak Faraday: PSA & derivative best; Pdir reflects emission angles."
        elif nrms > STRONG_MIN:
            rec = "Strong Faraday: PVA & Pdir best (screen-dominated)."
        else:
            rec = "Transition: combine PSA+PVA; Pdir useful; derivative helps."
        print(f"λ={lam:.3f} m  Nrms={nrms:.3f}  -> {rec}")

    # Derivative measure
    lam_mid, var_dPdl2 = dPdl2_variance(P_by_lambda)

    # === Figure 1: Directional spectrum Pdir(k) overlays for a few λ ===
    # Pick up to 3 lambdas: low, mid, high
    lam_plot = sorted(LAMBDA_LIST_M)
    pick = [lam_plot[0], lam_plot[len(lam_plot)//2], lam_plot[-1]]
    plt.figure()
    for lam in pick:
        chi = map_angles(P_by_lambda[lam])
        k, Pdir = directional_spectrum_from_angles(chi)
        plt.loglog(k[k>0], Pdir[k>0], label=f"λ={lam:.2f} m (Nrms={nrms_by_lambda[lam]:.2f})")
    plt.xlabel("k (pixels$^{-1}$)")
    plt.ylabel("$P_{dir}(k)$")
    plt.title("Directional Spectrum $P_{dir}(k)$ (external screen)")
    plt.legend()
    if SAVE_FIGS:
        plt.savefig(os.path.join(OUTDIR, "directional_spectrum_vs_k.png"), dpi=160, bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
    plt.close()

    # === Figure 2: Nrms(λ) scorecard with shaded regimes & labels ===
    lam_arr = np.array(sorted(nrms_by_lambda.keys()))
    nrms_arr = np.array([nrms_by_lambda[l] for l in lam_arr])
    plt.figure()
    plt.loglog(lam_arr, nrms_arr, marker="o")
    # Shade regions
    lam_min, lam_max = lam_arr.min(), lam_arr.max()
    y_lo, y_hi = np.min(nrms_arr[nrms_arr>0]) / 2.0, np.max(nrms_arr) * 2.0
    # Weak
    plt.fill_between([lam_min, lam_max], WEAK_MAX, y_lo, where=[True, True], alpha=0.1, label="Weak Faraday")
    # Transition (approx)
    plt.fill_between([lam_min, lam_max], STRONG_MIN, WEAK_MAX, where=[True, True], alpha=0.1, label="Transition")
    # Strong
    plt.fill_between([lam_min, lam_max], y_hi, STRONG_MIN, where=[True, True], alpha=0.1, label="Strong Faraday")
    plt.xlabel("λ (m)")
    plt.ylabel(r"$N_{\rm rms}(\lambda)=\lambda^2\sigma_{\Phi}/(2\pi)$")
    plt.title("Regime scorecard & which measure to use")
    # Text hints
    # Keep text simple to avoid overlapping; place three annotations
    plt.text(lam_arr[0], WEAK_MAX*0.6, "Weak: PSA + derivative\n(Pdir traces emission angles)", fontsize=9)
    plt.text(np.sqrt(lam_min*lam_max), np.sqrt(WEAK_MAX*STRONG_MIN), "Transition: combine PSA + PVA\nPdir helps", fontsize=9, ha="center")
    plt.text(lam_max, STRONG_MIN*1.5, "Strong: PVA + Pdir (screen)\nPSA suppressed", fontsize=9, ha="right")
    plt.legend(loc="lower right")
    if SAVE_FIGS:
        plt.savefig(os.path.join(OUTDIR, "nrms_vs_lambda_scorecard.png"), dpi=160, bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
    plt.close()

    # Print quick summary of PSA slope, PVA, derivative
    print("\n=== PSA power-spectrum slope of |P| ===")
    for lam in sorted(psa_by_lambda.keys()):
        print(f"  λ={lam:.3f} m : slope ≈ {psa_by_lambda[lam]:.3f}")
    print("\n=== PVA (Var|P|) vs λ ===")
    for lam in sorted(var_absP_by_lambda.keys()):
        print(f"  λ={lam:.3f} m : Var(|P|) = {var_absP_by_lambda[lam]:.6e}")
    print("\n=== Derivative Var(|dP/d(λ²)|) vs λ_mid ===")
    for lmid, vv in zip(lam_mid, var_dPdl2):
        print(f"  λ_mid ≈ {lmid:.3f} m : Var(|dP/d(λ²)|) = {vv:.6e}")

if __name__ == "__main__":
    main()
