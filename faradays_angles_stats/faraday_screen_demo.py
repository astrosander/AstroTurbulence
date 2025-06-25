#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faraday–screen angle statistics (separate-regions geometry)
===========================================================
Numerical experiment mirroring Lazarian & Pogosyan (2016) and the follow-up
discussion of 17 Jun 2025.

 * Generates a 2-D Kolmogorov random field Φ(X) that mimics the RM map
   of a turbulent Faraday screen of thickness L              (DΦ ∝ R^{5/3}).
 * Computes S(R,λ)  and the derived structure functions Dφ, DP.
 * Plots the results for three representative wavelengths.

Author: <your-name-here> – 2025-06-17
"""
from __future__ import annotations
import numpy as nppgadmin4
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# 1.  Synthetic Kolmogorov RM map
# ---------------------------------------------------------------------------

def generate_phi_field(
    n_pix: int = 1024,
    beta: float = 11 / 3,                 # 2-D power-spectrum slope
    rms_phi: float = 1.0,
    seed: int | None = 42
) -> np.ndarray:
    """
    Return a square 2-D field Φ(X) whose power spectrum P(k) ∝ k^{-β}.
    For Kolmogorov turbulence β = 11/3 ⇒ DΦ(R) ∝ R^{β−2} = R^{5/3}.
    """
    rng = np.random.default_rng(seed)

    # --- wavenumber grid ----------------------------------------------------
    kx = np.fft.fftfreq(n_pix)[:, None]          # column vector
    ky = np.fft.fftfreq(n_pix)[None, :]          # row vector
    kk = np.hypot(kx, ky)
    kk[0, 0] = np.inf                            # avoid division by zero

    # --- Fourier amplitudes with Kolmogorov scaling -------------------------
    amp = kk ** (-beta / 2.0)                    # |F(k)| ∝ k^{-β/2}
    phase = rng.uniform(0.0, 2 * np.pi, size=(n_pix, n_pix))
    ft = amp * np.exp(1j * phase)

    # --- back to real space -------------------------------------------------
    field = ifft2(ft).real
    field -= field.mean()
    field *= rms_phi / field.std(ddof=0)         # set σ_Φ exactly
    return field


# ---------------------------------------------------------------------------
# 2.  Radial-binning helper
# ---------------------------------------------------------------------------

def radial_profile(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Azimuthal average of *img* into integer-pixel radius bins.
    Returns (mean value, number of pixels) for r = 0 .. N/2−1.
    """
    n = img.shape[0]        # assume square & even
    assert img.shape[0] == img.shape[1], "Image must be square."

    # pixel-distance from (0,0) in the cyclic (FFT) sense
    y, x = np.indices((n, n))
    dx = (x + n // 2) % n - n // 2
    dy = (y + n // 2) % n - n // 2
    r   = np.hypot(dx, dy)
    r_int = r.astype(int)

    r_max = n // 2
    prof  = np.empty(r_max, dtype=np.float64)
    count = np.empty(r_max, dtype=np.int64)

    for ri in range(r_max):
        mask = (r_int == ri)
        count[ri] = mask.sum()
        prof[ri]  = img[mask].mean() if count[ri] else np.nan
    return prof, count


# ---------------------------------------------------------------------------
# 3.  S(R,λ)  and derived structure functions
# ---------------------------------------------------------------------------

def S_of_R(phi: np.ndarray, lam_m: float) -> np.ndarray:
    """
    Compute S(R,λ) = ⟨cos[2 λ² ΔΦ]⟩ for all integer R using FFT autocorrelation.
    ΔΦ = Φ(X) − Φ(X+R).  The trick:

        S = Re ⟨e^{i2λ²Φ(X)} e^{−i2λ²Φ(X+R)}⟩
          = Re 𝒞_{E,E}(R),

    where E(X) = exp(i 2 λ² Φ(X)).
    """
    E   = np.exp(1j * 2.0 * lam_m ** 2 * phi)
    Ehat = fft2(E)
    corr = ifft2(np.abs(Ehat) ** 2).real / phi.size        # ⟨E E*⟩
    corr = fftshift(corr)                                  # put R=(0,0) in centre
    S1d, _ = radial_profile(corr)
    return S1d


def structure_functions(phi: np.ndarray, lam_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns Dφ(R,λ)  and  DP(R,λ).
    For constant |P_i|=1,   DP = 2 ( 1 − S ).
    """
    S = S_of_R(phi, lam_m)
    Dphi = 0.5 * (1.0 - S)
    DP   = 2.0 * (1.0 - S)
    return Dphi, DP


# ---------------------------------------------------------------------------
# 4.  Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    # ---------------------------------------------------
    # Simulation parameters
    # ---------------------------------------------------
    NPIX       = 1024         # transverse map size (pixels)
    SIGMA_PHI  = 1.0          # σ_Φ in the toy model (arbitrary units)
    LAM_CM     = [11, 18, 28] # representative wavelengths [cm]
    COLORS     = ["C0", "C1", "C2"]

    # ---------------------------------------------------
    # Generate the RM field
    # ---------------------------------------------------
    phi = generate_phi_field(n_pix=NPIX, rms_phi=SIGMA_PHI, seed=20250617)
    print(f"Generated Φ field: size={NPIX}×{NPIX}, σ_Φ={phi.std():.3f}")

    # ---------------------------------------------------
    # Compute structure functions for each λ
    # ---------------------------------------------------
    r_pix = np.arange(NPIX // 2)                   # radius array (integer pixels)
    Dphi: Dict[float, np.ndarray] = {}
    DP:   Dict[float, np.ndarray] = {}

    for lam_cm, col in zip(LAM_CM, COLORS):
        lam_m = lam_cm / 100.0                     # convert to metres
        Dphi[lam_cm], DP[lam_cm] = structure_functions(phi, lam_m)
        print(f"λ = {lam_cm:5.1f} cm  →  max Dφ = {np.nanmax(Dphi[lam_cm]):.3f}")

    # ---------------------------------------------------
    # Plotting
    # ---------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for lam_cm, col in zip(LAM_CM, COLORS):
        ax1.loglog(r_pix[1:], Dphi[lam_cm][1:],  label=fr"$\lambda={lam_cm}\,\mathrm{{cm}}$", color=col)
        ax2.loglog(r_pix[1:], DP[lam_cm][1:] / (lam_cm / 100.0) ** 4,
                   label=fr"$\lambda={lam_cm}\,\mathrm{{cm}}$", color=col)

    # Reference Kolmogorov slope 5/3 --------------------
    ref = r_pix[1:] ** (5 / 3)
    ref *= Dphi[LAM_CM[0]][2] / ref[1]              # anchor visually
    ax1.loglog(r_pix[1:], ref, "k--", lw=1, label=r"$R^{5/3}$")
    ax2.loglog(r_pix[1:], ref, "k--", lw=1)

    # Cosmetics ----------------------------------------
    ax1.set_xlabel(r"$R$  [pixels]")
    ax1.set_ylabel(r"$D_{\varphi}(R,\lambda)$")
    ax2.set_xlabel(r"$R$  [pixels]")
    ax2.set_ylabel(r"$D_{P}(R,\lambda)\;/\;\lambda^{4}$")

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.tight_layout()

    # Save & show --------------------------------------
    out = Path("compare_measures.pdf")
    fig.savefig(out, dpi=300)
    print(f"Figure written to {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
