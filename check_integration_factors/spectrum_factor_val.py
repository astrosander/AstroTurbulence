"""
series_kappa.py
---------------

Recover the small‑R⊥ series
    Δρ(R⊥) = Σ κ_m (R⊥/L)^(5/3+m)    ,  m = 0,1,2,…

and compare the numerical κ₀,κ₁,κ₂,… with the leading analytic value.

Usage
~~~~~
    python series_kappa.py               # default: 3 terms + plot
    python series_kappa.py 5             # 5 terms
    python series_kappa.py 4 --no-plot   # fit only, no graphics

Inside a notebook:

    import series_kappa as sk
    sk.run(N_terms=4, fft_diagnose=True)

---------------------------------------------------------------------
"""

import argparse
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

# ----------------- model parameters -----------------
C2 = 1.0   # Kolmogorov prefactor (drops out of the κ_m ratios)
L0 = 1.0   # outer scale  𝓛
L  = L0    # slab depth    (keep equal for clarity)

# analytic κ₀  (same as before)
kappa0_analytic = (np.sqrt(np.pi) * gamma(-5/6)) / (2 * gamma(-1/3))  # ≈ 1.45719


# ----------------- Δρ(R⊥) numerical integral -----------------
def delta_rho(Rperp, N=8000):
    """
    Returns Δρ = ρ̄⊥(0) – ρ̄⊥(R⊥)   using the regularised integrand
        [(R⊥²+Δs²)^{1/3} – |Δs|^{2/3}] (1-|Δs|/L).
    High‑resolution trapezoidal rule is inexpensive and very accurate.
    """
    if Rperp == 0.0:
        return 0.0

    y = np.linspace(0.0, 1.0, N)                 # y = |Δs|/L ∈ [0,1]
    core = ((Rperp**2 + (y*L)**2)**(1./3.) - (y*L)**(2./3.)) * (1.0 - y)
    integral = 2.0 * L * np.trapz(core, y)       # ×2 for symmetry
    return (C2 / (2 * L0**(2./3.) * L)) * integral


# ----------------- series fit -----------------
def fit_kappa(Rvals, Δρ, N_terms=3):
    """
    Least‑squares fit for κ_m,  m = 0 … N_terms‑1,  where
        Δρ ≈ Σ κ_m x^{5/3+m},  x = R⊥/L.
    Returns κ, residuals, and the design matrix (for diagnostics).
    """
    exponents = (5.0 + 3*np.arange(N_terms)) / 3.0        # 5/3,8/3,…
    Xmat = (Rvals[:, None] / L)**exponents[None, :]
    # solve X κ = Δρ  in the least‑squares sense
    κ, *_ = np.linalg.lstsq(Xmat, Δρ, rcond=None)
    residual = Δρ - Xmat @ κ
    return κ, residual, exponents


# ----------------- driver -----------------
def run(N_terms=3,
        Rmin=4e-4, Rmax=2e-2, nR=40,
        N_int=8000,
        make_plot=True,
        fft_diagnose=False):
    """
    Main routine.  Increase N_terms to extract more κ_m.
    """
    # ---- generate Δρ data on an even grid in log R⊥ ----
    R = np.logspace(np.log10(Rmin*L), np.log10(Rmax*L), nR)
    Δρ = np.array([delta_rho(r, N=N_int) for r in R])

    # ---- fit the series ----
    κ, residual, exponents = fit_kappa(R, Δρ, N_terms=N_terms)

    # ---- print a tidy table ----
    print("\nκ‑coefficients from least‑squares fit\n"
          "m   exponent      κ_m (numeric)   sign")
    print("--  ---------  ----------------  ----")
    for m, (exp, km) in enumerate(zip(exponents, κ)):
        sign = "+" if km >= 0 else "–"
        print(f"{m:<2d}  {exp:7.3f}      {km:2.1f}   {sign}")
    print("---------")
    print(f"analytic κ₀  = {kappa0_analytic:.6e}")
    print(f"relative error on κ₀ : "
          f"{100*abs(κ[0]-kappa0_analytic)/kappa0_analytic:.3f} %")
    print(f"max |residual| / Δρ : "
          f"{100*np.max(np.abs(residual)/Δρ):.2f} % "
          "(fit quality in the chosen range)\n")

    # ---- graphics ----
    if make_plot:
        plt.figure(figsize=(5.1,4))
        plt.loglog(R/L, Δρ, 'k', lw=2, label='numerical Δρ')

        # truncated series with fitted κ
        series = sum(km * (R/L)**exp for km, exp in zip(κ, exponents))
        plt.loglog(R/L, series,
                   ls='--', lw=1.6, label=f'series ({N_terms} terms)')

        plt.xlabel(r'$R_\perp/L$')
        plt.ylabel(r'$\Delta\rho$')
        plt.title('Projected correlation and its small‑$R$ series')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---- FFT diagnostic (optional) ----
    if fft_diagnose:
        # FFT needs equally‑spaced grid in log10 R
        logR = np.log10(R/L)
        logΔρ = np.log10(Δρ)
        # detrend (subtract best straight line) ⇒ better dynamic range
        a,b = np.polyfit(logR, logΔρ, 1)
        detrended = logΔρ - (a*logR + b)

        # FFT of detrended signal
        fft = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(logR.size, d=logR[1]-logR[0])

        plt.figure(figsize=(5.1,3.5))
        plt.semilogy(freqs, np.abs(fft), lw=1.4)
        plt.xlim(0, 6)      # we only need the first few harmonics
        plt.xlabel(r'log‑space frequency  $k$')
        plt.ylabel(r'|FFT|  (arb.)')
        plt.title('FFT of  log[Δρ]  vs  log R⊥')
        plt.tight_layout()
        plt.show()


# ----------------- CLI entry‑point -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recover κ‑coefficients of the small‑R⊥ expansion "
                    "Δρ = Σ κ_m R⊥^{5/3+m}.")
    parser.add_argument("N_terms", nargs="?", type=int, default=10,
                        help="how many κ_m to fit (default 3)")
    parser.add_argument("--no-plot", action="store_true",
                        help="suppress plots")
    parser.add_argument("--fft", dest="fft", action="store_true",
                        help="show FFT diagnostic")
    args = parser.parse_args()

    run(N_terms=args.N_terms,
        make_plot=not args.no_plot,
        fft_diagnose=args.fft)
