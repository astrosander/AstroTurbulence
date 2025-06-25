"""
stable_kappa.py
===============

Extracts κ₀, κ₁, … from
    Δρ(R⊥) = Σ κ_n (R⊥/L)^{5/3+n}
with numerically *robust* linear algebra.

Key knobs:
    --nterms N       (default 3)   how many κ_n to keep
    --mp     D       (no default)  turn on mpmath at D decimal digits
    --rcond  X       (default 1e-10) truncates tiny SVD singular values
    --noplot
Run "python stable_kappa.py -h" for details.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import contextlib

# Optional high‑precision backend
try:
    import mpmath as mp
except ImportError:
    mp = None

# --------- physical constants (C₂, L) ----------------
C2 = 1.0
L  = 1.0           # set L = 𝓛 for clarity

# analytic κ₀
from math import sqrt
from scipy.special import gamma
kappa0_true = sqrt(np.pi) * gamma(-5/6) / (2*gamma(-1/3))   # ≃ 1.45719


# --------- high‑accuracy Δρ(R) ------------------------
def delta_rho_np(R: float, N=8000) -> float:
    """ Δρ using double precision numpy trapezoid """
    if R == 0: return 0.0
    y = np.linspace(0.0, 1.0, N)            # |Δs|/L
    core = ((R**2 + (y*L)**2)**(1/3) - (y*L)**(2/3)) * (1 - y)
    integral = 2*L * np.trapz(core, y)
    return (C2 / (2*L**(2/3)*L)) * integral


def delta_rho_mp(R: float, N=8000) -> float:
    """Same integrand but in arbitrary precision (mpmath)."""
    y = mp.linspace(0, 1, N)
    core = [( (R**2 + (yy*L)**2)**(mp.mpf(1)/3) - (yy*L)**(mp.mpf(2)/3) )
             * (1-yy) for yy in y]
    integral = 2*L * mp.quad(lambda i: core[int(i)], [0, N-1]) / (N-1)
    return (C2 / (2*L**(mp.mpf(2)/3)*L)) * integral


# --------- κ‑fit with whitening & trunc. SVD ----------
# --------- κ‑fit with whitening & truncated SVD ----------
def fit_kappa(R, dR, n_terms, rcond=1e-10):
    """
    Least‑squares fit for κ_n  (n = 0 … n_terms‑1).

    Returns
        κ        – 1‑D array of fitted coefficients
        residual – dR − model
        exps     – the exponents 5/3, 8/3, …
        rank     – numerical rank of the design matrix
    """
    exps = (5 + 3*np.arange(n_terms)) / 3.0          # 5/3, 8/3, …
    Xraw = (R[:, None] / L)**exps[None, :]

    # -------- whitening (unit‑norm columns) --------------
    col_norm = np.linalg.norm(Xraw, axis=0)
    X = Xraw / col_norm

    # -------- stable least‑squares via SVD ---------------
    κ_scaled, *_ , rank, _ = np.linalg.lstsq(X, dR, rcond=rcond)
    κ = κ_scaled / col_norm                             # un‑whiten

    residual = dR - Xraw @ κ
    return κ, residual, exps, rank



# --------- CLI driver --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nterms", "-n", type=int, default=10)
    ap.add_argument("--mp", type=int, help="decimal digits for mpmath back‑end")
    ap.add_argument("--rmin", type=float, default=2e-4)
    ap.add_argument("--rmax", type=float, default=3e-3)
    ap.add_argument("--nR",   type=int,   default=40)
    ap.add_argument("--noplot", action="store_true")
    ap.add_argument("--rcond", type=float, default=1e-10)
    args = ap.parse_args()

    # choose precision & Δρ evaluator
    if args.mp:
        if mp is None:
            print("mpmath not installed; aborting.")
            sys.exit(1)
        mp.mp.dps = args.mp
        delta_rho = delta_rho_mp
        ftype = mp.mpf
    else:
        delta_rho = delta_rho_np
        ftype = float

    # sample radii on logarithmic grid
    R = np.logspace(np.log10(args.rmin*L), np.log10(args.rmax*L), args.nR, dtype=float)
    dR = np.array([delta_rho(ftype(r)) for r in R], dtype=float)

    κ, res, exps, rank = fit_kappa(R, dR, args.nterms, rcond=args.rcond)

    # --------- report ---------
    print("\nκ‑coefficients (rank {:d} fit, rcond={:.1e})".format(rank, args.rcond))
    print("m  exponent   κ_m           sign      |remainder|/Δρ after subtracting term")
    print("-- --------- ------------- -----      -------------------------------------")
    remainder = dR.copy()
    for m,(e,km) in enumerate(zip(exps, κ)):
        remainder -= km*(R/L)**e
        sign = "+" if km>=0 else "–"
        print(f"{m:<1d}  {e:7.3f}  {km: .6e}   {sign}         "
              f"{np.max(np.abs(remainder/dR)):.2e}")
    print("analytic κ₀ = {:.6e}  (error {:5.2f} %)"
          .format(kappa0_true,100*abs(κ[0]-kappa0_true)/kappa0_true))

    # --------- plots ----------
    if not args.noplot:
        import matplotlib.ticker as mt
        fig,ax = plt.subplots(figsize=(5,4))
        ax.loglog(R/L, dR, 'k', lw=2, label='numerical Δρ')
        series = sum(k*(R/L)**e for k,e in zip(κ, exps))
        ax.loglog(R/L, series, 'r--', label=f'series ({rank} terms)')
        ax.set_xlabel(r'$R_\perp/L$')
        ax.set_ylabel(r'$\Delta\rho$')
        ax.legend()
        ax.set_title('Small‑$R$ expansion with stable κ fit')
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
