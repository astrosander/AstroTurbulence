import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import j0


def radial_integrand(R: float, k: float) -> float:
    return (R ** (-2.0 / 3.0)) * np.exp(-(R ** (5.0 / 3.0))) * j0(k * R) * R


def compute_integral(k: float) -> float:
    val, err = quad(
        radial_integrand,
        0.0,
        np.inf,
        args=(k,),
        limit=200
    )
    return val


def main():
    k_values = np.logspace(np.log10(0.1), np.log10(100.0), 1200)

    integral_values = np.array([compute_integral(k) for k in k_values])

    plt.figure(figsize=(7, 4.5))
    plt.loglog(k_values, np.abs(integral_values))
    
    ref_line = k_values[0] ** (-11.0 / 3.0) * (k_values / k_values[0]) ** (-11.0 / 3.0)
    ref_line = ref_line * np.abs(integral_values[0]) / ref_line[0]* 10**6
    
    mask = k_values > 2
    plt.loglog(k_values[mask], ref_line[mask], '-.', color="black", lw=1, label='k^(-11/3)', alpha=0.7)
    
    plt.xlabel("k")
    plt.ylabel("|Integral|")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
