import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import j0

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    # "savefig.transparent": True,
})


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
    plt.loglog(k_values, np.abs(integral_values),color="blue",lw=2)
    
    slope=-4/3
    ref_line = k_values[0] ** (slope) * (k_values / k_values[0]) ** (slope)
    ref_line = ref_line * np.abs(integral_values[0]) / ref_line[0]* 20**1
    
    mask = k_values > 2
    plt.loglog(k_values[mask], ref_line[mask], '-.', color="black", lw=1.5, label='$k^{-4/3}$', alpha=0.7)
    
    mask = (k_values < 10) & (k_values > 1.5)
    slope2=-5/3
    ref_line2 = k_values[0] ** (slope2) * (k_values / k_values[0]) ** (slope2)
    ref_line2 = ref_line2 * np.abs(integral_values[0]) / ref_line2[0]* 80**1
    
    plt.loglog(k_values[mask], ref_line2[mask], '--', color="red", lw=1.5, label='$k^{-5/3}$', alpha=0.7)
    
    plt.xlabel("$k$")
    plt.ylabel(r"integral")
    plt.title(r"$\int_{0}^{\infty} R^{-2/3}\, e^{-R^{5/3}}\, J_0(kR)\, R \, dR$")
    plt.legend()
    # plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("henkel.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
