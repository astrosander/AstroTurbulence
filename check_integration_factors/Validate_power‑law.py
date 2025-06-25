import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

C2 = 1.0          
L0 = 1.0          
L  = L0           

kappa = np.sqrt(np.pi)*gamma(-5/6)/(2*gamma(-1/3))   # ≈ 1.45719

# def delta_rho(Rperp, N=6000):
#     if Rperp == 0.0:
#         return 0.0

#     y = np.linspace(0.0, 1.0, N)
#     core = ((Rperp**2 + (y*L)**2)**(1./3.) - (y*L)**(2./3.)) * (1.0 - y)
#     integral = 2.0 * L * np.trapz(core, y)        

#     return (C2 / (2 * L0**(2./3.) * L)) * integral


def delta_rho(Rperp, N=600):
    # Parameters must be defined globally or passed in
    global L, C2, L0

    # Set up grid
    s = np.linspace(0, L, N)
    s_prime = s[:, None]  # Column vector for broadcasting
    delta_s = s - s_prime
    ds = L / (N - 1)

    # Define r^2 for both Rperp and for the baseline Rperp = 0
    r2 = Rperp**2 + delta_s**2
    r2_zero = delta_s**2  # Equivalent to Rperp = 0

    # Kolmogorov correlation functions
    rho = 1.0 - (C2 / L0**(2/3)) * r2**(1/3)
    rho0 = 1.0 - (C2 / L0**(2/3)) * r2_zero**(1/3)

    # Integrate both
    integral = np.trapz(np.trapz(rho, dx=ds, axis=0), dx=ds)
    integral0 = np.trapz(np.trapz(rho0, dx=ds, axis=0), dx=ds)

    # Normalize by L^2 and subtract
    bar_rho_perp = integral / L**2
    bar_rho_perp_0 = integral0 / L**2

    delta = bar_rho_perp - bar_rho_perp_0
    return delta


def delta_rho_theory(Rperp):
    return C2 * kappa * (Rperp / L0)**(5./3.)


def run(nR=30, Rmin=5e-4, Rmax=1, N_int=6000, make_plot=True):
    R = np.logspace(np.log10(Rmin), np.log10(Rmax), nR)
    dR = np.array([delta_rho(r, N=N_int) for r in R])

    slope = np.polyfit(np.log(R), np.log(dR), 1)[0]

    print(f"log–log fit of Δρ ∝ R⊥^m   gives  m = {slope:.4f}")
    print(f"Relative error: {100*abs(slope-5/3)/(5/3):.2f} %")

    if make_plot:
        plt.figure(figsize=(5,4))
        plt.loglog(R, dR,               label=r'$\Delta\rho_{\rm num}$')
        plt.loglog(R, delta_rho_theory(R),
                   linestyle=':', label='$5/3$ slope; eq. (5.26)')
        plt.xlabel(r'$R_\perp/L$')
        plt.ylabel(r'$\Delta\rho$')
        plt.title('Projected correlation: Δρ versus $R_\\perp$')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run()
