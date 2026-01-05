import numpy as np
import matplotlib.pyplot as plt

A_P  = 1.0
chi  = 0.05#4
R0   = 1.0
r_phi = 1.0
m_psi = 4/3
m_phi = 2/3
k_trans = 10.0

def fpsi(R):
    x = (R / R0) ** m_psi
    return x / (1.0 + x)

def fphi(R):
    y = (R / r_phi) ** m_phi
    return y / (1.0 + y)

def Du_over2(R):
    return A_P * (1.0 - (1.0 - fpsi(R)) * np.exp(-(chi**2) * fphi(R)))

def Du_over2_small_2term(R):
    return A_P * ((R / R0) ** m_psi + (chi**2) * (R / r_phi) ** m_phi)

def Rx_u():
    if abs(m_psi - m_phi) < 1e-14:
        return None
    return ((chi**2) * (R0**m_psi) / (r_phi**m_phi)) ** (1.0 / (m_psi - m_phi))

def effective_slope(R, y):
    logR = np.log(R)
    logy = np.log(np.maximum(y, 1e-300))
    return np.gradient(logy, logR)

def neff_2term(R, Rx):
    t = (R / Rx) ** (m_psi - m_phi)
    return (m_phi + m_psi * t) / (1.0 + t)

R = np.logspace(-6, 3, 2000)
y_exact = Du_over2(R)

Rx = Rx_u()

dm = abs(m_psi - m_phi)
R_left = R_right = None
if Rx is not None and np.isfinite(Rx) and dm > 0:
    R_left  = Rx * (k_trans ** (-1.0 / dm))
    R_right = Rx * (k_trans ** ( 1.0 / dm))

mask = y_exact > 0
n_exact = np.full_like(R, np.nan, dtype=float)
n_exact[mask] = effective_slope(R[mask], y_exact[mask])

n_2term = None
if Rx is not None and np.isfinite(Rx):
    n_2term = neff_2term(R, Rx)

plt.figure(figsize=(10, 5))
plt.semilogx(R, n_exact, color="black", lw=3, label=r"$d\ln(D_u/2)/d\ln R$")
if n_2term is not None:
    plt.semilogx(R, n_2term, "--", color="red", lw=2, alpha=1, label="$(R / R_0)^{m_\psi} + \chi^2 (R / r_\phi)^{m_\phi}$")
plt.axhline(m_psi, linestyle="--", linewidth=2, color="orange", label=r"$m_\psi$")
plt.axhline(m_phi, linestyle="--", linewidth=2, color="blue", label=r"$m_\Phi$")

if Rx is not None and np.isfinite(Rx) and (R.min() < Rx < R.max()):
    plt.axvline(Rx, linestyle="-.", linewidth=2, color="gray")
    plt.axhline(0.5 * (m_psi + m_phi), linestyle=":", linewidth=1,
                label=r"$(m_\psi+m_\Phi)/2$")

if R_left is not None and R_right is not None:
    if R.min() < R_left < R.max():
        plt.axvline(R_left, linestyle="--", color="gray", linewidth=1)
        y0, y1 = plt.ylim()
        plt.text(R_left, y1 *0.75, r"$R_\times k^{-\frac{1}{\Delta \rm{m}}}$", rotation=90,
                 va="bottom", ha="right", color="gray")
    if R.min() < R_right < R.max():
        plt.axvline(R_right, linestyle="--", color="gray", linewidth=1)
        y0, y1 = plt.ylim()
        plt.text(R_right, y1 *0.75, r"$R_\times k^{\frac{1}{\Delta \rm{m}}}$", rotation=90,
                 va="bottom", ha="right", color="gray")

plt.xlabel("R")
plt.ylabel("log-log slope")
plt.legend(fontsize=9, loc="best")
plt.show()

print("Parameters:")
print(f"A_P={A_P}, chi={chi}, R0={R0}, r_phi={r_phi}, m_psi={m_psi}, m_phi={m_phi}")
print(f"Analytic crossover R_x,u = {Rx}")
