import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

init_A_P  = 1.0
init_chi  = 0.05
init_R0   = 1.0
init_r_phi = 1.0
init_m_psi = 4/3
init_m_phi = 2/3
k_trans = 10.0

def fpsi(R, R0, m_psi):
    x = (R / R0) ** m_psi
    return x / (1.0 + x)

def fphi(R, r_phi, m_phi):
    y = (R / r_phi) ** m_phi
    return y / (1.0 + y)

def Du_over2(R, A_P, chi, R0, r_phi, m_psi, m_phi):
    return A_P * (1.0 - (1.0 - fpsi(R, R0, m_psi)) * np.exp(-(chi**2) * fphi(R, r_phi, m_phi)))

def Du_over2_small_2term(R, A_P, chi, R0, r_phi, m_psi, m_phi):
    return A_P * ((R / R0) ** m_psi + (chi**2) * (R / r_phi) ** m_phi)

def Rx_u(chi, R0, r_phi, m_psi, m_phi):
    if abs(m_psi - m_phi) < 1e-14:
        return None
    return ((chi**2) * (R0**m_psi) / (r_phi**m_phi)) ** (1.0 / (m_psi - m_phi))

def effective_slope(R, y):
    logR = np.log(R)
    logy = np.log(np.maximum(y, 1e-300))
    return np.gradient(logy, logR)

def neff_2term(R, Rx, m_psi, m_phi):
    t = (R / Rx) ** (m_psi - m_phi)
    return (m_phi + m_psi * t) / (1.0 + t)

R = np.logspace(-6, 3, 2000)

def compute_all_curves(A_P, chi, R0, r_phi, m_psi, m_phi):
    y_exact = Du_over2(R, A_P, chi, R0, r_phi, m_psi, m_phi)
    Rx = Rx_u(chi, R0, r_phi, m_psi, m_phi)
    
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
        n_2term = neff_2term(R, Rx, m_psi, m_phi)
    
    return n_exact, n_2term, Rx, R_left, R_right, m_psi, m_phi

fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.20, top=0.98, left=0.08, right=0.98)

def update_plot(chi, r_phi, m_psi, m_phi):
    n_exact, n_2term, Rx, R_left, R_right, m_psi, m_phi = compute_all_curves(
        init_A_P, chi, init_R0, r_phi, m_psi, m_phi
    )
    
    ax.clear()
    
    ax.semilogx(R, n_exact, color="black", lw=3.5, label=r"$d\ln(D_u/2)/d\ln R$")
    if n_2term is not None:
        ax.semilogx(R, n_2term, "--", color="red", lw=2, alpha=1, label="$(R / R_0)^{m_\psi} + \chi^2 (R / r_\phi)^{m_\phi}$")
    ax.axhline(m_psi, linestyle="--", linewidth=2, color="orange", label=r"$m_\psi$")
    ax.axhline(m_phi, linestyle="--", linewidth=2, color="blue", label=r"$m_\Phi$")
    
    if Rx is not None and np.isfinite(Rx) and (R.min() < Rx < R.max()):
        ax.axvline(Rx, linestyle="-.", linewidth=2, color="gray")
        ax.axhline(0.5 * (m_psi + m_phi), linestyle=":", linewidth=1,
                    label=r"$(m_\psi+m_\Phi)/2$")
    
    if R_left is not None and R_right is not None:
        if R.min() < R_left < R.max():
            ax.axvline(R_left, linestyle="--", color="gray", linewidth=1)
            y0, y1 = ax.get_ylim()
            ax.text(R_left, y1 * 0.75, r"$R_\times k^{-\frac{1}{\Delta \rm{m}}}$", rotation=90,
                     va="bottom", ha="right", color="gray")
        if R.min() < R_right < R.max():
            ax.axvline(R_right, linestyle="--", color="gray", linewidth=1)
            y0, y1 = ax.get_ylim()
            ax.text(R_right, y1 * 0.75, r"$R_\times k^{\frac{1}{\Delta \rm{m}}}$", rotation=90,
                     va="bottom", ha="right", color="gray")
    
    ax.set_xlabel("R")
    ax.set_ylabel("log-log slope")
    ax.legend(fontsize=9, loc="best")
    
    fig.canvas.draw_idle()

initial_n_exact, initial_n_2term, initial_Rx, initial_R_left, initial_R_right, initial_m_psi, initial_m_phi = compute_all_curves(
    init_A_P, init_chi, init_R0, init_r_phi, init_m_psi, init_m_phi
)

update_plot(init_chi, init_r_phi, init_m_psi, init_m_phi)

ax_chi = plt.axes([0.15, 0.14, 0.3, 0.02])
ax_r_phi = plt.axes([0.15, 0.11, 0.3, 0.02])
ax_m_psi = plt.axes([0.15, 0.08, 0.3, 0.02])
ax_m_phi = plt.axes([0.15, 0.05, 0.3, 0.02])

slider_chi = Slider(ax_chi, r'$\chi$', 0.01, 2.0, valinit=init_chi, valstep=0.01, color="black")
slider_r_phi = Slider(ax_r_phi, r'$r_\phi$', 0.1, 10.0, valinit=init_r_phi, valstep=0.1, color="black")
slider_m_psi = Slider(ax_m_psi, r'INTRISTIC: $m_\psi$', 0.1, 5.0, valinit=init_m_psi, valstep=0.1, color="orange")
slider_m_phi = Slider(ax_m_phi, r'FARADAY: $m_\Phi$', 0.1, 5.0, valinit=init_m_phi, valstep=0.1, color="blue")

def update(val):
    chi = slider_chi.val
    r_phi = slider_r_phi.val
    m_psi = slider_m_psi.val
    m_phi = slider_m_phi.val
    update_plot(chi, r_phi, m_psi, m_phi)

slider_chi.on_changed(update)
slider_r_phi.on_changed(update)
slider_m_psi.on_changed(update)
slider_m_phi.on_changed(update)

plt.show()

print("=== Initial Parameters ===")
print(f"A_P={init_A_P}, chi={init_chi}, R0={init_R0}, r_phi={init_r_phi}, m_psi={init_m_psi}, m_phi={init_m_phi}")
print(f"Analytic crossover R_x,u = {initial_Rx}")
