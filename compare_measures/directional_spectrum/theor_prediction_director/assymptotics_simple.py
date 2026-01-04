import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------------
# Initial parameter values
# --------------------------
init_A_P = 1.0
init_chi = 0.4
init_R0 = 1.0
init_r_phi = 3.0
init_m_psi = 3.0
init_m_phi = 2.0

# --------------------------
# R grid (log-spaced)
# --------------------------
R = np.logspace(-4, 1, 1500)  # wide range to show asymptotics

# --------------------------
# Definitions (functions that take parameters as arguments)
# --------------------------
def fpsi(R, R0, m_psi):
    x = (R / R0) ** m_psi
    return x / (1.0 + x)

def fphi(R, r_phi, m_phi):
    y = (R / r_phi) ** m_phi
    return y / (1.0 + y)

def F(R, A_P, chi, R0, r_phi, m_psi, m_phi):
    return A_P * (1.0 - (1.0 - fpsi(R, R0, m_psi)) * np.exp(-(chi**2) * fphi(R, r_phi, m_phi)))

# Small-R asymptotic pieces
def F_small_piece_psi(R, A_P, R0, m_psi):
    return A_P * (R / R0) ** m_psi

def F_small_piece_phi(R, A_P, chi, r_phi, m_phi):
    return A_P * (chi**2) * (R / r_phi) ** m_phi

# Large-R asymptotic (plateau deficit)
def deficit(R, A_P, chi, R0, r_phi, m_psi, m_phi):
    return A_P - F(R, A_P, chi, R0, r_phi, m_psi, m_phi)

def deficit_large(R, A_P, chi, R0, m_psi):
    return A_P * np.exp(-(chi**2)) * (R0 / R) ** m_psi

# --------------------------
# Create figure and axis
# --------------------------
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.35)  # Make room for sliders

# Initial plot
F_vals = F(R, init_A_P, init_chi, init_R0, init_r_phi, init_m_psi, init_m_phi)
F_psi = F_small_piece_psi(R, init_A_P, init_R0, init_m_psi)
F_phi = F_small_piece_phi(R, init_A_P, init_chi, init_r_phi, init_m_phi)

line_F, = ax.loglog(R, F_vals, label="F(R) exact")
line_psi, = ax.loglog(R, F_psi, "--", label=r"small-R: $A_P (R/R_0)^{m_\psi}$")
line_phi, = ax.loglog(R, F_phi, "--", label=r"small-R: $A_P \chi^2 (R/r_\phi)^{m_\Phi}$")

# Calculate and plot R_x (will be updated in update function)
rx_artists = []  # Store line and text for R_x

def update_rx_display(A_P, chi, R0, r_phi, m_psi, m_phi):
    """Update the R_x vertical line and label"""
    # Remove old artists
    for artist in rx_artists:
        artist.remove()
    rx_artists.clear()
    
    if abs(m_psi - m_phi) > 1e-12:
        R_x = ((chi**2) * (R0**m_psi) / (r_phi**m_phi)) ** (1.0 / (m_psi - m_phi))
        if np.isfinite(R_x) and R.min() < R_x < R.max():
            vline = ax.axvline(R_x, linestyle=":", linewidth=1, color='gray')
            y0, y1 = ax.get_ylim()
            text_Rx = ax.text(R_x, y0 * 2, r"$R_\times$", rotation=90, va="bottom", ha="right")
            rx_artists.extend([vline, text_Rx])

# Initial R_x display
update_rx_display(init_A_P, init_chi, init_R0, init_r_phi, init_m_psi, init_m_phi)

ax.set_xlabel("R")
ax.set_ylabel("F(R)")
ax.set_title("Small-R asymptotics (log-log)")
ax.legend()
# ax.grid(True, which="both", linestyle=":", alpha=0.3)

# --------------------------
# Create sliders
# --------------------------
ax_A_P = plt.axes([0.15, 0.25, 0.3, 0.03])
ax_chi = plt.axes([0.15, 0.21, 0.3, 0.03])
ax_R0 = plt.axes([0.15, 0.17, 0.3, 0.03])
ax_r_phi = plt.axes([0.15, 0.13, 0.3, 0.03])
ax_m_psi = plt.axes([0.15, 0.09, 0.3, 0.03])
ax_m_phi = plt.axes([0.15, 0.05, 0.3, 0.03])

slider_A_P = Slider(ax_A_P, r'$A_P$', 0.1, 5.0, valinit=init_A_P, valstep=0.1)
slider_chi = Slider(ax_chi, r'$\chi$', 0.1, 5.0, valinit=init_chi, valstep=0.1)
slider_R0 = Slider(ax_R0, r'$R_0$', 0.1, 5.0, valinit=init_R0, valstep=0.1)
slider_r_phi = Slider(ax_r_phi, r'$r_\phi$', 0.1, 5.0, valinit=init_r_phi, valstep=0.1)
slider_m_psi = Slider(ax_m_psi, r'$m_\psi$', 0.5, 10.0, valinit=init_m_psi, valstep=0.1)
slider_m_phi = Slider(ax_m_phi, r'$m_\Phi$', 0.5, 10.0, valinit=init_m_phi, valstep=0.1)

# --------------------------
# Update function
# --------------------------
def update(val):
    A_P = slider_A_P.val
    chi = slider_chi.val
    R0 = slider_R0.val
    r_phi = slider_r_phi.val
    m_psi = slider_m_psi.val
    m_phi = slider_m_phi.val
    
    # Recalculate functions
    F_vals = F(R, A_P, chi, R0, r_phi, m_psi, m_phi)
    F_psi = F_small_piece_psi(R, A_P, R0, m_psi)
    F_phi = F_small_piece_phi(R, A_P, chi, r_phi, m_phi)
    
    # Update plot lines
    line_F.set_ydata(F_vals)
    line_psi.set_ydata(F_psi)
    line_phi.set_ydata(F_phi)
    
    # Update R_x line if needed
    update_rx_display(A_P, chi, R0, r_phi, m_psi, m_phi)
    
    # Rescale axes
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_A_P.on_changed(update)
slider_chi.on_changed(update)
slider_R0.on_changed(update)
slider_r_phi.on_changed(update)
slider_m_psi.on_changed(update)
slider_m_phi.on_changed(update)

plt.show()
