import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import sys

# Import functions from rph_ri
sys.path.insert(0, str(Path(__file__).parent))
from rph_ri import (
    load_fields, separated_P_map, ring_average, 
    get_field_components_for_los, auto_select_los_axis,
    polarized_emissivity_simple, faraday_density,
    radial_corr_length_unbiased, fit_segment,
    h5_path, C, gamma, los_axis, auto_los, los_perpendicular
)

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# Fixed chi value
CHI_TARGET = 1.0

# Initial values
INIT_EMIT_START = 0.7
INIT_EMIT_END = 0.75
INIT_SCREEN_START = 0.00
INIT_SCREEN_END = 0.05

# Load data once
print("Loading data...")
Bx, By, Bz, ne = load_fields(h5_path)

# Determine LOS axis
use_los_axis = los_axis
if 'auto_los' in globals() and auto_los:
    perp = los_perpendicular if 'los_perpendicular' in globals() else True
    use_los_axis = auto_select_los_axis(Bx, By, Bz, perpendicular=perp)
    print(f"Auto-selected LOS axis: {use_los_axis}")

B_perp1, B_perp2, B_parallel = get_field_components_for_los(Bx, By, Bz, use_los_axis)
Pi = polarized_emissivity_simple(B_perp1, B_perp2, gamma)

# Compute sigma_RM for initial screen_frac to get lambda
phi_init = faraday_density(ne, B_parallel, C)
_, sigma_RM_init, _, _ = separated_P_map(
    Pi, phi_init, 0.0, use_los_axis, 
    (INIT_EMIT_START, INIT_EMIT_END), 
    (INIT_SCREEN_START, INIT_SCREEN_END)
)
lam_init = np.sqrt(CHI_TARGET / (2.0 * sigma_RM_init)) if sigma_RM_init > 0 else 0.0

print(f"Initial sigma_RM = {sigma_RM_init:.6f}")
print(f"Initial lambda = {lam_init:.6f}")

# Create figure and axis
fig = plt.figure(figsize=(14, 9))
ax = plt.subplot(2, 1, 1)
plt.subplots_adjust(bottom=0.25)

# Initial plot
ring_bins = 96
P, sigma_RM, P_emit_map, Phi_map = separated_P_map(
    Pi, phi_init, lam_init, use_los_axis,
    (INIT_EMIT_START, INIT_EMIT_END),
    (INIT_SCREEN_START, INIT_SCREEN_END)
)
Q, U = P.real, P.imag
chi_angle = 0.5*np.arctan2(U,Q)
c2, s2 = np.cos(2*chi_angle), np.sin(2*chi_angle)
kc, Pk, _, _, _, _ = ring_average(c2, ring_bins, 3.0, None, True, False)
_, Pk2, _, _, _, _ = ring_average(s2, ring_bins, 3.0, None, True, False)
Pdir = Pk + Pk2

# Compute r_i and r_phi
r_i, _, _ = radial_corr_length_unbiased(P_emit_map, bins=256, method="efold")
r_phi, _, _ = radial_corr_length_unbiased(Phi_map, bins=256, method="efold")
Ny, Nx = P_emit_map.shape
Kphi_idx = (1.0/r_phi)*Nx if (np.isfinite(r_phi) and r_phi>0) else None
Ki_idx = (1.0/r_i)*Nx if (np.isfinite(r_i) and r_i>0) else None

# Plot data
line_data, = ax.loglog(kc, Pdir, 'o-', color='#2C3E50', lw=2.5, alpha=0.8, label='Data', ms=5)
line_kphi = ax.axvline(Kphi_idx if Kphi_idx else 0, color="#9B59B6", lw=2.0, ls="--", alpha=0.9,
                       label=fr"$K_\phi=1/r_\phi$ = {Kphi_idx:.3f}" if Kphi_idx else "")
line_ki = ax.axvline(Ki_idx if Ki_idx else 0, color="#16A085", lw=2.0, ls="--", alpha=0.9,
                     label=fr"$K_i=1/r_i$ = {Ki_idx:.3f}" if Ki_idx else "")

# Track fit lines
fit_lines = []

kmin, kmax = kc.min(), kc.max()

# Fit slopes
sP = None
sM = None
sH = None

# 1. k < K_i (low range)
if Ki_idx is not None:
    k_i_upper = min(max(Ki_idx, kmin), kmax)
    if k_i_upper > kmin and Ki_idx > kmin:
        sP = fit_segment(ax, kc, Pdir, kmin, k_i_upper, "#7F8C8D", "$k<K_i$")
        if sP is not None:
            fit_lines.append(ax.lines[-1])

# 2. K_i < k < K_phi (mid range)
if (Ki_idx is not None) and (Kphi_idx is not None) and (Kphi_idx > Ki_idx):
    k_i_lower = max(Ki_idx, kmin)
    k_phi_upper = min(Kphi_idx, kmax)
    if k_phi_upper > k_i_lower:
        sM = fit_segment(ax, kc, Pdir, k_i_lower, k_phi_upper, "#E67E22", "$K_i<k<K_\phi$")
        if sM is not None:
            fit_lines.append(ax.lines[-1])

# 3. k > K_phi (high range)
if Kphi_idx is not None:
    k_phi_lower = max(Kphi_idx, kmin)
    if k_phi_lower < kmax:
        sH = fit_segment(ax, kc, Pdir, k_phi_lower, kmax, "#E74C3C", "$k>K_\phi$")
        if sH is not None:
            fit_lines.append(ax.lines[-1])

ax.set_xlabel("$k$", fontsize=18)
ax.set_ylabel("$P_{dir}(k)$", fontsize=18)
ax.set_title(rf"$\chi = {CHI_TARGET:.1f}$, $r_\phi={r_phi:.2f}$, $r_i={r_i:.2f}$", fontsize=20, fontweight='bold', pad=10)
ax.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.8)
ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)

# Create sliders
ax_emit_start = plt.axes([0.15, 0.15, 0.3, 0.03])
ax_emit_end = plt.axes([0.15, 0.10, 0.3, 0.03])
ax_screen_start = plt.axes([0.55, 0.15, 0.3, 0.03])
ax_screen_end = plt.axes([0.55, 0.10, 0.3, 0.03])

slider_emit_start = Slider(ax_emit_start, 'emit_start', 0.0, 1.0, valinit=INIT_EMIT_START, valstep=0.01)
slider_emit_end = Slider(ax_emit_end, 'emit_end', 0.0, 1.0, valinit=INIT_EMIT_END, valstep=0.01)
slider_screen_start = Slider(ax_screen_start, 'screen_start', 0.0, 1.0, valinit=INIT_SCREEN_START, valstep=0.01)
slider_screen_end = Slider(ax_screen_end, 'screen_end', 0.0, 1.0, valinit=INIT_SCREEN_END, valstep=0.01)

def update(val):
    """Update plot when sliders change."""
    emit_start = slider_emit_start.val
    emit_end = slider_emit_end.val
    screen_start = slider_screen_start.val
    screen_end = slider_screen_end.val
    
    # Ensure emit_end > emit_start and screen_end > screen_start
    if emit_end <= emit_start:
        emit_end = emit_start + 0.01
        slider_emit_end.set_val(emit_end)
    if screen_end <= screen_start:
        screen_end = screen_start + 0.01
        slider_screen_end.set_val(screen_end)
    
    emit_frac = (emit_start, emit_end)
    screen_frac = (screen_start, screen_end)
    
    # Recompute with new fractions
    phi = faraday_density(ne, B_parallel, C)
    P, sigma_RM, P_emit_map, Phi_map = separated_P_map(
        Pi, phi, 0.0, use_los_axis, emit_frac, screen_frac
    )
    
    # Calculate lambda to maintain chi = 1.0
    lam = np.sqrt(CHI_TARGET / (2.0 * sigma_RM)) if sigma_RM > 0 else 0.0
    
    # Recompute with correct lambda
    P, sigma_RM, P_emit_map, Phi_map = separated_P_map(
        Pi, phi, lam, use_los_axis, emit_frac, screen_frac
    )
    
    Q, U = P.real, P.imag
    chi_angle = 0.5*np.arctan2(U,Q)
    c2, s2 = np.cos(2*chi_angle), np.sin(2*chi_angle)
    kc, Pk, _, _, _, _ = ring_average(c2, ring_bins, 3.0, None, True, False)
    _, Pk2, _, _, _, _ = ring_average(s2, ring_bins, 3.0, None, True, False)
    Pdir = Pk + Pk2
    
    # Update data line
    line_data.set_data(kc, Pdir)
    
    # Compute r_i and r_phi
    r_i, _, _ = radial_corr_length_unbiased(P_emit_map, bins=256, method="efold")
    r_phi, _, _ = radial_corr_length_unbiased(Phi_map, bins=256, method="efold")
    Ny, Nx = P_emit_map.shape
    Kphi_idx = (1.0/r_phi)*Nx if (np.isfinite(r_phi) and r_phi>0) else None
    Ki_idx = (1.0/r_i)*Nx if (np.isfinite(r_i) and r_i>0) else None
    
    # Update vertical lines
    if Kphi_idx is not None:
        line_kphi.set_xdata([Kphi_idx, Kphi_idx])
        line_kphi.set_label(fr"$K_\phi=1/r_\phi$ = {Kphi_idx:.3f}")
    else:
        line_kphi.set_xdata([0, 0])
        line_kphi.set_label("")
    
    if Ki_idx is not None:
        line_ki.set_xdata([Ki_idx, Ki_idx])
        line_ki.set_label(fr"$K_i=1/r_i$ = {Ki_idx:.3f}")
    else:
        line_ki.set_xdata([0, 0])
        line_ki.set_label("")
    
    # Remove old fit lines
    for line in fit_lines:
        if line in ax.lines:
            line.remove()
    fit_lines.clear()
    
    # Update fit lines
    kmin, kmax = kc.min(), kc.max()
    
    # 1. k < K_i (low range)
    if Ki_idx is not None:
        k_i_upper = min(max(Ki_idx, kmin), kmax)
        if k_i_upper > kmin and Ki_idx > kmin:
            sP = fit_segment(ax, kc, Pdir, kmin, k_i_upper, "#7F8C8D", "$k<K_i$")
            if sP is not None:
                fit_lines.append(ax.lines[-1])
    
    # 2. K_i < k < K_phi (mid range)
    if (Ki_idx is not None) and (Kphi_idx is not None) and (Kphi_idx > Ki_idx):
        k_i_lower = max(Ki_idx, kmin)
        k_phi_upper = min(Kphi_idx, kmax)
        if k_phi_upper > k_i_lower:
            sM = fit_segment(ax, kc, Pdir, k_i_lower, k_phi_upper, "#E67E22", "$K_i<k<K_\phi$")
            if sM is not None:
                fit_lines.append(ax.lines[-1])
    
    # 3. k > K_phi (high range)
    if Kphi_idx is not None:
        k_phi_lower = max(Kphi_idx, kmin)
        if k_phi_lower < kmax:
            sH = fit_segment(ax, kc, Pdir, k_phi_lower, kmax, "#E74C3C", "$k>K_\phi$")
            if sH is not None:
                fit_lines.append(ax.lines[-1])
    
    # Update title
    ax.set_title(rf"$\chi = {CHI_TARGET:.1f}$, $r_\phi={r_phi:.2f}$, $r_i={r_i:.2f}$", fontsize=20, fontweight='bold', pad=10)
    
    # Update axis limits
    ax.relim()
    ax.autoscale()
    
    # Update legend
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_emit_start.on_changed(update)
slider_emit_end.on_changed(update)
slider_screen_start.on_changed(update)
slider_screen_end.on_changed(update)

plt.show()

