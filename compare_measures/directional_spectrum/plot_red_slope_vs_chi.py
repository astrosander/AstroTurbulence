import numpy as np
import matplotlib.pyplot as plt
from directional_spectrum_single_lambda import (
    load_fields, polarized_emissivity_simple, faraday_density,
    separated_P_map, ring_average, _two_segment_candidates, _select_break_stable, _ema, _whole_slope, BREAK_STATE,
    los_axis, emit_frac, screen_frac, C, h5_path, ring_bins
)

def compute_sigma_RM():
    Bx, By, Bz, ne = load_fields(h5_path)
    Pi = polarized_emissivity_simple(Bx, By, 2.0)
    Bpar = Bz
    phi = faraday_density(ne, Bpar, C)
    _, sigma_RM = separated_P_map(Pi, phi, 1.0, los_axis, emit_frac, screen_frac)
    return sigma_RM

def compute_red_slope(lam, sigma_RM):
    Bx, By, Bz, ne = load_fields(h5_path)
    Pi = polarized_emissivity_simple(Bx, By, 2.0)
    Bpar = Bz
    phi = faraday_density(ne, Bpar, C)
    
    P, _ = separated_P_map(Pi, phi, lam, los_axis, emit_frac, screen_frac)
    Q = P.real
    U = P.imag
    chi_angle = 0.5*np.arctan2(U,Q)
    
    c2 = np.cos(2*chi_angle)
    s2 = np.sin(2*chi_angle)
    kc, Pk, _, _, _, _ = ring_average(c2, ring_bins, 3.0, None, True, False)
    _, Pk2, _, _, _, _ = ring_average(s2, ring_bins, 3.0, None, True, False)
    Pdir = Pk + Pk2
    
    order_key = ("increasing", float(lam))
    if BREAK_STATE.get("last_order_key") is None:
        BREAK_STATE["last_order_key"] = order_key
    elif BREAK_STATE["last_order_key"][0] != "increasing":
        BREAK_STATE.update({"prev_frac": None, "prev_slope": None, "active": False, "prev_out": None, "last_order_key": order_key})

    cands, whole = _two_segment_candidates(kc, Pdir, min_pts=6, cont_eps=0.08)
    if whole is None:
        s_whole, _ = _whole_slope(kc, Pdir)
    else:
        s_whole = whole[0]
    if not cands:
        return _ema(s_whole, BREAK_STATE)

    sL, _ = _select_break_stable(cands, s_whole, state=BREAK_STATE, tau_on=6.0, tau_off=3.0, lam_frac=3.0, lam_s=1.2, monotonic=True)
    chi = 2*lam**2*sigma_RM
    if chi <= 0.3:
        BREAK_STATE.update(active=False, prev_frac=None, prev_slope=s_whole)
        return _ema(s_whole, BREAK_STATE)
    return _ema(sL, BREAK_STATE)

sigma_RM = compute_sigma_RM()
chi_values = np.linspace(0, 20, 50)
slopes = []

for chi in chi_values:
    if chi <= 0:
        lam = 0.0
    else:
        lam = np.sqrt(chi / (2.0 * sigma_RM))
    slope = compute_red_slope(lam, sigma_RM)
    slopes.append(slope)
    print(f"chi={chi:.2f}, slope={slope}")

chi_clean = []
slopes_clean = []
for chi, slope in zip(chi_values, slopes):
    if slope is not None:
        chi_clean.append(chi)
        slopes_clean.append(slope)

chi_values = np.array(chi_clean)
slopes = np.array(slopes_clean)

plt.figure(figsize=(10, 6))
plt.plot(chi_values, slopes, '-', color='#E74C3C', lw=3)
plt.xlabel('$\\chi$', fontsize=20)
plt.ylabel('Red Line Slope', fontsize=20)
plt.title('Red Line Slope vs $\\chi$', fontsize=22, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('red_slope_vs_chi.png', dpi=300, bbox_inches='tight')
plt.show()

