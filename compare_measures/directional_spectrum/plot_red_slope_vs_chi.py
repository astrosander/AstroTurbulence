import numpy as np
import matplotlib.pyplot as plt
from directional_spectrum_single_lambda import (
    load_fields, polarized_emissivity_simple, faraday_density,
    separated_P_map, ring_average, plot_fit,
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
    
    chi = 2*lam**2*sigma_RM
    if chi < 0.3:
        return None
    
    alphas = []
    slopes = []
    
    for i in np.linspace(0.01, 1.00, 1000):
        result = plot_fit(None, kc, Pdir, 0, i, "red", isPlot=False)
        alphas.append(i)
        if result is not None and isinstance(result, tuple) and len(result) == 2:
            intercept, slope = result
            slopes.append(slope)
        else:
            slopes.append(None)
    
    best_coarse = None
    min_abs_slope = float('inf')
    for i, slope in reversed(list(zip(alphas, slopes))):
        if slope is not None:
            abs_slope = abs(slope)
            if abs_slope <= min_abs_slope:
                min_abs_slope = abs_slope
                best_coarse = i
    
    best = best_coarse
    if best_coarse is not None:
        coarse_spacing = (alphas[-1] - alphas[0]) / (len(alphas) - 1) if len(alphas) > 1 else 0.01
        search_range = max(0.01, 2 * coarse_spacing)
        i_min = max(0.01, best_coarse - search_range)
        i_max = min(1.00, best_coarse + search_range)
        
        fine_alphas = np.linspace(i_min, i_max, 1000)
        min_abs_slope_fine = float('inf')
        for i_fine in reversed(fine_alphas):
            result = plot_fit(None, kc, Pdir, 0, i_fine, "red", isPlot=False)
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                intercept_fine, slope_fine = result
                abs_slope_fine = abs(slope_fine)
                if abs_slope_fine <= min_abs_slope_fine:
                    min_abs_slope_fine = abs_slope_fine
                    best = i_fine
    
    if chi < 4:
        best = 0.0
    
    xv = np.linspace(0.05, 0.95, 200)
    k0, k1 = kc.min(), kc.max()
    yl, yr = [], []
    for x in xv:
        a = plot_fit(None, kc, Pdir, 0, x, "r", isPlot=False)
        b = plot_fit(None, kc, Pdir, x, 1.0, "b", isPlot=False)
        if a and b:
            kx = k0 + x*(k1-k0)
            yl.append(np.exp(a[0]) * kx**a[1])
            yr.append(np.exp(b[0]) * kx**b[1])
        else:
            yl.append(np.nan)
            yr.append(np.nan)
    
    xv = np.asarray(xv)
    yl = np.asarray(yl)
    yr = np.asarray(yr)
    m = np.isfinite(yl)&np.isfinite(yr)&(yl>0)&(yr>0)
    xv, yl, yr = xv[m], yl[m], yr[m]
    d = np.log(yl) - np.log(yr)
    idx = np.where(d[:-1]*d[1:]<=0)[0]
    xs, ys = [], []
    for i in idx:
        x0, x1 = xv[i], xv[i+1]
        d0, d1 = d[i], d[i+1]
        t = 0 if d1==d0 else -d0/(d1-d0)
        xs.append(x0 + (x1-x0)*t)
        ys.append(np.exp(np.interp(xs[-1], [x0,x1], [np.log(yl[i]), np.log(yl[i+1])])))
    j = np.argmin(np.abs(np.array(xs)-0.18))
    x1 = xs[j]
    
    result = plot_fit(None, kc, Pdir, best, x1, "#E74C3C", isPlot=False)
    if result is not None:
        return result[1]
    return None

sigma_RM = compute_sigma_RM()
chi_values = np.linspace(0.3, 20.0, 100)
slopes = []

for chi in chi_values:
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

