import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 24


# Load data from structure_functions_points_1400.npz
input_file = 'spectrum/B0x_11.npz'
print(f"Loading data from {input_file}...")
data = np.load(input_file)

# Extract structure function data
rP = data['rP']
DP = data['DP']
ru = data['ru']
Du = data['Du']

# Extract power spectrum data
kP = data['kP']
Pk_amp = data['Pk_amp']
ku = data['ku']
Pk_u = data['Pk_u']

# Extract parameters for title/labels
N = int(data['N'])
L = float(data['L'])

print(f"Loaded data:")
print(f"  P_i structure bins: {len(rP)}")
print(f"  u_i structure bins: {len(ru)}")
print(f"  |P_i| spectrum bins: {len(kP)}")
print(f"  u_i spectrum bins: {len(ku)}")

# Create and save the structure function plot
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.loglog(rP, DP, 'o', color='blue', label=r'$D_P$ for $P_i$', markersize=4, linewidth=1.5)
# ax.loglog(ru, Du, 's', color='red', label=r'$D_u$ for $u_i$', markersize=4, linewidth=1.5)

# # Fit reference line: 2A_P * (R/R_0)^(m_psi) / (1 + (R/R_0)^(m_psi)) where m_psi = 11/3
# m_psi = 5.0 / 3.0

# def model_func(r, A_P, R_0):
#     x = r / R_0
#     return 2.0 * A_P * (x ** m_psi) / (1.0 + x ** m_psi)

# r_all = np.concatenate([rP, ru])
# r_min = np.min(r_all[r_all > 0])
# r_max = np.max(r_all)
# r_ref = np.geomspace(r_min, r_max, 100)

# # Fit to DP data
# mask_P = (rP > 0) & (DP > 0)
# r_fit_P = rP[mask_P]
# D_fit_P = DP[mask_P]

# # Initial guess for P_i
# R_0_init_P = np.sqrt(r_min * r_max)
# A_P_init_P = np.median(D_fit_P) / 2.0

# # Perform fit for P_i
# try:
#     popt_P, pcov_P = curve_fit(model_func, r_fit_P, D_fit_P, 
#                                p0=[A_P_init_P, R_0_init_P],
#                                bounds=([0, r_min*0.1], [np.inf, r_max*10]))
#     A_P_fit_P, R_0_fit_P = popt_P
#     print(f"Fitted parameters for P_i: A_P = {A_P_fit_P:.6e}, R_0 = {R_0_fit_P:.6e}")
# except Exception as e:
#     print(f"Fit failed for P_i: {e}, using initial guess")
#     A_P_fit_P, R_0_fit_P = A_P_init_P, R_0_init_P

# Dr_ref_P = model_func(r_ref, A_P_fit_P, R_0_fit_P)
# ax.loglog(r_ref, Dr_ref_P, '--', color='blue', linewidth=1.5, alpha=0.7,
#           label=r'$2A_P \frac{(R/R_0)^{m_\psi}}{1 + (R/R_0)^{m_\psi}}$ ($P_i$)')

# # Fit to Du data
# mask_u = (ru > 0) & (Du > 0)
# r_fit_u = ru[mask_u]
# D_fit_u = Du[mask_u]

# # Initial guess for u_i
# R_0_init_u = np.sqrt(r_min * r_max)
# A_P_init_u = np.median(D_fit_u) / 2.0

# # Perform fit for u_i
# try:
#     popt_u, pcov_u = curve_fit(model_func, r_fit_u, D_fit_u, 
#                                p0=[A_P_init_u, R_0_init_u],
#                                bounds=([0, r_min*0.1], [np.inf, r_max*10]))
#     A_P_fit_u, R_0_fit_u = popt_u
#     print(f"Fitted parameters for u_i: A_P = {A_P_fit_u:.6e}, R_0 = {R_0_fit_u:.6e}")
# except Exception as e:
#     print(f"Fit failed for u_i: {e}, using initial guess")
#     A_P_fit_u, R_0_fit_u = A_P_init_u, R_0_init_u

# Dr_ref_u = model_func(r_ref, A_P_fit_u, R_0_fit_u)
# ax.loglog(r_ref, Dr_ref_u, '--', color='red', linewidth=1.5, alpha=0.7,
#           label=r'$2A_P \frac{(R/R_0)^{m_\psi}}{1 + (R/R_0)^{m_\psi}}$ ($u_i$)')

# ax.set_xlabel(r'$r$')
# ax.set_ylabel(r'$D(r)$')
# # ax.set_title(rf'Structure Functions: $P_i$ vs $u_i$ (N={N}, L={L})', fontsize=14)
# ax.legend(fontsize=24)
# # ax.grid(True, alpha=0.3, which='both')
# r_data_all = np.concatenate([rP[rP > 0], ru[ru > 0]])
# D_data_all = np.concatenate([DP[DP > 0], Du[Du > 0]])
# ax.set_xlim(np.min(r_data_all), np.max(r_data_all))
# ax.set_ylim(np.min(D_data_all), np.max(D_data_all))
# plt.tight_layout()

# # Save structure function figure
# structure_file = 'structure_functions_from_points_1400.png'
# plt.savefig(structure_file, dpi=150, bbox_inches='tight')
# print(f"  Saved structure function figure to {structure_file}")

# # Also save as PDF
# structure_file_pdf = 'structure_functions_from_points_1400.svg'
# plt.savefig(structure_file_pdf, bbox_inches='tight')
# print(f"  Saved structure function figure to {structure_file_pdf}")

# plt.show()
# plt.close()

# Create and save the power spectrum plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(kP, Pk_amp, 'o', color='blue', label=r'$P(k)$ for $|P_i|$', markersize=4, linewidth=1.5)
ax.loglog(ku, Pk_u, 's', color='red', label=r'$P(k)$ for $u_i$', markersize=4, linewidth=1.5)

# Add -11/3 reference lines
k_all = np.concatenate([kP, ku])
k_min = np.min(k_all[k_all > 0])
k_max = np.max(k_all)
k_ref = np.geomspace(k_min, k_max, 100)
anchor_k = np.sqrt(k_min * k_max)

# Reference line for P_i
anchor_idx_P = np.argmin(np.abs(kP - anchor_k))
if anchor_idx_P >= len(Pk_amp):
    anchor_idx_P = len(Pk_amp) - 1
C_P = Pk_amp[anchor_idx_P] * (kP[anchor_idx_P] ** (11.0/3.0))
Pk_ref_P = C_P * k_ref ** (-11.0/3.0)
ax.loglog(k_ref, Pk_ref_P, '--', color='blue', linewidth=1.5, label=r'$k^{-11/3}$ ($|P_i|$)', alpha=0.7)

# Reference line for u_i
anchor_idx_u = np.argmin(np.abs(ku - anchor_k))
if anchor_idx_u >= len(Pk_u):
    anchor_idx_u = len(Pk_u) - 1
C_u = Pk_u[anchor_idx_u] * (ku[anchor_idx_u] ** (11.0/3.0))
Pk_ref_u = C_u * k_ref ** (-11.0/3.0)
ax.loglog(k_ref, Pk_ref_u, '--', color='red', linewidth=1.5, label=r'$k^{-11/3}$ ($u_i$)', alpha=0.7)

# Fit power law for u_i at k > 1000
def powerlaw_func(k, A, alpha):
    return A * (k ** alpha)

mask_fit = (ku > 30) & (ku < 400) & (Pk_u > 0)
if np.sum(mask_fit) > 2:
    k_fit = ku[mask_fit]
    Pk_fit = Pk_u[mask_fit]
    
    log_k = np.log(k_fit)
    log_Pk = np.log(Pk_fit)
    
    try:
        popt, pcov = np.polyfit(log_k, log_Pk, 1, cov=True)
        alpha_fit = popt[0]
        log_A = popt[1]
        A_fit = np.exp(log_A)
        
        k_fit_range = np.geomspace(30.0, 400.0, 200)
        Pk_fit_line = powerlaw_func(k_fit_range, A_fit, alpha_fit)
        ax.loglog(k_fit_range, Pk_fit_line, '-', color='black', linewidth=2.5, 
                  label=rf'$k^{{{alpha_fit:.2f}}}$', alpha=1, zorder=10)
        print(f"Fitted power law for u_i (k>1000): P(k) = {A_fit:.6e} * k^{alpha_fit:.4f}")
    except Exception as e:
        print(f"Fit failed for u_i (k>1000): {e}")
else:
    print(f"No data points with k > 1000 for u_i (found {np.sum(mask_fit)} points)")

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$P(k)$')
# ax.set_title(rf'Power Spectra: $|P_i|$ vs $u_i$ (N={N}, L={L})', fontsize=14)
ax.legend(fontsize=24)
# ax.grid(True, alpha=0.3, which='both')
k_data_all = np.concatenate([kP[kP > 0], ku[ku > 0]])
Pk_data_all = np.concatenate([Pk_amp[Pk_amp > 0], Pk_u[Pk_u > 0]])
ax.set_xlim(np.min(k_data_all), np.max(k_data_all))
ax.set_ylim(np.min(Pk_data_all), np.max(Pk_data_all))
plt.tight_layout()

# Save spectrum figure
spectrum_file = 'power_spectra_from_points_1400.png'
plt.savefig(spectrum_file, dpi=150, bbox_inches='tight')
print(f"  Saved spectrum figure to {spectrum_file}")

# Also save as PDF
spectrum_file_pdf = 'power_spectra_from_points_1400.svg'
plt.savefig(spectrum_file_pdf, bbox_inches='tight')
print(f"  Saved spectrum figure to {spectrum_file_pdf}")

plt.show()
plt.close()

# Compute and plot d(log P) / d(log k) (power-law index)
def compute_dlogP_dlogk(k, Pk, window_fraction=0.15, min_points=5, max_points=11):
    """
    Compute d(log P) / d(log k) using local power-law fitting with adaptive windowing.
    
    For 50 linearly spaced points, this uses:
    - Adaptive window size based on log spacing
    - Weighted fitting with Gaussian weights
    - Better handling of non-uniform log spacing
    
    Parameters:
    -----------
    k : array
        Wavenumber values (linearly spaced)
    Pk : array
        Power spectrum values
    window_fraction : float
        Fraction of total range in log space to use for window (default 0.15 = 15%)
    min_points : int
        Minimum number of points in window (default 5)
    max_points : int
        Maximum number of points in window (default 11)
    """
    # Filter out invalid values
    mask = (k > 0) & (Pk > 0)
    k_valid = k[mask]
    Pk_valid = Pk[mask]
    
    if len(k_valid) < min_points:
        return np.array([]), np.array([])
    
    log_k = np.log(k_valid)
    log_Pk = np.log(Pk_valid)
    
    # Determine window size in log space
    log_k_range = log_k.max() - log_k.min()
    window_size_log = window_fraction * log_k_range
    
    n_points = len(k_valid)
    dlogP_dlogk = np.zeros(n_points)
    k_centers = np.zeros(n_points)
    
    for i in range(n_points):
        # Find points within window_size_log in log space
        log_k_center = log_k[i]
        log_k_low = log_k_center - window_size_log / 2
        log_k_high = log_k_center + window_size_log / 2
        
        # Find indices within this range
        in_window = (log_k >= log_k_low) & (log_k <= log_k_high)
        window_indices = np.where(in_window)[0]
        
        # Ensure we have enough points
        if len(window_indices) < min_points:
            # Expand window if needed
            window_indices = np.arange(max(0, i - max_points//2), 
                                      min(n_points, i + max_points//2 + 1))
        elif len(window_indices) > max_points:
            # Limit to max_points closest to center
            distances = np.abs(log_k[window_indices] - log_k_center)
            sorted_idx = np.argsort(distances)
            window_indices = window_indices[sorted_idx[:max_points]]
        
        # Extract window data
        k_window = log_k[window_indices]
        Pk_window = log_Pk[window_indices]
        
        # Fit linear (power law in log space): log P = alpha * log k + const
        if len(k_window) >= 2:
            # Use Gaussian weights centered on current point
            distances = np.abs(k_window - log_k_center)
            sigma = window_size_log / 3.0  # 3-sigma covers most of window
            weights = np.exp(-0.5 * (distances / sigma)**2)
            weights = weights / (weights.sum() + 1e-30)
            
            # Weighted linear fit
            k_mean = np.average(k_window, weights=weights)
            Pk_mean = np.average(Pk_window, weights=weights)
            
            numerator = np.sum(weights * (k_window - k_mean) * (Pk_window - Pk_mean))
            denominator = np.sum(weights * (k_window - k_mean)**2)
            
            if abs(denominator) > 1e-10:
                alpha = numerator / denominator
            else:
                # Fallback to simple linear fit
                alpha = np.polyfit(k_window, Pk_window, 1)[0]
            
            dlogP_dlogk[i] = alpha
            k_centers[i] = k_valid[i]
        else:
            dlogP_dlogk[i] = np.nan
            k_centers[i] = k_valid[i]
    
    # Remove NaN values
    valid_mask = ~np.isnan(dlogP_dlogk)
    return k_centers[valid_mask], dlogP_dlogk[valid_mask]

# Compute derivatives for both spectra using adaptive windowing
# For 50 linearly spaced points, use 15% of log range with 5-11 points per window
# This gives good balance between smoothness and resolution
print(f"Computing derivatives: len(kP)={len(kP)}, len(ku)={len(ku)}")
kP_deriv, dlogP_dlogk_P = compute_dlogP_dlogk(kP, Pk_amp, window_fraction=0.15, min_points=5, max_points=11)
ku_deriv, dlogP_dlogk_u = compute_dlogP_dlogk(ku, Pk_u, window_fraction=0.15, min_points=5, max_points=11)

# Create plot for d(log P) / d(log k)
fig, ax = plt.subplots(figsize=(10, 6))
if len(kP_deriv) > 0:
    ax.semilogx(kP_deriv, dlogP_dlogk_P, 'o', color='blue', label=r'$\frac{d\log P}{d\log k}$ for $|P_i|$', markersize=4, linewidth=1.5, alpha=0.7)
if len(ku_deriv) > 0:
    ax.semilogx(ku_deriv, dlogP_dlogk_u, 's', color='red', label=r'$\frac{d\log P}{d\log k}$ for $u_i$', markersize=4, linewidth=1.5, alpha=0.7)

# Add reference line at -11/3
k_all_deriv = np.concatenate([kP_deriv, ku_deriv]) if len(kP_deriv) > 0 and len(ku_deriv) > 0 else (kP_deriv if len(kP_deriv) > 0 else ku_deriv)
if len(k_all_deriv) > 0:
    k_min_deriv = np.min(k_all_deriv)
    k_max_deriv = np.max(k_all_deriv)
    k_ref_deriv = np.geomspace(k_min_deriv, k_max_deriv, 100)
    ax.axhline(-11.0/3.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$-11/3$ reference')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\frac{d\log P}{d\log k}$')
ax.legend(fontsize=24)
if len(k_all_deriv) > 0:
    ax.set_xlim(np.min(k_all_deriv), np.max(k_all_deriv))
    if len(kP_deriv) > 0 and len(ku_deriv) > 0:
        y_min = min(np.min(dlogP_dlogk_P), np.min(dlogP_dlogk_u))
        y_max = max(np.max(dlogP_dlogk_P), np.max(dlogP_dlogk_u))
    elif len(kP_deriv) > 0:
        y_min, y_max = np.min(dlogP_dlogk_P), np.max(dlogP_dlogk_P)
    else:
        y_min, y_max = np.min(dlogP_dlogk_u), np.max(dlogP_dlogk_u)
    y_range = y_max - y_min
    # ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

    ax.set_ylim(-4, -3)
plt.tight_layout()

# Save derivative figure
derivative_file = 'dlogP_dlogk_from_points_1400.png'
plt.savefig(derivative_file, dpi=150, bbox_inches='tight')
print(f"  Saved derivative figure to {derivative_file}")

# Also save as SVG
derivative_file_svg = 'dlogP_dlogk_from_points_1400.svg'
plt.savefig(derivative_file_svg, bbox_inches='tight')
print(f"  Saved derivative figure to {derivative_file_svg}")

plt.show()
plt.close()

print("Done!")

