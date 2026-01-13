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
input_file = 'structure_functions_points_1400.npz'
print(f"Loading data from {input_file}...")
data = np.load(input_file)

# Extract structure function data
rP = data['rP']
DP = data['DP']
ru = data['ru']
Du = data['Du']

# Extract power spectrum data
kP = data['kP'][1:]
Pk_amp = data['Pk_amp'][1:]
ku = data['ku'][1:]
Pk_u = data['Pk_u'][1:]

# Extract parameters for title/labels
N = int(data['N'])
L = float(data['L'])

print(f"Loaded data:")
print(f"  P_i structure bins: {len(rP)}")
print(f"  u_i structure bins: {len(ru)}")
print(f"  |P_i| spectrum bins: {len(kP)}")
print(f"  u_i spectrum bins: {len(ku)}")

# Create and save the structure function plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(rP, DP, 'o', color='blue', label=r'$D_P$ for $P_i$', markersize=4, linewidth=1.5)
ax.loglog(ru, Du, 's', color='red', label=r'$D_u$ for $u_i$', markersize=4, linewidth=1.5)

# Fit reference line: 2A_P * (R/R_0)^(m_psi) / (1 + (R/R_0)^(m_psi)) where m_psi = 11/3
m_psi = 5.0 / 3.0

def model_func(r, A_P, R_0):
    x = r / R_0
    return 2.0 * A_P * (x ** m_psi) / (1.0 + x ** m_psi)

r_all = np.concatenate([rP, ru])
r_min = np.min(r_all[r_all > 0])
r_max = np.max(r_all)
r_ref = np.geomspace(r_min, r_max, 100)

# Fit to DP data
mask = (rP > 0) & (DP > 0)
r_fit = rP[mask]
D_fit = DP[mask]

# Initial guess
R_0_init = np.sqrt(r_min * r_max)
A_P_init = np.median(D_fit) / 2.0

# Perform fit
try:
    popt, pcov = curve_fit(model_func, r_fit, D_fit, 
                          p0=[A_P_init, R_0_init],
                          bounds=([0, r_min*0.1], [np.inf, r_max*10]))
    A_P_fit, R_0_fit = popt
    print(f"Fitted parameters: A_P = {A_P_fit:.6e}, R_0 = {R_0_fit:.6e}")
except Exception as e:
    print(f"Fit failed: {e}, using initial guess")
    A_P_fit, R_0_fit = A_P_init, R_0_init
# A_P_fit=1
# R_0_fit=0.2
Dr_ref = model_func(r_ref, A_P_fit, R_0_fit)
ax.loglog(r_ref, Dr_ref, '--', color='gray', linewidth=1.5, 
          label=r'$2A_P \frac{(R/R_0)^{m_\psi}}{1 + (R/R_0)^{m_\psi}}$', alpha=0.7)

ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$D(r)$')
# ax.set_title(rf'Structure Functions: $P_i$ vs $u_i$ (N={N}, L={L})', fontsize=14)
ax.legend(fontsize=24)
# ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

# Save structure function figure
structure_file = 'structure_functions_from_points_1400.png'
plt.savefig(structure_file, dpi=150, bbox_inches='tight')
print(f"  Saved structure function figure to {structure_file}")

# Also save as PDF
structure_file_pdf = 'structure_functions_from_points_1400.pdf'
plt.savefig(structure_file_pdf, bbox_inches='tight')
print(f"  Saved structure function figure to {structure_file_pdf}")

plt.show()
plt.close()

# Create and save the power spectrum plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(kP, Pk_amp, 'o', color='blue', label=r'$P(k)$ for $|P_i|$', markersize=4, linewidth=1.5)
ax.loglog(ku, Pk_u, 's', color='red', label=r'$P(k)$ for $u_i$', markersize=4, linewidth=1.5)

# Add -11/3 reference line
k_all = np.concatenate([kP, ku])
k_min = np.min(k_all[k_all > 0])
k_max = np.max(k_all)
k_ref = np.geomspace(k_min, k_max, 100)
anchor_k = np.sqrt(k_min * k_max)
anchor_idx = np.argmin(np.abs(kP - anchor_k))
if anchor_idx >= len(Pk_amp):
    anchor_idx = len(Pk_amp) - 1
C = Pk_amp[anchor_idx] * (kP[anchor_idx] ** (11.0/3.0))
Pk_ref = C * k_ref ** (-11.0/3.0)
ax.loglog(k_ref, Pk_ref, '--', color='gray', linewidth=1.5, label=r'$k^{-11/3}$', alpha=0.7)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$P(k)$')
# ax.set_title(rf'Power Spectra: $|P_i|$ vs $u_i$ (N={N}, L={L})', fontsize=14)
ax.legend(fontsize=24)
# ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

# Save spectrum figure
spectrum_file = 'power_spectra_from_points_1400.png'
plt.savefig(spectrum_file, dpi=150, bbox_inches='tight')
print(f"  Saved spectrum figure to {spectrum_file}")

# Also save as PDF
spectrum_file_pdf = 'power_spectra_from_points_1400.pdf'
plt.savefig(spectrum_file_pdf, bbox_inches='tight')
print(f"  Saved spectrum figure to {spectrum_file_pdf}")

plt.show()
plt.close()

print("Done!")

