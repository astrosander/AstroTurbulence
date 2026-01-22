import numpy as np
import matplotlib.pyplot as plt

# Load data from 1400.npz
input_file = '128.npz'
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

print(f"Loaded structure function data:")
print(f"  P_i structure bins: {len(rP)}")
print(f"  u_i structure bins: {len(ru)}")
print(f"Loaded power spectrum data:")
print(f"  |P_i| spectrum bins: {len(kP)}")
print(f"  u_i spectrum bins: {len(ku)}")

# Create and save the structure function plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(rP, DP, 'o-', label='D_P (P_i structure function)', markersize=4, linewidth=1.5)
ax.loglog(ru, Du, 's-', label='D_u (u_i structure function)', markersize=4, linewidth=1.5)
ax.set_xlabel('r (separation distance)', fontsize=12)
ax.set_ylabel('D(r) (structure function)', fontsize=12)
ax.set_title(f'Structure Functions: P_i vs u_i (N={N}, L={L})', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

# Save figure
figure_file = 'structure_functions_1400.png'
plt.savefig(figure_file, dpi=150, bbox_inches='tight')
print(f"  Saved figure to {figure_file}")

# Also save as PDF
figure_file_pdf = 'structure_functions_1400.pdf'
plt.savefig(figure_file_pdf, bbox_inches='tight')
print(f"  Saved figure to {figure_file_pdf}")

plt.close()

# Create and save the power spectrum plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(kP, Pk_amp, 'o-', label='P(k) for |P_i|', markersize=4, linewidth=1.5)
ax.loglog(ku, Pk_u, 's-', label='P(k) for u_i', markersize=4, linewidth=1.5)
ax.set_xlabel('k (wavenumber)', fontsize=12)
ax.set_ylabel('P(k) (power spectrum)', fontsize=12)
ax.set_title(f'Power Spectra: |P_i| vs u_i (N={N}, L={L})', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

# Save spectrum figure
spectrum_file = 'power_spectra_1400.png'
plt.savefig(spectrum_file, dpi=150, bbox_inches='tight')
print(f"  Saved spectrum figure to {spectrum_file}")

# Also save as PDF
spectrum_file_pdf = 'power_spectra_1400.pdf'
plt.savefig(spectrum_file_pdf, bbox_inches='tight')
print(f"  Saved spectrum figure to {spectrum_file_pdf}")

plt.close()

# Save structure function and spectrum points to the same npz
output_points_file = 'structure_functions_points_1400.npz'
np.savez(
    output_points_file,
    # Structure functions
    rP=rP,
    DP=DP,
    ru=ru,
    Du=Du,
    # Power spectra
    kP=kP,
    Pk_amp=Pk_amp,
    ku=ku,
    Pk_u=Pk_u,
    # Parameters
    N=N,
    L=L,
)
print(f"  Saved structure function and spectrum points to {output_points_file}")

print("Done!")

