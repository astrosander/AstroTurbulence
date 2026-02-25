import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load NPZ file
# -------------------------
npz_file = "dens_spectrum_ms10ma08_512_w_00021.npz"
data = np.load(npz_file)

# -------------------------
# Extract arrays
# -------------------------
k = data["k_centers"]
E = data["E_dens"]

# -------------------------
# Power law fit: E(k) = A * k^α
# In log space: log(E) = α * log(k) + log(A)
# -------------------------
# Fit range
k_min = 50
k_max = 500

# Use data from index 1 onwards (skip k=0)
k_fit = k[1:]
E_fit = E[1:]

# Filter: k between k_min and k_max, and positive values
valid = (k_fit >= k_min) & (k_fit <= k_max) & (k_fit > 0) & (E_fit > 0)
k_fit = k_fit[valid]
E_fit = E_fit[valid]

# Fit in log-log space
log_k = np.log10(k_fit)
log_E = np.log10(E_fit)

# Linear fit: log(E) = slope * log(k) + intercept
# slope = α (power law exponent), intercept = log10(A)
coeffs = np.polyfit(log_k, log_E, 1)
alpha = coeffs[0]  # Power law exponent
log_A = coeffs[1]  # log10 of amplitude
A = 10**log_A

# Generate fit curve over the fitted range
k_fit_plot = np.logspace(np.log10(k_min), np.log10(k_max), 100)
E_fit_plot = A * (k_fit_plot ** alpha)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

# Plot data
ax.loglog(k[1:], E[1:], 'o-', markersize=4, label='Data', alpha=0.7)

# Plot fit
ax.loglog(k_fit_plot, E_fit_plot, 'r--', linewidth=2, 
          label=f'Power law fit (k={k_min}-{k_max}): α = {alpha:.3f}')

ax.set_xlabel("k")
ax.set_ylabel("E(k)")
ax.set_title("Column Density Spectrum")
ax.grid(True, which="both")
ax.legend()

plt.tight_layout()
plt.savefig("spectrum_from_npz.png")
plt.show()

# Print fit results
print(f"Power law fit (k={k_min}-{k_max}): E(k) = {A:.6e} * k^{alpha:.3f}")
print(f"Exponent α = {alpha:.3f}")