import numpy as np
import matplotlib.pyplot as plt

# Parameters
C2 = 1.0  # Set C2 = 1 for normalization
L = 1.0   # Normalize L = 1

# R_perp values
R = np.linspace(0, 1, 100)

# Delta-s grid for convolution integrals
d = np.linspace(-L, L, 2001)
w = (1 - np.abs(d) / L)  # weight (L - |Δs|)/L

# Unsmeared 3D correlation: ρ(R) = 1 - (C2/2)*(R/L)^(2/3)
y_unsmoothed = 1 - 0.5 * (R / L)**(5/3)

# Exact projected correlation (eq. 5.3 through convolution trick)
y_exact = []
# Approximate projected correlation (eq. 5.4)
y_approx = []

for R_val in R:
    # Exact: integrand exponent 2/3
    integrand_exact = (R_val**2 + d**2)**(2/3) * w
    I_exact = np.trapz(integrand_exact, d)
    y_exact.append(1 - 0.5 * I_exact / L**(2/3))
    
    # Approx: integrand exponent 1/3
    integrand_approx = (R_val**2 + d**2)**(1/3) * w
    I_approx = np.trapz(integrand_approx, d)
    y_approx.append(1 - 0.5 * I_approx / L**(2/3))

y_exact = np.array(y_exact)
y_approx = np.array(y_approx)

# Plotting
plt.figure()
plt.plot(R, y_unsmoothed, label='Unsmeared: $1 - 0.5\\,R^{2/3}$')
plt.plot(R, y_exact, label='Exact projected ($\\bar{ρ}_⊥$ via eq.5.3)')
plt.plot(R, y_approx, '--', label='Approx projected (small $R$, eq.5.4)')

plt.xlabel('$R_⊥ / L$')
plt.ylabel('$\\bar{ρ}_⊥$')
plt.title('Comparison of Correlation Functions')
plt.legend()
plt.tight_layout()
plt.show()
