import numpy as np
import matplotlib.pyplot as plt

# Create an array of angles from 0 to 2π
theta = np.linspace(0, 2 * np.pi, 500)

# Define r = cos(2x) where x = theta
r = np.cos(2 * theta)

# Create a polar plot
plt.figure()
ax = plt.subplot(111, polar=True)
ax.plot(theta, r)

# Optional: Add title
ax.set_title(r"$r = \cos(2\theta)$", va='bottom')

plt.show()
