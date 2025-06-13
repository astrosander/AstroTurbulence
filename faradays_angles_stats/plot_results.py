#!/usr/bin/env python
import numpy as np, matplotlib.pyplot as plt, glob, re, pathlib

root = pathlib.Path("results")
files = sorted(glob.glob(str(root/"Dphi_*.npy")),
               key=lambda f: float(re.search(r"_(\d+\.\d+)cm",f).group(1)))

pick = [0, 10, 19]         # unsaturated, transition, saturated indices
labels = []
for i in pick:
    lam_cm = float(re.search(r"_(\d+\.\d+)cm", files[i]).group(1))
    D = np.load(files[i])
    R, Dphi = D[0], D[1]
    plt.loglog(R, Dphi, label=f"λ = {lam_cm:.2f} cm")
    labels.append(lam_cm)

# ----- slope from the FIRST (unsaturated) curve ----------
R_fit, D_fit = np.load(files[pick[0]])
mask = (R_fit > 0.02*R_fit.max()) & (R_fit < 0.1*R_fit.max())
m, _ = np.polyfit(np.log10(R_fit[mask]), np.log10(D_fit[mask]), 1)
plt.text(0.05*R_fit.max(), 0.3*D_fit.max(),
         rf"slope ≈ {m:.2f}", rotation=-30)

plt.xlabel(r"separation  $R\ \mathrm{[pc]}$")
plt.ylabel(r"$D_\phi(R)$  [rad$^2$]")
plt.legend()
plt.tight_layout()
plt.savefig(root/"Dphi_examples.png", dpi=300)
plt.show()
