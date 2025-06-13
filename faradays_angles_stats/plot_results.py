#!/usr/bin/env python
"""
plot_results.py  –  show three λ curves and fit 5/3 slope
"""
import numpy as np, matplotlib.pyplot as plt, glob, re, pathlib, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

root = pathlib.Path("results")
files = sorted(glob.glob(str(root/"Dphi_*.npy")),
               key=lambda f: float(re.search(r"_(\d+\.\d+)cm", f).group(1)))

pick = [0, len(files)//2, -1]        # unsat / mid / sat

for idx in pick:
    lam_cm = float(re.search(r"_(\d+\.\d+)cm", files[idx]).group(1))
    R,D = np.load(files[idx])
    good = D>0
    plt.loglog(R[good], D[good], label=fr"λ = {lam_cm:.2f} cm")

# -------- slope fit on first curve (unsaturated) -----------
R,D = np.load(files[pick[0]])
mask = (D>0) & (R>2) & (R<40)        # 2–40 pc inertial range
m,_ = np.polyfit(np.log10(R[mask]), np.log10(D[mask]), 1)
plt.text(0.07*R.max(), 0.4*D[mask].max(),
         fr"slope ≈ {m:.2f}", rotation=-28)

plt.xlabel(r"separation $R$ [pc]")
plt.ylabel(r"$D_\phi(R)$  [rad$^{2}$]")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(root/"Dphi_examples.png", dpi=300)
plt.show()
