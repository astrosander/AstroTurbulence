#!/usr/bin/env python3
"""
plot_results.py  –  three diagnostics + 5/3 and 2/3 reference slopes
"""
import numpy as np, matplotlib.pyplot as plt, glob, re, pathlib
root = pathlib.Path("results")

def files(tag):
    fl = sorted(glob.glob(str(root/f"{tag}_*.npy")),
                key=lambda f: float(re.search(r"_(\d+\.\d+)cm",f).group(1)))
    return fl

Dphi   = files("Dphi")
DphiLn = files("DphiLn")
DP     = files("DP")

pick   = [0, len(Dphi)//2, -1]
color  = ["tab:blue", "tab:orange", "tab:green"]

fig,ax = plt.subplots(1,3,figsize=(15,5),sharey=True)

# ---- helper to plot one set ----------------------------------------
def plot_set(fileset, axis, title, yscale=1.0, label_suffix=""):
    for c,i in zip(color, pick):
        lam = float(re.search(r"_(\d+\.\d+)cm", fileset[i]).group(1))
        R,D = np.load(fileset[i])
        if np.ndim(D)==1:
            D_plot = D/yscale
        else:
            D_plot = D
        axis.loglog(R, D_plot, c, label=f"λ={lam:.1f} cm{label_suffix}")
    axis.set_title(title)

plot_set(Dphi,   ax[0], r"$D_{\varphi}(R)$")
plot_set(DphiLn, ax[1], r"$-\ln S / 2\lambda^{4}$")
plot_set(DP,     ax[2], r"$D_{P}(R)/\lambda^{4}$",
         yscale=[(float(re.search(r"_(\d+\.\d+)cm", f).group(1)))**4
                  for f in Dphi][0],  # scale only legend text later
         label_suffix=" (λ⁻⁴)")

# ---- reference slopes ----------------------------------------------
R_the = np.array([2,40])
for axis in ax:
    ymin, ymax = axis.get_ylim()
    D0 = 10**(0.5*(np.log10(ymin)+np.log10(ymax)))
    axis.loglog(R_the, D0*(R_the/10)**(5/3), "k--", lw=0.8)
    axis.loglog(R_the, D0*(R_the/10)**(2/3), "k:",  lw=0.8)
    axis.set_xlabel(r"$R$ [pc]")
ax[0].set_ylabel(r"structure function  [rad$^{2}$]")
for axis in ax: axis.legend(frameon=False)
plt.tight_layout(); plt.savefig(root/"compare_measures.png", dpi=300)
plt.show()

# ---- slopes to console ---------------------------------------------
def slope(R,D):
    mask = np.isfinite(D) & (D>0) & (R>2) & (R<40)
    return np.polyfit(np.log10(R[mask]), np.log10(D[mask]), 1)[0]

print("fitted slopes:")
tags = ("Dphi", "DphiLn", "DP/λ⁴")
for tag,fl in zip(tags,(Dphi, DphiLn, DP)):
    R,D = np.load(fl[pick[0]])
    if tag=="DP/λ⁴":
        lam = float(re.search(r"_(\d+\.\d+)cm", fl[pick[0]]).group(1))
        D = D/lam**4
    print(f"  {tag}: {slope(R,D):.3f}")
