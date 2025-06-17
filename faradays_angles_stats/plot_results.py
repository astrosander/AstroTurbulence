#!/usr/bin/env python3
"""
plot_results.py  –  comparison figure with theory lines
Requires: matplotlib, numpy
"""
import numpy as np, matplotlib.pyplot as plt, glob, re, pathlib
root = pathlib.Path("results")

def grab(tag):
    fl = sorted(glob.glob(str(root/f"{tag}_*.npy")),
                key=lambda f: float(re.search(r"_(\d+\.\d+)cm",f).group(1)))
    return fl

Dphi_files   = grab("Dphi")
DphiLn_files = grab("DphiLn")
DP_files     = grab("DP")

pick = [0, len(Dphi_files)//2, -1]           # unsat / mid / sat
colors = ["tab:blue","tab:orange","tab:green"]

fig,axes = plt.subplots(1,3,figsize=(15,5),sharey=True)

# ---------- panel A  D_phi -------------------------------------------
for c,i in zip(colors, pick):
    lam = float(re.search(r"_(\d+\.\d+)cm", Dphi_files[i]).group(1))
    R,D = np.load(Dphi_files[i])
    axes[0].loglog(R,D,c,label=f"λ={lam:.1f} cm")
axes[0].set_title(r"$D_{\varphi}(R)$")

# ---------- panel B  D_phi^ln ----------------------------------------
for c,i in zip(colors, pick):
    lam = float(re.search(r"_(\d+\.\d+)cm", DphiLn_files[i]).group(1))
    R,D = np.load(DphiLn_files[i])
    axes[1].loglog(R,D,c,label=f"λ={lam:.1f} cm")
axes[1].set_title(r"$-\ln S / (2\lambda^{4})$")

# ---------- panel C  PSA / λ⁴ ----------------------------------------
for c,i in zip(colors, pick):
    lam = float(re.search(r"_(\d+\.\d+)cm", DP_files[i]).group(1))
    R,D = np.load(DP_files[i])
    axes[2].loglog(R,D/lam**4,c,label=f"λ={lam:.1f} cm")
axes[2].set_title(r"$D_{P}(R)/\lambda^{4}$")

# ---------- common decorations ---------------------------------------
for ax in axes:
    ax.set_xlabel(r"$R$  [pc]")
    ax.legend(frameon=False)
axes[0].set_ylabel(r"structure function  [rad$^{2}$]")

# theory slope guides anchored at R₀ = 10 pc
R0 = 10
for ax in axes:
    ymin, ymax = ax.get_ylim()
    R_theo = np.array([2,40])
    # anchor at mid of y-range to keep visible
    D0 = 10**(0.5*(np.log10(ymin)+np.log10(ymax)))
    ax.loglog(R_theo, D0*(R_theo/R0)**(5/3), "k--", lw=0.8)
    ax.loglog(R_theo, D0*(R_theo/R0)**(2/3), "k:",  lw=0.8)

plt.tight_layout()
plt.savefig(root/"compare_measures.png", dpi=300)
plt.show()

# ---------- console slopes for sanity --------------------------------
def slope(R,D):
    mask = (R>2) & (R<40)
    m,_ = np.polyfit(np.log10(R[mask]), np.log10(D[mask]), 1)
    return m
print("fitted slopes:")
for tag,fl in zip(("Dphi","DphiLn","DP/λ⁴"),
                  (Dphi_files, DphiLn_files, DP_files)):
    R,D = np.load(fl[pick[0]])
    if tag=="DP/λ⁴":
        lam = float(re.search(r"_(\d+\.\d+)cm", fl[pick[0]]).group(1))
        D = D/lam**4
    print(f"  {tag}: {slope(R,D):.3f}")
