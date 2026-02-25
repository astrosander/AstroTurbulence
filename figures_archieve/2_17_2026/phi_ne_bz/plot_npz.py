import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- inputs ----
npz_path = "Phi_neBz_integral_spectrum_compare.npz"
out_png = "Phi_neBz_integral_spectrum_compare_from_npz.png"

# ---- load ----
d = np.load(npz_path, allow_pickle=True)

# helper to fetch (works whether stored as str or 0-d object array)
def _get_str(key, default=""):
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, np.ndarray) and v.shape == () and v.dtype == object:
        return str(v.item())
    return str(v)

# ---- plot spectra ----
plt.figure(figsize=(7, 5), dpi=150)

series = [
    ("ms_10", "ms10_k_centers", "ms10_E"),
    ("ms_1",  "ms1_k_centers",  "ms1_E"),
]

for label, kkey, ekey in series:
    k = np.asarray(d[kkey])
    E = np.asarray(d[ekey])

    m = np.isfinite(k) & np.isfinite(E) & (k > 0) & (E > 0)
    plt.loglog(k[m], E[m], label=label)

plt.xlabel(r"$k$")
plt.ylabel(r"$P_{\int n_e B_z dz}(k)$")
plt.legend()
plt.tight_layout()
plt.savefig(out_png)
plt.close()

print(f"loaded: {npz_path}")
print(f"saved : {out_png}")
print("screen_frac:", float(d["screen_frac"]) if "screen_frac" in d else "N/A")
print("ms_10 bcc:", _get_str("ms10_bcc_path"))
print("ms_10 w  :", _get_str("ms10_w_path"))
print("ms_1  bcc:", _get_str("ms1_bcc_path"))
print("ms_1  w  :", _get_str("ms1_w_path"))

# ---- (optional) quick-look images of phi maps ----
# If you want 2D images of phi (integrated ne*Bz*dz), uncomment below.

# for label, phikey in [("ms_10", "ms10_phi"), ("ms_1", "ms1_phi")]:
#     phi = np.asarray(d[phikey])
#     plt.figure(figsize=(6, 5), dpi=150)
#     im = plt.imshow(phi.T, origin="lower", aspect="auto")  # transpose so x~horizontal, y~vertical
#     plt.colorbar(im, label=r"$\int n_e B_z dz$")
#     plt.title(label)
#     plt.tight_layout()
#     fn = f"{label}_phi_from_npz.png"
#     plt.savefig(fn)
#     plt.close()
#     print("saved :", fn)