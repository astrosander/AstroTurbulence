#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ------------------------ FFT helpers ------------------------
def _get_vec(group, base_names):
    """Return (fx,fy,fz) arrays from 'group' trying multiple name variants."""
    def find(name_opts):
        for n in name_opts:
            if n in group: return np.asarray(group[n])
        return None

    x = find([base_names[0], base_names[0].replace('_', ''), base_names[0].upper(), base_names[0].capitalize()])
    y = find([base_names[1], base_names[1].replace('_', ''), base_names[1].upper(), base_names[1].capitalize()])
    z = find([base_names[2], base_names[2].replace('_', ''), base_names[2].upper(), base_names[2].capitalize()])
    if x is None or y is None or z is None:
        return None
    return x, y, z

def _vector_spectrum(vx, vy, vz):
    """
    Isotropic 3D spectrum of a vector field.
    Returns k_centers (integer shells) and E(k) (shell-averaged power).
    """
    # FFTs
    Fvx = np.fft.fftn(vx); Fvy = np.fft.fftn(vy); Fvz = np.fft.fftn(vz)
    P = (np.abs(Fvx)**2 + np.abs(Fvy)**2 + np.abs(Fvz)**2) / (vx.size)

    nz, ny, nx = vx.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='xy')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    kmax = int(np.max(Kmag))  # integer shells
    Ek   = np.zeros(kmax+1, dtype=np.float64)
    cnt  = np.zeros(kmax+1, dtype=np.int64)

    # Bin to integer shells
    inds = np.rint(Kmag).astype(int)
    for i in range(kmax+1):
        m = (inds == i)
        if np.any(m):
            Ek[i] = P[m].mean()
            cnt[i] = m.sum()

    k = np.arange(kmax+1, dtype=float)
    # discard k=0 (mean)
    m = (k > 0) & (cnt > 0)
    return k[m], Ek[m]

def _fit_and_line(k, Ek, slope=-5/3, frac_ref=0.3):
    """Build a dashed slope line anchored at a reference k inside the inertial range."""
    m = (k > k.min()*2) & (k < k.max()*frac_ref)
    if not np.any(m):  # fallback
        m = slice(len(k)//4, len(k)//2)
    kref = np.median(k[m])
    # choose amplitude so line passes through spectrum near kref
    Eref = np.interp(kref, k, Ek)
    A = Eref / (kref**slope)
    return A * (k**slope)

# ------------------------ HDF5 loading ------------------------
def _parse_MA(name, grp):
    # attribute first
    for key in ('MA','M_A','Ma','mA'):
        if key in grp.attrs:
            try:
                return float(grp.attrs[key])
            except Exception:
                pass
    # from group name like "MA_3.0" or "run_MA1p5"
    m = re.search(r'MA[_ ]?([0-9]+(?:\.[0-9]+)?)', name, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None

def load_runs(h5path):
    runs = []
    with h5py.File(h5path, 'r') as f:
        groups = [f[g] for g in f.keys()] if len(f.keys())>0 and isinstance(f[list(f.keys())[0]], h5py.Group) else [f]
        for g in groups:
            name = g.name.split('/')[-1] or 'root'
            MA = _parse_MA(name, g)

            v = _get_vec(g, ('vx','vy','vz')) or _get_vec(g, ('v_x','v_y','v_z'))
            B = _get_vec(g, ('bx','by','bz')) or _get_vec(g, ('B_x','B_y','B_z'))

            if v is None or B is None:
                # try nested group (some formats keep fields in a sub-group)
                for subname, subg in g.items():
                    if not isinstance(subg, h5py.Group): continue
                    v = v or _get_vec(subg, ('vx','vy','vz')) or _get_vec(subg, ('v_x','v_y','v_z'))
                    B = B or _get_vec(subg, ('bx','by','bz')) or _get_vec(subg, ('B_x','B_y','B_z'))
            if v is None or B is None:
                print(f"[warn] Skipping '{name}': could not find velocity and magnetic vectors.")
                continue

            runs.append(dict(name=name, MA=MA, v=v, B=B))
    return runs

# ------------------------ Plot ------------------------
def plot_spectra(h5path, out="mhd_spectra.pdf",
                 inj_k=2.0, MA_for_kA=3.0,
                 show_kA=True, logy_limits=None):
    runs = load_runs(h5path)
    if not runs:
        raise RuntimeError("No runs with velocity+magnetic fields found.")

    plt.figure(figsize=(10,4))

    # Left: velocity
    ax1 = plt.subplot(1,2,1)
    colors = ["tab:green","tab:red","tab:blue","k","tab:purple","tab:orange"]
    labels = []
    for i, r in enumerate(runs):
        k, Ev = _vector_spectrum(*r['v'])
        ax1.loglog(k, Ev, color=colors[i%len(colors)], lw=1.8)
        labels.append(f"$M_s\\approx 1,\\ M_A\\approx {r['MA']:.1f}$" if r['MA'] else r['name'])

    # -5/3 guide
    guide = _fit_and_line(k, Ev, slope=-5/3)
    ax1.loglog(k, guide, 'k--', lw=1.5, label="slope: $-5/3$")

    if show_kA:
        kA = inj_k * (MA_for_kA**3)    # l_A = L/M_A^3  -> k_A ~ k_inj * M_A^3 (super-Alfvénic)
        ax1.axvline(kA, color='r', ls='--', lw=1.5, label=r"$k_A$ (for $M_A\simeq 3$)")

    ax1.set_xlabel(r"$k\,L_{\rm box}/(2\pi)$")
    ax1.set_ylabel(r"$E_v(k)$")
    ax1.set_title("Velocity Spectrum")
    if logy_limits: ax1.set_ylim(*logy_limits)
    # legend block
    from matplotlib.patches import Patch
    leg_elems = [Patch(color='none', label="slope: $-5/3$")]
    ax1.legend([plt.Line2D([],[], color='k', ls='--', lw=1.5, label="slope: $-5/3$")]+
               [plt.Line2D([],[], color=c, lw=1.8) for c in colors[:len(runs)]],
               ["slope: $-5/3$"]+labels, fontsize=9, loc='lower left', framealpha=0.9)

    # Right: magnetic
    ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
    for i, r in enumerate(runs):
        k, Eb = _vector_spectrum(*r['B'])
        ax2.loglog(k, Eb, color=colors[i%len(colors)], lw=1.8)

    guideB = _fit_and_line(k, Eb, slope=-5/3)
    ax2.loglog(k, guideB, 'k--', lw=1.5)
    if show_kA:
        kA = inj_k * (MA_for_kA**3)
        ax2.axvline(kA, color='b', ls='--', lw=1.5)

    ax2.set_xlabel(r"$k\,L_{\rm box}/(2\pi)$")
    ax2.set_ylabel(r"$E_B(k)$")
    ax2.set_title("Magnetic Field Spectrum")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot velocity and magnetic spectra from mhd_fields.h5")
    ap.add_argument("h5", nargs="?", default="mhd_fields.h5", help="input HDF5 file")
    ap.add_argument("--out", default="mhd_spectra.pdf", help="output figure")
    ap.add_argument("--inj_k", type=float, default=2.0, help="injection wavenumber in kL/2π units")
    ap.add_argument("--kA_MA", type=float, default=3.0, help="M_A used to draw k_A (= inj_k * M_A^3)")
    ap.add_argument("--no_kA", action="store_true", help="do not draw k_A line")
    args = ap.parse_args()
    plot_spectra(args.h5, out=args.out, inj_k=args.inj_k, MA_for_kA=args.kA_MA, show_kA=not args.no_kA)

if __name__ == "__main__":
    main()
