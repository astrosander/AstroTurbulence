#!/usr/bin/env python3
# plot_stokes_and_directional_spectrum.py

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


# Styling (close to what you used)
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 22


def fft2_power(field):
    """
    Return |FFT2(field)|^2 with fftshift applied.
    """
    fk = np.fft.fft2(field)
    pk = np.abs(fk) ** 2
    return np.fft.fftshift(pk)


def kgrid_2d(Ny, Nx, Ly, Lx):
    """
    Angular wavenumber grid (rad / length unit), fftshifted ordering.
    """
    dy = Ly / Ny
    dx = Lx / Nx
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.sqrt(KX * KX + KY * KY)
    return np.fft.fftshift(K)


def radial_bin_average(K, P2, nbins=50, kmin=None, kmax=None, logbins=True):
    """
    Radially average an isotropic 2D spectrum P2(Kx,Ky) -> P1(K).
    """
    k = K.ravel()
    p = P2.ravel()

    mask = np.isfinite(k) & np.isfinite(p) & (k > 0)
    k = k[mask]
    p = p[mask]

    if kmin is None:
        kmin = np.min(k)
    if kmax is None:
        kmax = np.max(k)

    if logbins:
        edges = np.geomspace(kmin, kmax, nbins + 1)
    else:
        edges = np.linspace(kmin, kmax, nbins + 1)

    idx = np.digitize(k, edges) - 1
    good = (idx >= 0) & (idx < nbins)

    idx = idx[good]
    k = k[good]
    p = p[good]

    # bin centers
    kc = np.sqrt(edges[:-1] * edges[1:]) if logbins else 0.5 * (edges[:-1] + edges[1:])

    # mean power per bin
    num = np.bincount(idx, weights=p, minlength=nbins)
    den = np.bincount(idx, minlength=nbins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pmean = num / den
    valid = den > 0
    return kc[valid], pmean[valid]


def overlay_powerlaw(ax, k, Pk, slope, anchor_k, label):
    """
    Draw a reference powerlaw ~ k^{-slope}, anchored at (anchor_k, P(anchor_k)).
    """
    if anchor_k <= k.min() or anchor_k >= k.max():
        anchor_k = np.sqrt(k.min() * k.max())
    # pick nearest point to anchor
    j = np.argmin(np.abs(np.log(k) - np.log(anchor_k)))
    C = Pk[j] * (k[j] ** slope)
    ax.loglog(k, C * k ** (-slope), lw=2.2, label=label)


def make_args():
    ap = argparse.ArgumentParser(description="Plot Stokes maps and directional spectrum from a saved cube.")
    ap.add_argument("--infile", type=str, default="screens_cube.npz")
    ap.add_argument("--chan", type=int, default=0, help="Which chi channel to use (index into saved chi array).")
    ap.add_argument("--nbins", type=int, default=60, help="Radial bins for the 1D spectrum.")
    ap.add_argument("--eps", type=float, default=1e-12, help="Floor for sqrt(Q^2+U^2) normalization.")
    ap.add_argument("--save_prefix", type=str, default="", help="If set, saves figures as <prefix>_*.png")
    return ap


def main():
    args = make_args().parse_args()

    dat = np.load(args.infile, allow_pickle=True)
    psi = dat["psi"]
    phi = dat["phi"]
    chi = dat["chi"]
    Q = dat["Q"]
    U = dat["U"]
    meta = json.loads(str(dat["meta"]))

    Ny, Nx = meta["Ny"], meta["Nx"]
    Ly, Lx = meta["Ly"], meta["Lx"]
    m_psi = meta["m_psi"]
    m_phi = meta["m_phi"]
    r_phi = meta["r_phi"]
    R0_psi = meta["R0_psi"]

    if args.chan < 0 or args.chan >= Q.shape[0]:
        raise ValueError(f"--chan out of range. Have {Q.shape[0]} channels.")

    chi0 = float(chi[args.chan])
    Q0 = Q[args.chan]
    U0 = U[args.chan]

    P_amp = np.sqrt(Q0 * Q0 + U0 * U0)

    # # --- Plot Stokes maps ---
    # fig1, ax = plt.subplots(1, 3, figsize=(15.6, 5.0))
    # im0 = ax[0].imshow(Q0, origin="lower")
    # ax[0].set_title(rf"$Q$ (chan={args.chan}, $\chi={chi0:g}$)")
    # plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # im1 = ax[1].imshow(U0, origin="lower")
    # ax[1].set_title(rf"$U$ (chan={args.chan})")
    # plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # im2 = ax[2].imshow(P_amp, origin="lower")
    # ax[2].set_title(r"$\sqrt{Q^2+U^2}$")
    # plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    # for a in ax:
    #     a.set_xticks([])
    #     a.set_yticks([])
    # fig1.tight_layout()

    # if args.save_prefix:
    #     fig1.savefig(f"{args.save_prefix}_stokes.png", dpi=200)

    # --- Directional field and directional spectrum (Eq. 29) ---
    # u(x) = (Q+iU)/sqrt(Q^2+U^2)
    u = (Q0 + 1j * U0) / np.maximum(P_amp, args.eps)

    Fphi_2d = fft2_power(u)

    K = kgrid_2d(Ny, Nx, Ly, Lx)
    k1, Fphi_1d = radial_bin_average(K, Fphi_2d, nbins=args.nbins, logbins=True)

    # --- Predicted crossover ---
    # Given: R_x,u = r * chi^{2/(m_psi - m_phi)} ; choose r=r_phi
    # Then K_x ~ 2pi / R_x,u
    exponent = 2.0 / (m_psi - m_phi)
    Rx = r_phi * (chi0 ** exponent)
    Kx = 2.0 * np.pi / Rx

    # # --- Plot 2D directional spectrum (optional but useful) ---
    # fig2, ax2 = plt.subplots(1, 1, figsize=(6.8, 5.8))
    # # log stretch; add a tiny offset to avoid log(0)
    # img = ax2.imshow(np.log10(Fphi_2d + 1e-30), origin="lower")
    # ax2.set_title(r"$\log_{10}\, \mathcal{F}_\phi(\mathbf{K})$")
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # plt.colorbar(img, ax=ax2, fraction=0.046, pad=0.04)
    # fig2.tight_layout()
    # if args.save_prefix:
    #     fig2.savefig(f"{args.save_prefix}_Fphi2d.png", dpi=200)

    # --- Plot 1D (radially averaged) directional spectrum with asymptotics ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(7.4, 4.9))
    ax3.loglog(k1, Fphi_1d, "o", ms=3.0, color="k", label=r"$\langle \mathcal{F}_\phi(K)\rangle_{\rm ang}$")
    ax3.axvline(Kx, color="green", ls="-.", lw=1.2, label=rf"$K_\times \approx 2\pi/R_\times$")

    # Overlay expected slopes (anchor at Kx)
    overlay_powerlaw(ax3, k1, Fphi_1d, slope=(m_psi + 2.0), anchor_k=np.max(k1),
                     label=rf"$K^{{-(m_\psi+2)}}$ (intrinsic)")
    overlay_powerlaw(ax3, k1, Fphi_1d, slope=(m_phi + 2.0), anchor_k=np.max(k1),#Kx
                     label=rf"$K^{{-(m_\phi+2)}}$ (Faraday)")

    ax3.set_xlabel(r"$K$ [rad / length]")
    ax3.set_ylabel(r"Directional spectrum (arb. units)")
    ax3.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax3.legend(loc="best")
    ax3.set_xlim(np.min(k1), np.max(k1))
    ax3.set_ylim(np.min(Fphi_1d), np.max(Fphi_1d))
    fig3.tight_layout()

    if args.save_prefix:
        fig3.savefig(f"{args.save_prefix}_Fphi1d.png", dpi=200)

    # Print key scales
    print("\n--- Loaded meta ---")
    print(json.dumps(meta, indent=2))
    print("\n--- Channel ---")
    print(f"chan = {args.chan}, chi = {chi0:g}")
    print("\n--- Crossover prediction ---")
    print(f"R_x,u = r_phi * chi^(2/(m_psi-m_phi)) = {Rx:.6g}")
    print(f"K_x ~ 2pi / R_x,u = {Kx:.6g}")
    print("--------------------------------\n")

    plt.savefig(f"img_{chi}.png")
    plt.savefig(f"img_{chi}.svg")
    plt.show()


if __name__ == "__main__":
    main()