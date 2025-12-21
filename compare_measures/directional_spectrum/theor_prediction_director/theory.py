import numpy as np
import matplotlib.pyplot as plt

def ring_average_2d(power2d, kmin=1.0, kmax=None, nbins=40):
    ny, nx = power2d.shape
    ky = np.arange(-ny//2, ny - ny//2)
    kx = np.arange(-nx//2, nx - nx//2)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)

    if kmax is None:
        kmax = K.max()

    edges = np.linspace(kmin, kmax, nbins + 1)
    kc = 0.5 * (edges[:-1] + edges[1:])
    Pk = np.zeros_like(kc)

    for i in range(nbins):
        m = (K >= edges[i]) & (K < edges[i+1])
        if np.any(m):
            Pk[i] = np.mean(power2d[m])
        else:
            Pk[i] = np.nan

    good = np.isfinite(Pk)
    return kc[good], Pk[good]

def D_P_model(R, A_P=1.0, R0=50.0, mpsi=1.0):
    x = (R / R0)**mpsi
    return 2.0 * A_P * x / (1.0 + x)

def D_Phi_model(R, sigma_RM=10.0, rphi=30.0, mPhi=1.0):
    x = (R / rphi)**mPhi
    return 2.0 * (sigma_RM**2) * x / (1.0 + x)

def radial_grid(N):
    y = np.arange(-N//2, N - N//2)
    x = np.arange(-N//2, N - N//2)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    return R

def corr_from_structure(D, C0):
    return C0 - 0.5 * D

def spectrum_from_corr(C):
    C0 = np.fft.ifftshift(C)
    Pk = np.fft.fft2(C0)
    Pk = np.fft.fftshift(Pk)
    return Pk.real

def directional_spectrum_from_models(
    N=512,
    A_P=1.0, R0=55.0, mpsi=1.37,
    sigma_RM=108.0, rphi=33.8, mPhi=0.93,
    chi_list=(0.2, 1.0, 5.0, 20.0),
    nbins=50,
    kmin=2.0,
):
    R = radial_grid(N)

    DP  = D_P_model(R, A_P=A_P, R0=R0, mpsi=mpsi)
    DPH = D_Phi_model(R, sigma_RM=sigma_RM, rphi=rphi, mPhi=mPhi)

    xi_P = corr_from_structure(DP, C0=1.0)

    xi_Phi = corr_from_structure(DPH, C0=sigma_RM**2)

    P_ui_2d  = spectrum_from_corr(xi_P)
    P_Phi_2d = spectrum_from_corr(xi_Phi)

    kc_ui,  P_ui  = ring_average_2d(P_ui_2d,  kmin=kmin, nbins=nbins)
    kc_ph,  P_ph  = ring_average_2d(P_Phi_2d, kmin=kmin, nbins=nbins)

    results = []

    for chi in chi_list:
        lam2 = chi / (2.0 * sigma_RM)
        lam4 = lam2**2

        C_dir = xi_P * np.exp(-2.0 * lam4 * DPH)

        Pdir_2d = spectrum_from_corr(C_dir)
        kc, Pdir = ring_average_2d(Pdir_2d, kmin=kmin, nbins=nbins)

        results.append((chi, kc, Pdir))

    plt.figure(figsize=(7.5, 5.5))

    for chi, kc, Pdir in results:
        plt.loglog(kc, Pdir, lw=1.5, label=rf"$\chi={chi:g}$")

    if results:
        _, kc_ref, Pdir_ref = results[0]
        if len(kc_ref) > 0:
            k_mid = kc_ref[len(kc_ref)//2]
            P_mid = Pdir_ref[len(Pdir_ref)//2]
            k_ref = np.logspace(np.log10(kc_ref.min()), np.log10(kc_ref.max()), 100)
            P_ref = P_mid * (k_ref / k_mid)**(-11.0/3.0) *0.3
            plt.loglog(k_ref, P_ref, 'k--', lw=1.5, alpha=0.7, label=r"$k^{-11/3}$")
            P_ref = P_mid * (k_ref / k_mid)**(-10.0/3.0) *0.9
            plt.loglog(k_ref, P_ref, 'r--', lw=1.5, alpha=0.7, label=r"$k^{-10/3}$")

    plt.xlabel(r"$k$ (FFT index radius)")
    plt.ylabel(r"$P_{\rm dir}(k)$ (arb. units)")
    plt.title(r"Directional spectrum from $D_P(R)$ and $D_\Phi(R)$")

    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.5, 5.5))
    plt.loglog(kc_ui, P_ui,  lw=2, label=r"$P_{u_i}(k)$ from $\xi_P$")
    plt.loglog(kc_ph, P_ph,  lw=2, label=r"$P_{\Phi}(k)$ from $\xi_\Phi$")
    plt.xlabel(r"$k$ (FFT index radius)")
    plt.ylabel(r"$P(k)$ (arb. units)")
    plt.title("Reference spectra implied by the input structure functions")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

def mPhi_from_beta3D(beta_3d):
    m3 = beta_3d - 3.0
    tilde = min(m3, 1.0)
    return 1.0 + tilde

if __name__ == "__main__":
    chi_list = [0]
    chi_list_item=0.5
    for i in range(8):
        chi_list.append(chi_list_item)
        chi_list_item*=2

    beta_kolm = 11.0/3.0
    mPhi_kolm = mPhi_from_beta3D(beta_kolm)

    directional_spectrum_from_models(
        N=1024,
        A_P=1.0,
        R0=55.38,
        mpsi=1.373,
        sigma_RM=108.251,
        rphi=33.76,
        mPhi=mPhi_kolm,
        chi_list=chi_list,
        nbins=100,
        kmin=3.0,
    )
