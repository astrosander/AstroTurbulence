import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# Publication-ready font sizes
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 22


def Xi_i(R, A_P=1.0, R0=1.0, m_psi=2/3):
    return A_P / (1.0 + (R / R0) ** m_psi)


def f_saturating_powerlaw(R, Rb, m):
    x = (R / Rb) ** m
    return x / (1.0 + x)


def chi2_from_lambda(lam, sigma_phi2, chi2_factor=2.0):
    return chi2_factor * (lam ** 4) * sigma_phi2


def xi_P(R, A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2, chi2_factor=2.0):
    Xi = Xi_i(R, A_P=A_P, R0=R0, m_psi=m_psi)
    fphi = f_saturating_powerlaw(R, r_phi, m_phi)
    chi2 = chi2_from_lambda(lam, sigma_phi2, chi2_factor=chi2_factor)
    return Xi * np.exp(-chi2 * fphi)


def Du_half(R, A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2, chi2_factor=2.0):
    return A_P - xi_P(
        R, A_P=A_P, R0=R0, m_psi=m_psi,
        r_phi=r_phi, m_phi=m_phi,
        lam=lam, sigma_phi2=sigma_phi2,
        chi2_factor=chi2_factor
    )


def sf_spectrum_proxy(k, R, Du2, sigma_ln=0.35):
    lnR = np.log(R)
    dlnR = np.diff(lnR)

    ln_kR = np.log(k[:, None]) + lnR[None, :]

    W = np.exp(-(ln_kR ** 2) / (2.0 * sigma_ln ** 2)) / (math.sqrt(2.0 * math.pi) * sigma_ln)

    integrand = Du2[None, :] * W
    return np.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dlnR[None, :], axis=1)


def compute_scales(A_P, R0, m_psi, r_phi, m_phi, lam, sigma_phi2,
                   chi2_factor=2.0, eps=0.1, F=30.0, x_eff=1.0):
    chi2 = chi2_from_lambda(lam, sigma_phi2, chi2_factor=chi2_factor)

    R_psi = R0 * (eps ** (1.0 / m_psi))
    R_phi = r_phi * (eps ** (1.0 / m_phi))
    R_exp = r_phi * ((eps / chi2) ** (1.0 / m_phi)) if chi2 > 0 else np.inf

    R_asym = min(R_psi, R_phi, R_exp)
    k_asym = x_eff / R_asym

    k_inert_min = x_eff * max(1.0 / R0, 1.0 / r_phi)

    if abs(m_psi - m_phi) < 1e-12:
        R_x = np.nan
        k_x = np.nan
        delta = 0.0
    else:
        R_x = (chi2 * (R0 ** m_psi) / (r_phi ** m_phi)) ** (1.0 / (m_psi - m_phi))
        k_x = x_eff / R_x
        delta = m_phi - m_psi

    def k_at_ratio(target):
        return k_x * (target ** (-1.0 / delta))

    k_far_sure = (np.nan, np.nan)
    k_int_sure = (np.nan, np.nan)
    if np.isfinite(k_x) and abs(delta) > 1e-12:
        kF = k_at_ratio(F)
        kInvF = k_at_ratio(1.0 / F)
        k_lo = min(kF, kInvF)
        k_hi = max(kF, kInvF)

        k_test = 10.0 * k_x
        W_test = (k_x / k_test) ** delta

        if W_test < 1.0:
            k_far_sure = (0.0, k_lo)
            k_int_sure = (k_hi, np.inf)
        else:
            k_far_sure = (k_hi, np.inf)
            k_int_sure = (0.0, k_lo)

        def intersect_with_asym(interval):
            lo, hi = interval
            lo = max(lo, k_asym)
            if hi < lo:
                return (np.nan, np.nan)
            return (lo, hi)

        k_far_sure = intersect_with_asym(k_far_sure)
        k_int_sure = intersect_with_asym(k_int_sure)

    return dict(
        chi2=chi2,
        eps=eps,
        F=F,
        x_eff=x_eff,
        R_psi=R_psi,
        R_phi=R_phi,
        R_exp=R_exp,
        R_asym=R_asym,
        R_x=R_x,
        delta=delta,
        k_inert_min=k_inert_min,
        k_asym=k_asym,
        k_x=k_x,
        k_far_sure=k_far_sure,
        k_int_sure=k_int_sure,
    )


def suggest_sigma_phi2_for_kx(kx_desired, R0, m_psi, r_phi, m_phi, lam,
                              chi2_factor=2.0):
    R_x = 1.0 / kx_desired
    if abs(m_psi - m_phi) < 1e-12:
        raise ValueError("m_psi == m_phi: crossover is prefactor-only; kx is not set by slope balance.")
    chi2 = (r_phi ** m_phi / R0 ** m_psi) * (R_x ** (m_psi - m_phi))
    return chi2 / (chi2_factor * lam ** 4)
def run_and_plot(params,
                 Rmin=1e-5, Rmax=1e3, NR=1400,
                 Nk=900,
                 eps=0.1, F=30.0,
                 sigma_ln=0.35,
                 out_prefix="sf_run",
                 save_scales=True,
                 show=True,
                 sigma_phi2_list=None):
    A_P = params["A_P"]
    R0 = params["R0"]
    m_psi = params["m_psi"]
    r_phi = params["r_phi"]
    m_phi = params["m_phi"]
    lam = params["lam"]
    sigma_phi2_default = params["sigma_phi2"]
    chi2_factor = params.get("chi2_factor", 2.0)

    # If sigma_phi2_list is provided, use it; otherwise use single value
    if sigma_phi2_list is None:
        sigma_phi2_list = [sigma_phi2_default]

    R = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)
    kmin = 1.0 / Rmax
    kmax = 1.0 / Rmin
    k = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

    # Compute all chi values first to create a logical color mapping
    chi_values = []
    for sigma_phi2 in sigma_phi2_list:
        chi2 = chi2_from_lambda(lam, sigma_phi2, chi2_factor=chi2_factor)
        chi_values.append(np.sqrt(chi2))
    
    chi_values = np.array(chi_values)
    chi_min, chi_max = chi_values.min(), chi_values.max()
    
    # Create a sequential colormap with highly distinct, professional colors
    # Top-tier ApJ style: maximum contrast, publication-quality palette
    # Transitions from deep cool colors to vibrant warm colors
    from matplotlib.colors import LinearSegmentedColormap
    # Nobel Prize level palette: very distinct, high contrast, memorable
    modern_colors = [
        '#003366',  # Deep Navy Blue - authoritative start
        '#0066CC',  # Royal Blue - clear and professional
        '#6600CC',  # Deep Purple - distinctive transition
        '#CC0066',  # Deep Magenta - strong and memorable
        '#FF3300',  # Vibrant Red-Orange - high impact finish
    ]
    cmap_chi = LinearSegmentedColormap.from_list('chi_palette', modern_colors, N=256)
    
    # Normalize chi values to [0, 1] for colormap
    if chi_max > chi_min:
        chi_normalized = (chi_values - chi_min) / (chi_max - chi_min)
    else:
        chi_normalized = np.ones_like(chi_values) * 0.5
    
    # Map chi values to colors
    colors = [cmap_chi(val) for val in chi_normalized]

    fig1, ax1 = plt.subplots(figsize=(7.6, 5.0))
    fig2, ax2 = plt.subplots(figsize=(7.6, 5.0))

    all_scales = []
    all_M = []
    
    # Set ylim early for consistent label positioning
    ylim_bottom = 1e-5
    ylim_top = 1e0
    
    # Helper function to mark R values with proper label positioning
    def mark_R(v, color, ls, text):
        if np.isfinite(v) and (Rmin < v < Rmax):
            # ax1.axvline(v, color=color, ls=ls, lw=3)
            # Position label at bottom of plot in data coordinates
            # Use a small offset from bottom ylim for log scale
            y_pos = ylim_bottom * 1.5  # Slightly above bottom for visibility
            # ax1.text(v, y_pos, text, rotation=90,
            #          va="bottom", ha="right", color=color, fontsize=16)
    
    # Nobel Prize / Top-tier ApJ color palette - highly distinct and professional
    # Inspired by leading astrophysics publications with maximum visual impact
    color_m_psi = "#0066CC"      # Deep Royal Blue - authoritative and clear
    color_m_phi = "#CC0066"      # Deep Magenta - strong contrast, memorable
    color_r_phi = "#6600CC"      # Deep Purple - distinctive and professional
    color_r_i = "#FF3300"        # Vibrant Red-Orange - high visibility
    color_faraday = "#0099CC"    # Cyan Blue - clear and distinct
    
    # Add common reference lines that don't depend on chi2
    # Top-tier ApJ style: prominent, high-visibility reference lines
    # m_psi line in one color
    Du2_int = A_P * (R / R0) ** m_psi
    ax1.loglog(R, Du2_int, ls="--", lw=3.5, alpha=0.85, color=color_m_psi, 
               label=rf"$\propto R^{{m_\psi}}$", zorder=5)
    
    # Mark r_phi (common to all curves)
    mark_R(r_phi, color_r_phi, "-.", r"$r_\phi$")
    
    # Reference k lines (will be added after we have M values)
    k_anchor = math.sqrt(kmin * kmax)
    
    # Compute first curve's chi2 to use for single m_phi reference line
    first_sigma_phi2 = sigma_phi2_list[0]
    first_xi = xi_P(R, A_P=A_P, R0=R0, m_psi=m_psi,
                    r_phi=r_phi, m_phi=m_phi,
                    lam=lam, sigma_phi2=first_sigma_phi2,
                    chi2_factor=chi2_factor)
    first_scales = compute_scales(
        A_P=A_P, R0=R0, m_psi=m_psi,
        r_phi=r_phi, m_phi=m_phi,
        lam=lam, sigma_phi2=first_sigma_phi2,
        chi2_factor=chi2_factor,
        eps=eps, F=F,
        x_eff=1.0
    )
    first_chi2 = first_scales["chi2"]
    
    # Add single m_phi reference line using first curve's chi2
    Du2_far = A_P * first_chi2 * (R / r_phi) ** m_phi
    ax1.loglog(R, Du2_far, ls="--", lw=3.5, alpha=0.85, color=color_m_phi, 
               label=rf"$\propto R^{{m_\Phi}}$", zorder=5)
    
    for idx, sigma_phi2 in enumerate(sigma_phi2_list):
        Xi = Xi_i(R, A_P=A_P, R0=R0, m_psi=m_psi)
        xi = xi_P(R, A_P=A_P, R0=R0, m_psi=m_psi,
                  r_phi=r_phi, m_phi=m_phi,
                  lam=lam, sigma_phi2=sigma_phi2,
                  chi2_factor=chi2_factor)
        Du2 = A_P - xi

        M = sf_spectrum_proxy(k, R, Du2, sigma_ln=sigma_ln)
        all_M.append(M)

        scales = compute_scales(
            A_P=A_P, R0=R0, m_psi=m_psi,
            r_phi=r_phi, m_phi=m_phi,
            lam=lam, sigma_phi2=sigma_phi2,
            chi2_factor=chi2_factor,
            eps=eps, F=F,
            x_eff=1.0
        )
        all_scales.append(scales)

        chi2 = scales["chi2"]
        chi = np.sqrt(chi2)
        color = colors[idx]
        
        # Plot Du2 curves
        label_chi = rf"$\eta={chi:.1g}$" if len(sigma_phi2_list) > 1 else r"$\frac{D_u(R;\lambda)}{2}$"
        ax1.loglog(R, Du2, color=color, lw=3, label=label_chi)
        
        # Mark chi2-dependent R scales (with reduced alpha for multiple curves)
        alpha_mark = 0.5 if len(sigma_phi2_list) > 1 else 1.0
        # Only add text labels for first curve to avoid overlap
        if idx == 0:
            mark_R(scales["R_asym"], color_r_i, ":", r"$r_{i}$")
        # else:
            # Just add vertical lines without text for subsequent curves
            # if np.isfinite(scales["R_asym"]) and (Rmin < scales["R_asym"] < Rmax):
            #     ax1.axvline(scales["R_asym"], color=color_r_i, ls=":", lw=3, alpha=alpha_mark)
        
        # Mark R_x as a point on the curve instead of vertical line
        if np.isfinite(scales["R_x"]) and (Rmin < scales["R_x"] < Rmax):
            # Find Du2 value at R_x by interpolation
            R_x = scales["R_x"]
            # Find closest R value or interpolate
            idx_closest = np.argmin(np.abs(R - R_x))
            Du2_at_Rx = Du2[idx_closest]
            # Nobel Prize-level star design: elegant, sophisticated, publication-quality
            # Multi-layer design for maximum visual impact and depth
            # Layer 1: Deep shadow for 3D effect (slightly offset)
            ax1.plot(R_x * 1.002, Du2_at_Rx * 0.998, '*', color='black', markersize=28,
                    markeredgecolor='black', markeredgewidth=0, alpha=0.4,
                    markerfacecolor='black', zorder=8)
            # Layer 2: Main shadow base
            ax1.plot(R_x, Du2_at_Rx, '*', color='black', markersize=26,
                    markeredgecolor='black', markeredgewidth=0,
                    markerfacecolor='black', zorder=9)
            # Layer 3: Colored star with elegant black border
            ax1.plot(R_x, Du2_at_Rx, '*', color=color, markersize=22,
                    markeredgecolor='black', markeredgewidth=1.2,
                    markerfacecolor=color, zorder=10, alpha=0.95)
            # Layer 4: Bright center highlight for depth and elegance
            ax1.plot(R_x, Du2_at_Rx, '*', color='white', markersize=11,
                    markeredgecolor='none', markeredgewidth=0,
                    markerfacecolor='white', zorder=11, alpha=0.9)
            # Add label only for first curve
            # if idx == 0:
            #     ax1.text(R_x, Du2_at_Rx, r"$R_\times$", color=color, 
            #             fontsize=16, ha="left", va="bottom")
        
        # Plot M curves
        label_M = rf"$\eta={chi:.1g}$" if len(sigma_phi2_list) > 1 else r"$\mathcal{M}(k)$"
        ax2.loglog(k, M, color=color, lw=3, label=label_M)
        
        # Add reference k lines (only once, using first M)
        # Top-tier ApJ style: prominent, high-visibility reference lines
        if idx == 0:
            y_anchor = M[np.argmin(np.abs(k - k_anchor))]
            ax2.loglog(k, y_anchor * (k / k_anchor) ** (-m_psi),
                       ls="--", lw=3.5, alpha=0.85, color=color_m_psi, 
                       label=rf"$k^{{m_\psi}}$", zorder=5)
            ax2.loglog(k, y_anchor * (k / k_anchor) ** (-m_phi),
                       ls="--", lw=3.5, alpha=0.85, color=color_m_phi, 
                       label=rf"$k^{{m_\Phi}}$", zorder=5)
        
        # Mark k scales (with reduced alpha for multiple curves)
        # ax2.axvline(scales["k_inert_min"], color=color_r_i, ls=":", lw=3, alpha=alpha_mark, 
        #            label=r"$k_{\rm inert,min}$" if idx == 0 else "")
        # ax2.axvline(scales["k_asym"], color=color_r_i, ls="--", lw=3, alpha=alpha_mark,
        #            label=r"$k_{i}$" if idx == 0 else "")
        
        # Mark k_x as a point on the curve instead of vertical line
        if np.isfinite(scales["k_x"]) and (kmin < scales["k_x"] < kmax):
            k_x = scales["k_x"]
            # Find closest k value
            idx_closest = np.argmin(np.abs(k - k_x))
            M_at_kx = M[idx_closest]
            # Nobel Prize-level star design: elegant, sophisticated, publication-quality
            # Multi-layer design for maximum visual impact and depth
            # Layer 1: Deep shadow for 3D effect (slightly offset)
            ax2.plot(k_x * 1.002, M_at_kx * 0.998, '*', color='black', markersize=28,
                    markeredgecolor='black', markeredgewidth=0, alpha=0.4,
                    markerfacecolor='black', zorder=8)
            # Layer 2: Main shadow base
            ax2.plot(k_x, M_at_kx, '*', color='black', markersize=26,
                    markeredgecolor='black', markeredgewidth=0,
                    markerfacecolor='black', zorder=9)
            # Layer 3: Colored star with elegant black border
            ax2.plot(k_x, M_at_kx, '*', color=color, markersize=22,
                    markeredgecolor='black', markeredgewidth=1.2,
                    markerfacecolor=color, zorder=10, alpha=0.95)
            # Layer 4: Bright center highlight for depth and elegance
            ax2.plot(k_x, M_at_kx, '*', color='white', markersize=11,
                    markeredgecolor='none', markeredgewidth=0,
                    markerfacecolor='white', zorder=11, alpha=0.9)
            # Add label only for first curve
            # if idx == 0:
            #     ax2.text(k_x, M_at_kx, r"$k_\times$", color=color,
            #             fontsize=16, ha="left", va="bottom")

        # Add "Faraday sure" and "Intrinsic sure" regions
        # for name, (lo, hi) in [("Faraday sure", scales["k_far_sure"]),
        #                        ("Intrinsic sure", scales["k_int_sure"])]:
        #     for v in (lo, hi):
        #         if np.isfinite(v) and (kmin < v < kmax):
        #             ax2.axvline(v, color=color_faraday, ls="--", lw=0.9, alpha=0.3)

    ax1.set_xlim(Rmin*10, Rmax*0.1)
    ax1.set_ylim(ylim_bottom, ylim_top)
    ax1.set_xlabel(r"$R$")
    ax1.set_ylabel(r"$D_u/2$")
    ax1.legend(loc="best", fontsize=18)
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_realspace.png", dpi=220)
    fig1.savefig(f"{out_prefix}_realspace.svg", dpi=220)
    print(f"Saved {out_prefix}_realspace.svg")

    ax2.set_xlim(kmin*10, kmax*0.1)
    ax2.set_ylim(1e-5, 1e0)
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$\mathcal{M}(k)$")
    ax2.legend(loc="best", fontsize=18)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_kproxy.png", dpi=220)
    fig2.savefig(f"{out_prefix}_kproxy.svg", dpi=220)
    print(f"Saved {out_prefix}_kproxy.svg")

    if show:
        pass
        # plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return all_scales[0] if len(all_scales) == 1 else all_scales


if __name__ == "__main__":
    params = dict(
        A_P=1.0,
        R0=1.0,
        m_psi=5/3,
        r_phi=1.0,
        m_phi=1/2,#1/2,
        lam=1,
        sigma_phi2=0.01,  # Default value (used if sigma_phi2_list not provided)
        chi2_factor=4.0,
    )

    # Generate more sequential chi values with denser spacing for smaller chi
    # Use geometric spacing to make it denser at lower values
    n_curves = 5
    chi_min_target = 0.05  # Minimum chi value
    chi_max_target = 0.5     # Maximum chi value
    
    # Generate geometrically spaced chi values (denser at lower values)
    chi_targets = np.geomspace(chi_min_target, chi_max_target, n_curves)
    
    # Compute corresponding sigma_phi2 values
    # chi = sqrt(chi2_factor * lam^4 * sigma_phi2)
    # so sigma_phi2 = chi^2 / (chi2_factor * lam^4)
    chi2_factor = params["chi2_factor"]
    lam = params["lam"]
    sigma_phi2_list = [chi**2 / (chi2_factor * lam**4) for chi in chi_targets]
    
    print(f"Generated sequential chi values (geometric spacing): {[f'{chi:.1g}' for chi in chi_targets]}")
    print(f"Corresponding sigma_phi2: {[f'{s:.4f}' for s in sigma_phi2_list]}")

    scales = run_and_plot(
        params,
        Rmin=1e-5, Rmax=1e2, NR=1400,
        Nk=900,
        eps=1.0,  # Set to 1.0 so that R_phi = 1 and R_psi = 1, making R_asym = 1
        F=30.0,
        sigma_ln=0.35,
        out_prefix="sf_demo",
        save_scales=True,
        show=True,
        sigma_phi2_list=sigma_phi2_list
    )

    print("\nComputed scales:")
    if isinstance(scales, list):
        for i, scale in enumerate(scales):
            print(f"\nCurve {i+1} (sigma_phi2={sigma_phi2_list[i]:.4f}):")
            for k, v in scale.items():
                print(f"  {k:>12s} : {v}")
    else:
        for k, v in scales.items():
            print(f"{k:>12s} : {v}")
