import os
import sys
import numpy as np

# Make project root importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from compare_measures.pfa_and_derivative_lp_16_like import (
    FieldKeys,
    PFAConfig,
    load_fields,
    polarized_emissivity_lp16,
    faraday_density,
    sigma_phi_total,
    compute_faraday_regime,
    dir_slopes_vs_lambda,
    plot_dir_slopes_vs_lambda,
    fit_log_slope_with_bounds,
    H5_PATH as DEFAULT_H5_PATH,
)


def build_phi_and_pi(h5_path: str, cfg: PFAConfig):
    keys = FieldKeys()
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    Pi = polarized_emissivity_lp16(Bx, By, gamma=cfg.gamma)

    if cfg.los_axis == 0:
        Bpar = Bx
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz

    phi = faraday_density(ne, Bpar, C=1.0)

    if cfg.use_auto_faraday:
        Nz = Bx.shape[cfg.los_axis]
        dz = 1.0 / float(Nz)
        Phi_tot = np.sum(np.moveaxis(phi, cfg.los_axis, 0), axis=0) * dz
        sigma_Phi = float(Phi_tot.std())
        C = 1.0 / (max(cfg.lam2_break_target, 1e-30) * sigma_Phi) if sigma_Phi > 0 else 1.0
        phi *= C
        print(f"[dir] σΦ={sigma_Phi:.3g}, C={C:.3g}")
    else:
        phi *= cfg.faraday_const
        sigma_Phi = sigma_phi_total(phi, cfg.los_axis)
        print(f"[dir] Using faraday_const={cfg.faraday_const}; σΦ={sigma_Phi:.3g}")

    return Pi, phi, sigma_Phi


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Directional spectrum slopes vs lambda")
    parser.add_argument("--h5", default=DEFAULT_H5_PATH, help="Path to HDF5 with fields")
    parser.add_argument("--geometry", choices=["mixed", "separated"], default="separated")
    parser.add_argument("--every", type=int, default=2, help="Use every Nth lambda for speed")
    parser.add_argument("--ring_bins", type=int, default=48)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--k_min", type=float, default=6.0)
    parser.add_argument("--min_counts", type=int, default=10)
    parser.add_argument("--out", default=None, help="Output PNG path; defaults into lp2016_outputs/")
    parser.add_argument("--sweep", action="store_true", help="Run λ² sweep like run_spatial_psa_comparison.py")
    args = parser.parse_args()

    cfg = PFAConfig()

    print("Building Pi and phi...")
    Pi, phi, sigmaPhi0 = build_phi_and_pi(args.h5, cfg)
    sigmaPhi0=1.9101312160491943
    if args.sweep:
        # Use same settings as run_spatial_psa_comparison.py
        psa_kwargs = dict(ring_bins=48, pad=1, apodize=True, k_min=6.0, min_counts_per_ring=10)
        fit_bounds = dict(kmin=4, kmax=25)
        
        # Define λ² grid for sweep so that 0 ≤ 2(σ_φ λ²) ≤ 20
        # This implies χ = 2σ_φλ² spans [0, 20] on the x-axis
        # Ensure λ=0 is included (λ²=0, χ=0)
        lam2_max = 20.0 / (2.0 * sigmaPhi0)  # χ = 20
        n_points = 250
        # Create grid that includes λ²=0 and goes to lam2_max
        lam2_values = np.linspace(0.0, lam2_max, n_points)
        
        print(f"\nλ² sweep: 0.0 to {lam2_max:.3f} ({n_points} points)")
        print(f"Corresponding σ_φλ² range: 0.0 to {sigmaPhi0*lam2_max:.2f}")
        print(f"Corresponding χ=2σ_φλ² range: 0.0 to {2*sigmaPhi0*lam2_max:.2f}")
        
        # Initialize arrays to store results
        chi_values = []
        slopes_dir = []
        errors_dir = []
        
        # Sweep over λ² values
        print(f"\nComputing directional spectrum slopes for each λ²...")
        for i, lam2 in enumerate(lam2_values):
            print(i+1, "/", n_points)
            lam = np.sqrt(lam2)
            chi = 2.0 * sigmaPhi0 * lam2
            
            print(f"  λ²={lam2:.3f}, χ={chi:.2f} ({i+1}/{n_points})")
            
            # Build P_map for this λ
            if args.geometry == "mixed":
                from pfa_and_derivative_lp_16_like import P_map_mixed
                P_map = P_map_mixed(Pi, phi, lam, cfg)
            else:
                from pfa_and_derivative_lp_16_like import P_map_separated
                P_map = P_map_separated(Pi, phi, lam, cfg)
            
            # Compute directional spectrum
            from pfa_and_derivative_lp_16_like import directional_spectrum_of_map
            k_dir, Ek_dir = directional_spectrum_of_map(P_map, **psa_kwargs)
            
            # Fit slope
            mDir, aDir, eDir, _ = fit_log_slope_with_bounds(k_dir, Ek_dir, **fit_bounds)
            
            # Store results
            chi_values.append(chi)
            slopes_dir.append(mDir)
            errors_dir.append(eDir)
            
            print(f"    Directional slope: {mDir:.2f}±{eDir:.2f}")
        
        # Save results to NPZ
        output_dir = os.path.join(THIS_DIR, "lp2016_outputs")
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, f"dir_slopes_vs_chi_{args.geometry}_data.npz")
        np.savez(
            npz_path,
            lam2_values=np.array(lam2_values),
            chi_values=np.array(chi_values),
            slopes_dir=np.array(slopes_dir),
            errors_dir=np.array(errors_dir),
            psa_kwargs=psa_kwargs,
            fit_bounds=fit_bounds,
            sigmaPhi0=np.array(sigmaPhi0),
            faraday_const=np.array(cfg.faraday_const),
            los_axis=np.array(cfg.los_axis),
            gamma=np.array(cfg.gamma),
        )
        print(f"\nSaved sweep data to: {npz_path}")
        
        # Create slope vs χ plot
        print(f"\nCreating slope vs χ plot...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.errorbar(chi_values, slopes_dir, yerr=errors_dir, fmt='d-', capsize=3, 
                     label=f'Directional spectrum ({args.geometry})', color='green', alpha=0.7)
        plt.xlabel('χ = 2σ_φ λ²')
        plt.ylabel('Directional Spectrum Slope')
        plt.title(f'Directional Spectrum Slope vs χ ({args.geometry.capitalize()} Geometry, χ ∈ [0, 20])')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        slope_plot_path = os.path.join(output_dir, f"dir_slopes_vs_chi_{args.geometry}.png")
        plt.savefig(slope_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saving slope vs χ plot to: {slope_plot_path}")
        
        # Print summary
        print(f"\n" + "="*70)
        print(f"SUMMARY - DIRECTIONAL SPECTRUM SLOPES vs χ")
        print(f"="*70)
        print(f"Computed χ range: {min(chi_values):.2f} to {max(chi_values):.2f}")
        print(f"Number of computed points: {len(chi_values)}")
        print(f"Directional slopes range: {min(slopes_dir):.2f} to {max(slopes_dir):.2f}")
        
    else:
        # Original single plot mode - restrict to 0 ≤ 2(σ_φ λ²) ≤ 20
        # Ensure λ=0 is included (λ²=0, χ=0)
        lam2_max = 20.0 / (2.0 * sigmaPhi0)  # χ = 20
        lam2 = np.linspace(0.0, lam2_max, 250)  # uniform in λ² over the requested region, includes λ²=0
        cfg.lam_grid = tuple(np.sqrt(lam2))
        # Ensure λ=0 is always included, even when thinning
        lam_list = cfg.lam_grid[:: max(1, int(args.every))]
        if lam_list[0] != 0.0:  # If λ=0 was skipped due to thinning, add it back
            lam_list = [0.0] + list(lam_list)

        print(f"Computing directional-spectrum slopes vs lambda ({args.geometry}) ...")
        print(f"λ² range: 0.0 to {lam2_max:.3f}")
        print(f"Corresponding σ_φλ² range: 0.0 to {sigmaPhi0*lam2_max:.2f}")
        print(f"Corresponding χ=2σ_φλ² range: 0.0 to {2*sigmaPhi0*lam2_max:.2f}")
        print(f"Lambda list starts with: {lam_list[:3]} (should include 0.0)")
        
        lam_arr, mDir, eDir = dir_slopes_vs_lambda(
            Pi,
            phi,
            cfg,
            geometry=args.geometry,
            lam_list=lam_list,
            ring_bins=args.ring_bins,
            pad=args.pad,
            k_min=args.k_min,
            min_counts_per_ring=args.min_counts,
        )

        # Save results to NPZ for reproducibility
        out_dir = os.path.join(THIS_DIR, "lp2016_outputs")
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, f"dir_slopes_vs_lambda_{args.geometry}_data.npz")
        np.savez(
            npz_path,
            lam_arr=np.array(lam_arr),
            slopes_dir=np.array(mDir),
            errors_dir=np.array(eDir),
            lam2_values=np.array([lam**2 for lam in lam_arr]),
            chi_values=np.array([2*sigmaPhi0*lam**2 for lam in lam_arr]),
            ring_bins=np.array(args.ring_bins),
            pad=np.array(args.pad),
            k_min=np.array(args.k_min),
            min_counts_per_ring=np.array(args.min_counts),
            sigmaPhi0=np.array(sigmaPhi0),
            faraday_const=np.array(cfg.faraday_const),
            los_axis=np.array(cfg.los_axis),
            gamma=np.array(cfg.gamma),
            geometry=args.geometry,
        )
        print(f"Saved data to: {npz_path}")

        print("Plotting...")
        title = f"Directional-spectrum slope ({args.geometry.capitalize()}, χ ∈ [0, 20])"
        # Pass sigma_phi so x-axis is χ = 2σ_φλ²
        plot_dir_slopes_vs_lambda(lam_arr, mDir, eDir, x_is_lambda2=True, title=title, sigma_phi=sigmaPhi0)

        out_path = (
            args.out
            if args.out is not None
            else os.path.join(out_dir, f"dir_slopes_vs_lambda_{args.geometry}.png")
        )
        import matplotlib.pyplot as plt
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()


