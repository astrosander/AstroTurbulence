import numpy as np
from main import (
    make_powerlaw_scalar_field_3d,
    make_solenoidal_vector_field_3d,
    make_lognormal_from_gaussian,
    intrinsic_polarization_from_density_and_Bperp,
    unit_polarization,
    spectrum_P_complex,
    spectrum_u_i,
    isotropic_structure_function_2d,
)

if __name__ == "__main__":
    rng = np.random.default_rng(1234)

    # Grid / box
    N = 356
    L = 1.0  # physical size (same in x,y,z here)

    # --- 1) Density cube with power-law spectrum ---
    beta_n = 11/3  # e.g. Kolmogorov-like 3D slope for a passive scalar (adjust as needed)
    g_n = make_powerlaw_scalar_field_3d(N, L, beta=beta_n, rng=rng)
    n = make_lognormal_from_gaussian(g_n, mean=1.0, sigma_ln=1.0)

    # --- 2) Perpendicular B-field cube with power-law spectrum ---
    beta_B = 11/3
    bx, by, bz = make_solenoidal_vector_field_3d(N, L, beta=beta_B, rng=rng)

    print(f"RMS before adding mean field:")
    print(f"  bx RMS: {np.sqrt(np.mean(bx**2)):.6e}")
    print(f"  by RMS: {np.sqrt(np.mean(by**2)):.6e}")
    print(f"  bz RMS: {np.sqrt(np.mean(bz**2)):.6e}")

    # --- 2a) Add mean magnetic field ---
    
    B0x = 4.0  # mean field in x direction
    B0y = 0.0  # mean field in y direction
    B0z = 0.0  # mean field in z direction (along LOS)
    bx = bx + B0x
    by = by + B0y
    bz = bz + B0z

    # --- 3) Build intrinsic complex polarization screen P_i(x,y) ---
    P_i = intrinsic_polarization_from_density_and_Bperp(
        n_cube=n,
        bx_cube=bx,
        by_cube=by,
        p0=0.7,
        alpha_B=2.0,  # emissivity dependence on B_perp
        alpha_n=1.0,  # emissivity dependence on density
        dz=L/N,
        evpa_from_B=True,
        chi_offset=0.0,
    )

    # --- 4) u_i(x,y) = P_i/|P_i| ---
    u_i = unit_polarization(P_i)

    # --- 4a) Mean subtraction for spectra (to avoid k=0 dominance) ---
    P_i_fluc = P_i - P_i.mean()
    u_i_fluc = u_i - u_i.mean()

    # --- 5) Spectra: complex P_i and u_i ---
    kP, Pk_amp = spectrum_P_complex(P_i_fluc, Lxy=L, nbins=50)
    ku, Pk_u = spectrum_u_i(u_i_fluc, Lxy=L, nbins=50)

    # --- 6) Structure functions directly ---
    rP, DP = isotropic_structure_function_2d(P_i, Lxy=L, nbins=50)
    ru, Du = isotropic_structure_function_2d(u_i, Lxy=L, nbins=50)

    # At this point, you can fit slopes in log-log space, plot, etc.
    print("Computed:")
    print(f"  |P_i| spectrum bins: {len(kP)}")
    print(f"  u_i spectrum bins:   {len(ku)}")
    print(f"  P_i structure bins:  {len(rP)}")
    print(f"  u_i structure bins:  {len(ru)}")

    # --- 7) Save only structure function and power spectrum data ---
    output_file = 'structure_functions_points_1400.npz'
    np.savez(
        output_file,
        # Parameters needed by plot_from_points.py
        N=N,
        L=L,
        # Structure function data
        rP=rP,
        DP=DP,
        ru=ru,
        Du=Du,
        # Power spectrum data
        kP=kP,
        Pk_amp=Pk_amp,
        ku=ku,
        Pk_u=Pk_u,
    )
    print(f"  Saved structure function and power spectrum data to {output_file}")

