import numpy as np
import os
from main import (
    isotropic_power_spectrum_2d,
    isotropic_structure_function_2d,
)

def generate_B0x_values(B0_min=0.0, B0_max=15.0):
    """
    Generate B0x values from 0 to 15 with more points near 1.
    Uses fine spacing around 1 and coarser spacing elsewhere.
    """
    # Fine spacing around 1 (below and above)
    # From 0 to 0.5: medium spacing
    B0_low = np.linspace(0.0, 0.5, 6)

    # From 0.5 to 1.0: fine spacing (below 1)
    B0_below1 = np.linspace(0.5, 0.95, 10)

    # From 1.0 to 1.5: very fine spacing (around 1, above 1)
    B0_around1 = np.linspace(1.0, 1.5, 12)

    # From 1.5 to 3.0: medium spacing (potential fast growth region)
    B0_growth = np.linspace(1.5, 3.0, 8)

    # From 3.0 to 15.0: coarser spacing
    B0_high = np.linspace(3.0, 15.0, 13)

    # Combine all and remove duplicates
    B0x_all = np.concatenate([B0_low, B0_below1, B0_around1, B0_growth, B0_high])
    B0x_unique = np.unique(B0x_all)

    # Sort and return
    return np.sort(B0x_unique)

if __name__ == "__main__":
    # Load data from 256.npz (read once)
    input_file = '1024_3.5.npz'
    print(f"Loading data from {input_file}...")
    data = np.load(input_file)

    # Extract B field components (original, will be modified in loop)
    bx_original = data['bx_cube'].copy()
    by_original = data['by_cube'].copy()
    bz_original = data['bz_cube'].copy()

    # Extract parameters
    N = int(data['N'])
    L = float(data['L'])

    print(f"Loaded B field cubes:")
    print(f"  Shape: {bx_original.shape}")
    print(f"  N: {N}, L: {L}")

    # Generate B0x values with more points near 1
    B0x_values = generate_B0x_values(0.0, 15.0)
    print(f"\nComputing spectra for {len(B0x_values)} B0x values:")
    print(f"  B0x range: {B0x_values[0]:.3f} to {B0x_values[-1]:.3f}")
    print(f"  Values near 1: {B0x_values[(B0x_values >= 0.9) & (B0x_values <= 1.1)]}")

    # Create output directory if it doesn't exist
    output_dir = 'spectrum'
    os.makedirs(output_dir, exist_ok=True)

    # Loop over B0x values
    for i, B0x in enumerate(B0x_values):
        print(f"\n[{i+1}/{len(B0x_values)}] Processing B0x = {B0x:.6f}")

        # Add mean magnetic field
        bx = bx_original + B0x
        by = by_original.copy()
        bz = bz_original.copy()

        # Compute B_perp^2 integrated along z (similar to how P_i is computed)
        # Use the complex representation like P_i
        # B_complex = (bx + 1j*by)**2 integrated along z
        B_complex_2d = np.sum((bx + 1j*by)**2, axis=2)

        # Use the complex representation for consistency with P_i calculation
        B_2d = B_complex_2d

        # Mean subtraction for spectra (to avoid k=0 dominance)
        B_2d_fluc = B_2d - B_2d.mean()

        # Compute power spectrum
        kB, Pk_B = isotropic_power_spectrum_2d(B_2d_fluc, Lxy=L, nbins=50)

        # Compute structure function
        rB, DB = isotropic_structure_function_2d(B_2d, Lxy=L, nbins=50)

        # Create unit B field (similar to u_i)
        B_unit = B_2d / (np.abs(B_2d) + 1e-30)
        B_unit_fluc = B_unit - B_unit.mean()

        # Compute spectrum of unit B field
        ku_B, Pk_u_B = isotropic_power_spectrum_2d(B_unit_fluc, Lxy=L, nbins=50)

        # Compute structure function of unit B field
        ru_B, Du_B = isotropic_structure_function_2d(B_unit, Lxy=L, nbins=50)

        # Create output filename with B0x value
        # Format: B0x value with appropriate precision
        B0x_str = f"{B0x:.6f}".rstrip('0').rstrip('.')
        output_file = os.path.join(output_dir, f'B0x_{B0x_str}.npz')

        # Save to npz file
        np.savez(
            output_file,
            # Parameters
            N=N,
            L=L,
            B0x=B0x,  # Save the B0x value used
            # Structure function data (using B field)
            rP=rB,
            DP=DB,
            ru=ru_B,
            Du=Du_B,
            # Power spectrum data (using B field)
            kP=kB,
            Pk_amp=Pk_B,
            ku=ku_B,
            Pk_u=Pk_u_B,
        )
        print(f"  Saved to {output_file}")

    print(f"\nCompleted! Processed {len(B0x_values)} files.")

