import numpy as np
import os
import h5py
from pathlib import Path
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
    # HDF5 file path - modify as needed
    h5_file_path = Path(r"D:\Рабочая папка\GitHub\AstroTurbulence\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5")  # Update with actual path
    
    print(f"Reading HDF5 file: {h5_file_path}")
    
    # Read HDF5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Print available keys for debugging
        print("\nAvailable keys in HDF5 file:")
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_keys)
        
        # Extract field names
        # Expected: gas_density, i_mag_field, j_mag_field, k_mag_field
        density_key = "gas_density"
        bx_key = "i_mag_field"
        by_key = "j_mag_field"
        bz_key = "k_mag_field"
        
        # Check if keys exist
        available_keys = list(f.keys())
        print(f"\nRoot keys: {available_keys}")
        
        if density_key not in f:
            print(f"Warning: '{density_key}' not found. Available keys: {available_keys}")
            # Try to find density field
            for key in available_keys:
                if 'density' in key.lower() or 'dens' in key.lower():
                    density_key = key
                    print(f"Using '{density_key}' as density field")
                    break
        
        if bx_key not in f or by_key not in f or bz_key not in f:
            print(f"Warning: B field keys not found. Looking for alternatives...")
            # Try to find B field components
            for key in available_keys:
                key_lower = key.lower()
                if 'mag' in key_lower or 'field' in key_lower:
                    if 'i' in key_lower or 'x' in key_lower:
                        if bx_key not in f:
                            bx_key = key
                            print(f"Using '{bx_key}' as bx field")
                    elif 'j' in key_lower or 'y' in key_lower:
                        if by_key not in f:
                            by_key = key
                            print(f"Using '{by_key}' as by field")
                    elif 'k' in key_lower or 'z' in key_lower:
                        if bz_key not in f:
                            bz_key = key
                            print(f"Using '{bz_key}' as bz field")
        
        # Extract arrays
        print(f"\nExtracting arrays...")
        print(f"  density: {density_key}")
        print(f"  bx: {bx_key}")
        print(f"  by: {by_key}")
        print(f"  bz: {bz_key}")
        
        ne = np.array(f[density_key])
        bx_original = np.array(f[bx_key])
        by_original = np.array(f[by_key])
        bz_original = np.array(f[bz_key])
        
        # Get coordinates to determine physical size
        if "x_coor" in f and "y_coor" in f and "z_coor" in f:
            x_coor = np.array(f["x_coor"])
            y_coor = np.array(f["y_coor"])
            z_coor = np.array(f["z_coor"])
            L = max(x_coor.max() - x_coor.min(), 
                   y_coor.max() - y_coor.min(), 
                   z_coor.max() - z_coor.min())
        else:
            # Fallback: assume unit size or estimate from array shape
            L = 1.0
            print("Warning: Coordinates not found. Using L=1.0")
    
    print(f"\nLoaded data:")
    print(f"  ne shape: {ne.shape}")
    print(f"  bx shape: {bx_original.shape}")
    print(f"  by shape: {by_original.shape}")
    print(f"  bz shape: {bz_original.shape}")
    
    # Get dimensions
    N = bx_original.shape[0]  # Assuming cubic grid
    print(f"  N: {N}, L: {L:.6f}")
    
    # Generate B0x values with more points near 1
    B0x_values = generate_B0x_values(0.0, 15.0)
    print(f"\nComputing spectra for {len(B0x_values)} B0x values:")
    print(f"  B0x range: {B0x_values[0]:.3f} to {B0x_values[-1]:.3f}")
    print(f"  Values near 1: {B0x_values[(B0x_values >= 0.9) & (B0x_values <= 1.1)]}")
    
    # Create output directory if it doesn't exist
    output_dir = 'spectrum_vtk(hu)'
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

