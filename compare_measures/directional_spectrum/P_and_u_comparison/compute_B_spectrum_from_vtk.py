import numpy as np
import os
import pyvista as pv
from pathlib import Path
from main import (
    isotropic_power_spectrum_2d,
    isotropic_structure_function_2d,
)

def _get_structured_array(mesh: pv.ImageData, name: str) -> np.ndarray:
    """
    Return a (nx, ny, nz) numpy array for scalar field `name` stored on VTK ImageData.
    Works for point or cell data; prefers point_data if both exist.
    Reshapes in Fortran order to match VTK memory layout.
    """
    nx, ny, nz = mesh.dimensions
    if name in mesh.point_data:
        arr = np.asarray(mesh.point_data[name]).reshape((nx, ny, nz), order="F")
        return arr.astype(np.float64)

    if name in mesh.cell_data:
        carr = np.asarray(mesh.cell_data[name]).reshape((nx - 1, ny - 1, nz - 1), order="F")
        cx, cy, cz = carr.shape
        out = np.empty((cx + 1, cy + 1, cz + 1), dtype=carr.dtype)

        # nearest-neighbor padding to points
        out[:cx, :cy, :cz] = carr
        out[cx, :cy, :cz] = carr[cx - 1, :, :]
        out[:cx, cy, :cz] = carr[:, cy - 1, :]
        out[:cx, :cy, cz] = carr[:, :, cz - 1]
        out[cx, cy, :cz] = carr[cx - 1, cy - 1, :]
        out[cx, :cy, cz] = carr[cx - 1, :, cz - 1]
        out[:cx, cy, cz] = carr[:, cy - 1, cz - 1]
        out[cx, cy, cz] = carr[cx - 1, cy - 1, cz - 1]
        return out.astype(np.float64)

    raise KeyError(
        f"'{name}' not found. Available: point={list(mesh.point_data.keys())}, cell={list(mesh.cell_data.keys())}"
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
    # VTK file paths - modify these as needed
    vtk_w_path = Path(r"D:\Downloads\1_11_2026\256_vtk\ms1ma2_256.mhd_w.00007.vtk")  # Update with actual path
    vtk_bcc_path = Path(r"D:\Downloads\1_11_2026\256_vtk\ms1ma2_256.mhd_bcc.00007.vtk")  # Update with actual path
    
    print(f"Reading VTK files...")
    print(f"  w_mesh: {vtk_w_path}")
    print(f"  b_mesh: {vtk_bcc_path}")
    
    # Read VTK files
    w_mesh = pv.read(str(vtk_w_path))
    b_mesh = pv.read(str(vtk_bcc_path))
    
    # Print available fields for debugging
    print("\nAvailable fields in w_mesh:")
    print(f"  point_data: {list(w_mesh.point_data.keys())}")
    print(f"  cell_data: {list(w_mesh.cell_data.keys())}")
    print("\nAvailable fields in b_mesh:")
    print(f"  point_data: {list(b_mesh.point_data.keys())}")
    print(f"  cell_data: {list(b_mesh.cell_data.keys())}")
    
    # Field names in VTK files - try common names
    # For w_mesh: try 'ne', 'dens', 'density', 'n'
    ne_candidates = ["ne", "dens", "density", "n"]
    ne_name = None
    for candidate in ne_candidates:
        if candidate in w_mesh.point_data or candidate in w_mesh.cell_data:
            ne_name = candidate
            break
    if ne_name is None:
        # Use first available field from w_mesh
        if w_mesh.cell_data.keys():
            ne_name = list(w_mesh.cell_data.keys())[0]
            print(f"Warning: Using '{ne_name}' as density field (not found in candidates)")
        else:
            raise KeyError(f"No suitable density field found. Available: {list(w_mesh.cell_data.keys())}")
    
    # For b_mesh: try various naming conventions
    # Common patterns: bx/by/bz, Bx/By/Bz, bcc1/bcc2/bcc3, b_x/b_y/b_z, etc.
    bx_candidates = ["bx", "Bx", "b_x", "B_x", "Bx_cc", "bx_cc", "bcc1", "Bcc1", "BCC1", "b1", "B1"]
    by_candidates = ["by", "By", "b_y", "B_y", "By_cc", "by_cc", "bcc2", "Bcc2", "BCC2", "b2", "B2"]
    bz_candidates = ["bz", "Bz", "b_z", "B_z", "Bz_cc", "bz_cc", "bcc3", "Bcc3", "BCC3", "b3", "B3"]
    
    def find_field(mesh, candidates):
        for candidate in candidates:
            if candidate in mesh.point_data or candidate in mesh.cell_data:
                return candidate
        # If not found, check all available fields with case-insensitive matching
        all_fields = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        for field in all_fields:
            field_lower = field.lower()
            for c in candidates:
                if c.lower() in field_lower or field_lower in c.lower():
                    return field
        return None
    
    bx_name = find_field(b_mesh, bx_candidates)
    by_name = find_field(b_mesh, by_candidates)
    bz_name = find_field(b_mesh, bz_candidates)
    
    # If still not found, try to infer from available fields
    if bx_name is None or by_name is None or bz_name is None:
        all_b_fields = list(b_mesh.point_data.keys()) + list(b_mesh.cell_data.keys())
        all_b_fields_sorted = sorted(all_b_fields)
        
        # Try common patterns: bcc1/bcc2/bcc3, b1/b2/b3, etc.
        if len(all_b_fields_sorted) >= 3:
            # Try to match by position or name pattern
            for i, field in enumerate(all_b_fields_sorted):
                field_lower = field.lower()
                if '1' in field_lower or 'x' in field_lower:
                    if bx_name is None:
                        bx_name = field
                elif '2' in field_lower or 'y' in field_lower:
                    if by_name is None:
                        by_name = field
                elif '3' in field_lower or 'z' in field_lower:
                    if bz_name is None:
                        bz_name = field
            
            # If still missing, assign by order (first three fields)
            if bx_name is None:
                bx_name = all_b_fields_sorted[0]
            if by_name is None:
                by_name = all_b_fields_sorted[1] if len(all_b_fields_sorted) > 1 else all_b_fields_sorted[0]
            if bz_name is None:
                bz_name = all_b_fields_sorted[2] if len(all_b_fields_sorted) > 2 else all_b_fields_sorted[0]
    
    if bx_name is None or by_name is None or bz_name is None:
        print(f"\nAvailable fields in b_mesh: {list(b_mesh.point_data.keys()) + list(b_mesh.cell_data.keys())}")
        raise KeyError(f"Could not find B field components. bx={bx_name}, by={by_name}, bz={bz_name}")
    
    print(f"\nUsing field names:")
    print(f"  ne/density: {ne_name}")
    print(f"  bx: {bx_name}")
    print(f"  by: {by_name}")
    print(f"  bz: {bz_name}")
    
    # Extract structured arrays
    print("\nExtracting structured arrays...")
    ne = _get_structured_array(w_mesh, ne_name)
    bx_original = _get_structured_array(b_mesh, bx_name)
    by_original = _get_structured_array(b_mesh, by_name)
    bz_original = _get_structured_array(b_mesh, bz_name)
    
    print(f"Loaded data:")
    print(f"  ne shape: {ne.shape}")
    print(f"  bx shape: {bx_original.shape}")
    print(f"  by shape: {by_original.shape}")
    print(f"  bz shape: {bz_original.shape}")
    
    # Compute mean magnetic field and RMS
    bx_mean = np.mean(bx_original)
    by_mean = np.mean(by_original)
    bz_mean = np.mean(bz_original)
    
    bx_rms = np.sqrt(np.mean(bx_original**2))
    by_rms = np.sqrt(np.mean(by_original**2))
    bz_rms = np.sqrt(np.mean(bz_original**2))
    
    print(f"\nMagnetic field statistics (before adding mean field):")
    print(f"  bx: mean = {bx_mean:.6e}, RMS = {bx_rms:.6e}")
    print(f"  by: mean = {by_mean:.6e}, RMS = {by_rms:.6e}")
    print(f"  bz: mean = {bz_mean:.6e}, RMS = {bz_rms:.6e}")
    
    # Get dimensions
    N = bx_original.shape[0]  # Assuming cubic grid
    # Get physical size from mesh bounds
    bounds = b_mesh.bounds
    L = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    
    print(f"\nGrid parameters:")
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

