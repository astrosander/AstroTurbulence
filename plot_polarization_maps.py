#!/usr/bin/env python3
"""
Plot polarization maps for separated and mixed geometries.

This script loads an HDF5 file and creates side-by-side plots of:
1. Separated screen geometry (external Faraday screen)
2. Mixed geometry (emission and rotation co-exist)

Can generate single frames or animation frames for a series of χ values.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

# Add the compare_measures directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'compare_measures'))

# Import required functions
from pfa_and_derivative_lp_16_like import (
    load_fields,
    polarized_emissivity_simple,
    faraday_density,
    P_map_mixed,
    PFAConfig,
    FieldKeys
)

def P_map_separated(Pi, phi, lam, cfg, emit_bounds=None, screen_bounds=None):
    """
    External screen: P(X,λ) = [∫_emit Pi dz] * exp{ 2i λ^2 Φ_screen(X) }.
    |P| does NOT depend on λ; all λ-dependence is in the phase.
    """
    Pi_los  = np.moveaxis(Pi,  cfg.los_axis, 0)
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz, Ny, Nx = Pi_los.shape

    # default: take front 10% as screen, rest as emitter
    if screen_bounds is None:
        scr_N = max(1, int(0.10 * Nz))
        screen_bounds = (0, scr_N)
    if emit_bounds is None:
        emit_bounds = (screen_bounds[1], Nz)

    z0e, z1e = emit_bounds
    z0s, z1s = screen_bounds

    # emission (no internal rotation)
    P_emit = Pi_los[z0e:z1e].sum(axis=0)
    # screen RM
    Phi_screen = phi_los[z0s:z1s].sum(axis=0)
    return P_emit * np.exp(2j * (lam**2) * Phi_screen)


def plot_polarization_maps(h5_path=None, lam=1.0, chi=None, output_path=None, 
                           save_frames=False, frames_dir=None):
    """
    Load HDF5 file and plot polarization maps for both geometries.
    
    Parameters:
    -----------
    h5_path : str, optional
        Path to HDF5 file. If None, will try common locations.
    lam : float
        Wavelength value (default: 1.0). Ignored if chi is provided.
    chi : float, optional
        χ = 2*σ_Φ*λ² value. If provided, lam will be computed from chi.
    output_path : str, optional
        Path to save the plot. If None, saves in compare_measures/lp2016_outputs/
    save_frames : bool
        If True, saves frame with frame number in filename
    frames_dir : str, optional
        Directory to save frames (only used if save_frames=True)
    """
    # Try different possible paths
    if h5_path is None:
        possible_paths = [
            r"faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
            r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
            os.path.join(os.path.dirname(__file__), "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
            os.path.join(os.path.dirname(__file__), "..", "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
        ]
        
        h5_path = None
        for path in possible_paths:
            if os.path.exists(path):
                h5_path = path
                break
        
        if h5_path is None:
            raise FileNotFoundError(f"Could not find HDF5 file. Tried: {possible_paths}")
    
    print(f"Loading MHD fields from: {h5_path}")
    
    # Load the data
    keys = FieldKeys()
    cfg = PFAConfig()
    
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    print(f"Loaded fields with shape: Bx={Bx.shape}, By={By.shape}, Bz={Bz.shape}, ne={ne.shape}")
    
    # Set configuration
    cfg.los_axis = 2  # z-axis is LOS
    gamma = cfg.gamma = 2.0
    cfg.faraday_const = 1.0
    
    # Choose B_parallel based on LOS axis
    if cfg.los_axis == 0:
        Bpar = Bx
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz
    
    # Compute polarization emissivity
    Pi = polarized_emissivity_simple(Bx, By, gamma=gamma)
    print(f"Computed polarization emissivity with shape: {Pi.shape}")
    
    # Compute Faraday density
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)
    print(f"Computed Faraday density with shape: {phi.shape}")
    
    # Set up separated geometry bounds
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz = phi_los.shape[0]
    scr_N = max(1, int(0.10 * Nz))
    screen_bounds = (0, scr_N)
    emit_bounds = (scr_N, Nz)
    
    # Compute σ_Φ from screen only (for separated case)
    Phi_screen = phi_los[screen_bounds[0]:screen_bounds[1]].sum(axis=0)
    sigmaPhi_screen = float(Phi_screen.std())
    
    # Fixed sigma for mixed case
    sigmaPhi0_mixed = 1.9101312160491943
    
    # If chi is provided, compute lambda from chi
    # For separated: use sigmaPhi_screen; for mixed: use sigmaPhi0_mixed
    # We'll use separated chi for the primary calculation
    if chi is not None:
        # chi = 2 * sigmaPhi_screen * cfg.faraday_const * lam^2 (for separated)
        lam2 = chi / (2.0 * sigmaPhi_screen * cfg.faraday_const)
        lam = np.sqrt(lam2)
        
        # Also compute chi_mixed for reference
        chi_mixed = 2.0 * sigmaPhi0_mixed * cfg.faraday_const * (lam**2)
        print(f"\nUsing χ = {chi:.3f} (separated, based on σ_Φ(screen) = {sigmaPhi_screen:.3f})")
        print(f"  Computed λ = {lam:.6f}")
        print(f"  χ_mixed (based on σ_Φ0 = {sigmaPhi0_mixed:.3f}) = {chi_mixed:.3f}")
    else:
        # Compute chi from lambda
        chi = 2.0 * sigmaPhi_screen * cfg.faraday_const * (lam**2)
        chi_mixed = 2.0 * sigmaPhi0_mixed * cfg.faraday_const * (lam**2)
        print(f"\nUsing λ = {lam:.6f}")
        print(f"  χ_separated (σ_Φ(screen) = {sigmaPhi_screen:.3f}) = {chi:.3f}")
        print(f"  χ_mixed (σ_Φ0 = {sigmaPhi0_mixed:.3f}) = {chi_mixed:.3f}")
    
    print(f"Separated geometry: screen bounds = {screen_bounds}, emit bounds = {emit_bounds}")
    
    # Compute P maps for both geometries
    P_separated = P_map_separated(Pi, phi, lam, cfg, 
                                   emit_bounds=emit_bounds, 
                                   screen_bounds=screen_bounds)
    P_mixed = P_map_mixed(Pi, phi, lam, cfg)
    
    print(f"P_separated shape: {P_separated.shape}")
    print(f"P_mixed shape: {P_mixed.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(17.7777777778, 10))
    
    # Determine regime and color based on chi
    if chi < 1.0:
        regime = "Synchrotron-dominated"
        regime_color = 'green'
    elif chi < 3.0:
        regime = "Transitional"
        regime_color = 'yellow'
    else:
        regime = "Faraday-dominated"
        regime_color = 'red'
    
    title = f'Polarization Maps Comparison ($\\chi = {chi:.3f}$) - {regime}'
    fig.suptitle(title, fontsize=28, fontweight='bold', color=regime_color)
    
    # Modern colormaps - rainbow for magnitude, seismic for complex parts
    cmap_magnitude = 'turbo'  # Modern rainbow colormap, excellent for presentations
    cmap_complex = 'seismic'   # Modern alternative to RdBu_r
    
    # Plot 1: Separated - Magnitude
    im1 = axes[0, 0].imshow(np.abs(P_separated), origin='lower', cmap=cmap_magnitude, aspect='auto')
    axes[0, 0].set_title('Separated: $|P|$ (Magnitude)', fontsize=24, fontweight='bold')
    axes[0, 0].set_xlabel('$X$', fontsize=20)
    axes[0, 0].set_ylabel('$Y$', fontsize=20)
    axes[0, 0].tick_params(labelsize=18)
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('$|P|$', fontsize=20)
    cbar1.ax.tick_params(labelsize=18)
    
    # Plot 2: Separated - Real part
    im2 = axes[0, 1].imshow(np.real(P_separated), origin='lower', cmap=cmap_complex, aspect='auto')
    axes[0, 1].set_title('Separated: $\\mathrm{Re}(P)$', fontsize=24, fontweight='bold')
    axes[0, 1].set_xlabel('$X$', fontsize=20)
    axes[0, 1].set_ylabel('$Y$', fontsize=20)
    axes[0, 1].tick_params(labelsize=18)
    cbar2 = plt.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('$\\mathrm{Re}(P)$', fontsize=20)
    cbar2.ax.tick_params(labelsize=18)
    
    # Plot 3: Separated - Imaginary part
    im3 = axes[0, 2].imshow(np.imag(P_separated), origin='lower', cmap=cmap_complex, aspect='auto')
    axes[0, 2].set_title('Separated: $\\mathrm{Im}(P)$', fontsize=24, fontweight='bold')
    axes[0, 2].set_xlabel('$X$', fontsize=20)
    axes[0, 2].set_ylabel('$Y$', fontsize=20)
    axes[0, 2].tick_params(labelsize=18)
    cbar3 = plt.colorbar(im3, ax=axes[0, 2])
    cbar3.set_label('$\\mathrm{Im}(P)$', fontsize=20)
    cbar3.ax.tick_params(labelsize=18)
    
    # Plot 4: Mixed - Magnitude
    im4 = axes[1, 0].imshow(np.abs(P_mixed), origin='lower', cmap=cmap_magnitude, aspect='auto')
    axes[1, 0].set_title('Mixed: $|P|$ (Magnitude)', fontsize=24, fontweight='bold')
    axes[1, 0].set_xlabel('$X$', fontsize=20)
    axes[1, 0].set_ylabel('$Y$', fontsize=20)
    axes[1, 0].tick_params(labelsize=18)
    cbar4 = plt.colorbar(im4, ax=axes[1, 0])
    cbar4.set_label('$|P|$', fontsize=20)
    cbar4.ax.tick_params(labelsize=18)
    
    # Plot 5: Mixed - Real part
    im5 = axes[1, 1].imshow(np.real(P_mixed), origin='lower', cmap=cmap_complex, aspect='auto')
    axes[1, 1].set_title('Mixed: $\\mathrm{Re}(P)$', fontsize=24, fontweight='bold')
    axes[1, 1].set_xlabel('$X$', fontsize=20)
    axes[1, 1].set_ylabel('$Y$', fontsize=20)
    axes[1, 1].tick_params(labelsize=18)
    cbar5 = plt.colorbar(im5, ax=axes[1, 1])
    cbar5.set_label('$\\mathrm{Re}(P)$', fontsize=20)
    cbar5.ax.tick_params(labelsize=18)
    
    # Plot 6: Mixed - Imaginary part
    im6 = axes[1, 2].imshow(np.imag(P_mixed), origin='lower', cmap=cmap_complex, aspect='auto')
    axes[1, 2].set_title('Mixed: $\\mathrm{Im}(P)$', fontsize=24, fontweight='bold')
    axes[1, 2].set_xlabel('$X$', fontsize=20)
    axes[1, 2].set_ylabel('$Y$', fontsize=20)
    axes[1, 2].tick_params(labelsize=18)
    cbar6 = plt.colorbar(im6, ax=axes[1, 2])
    cbar6.set_label('$\\mathrm{Im}(P)$', fontsize=20)
    cbar6.ax.tick_params(labelsize=18)
    
    plt.tight_layout()
    
    # Save the plot
    if save_frames and frames_dir is not None:
        # Save frame with frame number in filename
        frame_num = getattr(plot_polarization_maps, '_frame_counter', 0)
        output_path = os.path.join(frames_dir, f"frame_{frame_num:04d}_chi_{chi:.3f}.png")
        plot_polarization_maps._frame_counter = frame_num + 1
    elif output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), "compare_measures", "lp2016_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"polarization_maps_comparison_chi_{chi:.3f}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"STATISTICS")
    print(f"{'='*60}")
    print(f"Separated geometry:")
    print(f"  |P| range: [{np.abs(P_separated).min():.3e}, {np.abs(P_separated).max():.3e}]")
    print(f"  |P| mean: {np.abs(P_separated).mean():.3e}")
    print(f"  |P| std: {np.abs(P_separated).std():.3e}")
    print(f"\nMixed geometry:")
    print(f"  |P| range: [{np.abs(P_mixed).min():.3e}, {np.abs(P_mixed).max():.3e}]")
    print(f"  |P| mean: {np.abs(P_mixed).mean():.3e}")
    print(f"  |P| std: {np.abs(P_mixed).std():.3e}")
    
    if not save_frames:
        plt.show()
    
    plt.close()
    
    return P_separated, P_mixed


def generate_animation_frames(h5_path=None, chi_min=0.05, chi_max=20.0, n_frames=50, 
                               frames_dir=None, show_progress=True):
    """
    Generate animation frames for a series of χ values.
    
    Parameters:
    -----------
    h5_path : str, optional
        Path to HDF5 file. If None, will try common locations.
    chi_min : float
        Minimum χ value (default: 0.05)
    chi_max : float
        Maximum χ value (default: 20.0)
    n_frames : int
        Number of frames to generate (default: 50)
    frames_dir : str, optional
        Directory to save frames. If None, uses compare_measures/lp2016_outputs/animation_frames/
    show_progress : bool
        Print progress messages (default: True)
    """
    # Set up frames directory
    if frames_dir is None:
        frames_dir = os.path.join(os.path.dirname(__file__), "compare_measures", 
                                   "lp2016_outputs", "animation_frames3")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Frames will be saved to: {frames_dir}")
    
    # We need to load data first to compute sigmaPhi_screen and determine the relationship
    # But we'll do a preliminary load or use a placeholder approach
    # Actually, let's restructure to load data first, then compute chi ranges
    print(f"\nRequested χ range: [{chi_min:.3f}, {chi_max:.3f}]")
    
    # Reset frame counter
    plot_polarization_maps._frame_counter = 0
    
    # Load data once (will be reused for all frames)
    # Try different possible paths
    if h5_path is None:
        possible_paths = [
            r"faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
            r"..\faradays_angles_stats\lp_structure_tests\ms01ma08.mhd_w.00300.vtk.h5",
            os.path.join(os.path.dirname(__file__), "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
            os.path.join(os.path.dirname(__file__), "..", "faradays_angles_stats", "lp_structure_tests", "ms01ma08.mhd_w.00300.vtk.h5"),
        ]
        
        h5_path = None
        for path in possible_paths:
            if os.path.exists(path):
                h5_path = path
                break
        
        if h5_path is None:
            raise FileNotFoundError(f"Could not find HDF5 file. Tried: {possible_paths}")
    
    print(f"Loading MHD fields from: {h5_path}")
    
    # Load the data once
    keys = FieldKeys()
    cfg = PFAConfig()
    
    Bx, By, Bz, ne = load_fields(h5_path, keys)
    
    # Set configuration
    cfg.los_axis = 2  # z-axis is LOS
    gamma = cfg.gamma = 2.0
    cfg.faraday_const = 1.0
    
    # Choose B_parallel based on LOS axis
    if cfg.los_axis == 0:
        Bpar = Bx
    elif cfg.los_axis == 1:
        Bpar = By
    else:
        Bpar = Bz
    
    # Compute polarization emissivity and Faraday density once
    Pi = polarized_emissivity_simple(Bx, By, gamma=gamma)
    phi = faraday_density(ne, Bpar, C=cfg.faraday_const)
    
    # Set up separated geometry bounds
    phi_los = np.moveaxis(phi, cfg.los_axis, 0)
    Nz = phi_los.shape[0]
    scr_N = max(1, int(0.10 * Nz))
    screen_bounds = (0, scr_N)
    emit_bounds = (scr_N, Nz)
    
    # Compute σ_Φ from screen only (for separated case)
    Phi_screen = phi_los[screen_bounds[0]:screen_bounds[1]].sum(axis=0)
    sigmaPhi_screen = float(Phi_screen.std())
    
    # Fixed sigma for mixed case
    sigmaPhi0_mixed = 1.9101312160491943
    
    print(f"Loaded data. σ_Φ(screen) = {sigmaPhi_screen:.3f} (for separated)")
    print(f"  σ_Φ0 = {sigmaPhi0_mixed:.3f} (for mixed)")
    print(f"Separated geometry: screen bounds = {screen_bounds}, emit bounds = {emit_bounds}")
    
    # Compute the ratio between chi_mixed and chi_separated
    # chi_mixed = ratio * chi_separated, where ratio = sigmaPhi0_mixed / sigmaPhi_screen
    ratio = sigmaPhi0_mixed / sigmaPhi_screen
    print(f"\nRatio: χ_mixed / χ_separated = {ratio:.6f}")
    
    # To ensure both chi_separated and chi_mixed are in [chi_min, chi_max]:
    # 1. chi_separated ∈ [chi_min, chi_max]
    # 2. chi_mixed = ratio * chi_separated ∈ [chi_min, chi_max]
    #    => chi_separated ∈ [chi_min/ratio, chi_max/ratio]
    # The valid range is the intersection: [max(chi_min, chi_min/ratio), min(chi_max, chi_max/ratio)]
    
    # Handle both cases: ratio > 1 and ratio < 1
    if ratio > 1.0:
        # chi_mixed > chi_separated, so upper bound is tighter
        # chi_separated must be ≤ chi_max/ratio to keep chi_mixed ≤ chi_max
        valid_chi_sep_min = chi_min  # chi_min/ratio < chi_min, so chi_min is the constraint
        valid_chi_sep_max = min(chi_max, chi_max / ratio)
    else:
        # chi_mixed < chi_separated, so lower bound is tighter
        # chi_separated must be ≥ chi_min/ratio to keep chi_mixed ≥ chi_min
        valid_chi_sep_min = max(chi_min, chi_min / ratio)
        valid_chi_sep_max = chi_max  # chi_max/ratio > chi_max, so chi_max is the constraint
    
    if valid_chi_sep_min >= valid_chi_sep_max:
        # No valid intersection, use a fallback
        print(f"Warning: Cannot ensure both chi values in range simultaneously with ratio {ratio:.6f}")
        print(f"  Using chi_separated range: [{chi_min:.6f}, {chi_max:.6f}]")
        chi_sep_min = chi_min
        chi_sep_max = chi_max
    else:
        chi_sep_min = valid_chi_sep_min
        chi_sep_max = valid_chi_sep_max
        print(f"Adjusted chi_separated range to [{chi_sep_min:.6f}, {chi_sep_max:.6f}]")
        print(f"  This ensures both chi_separated and chi_mixed are in [{chi_min:.6f}, {chi_max:.6f}]")
    
    # Generate chi_separated values linearly
    chi_values = np.linspace(chi_sep_min, chi_sep_max, n_frames)
    
    # Verify chi array
    print(f"\nVerifying chi array:")
    print(f"  Length: {len(chi_values)} (expected {n_frames})")
    print(f"  First chi_sep (frame 1): {chi_values[0]:.6f}")
    print(f"  Second chi_sep (frame 2): {chi_values[1]:.6f}")
    print(f"  Last chi_sep (frame {n_frames}): {chi_values[-1]:.6f}")
    print(f"  Spacing: {(chi_values[1] - chi_values[0]):.6f} per frame")
    
    # Verify chi_mixed range
    chi_mix_first = ratio * chi_values[0]
    chi_mix_last = ratio * chi_values[-1]
    print(f"  First chi_mix (frame 1): {chi_mix_first:.6f}")
    print(f"  Last chi_mix (frame {n_frames}): {chi_mix_last:.6f}")
    
    # Generate frames
    for i, chi in enumerate(chi_values):
        if show_progress and (i % 10 == 0 or i == 0 or i == len(chi_values) - 1):
            progress_pct = 100 * (i + 1) / n_frames
            # chi is for separated case, compute lambda from it
            lam_current = np.sqrt(chi / (2.0 * sigmaPhi_screen * cfg.faraday_const))
            # Also compute chi_mixed for reference
            chi_mixed_current = 2.0 * sigmaPhi0_mixed * cfg.faraday_const * (lam_current**2)
            print(f"  Frame {i+1}/{n_frames} ({progress_pct:.1f}%): χ_sep = {chi:.6f}, χ_mix = {chi_mixed_current:.6f}, λ = {lam_current:.6f}        ")
        
        try:
            # Compute lambda from chi (chi is for separated case)
            lam2 = chi / (2.0 * sigmaPhi_screen * cfg.faraday_const)
            lam = np.sqrt(lam2)
            
            # Compute chi_mixed for reference (using same lambda)
            chi_mixed = 2.0 * sigmaPhi0_mixed * cfg.faraday_const * (lam**2)
            
            # Compute P maps
            P_separated = P_map_separated(Pi, phi, lam, cfg, 
                                         emit_bounds=emit_bounds, 
                                         screen_bounds=screen_bounds)
            P_mixed = P_map_mixed(Pi, phi, lam, cfg)
            
            # Create and save plot
            fig, axes = plt.subplots(2, 3, figsize=(17.7777777778, 10))
            
            # Determine regime and color based on chi
            if chi < 1.0:
                regime = "Synchrotron-dominated"
                regime_color = 'green'
            elif chi < 3.0:
                regime = "Transitional"
                regime_color = 'gold'  # Yellow/gold for transitional regime
            else:
                regime = "Faraday-dominated"
                regime_color = 'red'
            
            title = f'Polarization Maps Comparison ($\\chi = {chi:.3f}$) - {regime}'
            fig.suptitle(title, fontsize=28, fontweight='bold', color=regime_color)
            
            # Modern colormaps - rainbow for magnitude, seismic for complex parts
            cmap_magnitude = 'turbo'  # Modern rainbow colormap, excellent for presentations
            cmap_complex = 'seismic'   # Modern alternative to RdBu_r
            
            # Plot separated
            im1 = axes[0, 0].imshow(np.abs(P_separated), origin='lower', cmap=cmap_magnitude, aspect='auto')
            axes[0, 0].set_title('Separated: $|P|$ (Magnitude)', fontsize=24, fontweight='bold')
            axes[0, 0].set_xlabel('$X$', fontsize=20)
            axes[0, 0].set_ylabel('$Y$', fontsize=20)
            axes[0, 0].tick_params(labelsize=18)
            cbar1 = plt.colorbar(im1, ax=axes[0, 0])
            cbar1.set_label('$|P|$', fontsize=20)
            cbar1.ax.tick_params(labelsize=18)
            
            im2 = axes[0, 1].imshow(np.real(P_separated), origin='lower', cmap=cmap_complex, aspect='auto')
            axes[0, 1].set_title('Separated: $\\mathrm{Re}(P)$', fontsize=24, fontweight='bold')
            axes[0, 1].set_xlabel('$X$', fontsize=20)
            axes[0, 1].set_ylabel('$Y$', fontsize=20)
            axes[0, 1].tick_params(labelsize=18)
            cbar2 = plt.colorbar(im2, ax=axes[0, 1])
            cbar2.set_label('$\\mathrm{Re}(P)$', fontsize=20)
            cbar2.ax.tick_params(labelsize=18)
            
            im3 = axes[0, 2].imshow(np.imag(P_separated), origin='lower', cmap=cmap_complex, aspect='auto')
            axes[0, 2].set_title('Separated: $\\mathrm{Im}(P)$', fontsize=24, fontweight='bold')
            axes[0, 2].set_xlabel('$X$', fontsize=20)
            axes[0, 2].set_ylabel('$Y$', fontsize=20)
            axes[0, 2].tick_params(labelsize=18)
            cbar3 = plt.colorbar(im3, ax=axes[0, 2])
            cbar3.set_label('$\\mathrm{Im}(P)$', fontsize=20)
            cbar3.ax.tick_params(labelsize=18)
            
            # Plot mixed
            im4 = axes[1, 0].imshow(np.abs(P_mixed), origin='lower', cmap=cmap_magnitude, aspect='auto')
            axes[1, 0].set_title('Mixed: $|P|$ (Magnitude)', fontsize=24, fontweight='bold')
            axes[1, 0].set_xlabel('$X$', fontsize=20)
            axes[1, 0].set_ylabel('$Y$', fontsize=20)
            axes[1, 0].tick_params(labelsize=18)
            cbar4 = plt.colorbar(im4, ax=axes[1, 0])
            cbar4.set_label('$|P|$', fontsize=20)
            cbar4.ax.tick_params(labelsize=18)
            
            im5 = axes[1, 1].imshow(np.real(P_mixed), origin='lower', cmap=cmap_complex, aspect='auto')
            axes[1, 1].set_title('Mixed: $\\mathrm{Re}(P)$', fontsize=24, fontweight='bold')
            axes[1, 1].set_xlabel('$X$', fontsize=20)
            axes[1, 1].set_ylabel('$Y$', fontsize=20)
            axes[1, 1].tick_params(labelsize=18)
            cbar5 = plt.colorbar(im5, ax=axes[1, 1])
            cbar5.set_label('$\\mathrm{Re}(P)$', fontsize=20)
            cbar5.ax.tick_params(labelsize=18)
            
            im6 = axes[1, 2].imshow(np.imag(P_mixed), origin='lower', cmap=cmap_complex, aspect='auto')
            axes[1, 2].set_title('Mixed: $\\mathrm{Im}(P)$', fontsize=24, fontweight='bold')
            axes[1, 2].set_xlabel('$X$', fontsize=20)
            axes[1, 2].set_ylabel('$Y$', fontsize=20)
            axes[1, 2].tick_params(labelsize=18)
            cbar6 = plt.colorbar(im6, ax=axes[1, 2])
            cbar6.set_label('$\\mathrm{Im}(P)$', fontsize=20)
            cbar6.ax.tick_params(labelsize=18)
            
            plt.tight_layout()
            
            # Save frame
            frame_num = plot_polarization_maps._frame_counter
            output_path = os.path.join(frames_dir, f"frame_{frame_num:04d}_chi_{chi:.3f}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save NPZ file with all data for reproduction
            npz_path = os.path.join(frames_dir, f"frame_{frame_num:04d}_chi_{chi:.3f}.npz")
            
            # Determine regime for metadata
            if chi < 1.0:
                regime = "Synchrotron-dominated"
            elif chi < 3.0:
                regime = "Transitional"
            else:
                regime = "Faraday-dominated"
            
            np.savez(
                npz_path,
                # Polarization maps (complex arrays)
                P_separated=P_separated,
                P_mixed=P_mixed,
                # Chi values
                chi_separated=np.array(chi),
                chi_mixed=np.array(chi_mixed),
                # Lambda and configuration
                lambda_val=np.array(lam),
                sigmaPhi_screen=np.array(sigmaPhi_screen),
                sigmaPhi0_mixed=np.array(sigmaPhi0_mixed),
                # Frame metadata
                frame_number=np.array(frame_num),
                chi_min=np.array(chi_min),
                chi_max=np.array(chi_max),
                n_frames=np.array(n_frames),
                # Regime info
                regime=regime,
                # Geometry bounds
                screen_bounds=screen_bounds,
                emit_bounds=emit_bounds,
                # Configuration
                los_axis=np.array(cfg.los_axis),
                gamma=np.array(cfg.gamma),
                faraday_const=np.array(cfg.faraday_const),
            )
            
            plot_polarization_maps._frame_counter += 1
            
        except Exception as e:
            if show_progress:
                print(f"\n  Error at frame {i+1} (χ = {chi:.3f}): {e}")
            continue
    
    if show_progress:
        print(f"\n\nCompleted! Generated {plot_polarization_maps._frame_counter} frames")
        print(f"PNG frames saved in: {frames_dir}")
        print(f"NPZ data files saved in: {frames_dir} (for figure reproduction)")
    
    return frames_dir


def main():
    """Main function to run the polarization map plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot polarization maps for separated and mixed geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single frame at specific chi
  python plot_polarization_maps.py --chi 1.0
  
  # Generate animation frames
  python plot_polarization_maps.py --animate --chi_min 0.1 --chi_max 20 --n_frames 100
  
  # Single frame at specific lambda (legacy)
  python plot_polarization_maps.py --lambda 1.0
        """
    )
    parser.add_argument("--h5_path", type=str, default=None,
                        help="Path to HDF5 file (if not provided, will search common locations)")
    parser.add_argument("--lambda", type=float, default=None, dest="lam",
                        help="Wavelength value (ignored if --chi is provided)")
    parser.add_argument("--chi", type=float, default=None,
                        help="χ = 2*σ_Φ*λ² value (preferred over --lambda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the plot (default: auto-generated)")
    parser.add_argument("--animate", action="store_true",
                        help="Generate animation frames for a series of chi values")
    parser.add_argument("--chi_min", type=float, default=0.05,
                        help="Minimum χ value for animation (default: 0.05)")
    parser.add_argument("--chi_max", type=float, default=20.0,
                        help="Maximum χ value for animation (default: 20.0)")
    parser.add_argument("--n_frames", type=int, default=50,
                        help="Number of animation frames to generate (default: 50)")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Directory to save animation frames (default: auto-generated)")
    
    args = parser.parse_args()
    
    if args.animate:
        # Generate animation frames
        generate_animation_frames(
            h5_path=args.h5_path,
            chi_min=args.chi_min,
            chi_max=args.chi_max,
            n_frames=args.n_frames,
            frames_dir=args.frames_dir
        )
    else:
        # Single frame
        lam = args.lam if args.lam is not None else 1.0
        plot_polarization_maps(h5_path=args.h5_path, lam=lam, chi=args.chi, 
                              output_path=args.output)


if __name__ == "__main__":
    main()

