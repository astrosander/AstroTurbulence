import numpy as np
from pathlib import Path

from directional_spectrum_single_lambda import (
    load_fields, polarized_emissivity_simple, faraday_density,
    separated_P_map, los_axis, emit_frac, screen_frac, C, h5_path, main
)

def compute_sigma_RM():
    Bx, By, Bz, ne = load_fields(h5_path)
    Pi = polarized_emissivity_simple(Bx, By, 2.0)
    Bpar = Bz
    phi = faraday_density(ne, Bpar, C)
    _, sigma_RM = separated_P_map(Pi, phi, 1.0, los_axis, emit_frac, screen_frac)
    return sigma_RM

def generate_animation_frames(chi_min=3.0, chi_max=20.0, n_frames=104, 
                               frames_dir=None, show_progress=True):
    if frames_dir is None:
        script_dir = Path(__file__).parent
        frames_dir = script_dir / "frames"
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    if show_progress:
        print(f"Frames will be saved to: {frames_dir}")
        print(f"Generating {n_frames} frames for chi from {chi_min:.2f} to {chi_max:.2f}")
    
    if show_progress:
        print("\nComputing sigma_RM from data...")
    sigma_RM = compute_sigma_RM()
    if show_progress:
        print(f"sigma_RM = {sigma_RM:.6f}")
    
    chi_values = np.linspace(chi_min, chi_max, n_frames)
    
    if show_progress:
        print(f"\nGenerating frames...")
    
    for i1, chi_target in enumerate(chi_values):
        i=i1+47
        if chi_target <= 0:
            lam = 0.0
        else:
            lam = np.sqrt(chi_target / (2.0 * sigma_RM))
        
        frame_filename = frames_dir / f"{i:04d}.png"
        
        if show_progress:
            print(f"  Frame {i+1}/{n_frames}: chi={chi_target:.3f}, lam={lam:.6f}")
        
        try:
            main(lam, save_path=str(frame_filename), show_plots=False)
        except Exception as e:
            if show_progress:
                print(f"    Error: {e}")
            continue
    
    if show_progress:
        print(f"\n✅ Animation frames saved to: {frames_dir}")
        print(f"   Total frames: {len(list(frames_dir.glob('*.png')))}")

if __name__ == "__main__":
    generate_animation_frames()

