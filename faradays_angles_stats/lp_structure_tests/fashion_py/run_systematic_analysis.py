#!/usr/bin/env python3
"""
run_systematic_analysis.py
==========================

Workflow script to run systematic Faraday screen analysis as recommended by Dr. Lazarian.

This script:
1. Generates synthetic cubes with pure power laws
2. Runs comprehensive analysis including lambda dependence studies  
3. Tests mean field vs fluctuation effects
4. Addresses 2π ambiguity concerns
5. Compares different analysis methods

Usage:
    python run_systematic_analysis.py [--regenerate-cubes] [--skip-analysis]
    
Author : <you>
Date   : 2025-06-25
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def generate_synthetic_cubes():
    """Generate test suite of synthetic cubes with different parameters."""
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC CUBES WITH PURE POWER LAWS")
    print("="*80)
    print("As recommended by Dr. Lazarian for clean power law demonstration")
    
    # Generate test suite
    cmd = ["python", "make_powerlaw_cube.py", "--generate-suite", "--N", "512"]
    success = run_command(cmd, "Generating test suite of synthetic cubes")
    
    if not success:
        print("WARNING: Failed to generate test suite. Trying individual cubes...")
        
        # Generate individual test cases
        test_cases = [
            ("--beta_ne", "3.67", "--beta_bz", "3.67", "--mean_bz", "0.0", 
             "--out", "synthetic_kolmogorov_no_mean.h5"),
            ("--beta_ne", "3.67", "--beta_bz", "3.67", "--mean_bz", "1.0", 
             "--out", "synthetic_kolmogorov_with_mean.h5"),
            ("--beta_ne", "3.67", "--beta_bz", "3.67", "--mean_bz", "5.0", 
             "--out", "synthetic_kolmogorov_strong_mean.h5"),
            ("--beta_ne", "2.67", "--beta_bz", "2.67", "--mean_bz", "0.0", 
             "--out", "synthetic_steep_powerlaw.h5"),
            ("--beta_ne", "4.67", "--beta_bz", "4.67", "--mean_bz", "0.0", 
             "--out", "synthetic_shallow_powerlaw.h5"),
        ]
        
        for i, args in enumerate(test_cases):
            cmd = ["python", "make_powerlaw_cube.py", "--N", "512", "--seed", str(2025+i)] + list(args)
            run_command(cmd, f"Generating cube {i+1}/{len(test_cases)}")


def run_comprehensive_analysis():
    """Run the comprehensive analysis on all available cubes."""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE FARADAY ANALYSIS")
    print("="*80)
    print("Addressing Dr. Lazarian's concerns:")
    print("- Systematic lambda dependence study")
    print("- 2π ambiguity detection")
    print("- Mean field vs fluctuation comparison")
    print("- Power law range analysis")
    
    # Find all synthetic cube files
    cube_files = list(Path(".").glob("synthetic_*.h5"))
    
    if not cube_files:
        print("No synthetic cube files found! Make sure to generate them first.")
        return False
    
    print(f"Found {len(cube_files)} cube files:")
    for f in cube_files:
        print(f"  - {f}")
    
    # Run comprehensive analysis
    cmd = ["python", "comprehensive_faraday_analysis.py"] + [str(f) for f in cube_files] + [
        "--lambda-range", "0.01", "0.8",  # Wide lambda range as requested
        "--n-lambda", "15",               # More lambda values for better sampling
        "--output-dir", "systematic_analysis_results"
    ]
    
    return run_command(cmd, "Running comprehensive analysis")


def run_original_analysis_for_comparison():
    """Run original analysis scripts for comparison."""
    print("\n" + "="*80)
    print("RUNNING ORIGINAL ANALYSIS FOR COMPARISON")
    print("="*80)
    
    # Find a test cube
    cube_files = list(Path(".").glob("synthetic_*.h5"))
    if not cube_files:
        print("No cube files available for comparison analysis.")
        return
        
    test_cube = cube_files[0]
    print(f"Using {test_cube} for comparison analysis...")
    
    # Run original faraday screen analysis
    success1 = run_command(
        ["python", "faraday_screen_sim_new.py"],
        "Original Faraday screen analysis"
    )
    
    # Run logarithmic collapse test
    success2 = run_command(
        ["python", "faraday_screen_lncheck.py"],
        "Logarithmic collapse test"
    )
    
    return success1 and success2


def check_dependencies():
    """Check if required Python packages are available."""
    required_packages = ['numpy', 'scipy', 'matplotlib', 'h5py']
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic Faraday analysis as recommended by Dr. Lazarian"
    )
    parser.add_argument("--regenerate-cubes", action="store_true",
                       help="Regenerate synthetic cubes even if they exist")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis, only generate cubes")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip original analysis comparison")
    
    args = parser.parse_args()
    
    print("SYSTEMATIC FARADAY SCREEN ANALYSIS")
    print("Based on recommendations from Dr. Lazarian")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Change to the appropriate directory
    script_dir = Path(__file__).parent
    if script_dir != Path("."):
        print(f"Changing to directory: {script_dir}")
        import os
        os.chdir(script_dir)
    
    # Step 1: Generate synthetic cubes (if needed)
    cube_files = list(Path(".").glob("synthetic_*.h5"))
    if args.regenerate_cubes or not cube_files:
        generate_synthetic_cubes()
    else:
        print(f"Found {len(cube_files)} existing synthetic cubes. Use --regenerate-cubes to regenerate.")
    
    if args.skip_analysis:
        print("Skipping analysis as requested.")
        return 0
    
    # Step 2: Run comprehensive analysis
    success = run_comprehensive_analysis()
    if not success:
        print("Comprehensive analysis failed!")
        return 1
    
    # Step 3: Run original analysis for comparison (optional)
    if not args.skip_comparison:
        run_original_analysis_for_comparison()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Results are available in:")
    print("  - systematic_analysis_results/  (comprehensive analysis)")
    print("  - fig*.pdf  (individual analysis plots)")
    print("\nKey outputs addressing Dr. Lazarian's concerns:")
    print("  1. lambda_dependence_study.pdf - Systematic λ dependence")
    print("  2. collapse_comparison.pdf - Method comparison")
    print("  3. powerlaw_range.pdf - Inertial range analysis")
    print("  4. field_comparison.pdf - Mean field effects")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 