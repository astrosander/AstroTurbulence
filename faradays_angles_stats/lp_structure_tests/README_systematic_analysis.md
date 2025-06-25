# Systematic Faraday Screen Analysis - Fixes Based on Dr. Lazarian's Recommendations

## Overview

This directory contains improved analysis scripts that address the concerns raised by Dr. Lazarian about the Faraday screen simulations. The main issues identified and their solutions are documented below.

## Issues Identified by Dr. Lazarian

### 1. **Limited Power Law Range in MHD Data**
**Problem**: Structure functions from real MHD simulations show saturation rather than clean power laws, making it difficult to demonstrate the method clearly.

**Solution**: Use synthetic data with pure power law spectra to demonstrate the method without complications from real MHD physics.

### 2. **Insufficient Lambda Dependence Study** 
**Problem**: The dependence on wavelength λ was not systematically examined, particularly the λ⁴ scaling predicted by theory.

**Solution**: Comprehensive analysis with systematic λ variation and scaling tests.

### 3. **2π Ambiguity Concerns**
**Problem**: Large rotation measures (|Φ| > π) can cause measurement ambiguities in polarization angle observations.

**Solution**: Explicit detection and quantification of 2π ambiguity risks in the analysis.

### 4. **Mean Field vs Fluctuation Effects**
**Problem**: Need to understand how uniform mean magnetic fields affect the results compared to fluctuations alone.

**Solution**: Generate and analyze both total field (B + ΔB) and fluctuation-only (ΔB) scenarios.

### 5. **Method Comparison**
**Problem**: Need to compare direct rotation measure structure functions with angle-based structure functions to validate the collapse.

**Solution**: Side-by-side comparison of different analysis methods.

## New Analysis Scripts

### 1. `make_powerlaw_cube.py` (Enhanced)
**Enhancements**:
- Added `--generate-suite` option to create multiple test cases
- Stores both total field and fluctuations-only for comparison
- Saves metadata about power law indices and mean field strength
- Systematic parameter exploration

**Usage**:
```bash
# Generate a single cube
python make_powerlaw_cube.py --N 512 --beta_ne 3.67 --beta_bz 3.67 --mean_bz 1.0 --out test_cube.h5

# Generate entire test suite
python make_powerlaw_cube.py --generate-suite --N 512
```

### 2. `comprehensive_faraday_analysis.py` (New)
**Features**:
- Systematic λ dependence study with scaling tests
- 2π ambiguity detection and quantification
- Mean field vs fluctuation comparison
- Power law range analysis
- Multiple diagnostic plots

**Usage**:
```bash
python comprehensive_faraday_analysis.py synthetic_*.h5 --lambda-range 0.01 0.8 --n-lambda 15 --output-dir results/
```

### 3. `run_systematic_analysis.py` (New Workflow)
**Features**:
- Automated workflow for complete analysis
- Generates synthetic data and runs all analyses
- Produces publication-ready plots
- Addresses all of Dr. Lazarian's concerns systematically

**Usage**:
```bash
# Run complete analysis
python run_systematic_analysis.py

# Just generate cubes
python run_systematic_analysis.py --skip-analysis

# Regenerate everything
python run_systematic_analysis.py --regenerate-cubes
```

## Key Outputs

The analysis produces several plots that directly address Dr. Lazarian's concerns:

### 1. `lambda_dependence_study.pdf`
- Systematic study of λ dependence 
- λ⁴ scaling verification
- Angle structure functions for different wavelengths

### 2. `collapse_comparison.pdf`
- Logarithmic collapse test comparing direct vs angle-based methods
- Local power law slope analysis
- Validation of theoretical predictions

### 3. `powerlaw_range.pdf`
- Analysis of inertial range extent
- Comparison between synthetic and MHD data
- Quantification of clean power law regions

### 4. `field_comparison.pdf`
- Mean field vs fluctuation-only comparison
- 2π ambiguity risk assessment
- Maximum rotation measure analysis

## Theory Background

The Lazarian-Pogosyan theory predicts:

1. **Structure Function Scaling**: D_Φ(R) ∝ R^(5/3) for Kolmogorov turbulence
2. **Logarithmic Collapse**: -ln S(R,λ)/(2λ⁴) ≈ D_Φ(R) 
3. **Angle Structure Function**: D_φ(R,λ) = ½[1 - exp(-2λ⁴D_Φ)]
4. **Lambda Scaling**: D_φ ∝ λ⁴ in the weak fluctuation regime

## Addressing Specific Concerns

### **"The higher the power law, the better it is for us"**
→ Generated synthetic cubes with perfect power laws (β = 8/3, 11/3, 14/3) to test method sensitivity

### **"2π ambiguities somehow affect us"**  
→ Added explicit 2π ambiguity detection and quantification in analysis

### **"Compare this structural function obtained from the rotation measure map"**
→ Side-by-side comparison of direct RM structure function vs angle-based methods

### **"Look at the dependence on lambda, too"**
→ Comprehensive λ dependence study with systematic parameter exploration

### **"See what happened with synthetic data vs MHD data"**
→ Generated pure power law synthetic data for clean method demonstration

## Running the Analysis

### Prerequisites
```bash
pip install numpy scipy matplotlib h5py
```

### Quick Start
```bash
cd faradays_angles_stats/lp_structure_tests/
python run_systematic_analysis.py
```

This will:
1. Generate synthetic cubes with different parameters
2. Run comprehensive analysis addressing all concerns
3. Generate publication-ready plots
4. Provide detailed diagnostics

### For Paper Submission

The outputs from this analysis directly address the reviewer/advisor concerns:

1. **Demonstrate method with clean power laws** → `powerlaw_range.pdf`
2. **Show λ dependence systematically** → `lambda_dependence_study.pdf` 
3. **Address 2π ambiguity concerns** → `field_comparison.pdf`
4. **Compare different methods** → `collapse_comparison.pdf`
5. **Test mean field effects** → `field_comparison.pdf`

## Next Steps for Paper

1. **Use synthetic data results** to demonstrate clean method performance
2. **Include λ dependence plots** to show systematic wavelength effects  
3. **Discuss 2π ambiguity** limitations and when method is applicable
4. **Compare with MHD results** to show real-world performance
5. **Emphasize spatial vs frequency approach** advantage as Dr. Lazarian suggested

The analysis now provides a comprehensive foundation for addressing reviewer concerns and demonstrating the method's validity across different scenarios. 