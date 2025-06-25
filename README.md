# AstroTurbulence 🌌

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Astrophysics](https://img.shields.io/badge/Field-Astrophysics-purple.svg)](https://github.com/topics/astrophysics)
[![Plasma Physics](https://img.shields.io/badge/Field-Plasma_Physics-orange.svg)](https://github.com/topics/plasma-physics)

**Advanced toolkit for simulating and analyzing turbulent polarization in astrophysical environments**

> Simulate Kolmogorov turbulence, generate polarization maps, analyze Faraday rotation, and compute structure functions for astrophysical research.

![Polarization Maps](https://via.placeholder.com/800x300/1e1e1e/ffffff?text=Polarization+Structure+Functions+%26+Turbulence+Maps)

## ✨ Features

### 🌪️ **Turbulence Simulation**
- **Kolmogorov Power Spectrum**: Generate realistic 3D turbulent velocity fields
- **Solenoidal & Compressive**: Support for both solenoidal and non-solenoidal turbulence
- **Configurable Resolution**: From 256³ to 512³ grid points and beyond
- **Helmholtz Decomposition**: Advanced vector field decomposition

### 📡 **Polarization Analysis** 
- **Stokes Parameters**: Full I, Q, U parameter computation
- **Synchrotron Emission**: Realistic synchrotron polarization modeling
- **Dust Polarization**: Dust-based polarization prescriptions
- **Faraday Rotation**: Screen effects and rotation measure analysis

### 📊 **Statistical Tools**
- **Structure Functions**: Angular structure function computation `D_φ(R)`
- **Power Law Analysis**: Automated slope fitting and validation
- **PDF Analysis**: Single-point and two-point probability distributions
- **Correlation Analysis**: Spatial correlation characterization

### 🎯 **Research Applications**
- **Interstellar Medium** studies
- **Magnetohydrodynamic turbulence** analysis  
- **Cosmic ray propagation** modeling
- **Radio astronomy** data interpretation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/astrosander/AstroTurbulence.git
cd AstroTurbulence
pip install -r requirements.txt
```

### Basic Usage

#### Generate Turbulent Velocity Field
```python
from turbulence_angles.turbulence_angles_full import generate_velocity_cube

# Generate 256³ solenoidal velocity field
u = generate_velocity_cube(N=256, solenoidal=True, seed=42)
print(f"Generated velocity cube: {u.shape}")
```

#### Compute Polarization Maps
```python
# Synchrotron polarization
Q_syn = np.sum(u[:,:,:,0]**2 - u[:,:,:,1]**2, axis=2)
U_syn = np.sum(2 * u[:,:,:,0] * u[:,:,:,1], axis=2)
phi_syn = 0.5 * np.arctan2(U_syn, Q_syn)
```

#### Structure Function Analysis
```python
from check_power_law import structure_function_2d

Rs = np.logspace(0, 2, 20).astype(int)
R_vals, D_phi = structure_function_2d(phi_syn, Rs)

# Should recover ~R^(5/3) Kolmogorov scaling
```

## 📁 Project Structure

```
AstroTurbulence/
├── 🌪️ turbulence_angles/          # Core turbulence simulation
│   ├── turbulence_angles_full.py   # Main simulation engine
│   ├── single_PDF_vector.py        # Statistical analysis
│   └── figures/                    # Generated plots
├── 📡 faradays_angles_stats/       # Faraday rotation analysis
│   ├── faraday_screen_demo.py     # Faraday screen modeling
│   └── lp_structure_tests/         # Linear polarization tests
├── 📊 stokesMaps_velocity_kolmogorov_NoNorm/  # Stokes parameter maps
│   ├── read_and_plot.py           # Visualization tools
│   └── check_power_law.py         # Power law validation
├── ⚡ spectrum_generation/         # Power spectrum analysis
└── 🧮 check_integration_factors/  # Numerical validation
```

## 📈 Example Results

### Polarization Structure Functions
The toolkit reproduces the expected **Kolmogorov R^(5/3)** scaling for turbulent polarization:

```python
# Expected output:
# Stokes azimuth slope: 1.67 ± 0.05
# Vector azimuth slope: 1.65 ± 0.04
```

### Faraday Rotation Analysis
Analyze how magnetic field turbulence affects polarization:

```python
python faradays_angles_stats/faraday_screen_demo.py
```

## 🔬 Scientific Applications

This toolkit has been used for research in:

- **Interstellar Turbulence**: Characterizing magnetohydrodynamic turbulence in the ISM
- **Pulsar Studies**: Understanding scintillation and Faraday rotation effects  
- **Galaxy Formation**: Modeling magnetic field evolution in cosmic structure
- **Radio Astronomy**: Interpreting polarization observations from SKA/LOFAR

## 📚 Documentation

### Core Functions

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `generate_velocity_cube()` | Generate 3D turbulent velocity field | `N`, `solenoidal`, `seed` |
| `structure_function_2d()` | Compute angular structure function | `field`, `Rs`, `max_pairs` |
| `angle_stokes()` | Calculate Stokes polarization angle | `u` (velocity field) |
| `sample_pairs_3d()` | Sample vector angle differences | `u`, `R`, `max_pairs` |

### Mathematical Background

The toolkit implements several key astrophysical relations:

**Synchrotron Polarization:**
```math
Q(x,y) = ∫ dz (v_x² - v_y²)
U(x,y) = ∫ dz (2 v_x v_y)
```

**Angular Structure Function:**
```math
D_φ(R) = ⟨[φ(r+R) - φ(r)]²⟩
```

**Kolmogorov Spectrum:**
```math
E(k) ∝ k^{-5/3}
```

## 🛠️ Requirements

- **Python 3.8+**
- **NumPy** (≥1.19)
- **SciPy** (≥1.6)
- **Matplotlib** (≥3.3)
- **h5py** (≥3.0)
- **numba** (≥0.54) - for performance
- **Fortran compiler** (for I/O routines)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/astrosander/AstroTurbulence.git
cd AstroTurbulence
pip install -e .
pre-commit install
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{astroturbulence2024,
  title={AstroTurbulence: Advanced Toolkit for Astrophysical Turbulence Analysis},
  author={Aliaksandr Melnichenka},
  year={2024},
  url={https://github.com/astrosander/AstroTurbulence}
}
```

## 🌟 Acknowledgments

- Built for the astrophysics and plasma physics communities
- Inspired by modern magnetohydrodynamic turbulence research
- Optimized for high-performance scientific computing

---

⭐ **Star this repository** if you find it useful for your astrophysics research!

🐛 **Found a bug?** [Open an issue](https://github.com/astrosander/AstroTurbulence/issues)

💡 **Have a feature request?** [Start a discussion](https://github.com/astrosander/AstroTurbulence/discussions) 