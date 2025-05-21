**Polarization Maps from 3D Kolmogorov Turbulence**

This folder contains 512×512 maps of Stokes I, Q, U, and polarization angle, derived from a 3D Kolmogorov velocity field. The field was smoothed in Fourier space with a Gaussian filter (width W=2 grid cells), which limits small-scale structure.

The data is stored as Fortran binary files using `write_2D_field` from `io.f90`. If you're not using Fortran, you can mimic the `read_2D_field` logic — Python scripts in the folder do exactly that.

### Contents:

* `input/`: Binary data files for I, Q, U, and angle
* `figures/`: Plots of the maps and structure functions
* `read_and_plot.py`: Visualizes the I/Q/U and derived angle
* `check_power_law.py`: Computes angular structure functions
* `io.f90`: Fortran I/O routines for reading/writing the data