**Polarization Maps — Velocity-Based, Kolmogorov Turbulence (Synchrotron & Dust, Unnormalized)**

This folder contains 512×512 maps of Stokes I, Q, U, and polarization angle, computed from a 3D turbulent velocity field with a Kolmogorov power spectrum. The polarization was generated using synchrotron- and dust-style prescription, assuming Q and U come from the projected velocity components:

$$Q(x, y) = \int dz\\, (v_x^2 - v_y^2),$$

$$U(x, y) = \int dz\\, 2 v_x v_y.$$

The 3D velocity field was smoothed in Fourier space using a Gaussian filter:

$$
e^{-k^2 W^2}, \quad W = 2 \text{ (grid cells)},
$$

which reduces small-scale fluctuations and mimics limited observational resolution.


### Contents:

* `input/`: Binary data files for I, Q, U, and angle
* `figures/`: Plots of the maps and structure functions
* `read_and_plot.py`: Visualizes the I/Q/U and derived angle
* `check_power_law.py`: Computes angular structure functions
* `io.f90`: Fortran I/O routines for reading/writing the data
