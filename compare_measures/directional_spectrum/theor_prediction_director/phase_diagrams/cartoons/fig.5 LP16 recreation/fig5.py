#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LP16 (Lazarian & Pogosyan 2016) Figure-5 style code, upgraded to FULL Faraday
(single-wavelength PSA slice) using Eq. (35), with eta = 2*sigma_phi*lambda^2
as the independent variable.

What this script does
---------------------
- Keeps the intrinsic polarization correlation model xi_i from Eq. (30)
- Replaces the negligible-Faraday Eq. (44) correlation with the full Eq. (35)
  single-wavelength PSA correlation:
      <P(X1) P*(X2)>
- Computes the RM-depth structure function D_{ΔΦ}(R,z1,z2) from Eq. (18)
  using the saturated isotropic RM-density correlation model (Eq. 14-like model)
- Uses eta = 2*sigma_phi*lambda^2 instead of lambda directly
- Keeps Eq. (45)/(46) asymptotes/markers for eta=0 (Figure 5 limit)

Notes
-----
- sigma_phi here corresponds to LP16's sigma_f (rms RM-density fluctuation).
- eta=0 reproduces the negligible-Faraday Figure 5 limit numerically.
- This is substantially more expensive than the eta=0 Eq. (44) integral.
  Use moderate nz (e.g. 401-801) for practical runs.

Paper references (equations and context): Eq. (30), Eq. (31)-(35), Eq. (18)-(21),
Eq. (44)-(46), and Figure 5 caption. (See uploaded LP16 PDF.)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

# NumPy compatibility (np.trapezoid in newer NumPy, np.trapz in older)
TRAPZ = getattr(np, "trapezoid", np.trapz)


# ============================================================
# Exact paper-model ingredients (intrinsic + RM density)
# ============================================================

def xi_i_eq30(R, dz, r_i, m, sigma_i=1.0, Pbar_i=0.0):
    r"""
    Eq. (30): intrinsic polarization correlation model (saturated isotropic power law)
        xi_i(R, dz) = |Pbar_i|^2 + sigma_i^2 * r_i^m / [r_i^m + (R^2 + dz^2)^(m/2)]

    In Figure 5 use case, typically Pbar_i = 0.
    """
    rr = (R * R + dz * dz) ** (m / 2.0)
    return abs(Pbar_i) ** 2 + sigma_i**2 * (r_i**m) / (r_i**m + rr)


def xi_phi_eq14_model(R, dz, r_f, m_f, sigma_phi):
    r"""
    Saturated isotropic power-law model for RM-density correlation xi_f(R, dz),
    matching the LP16 illustrative model form (Eq. 14 context, with Eq. 13/14 definitions).

        xi_f(R, dz) = sigma_phi^2 * r_f^m_f / [r_f^m_f + (R^2 + dz^2)^(m_f/2)]

    Here sigma_phi corresponds to LP16 sigma_f (rms RM-density fluctuation).
    """
    if sigma_phi == 0.0:
        # No turbulent Faraday fluctuations
        return np.zeros_like(dz, dtype=float)

    rr = (R * R + dz * dz) ** (m_f / 2.0)
    return sigma_phi**2 * (r_f**m_f) / (r_f**m_f + rr)


# ============================================================
# Numerics helpers
# ============================================================

def make_uniform_z_grid(L, nz=601):
    """
    Uniform z grid on [0, L] (node grid, includes endpoints).
    Uniform grid is used because we need efficient 2D cumulative trapezoidal integrals
    for Eq. (18) over all (z1, z2) prefixes.
    """
    z = np.linspace(0.0, L, int(nz))
    if z.size < 2:
        raise ValueError("nz must be >= 2")
    return z


def make_z_grid_for_eq44(L, nz=2001, min_dz_ratio=1e-6):
    r"""
    Nonuniform Δz grid for Eq. (44) one-sided integral:
        ∫_0^L dΔz (L-Δz) ξ_i(R,Δz)

    Uses dense spacing near Δz=0 to resolve small-R asymptotics.
    The grid is on [0, L] with logarithmic spacing near 0.

    Parameters
    ----------
    L : float
        Path length.
    nz : int
        Total number of points.
    min_dz_ratio : float
        Minimum Δz / L to resolve near-diagonal contributions.

    Returns
    -------
    dz_grid : ndarray
        Nonuniform Δz grid on [0, L].
    """
    nz = int(nz)
    if nz < 2:
        raise ValueError("nz must be >= 2")

    # Logarithmic spacing near 0, transitioning to uniform
    # Use ~30% of points for the dense near-zero region
    n_log = max(2, int(0.3 * nz))
    n_unif = nz - n_log

    # Logarithmic part: from min_dz_ratio*L to transition point
    transition_frac = 0.1  # transition at 10% of L
    dz_log = np.logspace(
        np.log10(min_dz_ratio * L),
        np.log10(transition_frac * L),
        n_log
    )

    # Uniform part: from transition to L
    dz_unif = np.linspace(transition_frac * L, L, n_unif)

    # Combine and ensure 0 and L are included
    dz_grid = np.concatenate([[0.0], dz_log, dz_unif[1:]])
    dz_grid = np.unique(dz_grid)  # remove duplicates
    dz_grid = np.clip(dz_grid, 0.0, L)  # ensure bounds

    return dz_grid


def trapz_weights_uniform(z):
    """1D trapezoidal weights for a uniform node grid."""
    dz = float(z[1] - z[0])
    w = np.ones_like(z, dtype=float) * dz
    w[0] *= 0.5
    w[-1] *= 0.5
    return w, dz


def cumulative_trapz_prefix_2d_uniform(F, dz):
    r"""
    Compute all prefix integrals C[i,j] ~ ∫_0^{z_i} du ∫_0^{z_j} dv F(u,v)
    using cumulative trapezoidal integration on a UNIFORM node grid.

    Parameters
    ----------
    F : (nz, nz) ndarray
        Values on the tensor-product node grid.
    dz : float
        Uniform spacing.

    Returns
    -------
    C : (nz, nz) ndarray
        Prefix double-integral table.
    """
    # Integrate along axis 0 (u) first
    Iu = np.zeros_like(F, dtype=float)
    Iu[1:, :] = np.cumsum(0.5 * (F[1:, :] + F[:-1, :]) * dz, axis=0)

    # Then integrate along axis 1 (v)
    C = np.zeros_like(F, dtype=float)
    C[:, 1:] = np.cumsum(0.5 * (Iu[:, 1:] + Iu[:, :-1]) * dz, axis=1)
    return C


# ============================================================
# Eq. (44): Negligible-Faraday correlation (eta=0 limit)
# ============================================================

def corr_P_eq44_negligible_faraday(R_values, r_i, m, L, sigma_i=1.0, Pbar_i=0.0, z_grid=None):
    r"""
    Eq. (44): Negligible-Faraday correlation (weak/no rotation limit):
        C_P(R) = ∫_0^L dΔz (L-Δz) ξ_i(R,Δz)

    This is the one-sided integral form used in the original Figure 5 implementation.
    For eta=0, this should match the full Eq. (35) result (up to a factor-of-2 convention).

    Parameters
    ----------
    R_values : array-like
        Transverse separation values.
    r_i : float
        Intrinsic correlation length.
    m : float
        Power-law index.
    L : float
        Path length.
    sigma_i : float
        Intrinsic polarization rms.
    Pbar_i : complex or float
        Mean intrinsic polarization (typically 0).
    z_grid : ndarray, optional
        Nonuniform Δz grid. If None, a default dense grid is created.

    Returns
    -------
    C_P : ndarray
        Correlation values (real, same size as R_values).
    """
    R_values = np.asarray(R_values, dtype=float)
    if z_grid is None:
        z_grid = make_z_grid_for_eq44(L, nz=2001)

    dz_grid = np.asarray(z_grid, dtype=float)
    if dz_grid.size < 2:
        raise ValueError("z_grid must have at least 2 points")

    # Ensure grid is sorted and in [0, L]
    dz_grid = np.sort(dz_grid)
    dz_grid = np.clip(dz_grid, 0.0, L)

    # Weight function: (L - Δz) for the one-sided integral
    weights = L - dz_grid
    weights = np.maximum(weights, 0.0)  # ensure non-negative

    # Trapezoidal weights
    dz_spacings = np.diff(dz_grid)
    trapz_w = np.zeros_like(dz_grid, dtype=float)
    if dz_spacings.size > 0:
        trapz_w[0] = 0.5 * dz_spacings[0] if dz_spacings.size > 0 else 0.0
        trapz_w[-1] = 0.5 * dz_spacings[-1] if dz_spacings.size > 0 else 0.0
        if dz_spacings.size > 1:
            trapz_w[1:-1] = 0.5 * (dz_spacings[:-1] + dz_spacings[1:])

    # Combined weights
    w_total = weights * trapz_w

    # Compute correlation for each R
    out = np.empty(R_values.size, dtype=float)
    for i, R in enumerate(R_values):
        xi_vals = xi_i_eq30(R, dz_grid, r_i=r_i, m=m, sigma_i=sigma_i, Pbar_i=Pbar_i)
        out[i] = np.sum(w_total * xi_vals)

    return out


# ============================================================
# Corrected finite-η correlator (Δz + z2 formulation)
# ============================================================
# This implementation uses a nonuniform Δz grid for all η, ensuring
# continuity as η→0 and preserving small-R asymptotics.

def cumtrapz_1d(y, x):
    """Cumulative trapezoidal integral on nonuniform grid."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    out = np.zeros_like(y)
    if y.size < 2:
        return out
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out


def interp1(x, y, xq):
    """Vectorized linear interpolation."""
    return np.interp(np.asarray(xq, float), np.asarray(x, float), np.asarray(y, float))


def xi_phi_hat(R, dz, r_f, m_f):
    r"""
    RM-density correlation SHAPE (dimensionless; sigma_phi factored out).
    This is xi_phi / sigma_phi^2.
    """
    rr = (R * R + dz * dz) ** (m_f / 2.0)
    return (r_f**m_f) / (r_f**m_f + rr)


def build_A0A1_hat(R, s_grid, r_f, m_f):
    r"""
    Build cumulants A0(x)=∫0^x xi(s) ds and A1(x)=∫0^x s xi(s) ds
    for the dimensionless RM-density correlation shape.
    """
    xi = xi_phi_hat(R, s_grid, r_f=r_f, m_f=m_f)
    A0 = cumtrapz_1d(xi, s_grid)
    A1 = cumtrapz_1d(s_grid * xi, s_grid)
    return A0, A1


def Vhat_from_A(A0_0, A1_0, s_grid, zq):
    r"""
    Vhat(z) = ∫_0^z ∫_0^z xi0(u-v) du dv in terms of A0, A1 (R=0).
    Vhat(z) = 2*(z*A0(z) - A1(z))
    """
    A0z = interp1(s_grid, A0_0, zq)
    A1z = interp1(s_grid, A1_0, zq)
    zq = np.asarray(zq, float)
    return 2.0 * (zq * A0z - A1z)


def C_hat_bDelta(A0_R, A1_R, s_grid, b, delta, L):
    r"""
    Covariance C_R(a,b) = ∫_0^a du ∫_0^b dv xi_R(u-v)
    for a=b+Δ, b=z2, Δ=Δz, using piecewise weight (no 2D quadrature).
    
    Returns Chat_R(a=b+delta, b) for arrays:
      b: (nb,)
      delta: (nd,)
    output: (nb, nd), valid: (nb, nd) boolean
    """
    b = np.asarray(b, float)
    delta = np.asarray(delta, float)
    nb = b.size
    nd = delta.size

    # Precompute A at b and delta (1D)
    A0_b = interp1(s_grid, A0_R, b)
    A1_b = interp1(s_grid, A1_R, b)
    A0_d = interp1(s_grid, A0_R, delta)
    A1_d = interp1(s_grid, A1_R, delta)

    # 2D bplus = b + delta
    bplus = b[:, None] + delta[None, :]
    valid = (bplus <= L)  # only these pairs are physically in-domain

    # safe values for interpolation (won't be used where invalid)
    bplus_safe = np.where(valid, bplus, L)

    A0_bp = interp1(s_grid, A0_R, bplus_safe.ravel()).reshape(nb, nd)
    A1_bp = interp1(s_grid, A1_R, bplus_safe.ravel()).reshape(nb, nd)

    # Chat formula (derived from exact overlap weight for intervals [0,b+Δ] and [0,b])
    # Chat = b*A0(b) - A1(b) - Δ*A0(Δ) + (b+Δ)*A0(b+Δ) - A1(b+Δ) + A1(Δ)
    Chat = (
        b[:, None] * A0_b[:, None] - A1_b[:, None]
        - delta[None, :] * A0_d[None, :]
        + bplus_safe * A0_bp - A1_bp
        + A1_d[None, :]
    )

    # Only valid region matters; set invalid to 0 (it will be masked later anyway)
    Chat = np.where(valid, Chat, 0.0)
    return Chat, valid


def inner_integral_IF(delta, eta, b_grid, Dhat_matrix):
    r"""
    Compute inner z2 integral I_F(Δ) = ∫_0^{L-Δ} exp(-η^2 Dhat) dz2.
    
    Parameters
    ----------
    b_grid : (nb,) ndarray
        Uniform grid on [0,L] for z2.
    Dhat_matrix : (nb, nd) ndarray
        Dhat(b, delta) values.
    delta : (nd,) ndarray
        Δz values.
    eta : float
        Faraday parameter.
    
    Returns
    -------
    IF : (nd,) ndarray
        Inner integral values for each delta.
    """
    b = b_grid
    nb = b.size
    nd = delta.size
    db = b[1] - b[0]

    F = np.exp(-(eta * eta) * Dhat_matrix)

    # cumulative trapezoid in b for each delta-column
    Cum = np.zeros_like(F)
    Cum[1:, :] = np.cumsum(0.5 * (F[1:, :] + F[:-1, :]) * db, axis=0)

    # evaluate Cum at b_target = L - delta (vectorized linear interpolation on uniform grid)
    L = b[-1]
    b_target = L - delta
    t = b_target / db
    idx = np.floor(t).astype(int)
    idx = np.clip(idx, 0, nb - 2)
    frac = t - idx
    cols = np.arange(nd)
    IF = Cum[idx, cols] + frac * (Cum[idx + 1, cols] - Cum[idx, cols])
    return IF


def corr_P_full_eta_delta(
    R_values,
    eta,
    beta,
    L,
    # intrinsic
    r_i, m_i, sigma_i=1.0, Pbar_i=0.0,
    # Faraday RM-density shape
    r_f=1.0, m_f=2.0/3.0,
    # grids
    delta_grid=None,      # nonuniform Δz grid on [0,L]
    s_grid=None,          # grid for cumulants of xi_phi_hat on [0,L]
    nb_z2=800             # uniform z2 grid points
):
    r"""
    Computes C_P(R;eta) using Δz + z2 formulation.

    This implementation guarantees continuity as eta->0, and preserves small-R asymptotics
    because delta_grid is dense near 0.

    Parameters
    ----------
    R_values : array-like
        Transverse separation values.
    eta : float
        eta = 2*sigma_phi*lambda^2 (user variable)
    beta : float
        beta = phi_bar / sigma_phi (set 0 for turbulent-only)
    L : float
        Path length.
    r_i, m_i, sigma_i, Pbar_i : float
        Intrinsic polarization model parameters.
    r_f, m_f : float
        RM-density correlation shape parameters (sigma_phi factored out).
    delta_grid : ndarray
        Nonuniform Δz grid on [0,L] (dense near 0).
    s_grid : ndarray, optional
        Grid for cumulants of xi_phi_hat on [0,L]. If None, uses delta_grid.
    nb_z2 : int
        Number of uniform z2 grid points.

    Returns
    -------
    C_P : ndarray
        Correlation values (complex, same size as R_values).
    """
    R_values = np.asarray(R_values, float)
    eta = float(eta)
    beta = float(beta)

    if delta_grid is None:
        raise ValueError("Provide a nonuniform delta_grid on [0,L] (dense near 0).")
    if s_grid is None:
        s_grid = delta_grid  # usually OK: same resolution for RM cumulants

    delta = np.asarray(delta_grid, float)
    if delta[0] != 0.0:
        raise ValueError("delta_grid must start at 0.")
    if delta[-1] < L:
        raise ValueError("delta_grid must reach L (or very close).")

    # uniform z2 grid for inner integral
    b = np.linspace(0.0, L, int(nb_z2))

    # Precompute R=0 cumulants for RM-density (dimensionless hat)
    A0_0, A1_0 = build_A0A1_hat(0.0, s_grid, r_f=r_f, m_f=m_f)

    # Precompute Vhat on grids we will query
    Vhat_b = Vhat_from_A(A0_0, A1_0, s_grid, b)  # (nb,)

    # mean phase factor depends only on delta
    phase = np.exp(1j * eta * beta * delta) if (eta != 0.0 and beta != 0.0) else 1.0

    out = np.empty(R_values.size, dtype=np.complex128)

    for i, R in enumerate(R_values):
        # RM cumulants at this R
        A0_R, A1_R = build_A0A1_hat(R, s_grid, r_f=r_f, m_f=m_f)

        # Build Dhat(b,delta) matrix (dimensionless)
        # Need Vhat(b+delta) and Chat_R(b+delta,b)
        bplus = b[:, None] + delta[None, :]
        # validity region is b <= L - delta (same as bplus<=L)
        # We'll compute Dhat everywhere but mask invalid to 0 in the end.
        Chat, valid = C_hat_bDelta(A0_R, A1_R, s_grid, b=b, delta=delta, L=L)

        Vhat_bp = Vhat_from_A(A0_0, A1_0, s_grid, np.minimum(bplus, L).ravel()).reshape(b.size, delta.size)

        Dhat = 0.5 * (Vhat_b[:, None] + Vhat_bp - 2.0 * Chat)
        Dhat = np.maximum(Dhat, 0.0)
        Dhat = np.where(valid, Dhat, 0.0)

        # Inner integral over z2: IF(delta)
        IF = inner_integral_IF(delta=delta, eta=eta, b_grid=b, Dhat_matrix=Dhat)

        # intrinsic xi_i(R,delta) (outer integrand)
        xi_i = xi_i_eq30(R, delta, r_i=r_i, m=m_i, sigma_i=sigma_i, Pbar_i=Pbar_i)

        integrand = xi_i * phase * IF
        out[i] = TRAPZ(integrand, delta)

    return out


def D_P_full_eta_delta(
    R_values,
    eta,
    beta,
    L,
    r_i, m_i, sigma_i=1.0, Pbar_i=0.0,
    r_f=1.0, m_f=2.0/3.0,
    delta_grid=None,
    s_grid=None,
    nb_z2=800
):
    r"""
    Structure function D_P(R;eta) = 2 [ C(0;eta) - C(R;eta) ]
    using the corrected Δz + z2 formulation.
    """
    C_R = corr_P_full_eta_delta(
        R_values, eta, beta, L,
        r_i, m_i, sigma_i, Pbar_i,
        r_f, m_f,
        delta_grid=delta_grid, s_grid=s_grid, nb_z2=nb_z2
    )
    C_0 = corr_P_full_eta_delta(
        np.array([0.0]), eta, beta, L,
        r_i, m_i, sigma_i, Pbar_i,
        r_f, m_f,
        delta_grid=delta_grid, s_grid=s_grid, nb_z2=nb_z2
    )[0]

    D_R = 2.0 * (np.real(C_0) - np.real(C_R))
    D_inf = 2.0 * np.real(C_0)
    return D_R, float(np.real(C_0)), float(D_inf)


# ============================================================
# Full Faraday machinery (Eq. 18 + Eq. 35 at fixed wavelength)
# ============================================================

class LP16FullEtaPSA:
    r"""
    Full single-wavelength PSA correlator using Eq. (35), parameterized by
        eta = 2 * sigma_phi * lambda^2

    The turbulent Faraday decorrelation factor in Eq. (35),
        exp[-4 lambda^4 D_{ΔΦ}]
    is rewritten as
        exp[-eta^2 * Dhat_{ΔΦ}],
    where Dhat_{ΔΦ} = D_{ΔΦ} / sigma_phi^2.

    Mean Faraday term (optional):
        exp[ + i * eta * (phi_bar / sigma_phi) * (z1 - z2) ].
    """

    def __init__(self, L, z_grid,
                 r_f, m_f, sigma_phi,
                 r_i, m_i, sigma_i=1.0, Pbar_i=0.0,
                 phi_bar_over_sigma_phi=0.0):
        self.L = float(L)
        self.z = np.asarray(z_grid, dtype=float)
        self.nz = self.z.size
        self.w, self.dz = trapz_weights_uniform(self.z)

        # Physical/model params
        self.r_f = float(r_f)
        self.m_f = float(m_f)
        self.sigma_phi = float(sigma_phi)        # LP16 sigma_f
        self.r_i = float(r_i)
        self.m_i = float(m_i)
        self.sigma_i = float(sigma_i)
        self.Pbar_i = Pbar_i
        self.beta = float(phi_bar_over_sigma_phi)  # beta = phi_bar / sigma_phi

        # Common matrices on z-grid
        self.z1 = self.z[:, None]
        self.z2 = self.z[None, :]
        self.dz_mat = self.z1 - self.z2
        self.W2 = self.w[:, None] * self.w[None, :]

        # Precompute the R=0 covariance prefix table for xi_phi, used in Eq. (18)
        # D_{ΔΦ}(R,z1,z2) = 1/2 [V(z1)+V(z2)-2 C_R(z1,z2)]
        if self.sigma_phi > 0.0:
            K00 = xi_phi_eq14_model(0.0, self.dz_mat, self.r_f, self.m_f, self.sigma_phi)
            self.C00 = cumulative_trapz_prefix_2d_uniform(K00, self.dz)
            self.V0 = np.diag(self.C00).copy()
        else:
            self.C00 = None
            self.V0 = np.zeros(self.nz, dtype=float)

    def dDeltaPhi_eq18_matrix(self, R):
        r"""
        Eq. (18): D_{ΔΦ}(R,z1,z2) with LP16's 1/2 convention.
        Returns the matrix D_{ΔΦ}(R, z_i, z_j).
        """
        if self.sigma_phi <= 0.0:
            return np.zeros((self.nz, self.nz), dtype=float)

        K_R = xi_phi_eq14_model(R, self.dz_mat, self.r_f, self.m_f, self.sigma_phi)
        C_R = cumulative_trapz_prefix_2d_uniform(K_R, self.dz)

        D = 0.5 * (self.V0[:, None] + self.V0[None, :] - 2.0 * C_R)

        # Numerical cleanup
        np.maximum(D, 0.0, out=D)
        return D

    def corr_P_eq35_full_eta(self, R_values, eta):
        r"""
        Eq. (35) at fixed wavelength (single PSA slice), rewritten in eta:

            C_P(R; eta) = ∬_0^L dz1 dz2
                            xi_i(R, z1-z2)
                            exp[i * eta * beta * (z1-z2)]
                            exp[-eta^2 * Dhat_{ΔΦ}(R,z1,z2)]

        where:
            eta = 2 * sigma_phi * lambda^2
            beta = phi_bar / sigma_phi
            Dhat_{ΔΦ} = D_{ΔΦ} / sigma_phi^2

        For sigma_phi = 0, eta is effectively 0 and the turbulent Faraday factor is 1.
        """
        eta = float(eta)
        R_values = np.asarray(R_values, dtype=float)
        out = np.empty(R_values.size, dtype=np.complex128)

        # Phase term due to mean Faraday rotation (optional)
        # exp[2 i phi_bar lambda^2 (z1-z2)] = exp[i * eta * (phi_bar/sigma_phi) * (z1-z2)]
        if (self.sigma_phi > 0.0) and (self.beta != 0.0) and (eta != 0.0):
            phase = np.exp(1j * eta * self.beta * self.dz_mat)
        else:
            phase = 1.0

        for i, R in enumerate(R_values):
            xi_i_mat = xi_i_eq30(R, self.dz_mat,
                                 r_i=self.r_i, m=self.m_i,
                                 sigma_i=self.sigma_i, Pbar_i=self.Pbar_i)

            if (self.sigma_phi > 0.0) and (eta != 0.0):
                D_mat = self.dDeltaPhi_eq18_matrix(R)
                Dhat = D_mat / (self.sigma_phi ** 2)
                faraday = np.exp(-(eta ** 2) * Dhat)
            else:
                faraday = 1.0

            integrand = xi_i_mat * phase * faraday
            out[i] = np.sum(self.W2 * integrand)

        return out

    def D_P_from_corr(self, R_values, eta, use_eq44_eta0_fallback=True, eq44_prefactor=1.0):
        r"""
        Structure function of complex polarization:
            D_P(R;eta) = <|P1 - P2|^2> = 2 [ C_P(0;eta) - Re C_P(R;eta) ]

        For the isotropic/symmetric case used here, C_P is real (up to numerical noise).

        Parameters
        ----------
        R_values : array-like
            Transverse separation values.
        eta : float
            Faraday parameter: eta = 2 * sigma_phi * lambda^2
        use_eq44_eta0_fallback : bool
            If True and eta==0 (and no mean-Faraday phase), use the 1D weak/no-Faraday
            Eq.(44)-style integral for numerical robustness (Figure 5 limit).
        eq44_prefactor : float
            Use 1.0 to match your original Eq.(44) code convention exactly.
            Use 2.0 to match the direct square-domain z1,z2 integral convention from Eq.(35).

        Returns
        -------
        D_R : ndarray
            Structure function values.
        C_0_real : float
            Correlation at R=0.
        D_inf : float
            Structure function at R→∞ (2 * C_0_real).
        """
        R_values = np.asarray(R_values, dtype=float)
        eta_val = float(eta)

        # ---- robust eta=0 fallback (Fig.5 limit) ----
        if use_eq44_eta0_fallback and (abs(eta_val) == 0.0) and (self.beta == 0.0):
            # Mean-phase factor is unity when eta=0 regardless of beta,
            # but we only use fallback when beta=0 for exact match
            dz_grid = make_z_grid_for_eq44(self.L, nz=2001)
            C_R = eq44_prefactor * corr_P_eq44_negligible_faraday(
                R_values,
                r_i=self.r_i, m=self.m_i, L=self.L,
                sigma_i=self.sigma_i, Pbar_i=self.Pbar_i,
                z_grid=dz_grid
            )
            C_0 = eq44_prefactor * corr_P_eq44_negligible_faraday(
                np.array([0.0]),
                r_i=self.r_i, m=self.m_i, L=self.L,
                sigma_i=self.sigma_i, Pbar_i=self.Pbar_i,
                z_grid=dz_grid
            )[0]

            D_R = 2.0 * (float(C_0) - C_R)
            D_inf = 2.0 * float(C_0)
            return D_R, float(C_0), D_inf

        # ---- generic full Eq.(35) path ----
        C_R = self.corr_P_eq35_full_eta(R_values, eta=eta_val)
        C_0 = self.corr_P_eq35_full_eta(np.array([0.0]), eta=eta_val)[0]

        C_R_real = np.real(C_R)
        C_0_real = float(np.real(C_0))

        D_R = 2.0 * (C_0_real - C_R_real)
        D_inf = 2.0 * C_0_real  # if xi_i(R→∞)→0 and Pbar_i=0
        return D_R, C_0_real, D_inf


# ============================================================
# Eta=0 (Figure 5) asymptotics from LP16 Eq. (45)/(46)
# ============================================================

def D_P_asymptotic_eq45(R_values, r_i, m, L, sigma_i=1.0):
    r"""
    Eq. (45), m<1 form (Figure 5 negligible-Faraday asymptotic):
        D_P(R) ~ sigma_i^2 * L * R * (R/r_i)^mbar
    """
    mbar = min(float(m), 1.0)
    return sigma_i**2 * L * R_values * (R_values / r_i) ** mbar


def R_P_eq46(r_i, m, L):
    r"""
    Eq. (46), projected/observed correlation length in Figure 5 limit (eta=0):
        R_P ~ r_i * (L/r_i)^((1-mbar)/(1+mbar)), mbar=min(m,1)
    """
    mbar = min(float(m), 1.0)
    return r_i * (L / r_i) ** ((1.0 - mbar) / (1.0 + mbar))


# ============================================================
# Figure-5 style plotting, now with eta (full Eq. 35)
# ============================================================

def reproduce_figure5_full_eta(
    out_png="Figure5_full_eta.png",
    out_svg="Figure5_full_eta.svg",
    # Geometry / intrinsic polarization
    L_over_ri=100.0,
    r_i=1.0,
    sigma_i=1.0,
    Pbar_i=0.0,
    # RM-density fluctuation model (LP16 sigma_f -> sigma_phi here)
    r_f_over_ri=1.0,
    m_f=2.0/3.0,
    sigma_phi=1.0,
    phi_bar_over_sigma_phi=0.0,  # = \bar{phi} / sigma_phi ; set 0 for turbulent-only
    # Faraday variable(s)
    eta_left=0.0,                # eta = 2*sigma_phi*lambda^2 ; eta=0 recovers Fig. 5
    eta_right=0.0,               # same for right panel
    # Plot / numerics
    left_norm_factor=2.0,        # same convention as your original script
    left_asymptote_offset=3.0,   # Fig.5 caption note (offset dotted asymptote for clarity)
    nR_left=100,                 # reduced for speed
    nR_right=100,                # reduced for speed
    nz_full=501                  # not used with new implementation
):
    """
    Figure-5-style reproduction, using the corrected Δz + z2 formulation for Eq. (35).
    This ensures continuity as eta→0 and preserves small-R asymptotics for all η.

    IMPORTANT:
    - Eq. (45)/(46) asymptote/R_P markers are strictly the eta=0 (Fig. 5) limit.
      They are still plotted only as references (and only meaningful when eta_right=0).
    """
    L = L_over_ri * r_i
    r_f = r_f_over_ri * r_i
    
    # Use dense nonuniform Δz grid (key to correct small-R asymptotics)
    # Reduced nz for speed while maintaining accuracy
    delta_grid = make_z_grid_for_eq44(L, nz=1001, min_dz_ratio=1e-8)
    s_grid = delta_grid  # same grid for RM cumulants

    # Common x-axis: x = R/r_i
    x_left = np.logspace(-4, 2, nR_left)
    R_left = x_left * r_i

    x_right = np.logspace(-4, 2, nR_right)
    R_right = x_right * r_i

    # -----------------
    # Left panel: m_i = 2/3 (like Fig. 5)
    # -----------------
    m_left = 2.0 / 3.0
    
    D_left, C0_left, Dinf_left = D_P_full_eta_delta(
        R_left, eta=eta_left, beta=phi_bar_over_sigma_phi, L=L,
        r_i=r_i, m_i=m_left, sigma_i=sigma_i, Pbar_i=Pbar_i,
        r_f=r_f, m_f=m_f,
        delta_grid=delta_grid, s_grid=s_grid, nb_z2=400
    )

    # Eq. (45) asymptote is the eta=0 weak/no-Faraday Fig.5 asymptote
    D_asym_left = D_P_asymptotic_eq45(R_left, r_i=r_i, m=m_left, L=L, sigma_i=sigma_i)

    left_denom = left_norm_factor * sigma_i**2 * L**2
    y_left = D_left / left_denom
    y_asym_left = (left_asymptote_offset * D_asym_left) / left_denom

    # -----------------
    # Right panel: m_i = 1/3, 1/2, 4/5  (Fig. 5 style)
    # -----------------
    m_list = [1/3, 1/2, 4/5]
    colors = ["#5B7DB1", "#D9901A", "#88A92A"]
    curves_right = []
    Rp_markers = []

    for m_i in m_list:
        D, C0, Dinf = D_P_full_eta_delta(
            R_right, eta=eta_right, beta=phi_bar_over_sigma_phi, L=L,
            r_i=r_i, m_i=m_i, sigma_i=sigma_i, Pbar_i=Pbar_i,
            r_f=r_f, m_f=m_f,
            delta_grid=delta_grid, s_grid=s_grid, nb_z2=400
        )
        curves_right.append(D / Dinf)

        # Eq. (46) is an eta=0 reference (Figure 5 limit)
        Rp_markers.append(R_P_eq46(r_i, m_i, L) / r_i)

    # ========================================================
    # Plot styling
    # ========================================================
    fig = plt.figure(figsize=(13.8, 5.8))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.90, bottom=0.28, wspace=0.28)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1e2)
        ax.set_ylim(1e-8, 2)
        ax.tick_params(which="both", direction="in", labelsize=14, color="0.35")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # Left panel (full eta)
    axL.loglog(x_left, y_left, color="black", lw=2.2, zorder=5)

    # Show Fig.5 eta=0 asymptotic slope as a dotted reference
    mask_small = x_left <= 0.1
    idx = np.where(mask_small)[0][::4]
    axL.plot(x_left[idx], y_asym_left[idx], linestyle="None", marker="o",
             ms=5.0, color="#5F84B8", zorder=4)

    axL.text(-0.08, 1.05, r"$D_P(\mathrm{R})$", transform=axL.transAxes, fontsize=28)
    axL.text(1.03, -0.02, r"$\mathrm{R}/r_i$", transform=axL.transAxes, fontsize=28)
    axL.text(0.05, 0.90, rf"$\eta={eta_left:.3g}$", transform=axL.transAxes, fontsize=18)
    axL.text(0.11, 0.55, r"$\sim \mathbf{R}^{1+\bar m_i}$", transform=axL.transAxes, fontsize=24)

    # Right panel curves
    for y, c in zip(curves_right, colors):
        axR.loglog(x_right, y, color=c, lw=2.2, zorder=4)

    # y = 1/2 line
    axR.hlines(0.5, 1.0, 100.0, colors="black", lw=2.2, zorder=3)

    # Eq. (46) R_P markers (strictly eta=0 references)
    for xrp, c in zip(Rp_markers, colors):
        axR.vlines(xrp, 1e-8, 1e-5, colors=c, lw=2.0, zorder=2)

    axR.text(-0.17, 1.05, r"$D_P(\mathrm{R})/D_P(\infty)$", transform=axR.transAxes, fontsize=28)
    axR.text(1.03, -0.02, r"$\mathrm{R}/r_i$", transform=axR.transAxes, fontsize=28)
    axR.text(0.05, 0.90, rf"$\eta={eta_right:.3g}$", transform=axR.transAxes, fontsize=18)
    axR.text(0.06, 0.45, r"$\sim \mathbf{R}^{1+\bar m_i}$", transform=axR.transAxes, fontsize=24)
    axR.text(0.63, 0.47, r"$r_P/r_i$", transform=axR.transAxes, fontsize=22)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_svg)
    print(f"L/r_i = {L_over_ri:g}, r_f/r_i = {r_f_over_ri:g}, m_f = {m_f:g}")
    print(f"sigma_phi (LP16 sigma_f) = {sigma_phi:g}")
    print(f"eta_left = {eta_left:g}, eta_right = {eta_right:g}")
    if eta_right == 0.0:
        print("Eq.(46) R_P markers (eta=0 / Figure-5 reference):")
        for m, xrp in zip(m_list, Rp_markers):
            print(f"  m={m:.6g} -> R_P/r_i = {xrp:.6g}")


# ============================================================
# Convenience helpers: eta <-> lambda
# ============================================================

def eta_from_lambda(lambda_val, sigma_phi):
    """
    eta = 2 * sigma_phi * lambda^2
    """
    lam = np.asarray(lambda_val, dtype=float)
    return 2.0 * sigma_phi * lam * lam


def lambda_from_eta(eta, sigma_phi):
    """
    lambda = sqrt(eta / (2*sigma_phi))
    """
    eta = np.asarray(eta, dtype=float)
    if sigma_phi <= 0.0:
        raise ValueError("sigma_phi must be > 0 to invert eta -> lambda")
    return np.sqrt(eta / (2.0 * sigma_phi))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # --- Figure-5 limit (eta=0) ---
    reproduce_figure5_full_eta(
        out_png="Figure5_eta0_fullEq35.png",
        out_svg="Figure5_eta0_fullEq35.svg",
        L_over_ri=100.0,
        r_i=1.0,
        sigma_i=1.0,
        Pbar_i=0.0,
        r_f_over_ri=1.0,         # not relevant when eta=0, but kept for completeness
        m_f=2.0/3.0,
        sigma_phi=1.0,
        phi_bar_over_sigma_phi=0.0,
        eta_left=0.001,
        eta_right=0.001,
        left_norm_factor=2.0,
        left_asymptote_offset=3.0,
        nR_left=100,  # reduced for speed
        nR_right=100,  # reduced for speed
        nz_full=501   # not used with new implementation
    )

    # --- Example finite-Faraday slice (same code, now eta > 0) ---
    # Uncomment to generate a finite-eta version:
    # reproduce_figure5_full_eta(
    #     out_png="Figure5_eta1_fullEq35.png",
    #     out_svg="Figure5_eta1_fullEq35.svg",
    #     L_over_ri=100.0,
    #     r_i=1.0,
    #     sigma_i=1.0,
    #     Pbar_i=0.0,
    #     r_f_over_ri=1.0,
    #     m_f=2.0/3.0,
    #     sigma_phi=1.0,
    #     phi_bar_over_sigma_phi=0.0,  # set >0 to include mean Faraday phase
    #     eta_left=1.0,
    #     eta_right=1.0,
    #     left_norm_factor=2.0,
    #     left_asymptote_offset=3.0,
    #     nR_left=140,
    #     nR_right=140,
    #     nz_full=401
    # )