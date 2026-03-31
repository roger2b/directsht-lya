"""
Theory C_ℓ(k) prediction for the Lyman-α forest.

Implements:
  C_L^theory(k) = P_F(k_perp = L/chi_bar, k_par = k) / chi_bar^2

where P_F is the anisotropic 3D Ly-α flux power spectrum in the Kaiser
approximation:
  P_F(k_perp, k_par) = b1^2 (1 + beta mu^2)^2 P_lin(k)
"""

import numpy as np
from scipy.interpolate import interp1d


def P_flux(k_perp, k_par, plin_interp, b1=-0.1521, beta=0.2298):
    """
    Anisotropic Ly-α flux power spectrum (Kaiser approximation).

    Parameters
    ----------
    k_perp : array_like
        Transverse wavenumber (h/Mpc).
    k_par : float or array_like
        Line-of-sight wavenumber (h/Mpc).
    plin_interp : callable
        Interpolator for P_lin(k) at the target redshift.
    b1 : float
        Linear flux bias.
    beta : float
        Kaiser RSD parameter.

    Returns
    -------
    P_F : array
        Anisotropic power spectrum.
    """
    k_perp = np.atleast_1d(np.asarray(k_perp, dtype=float))
    k_par = np.atleast_1d(np.asarray(k_par, dtype=float))
    k = np.sqrt(k_perp**2 + k_par**2)
    mu = np.where(k > 0, k_par / k, 0.0)
    pk = plin_interp(k)
    return b1**2 * (1.0 + beta * mu**2)**2 * pk


def theory_cl_k(ell_arr, k_par, chi_bar, plin_interp,
                b1=-0.1521, beta=0.2298):
    """
    Theory angular power spectrum C_L(k) in the flat-sky (Limber-like) limit.

    C_L(k) = P_F(k_perp = L/chi_bar, k_par = k) / chi_bar^2

    Parameters
    ----------
    ell_arr : (Nell,) array
        Multipole values.
    k_par : float
        LOS wavenumber (h/Mpc).
    chi_bar : float
        Mean comoving distance (Mpc/h).
    plin_interp : callable
        P_lin(k) interpolator.
    b1, beta : float
        Ly-α bias parameters.

    Returns
    -------
    cl : (Nell,) array
        Theory C_ℓ(k).
    """
    ell_arr = np.asarray(ell_arr, dtype=float)
    k_perp = ell_arr / chi_bar
    pf = P_flux(k_perp, k_par, plin_interp, b1=b1, beta=beta)
    return pf / chi_bar**2


def compute_chi_bar_from_grid(chi_grid):
    """Mean comoving distance from a uniform LOS grid."""
    return 0.5 * (chi_grid.min() + chi_grid.max())
