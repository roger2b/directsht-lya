"""
Theory C_ℓ(k) prediction for the Lyman-α forest.

Two approaches for comparing theory to measured pseudo-C_ℓ:

1. **Pair-counting** (original pipeline, ``theory_pseudo_cl``):
   Uses Wigner 3j coupling with P_F as weights × pair-counting angular window.
   Result matches ``hp.alm2cl(hdat)`` directly (no deconvolution needed).

2. **MaskDeconvolution** (new modular approach, ``theory_cl_for_deconvolution``):
   Uses C_true = P_F(ℓ/χ̄, k) / (32π³ χ̄²) with MaskDeconvolution's
   ``convolve_theory_Cls`` for binned + deconvolved comparison.

The two are algebraically equivalent (verified numerically).

P_F is the anisotropic 3D Ly-α flux power spectrum in the Kaiser
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


def theory_pseudo_cl(ell_arr, k_par, chi_bar, plin_interp,
                     coupling_matrix_pk, angular_window_PL,
                     b1=1.0, beta=0.0):
    """
    Compute the theory prediction for the pseudo-C_ℓ(k) using the
    pair-counting approach from the original pipeline.

    The pseudo-C_ℓ as measured is:

        pseudo-Cl_plotted = C_theory / (4π)²

    where:

        C_theory = M_pk @ PL / (4π × 2π × chi_bar²)

    Here M_pk is the coupling matrix with P_F(λ/chi_bar, k) as weights,
    and PL is the pair-counting angular window:
        PL[λ] = Σ_{j,k} K_j K_k P_λ(cos θ_{jk})

    Parameters
    ----------
    ell_arr : (Nl,) array
        Multipole values.
    k_par : float
        LOS wavenumber (h/Mpc). For the original code at k_idx,
        use k_arr[k_idx] (in cycles per distance).
    chi_bar : float
        Mean comoving distance (Mpc/h).
    plin_interp : callable
        P_lin(k) interpolator.
    coupling_matrix_pk : (Nl, lambda_max) array
        Wigner 3j coupling matrix with P_F(λ/chi_bar, k) as input weights.
    angular_window_PL : (lambda_max,) array
        Pair-counting angular window.
    b1, beta : float
        Ly-α bias parameters.

    Returns
    -------
    pseudo_cl_predicted : (Nl,) array
        Theory pseudo-C_ℓ that should match measured hp.alm2cl(hdat).
    """
    Nl = len(ell_arr)
    C_theory = (coupling_matrix_pk[:Nl, :] @ angular_window_PL
                / (4.0 * np.pi) / (2.0 * np.pi * chi_bar**2))
    # The measured pseudo-Cl matches C_theory / (4π)²
    return C_theory / (4.0 * np.pi)**2


def theory_cl_for_deconvolution(ell_arr, k_par, chi_bar, plin_interp,
                                 b1=1.0, beta=0.0):
    """
    Theory C_ℓ(k) suitable for use with MaskDeconvolution.convolve_theory_Cls.

    The MaskDeconvolution mode-coupling matrix Mll relates the true C_ℓ to
    the pseudo-C_ℓ via:

        <pseudo-Cl> = Mll @ C_true

    The correct C_true for this framework is:

        C_true[ℓ] = P_F(ℓ/χ̄, k) / (32 π³ χ̄²)

    The factor 1/(32π³) arises from the normalization chain:
    - The pair-counting angular window PLKjKk[λ] = 4π × wl_ref[λ]
    - The coupling identity M_pk @ PLKjKk = 4π × Mll @ pk
    - The original plotting convention C_plot = C_theory/(4π)²

    Combined: C_plot = Mll @ pk / (32π³χ²), so C_true = pk / (32π³χ²).

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
    cl_true : (Nell,) array
        Theory C_ℓ(k) for use with MaskDeconvolution forward model.
    """
    pf = P_flux(ell_arr / chi_bar, k_par, plin_interp, b1=b1, beta=beta)
    return pf / (32.0 * np.pi**3 * chi_bar**2)
