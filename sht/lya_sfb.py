"""
Spherical Fourier-Bessel estimator for the Lyman-α forest.

Implements the C_ℓ(k) pseudo-power spectrum estimator:
  1. Line-of-sight Fourier transform (per sightline, complex)
  2. Angular SHT per k-bin (Re/Im split for real-valued DirectSHT)
  3. Pseudo-C_ℓ(k) = (1/(2ℓ+1)) Σ_m |a_ℓm^data(k)|²

For Ly-α, the data weights are w_j × δ_F, so pseudo-Cl = |a_lm(data)|²
directly — no D−R subtraction needed.  The window alm (from randoms) is
computed separately for use in MaskDeconvolution / mode coupling.

Depends on: sht.sht.DirectSHT
"""

import numpy as np
import healpy as hp


class LyaSFB:
    """Spherical Fourier-Bessel C_ℓ(k) estimator for Ly-α forest sightlines."""

    def __init__(self, sht_engine, Nl):
        """
        Parameters
        ----------
        sht_engine : DirectSHT
            Pre-initialized DirectSHT instance.
        Nl : int
            Number of multipoles (ℓ from 0 to Nl-1).
        """
        self.sht = sht_engine
        self.Nl = Nl

    # ------------------------------------------------------------------ #
    #  Step 1: Line-of-sight Fourier transform                           #
    # ------------------------------------------------------------------ #
    def compute_los_ft(self, chi_grid, delta_F, K_j=None, k_arr=None,
                        apply_dchi=False):
        """
        Compute the LOS Fourier transform for all sightlines.

        Parameters
        ----------
        chi_grid : (Npix,) array
            Comoving distances along the LOS (same for all sightlines in the
            periodic-box case).
        delta_F : (Nsight, Npix) array
            Flux fluctuation δ_F for each sightline.
        K_j : (Nsight, Npix) array or None
            Per-pixel weights.  None → uniform (K=1).
        k_arr : (Nk,) array or None
            k-modes to evaluate.  None → FFT angular frequencies from chi_grid.
        apply_dchi : bool
            If True, multiply by dchi (continuous FT convention).
            If False (default), unnormalized DFT.

        Returns
        -------
        k_arr : (Nk,) array
            Wavenumber array (h/Mpc, angular frequency = 2π × cycles).
        delta_2d : (Nsight, Nk) complex array
            Fourier-weighted flux: Σ_α K_j(χ_α) δ_F(χ_α) e^{ikχ_α} [× Δχ]
        K_tilde : (Nsight, Nk) complex array
            Fourier-weighted window: Σ_α K_j(χ_α) e^{ikχ_α} [× Δχ]
        """
        Nsight, Npix = delta_F.shape
        dchi = chi_grid[1] - chi_grid[0]

        if k_arr is None:
            k_arr = 2.0 * np.pi * np.fft.fftfreq(Npix, d=dchi)

        if K_j is None:
            K_j = np.ones_like(delta_F)

        # Phase matrix: (Nk, Npix)
        phase = np.exp(1j * np.outer(k_arr, chi_grid))  # e^{i k chi}

        # Weighted fields: (Nsight, Npix)
        weighted_delta = K_j * delta_F  # K_j * delta_F

        # Matrix multiply: (Nsight, Npix) @ (Npix, Nk) → (Nsight, Nk)
        norm = dchi if apply_dchi else 1.0
        delta_2d = (weighted_delta @ phase.T) * norm
        K_tilde = (K_j @ phase.T) * norm

        return k_arr, delta_2d, K_tilde

    # ------------------------------------------------------------------ #
    #  Step 2: SHT per k-bin  (handles Re/Im split)                      #
    # ------------------------------------------------------------------ #
    def sht_per_k(self, theta, phi, delta_2d_k, K_tilde_k):
        """
        Run DirectSHT for one k-bin with Re/Im split.

        Parameters
        ----------
        theta, phi : (Nsight,) arrays
            Sky positions of sightlines (radians).
        delta_2d_k : (Nsight,) complex array
            Fourier-weighted flux at this k.
        K_tilde_k : (Nsight,) complex array
            Fourier-weighted window at this k.

        Returns
        -------
        alm_data : (Nlm,) complex array
            a_ℓm^f(k) = a_ℓm^{Re} + i a_ℓm^{Im}
        alm_rand : (Nlm,) complex array
            w_ℓm(k) = w_ℓm^{Re} + i w_ℓm^{Im}
        """
        # Data SHT
        alm_re = self.sht(theta, phi, np.real(delta_2d_k))
        alm_im = self.sht(theta, phi, np.imag(delta_2d_k))
        alm_data = alm_re + 1j * alm_im

        # Window (randoms) SHT
        wlm_re = self.sht(theta, phi, np.real(K_tilde_k))
        wlm_im = self.sht(theta, phi, np.imag(K_tilde_k))
        alm_rand = wlm_re + 1j * wlm_im

        return alm_data, alm_rand

    # ------------------------------------------------------------------ #
    #  Step 3: Pseudo-C_ℓ(k)                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def pseudo_cl(alm_data, Nl):
        """
        Compute pseudo-C_ℓ(k) = (1/(2ℓ+1)) Σ_m |a_ℓm^data(k)|².

        For Ly-α forest the data weights are w_j × δ_F, so the
        correlation function is <DD> = <(w δ_F)(w δ_F)> directly,
        with no D−R subtraction needed.

        Parameters
        ----------
        alm_data : complex array (Nlm,)
            Healpix-convention alm from data SHT.
        Nl : int
            Number of multipoles.

        Returns
        -------
        cl : (Nl,) array
            Pseudo-power spectrum at this k.
        """
        cl = _alm2cl_complex(alm_data, Nl)
        return cl

    # ------------------------------------------------------------------ #
    #  Full pipeline: all k-bins                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_angular_window(sht_engine, theta, phi, Nsight, Nl):
        """
        Compute the angular window spectrum from uniform sightline weights.

        W_l = (1/(2l+1)) Σ_m |u_lm|^2 where u_lm = SHT(ones).
        Also returns the shot noise: sn = N_skew / (4π).
        """
        u_lm = sht_engine(theta, phi, np.ones(Nsight))
        W_l = _alm2cl_complex(u_lm, Nl)
        sn = Nsight / (4.0 * np.pi)
        return W_l, sn, u_lm

    def compute_all_cl_k(self, theta, phi, chi_grid, delta_F, K_j=None,
                         k_arr=None, k_indices=None):
        """
        Compute pseudo-C_ℓ(k) for all (or selected) k-bins.

        Parameters
        ----------
        theta, phi : (Nsight,) arrays
        chi_grid : (Npix,) array
        delta_F : (Nsight, Npix) array
        K_j : optional weight array
        k_arr : optional k-grid
        k_indices : optional list of k-bin indices to compute

        Returns
        -------
        k_arr : (Nk,) array
        cl_k : (Nk_sel, Nl) array
            Pseudo-C_ℓ at each selected k.
        wl_k : (Nk_sel, Nl) array
            Window power spectrum at each selected k.
        alm_data_all : list of (Nlm,) complex arrays
        alm_rand_all : list of (Nlm,) complex arrays
        """
        k_arr, delta_2d, K_tilde = self.compute_los_ft(
            chi_grid, delta_F, K_j=K_j, k_arr=k_arr)

        Nk = len(k_arr)
        if k_indices is None:
            k_indices = range(Nk)

        cl_k = np.zeros((len(k_indices), self.Nl))
        wl_k = np.zeros((len(k_indices), self.Nl))
        alm_data_all = []
        alm_rand_all = []

        for i, ki in enumerate(k_indices):
            alm_data, alm_rand = self.sht_per_k(
                theta, phi, delta_2d[:, ki], K_tilde[:, ki])
            alm_data_all.append(alm_data)
            alm_rand_all.append(alm_rand)

            cl_k[i, :] = self.pseudo_cl(alm_data, self.Nl)

            # Window spectrum from randoms
            wl_k[i, :] = _alm2cl_complex(alm_rand, self.Nl)

        return k_arr, cl_k, wl_k, alm_data_all, alm_rand_all


def _alm2cl_complex(alm, Nl):
    """
    Compute C_ℓ = (1/(2ℓ+1)) Σ_m |a_ℓm|² for complex alm
    stored in Healpix convention (m ≥ 0 only).

    For "doubly complex" alm (from Re+iIm SHT), we need:
      |a_ℓm|² = Re(a_ℓm)² + Im(a_ℓm)²
    where Re and Im here refer to the Healpix complex number,
    not the Re/Im Fourier split (that's already combined into the complex alm).
    """
    cl = np.zeros(Nl)
    for ell in range(Nl):
        idx0 = (0 * (2 * Nl - 1 - 0)) // 2 + ell  # m=0
        val = np.abs(alm[idx0])**2
        for m in range(1, ell + 1):
            idx = (m * (2 * Nl - 1 - m)) // 2 + ell
            val += 2.0 * np.abs(alm[idx])**2
        cl[ell] = val / (2.0 * ell + 1.0)
    return cl
