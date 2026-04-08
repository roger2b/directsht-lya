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
    def compute_los_ft(self, chi_grid, delta_F, K_j=None, k_arr=None):
        """
        Compute the LOS Fourier transform for all sightlines.

        Uses the K_j = 1/L normalization convention:
            δ̃_j(k) = (Δχ/L) Σ_α K_j(χ_α) δ_F(χ_α) e^{ikχ_α}
        so that K̃_j(k=0) = 1 for uniform weights.

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

        Returns
        -------
        k_arr : (Nk,) array
            Wavenumber array (h/Mpc, angular frequency = 2π × cycles).
        delta_2d : (Nsight, Nk) complex array
            Fourier-weighted flux: (Δχ/L) Σ_α K_j(χ_α) δ_F(χ_α) e^{ikχ_α}
        K_tilde : (Nsight, Nk) complex array
            Fourier-weighted window: (Δχ/L) Σ_α K_j(χ_α) e^{ikχ_α}
        """
        Nsight, Npix = delta_F.shape
        dchi = chi_grid[1] - chi_grid[0]
        L = Npix * dchi

        if k_arr is None:
            k_arr = 2.0 * np.pi * np.fft.fftfreq(Npix, d=dchi)

        if K_j is None:
            K_j = np.ones_like(delta_F)

        # Phase matrix: (Nk, Npix)
        phase = np.exp(1j * np.outer(k_arr, chi_grid))  # e^{i k chi}

        # Weighted fields: (Nsight, Npix)
        weighted_delta = K_j * delta_F  # K_j * delta_F

        # Matrix multiply: (Nsight, Npix) @ (Npix, Nk) → (Nsight, Nk)
        norm = dchi / L  # K_j = 1/L convention: K̃(k=0) = 1
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
        alm_data_re, alm_data_im : (Nlm,) complex arrays
            SHT of real / imaginary parts of data weights.
        alm_rand_re, alm_rand_im : (Nlm,) complex arrays
            SHT of real / imaginary parts of window weights.
        """
        # Data SHT
        alm_data_re = self.sht(theta, phi, np.real(delta_2d_k))
        alm_data_im = self.sht(theta, phi, np.imag(delta_2d_k))

        # Window (randoms) SHT
        alm_rand_re = self.sht(theta, phi, np.real(K_tilde_k))
        alm_rand_im = self.sht(theta, phi, np.imag(K_tilde_k))

        return alm_data_re, alm_data_im, alm_rand_re, alm_rand_im

    # ------------------------------------------------------------------ #
    #  Step 3: Pseudo-C_ℓ(k)                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def pseudo_cl(alm_re, alm_im, Nl):
        """
        Compute pseudo-C_ℓ(k) = (1/(2ℓ+1)) Σ_m |a_ℓm^data(k)|² for a
        complex-weighted field.

        For k≠0 the SHT weights are complex, so the alm splits into
        a_ℓm = a_ℓm^Re + i a_ℓm^Im  (each itself complex in HEALPix
        convention).  For real-valued SHTs, the HEALPix relation
        a_{ℓ,-m} = (-1)^m a*_{ℓm} holds independently for the Re and Im
        parts.  This means:
            |a_{ℓm}|² + |a_{ℓ,-m}|² = 2(|a^Re_{ℓm}|² + |a^Im_{ℓm}|²)
        i.e. the cross-terms cancel in the m + (-m) sum.  Therefore the
        correct formula is C_ℓ = C_ℓ^{Re} + C_ℓ^{Im}, computed from
        separate calls to _alm2cl_complex.

        Using _alm2cl_complex on the combined (a^Re + i a^Im) would
        introduce an erroneous cross-term -2 Im[a^Re (a^Im)*] per m>0.

        Parameters
        ----------
        alm_re : complex array (Nlm,)
            Healpix alm from SHT of Re[weights].
        alm_im : complex array (Nlm,)
            Healpix alm from SHT of Im[weights].
        Nl : int
            Number of multipoles.

        Returns
        -------
        cl : (Nl,) array
            Pseudo-power spectrum at this k.
        """
        cl = _alm2cl_complex(alm_re, Nl) + _alm2cl_complex(alm_im, Nl)
        return cl

    # ------------------------------------------------------------------ #
    #  Full pipeline: all k-bins                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_angular_window(sht_engine, theta, phi, Nsight, Nl):
        """
        Compute the angular window spectrum from uniform sightline weights.

        W_l = (1/(2l+1)) Σ_m |u_lm|^2 where u_lm = SHT(ones).

        Note: Nsight/(4π) is NOT a separate shot-noise term to subtract.
        For Ly-α with fixed sightline positions, the j=k diagonal of the
        pair-counting sum is cosmological signal, not Poisson noise.
        The MASTER framework <pseudo-Cl> = Mll @ C_true already includes
        this contribution through the full window function W_l.
        """
        u_lm = sht_engine(theta, phi, np.ones(Nsight))
        W_l = _alm2cl_complex(u_lm, Nl)
        return W_l, u_lm

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
        wfloor_k : (Nk_sel,) array
            Window floor per k: (1/4π) Σ_j |K̃_j(k)|².
        """
        k_arr, delta_2d, K_tilde = self.compute_los_ft(
            chi_grid, delta_F, K_j=K_j, k_arr=k_arr)

        Nk = len(k_arr)
        if k_indices is None:
            k_indices = range(Nk)

        cl_k = np.zeros((len(k_indices), self.Nl))
        wl_k = np.zeros((len(k_indices), self.Nl))
        wfloor_k = np.zeros(len(k_indices))

        for i, ki in enumerate(k_indices):
            alm_d_re, alm_d_im, alm_r_re, alm_r_im = self.sht_per_k(
                theta, phi, delta_2d[:, ki], K_tilde[:, ki])

            cl_k[i, :] = self.pseudo_cl(alm_d_re, alm_d_im, self.Nl)
            wl_k[i, :] = _alm2cl_complex(alm_r_re, self.Nl) + \
                         _alm2cl_complex(alm_r_im, self.Nl)
            wfloor_k[i] = np.sum(np.abs(K_tilde[:, ki])**2) / (4.0 * np.pi)

        return k_arr, cl_k, wl_k, wfloor_k


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
