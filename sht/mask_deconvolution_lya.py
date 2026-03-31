"""
k-dependent mode-coupling matrix for the Ly-α sFB estimator.

Extends MaskDeconvolution to handle k-dependent window spectra W_λ(k),
noise floor subtraction, and per-k-bin mode decoupling.

Reuses the Wigner 3j symbols from the upstream code.
"""

import numpy as np
from sht.threej000 import Wigner3j


class MaskDeconvolutionLya:
    """k-dependent mode-coupling for the Ly-α C_ℓ(k) estimator."""

    def __init__(self, Nl, verbose=True):
        """
        Parameters
        ----------
        Nl : int
            Number of multipoles (ℓ from 0 to Nl-1).
        verbose : bool
            Print progress messages.
        """
        self.Nl = Nl
        self.lmax = Nl - 1
        if verbose:
            print("Precomputing Wigner 3j symbols...")
        self.w3j000 = Wigner3j(2 * Nl - 1)
        if verbose:
            print("Done.")

    # ------------------------------------------------------------------ #
    #  Window spectrum and noise floor                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_Wl(alm_rand, Nl):
        """
        Window power spectrum from randoms alm.

        W_λ = (1/(2λ+1)) Σ_m |w_{λm}|²
        """
        from sht.lya_sfb import _alm2cl_complex
        return _alm2cl_complex(alm_rand, Nl)

    @staticmethod
    def noise_floor_w(K_tilde_k):
        """
        Mask shot noise: N_w(k) = (1/4π) Σ_j |K̃_j(k)|²

        Parameters
        ----------
        K_tilde_k : (Nsight,) complex array
            Fourier-weighted window weights at this k-bin.

        Returns
        -------
        Nw : float
        """
        return np.sum(np.abs(K_tilde_k)**2) / (4.0 * np.pi)

    @staticmethod
    def noise_floor_f(delta_2d_k):
        """
        Field shot noise: N_f(k) = (1/4π) Σ_j |δ_2D(n̂_j; k)|²
        """
        return np.sum(np.abs(delta_2d_k)**2) / (4.0 * np.pi)

    # ------------------------------------------------------------------ #
    #  Mode-coupling matrix (per k-bin)                                   #
    # ------------------------------------------------------------------ #
    def get_M(self, W_l):
        """
        Compute the mode-coupling matrix M_ℓℓ' for a given window spectrum.

        M_{ℓ₁,ℓ₂} = (2ℓ₂+1)/(4π) Σ_{ℓ₃} (2ℓ₃+1) W(ℓ₃) [3j(ℓ₁,ℓ₂,ℓ₃;0,0,0)]²

        Parameters
        ----------
        W_l : (Nw,) array
            Window power spectrum (possibly noise-subtracted).

        Returns
        -------
        M : (Nl, Nl) array
            Mode-coupling matrix.
        """
        M = np.zeros((self.lmax + 1, self.lmax + 1))
        Nw = len(W_l)
        for l1 in range(self.lmax + 1):
            for l2 in range(self.lmax + 1):
                val = 0.0
                l3_min = abs(l1 - l2)
                l3_max = min(l1 + l2, Nw - 1)
                for l3 in range(l3_min, l3_max + 1):
                    if (l1 + l2 + l3) % 2 == 0:
                        val += (2 * l3 + 1) * W_l[l3] * self.w3j000(l1, l2, l3)**2
                M[l1, l2] = val * (2 * l2 + 1)
        M /= 4.0 * np.pi
        return M

    # ------------------------------------------------------------------ #
    #  Binning and decoupling                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def binning_matrix(Nl, step=32, start=0):
        """
        Linear binning matrix.

        Returns
        -------
        bins : (Nbin, Nl) array
        """
        bins_list = []
        l0 = start
        while l0 + step <= Nl:
            row = np.zeros(Nl)
            row[l0:l0 + step] = 1.0 / step
            bins_list.append(row)
            l0 += step
        return np.array(bins_list)

    def decouple(self, cl, Mll, bins):
        """
        Bin and mode-decouple a pseudo-C_ℓ spectrum.

        Parameters
        ----------
        cl : (Nl,) array
        Mll : (Nl, Nl) array
        bins : (Nbin, Nl) array

        Returns
        -------
        binned_ells : (Nbin,) array
        cl_decoupled : (Nbin,) array
        """
        Cb = bins @ cl[:self.Nl]
        binned_ells = bins @ np.arange(self.Nl) / np.sum(bins, axis=1)

        # Bin-bin coupling
        bins_no_wt = np.zeros_like(bins)
        bins_no_wt[bins > 0] = 1.0
        Mbb = bins @ Mll @ bins_no_wt.T
        Mbb_inv = np.linalg.inv(Mbb)
        cl_decoupled = Mbb_inv @ Cb
        return binned_ells, cl_decoupled

    def convolve_theory(self, cl_theory, Mll, bins):
        """
        Window-convolve a theory C_ℓ and bin.

        Returns
        -------
        binned_ells : (Nbin,) array
        cl_conv : (Nbin,) array
        """
        binned_ells = bins @ np.arange(self.Nl) / np.sum(bins, axis=1)
        # Convolve: Ĉ_ℓ = Σ_L M_ℓL C_L^theory
        cl_pseudo = Mll @ cl_theory[:self.Nl]
        cl_conv = bins @ cl_pseudo
        return binned_ells, cl_conv
