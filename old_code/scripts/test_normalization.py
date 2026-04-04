#!/usr/bin/env python
"""
Focused normalization test.

Compares the ORIGINAL code's measurement+theory pipeline against the NEW
modular code, to pin down the exact normalization factor needed.

Runs at k_idx=0 (simplest case: FT is real, no Re/Im split needed).
"""

import sys, os
import numpy as np
import healpy as hp

# ----- paths --------------------------------------------------------------- #
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

# ----- imports ------------------------------------------------------------- #
from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
from sht.lya_sfb import LyaSFB, _alm2cl_complex
from sht.theory_lya import P_flux, theory_cl_k, compute_chi_bar_from_grid

# ----- settings ------------------------------------------------------------ #
chi_shift = 5000
Nl        = 200
lambda_max = 200
L_max     = 200
seed      = 1000
num_qso   = 5000          # NxN sightlines → ~GRF.N^2 = 512^2 if enough
k_idx     = 0             # test at k=0 first
add_rsd_  = False
NperBin   = 32

# ----- DirectSHT instance ------------------------------------------------- #
Nx   = 2 * Nl
xmax = 0.75
sht  = DirectSHT(Nl, Nx, xmax)
print(f"DirectSHT: Nl={Nl}, Nx={Nx}, xmax={xmax}")

# ----- GRF ----------------------------------------------------------------- #
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew=num_qso, shift=chi_shift)
all_theta, all_phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
chi_grid = all_x[0, :]
all_w_gal = all_w_gal - 1.0  # delta_F = (1+delta) - 1
print(f"Nskew = {Nskew}, Npix = {chi_grid.size}")

# ================================================================= #
#  APPROACH 1: Exact replica of original code (pair-counting theory) #
# ================================================================= #
print("\n=== APPROACH 1: Original code (real-only, pair-counting) ===")

# ----- step 1: DFT (original convention) ---------------------------------- #
k_arr_orig, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)
print(f"k_arr_orig[0:5] = {k_arr_orig[:5]}")
print(f"  (these are in cycles/(Mpc/h), NOT h/Mpc)")
dchi = chi_grid[1] - chi_grid[0]
L_box = chi_grid.size * dchi
print(f"L_box = {L_box:.1f} Mpc/h, dchi = {dchi:.4f} Mpc/h")

# ----- step 2: SHT on real-part only (original) --------------------------- #
hdat_orig = sht(all_theta, all_phi, FT_delta[:, k_idx])  # FT_delta already .real from compute_dft
hran_orig = sht(all_theta, all_phi, FT_mask[:, k_idx])

# ----- step 3: measured Cl (original) ------------------------------------- #
cl_data_orig = hp.alm2cl(hdat_orig)[:Nl]
cl_rand_orig = hp.alm2cl(hran_orig)[:Nl]
print(f"cl_data_orig[5] = {cl_data_orig[5]:.6e}")
print(f"cl_rand_orig[0] = {cl_rand_orig[0]:.6e}")

# ----- step 4: Theory (original pair-counting approach) -------------------- #
chi_bar = compute_chi_bar_from_grid(chi_grid)
L_range = np.arange(0, L_max, 1)

# Original uses k_arr_orig which is in cycles, NOT h/Mpc
# But with add_rsd_=False, Kaiser=1 and kh_par doesn't matter
# at k_idx=0, kh_par=0 regardless
def Power_spectrum_orig(kh_perp, kh_par):
    kh = np.sqrt(kh_par**2 + kh_perp**2)
    pk = GRF.plin(kh)
    return pk  # no Kaiser since add_rsd_=False

pk_L = Power_spectrum_orig(kh_perp=L_range / chi_bar, kh_par=k_arr_orig[k_idx])

# Pair counting
nhat = sht_lya.compute_nhat(all_theta, all_phi)
cos_theta_njnk = np.dot(nhat, nhat.T)
KjKk = GRF.N**2  # periodic case
print("Computing pair-counting Legendre sums... ", end="", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta_njnk, KjKk)[:-1]
print("done")

# Coupling matrix with P(k) as weights
couple_mat_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
coupling_matrix_pk_L = couple_mat_pk.compute_matrix()

# Theory C_ell
C_ell_theory_orig = np.dot(coupling_matrix_pk_L, PLKjKk) / (4.0 * np.pi) / (2.0 * np.pi * chi_bar**2)

# Binning
couple_mat_win = Wigner3j.CoupleMat(Nl, cl_rand_orig)
coupling_matrix_win = couple_mat_win.compute_matrix()
MD = MaskDeconvolution(Nl, cl_rand_orig, precomputed_Wigner=coupling_matrix_win)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl)
binned_ells = bins @ ells

# Binned theory (matching the original code's plot convention)
binned_C_ell_orig = bins @ C_ell_theory_orig[:Nl]
binned_cl_data_orig = bins @ cl_data_orig

# The original code plots theory / (4pi)^2
theory_plot_orig = binned_C_ell_orig / (4.0 * np.pi)**2

print(f"\nOriginal code results (binned):")
print(f"  theory (plotted) [5] = {theory_plot_orig[1]:.6e}")
print(f"  measured         [5] = {binned_cl_data_orig[1]:.6e}")
print(f"  ratio meas/theo =      {binned_cl_data_orig[1] / theory_plot_orig[1]:.4f}")

print(f"\nRatios at all bins:")
for i in range(min(6, len(binned_ells))):
    if theory_plot_orig[i] > 0:
        r = binned_cl_data_orig[i] / theory_plot_orig[i]
        print(f"  ell_bin={binned_ells[i]:.0f}: ratio = {r:.4f}")


# ================================================================= #
#  APPROACH 2: New modular code with Re+Im SHT                      #
# ================================================================= #
print("\n=== APPROACH 2: New modular code (Re+Im SHT) ===")

sfb = LyaSFB(sht, Nl)

# ----- FT using new code (unnormalized, apply_dchi=False) ------------------ #
k_arr_new, delta_2d, K_tilde = sfb.compute_los_ft(
    chi_grid, all_w_gal, K_j=all_w_rand, k_arr=None, apply_dchi=False)
print(f"k_arr_new[0:5] = {k_arr_new[:5]}")
print(f"  (these are angular frequency in h/Mpc)")

# Map k_idx=0 from original to new: k=0 for both
ki_new = 0  # k=0 is also index 0 in new array

# ----- Check: Re-only match with original --------------------------------- #
# The new code's delta_2d[:, 0] should have only real part (at k=0)
print(f"\ndelta_2d[:5, 0] = {delta_2d[:5, 0]}")
print(f"FT_delta[:5, 0] = {FT_delta[:5, 0]}")
print(f"Max |Im(delta_2d[:,0])| = {np.max(np.abs(np.imag(delta_2d[:, 0]))):.2e}")

# ----- SHT per k: Re+Im split --------------------------------------------- #
alm_data_new, alm_rand_new = sfb.sht_per_k(
    all_theta, all_phi, delta_2d[:, ki_new], K_tilde[:, ki_new])

# Complex alm → Cl
cl_data_new = _alm2cl_complex(alm_data_new, Nl)
cl_rand_new = _alm2cl_complex(alm_rand_new, Nl)

# Also compute Re-only for comparison
alm_re_only = sht(all_theta, all_phi, np.real(delta_2d[:, ki_new]))
cl_re_only = hp.alm2cl(alm_re_only)[:Nl]

print(f"\ncl_data_new[5] (Re+Im)   = {cl_data_new[5]:.6e}")
print(f"cl_re_only[5] (Re only)  = {cl_re_only[5]:.6e}")
print(f"cl_data_orig[5] (orig)   = {cl_data_orig[5]:.6e}")
print(f"ratio Re+Im / Re-only    = {cl_data_new[5] / cl_re_only[5]:.4f}")
print(f"ratio Re-only / orig     = {cl_re_only[5] / cl_data_orig[5]:.4f}")

# ----- Angular window from unit weights ------------------------------------ #
W_l_unit, sn_unit, u_lm = sfb.compute_angular_window(sht, all_theta, all_phi, Nskew, Nl)
print(f"\nAngular window W_0 (unit weights) = {W_l_unit[0]:.2e}")
print(f"Shot noise = Nskew/(4pi) = {sn_unit:.2e}")
print(f"Nskew^2/(4pi) = {Nskew**2/(4*np.pi):.2e}")

# ----- MaskDeconvolution approach ------------------------------------------ #
# Build mode-coupling from the angular window (unit weights)
MD_new = MaskDeconvolution(Nl, W_l_unit)
bins_new = MD_new.binning_matrix('linear', 0, NperBin)
Mbl = MD_new.window_matrix(bins_new)
binned_ells_new = bins_new @ ells

# Theory: C_L^true(k) for the new code
# With unnormalized DFT and Re+Im, what should the theory be?
ell_arr = np.arange(Nl, dtype=float)
k_par_test = 0.0  # k=0

# Simple theory: P_F(ell/chi_bar, k) / chi_bar^2
cl_theory_simple = theory_cl_k(ell_arr, k_par_test, chi_bar, GRF.plin,
                                b1=1.0, beta=0.0)  # no bias/RSD since add_rsd_=False
# Note: with b1=1.0 and beta=0, P_F = P_lin (same as Power_spectrum_orig)

print(f"\ncl_theory_simple[5] = {cl_theory_simple[5]:.6e}")

# Compute theory convolved with the angular window
binned_ells_conv, binned_theory_conv = MD_new.convolve_theory_Cls(
    cl_theory_simple, bins_new)

binned_cl_new = bins_new @ cl_data_new
binned_cl_re = bins_new @ cl_re_only

print(f"\nNew code results (binned, window-convolved theory):")
for i in range(min(6, len(binned_ells_conv))):
    if binned_theory_conv[i] > 0:
        r_reim = binned_cl_new[i] / binned_theory_conv[i]
        r_re = binned_cl_re[i] / binned_theory_conv[i]
        print(f"  ell={binned_ells_conv[i]:.0f}: "
              f"meas(Re+Im)={binned_cl_new[i]:.4e}, "
              f"theory_conv={binned_theory_conv[i]:.4e}, "
              f"ratio(Re+Im)={r_reim:.2f}, ratio(Re)={r_re:.2f}")

# ----- Try theory with various normalization factors ----------------------- #
print("\n=== Testing theory normalization factors ===")
# Factor candidates:
# L_box factor (from FT integral length)
# N factor (from unnormalized DFT → each weight is sum of N pixels)
# N^2 factor (pair counting has N^2 weight for periodic)

for label, factor in [
    ("P_F/chi^2", 1.0),
    ("L_box * P_F/chi^2", L_box),
    ("N * P_F/chi^2", chi_grid.size),
    ("N^2 * P_F/chi^2", chi_grid.size**2),
    ("N * dchi * P_F/chi^2", chi_grid.size * dchi),
]:
    cl_test = cl_theory_simple * factor
    _, binned_test = MD_new.convolve_theory_Cls(cl_test, bins_new)
    r = binned_cl_re[1] / binned_test[1] if binned_test[1] > 0 else np.inf
    print(f"  {label:30s}: ratio(Re-only) = {r:.6f}")

print()
for label, factor in [
    ("P_F/chi^2", 1.0),
    ("L_box * P_F/chi^2", L_box),
    ("N * P_F/chi^2", chi_grid.size),
    ("N^2 * P_F/chi^2", chi_grid.size**2),
    ("N * dchi * P_F/chi^2", chi_grid.size * dchi),
    ("dchi * P_F/chi^2", dchi),
]:
    cl_test = cl_theory_simple * factor
    _, binned_test = MD_new.convolve_theory_Cls(cl_test, bins_new)
    r = binned_cl_new[1] / binned_test[1] if binned_test[1] > 0 else np.inf
    print(f"  {label:30s}: ratio(Re+Im) = {r:.6f}")


# ----- Shot noise ---------------------------------------------------------- #
print("\n=== Shot noise analysis ===")
SN_data = np.sum(np.abs(delta_2d[:, ki_new])**2) / (4.0 * np.pi)
SN_data_re = np.sum(np.real(delta_2d[:, ki_new])**2) / (4.0 * np.pi)
print(f"Shot noise (|delta_2D|^2 / 4pi) = {SN_data:.6e}")
print(f"Shot noise (Re^2 / 4pi)         = {SN_data_re:.6e}")
print(f"Measured Cl[100] (Re+Im)         = {cl_data_new[min(100,Nl-1)]:.6e}")
print(f"Measured Cl[100] (Re only)       = {cl_re_only[min(100,Nl-1)]:.6e}")

# ----- Summary comparison ------------------------------------------------- #
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"chi_bar = {chi_bar:.2f} Mpc/h")
print(f"L_box   = {L_box:.2f} Mpc/h")
print(f"N       = {chi_grid.size}")
print(f"dchi    = {dchi:.4f} Mpc/h")
print(f"Nskew   = {Nskew}")
print(f"Original code: measured Cl[5] = {cl_data_orig[5]:.6e}")
print(f"New code:      Re-only  Cl[5] = {cl_re_only[5]:.6e}")
print(f"New code:      Re+Im    Cl[5] = {cl_data_new[5]:.6e}")
print(f"Simple theory: P_F/chi^2 [5]  = {cl_theory_simple[5]:.6e}")
