#!/usr/bin/env python
"""
Normalization test v2: determine correct theory normalization.

Strategy:
1. Generate GRF, measure pseudo-Cl at k=0
2. Build MaskDeconvolution with the CORRECT window (from N-weighted sightlines)
3. Forward-convolve theory predictions and compare to measured pseudo-Cl
4. Find the normalization factor that makes theory match data
"""
import sys, os
import numpy as np
import healpy as hp

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
from sht.lya_sfb import LyaSFB, _alm2cl_complex
from sht.theory_lya import theory_cl_k, compute_chi_bar_from_grid

# ---- settings ---- #
chi_shift = 5000
Nl = 200
seed = 1000
num_qso = 5000
k_idx = 0
add_rsd_ = False
NperBin = 32

Nx = 2 * Nl
xmax = 0.75
sht = DirectSHT(Nl, Nx, xmax)

# ---- GRF ---- #
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew=num_qso, shift=chi_shift)
all_theta, all_phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
chi_grid = all_x[0, :]
all_w_gal = all_w_gal - 1.0
dchi = chi_grid[1] - chi_grid[0]
L_box = chi_grid.size * dchi
N = chi_grid.size
chi_bar = compute_chi_bar_from_grid(chi_grid)
print(f"Nskew={Nskew}, N={N}, L_box={L_box:.1f}, dchi={dchi:.4f}, chi_bar={chi_bar:.1f}")

# ========================================================== #
# STEP 1: Measure pseudo-Cl (original code convention)       #
# ========================================================== #
k_arr_orig, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)
# k_arr_orig is in cycles/(Mpc/h)

# SHT with real FT weights at k=0
hdat = sht(all_theta, all_phi, FT_delta[:, k_idx])
hran = sht(all_theta, all_phi, FT_mask[:, k_idx])

# Measured pseudo-Cl
cl_data = hp.alm2cl(hdat)[:Nl]
cl_rand = hp.alm2cl(hran)[:Nl]

print(f"\n--- Measured pseudo-Cl ---")
print(f"cl_data[5]  = {cl_data[5]:.4e}")
print(f"cl_rand[0]  = {cl_rand[0]:.4e}")
print(f"(cl_rand[0] should be ~ (N*Nskew)^2/(4pi) = {(N*Nskew)**2/(4*np.pi):.4e})")

# ========================================================== #
# STEP 2: Build MaskDeconvolution with N-weighted window      #
# ========================================================== #
# The window from the original code uses cl_rand = hp.alm2cl(sht(theta,phi, N*ones))
# because FT_mask[:, 0] = N for all sightlines in the periodic case
print(f"\n--- Window analysis ---")
print(f"FT_mask[:5, 0] = {FT_mask[:5, 0]}")
print(f"Expected = N = {N} for periodic box")

# The standard MaskDeconvolution approach:
MD = MaskDeconvolution(Nl, cl_rand)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

# ========================================================== #
# STEP 3: Forward convolution of theory                       #
# ========================================================== #
# The mode-coupling matrix Mll relates:
#   <pseudo-Cl> = Mll @ Cl_true
#
# Mll[l,L] = (2L+1)/(4pi) * sum_lambda (2*lambda+1) W_lambda 3j^2
# where W_lambda = cl_rand
#
# The theory Cl_true depends on what we mean:
#   - If weights w_j = delta_2D_j (FT output), then
#     pseudo-Cl = |sum_j w_j Y*(n_j)|^2 / (2l+1)
#   - And <pseudo-Cl> = sum_{j,k} <w_j w_k*> P_l(cos theta_jk) / (2l+1)
#
# For periodic box at k=0:
#   w_j = sum_n delta_F[j,n]  (unnormalized DFT, real)
#   <w_j w_k> = sum_{n,n'} <delta[j,n] delta[k,n']>
#
# For j != k (different sightlines):
#   <delta[j,n] delta[k,n']> depends on transverse and LOS separation
#
# The key insight: the relationship between P(k) and the angular Cl 
# in the pair-counting formalism involves the coupling matrix.

# Theory: P_F(l/chi_bar, k_par=0) / chi_bar^2
cl_theory_raw = theory_cl_k(ells, 0.0, chi_bar, GRF.plin, b1=1.0, beta=0.0)
print(f"\ncl_theory_raw[5] = {cl_theory_raw[5]:.4e} (P_F/chi_bar^2 at ell=5)")

# Forward convolution: Mll @ Cl_true
# Mll is already computed inside MD
Mll = MD.Mll  # shape (Nl, Nl)
pseudo_cl_theory = Mll @ cl_theory_raw
binned_pseudo_theory = bins @ pseudo_cl_theory
binned_pseudo_data = bins @ cl_data

print(f"\n--- Forward convolution comparison ---")
print(f"Predicted pseudo-Cl[5]    = {pseudo_cl_theory[5]:.4e}")
print(f"Measured pseudo-Cl[5]     = {cl_data[5]:.4e}")
print(f"Ratio measured/predicted  = {cl_data[5] / pseudo_cl_theory[5]:.6e}")

# Now test various normalization factors on top of P_F/chi^2
print(f"\n--- Testing normalization factors (forward convolution) ---")
for label, factor in [
    ("P_F/chi^2 (raw)", 1.0),
    ("N * P_F/chi^2", N),
    ("N^2 * P_F/chi^2", N**2),
    ("L * P_F/chi^2", L_box),
    ("N*dchi * P_F/chi^2", N * dchi),
    ("dchi * P_F/chi^2", dchi),
    ("1/(4pi) * P_F/chi^2", 1.0/(4*np.pi)),
    ("N/(4pi) * P_F/chi^2", N/(4*np.pi)),
    ("L/(4pi) * P_F/chi^2", L_box/(4*np.pi)),
    ("dchi/(4pi) * P_F/chi^2", dchi/(4*np.pi)),
]:
    pred = Mll @ (cl_theory_raw * factor)
    binned = bins @ pred
    r = binned_pseudo_data[1] / binned[1] if binned[1] > 0 else np.inf
    print(f"  {label:30s}: ratio = {r:.6f}")

# ========================================================== #
# STEP 4: Pair-counting comparison (replicate original theory)#
# ========================================================== #
print(f"\n--- Pair-counting theory (original code) ---")

# Compute pair-counting sums
nhat = sht_lya.compute_nhat(all_theta, all_phi)
cos_theta_njnk = np.dot(nhat, nhat.T)
KjKk = N**2  # periodic
print("Computing Legendre sums...", end="", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(Nl, cos_theta_njnk, KjKk)[:Nl]
print("done")

# Look at what PLKjKk gives us
print(f"PLKjKk[0] = {PLKjKk[0]:.4e}  (should be ~ KjKk * Nskew^2 = {N**2 * Nskew**2:.4e})")
print(f"PLKjKk[1] = {PLKjKk[1]:.4e}")

# Compare PLKjKk to the angular window W_l:
# W_l from unit weights: (1/(2l+1)) sum_m |u_lm|^2
# This should relate to PLKjKk by:
# PLKjKk[l] = KjKk * sum_{j,k} P_l(cos theta_jk)
# Angular window: u_lm = sum_j Y_lm*(n_j), W_l = sum_m |u_lm|^2 / (2l+1)
# By addition theorem: sum_m Y_lm*(n_j) Y_lm(n_k) = (2l+1)/(4pi) P_l(cos theta_jk)
# So: (2l+1) W_l = sum_m |u_lm|^2 = sum_{j,k} sum_m Y_lm*(nj) Y_lm(nk)
#                = (2l+1)/(4pi) sum_{j,k} P_l(cos theta_jk)
# Hence: W_l = (1/(4pi)) sum_{j,k} P_l(cos theta_jk)  ← pair count / (4pi)
# And: PLKjKk[l] = KjKk * 4pi * W_l_unit  (unit-weight angular window)

# N-weighted window: cl_rand[l] = W_l(N-weight) = N^2 * W_l(unit)
# So: PLKjKk[l] = KjKk * 4pi * cl_rand[l] / N^2
#               = N^2 * 4pi * cl_rand[l] / N^2 = 4pi * cl_rand[l]

# TEST this:
print(f"\nPLKjKk[0]           = {PLKjKk[0]:.4e}")
print(f"4pi * cl_rand[0]    = {4*np.pi * cl_rand[0]:.4e}")
print(f"Ratio               = {PLKjKk[0] / (4*np.pi * cl_rand[0]):.6f}")

print(f"\nPLKjKk[5]           = {PLKjKk[5]:.4e}")
print(f"4pi * cl_rand[5]    = {4*np.pi * cl_rand[5]:.4e}")
print(f"Ratio               = {PLKjKk[5] / (4*np.pi * cl_rand[5]):.6f}")

# Original theory:
# C_ell = coupling_pk @ PLKjKk / (4pi) / (2pi chi_bar^2)
# Plotted: C_ell / (4pi)^2
#
# coupling_pk[l,L] = (2L+1)/(4pi) * sum_lambda (2lambda+1) pk[lambda] 3j^2
#
# Substituting PLKjKk[L] = 4pi * cl_rand[L]:
# C_ell = sum_L coupling_pk[l,L] * 4pi * cl_rand[L] / (4pi) / (2pi chi_bar^2)
#       = sum_L coupling_pk[l,L] * cl_rand[L] / (2pi chi_bar^2)
#
# And coupling_pk[l,L] = (2L+1)/(4pi) * sum_lam (2lam+1) pk[lam] 3j^2
#
# The Mll from MaskDeconvolution with window=cl_rand is:
# Mll[l,L] = (2L+1)/(4pi) * sum_lam (2lam+1) cl_rand[lam] 3j^2
#
# So C_ell = sum_L {(2L+1)/(4pi) sum_lam (2lam+1) pk[lam] 3j²} * cl_rand[L] / (2pi chi^2)
# This is NOT the same as Mll @ pk/chi^2 because the spectrum and window are in different spots.
#
# In the pair-counting, the coupling_matrix_pk couples pk (spectrum) in the lambda direction,
# while PLKjKk couples the angular window in the L direction.
# In MaskDeconvolution, it's the opposite: Mll couples the window in lambda direction
# and applies to the spectrum in the L direction.
#
# So: coupling_pk @ PLKjKk != Mll @ (pk/chi^2) in general!
# But by symmetry of the Wigner 3j these might be related...

# Let's verify numerically:
L_range = np.arange(Nl, dtype=float)
pk_L = GRF.plin(L_range / chi_bar)

couple_mat_pk = Wigner3j.CoupleMat(Nl, pk_L)
coupling_pk = couple_mat_pk.compute_matrix()[:Nl, :Nl]

C_ell_pair = coupling_pk @ PLKjKk / (4.0 * np.pi) / (2.0 * np.pi * chi_bar**2)
C_ell_pair_plotted = C_ell_pair / (4.0 * np.pi)**2

# MaskDeconvolution forward: Mll @ (pk/chi^2)
pseudo_cl_md = Mll @ cl_theory_raw

print(f"\n--- Pair-counting vs MaskDeconvolution ---")
print(f"C_ell_pair_plotted[5]  = {C_ell_pair_plotted[5]:.4e}")
print(f"cl_data[5]             = {cl_data[5]:.4e}")
print(f"Ratio data/pair_theory = {cl_data[5]/C_ell_pair_plotted[5]:.4f}")

# That ratio in the original code output was ~0.03. The original code averages over
# multiple sims to reduce cosmic variance. For 1 sim, the ratio fluctuates but
# should be O(1) if the normalization is correct.

# Wait — maybe the original code HAD a normalization issue that they fixed by
# averaging over sims and plotting both on the same axes? Let me check if
# the original code's theory should actually be divided by something more.

# Actually the CRUCIAL question: does the original code's theory match the 
# original code's measurement (averaged over sims)?
# From the prior test output, the ratio was 0.023-0.042 (~ 1/36).
# This suggests the normalization IS wrong in my replica.

# Let me try the Wigner coupling with pk as INPUT weights to CoupleMat
# but using HIGHER lambda_max for the sum:
lambda_max_high = 2*Nl - 1  # extend coupling
couple_hi = Wigner3j.CoupleMat(lambda_max_high, np.pad(pk_L, (0, lambda_max_high - len(pk_L))))
coupling_hi = couple_hi.compute_matrix()

PLKjKk_hi = sht_lya.legendre_polynomials_sum(lambda_max_high, cos_theta_njnk, KjKk)[:lambda_max_high]

C_ell_hi = coupling_hi[:Nl, :] @ PLKjKk_hi / (4*np.pi) / (2*np.pi*chi_bar**2)
C_ell_hi_plot = C_ell_hi / (4*np.pi)**2

print(f"\n--- Extended lambda_max = {lambda_max_high} ---")
print(f"C_ell_hi_plot[5]             = {C_ell_hi_plot[5]:.4e}")
print(f"cl_data[5]                   = {cl_data[5]:.4e}")
print(f"Ratio data/theory_hi         = {cl_data[5]/C_ell_hi_plot[5]:.4f}")

for ell in [5, 10, 20, 50, 100]:
    if ell < Nl:
        r = cl_data[ell] / C_ell_hi_plot[ell] if C_ell_hi_plot[ell] > 0 else np.inf
        print(f"  ell={ell}: data={cl_data[ell]:.4e}, theory={C_ell_hi_plot[ell]:.4e}, ratio={r:.4f}")

# ========================================================== #
# STEP 5: Direct approach — what IS the correct theory?       #
# ========================================================== #
print(f"\n--- Direct approach: what normalization makes data match theory? ---")

# For forward-convolved theory using MaskDeconvolution:
# pseudo-Cl_predicted = Mll @ C_true
# We want pseudo-Cl_predicted ≈ cl_data
# So C_true ≈ Mll^{-1} @ cl_data (for unbinned)
# Or: normalization = (bins @ cl_data) / (bins @ (Mll @ cl_theory_raw * alpha))
# Find alpha such that ratio ≈ 1

# Simple least-squares fit
binned_data = bins @ cl_data
binned_model_raw = bins @ (Mll @ cl_theory_raw)
# alpha_fit = sum(data * model) / sum(model^2)
alpha_fit = np.sum(binned_data * binned_model_raw) / np.sum(binned_model_raw**2)
print(f"Best-fit alpha (data = alpha * Mll @ Cl_raw): {alpha_fit:.4e}")
print(f"  N     = {N}")
print(f"  N^2   = {N**2}")
print(f"  L_box = {L_box:.1f}")
print(f"  N^2 * dchi = {N**2 * dchi:.1f}")
print(f"  alpha / N  = {alpha_fit / N:.4f}")
print(f"  alpha / N^2 = {alpha_fit / N**2:.4f}")

# Also try with deconvolved measurement:
deconv_ells, deconv_data = MD(cl_data, bins)
_, deconv_model = MD.convolve_theory_Cls(cl_theory_raw, bins)
alpha_deconv = np.sum(deconv_data * deconv_model) / np.sum(deconv_model**2)
print(f"\nBest-fit alpha (deconvolved): {alpha_deconv:.4e}")
print(f"  alpha_deconv / alpha = {alpha_deconv / alpha_fit:.6f}")
