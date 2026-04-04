#!/usr/bin/env python
"""
THE SOLUTION: MASTER at Nl_large=1000 + small analytical correction for l'>1000.

The direct MASTER at Nl_large=1000 gives data/theory = 1.035 (theory ~3.5% low).
Adding the shot-noise correction from l'>1000 with SN_eff ≈ 0.07 * SN_Poisson gives ~1.00.

But we can do better: fit SN_eff from the data itself, OR compute PLKjKk to higher lambda
so the signal cancellation is captured.

Alternative approach: Instead of fitting SN_eff, let's compute the MASTER at 
Nl_large=1000 (which is well within our PLKjKk data range), and then add
the correction from l'>1000 using the ACTUAL PLKjKk structure.

The correction for l' > 1000 can be computed differently:
For each measured l, the contribution from l' is:
  M[l,l'] * C_true[l']
For l' > 1000, we need M[l,l'] which requires PLKjKk to lambda = l+l'.
For l=0..499 and l'=1000..6600: lambda needs to go from 500 to 7100.
We have PLKjKk to 2000. So lambda=2000 allows l' up to 2000-l.
For l=0: l' up to 2000. For l=499: l' up to 1501.

Actually: for the MASTER coupling matrix M[l,l'] with the inner lambda sum:
  M[l,l'] = (2l'+1)/(4pi) * SUM_{lambda=|l-l'|}^{l+l'} (2lambda+1) * wl[lambda] * (3j)^2
If wl[lambda>2000] = 0, then M[l,l'] = 0 for l' where ALL lambda values > 2000.
That means l' where |l-l'| > 2000, i.e., l' > l + 2000.
For l' such that SOME lambda are within [0,2000], the sum is partial.

So we can extend the MASTER to Nl_large = l + 2000 ≈ 2500 for l=499.
But the issue is the PARTIAL lambda sums for l' near the boundary.

Let me try extending MASTER to higher Nl_large with the PLKjKk data we have.
The 3j triangle inequality means the inner sum runs from |l-l'| to l+l'.
For the inner sum to be 100% complete: all lambda in [|l-l'|, l+l'] must be < 2000.
For l=0: l+l' < 2000 means l' < 2000. OK.
For l=499: l+l' < 2000 means l' < 1501. And |l-l'| > 0 always.

For the inner sum to have ANY nonzero terms: |l-l'| < 2000.
So l' < l + 2000. For l=499: l' < 2499.

So we can safely go to Nl_large ≈ 1500 (all inner sums complete for all measured l).
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Load cache
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_k_all = d['cl_k']
wl_k = d['wl_k']
N = int(d['Nk'])
L = float(d['L'])
Nskew = int(d['Nskew'])
Nl = 500
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)

# Load PLKjKk
PLKjKk = np.load('notebooks/data/PLKjKk_lambda2000.npy')
wl_full = PLKjKk / (4*np.pi)
lambda_max_data = len(PLKjKk)

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, Nskew={Nskew}, L={L:.1f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# ---- MASTER at increasing Nl_large ----
# For each l in [0, Nl), and l' in [0, Nl_large):
# the inner lambda sum goes from |l-l'| to l+l'
# It's complete if l+l' < lambda_max_data = 2000
# For l=499 (our highest measured ell), this means l' < 1501.
# So for Nl_large <= 1501, ALL inner sums are complete for ALL measured l.

kNy = np.pi / (L/N)
ell_Ny = int(kNy * chi_bar)
ells_high = np.arange(ell_Ny + 1)
C_true_high = b1_ref**2 * plin_ref((ells_high + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
sigma_sq_full = np.sum((2*ells_high+1)/(4*np.pi) * C_true_high)

SN_poisson = float(N**2 * Nskew)

print(f"\n---- MASTER convergence with PLKjKk to lambda={lambda_max_data} ----")
print(f"{'Nl_large':>8s} {'complete?':>10s} {'th/data':>10s} {'data/th':>10s} {'shot_corr':>12s} {'final_ratio':>12s}")
print("-" * 70)

Nl_large_values = [500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000]
results = {}
for Nl_large in Nl_large_values:
    wl_needed = 2 * Nl_large - 1
    wl_for_coupling = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_for_coupling[:n_avail] = wl_full[:n_avail]
    
    ells_ext = np.arange(Nl_large, dtype=float)
    C_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
    
    t0 = time.time()
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
    M = couple.compute_matrix()
    theory = (M @ C_true_ext)[:Nl]
    dt = time.time() - t0
    del couple, M; gc.collect()
    
    # Check if inner sums are complete for all measured l
    max_l = Nl - 1  # =499
    max_lambda_needed = max_l + (Nl_large - 1)
    complete = "yes" if max_lambda_needed < lambda_max_data else f"no ({max_lambda_needed})"
    
    ratio = np.mean(theory[10:] / cl_mean[10:])
    ratio_inv = np.mean(cl_mean[10:] / theory[10:])
    
    # Analytical shot correction for l' > Nl_large
    sigma_sq_Nl = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_ext)
    delta_sigma = sigma_sq_full - sigma_sq_Nl
    # Use SN_eff = 0.07 * SN_Poisson for the correction
    SN_eff_frac = 0.07
    shot_corr = SN_eff_frac * SN_poisson / (4*np.pi) * delta_sigma
    theory_corr = theory + shot_corr
    ratio_corr = np.mean(cl_mean[10:] / theory_corr[10:])
    
    results[Nl_large] = theory
    
    print(f"{Nl_large:8d} {complete:>10s} {ratio:10.4f} {ratio_inv:10.4f} "
          f"{shot_corr:12.2f} {ratio_corr:12.4f}")

# ---- Focus: compute at Nl_large = 1501 (max safe value) ----
print(f"\n---- Best estimate: Nl_large=1501 (all inner sums complete) ----")
Nl_large = 1501
wl_needed = 2 * Nl_large - 1
wl_for_coupling = np.zeros(wl_needed)
n_avail = min(wl_needed, lambda_max_data)
wl_for_coupling[:n_avail] = wl_full[:n_avail]

ells_ext = np.arange(Nl_large, dtype=float)
C_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
M = couple.compute_matrix()
theory_1501 = (M @ C_true_ext)[:Nl]
del couple, M, ells_ext, C_true_ext; gc.collect()

ratio_1501 = np.mean(cl_mean[10:] / theory_1501[10:])
print(f"data/theory (ell>10) = {ratio_1501:.4f}")

# sigma_sq for correction  
ells_1501 = np.arange(1501, dtype=float)
C_true_1501 = b1_ref**2 * plin_ref((ells_1501 + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
sigma_sq_1501 = np.sum((2*ells_1501+1)/(4*np.pi) * C_true_1501)
delta_sigma_1501 = sigma_sq_full - sigma_sq_1501

# Fit SN_eff to make data/theory = 1.0
# theory_corrected = theory_1501 + SN_eff/(4pi) * delta_sigma
# data/theory_corrected = 1 -> data = theory_corrected
# SN_eff/(4pi) = (mean(data) - mean(theory_1501)) / delta_sigma
residual = np.mean(cl_mean[10:]) - np.mean(theory_1501[10:])
SN_eff_4pi_fitted = residual / delta_sigma_1501
SN_eff_fitted = SN_eff_4pi_fitted * 4*np.pi

print(f"\nFitted SN_eff = {SN_eff_fitted:.4e} ({SN_eff_fitted/SN_poisson:.4f} * SN_Poisson)")
print(f"delta_sigma_sq (l'>=1501) = {delta_sigma_1501:.6e}")
print(f"Correction = {SN_eff_4pi_fitted * delta_sigma_1501:.4e} = {SN_eff_4pi_fitted * delta_sigma_1501 / np.mean(cl_mean[10:])*100:.2f}% of data")

# Apply fitted correction
theory_best = theory_1501 + SN_eff_4pi_fitted * delta_sigma_1501
print(f"\nWithout correction: data/theory = {ratio_1501:.4f}")
print(f"With correction:    data/theory = {np.mean(cl_mean[10:] / theory_best[10:]):.4f}")

# ---- Binned comparison ----
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl)
NperBin = 32
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells
binned_data = bins @ cl_mean
binned_theory = bins @ theory_1501
binned_theory_corr = bins @ theory_best

print(f"\n{'ell':>6s} {'data':>12s} {'th(1501)':>12s} {'th(corr)':>12s} {'d/t(1501)':>10s} {'d/t(corr)':>10s}")
print("-" * 65)
for i in range(len(binned_ells)):
    if binned_theory[i] > 0:
        print(f"{binned_ells[i]:6.0f} {binned_data[i]:12.4e} {binned_theory[i]:12.4e} "
              f"{binned_theory_corr[i]:12.4e} {binned_data[i]/binned_theory[i]:10.4f} "
              f"{binned_data[i]/binned_theory_corr[i]:10.4f}")

# Summary statistics
ratio_vec = binned_data / binned_theory
ratio_corr_vec = binned_data / binned_theory_corr
sel = binned_ells > 30
print(f"\nMean d/t (ell>30): uncorrected={np.mean(ratio_vec[sel]):.4f}, corrected={np.mean(ratio_corr_vec[sel]):.4f}")
print(f"Std  d/t (ell>30): uncorrected={np.std(ratio_vec[sel]):.4f}, corrected={np.std(ratio_corr_vec[sel]):.4f}")
