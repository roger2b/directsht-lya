#!/usr/bin/env python
"""
Study the MASTER convergence in detail.

At Nl_large=1501 (inner sums 100% complete for l=0..499), the ratio is 0.966.
But the series keeps growing past 1.0 and doesn't converge.

Key question: After accounting for the EXACT inner sums being complete,
does the outer (l') sum converge or still diverge?

Also: what if we compute PLKjKk to higher lambda?
For Nl_large=3000, we need lambda up to 499+2999 = 3498.
Time for Legendre sum to lambda=3500 ~ 3.5/2 * 185s ~ 320s.

Let's also try: compute ONLY the correction terms at high l',
using the PLKjKk data we have. For l' = 1501..2000, the lambda range is
1001..2499 (for l=499). We have lambda up to 1999.
So part of the inner sum is captured. Let's see if partially-captured
high-l' terms help or hurt.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
import SHT_lya as sht_lya

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
PLKjKk_file = 'notebooks/data/PLKjKk_lambda2000.npy'
PLKjKk = np.load(PLKjKk_file)
wl_full = PLKjKk / (4*np.pi)
lambda_max_data = len(PLKjKk)

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0

# Sightline positions (for computing PLKjKk)
GRF_pos = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=1000, verbose=False)
num_qso = 9797
all_x, all_y, all_z, _, _, _ = GRF_pos.process_skewers(Nskew=num_qso, shift=5000)
theta_ref, phi_ref = GRF_pos.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
del GRF_pos, all_x, all_y, all_z; gc.collect()

nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)  # 9797 x 9797
KjKk = N**2
del nhat; gc.collect()

print(f"N={N}, Nskew={Nskew}, L={L:.1f}")
print(f"cos_theta: {cos_theta.shape}, memory: {cos_theta.nbytes/1e9:.2f} GB")

# ---- Option 1: Extend PLKjKk computation to higher lambda ----
# Cost per 100 lambdas: ~185/20 ≈ 9s  (since lambda_max=2000 took 185s)
# Actually the cost per Legendre polynomial is constant.
# Total for 2000 took 185s, so per-lambda cost is 185/2000 ~ 0.09s
# For 1000 more: 92s. For 2000 more: 185s. For 5000 more: 462s.

# The question is: HOW MUCH higher do we need?
# At Nl_large=1501 (complete inner sums), data/theory = 0.966.
# We need the outer sum to converge. The fact that it's AT 0.966 with complete inner sums
# means continuing to add more l' with COMPLETE inner sums would push it above 1.0.
# Then at some point the additional l' contributions must turn NEGATIVE to bring it back.
# BUT: for a coupling matrix with positive 3j^2 and positive wl, ALL M[l,l'] > 0.
# And C_true > 0. So all terms are positive. The sum CAN'T decrease!

# WAIT. This means the sum strictly increases. If it's already AT 1.04 at Nl_large=1500
# (from our earlier data), then the true converged value is > 1.04.
# And that means our C_true formula OVERCOUNTS. The theory is too HIGH.

# So the agreement at Nl_large=1000 (ratio ~0.97) is COINCIDENTAL.
# The true converged prediction is significantly ABOVE the data.
# And the C_true formula must be wrong by ~5-10%.

# Unless... some of the wl values are NEGATIVE! Let me check.
print(f"\nwl_full range: min={np.min(wl_full):.4e}, max={np.max(wl_full):.4e}")
print(f"Any negative wl? {np.any(wl_full < 0)}")
if np.any(wl_full < 0):
    neg_idx = np.where(wl_full < 0)[0]
    print(f"Negative wl at lambda = {neg_idx}")
    print(f"Values: {wl_full[neg_idx]}")

# Since wl = PLKjKk/(4pi) and PLKjKk = SUM_{j,k} KjKk * Pl(cos_jk),
# and KjKk = N^2 > 0, and Pl can be negative, yes wl can be negative!
# And if wl is negative at some lambda, then M[l,l'] can have negative contributions.
# This could make the sum non-monotonic!

# Let me check the PLKjKk values more carefully
print(f"\nPLKjKk range: min={np.min(PLKjKk):.4e}, max={np.max(PLKjKk):.4e}")
print(f"Any negative PLKjKk? {np.any(PLKjKk < 0)}")

# How many are negative?
neg_mask = PLKjKk < 0
print(f"Number of negative PLKjKk values: {np.sum(neg_mask)} out of {len(PLKjKk)}")
print(f"Fraction negative: {np.mean(neg_mask):.4f}")

# Plot the structure
print(f"\nPLKjKk sign structure (100-bin averages):")
for lam_start in range(0, 2000, 100):
    chunk = PLKjKk[lam_start:lam_start+100]
    n_neg = np.sum(chunk < 0)
    print(f"  lambda {lam_start:4d}-{lam_start+99:4d}: "
          f"mean={np.mean(chunk):+.4e}, n_neg={n_neg:3d}/100, "
          f"min={np.min(chunk):+.4e}")

# So all PLKjKk > 0? That would mean all wl > 0 and M[l,l'] > 0 for all entries.
# Then the sum strictly increases. Let me verify.

# Actually, this can't be right. PLKjKk = SUM Kj*Kk*Pl(cos).
# For k=0: Kj = N for all j. So PLKjKk = N^2 * SUM_{j,k} Pl(cos_jk).
# For large lambda, the Legendre polynomials oscillate and the sum can be positive or negative.
# But the sum of MANY random terms with random cos should hover around 0 + a Poisson offset.
# Actually no: Pl(cos=1) = 1 always. The diagonal terms (j=k) contribute Nskew * N^2.
# Off-diagonal: N^2 * SUM_{j!=k} Pl(cos_jk) which oscillates.
# Total: N^2 * Nskew + N^2 * SUM_{j!=k} Pl(cos_jk)
# The SN = N^2 * Nskew comes from the diagonal.
# The off-diagonal sum at high lambda should average to 0 with fluctuations ~sqrt(Nskew^2).
# So PLKjKk = SN + O(SN * sqrt(Nskew)) = SN * (1 + O(1))
# The fluctuations are O(SN), so PLKjKk can be negative!

# But our data shows all positive. Let me check for very high lambda where fluctuations
# should be larger relative to the mean:
print(f"\nNskew = {Nskew}, N^2 = {N**2}")
print(f"SN = N^2 * Nskew = {N**2 * Nskew:.4e}")
print(f"Diagonal: N^2 * Nskew = {N**2 * Nskew:.4e}")
print(f"Off-diag pairs: Nskew*(Nskew-1) = {Nskew*(Nskew-1):.4e}")
print(f"Fluctuation scale ~ N^2 * sqrt(Nskew*(Nskew-1)) ~ {N**2 * np.sqrt(Nskew*(Nskew-1)):.4e}")
print(f"This is {N**2 * np.sqrt(Nskew*(Nskew-1)) / (N**2*Nskew):.4f} of SN")

# So fluctuations are ~sqrt(Nskew*(Nskew-1))/Nskew = sqrt((Nskew-1)/Nskew) ~ 1.
# PLKjKk fluctuates between ~0 and ~2*SN. It CAN be negative but it's rare.

# Given all PLKjKk > 0 (at least up to lambda=2000), ALL wl > 0, so ALL M[l,l'] >= 0.
# This means the MASTER sum strictly increases. It will NOT come back down.
# The series converges (C_true -> 0 at Nyquist), but to a value ABOVE the data.

# THIS MEANS C_TRUE IS WRONG (too high).
# Or our normalization of the pseudo-Cl is wrong.

# Let me directly check: what value of C_true normalization gives the right answer?
# C_true = alpha * b1^2 * Plin(k) / (32*pi^3*chi^2)
# We need to find alpha such that SUM M[l,l'] * alpha * C_true = cl_mean

# For the case Nl_large=1501:
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
del couple, M; gc.collect()

# Alpha from fit (each ell):
alpha_per_ell = cl_mean / theory_1501
print(f"\nalpha = data / theory_1501:")
print(f"  Mean (ell>10): {np.mean(alpha_per_ell[10:]):.4f}")
print(f"  Std  (ell>10): {np.std(alpha_per_ell[10:]):.4f}")

# But this is for Nl_large=1501. The TRUE converged theory is HIGHER.
# From the SN decomposition, the converged value adds ~delta_sigma * SN_poisson/(4pi)
# But we showed that overcounts. Let me try to estimate the converged value differently.

# The total theory = integral_over_l' M[l,l'] C[l'] = theory_1501 + correction
# correction = SUM_{l'>=1501} M[l,l'] C[l']
# We know M[l,l'] > 0 and C[l'] > 0, so correction > 0.
# So theory_converged > theory_1501, and alpha < 0.966.

# The MONOTONIC growth + all positive contributions means the converged theory
# is HIGHER than any partial sum. We can bound it from above.
# Upper bound: theory_1501 + Poisson_shot_correction
SN_4pi = float(N**2 * Nskew) / (4*np.pi)
kNy = np.pi / (L/N)
ell_Ny = int(kNy * chi_bar)
ells_high = np.arange(ell_Ny + 1)
C_true_high = b1_ref**2 * plin_ref((ells_high + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
sigma_sq_full = np.sum((2*ells_high+1)/(4*np.pi) * C_true_high)
sigma_sq_1501 = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_ext)
delta_sigma = sigma_sq_full - sigma_sq_1501

theory_upper = theory_1501 + SN_4pi * delta_sigma  
alpha_upper = np.mean(cl_mean[10:] / theory_upper[10:])

print(f"\nBounds:")
print(f"  theory_1501 / data = {np.mean(theory_1501[10:])/np.mean(cl_mean[10:]):.4f} (lower bound on converged)")
print(f"  theory_upper / data = {np.mean(theory_upper[10:])/np.mean(cl_mean[10:]):.4f} (upper bound)")
print(f"  True converged theory/data is between {np.mean(theory_1501[10:])/np.mean(cl_mean[10:]):.4f} and {np.mean(theory_upper[10:])/np.mean(cl_mean[10:]):.4f}")

# ---- Let's actually just compute PLKjKk to lambda=4000 ----
# This gives us Nl_large up to 3501 with complete inner sums.
# Time: ~4000/2000 * 185s = 370s ≈ 6 min
print(f"\n---- Computing PLKjKk to lambda=4000 ----")
lambda_max_new = 4000
t0 = time.time()
PLKjKk_ext = sht_lya.legendre_polynomials_sum(lambda_max_new, cos_theta, KjKk)[:lambda_max_new]
dt = time.time() - t0
print(f"Done in {dt:.1f}s")

# Save extended PLKjKk
outfile = os.path.join('notebooks', 'data', f'PLKjKk_lambda{lambda_max_new}.npy')
np.save(outfile, PLKjKk_ext)
print(f"Saved to {outfile}")

# Verify consistency with original
print(f"\nConsistency check with PLKjKk_lambda2000:")
print(f"  Max diff for lambda<2000: {np.max(np.abs(PLKjKk_ext[:2000] - PLKjKk[:2000])):.4e}")

# Now compute MASTER at higher Nl_large with complete inner sums
wl_ext = PLKjKk_ext / (4*np.pi)

print("\n---- MASTER with extended PLKjKk (lambda=4000) ----")
print(f"{'Nl_large':>8s} {'complete?':>10s} {'th/data':>10s} {'data/th':>10s}")
print("-" * 45)

for Nl_large in [1000, 1500, 2000, 2500, 3000, 3500]:
    wl_needed = 2 * Nl_large - 1
    wl_for_coupling = np.zeros(wl_needed)
    n_avail_ext = min(wl_needed, lambda_max_new)
    wl_for_coupling[:n_avail_ext] = wl_ext[:n_avail_ext]
    
    ells_nl = np.arange(Nl_large, dtype=float)
    C_true_nl = b1_ref**2 * plin_ref((ells_nl + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
    
    t0 = time.time()
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
    M = couple.compute_matrix()
    theory_nl = (M @ C_true_nl)[:Nl]
    dt = time.time() - t0
    del couple, M; gc.collect()
    
    max_lambda_needed = Nl - 1 + Nl_large - 1
    complete = "yes" if max_lambda_needed < lambda_max_new else f"no ({max_lambda_needed})"
    
    ratio = np.mean(theory_nl[10:] / cl_mean[10:])
    ratio_inv = np.mean(cl_mean[10:] / theory_nl[10:])
    
    print(f"{Nl_large:8d} {complete:>10s} {ratio:10.4f} {ratio_inv:10.4f} ({dt:.1f}s)")
