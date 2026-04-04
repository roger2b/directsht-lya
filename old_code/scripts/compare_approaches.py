#!/usr/bin/env python
"""
Compare three approaches for the theory pseudo-Cl:
1. Direct MASTER with wl_full (truncated at lambda=2000)
2. Shot+signal decomposition (Poisson SN)
3. Shot+signal decomposition (fitted SN from data)

Also investigate: what is happening with the MASTER at Nl_large=1000?
The direct MASTER gave ratio ~0.975 before, but the decomposition gives 0.77?!
Something is wrong with the normalization.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

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

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, Nskew={Nskew}, L={L:.1f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# ---- 1. Direct MASTER with wl_full ----
Nl_large = 1000
wl_needed = 2 * Nl_large - 1
wl_for_coupling = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_full))
wl_for_coupling[:n_avail] = wl_full[:n_avail]

ells_ext = np.arange(Nl_large, dtype=float)
C_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

t0 = time.time()
couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
M = couple.compute_matrix()
theory_master = (M @ C_true_ext)[:Nl]
print(f"\n1. Direct MASTER (Nl_large={Nl_large}): {time.time()-t0:.1f}s")
print(f"   Mean theory/measured (ell>10) = {np.mean(theory_master[10:] / cl_mean[10:]):.4f}")
print(f"   Mean measured/theory (ell>10) = {np.mean(cl_mean[10:] / theory_master[10:]):.4f}")
del couple; gc.collect()

# ---- 2. Decomposition with Poisson SN ----
SN = float(N**2 * Nskew)
SN_4pi = SN / (4*np.pi)

# Shot part: analytical
kNy = np.pi / (L/N)
ell_Ny = int(kNy * chi_bar)
ells_high = np.arange(ell_Ny + 1)
C_true_high = b1_ref**2 * plin_ref((ells_high + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
sigma_sq = np.sum((2*ells_high+1)/(4*np.pi) * C_true_high)
shot_contrib = SN_4pi * sigma_sq

# Signal part: MASTER with wl_signal
wl_signal = wl_full[:len(wl_full)] - SN_4pi
wl_sig_for_coupling = np.zeros(wl_needed)
n_avail_sig = min(wl_needed, len(wl_signal))
wl_sig_for_coupling[:n_avail_sig] = wl_signal[:n_avail_sig]

couple_sig = Wigner3j.CoupleMat(Nl_large, wl_sig_for_coupling)
M_sig = couple_sig.compute_matrix()
signal_Cl = (M_sig @ C_true_ext)[:Nl]
theory_decomp = shot_contrib + signal_Cl

print(f"\n2. Decomposition (Poisson SN={SN:.4e}):")
print(f"   Shot = {shot_contrib:.4e} (constant)")
print(f"   Signal mean (ell>10) = {np.mean(signal_Cl[10:]):.4e}")
print(f"   Total mean (ell>10) = {np.mean(theory_decomp[10:]):.4e}")
print(f"   Direct MASTER mean (ell>10) = {np.mean(theory_master[10:]):.4e}")
print(f"   Decomp / Direct = {np.mean(theory_decomp[10:]) / np.mean(theory_master[10:]):.4f}")
del couple_sig, M_sig; gc.collect()

# ---- 3. Check: is decomp = direct MASTER? ----
# It should be if: shot part + signal MASTER = direct MASTER
# Shot part: SN_4pi * sigma_sq = SN_4pi * SUM (2l'+1)/(4pi) C_true[l'] (summed to ell_Ny)
# But the MASTER sums to Nl_large=1000, not ell_Ny=6620.
# The shot part of the MASTER at Nl_large:
# SUM_{l'=0}^{Nl_large-1} M_shot[l,l'] C_true[l']
# where M_shot[l,l'] = (2l'+1)/(4pi) * SN_4pi * SUM_lambda (2lam+1)(3j)^2

# For the 3j completeness to hold, the lambda sum must go to l+l'.
# In CoupleMat(Nl_large), the lambda sum goes to 2*(Nl_large-1).
# For l=499, l'=999: l+l'=1498 < 2*999=1998 -- OK, complete.
# So M_shot[l,l'] = (2l'+1)/(4pi) * SN_4pi for all l,l' < Nl_large.
# The shot part from the MASTER at Nl_large=1000 is:
# SUM_{l'=0}^{999} (2l'+1)/(4pi) * SN_4pi * C_true[l']
# = SN_4pi * SUM_{l'=0}^{999} (2l'+1)/(4pi) C_true[l']
# = SN_4pi * sigma_sq_1000
sigma_sq_1000 = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_ext)
shot_from_master_1000 = SN_4pi * sigma_sq_1000
print(f"\n3. Consistency check:")
print(f"   sigma_sq (to ell_Ny={ell_Ny}) = {sigma_sq:.6e}")
print(f"   sigma_sq (to {Nl_large-1}) = {sigma_sq_1000:.6e}")
print(f"   Ratio: {sigma_sq_1000/sigma_sq:.4f}")
print(f"   Shot from analytical (to ell_Ny) = {shot_contrib:.4e}")
print(f"   Shot from MASTER (to {Nl_large}) = {shot_from_master_1000:.4e}")
print(f"   Difference: {(shot_contrib - shot_from_master_1000)/shot_contrib*100:.2f}%")

# AH HA! The issue: in the decomposition, I used sigma_sq summed to ell_Ny=6620,
# but the signal MASTER only goes to Nl_large=1000. The shot and signal DON'T add up
# to the direct MASTER because they sum over DIFFERENT l' ranges.
# 
# The correct decomposition for Nl_large=1000:
# Direct MASTER = shot_part(l'=0..999) + signal_part(l'=0..999)
# = SN_4pi*sigma_sq_1000 + signal_Cl_1000
# Let me check this:
corrected_decomp = shot_from_master_1000 + signal_Cl
print(f"\n   Corrected decomp (shot to {Nl_large}) + signal:")
print(f"   Mean (ell>10) = {np.mean(corrected_decomp[10:]):.4e}")
print(f"   Direct MASTER = {np.mean(theory_master[10:]):.4e}")
print(f"   Ratio: {np.mean(corrected_decomp[10:]) / np.mean(theory_master[10:]):.6f}")
# This SHOULD be ~1.0 if the decomposition is consistent.

# ---- 4. The REAL question ----
# The direct MASTER at Nl_large=1000 gives ratio ~0.975 (data/theory).
# This means theory is ~97.5% of the measured value.
# Missing 2.5% comes from ell' > 1000.
# But the ANALYTICAL shot part shows that ell' from 1000 to 6620 contribute
# (sigma_sq_full - sigma_sq_1000) / sigma_sq_full of the shot power:
missing_frac = (sigma_sq - sigma_sq_1000) / sigma_sq
print(f"\n4. Missing power from l'>={Nl_large}:")
print(f"   Fraction of sigma^2 above l'={Nl_large}: {missing_frac:.4f} ({100*missing_frac:.1f}%)")
print(f"   Missing shot power = {SN_4pi * (sigma_sq - sigma_sq_1000):.4e}")
print(f"   As fraction of total theory: {SN_4pi*(sigma_sq-sigma_sq_1000) / np.mean(theory_master[10:]):.4f}")
print()

# So the analytical shot part says there's ~X% missing from l'>1000.
# If we ADD the analytical shot contribution for l'>1000 to the MASTER at l'<=1000:
theory_corrected = theory_master + SN_4pi * (sigma_sq - sigma_sq_1000)
ratio_corrected = cl_mean / theory_corrected
print(f"5. Corrected theory (MASTER l'<{Nl_large} + analytical shot l'>={Nl_large}):")
print(f"   Mean theory = {np.mean(theory_corrected[10:]):.4e}")
print(f"   Data/Theory (ell>10) = {np.mean(ratio_corrected[10:]):.4f}")
print(f"   Data/Theory per bin:")
from sht.mask_deconvolution import MaskDeconvolution as MD_class
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MD_class(Nl, wl_ref, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, 32)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells
binned_data = bins @ cl_mean
binned_theory = bins @ theory_corrected
for i in range(len(binned_ells)):
    if binned_theory[i] > 0:
        print(f"   ell={binned_ells[i]:6.0f}: data/theory = {binned_data[i]/binned_theory[i]:.4f}")

# Also: the signal part for l'>1000 is approximately 0 (since wl_signal oscillates around 0
# at high lambda). So the correction is predominantly from the shot-noise constant.
# But the signal coupling from l'>1000 actually SUBTRACTS a small amount (since wl_signal
# is slightly negative on average). Let me not worry about that.

print(f"\n6. Trying different SN levels for the high-l' correction:")
for SN_try_label, SN_try in [("Poisson", SN), ("mean(PLKjKk[1000:])", np.mean(PLKjKk[1000:])),
                               ("mean(PLKjKk[1500:])", np.mean(PLKjKk[1500:])),
                               ("mean(PLKjKk[100:])", np.mean(PLKjKk[100:]))]:
    SN_try_4pi = SN_try / (4*np.pi)
    theory_try = theory_master + SN_try_4pi * (sigma_sq - sigma_sq_1000)
    ratio_try = cl_mean[10:] / theory_try[10:]
    print(f"   SN={SN_try_label:25s}: mean ratio = {np.mean(ratio_try):.4f}")
