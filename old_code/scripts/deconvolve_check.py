#!/usr/bin/env python
"""
Compute C_true empirically from the simulations themselves.

Method:
  1. For each sim, compute w_j at k=0 for ALL N^2 sightlines (full grid)
  2. DirectSHT with ALL sightlines -> get alm
  3. alm2cl gives pseudo-Cl_fullsky (no partial-sky masking needed if using all grid points)
  4. Actually: for N^2 sightlines covering the full box face,
     the "mask" covers a square patch. Not full sky.

Alternative: compute C_true by inverting M.
  Cl_data = M @ C_true  =>  C_true = M^{-1} @ Cl_data
  This is what MaskDeconvolution does!

Let me just run MaskDeconvolution and see what C_true it gives.
Then compare with b1^2 P/(L chi^2) and b1^2 P/(32pi^3 chi^2).
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_mean = np.mean(d['cl_k'], axis=0)
N = int(d['Nk'])
L = float(d['L'])
Nl = 500
Nskew = int(d['Nskew'])
wl_ref = d['wl_k'][0, :Nl]

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, L={L:.2f}, chi_bar={chi_bar:.2f}, b1={b1:.4f}, Nskew={Nskew}")

# MaskDeconvolution: inverts the MASTER relation
# <Cl> = M @ C_true  =>  C_true = M^{-1} @ <Cl>
# This works in binned space to handle the singular nature of M.

# Use wl_ref for M (which goes to Nl=500)
NperBin = 32
couple = Wigner3j.CoupleMat(Nl, wl_ref)
M = couple.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=M)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

# Deconvolve the data
ells_dec, cl_dec = MD(cl_mean, bins)

# Theory Cl at the deconvolved ell values:
cl_true_L = b1**2 * plin((ells_dec + 0.5)/chi_bar) / (L * chi_bar**2)
cl_true_32pi3 = b1**2 * plin((ells_dec + 0.5)/chi_bar) / (32*np.pi**3 * chi_bar**2)

print(f"\n{'ell':>6s} {'cl_dec':>12s} {'th(L)':>12s} {'th(32pi3)':>12s} {'r(L)':>8s} {'r(32pi3)':>10s}")
print("-" * 70)
for i in range(len(ells_dec)):
    r_L = cl_dec[i] / cl_true_L[i] if cl_true_L[i] > 0 else np.nan
    r_32 = cl_dec[i] / cl_true_32pi3[i] if cl_true_32pi3[i] > 0 else np.nan
    print(f"  {ells_dec[i]:4.0f}  {cl_dec[i]:12.4e} {cl_true_L[i]:12.4e} {cl_true_32pi3[i]:12.4e} {r_L:8.4f} {r_32:10.4f}")

# Average ratios (excl monopole):
r_L_arr = cl_dec[1:] / cl_true_L[1:]
r_32_arr = cl_dec[1:] / cl_true_32pi3[1:]
print(f"\nMean ratio (excl mono): X=L: {np.mean(r_L_arr):.4f}, X=32pi3: {np.mean(r_32_arr):.4f}")
print(f"Std: X=L: {np.std(r_L_arr):.4f}, X=32pi3: {np.std(r_32_arr):.4f}")

# Now try to understand the deconvolution.
# The binned deconvolution is: Cl_dec = (B M B^T)^{-1} B @ <Cl>
# This is the LEAST-SQUARES solution for C_true in the MASTER equation with binning.
# It accounts for the full coupling matrix, including the white-noise floor.
# So cl_dec IS the "true" C_true (as seen by the data), and comparing it with theory
# directly tells us the normalization.

# The ratio with X=L is the ell-dependent effective correction.
# Let me check if the scatter is consistent with MC noise.
cl_std = np.std(d['cl_k'], axis=0) / np.sqrt(100)
ells_std_dec, cl_std_dec = MD(cl_std, bins)
# Actually, the error on the deconvolved Cl is more complex...
# Let me just use the per-sim scatter.
cl_dec_per_sim = np.zeros((100, len(ells_dec)))
for s in range(100):
    _, cl_dec_per_sim[s] = MD(d['cl_k'][s], bins)
    
cl_dec_std = np.std(cl_dec_per_sim, axis=0) / np.sqrt(100)
cl_dec_mean = np.mean(cl_dec_per_sim, axis=0)

print(f"\n=== Deconvolved with per-sim scatter ===")
print(f"{'ell':>6s} {'cl_dec_mean':>12s} {'cl_dec_std':>12s} {'r(L)':>8s} {'err_r(L)':>10s} {'signif':>8s}")
for i in range(1, len(ells_dec)):
    r = cl_dec_mean[i] / cl_true_L[i]
    dr = cl_dec_std[i] / cl_true_L[i]
    sig = (r - 1.0) / dr if dr > 0 else 0
    print(f"  {ells_dec[i]:4.0f}  {cl_dec_mean[i]:12.4e} {cl_dec_std[i]:12.4e} {r:8.4f} {dr:10.4f} {sig:8.1f}σ")

print(f"\nOverall: mean r(L) = {np.mean(cl_dec_mean[1:]/cl_true_L[1:]):.4f}")
print(f"         mean r(32pi3) = {np.mean(cl_dec_mean[1:]/cl_true_32pi3[1:]):.4f}")
