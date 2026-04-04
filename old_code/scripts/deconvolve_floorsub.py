#!/usr/bin/env python
"""
Floor-subtracted deconvolution.

The raw pseudo-Cl contains contributions from:
  1. "Clustered" part: M_clust @ C_true   (converges with Nl_large)
  2. "Floor" (diagonal) part: W_floor * sigma^2 = Nskew * <w^2> / (4pi)

The diagonal part is ell-INDEPENDENT, so we can subtract it from the data,
then deconvolve only with M_clust (which is NOT ill-conditioned).

Steps:
  1. Compute <w^2> from the actual discrete mode grid
  2. diag_cl = Nskew * <w^2> / (4pi)
  3. cl_clust = cl_mean - diag_cl
  4. Build M_clust from wl_clust = wl - W_floor
  5. Deconvolve: C_true_emp = M_clust^{-1} @ cl_clust (binned)
  6. Compare with b1^2 P_lin(l/chi) / (X * chi^2) for X = L, 32pi^3
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Load data
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_all = d['cl_k']       # (100, 500)
cl_mean = np.mean(cl_all, axis=0)
N = int(d['Nk'])
L = float(d['L'])
Nl = 500
Nskew = int(d['Nskew'])

PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

W_floor = N**2 * Nskew / (4*np.pi)
print(f"N={N}, L={L:.2f}, chi_bar={chi_bar:.2f}, b1={b1:.4f}, Nskew={Nskew}")
print(f"W_floor = {W_floor:.4e}")

# Step 1: compute <w^2> from discrete mode grid
dk = 2*np.pi / L
kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2)
K_flat = K_perp.ravel()
Pk_flat = plin(np.where(K_flat > 0, K_flat, 1e-10))
Pk_flat[K_flat == 0] = 0
w2_theory = b1**2 * N**2 / L**3 * np.sum(Pk_flat)

# Step 2: diagonal contribution
diag_cl = Nskew * w2_theory / (4*np.pi)
print(f"\n<w^2>_theory  = {w2_theory:.6e}")
print(f"diag_cl       = {diag_cl:.4e}")
print(f"cl_mean[100]  = {cl_mean[100]:.4e}")
print(f"diag/cl_mean  = {diag_cl/cl_mean[100]:.4f}")

# Step 3: subtract diagonal from data
cl_clust_mean = cl_mean - diag_cl
cl_clust_all = cl_all - diag_cl  # per-sim

print(f"cl_clust[100] = {cl_clust_mean[100]:.4e}")
print(f"Fraction remaining (ell=100): {cl_clust_mean[100]/cl_mean[100]:.4f}")

# Step 4: Build M_clust with Nl_large=2000 and wl_clust
Nl_large = 2000
wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

# Build coupling matrix with clustered wl only
couple_c = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_full = couple_c.compute_matrix()
# Extract the Nl x Nl_large sub-block (we only have data for Nl modes)
M_clust = M_full[:Nl, :Nl_large]

print(f"\nM_clust shape: {M_clust.shape}")
print(f"M_clust[100,100] = {M_clust[100,100]:.4e}")
print(f"M_clust condition number (Nl block): {np.linalg.cond(M_clust[:Nl,:Nl]):.2e}")

# Step 5: Deconvolve with MaskDeconvolution using the Nl x Nl block
# We need the square sub-block for inversion
M_clust_sq = M_full[:Nl, :Nl]
MD = MaskDeconvolution(Nl, wl_clust[:2*Nl-1], precomputed_Wigner=M_clust_sq)
NperBin = 32
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

# Deconvolve the clustered Cl (mean)
ells_dec, cl_dec = MD(cl_clust_mean, bins)

# Theory: C_true(X) = b1^2 P_lin(l/chi) / (X * chi^2)
cl_true_L = b1**2 * plin((ells_dec + 0.5)/chi_bar) / (L * chi_bar**2)
cl_true_32pi3 = b1**2 * plin((ells_dec + 0.5)/chi_bar) / (32*np.pi**3 * chi_bar**2)

print(f"\n{'='*80}")
print(f"Floor-subtracted deconvolution results")
print(f"{'='*80}")
print(f"\n{'ell':>6s} {'cl_dec':>12s} {'th(L)':>12s} {'th(32pi3)':>12s} {'r(L)':>8s} {'r(32pi3)':>10s}")
print("-" * 70)
for i in range(len(ells_dec)):
    r_L = cl_dec[i] / cl_true_L[i] if cl_true_L[i] > 0 else np.nan
    r_32 = cl_dec[i] / cl_true_32pi3[i] if cl_true_32pi3[i] > 0 else np.nan
    print(f"  {ells_dec[i]:4.0f}  {cl_dec[i]:12.4e} {cl_true_L[i]:12.4e} {cl_true_32pi3[i]:12.4e} {r_L:8.4f} {r_32:10.4f}")

r_L_arr = cl_dec[1:] / cl_true_L[1:]
r_32_arr = cl_dec[1:] / cl_true_32pi3[1:]
print(f"\nMean ratio (excl mono): X=L: {np.mean(r_L_arr):.4f} ± {np.std(r_L_arr):.4f}")
print(f"                       X=32pi3: {np.mean(r_32_arr):.4f} ± {np.std(r_32_arr):.4f}")

# Per-sim scatter
print(f"\n{'='*80}")
print(f"Per-sim scatter (100 sims)")
print(f"{'='*80}")
cl_dec_per_sim = np.zeros((100, len(ells_dec)))
for s in range(100):
    _, cl_dec_per_sim[s] = MD(cl_clust_all[s], bins)
    
cl_dec_mean = np.mean(cl_dec_per_sim, axis=0)
cl_dec_std = np.std(cl_dec_per_sim, axis=0) / np.sqrt(100)  # error on mean

print(f"\n{'ell':>6s} {'cl_dec':>12s} {'err':>12s} {'r(L)':>8s} {'err_r':>8s} {'signif':>8s} {'r(32pi3)':>10s}")
print("-" * 80)
for i in range(1, len(ells_dec)):
    r = cl_dec_mean[i] / cl_true_L[i]
    dr = cl_dec_std[i] / cl_true_L[i]
    sig = (r - 1.0) / dr if dr > 0 else 0
    r32 = cl_dec_mean[i] / cl_true_32pi3[i]
    print(f"  {ells_dec[i]:4.0f}  {cl_dec_mean[i]:12.4e} {cl_dec_std[i]:12.4e} {r:8.4f} {dr:8.4f} {sig:8.1f}σ {r32:10.4f}")

print(f"\nOverall mean r(L) = {np.mean(cl_dec_mean[1:]/cl_true_L[1:]):.4f}")
print(f"Overall mean r(32pi3) = {np.mean(cl_dec_mean[1:]/cl_true_32pi3[1:]):.4f}")

# Also try: what X_fit gives ratio = 1 for the deconvolved data?
X_fit_dec = np.mean(cl_dec_mean[1:]) / np.mean(b1**2 * plin((ells_dec[1:]+0.5)/chi_bar) / chi_bar**2)
print(f"\nX_fit from deconvolution: {X_fit_dec:.4f}")
print(f"X_fit/L = {X_fit_dec/L:.4f}")
print(f"X_fit/(32pi3) = {X_fit_dec/(32*np.pi**3):.4f}")

# Step 6: check stability with different Nl_large for M_clust
print(f"\n{'='*80}")
print(f"Stability with Nl_large for M_clust")
print(f"{'='*80}")
for Nl_lg in [500, 1000, 1500, 2000, 3000]:
    t0 = time.time()
    wl_n = 2*Nl_lg - 1
    wl_r = np.zeros(wl_n)
    na = min(wl_n, len(wl_ext))
    wl_r[:na] = wl_ext[:na]
    wl_r[na:] = W_floor
    wl_c = wl_r - W_floor
    
    cp = Wigner3j.CoupleMat(Nl_lg, wl_c)
    Mf = cp.compute_matrix()
    M_sq = Mf[:Nl, :Nl]
    
    md = MaskDeconvolution(Nl, wl_c[:2*Nl-1], precomputed_Wigner=M_sq)
    _, cld = md(cl_clust_mean, bins)
    
    r_mean = np.mean(cld[1:] / cl_true_L[1:])
    print(f"  Nl_large={Nl_lg:5d}: mean r(L) = {r_mean:.4f}  [{time.time()-t0:.1f}s]")
    del cp, Mf; gc.collect()
