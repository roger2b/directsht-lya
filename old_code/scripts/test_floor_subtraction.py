#!/usr/bin/env python
"""
Subtract the white-noise floor from the window and check convergence.

The window W_lambda has a white-noise floor W_floor = N^2 * Nskew / (4*pi).
The MASTER matrix splits:
  M = M_clust + M_floor
where M_floor[l,L] = (2L+1) * W_floor / (4*pi)  (independent of l)

The M_floor contribution to the theory:
  (M_floor @ C_true)_l = (W_floor / (4*pi)) * SUM_L (2L+1) C_true[L]
  = W_floor * sigma^2_field

This is an ell-independent constant.

Strategy:
  1. Compute W_floor from the high-lambda plateau of wl
  2. wl_clust = wl - W_floor
  3. M_clust = CoupleMat(Nl_large, wl_clust)
  4. theory = M_clust @ C_true + W_floor * sigma^2
  5. Check if this converges with Nl_large
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_mean = np.mean(d['cl_k'], axis=0)
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

# White noise floor
W_floor = N**2 * Nskew / (4*np.pi)
print(f"W_floor = N^2 * Nskew / (4pi) = {W_floor:.4e}")

# Verify floor
print(f"wl at high lambda: {wl_ext[2000]:.4e}, {wl_ext[3000]:.4e}, {wl_ext[3500]:.4e}")
print(f"Ratio to W_floor: {wl_ext[2000]/W_floor:.4f}, {wl_ext[3000]/W_floor:.4f}")
print()

# Clustered window
wl_clust = wl_ext.copy()
wl_clust -= W_floor  # Remove floor

# For the clustered part, it should decay to ~0 at high lambda
print(f"wl_clust at high lambda:")
for lam in [0, 100, 200, 500, 1000, 2000, 3000]:
    print(f"  lam={lam}: {wl_clust[lam]:.4e} (fraction of floor: {wl_clust[lam]/W_floor:.4f})")

# Compute sigma^2 = (1/(4pi)) SUM (2L+1) C_true[L]
# For C_true = b1^2 P_lin(L/chi) / (X * chi^2), we need to choose X.
# Let's use X=L (my derivation) and X=32pi^3 (notes) and see.

def compute_sigma2(plin, b1, chi_bar, L_max, X):
    """Compute sigma^2_field = (1/(4pi)) SUM_{L=0}^{L_max} (2L+1) C_true[L]"""
    ells = np.arange(L_max, dtype=float)
    cl = b1**2 * plin((ells + 0.5)/chi_bar) / (X * chi_bar**2)
    return np.sum((2*ells + 1) * cl) / (4*np.pi)

# sigma^2 needs to run to ell_Ny since the GRF has no power above k_Ny
ell_Ny = int(np.pi * N / L * chi_bar)
print(f"\nell_Ny = {ell_Ny}")

# Compute sigma^2 at different L_max to check convergence
print(f"\n{'L_max':>8s} {'sigma2(L)':>14s} {'sigma2(32pi3)':>14s}")
for L_max in [500, 1000, 2000, 3000, 5000, 6000, 6621, 10000]:
    s2_L = compute_sigma2(plin, b1, chi_bar, L_max, L)
    s2_32 = compute_sigma2(plin, b1, chi_bar, L_max, 32*np.pi**3)
    print(f"  {L_max:5d}  {s2_L:14.6e} {s2_32:14.6e}")

# Use ell_Ny as the cutoff for sigma^2
sigma2_L = compute_sigma2(plin, b1, chi_bar, ell_Ny, L)
sigma2_32 = compute_sigma2(plin, b1, chi_bar, ell_Ny, 32*np.pi**3)
print(f"\nsigma2 at ell_Ny={ell_Ny}:")
print(f"  sigma2(X=L) = {sigma2_L:.6e}")
print(f"  sigma2(X=32pi3) = {sigma2_32:.6e}")

# The floor contribution to theory:
# theory_floor = W_floor * sigma^2
theory_floor_L = W_floor * sigma2_L
theory_floor_32 = W_floor * sigma2_32

print(f"\nW_floor * sigma2:")
print(f"  X=L: {theory_floor_L:.4e}")
print(f"  X=32pi3: {theory_floor_32:.4e}")
print(f"  cl_mean[100] = {cl_mean[100]:.4e}")
print(f"  floor/data ratio: X=L: {theory_floor_L/cl_mean[100]:.4f}, X=32pi3: {theory_floor_32/cl_mean[100]:.4f}")

# Now test convergence of M_clust @ C_true + floor
mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

print(f"\n{'Nl_large':>8s} {'ratio(L)':>10s} {'ratio(32pi3)':>12s}")
print("-" * 40)

for Nl_large in [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500]:
    ells_ext = np.arange(Nl_large, dtype=float)
    plin_vals = plin((ells_ext + 0.5)/chi_bar)
    
    cl_L = b1**2 * plin_vals / (L * chi_bar**2)
    cl_32 = b1**2 * plin_vals / (32*np.pi**3 * chi_bar**2)
    
    # Clustered window (subtract floor)
    wl_needed = 2*Nl_large - 1
    wl_clust_for_c = np.zeros(wl_needed)
    n_avail = min(wl_needed, len(wl_ext))
    wl_raw = np.zeros(wl_needed)
    wl_raw[:n_avail] = wl_ext[:n_avail]
    # For lambda beyond our data, assume wl = W_floor
    wl_raw[n_avail:] = W_floor
    wl_clust_for_c = wl_raw - W_floor
    
    couple_c = Wigner3j.CoupleMat(Nl_large, wl_clust_for_c)
    M_clust = couple_c.compute_matrix()
    
    th_L = (M_clust @ cl_L)[:Nl] + theory_floor_L
    th_32 = (M_clust @ cl_32)[:Nl] + theory_floor_32
    
    r_L = np.mean(cl_mean[mask]) / np.mean(th_L[mask])
    r_32 = np.mean(cl_mean[mask]) / np.mean(th_32[mask])
    
    print(f"  {Nl_large:5d}  {r_L:10.4f} {r_32:12.4f}")
    
    del couple_c, M_clust; gc.collect()

print(f"\nIf converged, the correct formula is C_true = b1^2 P / (X chi^2)")
print(f"with X such that the ratio -> 1.0")
