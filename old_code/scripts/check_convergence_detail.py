#!/usr/bin/env python
"""
Check the convergence of the MASTER sum by looking at the 
per-ell'-bin contribution to the theory.

(M @ C_true)_l = SUM_L M[l,L] C_true[L]

We want to see how M[l,L] * C_true[L] varies with L for a fixed l.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# Load
PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
N = int(d['Nk'])
L = float(d['L'])
cl_mean = np.mean(d['cl_k'], axis=0)

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

# Compute M at Nl_large=3500
Nl_large = 3500
ells_ext = np.arange(Nl_large, dtype=float)
cl_true_32pi3 = b1**2 * plin((ells_ext + 0.5)/chi_bar) / (32*np.pi**3 * chi_bar**2)

wl_needed = 2 * Nl_large - 1
wl_for_couple = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_for_couple[:n_avail] = wl_ext[:n_avail]

print("Computing coupling matrix...")
t0 = time.time()
couple = Wigner3j.CoupleMat(Nl_large, wl_for_couple)
M = couple.compute_matrix()
print(f"Done in {time.time()-t0:.1f}s")

# For a representative ell (say l=100), look at M[l, L] * C_true[L] vs L
for l_check in [50, 100, 200, 300, 400]:
    row = M[l_check, :] * cl_true_32pi3
    
    # Cumulative sum
    cumsum = np.cumsum(row)
    
    print(f"\nl={l_check}: cl_data={cl_mean[l_check]:.4e}")
    print(f"  Cumulative (M @ Ctrue) up to L_max:")
    for L_max_check in [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500]:
        if L_max_check <= Nl_large:
            val = cumsum[L_max_check-1]
            ratio = cl_mean[l_check] / val if val > 0 else np.inf
            print(f"    L_max={L_max_check:4d}: theory={val:.4e}, data/theory={ratio:.4f}")
    
    # Check the per-L-bin contributions at high L
    print(f"  Per-L-bin contribution M[l,L]*Ctrue[L] at high L:")
    for L_check in [500, 1000, 1500, 2000, 2500, 3000, 3400]:
        print(f"    L={L_check}: {row[L_check]:.4e} (fraction of total: {row[L_check]/cumsum[-1]:.6f})")

print(f"\n--- Check diagonal structure of M at high L ---")
for l_check in [100, 200]:
    print(f"l={l_check}:")
    for L_check in [2000, 2500, 3000, 3400]:
        print(f"  M[{l_check},{L_check}] = {M[l_check,L_check]:.4e}, "
              f"(2L+1)/(4pi)*W_floor = {(2*L_check+1)/(4*np.pi)*2e8:.4e}")
