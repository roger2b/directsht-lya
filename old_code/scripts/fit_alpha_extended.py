#!/usr/bin/env python
"""
Fit alpha using PLKjKk to lambda=4000 for maximum convergence,
scanning over many Nl_large values.
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
Nl = 500
cl_mean = np.mean(cl_k_all, axis=0)

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

# Extended PLKjKk
PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)
print(f"PLKjKk: {len(PLKjKk)} multipoles available")

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

alpha_ref = 1.0 / (32 * np.pi**3)

print(f"\nNl_large | alpha_fit      | alpha/[1/(32pi^3)] | data/theory")
print("-" * 70)

for Nl_large in [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500]:
    t1 = time.time()
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_unnorm = b1**2 * plin((ells_ext + 0.5) / chi_bar) / chi_bar**2
    
    wl_needed = 2 * Nl_large - 1
    wl_for_couple = np.zeros(wl_needed)
    n_avail = min(wl_needed, len(wl_ext))
    wl_for_couple[:n_avail] = wl_ext[:n_avail]
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_couple)
    M = couple.compute_matrix()
    
    cl_theory_unnorm = (M @ cl_unnorm)[:Nl]
    
    alpha_fit = np.mean(cl_mean[mask]) / np.mean(cl_theory_unnorm[mask])
    ratio = alpha_fit / alpha_ref
    
    # Also compute data/theory with the notes formula:
    cl_theory_notes = cl_theory_unnorm * alpha_ref
    dt_ratio = np.mean(cl_mean[mask]) / np.mean(cl_theory_notes[mask])
    
    dt = time.time() - t1
    print(f"  {Nl_large:5d}  | {alpha_fit:.6e} | {ratio:8.4f}            | {dt_ratio:8.4f}  [{dt:.1f}s]")
    
    del couple, M; gc.collect()

print(f"\nReference: 1/(32*pi^3) = {alpha_ref:.6e}")
print(f"If alpha converges to 1/(32*pi^3), the notes formula is correct.")
print(f"If it converges to something else, there's a normalization error.")
