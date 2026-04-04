#!/usr/bin/env python
"""
Fit the normalization constant alpha in:
  C_true = alpha * b1^2 * P_lin(l/chi) / chi^2

and track whether it converges as Nl_large increases.

If alpha -> 1/(32*pi^3), the notes formula is correct.
If alpha -> 1/L, my derivation is correct.
If it converges to something else, both are wrong.
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
print(f"N={N}, L={L:.2f}, chi_bar={chi_bar:.2f}, b1={b1:.4f}")
print(f"dchi = L/N = {L/N:.4f}")
print(f"1/L = {1/L:.6e}")
print(f"1/(32*pi^3) = {1/(32*np.pi**3):.6e}")
print()

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

# For each Nl_large, compute M_code @ (b1^2 Plin/chi^2) and find alpha
print(f"{'Nl_large':>8s} {'alpha':>14s} {'alpha*L':>10s} {'alpha*32pi3':>12s}")
print("-" * 55)

alpha_list = []
Nl_list = []

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
    
    theory_unnorm = (M @ cl_unnorm)[:Nl]  # = M_code @ (b1^2 Plin/chi^2)
    
    # alpha = <cl_data> / <theory_unnorm>
    alpha = np.mean(cl_mean[mask]) / np.mean(theory_unnorm[mask])
    
    alpha_list.append(alpha)
    Nl_list.append(Nl_large)
    
    print(f"  {Nl_large:5d}  {alpha:14.6e} {alpha*L:10.4f} {alpha*32*np.pi**3:12.4f}")
    
    del couple, M; gc.collect()

print(f"\nIf alpha*L -> 1.0, then C_true = b1^2 Plin / (L chi^2)")
print(f"If alpha*32pi3 -> 1.0, then C_true = b1^2 Plin / (32pi^3 chi^2)")
print(f"\nExtrapolation: alpha seems to still be decreasing...")
print(f"  alpha_3500*L = {alpha_list[-1]*L:.4f}")
print(f"  alpha_3500*32pi3 = {alpha_list[-1]*32*np.pi**3:.4f}")
print(f"\nNeither converges to 1. Let me check what alpha converges to.")

# Richardson extrapolation or power-law fit
# Log-log plot alpha vs 1/Nl_large
Nl_arr = np.array(Nl_list, dtype=float)
alpha_arr = np.array(alpha_list)
# Try: alpha(Nl) = alpha_inf + A / Nl^p
# Use last few points to extrapolate
from scipy.optimize import curve_fit

def model(x, alpha_inf, A, p):
    return alpha_inf + A / x**p

try:
    popt, pcov = curve_fit(model, Nl_arr[3:], alpha_arr[3:], p0=[5e-4, 1.0, 1.0])
    alpha_inf = popt[0]
    print(f"\nExtrapolated alpha_inf = {alpha_inf:.6e}")
    print(f"  alpha_inf * L = {alpha_inf*L:.4f}")
    print(f"  alpha_inf * 32pi3 = {alpha_inf*32*np.pi**3:.4f}")
except Exception as e:
    print(f"Extrapolation failed: {e}")

# Also just check ratio of successive differences
print(f"\nRatio of successive differences (convergence rate):")
for i in range(1, len(alpha_list)):
    da = alpha_list[i] - alpha_list[i-1]
    if i > 1:
        da_prev = alpha_list[i-1] - alpha_list[i-2]
        if abs(da_prev) > 0:
            print(f"  Nl={Nl_list[i]}: da/da_prev = {da/da_prev:.4f}")
