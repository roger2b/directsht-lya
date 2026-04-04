#!/usr/bin/env python
"""
Test whether setting C_true = 0 above ell_Nyquist fixes the convergence.

In the simulation, the density field is band-limited at k_Ny = pi*N/L.
For the TRANSVERSE k-modes (kz=0 plane), the maximum k_perp is:
  k_perp_max = sqrt(2) * k_Ny  (diagonal of the 2D Nyquist box)
  or  k_perp_max = k_Ny   (per axis)

The corresponding ell_max = k_perp_max * chi_bar.

Let me test with C_true(ell') = 0 for ell' > ell_Ny.
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

# Nyquist
k_Ny = np.pi * N / L
ell_Ny_1d = k_Ny * chi_bar  # 1D Nyquist ~ 6620
ell_Ny_2d = np.sqrt(2) * k_Ny * chi_bar  # 2D diagonal ~ 9360
print(f"k_Ny = {k_Ny:.4f}, ell_Ny(1D) = {ell_Ny_1d:.0f}, ell_Ny(2D) = {ell_Ny_2d:.0f}")

# Extended PLKjKk
PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

alpha_ref = 1.0 / (32 * np.pi**3)

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

print(f"\n{'Nl_large':>8} | {'cutoff':>6} | {'alpha/ref':>10} | {'ell_max':>8}")
print("-" * 55)

for Nl_large in [1000, 1500, 2000, 2500, 3000, 3500]:
    for ell_cut_label, ell_cut in [('none', Nl_large), ('Ny1D', int(ell_Ny_1d)), ('Ny2D', int(ell_Ny_2d))]:
        ells_ext = np.arange(Nl_large, dtype=float)
        
        # C_true with cutoff
        cl_true = b1**2 * plin((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
        # Apply cutoff
        actual_cut = min(ell_cut, Nl_large)
        cl_true[actual_cut:] = 0.0
        
        wl_needed = 2 * Nl_large - 1
        wl_for_couple = np.zeros(wl_needed)
        n_avail = min(wl_needed, len(wl_ext))
        wl_for_couple[:n_avail] = wl_ext[:n_avail]
        
        couple = Wigner3j.CoupleMat(Nl_large, wl_for_couple)
        M = couple.compute_matrix()
        
        cl_theory = (M @ cl_true)[:Nl]
        
        ratio = np.mean(cl_mean[mask]) / np.mean(cl_theory[mask])
        print(f"  {Nl_large:5d}   | {ell_cut_label:>6} | {ratio:10.4f} | {actual_cut:>8}")
        
        del couple, M; gc.collect()

# Also check: what is C_true at the Nyquist?
print(f"\n---- C_true at Nyquist ----")
print(f"P_lin(k_Ny = {k_Ny:.3f}) = {plin(k_Ny):.4e}")
print(f"P_lin(2*k_Ny) = {plin(2*k_Ny):.4e}")
print(f"P_lin(0.01) = {plin(0.01):.4e}")
print(f"\nC_true(ell_Ny={ell_Ny_1d:.0f}) = {b1**2 * plin(k_Ny) / (32*np.pi**3*chi_bar**2):.4e}")
print(f"C_true(ell=500) = {b1**2 * plin(500/chi_bar) / (32*np.pi**3*chi_bar**2):.4e}")
print(f"C_true(ell=1000) = {b1**2 * plin(1000/chi_bar) / (32*np.pi**3*chi_bar**2):.4e}")
print(f"C_true(ell=3000) = {b1**2 * plin(3000/chi_bar) / (32*np.pi**3*chi_bar**2):.4e}")
print(f"Ratio C_true(3000)/C_true(100) = {plin(3000/chi_bar)/plin(100/chi_bar):.4e}")
