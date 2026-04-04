#!/usr/bin/env python
"""
Test the CORRECT C_true formula:
   C_true = b1^2 * P_lin(l/chi) / (L * chi^2)

vs the notes formula:
   C_true = b1^2 * P_lin(l/chi) / (32*pi^3 * chi^2)

The derivation shows:
  1. P_2D(k_perp) = b1^2 * N^2 * P_lin(k_perp) / L   (verified numerically)
  2. C_l^{ww} = P_2D(l/chi) / chi^2 = b1^2 * N^2 * P_lin(l/chi) / (L * chi^2)
  3. M_code = N^2 * M_mask  (because wl_code uses weight N per sightline)
  4. <Cl_pseudo> = M_mask @ C_l^{ww} = (M_code / N^2) @ C_l^{ww}
     = M_code @ (C_l^{ww} / N^2)
  5. Therefore: C_true(for M_code) = b1^2 * P_lin(l/chi) / (L * chi^2)

The ratio 32*pi^3 / L = 993 / 1380 = 0.72.
If this derivation is correct, replacing 32*pi^3 with L should make
the theory converge to data as Nl_large increases.
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
print(f"Parameters: N={N}, L={L:.1f}, chi_bar={chi_bar:.1f}, b1={b1:.4f}")
print(f"32*pi^3 = {32*np.pi**3:.2f}")
print(f"L = {L:.2f}")
print(f"Ratio 32*pi^3/L = {32*np.pi**3/L:.4f}")
print()

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

print(f"{'Nl_large':>8s} | {'data/th(notes)':>14s} | {'data/th(L)':>14s} | {'data/th(2piN/dchi)':>18s}")
print("-" * 70)

for Nl_large in [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500]:
    t1 = time.time()
    ells_ext = np.arange(Nl_large, dtype=float)
    plin_vals = plin((ells_ext + 0.5) / chi_bar)
    
    # Three candidate formulas:
    # (a) Notes: C_true = b1^2 Plin / (32*pi^3 * chi^2)
    cl_notes = b1**2 * plin_vals / (32 * np.pi**3 * chi_bar**2)
    
    # (b) My derivation: C_true = b1^2 Plin / (L * chi^2)
    cl_mine = b1**2 * plin_vals / (L * chi_bar**2)
    
    # (c) Alternative: maybe the 2pi comes from 2piN/dchi = 2piN^2/L?
    # If the LOS factor is 2piN/dchi instead of N^2/L:
    # C_ww = b1^2 * (2piN/dchi) * Plin/(chi^2) / N^2 = b1^2 * 2pi * Plin / (dchi * chi^2)
    # This gives C_true = b1^2 * 2pi * Plin / (dchi * chi^2)
    # Hmm, dchi = L/N, so 2pi/dchi = 2piN/L
    # Actually the Fejer approximation was: F_N(k_par dchi) -> (2piN/dchi) delta(k_par)
    # But then the dk_par integral goes from -inf to inf, giving 2piN/dchi.
    # Then the full integral: integral dk_par/(2pi) ... (2piN/dchi) delta(k_par) 
    #   = (N/dchi) = N^2/L. So 2pi cancels with 1/(2pi).
    # This is the same as my formula (b). Just verifying.
    dchi = L/N
    cl_alt = b1**2 * plin_vals / (L * chi_bar**2)  # same as (b)!
    
    wl_needed = 2 * Nl_large - 1
    wl_for_couple = np.zeros(wl_needed)
    n_avail = min(wl_needed, len(wl_ext))
    wl_for_couple[:n_avail] = wl_ext[:n_avail]
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_couple)
    M = couple.compute_matrix()
    
    theory_notes = (M @ cl_notes)[:Nl]
    theory_mine  = (M @ cl_mine)[:Nl]
    
    r_notes = np.mean(cl_mean[mask]) / np.mean(theory_notes[mask])
    r_mine  = np.mean(cl_mean[mask]) / np.mean(theory_mine[mask])
    
    dt = time.time() - t1
    print(f"  {Nl_large:5d}  | {r_notes:14.4f} | {r_mine:14.4f} | [{dt:.1f}s]")
    
    del couple, M; gc.collect()

print(f"\nIf data/th(L) converges to 1.0 as Nl_large→∞, the correct formula is")
print(f"  C_true = b1^2 * P_lin(l/chi) / (L * chi^2)")
print(f"\nNotes formula has 32*pi^3 = {32*np.pi**3:.2f} instead of L = {L:.2f}")
print(f"Overshoot factor = L / (32*pi^3) = {L/(32*np.pi**3):.4f}")
