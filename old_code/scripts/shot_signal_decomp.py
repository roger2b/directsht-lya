#!/usr/bin/env python
"""
Compute the theory pseudo-Cl using the ANALYTIC shot-noise decomposition.

Key insight:
  wl_true[lambda] = wl_signal[lambda] + wl_shot
  where wl_shot = SN/(4pi) = N^2 * Nskew / (4pi) is a constant.

For the shot-noise part, the coupling is ANALYTICALLY summable:
  <pseudo-Cl>_shot = SN/(4pi) * SUM_{l'=0}^{l_max} (2l'+1)/(4pi) * C_true[l']
                   = SN/(4pi)^2 * SUM_{l'} (2l'+1) C_true[l']
                   = SN/(4pi) * sigma^2
This is a CONSTANT for all l (independent of l).

For the signal part (wl_signal = wl_true - wl_shot):
  <pseudo-Cl>_signal = SUM_{l'} M_signal[l,l'] C_true[l']
  where M_signal uses wl_signal which has ZERO mean at high lambda.
  This coupling decays faster and should converge with fewer terms.

Total: <pseudo-Cl> = <pseudo-Cl>_shot + <pseudo-Cl>_signal
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
lambda_max_data = len(PLKjKk)

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, Nskew={Nskew}, L={L:.1f}, chi_bar={chi_bar:.1f}")

# ---- Shot-noise constant ----
SN = float(N**2 * Nskew)  # PLKjKk at high lambda oscillates around this

# But wait: is SN the right value? Let me check from the data.
# At high lambda, PLKjKk oscillates. Let me check different estimates:
print(f"\nShot-noise estimates:")
print(f"  N^2 * Nskew (Poisson) = {SN:.4e}")
print(f"  wl_full[-1] * (4pi)   = {wl_full[-1]*4*np.pi:.4e}")
print(f"  Mean PLKjKk[1500:]    = {np.mean(PLKjKk[1500:]):.4e}")
print(f"  Mean PLKjKk[1000:]    = {np.mean(PLKjKk[1000:]):.4e}")

# Use the Poisson value since the oscillations average out
SN_4pi = SN / (4*np.pi)  # = wl_shot

# ---- C_true to high ell (up to Nyquist) ----
kNy = np.pi / (L/N)
ell_Ny = int(kNy * chi_bar)
ell_max_theory = ell_Ny + 500  # go a bit beyond Nyquist
ells_ext = np.arange(ell_max_theory)
C_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

# Set C_true = 0 beyond Nyquist (the box can't produce these modes)
C_true_ext[ells_ext > ell_Ny] = 0.0
print(f"\nell_Ny = {ell_Ny}, using C_true up to ell = {ell_max_theory-1}")
print(f"C_true[ell_Ny] = {C_true_ext[ell_Ny]:.4e}")
print(f"C_true set to 0 for ell > {ell_Ny}")

# ---- Part 1: Shot-noise contribution (analytical) ----
# <pseudo-Cl>_shot = wl_shot * SUM_{l'=0}^{ell_Ny} (2l'+1)/(4pi) C_true[l']
# = SN_4pi * sigma^2   where sigma^2 = SUM (2l'+1)/(4pi) C_true[l']

sigma_sq = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_ext)
shot_contribution = SN_4pi * sigma_sq
print(f"\nsigma^2 = {sigma_sq:.6e}")
print(f"Shot contribution (constant for all l) = {shot_contribution:.6e}")
print(f"Mean measured pseudo-Cl = {np.mean(cl_mean[1:]):.6e}")
print(f"Shot / measured = {shot_contribution / np.mean(cl_mean[1:]):.4f}")

# ---- Part 2: Signal contribution (using wl_signal = wl_full - SN_4pi) ----
# The signal part of the window
wl_signal = wl_full[:lambda_max_data] - SN_4pi
print(f"\nwl_signal statistics:")
print(f"  Mean: {np.mean(wl_signal):.4e}")
print(f"  Mean (l>500): {np.mean(wl_signal[500:]):.4e}")
print(f"  Std (l>500): {np.std(wl_signal[500:]):.4e}")
print(f"  Max |wl_signal/wl_shot| (l>500): {np.max(np.abs(wl_signal[500:]))/SN_4pi:.4f}")

# Compute the signal coupling matrix for different Nl_large
print("\n---- Signal coupling (MASTER with wl_signal) ----")
for Nl_large in [500, 700, 1000, 1200, 1500]:
    wl_needed = 2 * Nl_large - 1
    wl_sig_for_coupling = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_sig_for_coupling[:n_avail] = wl_signal[:n_avail]
    
    # C_true for this ell' range
    C_true_for_ell = np.zeros(Nl_large)
    n_ct = min(Nl_large, len(C_true_ext))
    C_true_for_ell[:n_ct] = C_true_ext[:n_ct]
    
    # Coupling matrix with signal part of window
    couple = Wigner3j.CoupleMat(Nl_large, wl_sig_for_coupling)
    M_sig = couple.compute_matrix()
    signal_Cl = (M_sig @ C_true_for_ell)[:Nl]
    del couple, M_sig; gc.collect()
    
    # Total theory = shot + signal
    theory_Cl = shot_contribution + signal_Cl
    
    ratio = cl_mean / theory_Cl
    print(f"  Nl_large={Nl_large:5d}: signal mean={np.mean(signal_Cl[1:]):.4e}, "
          f"total/measured = {np.mean(ratio[1:]):.4f} (ell>0), "
          f"total/measured = {np.mean(ratio[10:]):.4f} (ell>10)")

# Let's also compute without the Nyquist cutoff to see the effect
print("\n---- Without Nyquist cutoff ----")
C_true_nonyq = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
sigma_sq_nonyq = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_nonyq)
shot_nonyq = SN_4pi * sigma_sq_nonyq
print(f"sigma^2 (no cutoff, ell to {ell_max_theory}) = {sigma_sq_nonyq:.6e}")
print(f"Shot contribution (no cutoff) = {shot_nonyq:.6e}")
print(f"Shot (no cutoff) / measured = {shot_nonyq / np.mean(cl_mean[1:]):.4f}")

# Also try with a proper SN value from the data instead of Poisson
SN_data = np.mean(PLKjKk[1500:])
SN_data_4pi = SN_data / (4*np.pi)
wl_signal_data = wl_full[:lambda_max_data] - SN_data_4pi
sigma_sq_data = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_ext)
shot_data = SN_data_4pi * sigma_sq_data

print(f"\n---- Using SN from data mean(PLKjKk[1500:]) ----")
print(f"SN_data = {SN_data:.4e} vs SN_Poisson = {SN:.4e}")
print(f"  Difference: {(SN_data-SN)/SN*100:.2f}%")
print(f"Shot (data SN) = {shot_data:.6e}")

for Nl_large in [500, 700, 1000]:
    wl_needed = 2 * Nl_large - 1
    wl_sig_d = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_sig_d[:n_avail] = wl_signal_data[:n_avail]
    
    C_true_for_ell = np.zeros(Nl_large)
    n_ct = min(Nl_large, len(C_true_ext))
    C_true_for_ell[:n_ct] = C_true_ext[:n_ct]
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_sig_d)
    M_sig = couple.compute_matrix()
    signal_Cl = (M_sig @ C_true_for_ell)[:Nl]
    del couple, M_sig; gc.collect()
    
    theory_Cl = shot_data + signal_Cl
    ratio = cl_mean / theory_Cl
    print(f"  Nl_large={Nl_large:5d}: ratio = {np.mean(ratio[1:]):.4f} (ell>0), "
          f"{np.mean(ratio[10:]):.4f} (ell>10)")

# The key question: Does the signal part converge faster?
print("\n---- Signal convergence test (Poisson SN) ----")
print("  Nl_large    shot       signal_mean   total_mean  ratio_ell>10")
for Nl_large in [500, 600, 700, 800, 900, 1000]:
    wl_needed = 2 * Nl_large - 1
    wl_sig_for_coupling = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_sig_for_coupling[:n_avail] = wl_signal[:n_avail]
    
    C_true_for_ell = np.zeros(Nl_large)
    n_ct = min(Nl_large, len(C_true_ext))
    C_true_for_ell[:n_ct] = C_true_ext[:n_ct]
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_sig_for_coupling)
    M_sig = couple.compute_matrix()
    signal_Cl = (M_sig @ C_true_for_ell)[:Nl]
    del couple, M_sig; gc.collect()
    
    theory_Cl = shot_contribution + signal_Cl
    ratio = cl_mean / theory_Cl
    print(f"  {Nl_large:5d}  {shot_contribution:.4e}  {np.mean(signal_Cl[1:]):+.4e}  "
          f"{np.mean(theory_Cl[1:]):.4e}  {np.mean(ratio[10:]):.4f}")
