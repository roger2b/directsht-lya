#!/usr/bin/env python
"""
Verify Identity 2 from the notes:
  M^(p)_{l,L} * W_L = M_{l,L} * P_F(L/chi)

where:
  M^(p)_{l,L} = (2L+1) SUM_lam (2lam+1) P_F(lam/chi) 3j(l,L,lam)^2
  M_{l,L}     = (2L+1)/(4pi) SUM_lam (2lam+1) W_lam 3j(l,L,lam)^2

Also verify Identity 1: PLKjKk = 4*pi * W_lambda
"""
import sys, os, gc
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + float(np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')['L']) / 2.0
del GRF_tmp; gc.collect()

# Data
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
N = int(d['Nk'])
wl_k = d['wl_k']
Nl = 500
wl_ref = wl_k[0, :Nl]  # W_lambda from SHT (includes N^2)

# PLKjKk
PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')

# Verify Identity 1: PLKjKk_lam = 4*pi * wl_ref for lam < Nl
print("=== Identity 1: PLKjKk = 4*pi * W_lambda ===")
for lam in [0, 10, 50, 100, 200, 400]:
    ratio = PLKjKk[lam] / (4*np.pi * wl_ref[lam])
    print(f"  lam={lam}: PLKjKk={PLKjKk[lam]:.6e}, 4pi*wl={4*np.pi*wl_ref[lam]:.6e}, ratio={ratio:.6f}")

# Now build M^(p) and M for small Nl to verify Identity 2
Nl_test = 100
ells = np.arange(Nl_test, dtype=float)

# P_F at ell/chi
pf_ell = b1**2 * plin((ells + 0.5)/chi_bar)

# M: uses wl_ref as coupling weight
# CoupleMat with wl_ref builds: M_lL = (2L+1)/(4pi) SUM_lam (2lam+1) wl[lam] 3j^2
wl_for_M = np.zeros(2*Nl_test - 1)
wl_for_M[:min(Nl, 2*Nl_test-1)] = wl_ref[:min(Nl, 2*Nl_test-1)]
couple_M = Wigner3j.CoupleMat(Nl_test, wl_for_M)
M_mat = couple_M.compute_matrix()  # shape (Nl_test, Nl_test)

# M^(p): uses P_F(lam/chi) as coupling weight
# Build P_F over lambda range 0 to 2*Nl_test-1
lam_range = np.arange(2*Nl_test - 1, dtype=float)
pf_lam = b1**2 * plin((lam_range + 0.5)/chi_bar)
couple_Mp = Wigner3j.CoupleMat(Nl_test, pf_lam)
Mp_mat = couple_Mp.compute_matrix()  # This builds (2L+1)/(4pi) SUM (2lam+1) pf[lam] 3j^2

# BUT: the CoupleMat formula has a 1/(4pi) factor. The notes' M^(p) does NOT.
# Notes: M^(p)_{l,L} = (2L+1) SUM (2lam+1) P_F(lam) 3j^2
# CoupleMat: returns (2L+1)/(4pi) SUM (2lam+1) weight[lam] 3j^2
# So Mp_mat = M^(p) / (4pi)
Mp_mat_notes = Mp_mat * (4*np.pi)  # = M^(p) from the notes

print(f"\n=== Identity 2: M^(p)_lL * W_L = M_lL * P_F(L/chi) ===")
print(f"Using Nl_test={Nl_test}")

# Check element-wise
for l in [10, 30, 50, 80]:
    for L in [20, 40, 60, 90]:
        lhs = Mp_mat_notes[l, L] * wl_ref[L]
        rhs = M_mat[l, L] * pf_ell[L]
        if abs(rhs) > 0:
            ratio = lhs / rhs
        else:
            ratio = np.nan
        if l == 10 or L == 20:
            print(f"  l={l:3d}, L={L:3d}: LHS={lhs:.6e}, RHS={rhs:.6e}, ratio={ratio:.6f}")

# Now check for many elements
ratios = []
for l in range(5, Nl_test-5):
    for L in range(5, Nl_test-5):
        if abs(M_mat[l,L]) > 1e-10 * np.abs(M_mat).max():
            lhs = Mp_mat_notes[l, L] * wl_ref[L]
            rhs = M_mat[l, L] * pf_ell[L]
            if abs(rhs) > 0:
                ratios.append(lhs/rhs)
ratios = np.array(ratios)
print(f"\n  All elements: mean ratio = {np.mean(ratios):.6f}, std = {np.std(ratios):.6f}")
print(f"  min = {np.min(ratios):.6f}, max = {np.max(ratios):.6f}")

# ---- What the code actually computes ----
# theory_code = M_mat @ cl_true_notes
# where cl_true_notes = P_F / (32 pi^3 chi^2)
# 
# What should theory_pair be?
# <Cl> = 1/(4pi * 2pi * chi^2) * SUM_L M^(p)_lL * PLKjKk_L
#       = 1/(4pi * 2pi * chi^2) * SUM_L (4pi * Mp_mat[l,L]) * (4pi * wl_ref[L])
#       = (4pi)^2 / (4pi * 2pi * chi^2) * SUM_L Mp_mat[l,L] * wl_ref[L]
#       = (4pi) / (2pi * chi^2) * SUM_L Mp_mat[l,L] * wl_ref[L]
#       = 2/(chi^2) * SUM_L Mp_mat[l,L] * wl_ref[L]
#
# Hmm, let me reconsider. Let me use the actual pair-counting formula:
# <Cl> = 1/(4pi) * SUM_{j,k} <w_j w_k> P_l(cos gamma)
# 
# From the Limber approx:
# <w_j w_k> = (N^2/L) P_F(k_perp=0) / chi^2  [my derivation, corrected]
# Wait, that's wrong too. <w_j w_k> is a function of gamma, not a constant.
# 
# The pair counting formula (Eq 14 in notes):
# <Cl> = 1/(4pi * 2pi * chi^2) * SUM_L M^(p)_lL * PLKjKk_L
#
# This should match cl_mean.
#
# Let me compute this directly.
print(f"\n=== Direct pair-counting theory ===")
cl_mean = np.mean(np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')['cl_k'], axis=0)

# M^(p) at Nl=500 (with more Lambda for inner sum)
Nl_pc = 500
lam_max_pc = 999  # inner sum range (lambda from 0 to lam_max_pc)
lam_ext = np.arange(lam_max_pc, dtype=float)
pf_lam_ext = b1**2 * plin((lam_ext + 0.5)/chi_bar)

# CoupleMat gives M/(4pi) type matrix: (2L+1)/(4pi) SUM (2lam+1) weight 3j^2
couple_pc = Wigner3j.CoupleMat(Nl_pc, pf_lam_ext)
Mp_code = couple_pc.compute_matrix()  # shape (Nl_pc, Nl_pc)
# M^(p)_notes = 4*pi * Mp_code
Mp_notes = 4*np.pi * Mp_code

# PLKjKk at L from 0 to Nl_pc-1
PLKjKk_L = PLKjKk[:Nl_pc]

# Theory from pair counting:
# <Cl> = 1/(4*pi * 2*pi * chi^2) SUM_L M^(p)_lL * PLKjKk_L
theory_pair = (1.0 / (4*np.pi * 2*np.pi * chi_bar**2)) * (Mp_notes @ PLKjKk_L)

print(f"Ratio data/theory_pair at ell=100: {cl_mean[100]/theory_pair[100]:.4f}")
print(f"Ratio data/theory_pair at ell=200: {cl_mean[200]/theory_pair[200]:.4f}")
print(f"Ratio data/theory_pair at ell=300: {cl_mean[300]/theory_pair[300]:.4f}")
print(f"Mean ratio (ell 30-450): {np.mean(cl_mean[30:450]/theory_pair[30:450]):.4f}")

# Also try without the (4*pi)^2 convention:
# <Cl_plot> = <Cl> / (4*pi)^2, so
# theory_pair_plot = theory_pair / (4*pi)^2
print(f"\nWith (4pi)^2 in denominator:")
theory_pair_plot = theory_pair / (4*np.pi)**2
print(f"Ratio data/theory_pair_plot at ell=100: {cl_mean[100]/theory_pair_plot[100]:.6f}")

# Compare with the MASTER approach used in the code:
# M_code @ cl_true_notes
wl_code = np.zeros(2*Nl_pc - 1)
wl_code[:min(Nl, 2*Nl_pc - 1)] = wl_ref[:min(Nl, 2*Nl_pc - 1)]
couple_code = Wigner3j.CoupleMat(Nl_pc, wl_code)
M_code = couple_code.compute_matrix()

ells_pc = np.arange(Nl_pc, dtype=float)
cl_true_notes = b1**2 * plin((ells_pc + 0.5)/chi_bar) / (32*np.pi**3 * chi_bar**2)
theory_master = M_code @ cl_true_notes

print(f"\n=== MASTER code formula ===")
print(f"Ratio data/theory_master at ell=100: {cl_mean[100]/theory_master[100]:.4f}")
print(f"Ratio data/theory_master at ell=200: {cl_mean[200]/theory_master[200]:.4f}")
print(f"Mean ratio (ell 30-450): {np.mean(cl_mean[30:450]/theory_master[30:450]):.4f}")

# Compare pair-counting theory vs MASTER theory
print(f"\n=== pair-counting vs MASTER ===")
print(f"theory_pair / theory_master at ell=100: {theory_pair[100]/theory_master[100]:.4f}")
print(f"theory_pair / theory_master at ell=200: {theory_pair[200]/theory_master[200]:.4f}")
