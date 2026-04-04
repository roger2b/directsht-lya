#!/usr/bin/env python
"""
Verify: pair-counting at lambda_max=1000 ≡ MASTER at Nl_large=1000

The old code (make_final_plots.py) uses:
  couple_pk = CoupleMat(lambda_max=1000, pk_L)     # 1000x1000 matrix
  PLKjKk computed to lambda=1000                    # 1000 entries
  C_theory = coupling_pk @ PLKjKk / (4*pi) / (2*pi*chi^2) / (4*pi)^2

BUT: pk_L inside CoupleMat is ALSO zero-padded to 2*999=1998 entries.
pk_L has 1000 entries, so pk_L[1000:1998] = 0. This truncates the inner sum.

Meanwhile the MASTER at Nl_large=1000 uses wl (from PLKjKk/(4pi)) zero-padded to 1998.
But PLKjKk is only 1000 entries (in old code) → wl[0:999] used, wl[1000:1998]=0.

So the inner sums are different:
  Pair-counting inner: uses pk_L[lambda] for lambda=0..1998, but pk_L[1000:]=0
  MASTER inner: uses wl[lambda] for lambda=0..1998, but wl[1000:]=0

These are NOT the same because the roles of lambda and L differ.

Actually, let me re-read Identity 2 carefully:
  M^(p)_{l,L} * W_L = M^(W)_{l,L} * P_F(L/chi)

M^(p) is built with P(k) as inner weight:
  M^(p)_{l,L} = (2L+1) * SUM_lambda (2lam+1) P_F(lam/chi) * (3j(l,L,lam;0,0,0))^2

M^(W) is built with W as inner weight:
  M^(W)_{l,L} = (2L+1)/(4pi) * SUM_lambda (2lam+1) W_lambda * (3j(l,L,lam;0,0,0))^2

The 3j is symmetric in (l,L,lambda). So:
  M^(p)_{l,L} = (2L+1) * SUM_lambda (2lam+1) P_F(lam/chi) * (3j)^2
The P_F enters as a weight on LAMBDA in the inner sum.

The pair-counting theory:
  C_pair_plot[l] = SUM_L M^(p)_{l,L} * PLKjKk[L] / (4*pi) / (2*pi*chi^2) / (4*pi)^2

Substituting PLKjKk = 4*pi * W:
  = SUM_L M^(p)_{l,L} * 4pi * W_L / (4*pi) / (2*pi*chi^2) / (4*pi)^2
  = SUM_L M^(p)_{l,L} * W_L / (2*pi*chi^2) / (4*pi)^2

Using Identity 2: M^(p)_{l,L} * W_L = M^(W)_{l,L} * P_F(L/chi)
  = SUM_L M^(W)_{l,L} * P_F(L/chi) / (2*pi*chi^2) / (4*pi)^2
  = SUM_L M^(W)_{l,L} * P_F(L/chi) / (32*pi^3 * chi^2)
  = SUM_L M^(W)_{l,L} * C_true[L]
  = MASTER[l]

So they ARE equivalent IF both M^(p) and M^(W) have the same lambda range.

In the old code with lambda_max=1000:
  M^(p) = CoupleMat(1000, pk_L) where pk_L has 1000 entries → inner lambda to 998 (padded to 1998 but pk[1000:]=0)
  PLKjKk has 1000 entries → outer L sum goes to L=999

For MASTER Nl_large=1000 with wl from PLKjKk (1000 entries → padded to 1998 but wl[1000:]=0):
  M^(W) = CoupleMat(1000, wl_1000) where wl has 1000 entries → inner lambda to 998 (same)
  C_true has 1000 entries → outer L sum goes to L=999

Hmm, but the lambda range in the inner sum of M^(p) uses pk_L which is the POWER SPECTRUM,
while M^(W) uses wl which is the WINDOW. These are very different arrays.
Let me think about this more carefully...

The inner sum in _compute_matrix:
  for l in range(j-i, i+j+1, 2):
    tmp += (2*l+1) * wl[l] * [3j stuff]

When computing M^(p), wl = pk_L (1000 entries, padded to 1998).
  pk_L[l] = b1^2 * Plin((l+0.5)/chi) for l < 1000, 0 for l >= 1000.

When computing M^(W), wl = wl_window (1000 entries from PLKjKk/4pi, padded to 1998).
  wl[l] = PLKjKk[l]/(4pi) for l < 1000, 0 for l >= 1000.

These are completely different wl arrays! The identity M^(p)*W = M^(W)*P does NOT
mean the two approaches are identical when we truncate at the same Nl.
It's a POINTWISE identity, but:
  M^(p) is built with pk, M^(W) is built with W_lambda.
The inner sums use DIFFERENT weights.

Let me just compute both and see if they match numerically.
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
L_ = float(d['L'])
Nskew = int(d['Nskew'])
Nl = 500
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)

# Load PLKjKk
PLKjKk_2000 = np.load('notebooks/data/PLKjKk_lambda2000.npy')

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L_/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, L={L_:.1f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# ---- Pair-counting at lambda_max=1000 (old code approach) ----
lambda_max = 1000
L_range = np.arange(lambda_max, dtype=float)
pk_L = b1_ref**2 * plin_ref((L_range + 0.5) / chi_bar)

PLKjKk_1000 = PLKjKk_2000[:lambda_max]  # first 1000 entries

couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
M_pk = couple_pk.compute_matrix()

# Full pair-counting: matrix times PLKjKk, divided by norms
C_pair = M_pk @ PLKjKk_1000 / (4*np.pi) / (2*np.pi * chi_bar**2)
C_pair_plot = C_pair / (4*np.pi)**2
del couple_pk, M_pk; gc.collect()

print(f"\nPair-counting (lambda_max={lambda_max}):")
print(f"  Mean C_pair_plot[10:500] = {np.mean(C_pair_plot[10:500]):.6e}")
print(f"  Mean cl_mean[10:500]    = {np.mean(cl_mean[10:]):.6e}")
print(f"  Ratio data/theory (ell 10..499) = {np.mean(cl_mean[10:] / C_pair_plot[10:500]):.4f}")

# ---- MASTER at Nl_large=1000 with wl from PLKjKk ----
wl_1000 = PLKjKk_2000[:lambda_max] / (4*np.pi)  # only first 1000 lambdas

# Coupling matrix uses wl (window function)
couple_W = Wigner3j.CoupleMat(lambda_max, wl_1000)
M_W = couple_W.compute_matrix()

# C_true
ells_1000 = np.arange(lambda_max, dtype=float)
C_true_1000 = b1_ref**2 * plin_ref((ells_1000 + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

C_master = (M_W @ C_true_1000)[:Nl]
del couple_W, M_W; gc.collect()

print(f"\nMASTER (Nl_large={lambda_max}, wl to {lambda_max}):")
print(f"  Mean C_master[10:500]    = {np.mean(C_master[10:Nl]):.6e}")
print(f"  Ratio data/theory (ell 10..499) = {np.mean(cl_mean[10:] / C_master[10:]):.4f}")

# ---- Compare ----
print(f"\nPair/MASTER ratio (ell 10..499): {np.mean(C_pair_plot[10:500] / C_master[10:Nl]):.6f}")

# ---- MASTER at Nl_large=1000 with wl from PLKjKk to 2000 ----
wl_2000_for_1000 = np.zeros(2*lambda_max - 1)  # needs 1999 entries
n_avail = min(1999, len(PLKjKk_2000))
wl_2000_for_1000[:n_avail] = PLKjKk_2000[:n_avail] / (4*np.pi)

couple_W2 = Wigner3j.CoupleMat(lambda_max, wl_2000_for_1000)
M_W2 = couple_W2.compute_matrix()
C_master2 = (M_W2 @ C_true_1000)[:Nl]
del couple_W2, M_W2; gc.collect()

print(f"\nMASTER (Nl_large={lambda_max}, wl to 2000, inner sums complete):")
print(f"  Mean C_master2[10:500]   = {np.mean(C_master2[10:Nl]):.6e}")
print(f"  Ratio data/theory (ell 10..499) = {np.mean(cl_mean[10:] / C_master2[10:]):.4f}")

# ---- So the pair-counting approach at lambda_max=1000 with PLKjKk to 1000 ----
# gives data/theory = ???. Let me check if this matches the notes' 0.96.
# The notes say "ratio ≈ 0.96–1.04" with 20 sims and lambda_max=1000.
# Wait, that's data/theory or theory/data?
# The notes say "langle Cl rangle_meas / langle Cl rangle_theory ≈ 0.96–1.04"
# So data/theory. If we get ~1.0 at lambda_max=1000, that matches.

print(f"\n---- Summary ----")
print(f"Pair-counting (lambda=1000, PLKjKk 1000): data/th = {np.mean(cl_mean[10:]/C_pair_plot[10:500]):.4f}")
print(f"MASTER (Nl_large=1000, wl 1000):          data/th = {np.mean(cl_mean[10:]/C_master[10:]):.4f}")
print(f"MASTER (Nl_large=1000, wl 2000):          data/th = {np.mean(cl_mean[10:]/C_master2[10:]):.4f}")
