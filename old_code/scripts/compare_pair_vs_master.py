#!/usr/bin/env python
"""
Compare pair-counting theory vs MASTER theory.

Pair-counting:  C_theory = M^(p) @ PLKjKk / (4*pi) / (2*pi*chi^2) / (4*pi)^2
MASTER:         C_theory = M_W @ C_true = M_W @ (P_F / (32*pi^3*chi^2))

Identity 2 says: M^(p) * W = M_W * P_F
So: M^(p) @ PLKjKk = M^(p) @ (4*pi * W) = 4*pi * M^(p) @ W
And: C_pair = M^(p) @ (4*pi*W) / (4*pi) / (2*pi*chi^2) / (4*pi)^2
           = M^(p) @ W / (2*pi*chi^2) / (4*pi)^2

But M^(p)_lL * W_L = M_lL * P_F(L/chi), so:
  SUM_L M^(p)_lL * W_L = SUM_L M_lL * P_F(L/chi)
So C_pair = SUM_L M_lL * P_F(L/chi) / (2*pi*chi^2) / (4*pi)^2
          = SUM_L M_lL * P_F(L/chi) / (32*pi^3*chi^2)
          = SUM_L M_lL * C_true[L]
          = C_MASTER

BUT WAIT: The identity M^(p)*W = M_W*P_F holds POINTWISE for each (l,L).
However, in the pair-counting approach, BOTH M^(p) and PLKjKk are limited to L<Nl.
In the MASTER approach, M_W extends to L<Nl_large >> Nl.

The key: coupling_pk = CoupleMat(Nl, pk_array) where pk_array has the P(k) values.
This matrix is Nl x Nl. It's multiplied by PLKjKk which is also length Nl.
Both sums run from 0 to Nl-1.

In contrast, the MASTER uses M_W which is Nl_large x Nl_large, multiplied by
C_true (also length Nl_large), summed over L from 0 to Nl_large-1.

So the pair-counting approach computes:
  SUM_{L=0}^{Nl-1} M^(p)_{l,L} * PLKjKk[L] / (4*pi) / (2*pi*chi^2)

While the MASTER computes:
  SUM_{L=0}^{Nl_large-1} M_{l,L}^{(W)} * C_true[L]

These are the SAME only if:
  SUM_{L=0}^{Nl-1} M^(p)_{l,L} * W_L = SUM_{L=0}^{Nl-1} M_{l,L}^{(W)} * P_F(L/chi)

The identity says M^(p)_{l,L} * W_L = M_{l,L}^{(W)} * P_F(L/chi} for EACH (l,L).
So the sum identity holds term by term. BOTH approaches truncate at L<Nl.

So the pair-counting approach gives the SAME result as MASTER at Nl_large=Nl=500!
And we showed that MASTER at Nl_large=500 gives theory/data = 0.865.
That contradicts the notes claiming ratio 0.96!

Unless... the pair-counting approach in the actual code includes DIFFERENT lambda ranges
in the coupling matrix. Let me check.
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

# Load PLKjKk (both lambda=2000 and lambda=4000)
PLKjKk_2000 = np.load('notebooks/data/PLKjKk_lambda2000.npy')
PLKjKk_4000 = np.load('notebooks/data/PLKjKk_lambda4000.npy')

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, L={L:.1f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# ---- Approach 1: Pair-counting (original code) ----
# coupling_pk = CoupleMat(Nl, pk_array) with P(k) as input
# C_theory = coupling_pk @ PLKjKk[:Nl] / (4*pi) / (2*pi*chi^2) / (4*pi)^2
ells = np.arange(Nl, dtype=float)
pk_for_coupling = b1_ref**2 * plin_ref((ells + 0.5) / chi_bar)  # P_F(ell/chi)

couple_pk = Wigner3j.CoupleMat(Nl, pk_for_coupling)
M_pk = couple_pk.compute_matrix()

# PLKjKk at L=0..Nl-1
PLKjKk_Nl = PLKjKk_2000[:Nl]

C_pair_raw = M_pk @ PLKjKk_Nl  # raw pair-counting product
C_pair = C_pair_raw / (4*np.pi) / (2*np.pi * chi_bar**2)
C_pair_plot = C_pair / (4*np.pi)**2
print(f"\nPair-counting theory (Nl={Nl}):")
print(f"  Mean C_pair_plot[10:]  = {np.mean(C_pair_plot[10:]):.6e}")
print(f"  Mean cl_mean[10:]     = {np.mean(cl_mean[10:]):.6e}")
print(f"  Ratio cl_mean/C_pair_plot (ell>10) = {np.mean(cl_mean[10:]/C_pair_plot[10:]):.4f}")

# ---- Approach 2: MASTER at Nl_large=500 ----
C_true = b1_ref**2 * plin_ref((ells + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

couple_W = Wigner3j.CoupleMat(Nl, wl_ref)  # uses wl_ref as window
M_W = couple_W.compute_matrix()
C_master_500 = M_W @ C_true

print(f"\nMASTER (Nl={Nl}, wl_ref truncated at {Nl}):")
print(f"  Mean C_master_500[10:]  = {np.mean(C_master_500[10:]):.6e}")
print(f"  Ratio cl_mean/C_master_500 (ell>10) = {np.mean(cl_mean[10:]/C_master_500[10:]):.4f}")

# ---- Compare pair-counting vs MASTER ----
print(f"\nPair-counting / MASTER ratio (ell>10): {np.mean(C_pair_plot[10:]/C_master_500[10:]):.6f}")

# ---- Approach 3: MASTER at Nl_large=500 with EXTENDED wl (PLKjKk to 2000) ----
wl_ext_2000 = PLKjKk_2000 / (4*np.pi)
wl_for_500 = np.zeros(2*Nl-1)  # 999 entries
wl_for_500[:min(999, len(wl_ext_2000))] = wl_ext_2000[:min(999, len(wl_ext_2000))]

couple_W_ext = Wigner3j.CoupleMat(Nl, wl_for_500)
M_W_ext = couple_W_ext.compute_matrix()
C_master_500_ext = M_W_ext @ C_true

print(f"\nMASTER (Nl={Nl}, wl from PLKjKk to lambda={min(999, len(wl_ext_2000))}):")
print(f"  Mean C_master_500_ext[10:]  = {np.mean(C_master_500_ext[10:]):.6e}")
print(f"  Ratio cl_mean/C_master_500_ext (ell>10) = {np.mean(cl_mean[10:]/C_master_500_ext[10:]):.4f}")

# ---- What's different between wl_ref and wl_for_500? ----
# wl_ref comes from hp.alm2cl(hran) which is the SHT window with Nl=500.
# wl_for_500 comes from PLKjKk/(4pi) — the pair-counting with arbitrary lambda range.
# These should be the same for lambda < Nl=500, but wl_for_500 extends to 999
# while wl_ref stops at 499.
print(f"\nwl comparisons:")
print(f"  wl_ref length: {len(wl_ref)}")
print(f"  wl_for_500 length: {len(wl_for_500)}")
print(f"  wl_ref vs wl_ext for lambda<Nl:")
diff = np.max(np.abs(wl_ref[:Nl] - wl_ext_2000[:Nl]))
print(f"    Max absolute diff: {diff:.4e}")
print(f"    Max relative diff: {np.max(np.abs(wl_ref[:Nl] - wl_ext_2000[:Nl]) / wl_ext_2000[:Nl]):.6e}")

# AH HA! The CoupleMat(Nl, wl_ref) uses wl_ref which has length Nl=500.
# But CoupleMat needs wl up to lambda = 2*(Nl-1) = 998.
# If wl_ref only has 500 entries, the inner lambda sum is truncated at 499!
# This means the MASTER at Nl=500 with wl_ref only sums lambda from 0 to 499.
# But with wl_for_500 (which has lambda up to 999), the inner sum goes to 998.
# This matters because for diagonal elements l=l', the sum goes to 2*l.
# For l=499: lambda goes up to 998. With wl_ref (length 500), lambda only goes to 499.
# HALF the inner sum is missing!

# And the pair-counting approach: CoupleMat(Nl, pk_array) uses pk_array with length Nl=500.
# Its inner lambda sum also uses wl_ref (the pk_array) up to lambda=499 only.
# But wait — in the pair-counting approach, coupling_pk@PLKjKk is:
# SUM_L SUM_lambda (2lambda+1) pk[lambda] (3j)^2 * PLKjKk[L]
# Here pk acts as the inner sum weight, and PLKjKk[L] acts as the outer.
# The inner lambda sum goes from 0 to min(l+L, Nl-1).
# But wait, pk_array is only Nl=500 long, so the inner sum is truncated at 499.

# ACTUALLY, let me look at how CoupleMat works:
print(f"\nCoupleMat inner sum range:")
print(f"  Wigner3j.CoupleMat(Nl={Nl}, wl of length {len(wl_ref)})")
print(f"  Internal: Nl_3j = 2*(Nl-1) = {2*(Nl-1)}")
print(f"  But wl is zero-padded to 2*(Nl-1)+1 = {2*(Nl-1)+1}")

# Let me check: does CoupleMat zero-pad wl if it's shorter than 2*Nl-1?
# Looking at fast_Wigner3j.py...
