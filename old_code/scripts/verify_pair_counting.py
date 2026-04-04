#!/usr/bin/env python
"""
Direct verification of the pair-counting formula.

The pair sum (exact):
  <Cl> = (1/(4pi)) SUM_{j,k} <w_j w_k> P_l(cos gamma_{jk})

Rewriting using products of Legendre polynomials:
  P_l(x) P_lambda(x) = SUM_L (2L+1) C(l,lambda,L) P_L(x)
where C(l,lambda,L) = 3j(l,lambda,L;0,0,0)^2

So <Cl> = (1/(4pi)) SUM_{j,k} SUM_lambda (2lambda+1)/(4pi) C_lambda P_lambda(cos gamma) P_l(cos gamma)

Hmm this is getting complicated. Let me instead just _directly compute_ the pair-counting 
prediction using the MASTER matrix and various C_true formulas, and check convergence.

The key insight: the notes say the pair-counting formula is
  <Cl> = 1/(4pi * 2pi * chi^2) SUM_L M^(p)_lL * PLKjKk_L

But this 2pi might be wrong. Let me figure out where the 2pi actually comes from.

For the Limber approximation at k_par=0:
  <w_j w_k> = (b1^2 N^2/L^3) SUM_{k_perp} P_lin(K_perp) exp(i K_perp . Delta_r)

where K_perp = 2pi k_perp/L and Delta_r = chi_bar * (n_j - n_k) projected.

Converting sum to integral:
  SUM -> (L/(2pi))^2 integral d^2K = L^2/(2pi)^2 integral 2pi K dK

So:
  <w_j w_k> = (b1^2 N^2/L^3) * L^2/(2pi) integral K dK P_lin(K) J_0(K chi_bar gamma)
            = (b1^2 N^2)/(2pi L) integral K dK P_lin(K) J_0(K chi_bar gamma)

Now, using the flat-sky angular power spectrum relation:
  xi(gamma) = integral l dl/(2pi) C_l J_0(l gamma)
  with C_l = P_2D(l/chi)/chi^2

So:
  <w_j w_k> = integral K dK/(2pi) * (b1^2 N^2/L) * P_lin(K) * J_0(K chi gamma)
            = integral l dl/(2pi) * (b1^2 N^2/L) P_lin(l/chi) J_0(l gamma) / chi^2

[using K = l/chi, dK = dl/chi, K dK = l dl/chi^2]

So the angular power spectrum of w is:
  C_l^{ww} = (b1^2 N^2 / L) * P_lin(l/chi) / chi^2

Now, the pair sum:
  <Cl_pseudo> = (1/(4pi)) SUM_{j,k} <w_j w_k> P_l(cos gamma_{jk})

Using <w_j w_k> = SUM_L (2L+1)/(4pi) C_L^{ww} P_L(cos gamma):
  <Cl_pseudo> = (1/(4pi)) SUM_{j,k} SUM_L (2L+1)/(4pi) C_L^{ww} P_L(cos gamma) * P_l(cos gamma)

Now, P_l(x) P_L(x) = SUM_lambda (2lambda+1) 3j(l,L,lambda;0,0,0)^2 P_lambda(x)

So:
  <Cl_pseudo> = (1/(4pi)) SUM_L (2L+1)/(4pi) C_L SUM_lambda (2lambda+1) 3j^2 SUM_{j,k} P_lambda(cos gamma)

And SUM_{j,k} P_lambda(cos gamma) = (4pi) W_lambda_mask  where W_lambda_mask = (1/(2lambda+1)) |SUM_j Y_lm|^2

So: <Cl_pseudo> = SUM_L (2L+1)/(4pi) C_L SUM_lambda (2lambda+1) W_lambda_mask 3j^2
    = SUM_L M_mask[l,L] C_L

This is the standard MASTER formula with M_mask built from W_lambda_mask.

In the code: wl_code = N^2 * W_lambda_mask (because the SHT uses weight N).
So CoupleMat gives: M_code[l,L] = (2L+1)/(4pi) SUM (2lambda+1) wl_code[lambda] 3j^2
                                  = N^2 * M_mask[l,L]

Therefore: <Cl_pseudo> = M_mask @ C_ww = (M_code/N^2) @ C_ww = M_code @ (C_ww/N^2)

So C_true_for_code = C_ww / N^2 = b1^2 P_lin(l/chi) / (L chi^2)

Let me verify this numerically.
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
wl_ref = d['wl_k'][0, :Nl]

PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, L={L:.2f}, chi_bar={chi_bar:.2f}, b1={b1:.4f}")
print(f"N^2={N**2}")
print()

# KEY: the wl used by CoupleMat includes N^2 already.
# wl_ext = PLKjKk/(4pi) = W_lambda from SHT with weight N per sightline
# So wl_ext = N^2 * W_lambda_mask_pure (where pure = SHT with weight 1)
# And M_code = N^2 * M_mask_pure

# To get the right C_true for the code:
# <Cl> = M_mask_pure @ C_ww = (M_code/N^2) @ C_ww
# So M_code @ C_true = M_code @ (C_ww/N^2)
# C_true = C_ww / N^2 = b1^2 P_lin / (L chi^2)

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

print(f"{'Nl_large':>8s} {'data/th(L)':>12s} {'data/th(32pi3)':>14s} {'data/th(2piL)':>14s}")
print("-" * 55)

for Nl_large in [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500]:
    ells_ext = np.arange(Nl_large, dtype=float)
    plin_vals = plin((ells_ext + 0.5)/chi_bar)
    
    # Theory: C_true = b1^2 P / (X * chi^2) where X is the unknown factor
    cl_over_chi2 = b1**2 * plin_vals / chi_bar**2
    
    wl_needed = 2*Nl_large - 1
    wl_for_c = np.zeros(wl_needed)
    n_avail = min(wl_needed, len(wl_ext))
    wl_for_c[:n_avail] = wl_ext[:n_avail]
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_c)
    M = couple.compute_matrix()
    
    theory_unnorm = (M @ cl_over_chi2)[:Nl]  # = M_code @ (b1^2 Plin/chi^2)
    
    # data/theory = X (the unknown denominator)
    # data = M_code @ C_true = M_code @ [b1^2 P / (X chi^2)]
    # So data/theory_unnorm = 1/X
    X_inv = np.mean(cl_mean[mask]) / np.mean(theory_unnorm[mask])
    X = 1.0 / X_inv
    
    r_L = X / L  # If X=L, this should be 1
    r_32pi3 = X / (32*np.pi**3)  # If X=32pi^3, this should be 1
    r_2piL = X / (2*np.pi*L)  # Let's also check 2piL
    
    print(f"  {Nl_large:5d}  {r_L:12.4f} {r_32pi3:14.4f} {r_2piL:14.4f}  (X={X:.2f})")
    
    del couple, M; gc.collect()

# Now let me also try the pair-counting route directly.
# <Cl> = (1/(4pi)) SUM_{j,k} <w_j w_k> P_l(cos gamma)
# where SUM_{j,k} P_lambda(cos gamma) = PLKjKk_lambda / N^2

# The Limber expression:
# <w_j w_k>(gamma) = SUM_L (2L+1)/(4pi) * C_L^{ww} * P_L(cos gamma)
# where C_L^{ww} = b1^2 N^2 P_lin(L/chi) / (L chi^2)

# <Cl> = (1/(4pi)) SUM_{j,k} SUM_L (2L+1)/(4pi) C_L P_L(cos gamma) P_l(cos gamma)
# Expanding P_l * P_L = SUM_lambda (2lambda+1) 3j^2 P_lambda:
# <Cl> = (1/(4pi)) SUM_L (2L+1)/(4pi) C_L SUM_lambda (2lambda+1) 3j^2 * [SUM_{j,k} P_lambda(cos gamma)]

# Now SUM_{j,k} P_lambda(cos gamma) = PLKjKk_lambda / N^2
# (because PLKjKk = N^2 * SUM_{jk} P_lambda)

# So: <Cl> = (1/(4pi)) SUM_L (2L+1)/(4pi) C_L SUM_lambda (2lambda+1) 3j^2 PLKjKk_lambda / N^2

# The inner sum SUM_lambda (2lambda+1) PLKjKk_lambda 3j^2 / N^2 = SUM (2lambda+1) (4pi wl_ref[lambda]) 3j^2 / N^2

# WAIT: wl_ref includes N^2 (from the SHT with weight N). Let me clarify.
# PLKjKk_lambda = 4pi * wl_ref  [Identity 1, verified]
# wl_ref = N^2 * (1/(2lambda+1)) |SUM_j Y_lm|^2  [from the SHT with weight N]

# SUM_{j,k} P_lambda(cos gamma) = (4pi/(2lambda+1)) * |SUM_j Y_lm|^2 [from addition thm]
#                                 = (4pi/N^2) * wl_ref

# So PLKjKk_lambda = N^2 * SUM_{j,k} P_lambda = N^2 * (4pi/N^2) * wl_ref = 4pi * wl_ref
# Consistent!

# And in the pair sum:
# <Cl> = (1/(4pi)) SUM_L (2L+1)/(4pi) C_L SUM_lambda (2lambda+1) 3j^2 (4pi/N^2)*wl_ref[lambda]
#       = (1/N^2) SUM_L (2L+1)/(4pi) C_L SUM_lambda (2lambda+1) wl_ref[lambda] 3j^2
#       = (1/N^2) SUM_L M_code[l,L] * C_L^{ww}
#       = M_code @ (C_L^{ww}/N^2)

# This confirms: C_true_for_code = C_ww / N^2 = b1^2 P_lin / (L chi^2)

# But we just showed that X converges to something that's NOT L!
# Let's see what value X actually converges to...

print(f"\n=== Summary ===")
print(f"L = {L:.4f}")
print(f"32*pi^3 = {32*np.pi**3:.4f}")
print(f"2*pi*L = {2*np.pi*L:.4f}")
print(f"\nThe derivation says C_true = b1^2 P / (L chi^2)")
print(f"But the numerical X value keeps growing, meaning the sum isn't converged.")
print(f"The question is: WHERE does the extra power come from?")
print(f"\nPossibility: the Limber approximation <w_j w_k> = integral P_F exp(ik.dr)")
print(f"breaks down for pairs at EXACTLY the same position (j=k)")
print(f"The diagonal term <w_j^2> has a different structure.")
