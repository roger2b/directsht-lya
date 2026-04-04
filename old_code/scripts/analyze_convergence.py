#!/usr/bin/env python
"""
Analyze why the MASTER ell' sum doesn't converge.
Key insight: compute C_true at high ell to see where the power lies,
then compute the expected pair-counting prediction directly.
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
print(f"b1={b1_ref:.4f}")

# ---- Part 1: C_true power distribution at high ell ----
kNy = np.pi / (L/N)
ell_Ny = kNy * chi_bar
print(f"\nk_Nyquist = {kNy:.4f} h/Mpc, ell_Nyquist = {ell_Ny:.0f}")

ells_high = np.arange(7000)
C_true_high = b1_ref**2 * plin_ref((ells_high + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

print("\nC_true at selected ell:")
for ell in [0, 100, 499, 1000, 2000, 3000, 5000, 6600]:
    k = (ell + 0.5) / chi_bar
    print(f"  ell={ell:5d}: C_true={C_true_high[ell]:.4e}, k={k:.4f} h/Mpc")

print("\nTotal (2l+1)*Cl per band:")
for l_lo, l_hi in [(0,500), (500,1000), (1000,2000), (2000,3000), (3000,5000), (5000,7000)]:
    band = C_true_high[l_lo:l_hi]
    total = np.sum((2*ells_high[l_lo:l_hi]+1) * band)
    frac = total / np.sum((2*ells_high[:7000]+1) * C_true_high[:7000])
    print(f"  l={l_lo:5d}..{l_hi-1:5d}: {total:.4e} ({100*frac:.1f}%)")

# ---- Part 2: Check PLKjKk = 4pi * wl identity ----
print("\n---- PLKjKk vs wl_ref ----")
print(f"wl_ref[0] = {wl_ref[0]:.4e}")
print(f"PLKjKk[0]/(4pi) = {PLKjKk[0]/(4*np.pi):.4e}")
ratios_wl = wl_ref[:min(Nl, lambda_max_data)] / wl_full[:min(Nl, lambda_max_data)]
print(f"Ratio wl_ref/(PLKjKk/(4pi)): mean={np.mean(ratios_wl):.6f}, std={np.std(ratios_wl):.6f}")
print(f"  first 5: {ratios_wl[:5]}")

# ---- Part 3: Direct approach ----
# The MASTER formula has ell' sum convergence issues.
# Alternative: compute <pseudo-Cl> = (1/(4pi)) * SUM_lambda (2lambda+1) C_true[lambda] * PLKjKk[lambda] / W0
# where W0 normalizes.
#
# Actually, the pair-counting (coupling by pk) approach is:
#   coupling_pk[l,lambda] = (2*lambda+1)/(4*pi) * (3j(l,0,lambda;0,0,0))^2
#   theory_pseudo_Cl[l] = SUM_lambda coupling_pk[l,lambda] * PLKjKk[lambda] * C_true[l] ... no
#
# Wait. The pair-counting theory is:
#   <pseudo-Cl> = (1/(4pi)) SUM_{j,k} C_jk * Pl(cos_jk)
# where C_jk = <delta_j * delta_k> is the angular correlation at the angular separation of j,k.
# 
# For a field with angular power spectrum C_true[l']:
#   C_jk = C(theta_jk) = SUM_{l'} (2l'+1)/(4pi) * C_true[l'] * Pl'(cos_jk)
#
# So <pseudo-Cl> = (1/(4pi)) SUM_{j,k} [SUM_{l'} (2l'+1)/(4pi) C_true[l'] Pl'(cos_jk)] * Pl(cos_jk)
#               = SUM_{l'} (2l'+1)/(4pi) C_true[l'] * [(1/(4pi)) SUM_{j,k} Pl'(cos_jk) Pl(cos_jk)]
#
# The inner piece [(1/(4pi)) SUM_{j,k} Pl'(cos_jk) Pl(cos_jk)] is NOT the coupling matrix.
# The coupling matrix defined by CoupleMat uses the 3j symbols:
#   M[l,l'] = (2l'+1)/(4pi) * SUM_lambda (2lambda+1) * wl[lambda] * (3j(l,l',lambda))^2
#
# The pair sum of two Legendre polynomials can be expanded:
#   Pl(cos) * Pl'(cos) = SUM_lambda (2lambda+1) (3j(l,l',lambda))^2 P_lambda(cos)
# So (1/(4pi)) SUM_{j,k} Pl'(cos_jk) Pl(cos_jk)
#   = SUM_lambda (2lambda+1) (3j)^2 * [(1/(4pi)) SUM_{j,k} P_lambda(cos_jk)]
#   = SUM_lambda (2lambda+1) (3j)^2 * wl[lambda]    
#     [if wl[lambda] = (1/(4pi)) SUM_{j,k} P_lambda(cos_jk) -- the FULL window function]
#
# So M[l,l'] = (2l'+1)/(4pi) * SUM_lambda (2lambda+1) * wl_true[lambda] * (3j(l,l',lambda))^2
# where wl_true[lambda] = PLKjKk[lambda] / (4pi) for ALL lambda.
#
# The lambda sum runs from |l-l'| to l+l' (triangle inequality for 3j).
# So for l=499, l'=6000, lambda ranges from 5501 to 6499.
# We need PLKjKk[5501..6499] -- which we don't have.
#
# BUT: PLKjKk at high lambda oscillates around the shot-noise level SN = N^2 * Nskew.
# If we EXTRAPOLATE PLKjKk[lambda] = SN for lambda > 2000, then:
# M_signal[l,l'] uses the oscillations, M_shot[l,l'] uses the constant.
# For the shot-noise part: M_shot[l,l'] = (2l'+1)/(4pi) * SN * SUM_lambda (2lambda+1)(3j)^2 = (2l'+1)/(4pi) * SN * 1
#   because the 3j completeness says SUM_lambda (2lambda+1)(3j(l,l',lam;0,0,0))^2 = 1
#
# This is the same as M_shot[l,l'] = (2l'+1)/(4pi) * SN  -- a DIAGONAL-FREE coupling.
# It couples ALL l' to l with weight (2l'+1).
# The total theory for the shot contribution is:
# SUM_{l'=0}^{l_max} M_shot[l,l'] * C_true[l'] = SN/(4pi) * SUM_{l'} (2l'+1) C_true[l']
# This is a CONSTANT (independent of l), equal to SN/(4pi) * C_total, where
# C_total = SUM_{l'} (2l'+1) C_true[l'] = 4pi * sigma^2  (the total angular variance)
# So the shot contribution = SN * sigma^2.
#
# For the signal part beyond lambda=2000:
# The oscillations of PLKjKk around SN are small (3-10%).
# Their contribution is small. Let me compute what we get with SN extrapolation.

# First: the exact pair-counting approach with PLKjKk data up to 2000
# <pseudo-Cl> for each l in the MASTER formalism requires:
# M[l,l'] which needs lambda up to l+l'.
# For l' > 2000-l (e.g., l'=1501 for l=499), the lambda sum is incomplete.
#
# Alternative approach: compute the "pair-counting" formula differently.
# The pair-counting formula says:
# <pseudo-Cl> = (1/(4pi)) SUM_{j,k} KjKk * C_true_jk * Pl(cos_jk)
# where C_true_jk is the angular correlation function.
# 
# We have cos_jk pre-computed (the same matrix used for PLKjKk).
# The correlation function C(cos_jk) = SUM_{l'} (2l'+1)/(4pi) C_true[l'] Pl'(cos_jk)
# This is a high-ell Legendre sum that can be evaluated ONCE per unique angle,
# then combined with Pl(cos_jk).
#
# But Ns=9797 so cos_jk is a 9797x9797 matrix -- 96M pairs.
# And Pl(cos_jk) for l=0..499 requires evaluating Legendre at each pair.
# This is essentially recomputing PLKjKk but with C_true weights.
# Total: O(Nl * Ns^2 * ell_max) which is 500 * 96e6 * 7000 ~ 3.4e14 -- way too expensive.
#
# BUT we can be smarter. Instead of computing at each pair, we can use the PLKjKk data.
# <pseudo-Cl> = (1/(4pi)) SUM_lambda (2lambda+1) (3j(l,0,lambda))^2 ... no, this doesn't work directly.
#
# Actually the key identity is:
# <pseudo-Cl> = SUM_{l'} M[l,l'] C_true[l']
# = SUM_{l'} (2l'+1)/(4pi) C_true[l'] * SUM_lambda (2lambda+1) wl_true[lambda] (3j(l,l',lambda))^2
# = SUM_lambda (2lambda+1) wl_true[lambda] * SUM_{l'} (2l'+1)/(4pi) C_true[l'] (3j(l,l',lambda))^2
#
# Changing the order of summation:
# = SUM_lambda (2lambda+1) wl_true[lambda] * F[l, lambda]
# where F[l, lambda] = SUM_{l'} (2l'+1)/(4pi) C_true[l'] (3j(l,l',lambda))^2
#
# For F[l, lambda], the l' sum is bounded by the triangle inequality: |l-lambda| <= l' <= l+lambda.
# If lambda >> l, then l' ranges from lambda-l to lambda+l, so it's a narrow window around lambda.
# F[l, lambda] ~ (2lambda+1)/(4pi) C_true[lambda] * SUM_{l'~lambda} (3j)^2
# ~ C_true[lambda] / (4pi)   [using 3j completeness on l']
#
# Actually, by the orthogonality of 3j:
# SUM_{l'=0}^{infty} (2l'+1) (3j(l,l',lambda;0,0,0))^2 = 1
# So F[l, lambda] = (1/(4pi)) * [SUM_{l'} (2l'+1) C_true[l'] (3j)^2]
# This is like a "filtered" version of C_true.
# 
# The LAMBDA sum then becomes:
# <pseudo-Cl> = SUM_lambda (2lambda+1) wl_true[lambda] * F[l, lambda]
# = (1/(4pi)) SUM_lambda (2lambda+1) wl_true[lambda] * SUM_{l'} (2l'+1) C_true[l'] (3j(l,l',lambda))^2
#
# Now we can compute F[l, lambda] efficiently because for each (l, lambda), the l' range is
# |l-lambda| to l+lambda, which for large lambda is narrow: width = 2l+1.
# So for l=499, lambda=5000: l' ranges from 4501 to 5499, width 999.
# F[499, 5000] = (1/(4pi)) * SUM_{l'=4501}^{5499} (2l'+1) C_true[l'] (3j(499,l',5000))^2
# 
# This is computable, but we need (3j) values. The key: (3j(l,l',lambda;0,0,0))^2 for a range of l'.
# For large l and lambda, this is approximately (see e.g. Swarztrauber):
# (3j(l,l',lambda;0,0,0))^2 ~ 2/(pi * sqrt((l+l'+lambda)(l+l'-lambda+1)(l-l'+lambda)(l'-l+lambda+1)))
# for l+l'+lambda even (and 0 for odd).
#
# Alternative: use the Gaunt integral identity. Skip this for now.

# ---- Key realization ----
# Instead of all this, note that the F order-swap gives:
# <pseudo-Cl> = (1/(4pi)) SUM_lambda (2lambda+1) PLKjKk[lambda]/(4pi) * SUM_{l'} (2l'+1) C_true[l'] (3j)^2
# But using the completeness: SUM_{l'} (2l'+1) (3j(l,l',lambda;0,0,0))^2 = 1
# We get:  SUM_{l'} (2l'+1) C_true[l'] (3j)^2 ISN'T the same as C_true[lambda] or1.
# It's a WEIGHTED AVERAGE of C_true over l' near lambda (within 2l of lambda).
# For l<<lambda, the (3j)^2 acts like a narrow kernel of width ~2l+1 centered at l'~lambda.
# So F[l, lambda] ~ C_true[lambda] * (weighted 3j average).
# For l small, the 3j is peaked near l'=lambda, so F ~ C_true[lambda]/(4pi).
#
# But what's killing us is the UNIFORM shot-noise background in PLKjKk.
# For a uniform PLKjKk[lambda] = SN = const:
# <pseudo-Cl> = SN/(4pi) * SUM_lambda (2lambda+1) * F[l, lambda]
# = SN/(4pi) * (1/(4pi)) * SUM_lambda (2lambda+1) SUM_{l'} (2l'+1) C_true[l'] (3j)^2
# Exchange order: = SN/(4pi)^2 * SUM_{l'} (2l'+1) C_true[l'] * SUM_lambda(2lam+1)(3j)^2
# = SN/(4pi)^2 * SUM_{l'} (2l'+1) C_true[l'] * 1
# = SN/(4pi)^2 * 4pi * sigma^2 = SN * sigma^2 / (4pi)
# This is a CONSTANT for all l.

# So the shot-noise part of PLKjKk couples all power into every measured ell.
# And the TOTAL angular variance sigma^2 = SUM (2l'+1)/(4pi) C_true[l'] runs up to ell_Nyquist.

sigma_sq = np.sum((2*ells_high+1)/(4*np.pi) * C_true_high)
SN = float(N**2 * Nskew)
print(f"\nsigma^2 (angular) = {sigma_sq:.6e}")
print(f"SN = N^2*Nskew = {SN:.4e}")
print(f"Shot contribution = SN/(4pi)*sigma^2 = {SN/(4*np.pi)*sigma_sq:.6e}")
print()

# Compare with the measured mean pseudo-Cl
print(f"Mean measured pseudo-Cl (ell=1..499): {np.mean(cl_mean[1:]):.6e}")
print(f"Shot contribution / mean measured: {SN/(4*np.pi)*sigma_sq / np.mean(cl_mean[1:]):.4f}")
print()

# Hmm wait. Let me reconsider. The pseudo-Cl has a different normalization.
# In the DirectSHT code, the pseudo-Cl is computed as:
# alm = SUM_j delta_j * Y*_lm(nhat_j)
# Cl = (1/(2l+1)) SUM_m |alm|^2
# And then it's divided by the window: Cl_normalized = Cl / wl
# OR: the wl is used in the coupling matrix.
# Let me check what cl_k_all actually stores.

# Let me look at the SHT code to understand the normalization
import SHT_lya
help_text = SHT_lya.__doc__ if SHT_lya.__doc__ else "no docstring"
print(f"SHT_lya module: {help_text[:200]}")
print(f"dir(SHT_lya): {[x for x in dir(SHT_lya) if not x.startswith('_')]}")
