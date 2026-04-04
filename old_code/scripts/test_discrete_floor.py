#!/usr/bin/env python
"""
Compute the correct floor contribution using the ACTUAL discrete k-modes
instead of the smooth Limber approximation.

The diagonal Cl: (1/(4pi)) * SUM_j <w_j^2> = Nskew * <w^2> / (4pi)
where <w^2> = (b1^2 N^2/L^3) * SUM_{kx,ky} P_lin(|K_perp|)

The floor contribution in MASTER:
  (M_floor @ C_true)_l = (W_floor/(4pi)) SUM_L (2L+1) C_true[L]

These should match: the floor captures the diagonal (shot noise-like) term.
But the smooth Limber sigma^2 overcounts because it replaces the discrete SUM
with a continuous integral.

Solution: compute the floor contribution from the ACTUAL discrete variance.
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
Nskew = int(d['Nskew'])

PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

W_floor = N**2 * Nskew / (4*np.pi)

# ---- Compute <w^2> from discrete modes ----
kvals = np.fft.fftfreq(N, d=L/N) * (2*np.pi)  # physical k in h/Mpc
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2)
K_flat = K_perp.ravel()
# DC mode: k=0
mask_dc = K_flat > 0
Pk_flat = np.zeros_like(K_flat)
Pk_flat[mask_dc] = plin(K_flat[mask_dc])
# <w^2> = (b1^2 N^2 / L^3) * SUM P_lin(K_perp)
w2_discrete = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
print(f"<w^2>_discrete = {w2_discrete:.6e}")

# The diagonal Cl (ell-independent):
diag_cl = Nskew * w2_discrete / (4*np.pi)
print(f"Diagonal Cl = Nskew * <w^2> / (4*pi) = {diag_cl:.4e}")

# ---- The theory floor term should equal the diagonal ----
# theory_floor = W_floor * sigma^2(C_true)
# For C_true = b1^2 P / (X chi^2):
# theory_floor = W_floor * b1^2/(4pi X chi^2) * SUM_L (2L+1) P(L/chi)
# 
# If theory_floor should equal diag_cl:
# W_floor * sigma^2 = Nskew * <w^2> / (4pi)
# 
# Actually, in the MASTER framework, the floor contribution captures
# BOTH diagonal AND the "isotropic" part of the off-diagonal.
# So theory_floor ≠ diag_cl in general.
#
# Instead, the approach should be:
# Use C_true = b1^2 P / (L chi^2) [my derivation]
# Compute theory_clust + theory_floor
# And see if it matches data.
#
# But we already showed X_fit ≈ 1265, not L ≈ 1383.
# The question is: is the 8.5% due to the smooth Limber sigma^2 approximation?

# Let me instead compute sigma^2 using discrete modes:
# sigma^2 = <w^2>_on_sphere / N^2
# But this is circular because we're trying to derive C_true...

# Better approach: for C_true = b1^2 P / (L chi^2), and using the actual X_fit:
# X_fit = L would mean sigma^2_smooth is exactly right.
# X_fit = 1265 means sigma^2_smooth overestimates.

# The smooth sigma^2:
ell_Ny = int(np.pi * N / L * chi_bar)
ells_s = np.arange(ell_Ny, dtype=float)
plin_s = plin((ells_s + 0.5)/chi_bar)
sigma2_smooth = b1**2 / chi_bar**2 * np.sum((2*ells_s+1) * plin_s) / (4*np.pi)
print(f"\nsigma^2_smooth (from Limber sum to ell_Ny) = {sigma2_smooth:.6e}")

# The discrete sigma^2 (from actual k-modes):
# C_l^{ww} = P_2D(l/chi) / chi^2, but the actual P_2D is a discrete sum
# <|w_j|^2> = variance of the field w_j
# Total: Nskew * SUM_l (2l+1)/(4pi) C_l^{ww} = Nskew * <w^2>... 
# This doesn't work because of the mask.

# Let me try a COMPLETELY different approach to get the correct floor term.
# Instead of Limber sigma^2, use the MEASURED variance of w_j.

# From 100 sims, the variance is encoded in cl_mean.
# SUM_l (2l+1) cl_mean[l] / (4pi) = total angular power = "sky variance"
# But cl_mean only goes to l=500, missing high-l power.

# Actually, the simplest test: compute the FULL theory prediction
# using the pair-counting formula directly, without the MASTER framework.

# The pair counting formula:
# <Cl> = (1/(4pi)) SUM_{j,k} <w_j w_k> P_l(cos gamma)
# 
# where <w_j w_k> = xi_w(gamma_jk) is the angular correlation of the w field.
#
# For each pair (j,k), <w_j w_k> = (b1^2 N^2/L^3) SUM_{kperp} P(K) exp(iK.Delta_r)
# where Delta_r is the 3D separation vector (projected transversely).
#
# This is a SUM over N^2 k-modes for EACH pair — O(Nskew^2 * N^2) total.
# Too expensive for all pairs, but we can use the PLKjKk shortcut.

# Key insight: <w_j w_k> depends only on the angular separation gamma_jk.
# In Legendre space: <w_j w_k> = SUM_L (2L+1)/(4pi) xi_L P_L(cos gamma)
# where xi_L = angular correlation multipole.
#
# The pair counting then gives:
# <Cl> = SUM_L (2L+1)/(4pi) xi_L * [SUM_{jk} P_L(cos gamma) P_l(cos gamma)] / (4pi)
#       = SUM_L xi_L * M[l,L]  (standard MASTER)
# where M uses the mask window.
#
# So C_true = xi_L / N^2 (because M_code = N^2 * M_mask).
#
# Now xi_L needs to be computed from the actual discrete k-modes:
# xi_L = integral / sum that maps P_2D(k) to angular multipoles

# The FLAT-SKY Limber gives xi_L = P_2D(L/chi)/chi^2 = b1^2 N^2 P/(L chi^2)
# But this is approximate.
#
# The FULL-SKY exact formula (for point at distance chi):
# xi_L = (2/pi) integral k^2 dk P_2D(k) j_L(k*chi)^2
# where j_L is the spherical Bessel function.
# But this is for 3D -> angular projection, which involves integration over LOS.
# For our kz=0 case (2D field), the projection is simpler.

# For a 2D field on a sphere at radius chi:
# f(n_hat) = integral d^2k/(2pi)^2 P_2D(k) exp(i k . chi*n_hat_perp)
# In the flat-sky limit, C_l = P_2D(l/chi)/chi^2.
# The correction to this is O((l/chi)^2 / chi^2) or so.

# Actually, the flat-sky -> full-sky correction should be negligible at chi=5691
# and the box subtends only ~14 degrees.

# So the 8.5% discrepancy must come from something else.
# Let me check: is the smooth Limber integral the right way to compute sigma^2?

# sigma^2_smooth = b1^2/(4pi chi^2) SUM_l (2l+1) P(l/chi) 
# This converts to: = b1^2/(4pi chi^2) * integral 2l dl P(l/chi) 
#                   = b1^2/(4pi) * integral 2k dk P(k) 
#
# And <w^2> = b1^2 N^2/L^3 SUM_k P(K) 
# Converting: SUM_k = (L/(2pi))^2 integral d^2K = L^2/(2pi) integral K dK
# <w^2> = b1^2 N^2/L^3 * L^2/(2pi) integral K dK P(K) 
#        = b1^2 N^2/(2pi L) integral K dK P(K)
#
# So: sigma^2_smooth = b1^2/(4pi) integral 2K dK P(K) = b1^2/(2pi) integral K dK P(K)
# And: <w^2> = b1^2 N^2/(2pi L) integral K dK P(K)
# 
# Ratio: sigma^2_smooth = <w^2> * L/N^2

sigma2_from_w2 = w2_discrete * L / N**2
print(f"sigma^2 from <w^2> * L / N^2 = {sigma2_from_w2:.6e}")
print(f"sigma^2_smooth = {sigma2_smooth:.6e}")
print(f"Ratio sigma2_smooth/sigma2_from_w2 = {sigma2_smooth/sigma2_from_w2:.6f}")

# If the smooth integral equals the discrete sum: ratio should be 1
# If not, the difference is the discretization error.

# The correct floor term (from discrete modes):
# theory_floor_correct = W_floor * <w^2> * L / N^2
# But wait, this doesn't depend on X -- it's derived from the actual <w^2>.
# The floor captures: (W_floor/(4pi)) SUM (2L+1) C_true = W_floor * sigma^2
# If C_true = b1^2 P/(L chi^2): sigma^2 = b1^2/(4pi L chi^2) SUM (2L+1)P(l/chi)
# = (from continuous integral) = b1^2/(2pi L) integral K dK P(K)  [divided by chi^2, 
#   converted L to K]
# Hmm wait, I need to be careful.
# sigma^2(X=L) = S_sigma / L where S_sigma = b1^2/(chi^2) * SUM (2L+1)P / (4pi)
# S_sigma using continuous approx: = b1^2 * integral 2K dK P(K) / (4pi) = b1^2/(2pi) integral K dK P
# S_sigma from discrete: = <w^2> * N^2/L? Let me check:
# S_sigma = b1^2/chi^2 * SUM_L (2L+1) P(L/chi) / (4pi)
# Continuous: SUM_L (2L+1) P(L/chi) ~ integral 2 k chi^2 dk P(k) = 2 chi^2 integral k dk P
# S_sigma = b1^2/chi^2 * 2 chi^2 integral k dk P / (4pi) = b1^2 integral k dk P / (2pi)
# <w^2> = b1^2 N^2/(2pi L) integral k dk P
# So S_sigma = <w^2> * L / N^2
# And sigma^2(X=L) = S_sigma / L = <w^2> / N^2 = w2_discrete / N^2

sigma2_discrete = w2_discrete / N**2
print(f"\nsigma^2(X=L, discrete) = <w^2> / N^2 = {sigma2_discrete:.6e}")
print(f"sigma^2(X=L, smooth) = {sigma2_smooth/L:.6e}")
print(f"Ratio smooth/discrete = {(sigma2_smooth/L) / sigma2_discrete:.6f}")

# So the smooth sigma^2 / discrete sigma^2 = sigma2_smooth / sigma2_from_w2 * 1/L
# Let me just compute the floor terms with both:
floor_discrete = W_floor * sigma2_discrete
floor_smooth = W_floor * sigma2_smooth / L  # sigma^2(X=L, smooth)
print(f"\nfloor_discrete = {floor_discrete:.4e}")
print(f"floor_smooth = {floor_smooth:.4e}")
print(f"diag_cl = {diag_cl:.4e}")
print(f"floor_discrete / diag_cl = {floor_discrete/diag_cl:.4f}")
print(f"floor_smooth / diag_cl = {floor_smooth/diag_cl:.4f}")

# Now redo the convergence test with discrete sigma^2:
mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False
data_mean = np.mean(cl_mean[mask])

print(f"\n=== Convergence with discrete floor ===")
print(f"{'Nl_large':>8s} {'ratio(L,discr)':>14s} {'ratio(L,smooth)':>14s}")
print("-" * 45)

for Nl_large in [500, 1000, 1500, 2000, 2500, 3000, 3500]:
    ells_ext = np.arange(Nl_large, dtype=float)
    plin_vals = plin((ells_ext + 0.5)/chi_bar)
    cl_L = b1**2 * plin_vals / (L * chi_bar**2)
    
    wl_needed = 2*Nl_large - 1
    wl_raw = np.zeros(wl_needed)
    na = min(wl_needed, len(wl_ext))
    wl_raw[:na] = wl_ext[:na]
    wl_raw[na:] = W_floor
    wl_clust = wl_raw - W_floor
    
    cp = Wigner3j.CoupleMat(Nl_large, wl_clust)
    Mc = cp.compute_matrix()
    tc = (Mc @ cl_L)[:Nl]
    
    # With discrete floor:
    th_discr = tc + floor_discrete
    r_discr = np.mean(cl_mean[mask]) / np.mean(th_discr[mask])
    
    # With smooth floor:
    th_smooth = tc + floor_smooth
    r_smooth = np.mean(cl_mean[mask]) / np.mean(th_smooth[mask])
    
    print(f"  {Nl_large:5d}  {r_discr:14.4f} {r_smooth:14.4f}")
    del cp, Mc; gc.collect()
