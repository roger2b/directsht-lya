#!/usr/bin/env python
"""
Carefully re-derive the MASTER equation normalization.

Key facts (all verified numerically):
  1. SHT weight per sightline: w_j (the LOS-integrated density)
  2. wl uses weights N (not w_j), so wl = N^2 * |SUM_j Y_lm|^2/(2l+1)
     Verified: wl[0] = N^2 * Nskew^2/(4pi)
  3. W_floor = N^2 * Nskew/(4pi) at high lambda
  4. P_2D(k) = b1^2 * N^2 * P_lin(k) / L
  5. <w^2> = b1^2 * N^2 / L^3 * SUM P_lin(k_perp)

The MASTER equation is:
  <Cl_pseudo> = M @ C_true

where:
  Cl_pseudo = SUM_m |alm|^2 / (2l+1)
  alm = SUM_j w_j Y_lm^*(nj)
  M_lL = (2L+1)/(4pi) SUM_lambda (2lambda+1) wl[lambda] ThreeJ^2(l,L,lambda)

Since wl = N^2 * wl_mask (where wl_mask uses unit weights), we have:
  M = N^2 * M_mask

Now, <Cl_pseudo> = SUM_{jk} <wj wk> P_l(nj.nk) / (4pi)

Split into diagonal (j=k) and off-diagonal (j≠k):
  <Cl_pseudo> = (1/(4pi)) * [Nskew * <w^2> + SUM_{j≠k} <wj wk> P_l(cos gamma_jk)]

  diagonal = Nskew * <w^2> / (4pi)   [ell-independent]
  off-diag = SUM_{j≠k} <wj wk> P_l(cos gamma_jk) / (4pi)

For the MASTER relation C_true, we need:
  <Cl_pseudo> = M @ C_true = N^2 M_mask @ C_true

The off-diagonal part involves the correlation function:
  <wj wk> = xi_2D(|rj_perp - rk_perp|)  [for sightlines at same chi_bar]
  
where xi_2D(r) = FT^{-1}[P_2D](r)

On the full sky, for a field with angular Cl = C_l:
  <Cl_pseudo> = M_mask @ C_l  (with unit-weight mask)

Wait, that's for unit weights in both SHT and wl. In our case:
- SHT uses weights w_j (signal)
- wl uses weights N (uniform)

So the MASTER equation becomes:
  <Cl_pseudo> = SUM_L C_true_L * (2L+1)/(4pi) * SUM_lambda (2lambda+1) * 
                wl_mask[lambda] * ThreeJ^2
where wl_mask = |SUM Y_lm(nj)|^2 / (2l+1) (unit weights).

But wl_code = N^2 * wl_mask, so in terms of code quantities:
  <Cl_pseudo> = SUM_L C_true_L * (2L+1)/(4pi) * SUM_lambda (2lambda+1) * 
                wl_code[lambda]/N^2 * ThreeJ^2
              = (1/N^2) * M_code @ C_true

So: C_true = N^2 * M_code^{-1} @ <Cl_pseudo>

Wait, this is DIFFERENT from what I had before. Previously I had:
  <Cl_pseudo> = M_code @ C_true, so C_true = M_code^{-1} @ <Cl_pseudo>

If instead: <Cl_pseudo> = (1/N^2) * M_code @ C_true, then:
  C_true = N^2 * M_code^{-1} @ <Cl_pseudo>

This would explain the missing factor! The deconvolved C_true I computed was 
M_code^{-1} @ Cl_data, but it should be N^2 * M_code^{-1} @ Cl_data.

Hmm, but... I need to check more carefully. Let me verify with the floor.

The diagonal part: Nskew * <w^2> / (4pi) [independent of normalization convention]
This should equal M_floor @ C_true = (1/N^2) * (W_floor_code/(4pi)) * SUM (2L+1) C_true_L

If C_true_L = P_2D(L/chi) / chi^2 = b1^2 N^2 P_lin / (L chi^2):
  (1/N^2) * W_floor_code/(4pi) * SUM (2L+1) b1^2 N^2 P / (L chi^2)
  = (N^2 Nskew/(4pi)) / (4pi N^2) * SUM (2L+1) b1^2 N^2 P / (L chi^2)
  = Nskew / (4pi)^2 * b1^2 N^2 / chi^2 * SUM (2L+1) P(L/chi) / L

Meanwhile: Nskew * <w^2> / (4pi) = Nskew * b1^2 * N^2 / (4pi L^3) * SUM P(kperp)

For these to match:
  1/(4pi)^2 * N^2/chi^2 * SUM (2L+1)P/L = N^2/(4pi L^3) SUM_k P(k)

  SUM (2L+1)P(L/chi)/(L chi^2) = (4pi/L^3) SUM_k P(k)

Converting L sum to k integral: L = k*chi, dL = chi dk
  SUM (2L+1)P ~ integral 2k*chi P(k) chi dk / chi^2 = integral 2k P(k) dk
  And SUM_k P(k) ~ (L/(2pi))^2 integral d^2k P(k) = (L^2/(4pi^2)) * 2pi integral k dk P(k)
                  = (L^2/(2pi)) integral k dk P(k)

  So LHS: 2 integral k P(k) dk / chi^2 ??? No, too sloppy.

Let me just check NUMERICALLY instead.
"""
import sys, os, gc
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF

d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_mean = np.mean(d['cl_k'], axis=0)
N = int(d['Nk'])
L = float(d['L'])
Nl = 500
Nskew = int(d['Nskew'])

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

# Compute <w^2> from discrete modes
kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2)
K_flat = K_perp.ravel()
Pk_flat = plin(np.where(K_flat > 0, K_flat, 1e-10))
Pk_flat[K_flat == 0] = 0
w2 = b1**2 * N**2 / L**3 * np.sum(Pk_flat)

# Diagonal of pseudo-Cl
diag_cl = Nskew * w2 / (4*np.pi)

# Now compute floor theory with DIFFERENT C_true normalizations
W_floor = N**2 * Nskew / (4*np.pi)

# C_true(1) = b1^2 N^2 P(l/chi) / (L chi^2)  [includes N^2 from P_2D]
# sigma^2(1) = (1/(4pi)) SUM (2L+1) * b1^2 N^2 P(L/chi) / (L chi^2)
ell_Ny = int(np.pi * N / L * chi_bar)
ells = np.arange(1, ell_Ny, dtype=float)
pvals = plin((ells + 0.5)/chi_bar)
sigma2_1 = b1**2 * N**2 / (L * chi_bar**2 * 4*np.pi) * np.sum((2*ells+1) * pvals)

# C_true(2) = b1^2 P(l/chi) / (L chi^2)  [NO N^2]
sigma2_2 = b1**2 / (L * chi_bar**2 * 4*np.pi) * np.sum((2*ells+1) * pvals)

# Floor theories:
# If <Cl_pseudo> = M_code @ C_true: floor_theory = W_floor * sigma^2
floor_1 = W_floor * sigma2_1  # C_true includes N^2
floor_2 = W_floor * sigma2_2  # C_true without N^2

# If <Cl_pseudo> = (1/N^2) * M_code @ C_true: floor_theory = W_floor * sigma^2 / N^2
floor_1b = W_floor * sigma2_1 / N**2
floor_2b = W_floor * sigma2_2 / N**2

print(f"{'='*70}")
print(f"Floor diagnostics")
print(f"{'='*70}")
print(f"diag_cl (data) = {diag_cl:.6e}")
print()
print(f"--- If <Cl> = M_code @ C_true ---")
print(f"  C_true = b1^2 N^2 P/(L chi^2): floor = {floor_1:.6e}  ratio = {floor_1/diag_cl:.4f}")
print(f"  C_true = b1^2 P/(L chi^2):     floor = {floor_2:.6e}  ratio = {floor_2/diag_cl:.4f}")
print()
print(f"--- If <Cl> = (1/N^2) * M_code @ C_true ---")
print(f"  C_true = b1^2 N^2 P/(L chi^2): floor = {floor_1b:.6e}  ratio = {floor_1b/diag_cl:.4f}")
print(f"  C_true = b1^2 P/(L chi^2):     floor = {floor_2b:.6e}  ratio = {floor_2b/diag_cl:.4f}")

print()
print(f"\n{'='*70}")
print(f"Alternatively: what sigma^2 is needed to match diag_cl?")
print(f"{'='*70}")
# diag_cl = W_floor * sigma^2_needed  => sigma^2_needed = diag_cl / W_floor
sigma2_needed = diag_cl / W_floor
print(f"sigma^2_needed = {sigma2_needed:.6e}")
print(f"sigma2_1 (N^2) = {sigma2_1:.6e}, ratio = {sigma2_1/sigma2_needed:.4f}")
print(f"sigma2_2 (no N^2) = {sigma2_2:.6e}, ratio = {sigma2_2/sigma2_needed:.4f}")

# Or: diag_cl = W_floor * sigma^2_needed / N^2  => sigma^2_needed = N^2 * diag_cl / W_floor
sigma2_needed_b = N**2 * diag_cl / W_floor
print(f"\n  If (1/N^2) model: sigma^2_needed = {sigma2_needed_b:.6e}")
print(f"  sigma2_1 (N^2) ratio = {sigma2_1/sigma2_needed_b:.4f}")
print(f"  sigma2_2 (no N^2) ratio = {sigma2_2/sigma2_needed_b:.4f}")

# Actually, I already established that diag_cl = W_floor * <w^2>/Nskew  ??? No.
# diag_cl = Nskew * <w^2> / (4pi)
# W_floor = N^2 * Nskew / (4pi)
# So diag_cl / W_floor = <w^2> / N^2
print(f"\n{'='*70}")
print(f"Simple check:")
print(f"  diag_cl / W_floor = {diag_cl/W_floor:.6e}")
print(f"  <w^2> / N^2 = {w2 / N**2:.6e}")
print(f"  Ratio: {(diag_cl/W_floor) / (w2/N**2):.6f}")
print(f"{'='*70}")

# So the floor contribution to the MASTER sum is:
# floor_MASTER = W_floor * sigma^2
# This should equal diag_cl. So sigma^2 = diag_cl/W_floor = <w^2>/N^2.
#
# If sigma^2 = (1/(4pi)) SUM (2L+1) C_true, and sigma^2 = <w^2>/N^2:
# (1/(4pi)) SUM (2L+1) C_true = <w^2>/N^2 = b1^2/L^3 SUM_k P(k)
#
# So (1/(4pi)) SUM (2L+1) C_true = b1^2/L^3 SUM_kperp P(k_perp)
#
# If C_true = b1^2 P(L/chi)/(L chi^2):
# LHS = b1^2/(4pi L chi^2) SUM (2L+1) P(L/chi)
# Converting: L -> k*chi, sum over L ~ integral chi dk, (2L+1) ~ 2k*chi
# ~ b1^2/(4pi L chi^2) * integral 2k*chi^2 dk P(k) / (chi spacing)
# Hmm, it's a discrete sum. Let me just compute numerically.

LHS = sigma2_2  # = b1^2/(4pi L chi^2) SUM (2L+1)P(L/chi) [without N^2] 
RHS = w2 / N**2  # = b1^2/L^3 SUM P(kperp)

print(f"\n{'='*70}")
print(f"Checking C_true = b1^2 P/(L chi^2) gives correct variance")
print(f"  LHS = (1/(4pi)) SUM (2L+1) C_true = {LHS:.6e}")
print(f"  RHS = <w^2>/N^2 = {RHS:.6e}")
print(f"  Ratio LHS/RHS = {LHS/RHS:.6f}")
print(f"{'='*70}")

# Now separately, let me compute the angular sum and the k-sum numerically
# to see if they match.
sum_angular = np.sum((2*ells+1) * pvals)
sum_kmode = np.sum(Pk_flat)

print(f"\n  Angular: SUM (2L+1) P(L/chi) = {sum_angular:.6e}  (L=1 to {ell_Ny})")
print(f"  k-mode:  SUM P(kperp) = {sum_kmode:.6e}  (N^2 = {N**2} modes)")
print(f"  Ratio angular/k = {sum_angular/sum_kmode:.6f}")
print(f"  Expected ratio (4pi L chi^2 / (N^2 L^3)) = {4*np.pi*L*chi_bar**2 / (N**2*L**3):.6f}")
print(f"  So angular = k * (4pi chi^2/L^2) * (L^2/N^2) ??? ")
# Actually: LHS = b1^2/(4pi L chi^2) * sum_angular
#            RHS = b1^2 * N^2 / (N^2 * L^3) * sum_kmode = b1^2/L^3 * sum_kmode
# LHS = RHS => sum_angular / (4pi L chi^2) = sum_kmode / L^3
# => sum_angular = 4pi chi^2 sum_kmode / L^2
sum_kmode_check = sum_angular * L**2 / (4*np.pi * chi_bar**2)
print(f"\n  sum_kmode expected  = sum_angular * L^2/(4pi chi^2) = {sum_kmode_check:.6e}")
print(f"  sum_kmode actual   = {sum_kmode:.6e}")
print(f"  Ratio = {sum_angular * L**2 / (4*np.pi * chi_bar**2 * sum_kmode):.6f}")
