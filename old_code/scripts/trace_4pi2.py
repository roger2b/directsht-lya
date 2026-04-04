#!/usr/bin/env python
"""
Definitive test: compare the ACTUAL measured pseudo-Cl with M @ C_true.

The notes claim: C_plot ≡ <Cl> / (4π)² = Σ M_ℓL C_true
where C_true = P_F / (32π³ χ²).

But the CODE computes: ratio = cl_mean / (M @ C_true)
If the notes are right, this ratio should be (4π)².

If the code is right (ratio ≈ 1), then the notes' (4π)² is already absorbed.

Let me test with the cached data.
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
wl_ref = wl_k[0, :Nl]

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

# C_true from notes
ells = np.arange(Nl, dtype=float)
cl_true_notes = b1**2 * plin((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)

# MASTER M with wl truncated to Nl=500
wl_couple = np.zeros(2*Nl - 1)
wl_couple[:Nl] = wl_ref
couple = Wigner3j.CoupleMat(Nl, wl_couple)
M = couple.compute_matrix()

theory_master = M @ cl_true_notes

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

ratio_no_fac = np.mean(cl_mean[mask]) / np.mean(theory_master[mask])
ratio_with_16pi2 = np.mean(cl_mean[mask]) / (16 * np.pi**2 * np.mean(theory_master[mask]))

print(f"cl_mean[50] = {cl_mean[50]:.4e}")
print(f"(M @ C_true_notes)[50] = {theory_master[50]:.4e}")
print(f"16π² × (M @ C_true_notes)[50] = {16*np.pi**2 * theory_master[50]:.4e}")
print()
print(f"Ratio cl_mean / (M @ C_true) [no (4π)²]  = {ratio_no_fac:.4f}")
print(f"Ratio cl_mean / (16π² × M @ C_true)       = {ratio_with_16pi2:.4f}")
print(f"(4π)² = {(4*np.pi)**2:.4f}")
print(f"16π² = {16*np.pi**2:.4f}")

# Also check: what's wl[0] / (N^2 * Nskew)?
Nskew = 9600
print(f"\nwl[0] = {wl_ref[0]:.4e}")
print(f"N^2 * Nskew ^2 = {N**2 * Nskew**2:.4e}")
print(f"wl[0] / (N^2 * Nskew^2) = {wl_ref[0] / (N**2 * Nskew**2):.4f}")
print(f"wl[0] / (N^2 * Nskew) = {wl_ref[0] / (N**2 * Nskew):.4f}")
print(f"wl[0] / Nskew^2 = {wl_ref[0] / Nskew**2:.4f}")

# The window: wl = |SUM_j N*Y_lm|^2 / (2l+1)
# For l=0: Y_00 = 1/sqrt(4pi), so:
# w_0 = |N * Nskew * 1/sqrt(4pi)|^2 = N^2 * Nskew^2 / (4pi)
print(f"N^2 * Nskew^2 / (4π) = {N**2 * Nskew**2 / (4*np.pi):.4e}")
print(f"wl[0] / [N^2 * Nskew^2 / (4π)] = {wl_ref[0] / (N**2 * Nskew**2 / (4*np.pi)):.4f}")

# Good: wl[0] ≈ N^2 * Nskew^2 / (4pi)

# M[l,l'] = (2l'+1)/(4pi) SUM (2lamb+1) wl[lamb] 3j^2
# M[0,0] = (1)/(4pi) * (1) * wl[0] * 1^2 = wl[0]/(4pi)
print(f"\nM[0,0] = {M[0,0]:.4e}")
print(f"wl[0]/(4π) = {wl_ref[0]/(4*np.pi):.4e}")

# So M[0,0] = wl[0]/(4pi) ≈ N^2 * Nskew^2 / (4pi)^2
# And theory[0] = M[0,0]*C_true[0] + ... ≈ (N^2*Nskew^2/(4pi)^2) * b1^2*P/(32pi^3*chi^2)

# Hmm, the (4pi)^2 in the denominator of M[0,0] partly explains things.
# M already has 1/(4pi) from the standard definition.
# Then the notes' C_true also has 1/(4pi)^2 built in.
# So the total is 1/(4pi) × 1/(4pi)^2 = 1/(4pi)^3 from these factors alone?
# No, that's not right either.

# Let me trace through more carefully.
# Standard MASTER: <pseudo_Cl> = SUM_L M[l,L] C_true[L]
# with M = (2L+1)/(4pi) SUM (2lamb+1) Wl 3j^2
# and C_true is the TRUE angular Cl of the signal field.

# For us: signal field s(nhat) = w(r_perp(nhat)) = SUM_z delta(r_perp, z)
# at sightline positions: alm = SUM_j s(n_j) Y_lm^*(n_j)
# = SUM_j w_j Y_lm^*(n_j) [if all sightlines have equal unit weight]

# But our window is: u_lm = SUM_j K_j Y_lm(n_j) = SUM_j N Y_lm(n_j) = N * SUM Y_lm
# And our data: a_lm = SUM_j w_j Y_lm^*(n_j) where w_j is the field value.

# Wait, what's the actual field? w_j = SUM_alpha delta_F(j, alpha).
# And K_j = N (the DFT mask at k=0).
# The data alm is: a_lm = SUM_j w_j Y_lm^*(n_j)
# = SUM_j [K_j * delta_F_LOS_avg(j)] * Y_lm^*(n_j)... no.

# Actually from the code:
# data: wdata[:,0] = FT_delta[:,0] = SUM_alpha delta_F(j,alpha) * 1 = SUM delta_F
# mask: wrand[:,0] = FT_mask[:,0] = SUM_alpha 1 = N

# In the OLD code: hdat = sht(theta, phi, wdata[:,0]) and hran = sht(theta, phi, wrand[:,0])
# hdif = hp.alm2cl(hdat - hran)
# wl = hp.alm2cl(hran)

# In the NEW 100-sim code: w_j = FT_delta[:,0] = SUM delta_F (since delta_F = w_gal - 1)
# alm_data = sht(theta, phi, w_j)
# cl = |alm_data|^2 / (2l+1) = _alm2cl_complex(alm_data)

# So the DATA alm does NOT include the N factor for the mask.
# The MASK alm DOES include N: hran = sht(theta, phi, N*ones) = N * SUM Y_lm = N*u_lm_unit

# And the OLD code subtracted: hdat - hran.
# hdat = SUM_j (SUM_alpha (delta_F+1)) Y_lm^* = SUM_j (SUM delta_F + N) Y_lm^*
#      = SUM w_j Y_lm^* + N * SUM Y_lm^*
# hran = SUM N Y_lm^* = N SUM Y_lm^*
# hdat - hran = SUM w_j Y_lm^* = data alm

# So both old and new give the same thing: a_lm = SUM_j w_j Y_lm^*(n_j)
# where w_j = SUM_alpha delta_F(j,alpha).

# And the window: wl = |SUM_j N Y_lm(n_j)|^2 / (2l+1) = N^2 |SUM Y_lm|^2 / (2l+1)

# NOW: in the STANDARD MASTER derivation (Hivon+ 2002):
# a_lm^{pseudo} = SUM_j w(n_j) W(n_j) Y_lm^*(n_j)   [w = signal, W = window]
# In our case: w(n_j) = s_j (the signal) and W(n_j) comes from the discrete sampling.
# The effective window is: W(nhat) = SUM_j delta_D(nhat - nhat_j)
# so a_lm = SUM_j s_j Y_lm^*(n_j) = integral s(nhat) W(nhat) Y_lm^*(nhat) dOmega

# The window alm: W_lm = integral W(nhat) Y_lm(nhat) dOmega = SUM_j Y_lm(n_j)
# And the window Cl: Wl = |W_lm|^2 / (2l+1) = |SUM_j Y_lm(n_j)|^2 / (2l+1)

# But the code has hran with an N factor: u_lm = N * SUM Y_lm.
# So the CODE's wl = N^2 * |SUM Y_lm|^2 / (2l+1) = N^2 * Wl_standard.

# Therefore the STANDARD M matrix uses Wl_standard = wl_code / N^2.
# M_standard = (2L+1)/(4pi) SUM (2lamb+1) Wl_standard 3j^2
# = (1/N^2) × (2L+1)/(4pi) SUM (2lamb+1) wl_code 3j^2
# = M_code / N^2

# And the STANDARD MASTER: <pseudo_Cl> = SUM M_standard × C_true_standard
# = (1/N^2) × SUM M_code × C_true_standard
# So: <pseudo_Cl> = M_code @ C_true_standard / N^2

# But the plot_money code computes: theory = M_code @ cl_true_notes
# Comparing: cl_mean = M_code @ cl_true_notes → cl_true_notes = C_true_standard / N^2?

# C_true_standard = the full-sky C_l of s(nhat) = SUM_z density(r_perp(nhat), z)
# C_true_notes = P_F / (32 pi^3 chi^2)
# cl_true_notes = C_true_standard / N^2

# So C_true_standard = N^2 × P_F / (32 pi^3 chi^2) = N^2 b1^2 P_lin / (32 pi^3 chi^2)

# From my Limber derivation: C_true_standard = b1^2 N^2 P_lin / (L chi^2)
# Ratio: (32 pi^3) / L = 32 pi^3 / 1382.7 = 225.1
# Hmm, that's not 1.

# Wait, let me recalculate. C_true in the CODE is compared DIRECTLY with cl_mean:
# theory = M_code @ cl_true_notes, and cl_mean ≈ theory.
# So <Cl> = M_code @ cl_true_notes.
# The standard MASTER: <Cl> = M_standard @ C_true_standard = (M_code/N^2) @ C_true_standard
# Therefore: M_code @ cl_true_notes = M_code @ (C_true_standard/N^2)
# → cl_true_notes = C_true_standard / N^2

# From Limber: C_true_standard = b1^2 N^2 P_lin / (L chi^2)
# → cl_true_notes = b1^2 P_lin / (L chi^2)

# But the notes say: cl_true_notes = b1^2 P_lin / (32 pi^3 chi^2)

# So the claim is: L = 32 pi^3 ≈ 993?
# But L = 1382.7 Mpc/h!

# Ratio: 32*pi^3 / L:
print(f"\n32 pi^3 = {32*np.pi**3:.1f}")
print(f"L = {L:.1f}")
print(f"32 pi^3 / L = {32*np.pi**3 / L:.4f}")

# 32*pi^3 = 993 vs L = 1382.7 → ratio = 0.719
# So the notes claim cl_true ∝ 1/L effectively, but the coefficient differs by ~0.72.
# This factor of 0.72 would explain the 1/0.72 ≈ 1.39 overcounting... wait,
# actually it's wrong direction. If L > 32pi^3, then cl_true_notes > cl_true_my,
# meaning theory OVERSHOOTS → consistent with what we see!

# theory_notes = b1^2 P / (32pi^3 chi^2) = (b1^2 P / (L chi^2)) × L/(32pi^3)
# theory_mine = b1^2 P / (L chi^2)
# ratio = notes/mine = L/(32pi^3) = 1382.7/993 = 1.39

# So notes theory is 39% HIGHER than my derivation.
# And data/theory_notes approaches ~0.85 → data/theory_mine = 0.85 × 1.39 = 1.18
# Hmm, that doesn't work either. Or:
# notes theory = (L/32pi^3) × mine_theory  [notes is LARGER]
# If data = mine_theory: then data/notes = 32pi^3/L = 0.72.
# But we see data/notes ≈ 0.85 (at large Nl), not 0.72.

# So neither formula is perfectly right. But let me verify numerically:
# what alpha gives ratio=1 when fully converged?

# From fit_alpha_extended.py: at Nl_large=3500, alpha = 8.54e-4
# Notes: alpha = 1/(32pi^3) = 1.008e-3
# My: alpha = 1/L = 7.24e-4

print(f"\nalpha_notes = {1/(32*np.pi**3):.6e}")
print(f"alpha_mine = {1/L:.6e}")
print(f"alpha_fit@3500 = 8.54e-4")
print(f"alpha_fit@3500 / alpha_notes = {8.54e-4 / (1/(32*np.pi**3)):.4f}")
print(f"alpha_fit@3500 / alpha_mine = {8.54e-4 / (1/L):.4f}")

# alpha_fit ≈ 8.54e-4, between notes (1.01e-3) and mine (7.24e-4).
# This makes sense given the truncation — alpha_fit is still decreasing.
# If we could extend to ell_Ny ≈ 6600, alpha would converge to...?

# From f(3500) = 0.847, the extrapolation assuming hyperbolic decay:
# alpha → some limit between 0.72 and 0.85.

# Key insight: my derivation gives alpha = 1/L, and the notes give 1/(32pi^3).
# L ≈ 1382.7, 32pi^3 ≈ 993. The TRUTH should be between these.
# The discrepancy is that my flat-sky Limber derivation assumed
# CONTINUOUS k-modes, while the box has DISCRETE modes.
# The continuous Limber integral ∫ dk k P(k) J_0(kr) × J_0(lr/chi)
# should exactly equal the discrete sum.

# But: maybe the issue is that the 2D Limber projection
# C_l = P_2D(l/chi) / chi^2 has an extra angular factor.
# In the STANDARD Limber approximation:
# C_l = 1/chi^2 × P_2D(l/chi)
# where P_2D is the angular power spectrum coefficient:
# ξ(r) = ∫ dl l/(2pi) P_2D(l) J_0(lr)
# So: P_2D(l) = 2pi ∫ dr r ξ(r) J_0(lr)

# And P_2D_from_P3D: in the Limber approx,
# P_2D(l) = ∫ dchi/chi^2 × P_3D(l/chi) × W^2(chi)
# For our single thin shell at chi_bar with LOS window squared = L:
# P_2D(l) = (1/chi_bar^2) × P_3D(l/chi) × L
# = b1^2 P_lin(l/chi) L / chi^2

# Wait — that gives C_l = P_2D / chi^2 = b1^2 P_lin L / chi^4?
# That has chi^4, not chi^2. Let me reconsider.

# Standard Limber: C_l = ∫ (W(chi))^2/chi^2 P(l/chi) dchi
# Our W(chi) = 1 (constant weight along LOS of length L at chi_bar)
# Actually, the Limber integral for a discrete sum is:
# C_l = Σ_α Σ_β ξ(Δr_perp, Δz_αβ) P_l(cos γ) / (4pi) / ...
# This is getting circular.

# Let me just do the DIRECT NUMERICAL VERIFICATION.
# I have P_2D(k) = b1^2 * N^2 * P_lin(k) / L (verified to 0.1%).
# The angular Cl of the field s(nhat) = w(r_perp(nhat)):
# In flat-sky: C_l = P_2D(l/chi) / Omega_survey
# where Omega_survey is the solid angle of the survey.
# Wait, is that right? Let me think again.

# s(nhat) is defined on a small patch of solid angle Omega = L^2/chi^2.
# The full-sky Cl of this field:
# C_l = ∫ |a_lm|^2 / (2l+1)
# where a_lm = ∫ s(nhat) Y_lm^*(nhat) dOmega

# For a flat patch at distance chi:
# a_lm ≈ ∫_patch s(theta) Y_lm^*(theta) d^2theta [flat sky]
# ≈ (1/chi^2) ∫_box w(r) exp(-il.r/chi) d^2r  [change to physical coords]
# = (1/chi^2) w_FT(l/chi)

# where w_FT(k) = ∫ w(r) exp(-ik.r) d^2r is the 2D Fourier transform.
# P_2D(k) = <|w_FT(k)|^2> / A_box = <|w_FT|^2> / L^2

# So <|a_lm|^2> = (1/chi^4) <|w_FT(l/chi)|^2> = (1/chi^4) P_2D(l/chi) L^2

# And the m-sum: for flat-sky approximation, l has 2D components.
# The number of m-modes at a given l is ≈ 2l+1 (full sky) or
# ≈ (Omega × (2l+1)/(4pi)) (for a patch of solid angle Omega).
# This is encoded in the MASTER coupling matrix.

# So C_l^true_standard = <|a_lm|^2> / (2l+1) summed appropriately.
# Hmm, but this is exactly what M handles.

# I think the issue is that:
# C_l_true in the STANDARD MASTER framework = (1/chi^4) P_2D(l/chi) L^2
# Then: <pseudo_Cl> = M_standard @ C_true = (M_code/N^2) @ C_true
# And in the code: M_code @ cl_true_notes = (M_code/N^2) @ (N^2 cl_true_notes) = M_code @ cl_true_notes
# So: N^2 cl_true_notes = C_true_standard = P_2D L^2 / chi^4

# P_2D = b1^2 N^2 P_lin / L
# C_true_standard = b1^2 N^2 P_lin L^2 / (L chi^4) = b1^2 N^2 P_lin L / chi^4

# cl_true_notes = b1^2 P_lin L / (N^2 chi^4)... wait, this has chi^4, not chi^2.

# Hmm. I think the flat-sky approximation is more subtle.
# For the FULL spherical harmonics (not flat sky),
# a_lm = integral of s(nhat) Y_lm^*(nhat) dOmega  on the FULL sphere.
# Since s(nhat) = w(chi*sin(theta)*cos(phi), chi*sin(theta)*sin(phi)) 
# only on the small patch (theta < L/(2chi)),
# the integral is over just the patch.

# Actually, in real SHT:
# a_lm = SUM_j w_j Y_lm^*(n_j) [discrete sum, no dOmega factor]
# This is NOT the standard CMB integral definition!
# There's NO dOmega factor.

# In standard CMB:
# a_lm = ∫ f(nhat) Y_lm^*(nhat) dOmega  [with dOmega weight]
# pseudo_a_lm = ∫ W(nhat) f(nhat) Y_lm^*(nhat) dOmega
# W is typically pixels: W = 1 in observed region, 0 elsewhere.
# Wl = Cl of W(nhat).

# In our case: the discrete version is:
# a_lm = SUM_j w_j Y_lm^*(n_j)
# This is a QUADRATURE approximation of the integral... but WITHOUT the proper
# quadrature weights (dOmega). The DirectSHT might include weights.

# Let me check what DirectSHT does.
print("\n---- Checking DirectSHT weights ----")

del couple, M; gc.collect()
