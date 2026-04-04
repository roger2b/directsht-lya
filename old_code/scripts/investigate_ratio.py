#!/usr/bin/env python
"""
Investigate what's causing the ell-dependent ratio in the deconvolved C_true.

Possible explanations:
1. Limber argument: l vs l+0.5 vs l+1
2. Beyond-Limber corrections (finite box width)
3. Discrete k-mode structure (C_true is a comb, not smooth)
4. The P_2D relation has ell-dependent corrections

Let me check each.
"""
import sys, os, gc
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Load data + deconvolve the clustered part
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

# Compute diag_cl
kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2)
K_flat = K_perp.ravel()
Pk_flat = plin(np.where(K_flat > 0, K_flat, 1e-10))
Pk_flat[K_flat == 0] = 0
w2_theory = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2_theory / (4*np.pi)

# Subtract floor
cl_clust = cl_mean - diag_cl

# Build M_clust (Nl_large=2000 — already shown to be converged)
Nl_large = 2000
wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust_arr = wl_raw - W_floor

couple_c = Wigner3j.CoupleMat(Nl_large, wl_clust_arr)
M_full = couple_c.compute_matrix()
M_sq = M_full[:Nl, :Nl]
MD = MaskDeconvolution(Nl, wl_clust_arr[:2*Nl-1], precomputed_Wigner=M_sq)
NperBin = 32
bins = MD.binning_matrix('linear', 0, NperBin)
ells_arr = np.arange(Nl, dtype=float)
binned_ells = bins @ ells_arr

ells_dec, cl_dec = MD(cl_clust, bins)

print(f"  bin centers: {ells_dec}")

# Test 1: Different Limber arguments
print(f"\n{'='*80}")
print("Test 1: Different Limber arguments")
print(f"{'='*80}")
for label, ell_shift in [("l", 0.0), ("l+0.5", 0.5), ("l+1", 1.0)]:
    k_vals = (ells_dec + ell_shift) / chi_bar
    cl_th = b1**2 * plin(k_vals) / (L * chi_bar**2)
    r = cl_dec / cl_th
    print(f"\n  Limber arg = {label}:")
    for i in range(1, len(ells_dec)):
        print(f"    ell={ells_dec[i]:4.0f}  r={r[i]:.4f}")
    print(f"  Mean r = {np.mean(r[1:]):.4f}, std = {np.std(r[1:]):.4f}")

# Test 2: Compute C_true from discrete modes directly
# The "true" angular power spectrum from a periodic box is:
# C_l = (1/N^2) * SUM over N^2 sightlines |alm_true|^2 / (2l+1)
# where alm_true = SUM_j w_j Y_lm^*(nhat_j) 
# and w_j is the actual field value.
# 
# But we can compute C_true from P_2D using:
# C_l = P_2D(l/chi) / (chi^2 * N^2)   [Limber]
# where P_2D(k) = b1^2 N^2 P(k) / L   [verified]
# So C_l^{true} = b1^2 P(l/chi) / (L chi^2)  [what we've been using]
#
# Beyond Limber: the actual SHT computes at a specific chi=chi_bar, 
# but w_j involves integrating along the LOS from chi_min to chi_max.
# For our setup, all sightlines are at the same chi_bar (single slab).
# So Limber should be EXACT for a single thin slab. No beyond-Limber correction.
#
# Wait — the slab has ZERO width (all at chi_bar), and the field is 2D.
# So the Limber-like relation C_l = P_2D(l/chi) / chi^2 should be exact 
# for a delta-function window in chi. This IS the flat-sky limit.
#
# But IS it exact for the spherical geometry? Let me check.
# On the flat sky: C_l = integral dk k P_2D(k) B_l^2(k) where B_l is a 
# Bessel function window. For a delta function at chi_bar:
# C_l = P_2D(l/chi_bar) / chi_bar^2  [exactly]
# This uses the Limber approximation jl(x) ~ delta(x - l - 0.5) / sqrt(x).
# Actually for a DELTA source, the exact result involves jl^2(k*chi_bar):
# C_l = (2/pi) integral k^2 dk P_2D(k) jl^2(k chi_bar) / chi_bar^4  ← NO
#
# Hmm, let me think more carefully. The definition is:
# a_lm = integral d^2n w(n) Y_lm^*(n)
# For a 2D field at distance chi_bar, w(n) = delta_2D(chi_bar * theta)
# In the flat-sky limit: a_lm ~ integral d^2theta w(theta) e^{-i l.theta}
# and Cl = P_2D(l/chi_bar) / chi_bar^2.
#
# But spherical effects introduce corrections at low l (large angles).
# For l >> 1, Limber is exact. At l ~ 10, there could be O(1/l) corrections.
# Our lowest bin is l=48, which should be fine.
# The bump at l=48 suggests something else is going on.

# Test 3: What about the box geometry? The sightlines cover a square patch
# on the sky, not a random distribution. The SHT sees this as a specific
# geometry. The ANGULAR size of the box face is theta_box = L / chi_bar.
theta_box = L / chi_bar  # in radians
print(f"\n{'='*80}")
print(f"Test 2: Box geometry")
print(f"  theta_box = {theta_box:.4f} rad = {np.degrees(theta_box):.2f} deg")
print(f"  l_box = pi/theta_box = {np.pi/theta_box:.1f}")
print(f"  l_fund = 2*pi*chi/L = {2*np.pi*chi_bar/L:.1f}")
print(f"  (modes below l_fund are poorly sampled)")
print(f"{'='*80}")

# Test 4: What if the correct formula involves a 2/pi factor from Limber?
# The exact Limber relation for a single shell at chi_bar:
# C_l = P_2D(nu/chi_bar) / chi_bar^2  where nu = l + 1/2
# But there's a well-known Limber correction factor:
# C_l = P_2D(nu/chi_bar) / chi_bar^2 * [1 + O(1/l^2)]
# Let me check if multiplying by (2l+1)/(2l) or similar helps.

print(f"\n{'='*80}")
print("Test 3: Limber corrections and normalization factors")
print(f"{'='*80}")

for label, factor_fn in [
    ("1 (baseline)", lambda l: np.ones_like(l)),
    ("2l/(2l+1)", lambda l: 2*l/(2*l+1)),
    ("l/(l+0.5)", lambda l: l/(l+0.5)),
    ("(l+0.5)/l", lambda l: (l+0.5)/l),
    # ("4pi/(4pi-1)", lambda l: np.full_like(l, 4*np.pi/(4*np.pi-1))),
]:
    k_vals = (ells_dec + 0.5) / chi_bar
    cl_th = b1**2 * plin(k_vals) / (L * chi_bar**2) * factor_fn(ells_dec)
    r = cl_dec / cl_th
    print(f"\n  Factor = {label}:")
    print(f"  Mean r = {np.mean(r[1:]):.4f}, std = {np.std(r[1:]):.4f}")

# Test 5: Compute C_true by summing over discrete k-modes 
# C_l = (1/chi_bar^2) * (1/N^2) * SUM_{kx,ky} b1^2 |a(kx,ky)|^2 * W_l(kperp)
# where W_l(k) = delta(l/chi - k) in Limber limit
# In practice: C_l = (1/chi_bar^2) * P_2D(l/chi_bar) but with discrete modes
# P_2D_discrete(k) = b1^2 N^2 / L * P_lin(k) * delta(k - k_grid)
# averaged over annulus...

# Let me just compute the discrete-mode C_l directly:
# For each ell, find k = ell/chi_bar, then find the nearest grid modes
# and average P_2D over those modes.
print(f"\n{'='*80}")
print("Test 4: Discrete-mode C_l (exact grid P_2D)")
print(f"{'='*80}")

k_fund = 2*np.pi / L
# All discrete k-values in 2D
kx_1d = np.fft.fftfreq(N, d=L/(2*np.pi*N))  # = 2*pi*n/L for n=0,...,N-1
ky_1d = kx_1d.copy()
KX, KY = np.meshgrid(kx_1d, ky_1d)
K2D = np.sqrt(KX**2 + KY**2).ravel()
PK2D = b1**2 * N**2 / L * plin(np.where(K2D > 0, K2D, 1e-10))
PK2D[K2D == 0] = 0

# For each ell bin, average P_2D over modes in [k_lo, k_hi]
for i in range(1, len(ells_dec)):
    ell_lo = ells_dec[i] - NperBin/2
    ell_hi = ells_dec[i] + NperBin/2
    k_lo = ell_lo / chi_bar
    k_hi = ell_hi / chi_bar
    mask_k = (K2D >= k_lo) & (K2D < k_hi)
    n_modes = np.sum(mask_k)
    if n_modes > 0:
        P2D_avg = np.mean(PK2D[mask_k])
        cl_discrete = P2D_avg / (chi_bar**2 * N**2)  # C_true from discrete modes
        cl_limber = b1**2 * plin((ells_dec[i]+0.5)/chi_bar) / (L * chi_bar**2)
        r_disc = cl_dec[i] / cl_discrete if cl_discrete > 0 else np.nan
        print(f"  ell={ells_dec[i]:4.0f}: n_modes={n_modes:5d}, "
              f"P2D_avg/P2D_limber={P2D_avg/(b1**2*N**2/L*plin((ells_dec[i]+0.5)/chi_bar)):.4f}, "
              f"r(discrete)={r_disc:.4f}")
    else:
        print(f"  ell={ells_dec[i]:4.0f}: NO modes in bin!")

# Test 6: The issue with C_true = P_2D/(N^2 chi^2)
# Wait: C_l^{ww} = P_2D(l/chi) / chi^2 is for the CONTINUOUS field.
# The SHT computes alm = SUM_j w_j Y_lm^*, so <|alm|^2> = SUM_{jk} <wj wk> Y*Y
# The PAIR-COUNTING Cl is (1/(2l+1))<|alm|^2> = (1/(2l+1)) SUM_{jk} <wj wk> Y*Y
# = (1/(4pi)) SUM_{jk} <wj wk> P_l(cos gamma_jk)
# The off-diagonal (j≠k) part: <wj wk> = b1^2 N^2/L^3 SUM_kperp P(k) e^{ik.(rj-rk)}
# = P_2D(|rj-rk|) at 2D separation
# and SUM_{j≠k} P_2D(r_jk) P_l(cos gamma_jk) / (4pi)
# = SUM_{j≠k} SUM_L C_L^{ww} (2L+1)/(4pi) P_L(nj) P_L(nk) ... hmm this is M @ C

# Actually let me step back and think about what the deconvolution gives us.
# The relationship is: <Cl_pseudo> = M @ C_true
# We subtract the diagonal: <Cl_clust> = M_clust @ C_true 
# Then invert: C_true_emp = M_clust^{-1} @ <Cl_clust>
# The C_true here is the true angular Cl of the WEIGHT field (not normalized by N^2).
# Wait, no. The pseudo-Cl already includes the W_lambda normalization from the mask.
# M includes the mask coupling. So C_true is the TRUE angular power spectrum 
# of the unmasked field on the full sky. But our field is ZERO outside the box patch.
# So C_true = C_l of the field-times-mask? No, C_true should be the underlying 
# signal that would be observed on the full sky.
# 
# Hmm, for the MASTER formalism: Cl_pseudo = M_lL C_L^{true}
# where C_L^{true} is the FULL-SKY angular Cl of the field.
# But our field only exists on a tiny patch. So C_true is really the Cl
# of a field defined on the full sphere that happens to match our data on the patch.
# The Limber relation C_l = P_2D(l/chi)/chi^2 gives the full-sky Cl for 
# a field with that power spectrum.
# 
# The key question: is the deconvolved C_true consistent with 
# C_l = P_2D(l/chi)/(N^2 chi^2) = b1^2 P(l/chi) / (L chi^2)?
# The factor of N^2 in the denominator is because M uses wl which includes N^2
# from the SHT weight (each sightline has weight N, there are Nskew of them).
# Actually, NO. The SHT weight is w_j directly (not N*w_j). Let me re-check.

# SHT computes: alm = SUM_j w_j Y_lm^*(nhat_j)
# <|alm|^2> = SUM_{jk} <wj wk> Y_lm^*(nj) Y_lm(nk)
# Pseudo-Cl = SUM_m |alm|^2 / (2l+1) = SUM_{jk} <wj wk> P_l(nj.nk) / (4pi)
# 
# wl = SUM_m |SUM_j Y_lm(nj)|^2 / (2l+1)  [window function of UNIT weights]
# 
# Wait, but the code uses NON-UNIT weights in the SHT!
# Actually looking back at the conversation summary:
# "DirectSHT computes a_lm = SUM_j w_j Y_lm^*(n_j) with NO dΩ quadrature weight"
# I assumed w_j here is the pixel weight, which IS the LOS-integrated density.
#
# And for the window function: wl is computed FROM the data with unit weights?
# Or with the actual weights? Let me check what wl_ref actually is.

# From the conversation summary:
# "wl_code = |SUM_j N×Y_lm(n_j)|²/(2l+1) = N² × |SUM Y_lm|²/(2l+1)"
# So wl uses weights N (not 1). Thus M_code = N² × M_mask.
# And the pseudo-Cl uses w_j = (density values).

# Then: <Cl_pseudo> = M_code @ C_true
# = N^2 M_mask @ C_true
# where C_true = Cl of the field w_j (NOT divided by N).

# So C_true = P_2D(l/chi) / chi^2 = b1^2 N^2 P(l/chi) / (L chi^2)

# But then theory = M_code @ C_true = N^2 M_mask @ b1^2 N^2 P / (L chi^2)
# which has N^4, and the data should scale as N^2... that's wrong.

# Let me re-derive carefully, checking what wl_ref actually contains.
print(f"\n{'='*80}")
print("Checking wl_ref normalization")
print(f"{'='*80}")
wl_ref = d['wl_k'][0, :Nl]
# If wl = |SUM_j Y_lm(nj)|^2 / (2l+1) with UNIT weights:
# wl ~ Nskew^2 / (4pi) for isotropic uniform distribution
# If wl = |SUM_j N*Y_lm(nj)|^2 / (2l+1):
# wl ~ N^2 * Nskew^2 / (4pi) ??? No, that doesn't make sense either.

# At lambda=0: Y_00 = 1/sqrt(4pi), so w0 = |SUM_j w0_j * Y_00|^2 
# where w0_j is the SHT weight for j-th sightline.
# If w0_j = 1: w0 = Nskew^2 / (4pi)
# If w0_j = N: w0 = N^2 * Nskew^2 / (4pi)

print(f"wl[0] = {wl_ref[0]:.4e}")
print(f"Nskew^2/(4pi) = {Nskew**2/(4*np.pi):.4e}")
print(f"N^2 * Nskew^2/(4pi) = {N**2 * Nskew**2/(4*np.pi):.4e}")
print(f"wl[0] / (Nskew^2/(4pi)) = {wl_ref[0]/(Nskew**2/(4*np.pi)):.4f}")
print(f"wl[0] / (N^2*Nskew^2/(4pi)) = {wl_ref[0]/(N**2*Nskew**2/(4*np.pi)):.4f}")

# At high lambda: W_floor = N^2 * Nskew / (4pi)
# If w0_j = 1: floor = Nskew/(4pi) [each sightline is independent]
# If w0_j = N: floor = N^2 * Nskew/(4pi) ← matches!
print(f"\nW_floor = {W_floor:.4e}")
print(f"N^2 * Nskew/(4pi) = {N**2*Nskew/(4*np.pi):.4e}")
print(f"Nskew/(4pi) = {Nskew/(4*np.pi):.4e}")
print(f"wl[490] = {wl_ref[490]:.4e}")
