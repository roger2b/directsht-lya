#!/usr/bin/env python
"""
Find the value of X such that data/theory = 1 after floor subtraction.

theory = M_clust @ C_true + W_floor * sigma^2
C_true = b1^2 P_lin / (X * chi^2)
sigma^2 = (1/(4pi)) SUM (2L+1) C_true = b1^2/(4pi*X*chi^2) SUM (2L+1) P_lin(L/chi)
"""
import sys, os, gc, time
import numpy as np
from scipy.optimize import brentq

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
print(f"Parameters: N={N}, L={L:.2f}, chi_bar={chi_bar:.2f}, b1={b1:.4f}")
print(f"W_floor = {W_floor:.4e}")

# Compute sigma^2 sum (goes to ell_Ny since GRF has no power above k_Ny)
ell_Ny = int(np.pi * N / L * chi_bar)
ells_sigma = np.arange(ell_Ny, dtype=float)
plin_sigma = plin((ells_sigma + 0.5)/chi_bar)
# sigma^2 = (1/(4pi)) * (b1^2/chi^2) * SUM (2L+1)*P_lin(L/chi) / X = S / X
S_sigma = b1**2 / chi_bar**2 * np.sum((2*ells_sigma+1) * plin_sigma) / (4*np.pi)
print(f"S_sigma = {S_sigma:.6e} (sigma^2 = S_sigma / X)")

# Also check: what's <w^2> / (Nskew)?
# <w^2> = b1^2 * N^2 / L^3 * SUM_kperp P_lin(K_perp)
# From previous verification: <w^2> matches theory to 0.01%
# The variance sigma^2 of the angular field is <w^2> * (some angular factor)

# Compute M_clust @ (b1^2 P_lin / chi^2) for Nl_large=2000 (well converged after floor sub)
Nl_large = 2000
ells_ext = np.arange(Nl_large, dtype=float)
plin_vals = plin((ells_ext + 0.5)/chi_bar)
cl_over_chi2 = b1**2 * plin_vals / chi_bar**2  # C_true without 1/X factor

wl_needed = 2*Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor  # Fill beyond data with floor
wl_clust = wl_raw - W_floor

couple_c = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple_c.compute_matrix()
theory_clust = (M_clust @ cl_over_chi2)[:Nl]  # Without 1/X factor

# theory(X) = theory_clust / X + W_floor * S_sigma / X = (theory_clust + W_floor * S_sigma) / X
theory_total_unnorm = theory_clust + W_floor * S_sigma

# Check: print some values
print(f"\ntheory_clust[100] = {theory_clust[100]:.6e}")
print(f"W_floor * S_sigma = {W_floor * S_sigma:.6e}")
print(f"theory_total[100] / X = (above + above) / X")

mask = np.ones(Nl, dtype=bool)
mask[:30] = False
mask[450:] = False

# data / (theory_total / X) = X * data / theory_total
# Want this = 1, so X = theory_total_mean / data_mean
data_mean = np.mean(cl_mean[mask])
theory_total_mean = np.mean(theory_total_unnorm[mask])
X_fit = theory_total_mean / data_mean
print(f"\nX_fit = {X_fit:.4f}")
print(f"L = {L:.4f}")
print(f"32*pi^3 = {32*np.pi**3:.4f}")
print(f"X_fit / L = {X_fit/L:.6f}")
print(f"X_fit / (32*pi^3) = {X_fit/(32*np.pi**3):.6f}")

# Check per-ell ratios with X_fit
theory_Xfit = theory_total_unnorm / X_fit
print(f"\n=== Ratio data/theory(X_fit) per ell bin ===")
from sht.mask_deconvolution import MaskDeconvolution
NperBin = 32
wl_ref = d['wl_k'][0, :Nl]
# Need standard M for MaskDeconv (not the clust one)
couple_std = Wigner3j.CoupleMat(Nl, wl_ref)
M_std = couple_std.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=M_std)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells
binned_data = bins @ cl_mean
binned_theory = bins @ theory_Xfit

for i in range(len(binned_ells)):
    if binned_theory[i] > 0:
        print(f"  ell={binned_ells[i]:6.0f}: data/theory = {binned_data[i]/binned_theory[i]:.4f}")

# What's the mean ratio excluding monopole?
ratios = binned_data[1:] / binned_theory[1:]
print(f"\nMean ratio (excl mono): {np.mean(ratios):.4f}")
print(f"Std of ratio: {np.std(ratios):.4f}")

# Now try different Nl_large to check stability of X_fit
print(f"\n=== X_fit stability with Nl_large ===")
print(f"{'Nl_large':>8s} {'X_fit':>10s} {'X_fit/L':>10s}")
for Nl_lg in [500, 1000, 1500, 2000, 3000, 3500]:
    t0 = time.time()
    ells_e = np.arange(Nl_lg, dtype=float)
    cl_e = b1**2 * plin((ells_e + 0.5)/chi_bar) / chi_bar**2
    
    wl_n = 2*Nl_lg - 1
    wl_r = np.zeros(wl_n)
    na = min(wl_n, len(wl_ext))
    wl_r[:na] = wl_ext[:na]
    wl_r[na:] = W_floor
    wl_c = wl_r - W_floor
    
    cp = Wigner3j.CoupleMat(Nl_lg, wl_c)
    Mc = cp.compute_matrix()
    tc = (Mc @ cl_e)[:Nl]
    tt = tc + W_floor * S_sigma
    Xf = np.mean(tt[mask]) / data_mean
    print(f"  {Nl_lg:5d}  {Xf:10.4f} {Xf/L:10.6f}  [{time.time()-t0:.1f}s]")
    del cp, Mc; gc.collect()

# Check: maybe the 9% discrepancy from X_fit/L = 0.917 is because the 
# sigma^2 sum should use the actual DISCRETE P_2D, not the smooth interpolation.
# 
# P_2D(k) = b1^2 N^2 P_lin(k) / L  [verified]
# But plin is an interpolation of the transfer function. At k > k_Ny, 
# the GRF actually has zero power, but plin still returns finite values.
# The sigma^2 sum should only include k < k_Ny.
# Wait, I already cut at ell_Ny. But should I cut EXACTLY at the grid modes?

# Actually, the Limber conversion ell = k*chi maps discrete k-modes to discrete ell values.
# The correct C_true is a "comb" at those ell values, not a smooth function.
# The smooth Limber C_true overcounts by filling in between the comb teeth.

# Mode counting: in 2D annulus at k, the number of modes is ~2*pi*k*dk*(L/(2pi))^2
# = L^2 k dk / (2*pi)
# Integrated over all k: total modes = (pi*N^2/L) * integral from k_f to k_Ny dk*k
# Hmm, not useful directly.

# Instead, let me compute sigma^2 from the actual grid modes.
# <w^2> = (b1^2 N^2/L^3) SUM_{kx,ky} P_lin(K_perp)
# where K_perp = 2*pi*sqrt(kx^2+ky^2)/L and kx,ky = 0,...,N-1
# This should match sigma^2 * Nskew * N^2/(4*pi) or something...

# Actually, let me just compute <w^2> directly:
# <w^2> = (b1^2 N^2/L^3) SUM_{kx,ky=0}^{N-1} P_lin(2*pi*sqrt(kx^2+ky^2)/L)
dk = 2*np.pi / L
kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)  # physical k values
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2)
K_perp_flat = K_perp.ravel()
# Remove k=0 mode (DC)
Pk_flat = plin(np.where(K_perp_flat > 0, K_perp_flat, 1e-10))
Pk_flat[K_perp_flat == 0] = 0

w2_theory = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
print(f"\n=== Variance from discrete modes ===")
print(f"<w^2>_discrete = {w2_theory:.6e}")

# Now, sigma^2 of the angular field:
# C_l^{ww} = P_2D(l/chi) / chi^2
# sigma^2 = (1/(4pi)) SUM_l (2l+1) C_l = (1/(4pi)) SUM (2l+1) P_2D(l/chi) / chi^2
# Converting to k: l = k*chi, SUM_l ~ integral dl = integral chi dk
# This is complex for discrete modes...

# Actually: the field variance is simpler.
# Cl from pair counting: <Cl> = (1/(4pi)) SUM_{jk} <wj wk> P_l(cos gamma)
# Total variance: SUM_l (2l+1) <Cl> / (4pi) = (1/(4pi)^2) SUM_{jk} <wj wk> SUM_l (2l+1) P_l
# But SUM (2l+1) P_l(x) / (4pi) = delta(x-1)/sin(x). So this gives for j=k: <w^2> * Nskew * something

# Better: the "data" Cl already includes BOTH diagonal and off-diagonal.
# The diagonal contribution: (1/(4pi)) SUM_j <wj^2> P_l(1) = Nskew * <w^2> / (4pi)
# This is ell-independent. So:
diag_cl = Nskew * w2_theory / (4*np.pi)
print(f"Diagonal contribution to Cl: Nskew * <w^2> / (4*pi) = {diag_cl:.4e}")
print(f"cl_mean[100] = {cl_mean[100]:.4e}")
print(f"diag/cl_mean[100] = {diag_cl/cl_mean[100]:.4f}")

# Similarly, the W_floor contribution to theory:
# M_floor[l,L] = (2L+1) W_floor / (4pi)  [from completeness, summing 3j^2 = 1]
# (M_floor @ C_true)_l = (W_floor/(4pi)) SUM (2L+1) C_L = W_floor * sigma^2
# = W_floor * Nskew * <w^2> / (4pi * N^2)  ???  No, sigma^2 ≠ Nskew*<w^2>/(4pi*N^2)

# Let me rethink. sigma^2 is defined as (1/(4pi)) SUM (2L+1) C_true.
# And C_true = C_ww / N^2 = P_2D(l/chi) / (N^2 chi^2)
# sigma^2 = (1/(4pi*N^2*chi^2)) SUM (2L+1) P_2D(L/chi)

# P_2D = b1^2 N^2 P_lin(k) / L, so P_2D(L/chi) = b1^2 N^2 P_lin(L/chi) / L
# sigma^2 = (b1^2/(4pi L chi^2)) SUM (2L+1) P_lin(L/chi)

# Now <w^2> = (b1^2 N^2/L^3) SUM_{kperp} P_lin(K_perp) [discrete sum over N^2 modes]
# Converting the angular SUM to k-space: L = k*chi, dL = chi*dk
# SUM_L (2L+1) P(L/chi) ~ integral 2k*chi^2 dk P(k) = 2 chi^2 integral k dk P(k)
# And integral k dk P(k) for discrete modes = SUM |k_perp| delta_k P(k_perp) 
# ~ (2pi/L^2) SUM P(K) * integral over annuli

# This is getting too abstract. Let me just compare numerically.
# 
# W_floor * sigma^2(X=L) with diag_cl:
# theory_floor = W_floor * b1^2/(4pi*L*chi^2) * SUM (2L+1) P(L/chi)
# diag_cl = Nskew * b1^2 N^2 / (4pi L^3) * SUM_{kperp} P(K)

# Ratio:
# theory_floor / diag_cl = [W_floor * SUM(2L+1)P/(L*chi^2)] / [Nskew*N^2*SUM_k P(K)/L^3]
# W_floor = N^2*Nskew/(4pi), so
# = [N^2*Nskew/(4pi) * SUM(2L+1)P/(L*chi^2)] / [Nskew*N^2*SUM P/L^3]
# = [SUM(2L+1)P/(4pi*L*chi^2)] / [SUM P/L^3]
# = [L^3 SUM(2L+1) P(L/chi)] / [4pi*L*chi^2 * SUM_k P(K)]
# = [L^2 SUM(2L+1)P(L/chi)] / [4pi*chi^2 * SUM_k P(K)]
# Converting L to k: SUM(2L+1)P(L/chi) ~ integral 2k*chi^2*dk P(k) * (chi/dk_ell)...

# Too complicated. Let me just compute both numerically.
print(f"\ntheory_floor(X=L) = {theory_floor_L:.4e}")
print(f"diag_cl = {diag_cl:.4e}")
print(f"ratio = {theory_floor_L/diag_cl:.4f}")

# If they should match (floor term = diagonal Cl), we'd expect ratio = 1.
# If not 1, the mismatch tells us about the L vs 32pi^3 question.
