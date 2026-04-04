#!/usr/bin/env python
"""
Test if using a C_true computed from DISCRETE 2D k-modes (instead of smooth Limber)
improves the forward model.

Discrete C_true: for each ℓ, compute k = ℓ/chi, find the NEAREST discrete 
k-modes on the 2D grid, and average P_2D over them.

This accounts for the fact that the GRF only has power at discrete k-values,
not at all k. In the Limber limit, C_ℓ = P_2D(ℓ/chi) / chi^2, but since P_2D
is only nonzero at grid points, the effective C_ℓ should be weighted by the
mode density.
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

# Compute diag_cl
kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
KX, KY = np.meshgrid(kvals, kvals)
K2D = np.sqrt(KX**2 + KY**2).ravel()
PK_flat = plin(np.where(K2D > 0, K2D, 1e-10))
PK_flat[K2D == 0] = 0
w2 = b1**2 * N**2 / L**3 * np.sum(PK_flat)
diag_cl = Nskew * w2 / (4*np.pi)

k_fund = 2*np.pi / L

# Compute discrete C_true for each ℓ
# For ℓ-bin [ℓ, ℓ+1): k-range = [ℓ/chi, (ℓ+1)/chi]
# The number of 2D grid modes in this annulus tells us the mode density.
# C_true(ℓ) should be: (1/chi^2) * (SUM_{k in annulus} P_2D(k)) / (# modes * chi^2)
# No, wrong. Let me think more carefully.

# In Limber: C_ℓ = P_2D(ℓ/chi) / chi^2  [continuous]
# For discrete: the field has power only at grid k-values.
# The SHT at multipole ℓ picks up the field at scale ℓ/chi, which may or may
# not coincide with a grid mode.
# 
# Actually, the FIELD is defined on the grid, so it's periodic and has
# a smooth (not discrete) Fourier representation:
# w(r) = SUM_k a_k e^{ikr}  where k are on the grid
# On the sphere: the angular PS C_ℓ is the Legendre transform of xi(theta).
# xi(theta) = <w(n1) w(n2)>_{n1.n2=cos theta}
#            = SUM_k |a_k|^2 e^{ik.chi(n1-n2)}
#            = SUM_k |a_k|^2 cos(k chi theta)   [for small theta, 2D]
# 
# Using Limber (Bessel → delta): C_ℓ = ∫ dk k P_2D(k) δ(ℓ-kχ)/(kχ^2)
#                                     = P_2D(ℓ/χ) / χ^2
#
# For discrete modes: C_ℓ = (1/χ^2) Σ_k (P_2D(k)/ΔK^2) × δ_K(ℓ-kχ) × ΔK
# where ΔK = 2π/L, and the sum is over the 2D grid.
# In the Limber limit, the Bessel function selects k = ℓ/χ. 
# For discrete modes, it selects the NEAREST grid mode(s).
# The correction depends on how ℓ/χ falls relative to grid modes.
#
# This would show up as an oscillation in the ratio with period δℓ = χ × k_fund.
# δℓ = χ × 2π/L = 25.86
# Our bins are 32 ℓ-wide, which is close to δℓ, so we might be averaging
# over the oscillation → no effect on average.

# Let me compute C_true from BOTH the smooth plin and the discrete-mode average
# and see if the COUPLING of discrete C_true through M_clust is different.

# Build discrete C_true for ℓ = 0 to Nl_large
Nl_large = 2000
C_true_smooth = np.zeros(Nl_large)
C_true_discrete = np.zeros(Nl_large)
n_modes_per_ell = np.zeros(Nl_large)

ell_from_k = K2D * chi_bar  # ℓ = k × χ for each 2D grid mode
P2D_flat = b1**2 * N**2 / L * PK_flat  # P_2D per mode

for ell in range(Nl_large):
    k_eff = (ell + 0.5) / chi_bar
    C_true_smooth[ell] = b1**2 * plin(k_eff) / (L * chi_bar**2)
    
    # Find modes in [ell, ell+1)
    mask_k = (ell_from_k >= ell) & (ell_from_k < ell + 1)
    n_modes = np.sum(mask_k)
    n_modes_per_ell[ell] = n_modes
    if n_modes > 0:
        # Average P_2D from these modes, convert to C_true
        # C_ℓ = <P_2D>_annulus / (N^2 χ^2) ??? No.
        # C_ℓ = P_2D(k) / χ^2 where P_2D includes N^2/L factor
        # So C_ℓ = b1^2 N^2 P(k) / (L χ^2) / N^2 ← the N^2s cancel?
        # Actually, C_true = P_2D(k) / (N^2 χ^2) for the MASTER convention
        # where M uses N^2-weighted wl.
        # NO — we showed C_true = b1^2 P(k) / (L χ^2) [without N^2 denom].
        # P_2D = b1^2 N^2 P/L, so C_true = P_2D / (N^2 χ^2)? Or P_2D/χ^2?
        # From variance: σ^2 = <w^2>/N^2, and σ^2 = (1/(4π)) Σ (2L+1) C_true
        # <w^2>/N^2 = b1^2/L^3 Σ P(k)
        # (1/(4π)) Σ (2L+1) C_true = b1^2/(4π L χ^2) Σ (2L+1) P(L/χ)
        # These should match: b1^2/(4π L χ^2) Σ(2L+1)P ≈ b1^2/L^3 Σ P
        # Which gives: Σ(2L+1)P / (4π L χ^2) ≈ Σ P / L^3
        # → Σ(2L+1)P ≈ 4π χ^2/L^2 Σ P  ← the Limber conversion we checked (0.97)
        
        # So C_true = b1^2 P(k) / (L χ^2) where k = ℓ/χ
        # For discrete: C_true = b1^2 <P(k)>_annulus / (L χ^2)
        C_true_discrete[ell] = b1**2 * np.mean(PK_flat[mask_k]) / (L * chi_bar**2)
    else:
        C_true_discrete[ell] = C_true_smooth[ell]  # use smooth where no modes

print(f"Mode density check:")
for ell_c in [50, 100, 200, 300, 400]:
    nm = int(n_modes_per_ell[ell_c])
    r = C_true_discrete[ell_c] / C_true_smooth[ell_c] if C_true_smooth[ell_c] > 0 else 0
    print(f"  ℓ={ell_c}: {nm} modes, C_discrete/C_smooth = {r:.4f}")

# Build M_clust
wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

# Forward model with both C_trues
theory_smooth = (M_clust @ C_true_smooth)[:Nl] + diag_cl
theory_discrete = (M_clust @ C_true_discrete)[:Nl] + diag_cl

NperBin = 32
n_bins = Nl // NperBin

print(f"\n{'='*80}")
print(f"Forward model comparison: smooth vs discrete C_true")
print(f"{'='*80}")
print(f"  {'ell':>6s} {'r(smooth)':>10s} {'r(discrete)':>12s} {'improvement':>12s}")
print("-" * 50)
for b in range(n_bins):
    lo = b * NperBin
    hi = (b+1) * NperBin
    ell_c = (lo + hi - 1) / 2.0
    d_avg = np.mean(cl_mean[lo:hi])
    ts_avg = np.mean(theory_smooth[lo:hi])
    td_avg = np.mean(theory_discrete[lo:hi])
    rs = d_avg / ts_avg
    rd = d_avg / td_avg
    print(f"  {ell_c:6.1f} {rs:10.4f} {rd:12.4f} {abs(rd-1)-abs(rs-1):12.4f}")

mask = np.ones(Nl, dtype=bool)
mask[:32] = False
mask[480:] = False
r_smooth = np.mean(cl_mean[mask] / theory_smooth[mask])
r_discrete = np.mean(cl_mean[mask] / theory_discrete[mask])
print(f"\n  Mean ratio (smooth):   {r_smooth:.4f}")
print(f"  Mean ratio (discrete): {r_discrete:.4f}")

# Also try computing the variance from discrete C_true to see if it matches
sigma2_discrete = np.sum((2*np.arange(Nl_large)+1) * C_true_discrete) / (4*np.pi)
sigma2_smooth = np.sum((2*np.arange(Nl_large)+1) * C_true_smooth) / (4*np.pi)
sigma2_target = w2 / N**2

print(f"\n  σ²(discrete C_true) = {sigma2_discrete:.6e}")
print(f"  σ²(smooth C_true)   = {sigma2_smooth:.6e}")
print(f"  σ²(target = w²/N²)  = {sigma2_target:.6e}")
print(f"  discrete/target = {sigma2_discrete/sigma2_target:.4f}")
print(f"  smooth/target   = {sigma2_smooth/sigma2_target:.4f}")
