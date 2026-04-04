#!/usr/bin/env python
"""
Compute C_true from discrete 2D k-modes mapped to angular ℓ, 
instead of using smooth Limber interpolation.

The exact relation for a flat sky at distance chi_bar:
  C_ℓ = (1/Omega_patch) * SUM_{k_perp in annulus ℓ/chi ± δk/2} P_2D(k_perp) / chi^2

But for the full-sky SHT which sees the whole sphere:
  C_ℓ^{true} = angular PS of the field defined on the sphere.

For a 2D field at chi_bar: the exact formula (no Limber) is:
  a_ℓm = SUM_j w_j Y_ℓm*(nhat_j)
  
In the continuum limit (field w(nhat) = SUM_k a_k e^{ik.r_perp} where r_perp = chi*nhat_patch):
  a_ℓm = integral d^2n w(n) Y_ℓm*(n)
  
For small patch at pole: n = (theta*cos(phi), theta*sin(phi), 1-theta^2/2)
  r_perp = chi_bar * theta, so k.r_perp = k chi theta cos(phi - phi_k)
  exp(ik.r) ~ SUM iℓ (2ℓ+1) jℓ(k chi theta) Pℓ(cos angle) ← NO, wrong

Actually for a 2D DELTA shell at chi_bar, from the FRW derivation:
  C_ℓ = (2/pi) integral k^2 dk P_3D(k) [jℓ(k chi_bar)]^2

But we have a 2D field (no LOS integration), so:
  C_ℓ = P_2D(ℓ/chi) / chi^2   (Limber approximation)

The exact formula for a 2D field on a shell is:
  C_ℓ = (1/chi^2) integral d^2k_perp/(2pi)^2 P_2D(k_perp) |Ylm_FT(k)|^2
  ≈ P_2D(ℓ/chi_bar) / chi_bar^2 in the Limber limit.

For discrete modes, I should sum over actual k-modes.

But there's another issue: the MASTER relation assumes the signal is isotropic
on the sphere. Our signal is a SQUARE patch. The coupling from the square geometry
might affect the off-diagonal M elements differently than assumed.

Let me try something different: compute C_ℓ by directly summing pairs.
"""
import sys, os, gc, time
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

# ============================================================
# Method 1: Compute C_true by binning P_2D into ℓ shells
# ============================================================
print(f"{'='*80}")
print(f"Method 1: Bin discrete P_2D(k_perp) into ℓ shells via ℓ = k*chi")
print(f"{'='*80}")

kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
KX, KY = np.meshgrid(kvals, kvals)
K2D = np.sqrt(KX**2 + KY**2).ravel()
PK2D_vals = plin(np.where(K2D > 0, K2D, 1e-10))
PK2D_vals[K2D == 0] = 0

# Map k to ℓ
ell_vals = K2D * chi_bar  # ℓ = k * chi_bar

# For each ℓ bin of width 1, count modes and sum P
Nl_max = 700
# P_2D(k) = b1^2 N^2 P_lin(k) / L
P2D = b1**2 * N**2 / L * PK2D_vals
Pk_flat = PK2D_vals  # alias for sum below

# C_true(ℓ) in Limber: P_2D(ℓ/chi) / (chi^2)
# But for discrete modes assigned to ℓ-bin:
# C_true(ℓ) = (1/chi^2) * (1/(2pi k dk)) * SUM_{k in bin} P_2D(k) * delta_k
# Actually for a square survey of area Omega_survey = (L/chi_bar)^2:
# The flat-sky ang PS: C_ℓ = P_2D(ℓ/chi) / (chi^2) [per steradian]
# But on a discrete Fourier grid, each mode occupies (2pi/L)^2 in k-space.
# In the annulus ℓ, ℓ+1: area = 2pi*k*dk = 2pi*(ℓ/chi)*(1/chi)
# Number of modes in annulus = area / (2pi/L)^2 = 2pi*ℓ/(chi * (2pi/L)^2*chi)
#                            = 2pi ℓ L^2 / ((2pi)^2 chi^2)
#                            = ℓ L^2 / (2pi chi^2)
# Integral C_ℓ dℓ = integral P_2D(ℓ/chi)/(chi^2) dℓ 
#                  = integral P_2D(k)/chi dk = integral d^2k/(2pi) P_2D(k)/chi^2 * chi/(2pi k)
# This is the standard Limber result.

# For discrete modes:
# C_ℓ_discrete = (1/(#modes * chi^2)) Sum P_2D(k_i) ??? Not obviously.

# The simplest approach: the angular variance from the field equals:
# sigma^2 = (1/(4pi)) SUM_ℓ (2ℓ+1) C_ℓ = <|f(n)|^2> over the sphere
# For our patch field: <|f(n)|^2> = <w^2> * (Nskew/N_sky_pixels)
# But this mixes signal and geometry.

# Let me try a completely different approach.
# The FLAT-SKY C_ℓ is defined as:
# <|a_ℓ|^2> = C_ℓ_flat = P_2D(ℓ/chi) / chi^2 = b1^2 P_lin(ℓ/chi) / (L chi^2)
# This is per Fourier mode ℓ.
# The FULL-SKY C_ℓ should be the same in the Limber limit:
# <|a_ℓm|^2>/(2ℓ+1) ≈ C_ℓ_flat
# but this only holds for ℓ >> 1 and for a full-sky field.

# For a partial-sky field with mask W(n):
# <|a_ℓm|^2> = SUM_L M_ℓL C_L_true
# where C_L_true is the full-sky PS of the UNMASKED field.
# But our field is zero outside the patch — it IS masked.
# So C_true_full = C_ℓ_flat (the PS the field WOULD have on the full sky).

# The issue might be that C_ℓ_flat = P_2D(ℓ/chi)/chi^2 is NOT the right
# full-sky Cl. The flat-sky Cl corresponds to:
# C_ℓ_fullsky ≈ C_ℓ_flat / f_sky for some definitions???

# Wait, no. The standard MASTER convention is:
# <C_ℓ_pseudo> = SUM_L M_ℓL C_L_fullsky
# C_ℓ_pseudo = SUM_m |a_ℓm|^2 / (2ℓ+1)
# where a_ℓm = integral d^2n [W(n) * s(n)] Y_ℓm*(n)
# s(n) = signal, W(n) = mask (0 or 1)
# C_fullsky = PS of s(n) on the full sphere

# For us: s(n) is defined everywhere but is zero outside the patch.
# Actually, s(n) = signal only on the patch, zero elsewhere.
# So W(n) * s(n) = s(n) for us — the mask and field are inseparable.
# The "true" C_ℓ is the PS of s(n) defined on the FULL sphere, 
# which equals the PS of the partial-sky field.

# Hmm, this is getting circular. Let me just think about what
# a_ℓm = SUM_j w_j Y_ℓm*(nj) gives.

# For unit weight (wj=1):
# b_ℓm = SUM_j Y_ℓm*(nj) → W_ℓ = SUM_m |b_ℓm|^2/(2ℓ+1)
# = SUM_{jk} P_ℓ(nj.nk)/(4pi) = Nskew/(4pi) + SUM_{j≠k} P_ℓ/(4pi)

# For signal:
# a_ℓm = SUM_j wj Y_ℓm*(nj)
# Pseudo-Cℓ = SUM_m |a_ℓm|^2/(2ℓ+1) = SUM_{jk} wj wk P_ℓ(nj.nk)/(4pi)

# MASTER: <Pseudo-Cℓ> = SUM_L M_ℓL_(mask) C_L^{true}
# where mask means: wl computed with unit weights for mask
# and C_true is the PS of the signal that, when modulated by the mask,
# gives the observed field.

# But wait, in the code:
# wl_code = |SUM N*Y_lm|^2/(2l+1) = N^2 * |SUM Y_lm|^2/(2l+1) = N^2 wl_mask
# So M_code = N^2 M_mask
# And <Pseudo-Cl> = M_mask @ C_true = (1/N^2) M_code @ C_true
# So C_true = N^2 * M_code^{-1} @ Pseudo-Cl

# Now what IS C_true?
# By the MASTER theorem: C_true_L is the angular PS that satisfies
# <SUM_{jk} wj wk P_L(gamma_jk)/(4pi)> = SUM_L' M_LL' C_L'
#
# For j≠k, <wj wk> depends on separation rjk. The angular correlation function:
# xi(gamma) = SUM_L (2L+1)/(4pi) C_true_L P_L(gamma)
# should equal the SIGNAL correlation: <wj wk> / <wj> <wk> or something...
# Actually no, C_true IS defined such that the MASTER relation holds.

# In the flat-sky limit for a finite patch:
# The underlying 2D field has PS: p(k) = b1^2 P_lin(k) / L  [per mode, without N^2 factor]
# The angular Cl of this field on the FULL sphere:
# C_ℓ = (2/pi) integral k^2 dk |jℓ(k chi)|^2 p(k) / chi^4 ??? NO, that's for 3D.

# OK, for a 2D field painted on a sphere at radius chi_bar:
# f(nhat) = SUM_k a_k exp(i k_perp . chi_bar*nhat_perp) for nhat near pole
# <a_k a_k'*> = (2pi)^2/Area * P_2D(k) delta(k-k')  where Area = L^2

# The angular PS:
# C_ℓ = integral d^2k/(2pi)^2 P_2D(k) |I_ℓm(k)|^2 
# ... this is getting complicated. 

# Let me just test numerically: the angular sums.

# ============================================================
# Method 2: Check the ratio SUM(2L+1)P(L/chi) vs 4pi chi^2/L^2 SUM_k P(k)
# ============================================================
print(f"\nMethod 2: Angular sum vs k-sum")

ell_Ny = int(np.pi * N / L * chi_bar)
ells = np.arange(1, ell_Ny + 1, dtype=float)

# Sum1: exact angular SUM (2L+1) P(L/chi)
sum_ang = np.sum((2*ells+1) * plin((ells+0.5)/chi_bar))

# Sum2: discrete 2D k-modes SUM P(k_perp) (excluding k=0)
sum_k = np.sum(Pk_flat)

# Expected Limber relation: sum_ang ≈ 4pi chi^2/L^2 * sum_k
# But angular sum samples CONTINUOUSLY in L, while k-sum is on a GRID.
# L range: 1 to ell_Ny (~6620), spacing dL = 1
# k range: k_fund to k_Ny, spacing 2pi/L
# L = k*chi, so dL = chi*dk. The L-sum is finer than the k-grid by factor chi * dk_fund = chi * 2pi/L ≈ 25.9
# So the angular sum has ~26x more "samples" per unit k.

# The angular sum is essentially: integral dL (2L) P(L/chi) 
# = integral dk 2k*chi^2 P(k) = 2 chi^2 integral k P(k) dk

# The k-sum: SUM_k P(k) ≈ (L/(2pi))^2 integral d^2k P(k) = L^2/(2pi) integral k dk P(k)

# So: sum_ang / sum_k ≈ 2 chi^2 integral k P dk / [(L^2/(2pi)) integral k P dk]
#                      = 4pi chi^2 / L^2

ratio_sums = sum_ang / sum_k
expected_ratio = 4*np.pi * chi_bar**2 / L**2
print(f"  sum_ang = {sum_ang:.6e}")
print(f"  sum_k = {sum_k:.6e}")
print(f"  ratio = {ratio_sums:.4f}")
print(f"  expected (4pi chi^2/L^2) = {expected_ratio:.4f}")
print(f"  ratio / expected = {ratio_sums/expected_ratio:.6f}")

# The discrepancy from 1 in ratio/expected tells us the Limber integral mismatch.
# Let me also check what this implies for C_true normalization.

# If C_true = b1^2 P/(X chi^2), then:
# sigma^2 = b1^2/(4pi X chi^2) * sum_ang
# And this should = <w^2>/N^2 = b1^2/(N^2 L^3) * N^2 sum_k = b1^2 * sum_k / L^3

# So b1^2 sum_ang / (4pi X chi^2) = b1^2 sum_k / L^3
# X = sum_ang * L^3 / (4pi chi^2 sum_k) = (sum_ang/sum_k) * L^3/(4pi chi^2)
#   = (4pi chi^2/L^2 * correction) * L^3/(4pi chi^2) = L * correction

X_from_variance = ratio_sums * L**3 / (4*np.pi * chi_bar**2)
print(f"\n  X from variance matching: {X_from_variance:.4f}")
print(f"  X/L = {X_from_variance/L:.6f}")
print(f"  (ratio/expected) = {ratio_sums/expected_ratio:.6f} = X/L")
print(f"  => The 3% deficit from Limber sum gives X = 0.971*L")

# So the VARIANCE matching says X = 0.971*L = 1343.
# But the forward model says X_best ≈ 1225 = 0.886*L.
# The remaining ~8% comes from somewhere in the COUPLING, not just the sigma^2.

# ============================================================
# Method 3: Check if the coupling matrix M_clust itself has a mismatch
# ============================================================
# The off-diagonal pairs contribute: M_clust @ C_true
# where M_clust uses wl_clust = wl - W_floor
# 
# In the flat-sky limit, for a square mask:
# M_lL = (2L+1)/(4pi) * wl(|l-L|)  [for a symmetric mask]
# and the correction would involve the exact Fourier transform of the square mask.
# 
# The square mask FT has sinc-like oscillations, which differ from 
# the spherical harmonic wl of the same mask. This could cause ~% level corrections.

# Let me check: what's the theoretical M for a uniform square mask on the sky?
# Omega_mask = (L/chi)^2
# f_sky = Omega_mask / (4pi)
# For ℓ >> 1: M_ℓL ~ f_sky * delta_ℓL (nearly diagonal)
# wl_mask ~ f_sky * N_sightlines for ℓ < ℓ_fund = 2pi*chi/L
# 
# At ℓ=0: wl_mask[0] = Nskew^2 / (4pi) (check)
#   f_sky * Nskew ≈ Nskew * (L/chi)^2/(4pi)

Omega_mask = (L/chi_bar)**2
f_sky = Omega_mask / (4*np.pi)
ell_fund = 2*np.pi * chi_bar / L

print(f"\n{'='*80}")
print(f"Mask geometry:")
print(f"  Omega_mask = {Omega_mask:.6f} sr")
print(f"  f_sky = {f_sky:.6f}")
print(f"  ell_fund = {ell_fund:.1f}")
print(f"  Nskew = {Nskew}")
print(f"  Nskew * Omega_mask / (4pi) = {Nskew * f_sky:.4f}")
print(f"{'='*80}")

# ============================================================
# Method 4: Cross-check — compute SUM wl_clust and compare with theory
# ============================================================
PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)
wl_ref = d['wl_k'][0, :Nl]

W_floor = N**2 * Nskew / (4*np.pi)

# SUM wl_clust = SUM (wl - W_floor) 
# For the unit-weight mask: SUM wl_mask = (1/(4pi)) SUM (2l+1) wl_mask
# = (1/(4pi)) SUM_m SUM_j,k Yi(nj) Yi(nk) over all l,m
# = (1/(4pi)) SUM_{j,k} 4pi delta(nj,nk) [orthogonality of Ylm]
# = SUM_j delta(nj, nj) = divergent for point sources!
# That's the floor issue.

# For the clustered part: SUM wl_clust captures the excess over shot noise.
# Theory: SUM_L (2L+1)/(4pi) wl_clust[L] = ?
# wl_mask_clust = SUM_m |SUM_j Y_lm(nj)|^2/(2l+1) - Nskew/(4pi)
#               = [SUM_{j≠k} P_l(nj.nk)]/(4pi)  [off-diagonal pairs only]

# For a uniform grid of Nskew points on a square patch:
# SUM_l (2l+1) wl_mask_clust/(4pi) = (1/(4pi)^2) SUM_{j≠k} SUM_l (2l+1) P_l(nj.nk)
# = (1/(4pi)) SUM_{j≠k} delta(nj - nk) = 0 for distinct points
# (since all sightlines are at distinct positions)

# But numerically:
wl_clust_500 = wl_ref - W_floor/N**2  # wait, wl_ref already includes N^2
# Actually wl_ref = N^2 * wl_mask, so wl_ref - W_floor = N^2 * (wl_mask - Nskew/(4pi))
# = N^2 * wl_mask_clust
sum_wl_clust = np.sum((2*np.arange(Nl)+1) * (wl_ref - W_floor)) / (4*np.pi)
print(f"\n  SUM (2L+1) wl_clust/(4pi) for L=0..499 = {sum_wl_clust:.4e}")
print(f"  SUM (2L+1) wl_ref/(4pi) for L=0..499 = {np.sum((2*np.arange(Nl)+1)*wl_ref)/(4*np.pi):.4e}")
print(f"  SUM (2L+1) W_floor/(4pi) = {np.sum((2*np.arange(Nl)+1)*W_floor)/(4*np.pi):.4e}")

# ============================================================
# Method 5: Compute theory by PAIR COUNTING directly (small test)
# ============================================================
# For a subsample of sightlines, compute:
# theory_cl = (1/(4pi)) SUM_{j≠k} xi(rjk) P_l(cos gamma_jk)
# where xi(rjk) = <wj wk> = FT^{-1}[P_2D](rjk)
# This is EXACT (no Limber), but expensive.
# Let me do it for a small subset.

print(f"\n{'='*80}")
print(f"Method 3: Pair counting with a subsample")
print(f"{'='*80}")

# Load sightline angles
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
theta_arr = GRF_tmp.theta_gal
phi_arr = GRF_tmp.phi_gal
del GRF_tmp

# Get Cartesian unit vectors on sphere for fast cos(gamma) computation
x = np.sin(theta_arr) * np.cos(phi_arr)
y = np.sin(theta_arr) * np.sin(phi_arr)
z = np.cos(theta_arr)

# 2D coordinates on the box face (in physical Mpc/h)
# theta is small (near pole), so r_perp ≈ chi_bar * theta (projected)
# More precisely: use the actual GRF grid positions
# The sightlines are on a regular NxN grid subsampled to Nskew.
# Let me get the actual positions.
# From GRF_class: positions are theta, phi from process_skewers
# Actually, I need the 2D r_perp for computing xi(r_perp).

# Use flat-sky approximation for r_perp:
# r_perp between j,k = chi_bar * |nhat_j - nhat_k|_perp ≈ chi_bar * sqrt(dtheta^2 + dphi^2 sin^2(theta))
# For small angles near the pole.

# Actually, let me just compute the 2D correlation function xi(r),
# then for each pair, get xi(r_jk) and Pl(gamma_jk).

# First compute xi(r) from P_2D(k):
from scipy.interpolate import interp1d
from scipy.special import j0

# P_2D(k) = b1^2 * N^2/L * P_lin(k)
# xi_2D(r) = integral k dk/(2pi) P_2D(k) J_0(kr)

k_arr = np.logspace(-4, np.log10(2*np.pi*N/(2*L)), 5000)
P2D_arr = b1**2 * N**2 / L * plin(k_arr)

# For a subsample of Nsub sightlines:
Nsub = 200
rng = np.random.default_rng(42)
idx = rng.choice(Nskew, Nsub, replace=False)

# Compute pairwise cos(gamma)
cos_gamma = np.zeros((Nsub, Nsub))
for i in range(Nsub):
    ii = idx[i]
    cos_gamma[i] = x[ii]*x[idx] + y[ii]*y[idx] + z[ii]*z[idx]

# Pairwise 2D separation (flat-sky)
# chi * angle = chi * arccos(cos_gamma) ≈ chi * sqrt(2*(1-cos_gamma)) for small angles
r_perp = chi_bar * np.arccos(np.clip(cos_gamma, -1, 1))

# xi_2D(r) by numerical integration for unique r values
# This is slow but let's do a few
from scipy.integrate import quad

def xi_2D(r, k_arr, P2D_arr):
    """Compute 2D correlation function by trapezoid integration."""
    integrand = k_arr * P2D_arr * j0(k_arr * r) / (2*np.pi)
    return np.trapz(integrand, k_arr)

# Compute xi for all unique separations
r_flat = r_perp[np.triu_indices(Nsub, k=1)]
print(f"  Computing xi_2D for {len(r_flat)} pairs...")

# Use a lookup table instead
r_table = np.linspace(0, r_flat.max()*1.1, 2000)
xi_table = np.zeros(len(r_table))
for i, r in enumerate(r_table):
    integrand = k_arr * P2D_arr * j0(k_arr * r) / (2*np.pi)
    xi_table[i] = np.trapz(integrand, k_arr)
xi_interp = interp1d(r_table, xi_table, kind='linear', fill_value=0, bounds_error=False)

# Now compute the pair-counting Cl for a few ℓ values
from scipy.special import legendre

ell_test = [50, 100, 200, 300, 400]
print(f"  Pair-counting C_ℓ (subsample of {Nsub} sightlines):")
print(f"  {'ell':>6s} {'Cl_pair':>12s} {'Cl_Limber':>12s} {'ratio':>8s}")

for ell in ell_test:
    Pl = np.polynomial.legendre.legval(cos_gamma, [0]*ell + [1])
    
    xi_mat = xi_interp(r_perp)
    np.fill_diagonal(xi_mat, 0)  # exclude j=k pairs
    
    # Cl_pair = (1/(4pi)) * SUM_{j≠k} xi(rjk) Pl(cos gamma) / Nsub_pairs
    # Wait, but C_true from pair counting should give the FULL Cl, not normalized by N_pairs.
    # Actually: <pseudo-Cl> = (1/(4pi)) SUM_{jk} <wjwk> Pl(gamma)
    # The off-diagonal part: (1/(4pi)) SUM_{j≠k} xi(rjk) Pl(gamma)
    # For Nsub sightlines:
    Cl_pair_offdiag = np.sum(np.triu(xi_mat * Pl, k=1)) / (2*np.pi)  # factor of 2 for symmetry
    
    # Hmm, the normalization is tricky for a subsample.
    # Let me use the MASTER relation instead.
    # <Cl_pseudo> = M @ C_true
    # Off-diagonal pseudo-Cl = M_clust @ C_true
    # And M_clust scales as N^2 * Nskew * (Nskew-1) for the off-diagonal part... no.
    
    # Actually, let's just check the RATIO of pair-counting at different ℓ
    # vs Limber prediction at different ℓ. The absolute normalization cancels.
    Cl_limber = b1**2 * plin((ell+0.5)/chi_bar) / (L * chi_bar**2)
    print(f"    {ell:4d}    {Cl_pair_offdiag:.4e}    {Cl_limber:.4e}    (ratio not meaningful for subsample)")

# ============================================================
# Method 6: Compare wl_clust shape with theoretical wl of square mask
# ============================================================
# For a square mask of angular size theta_box x theta_box:
# wl ~ Nskew^2 * |W_l|^2 where W_l is the SH transform of the mask shape.
# For a square cap at the pole: W_l ~ sinc-like function.
# The wl should follow a specific pattern related to the mask geometry.

print(f"\n{'='*80}")
print(f"Method 4: wl_clust shape analysis")
print(f"{'='*80}")

wl_clust_code = wl_ref - W_floor  # These are in code units (N^2 * wl_mask_clust)
wl_mask_clust = wl_clust_code / N**2

# The theoretical wl for a uniform square mask:
# At l=0: wl_mask[0] = Nskew^2/(4pi), so wl_clust[0] = (Nskew^2 - Nskew)/(4pi) = Nskew(Nskew-1)/(4pi)
print(f"  wl_mask_clust[0] = {wl_mask_clust[0]:.4e}")
print(f"  Nskew(Nskew-1)/(4pi) = {Nskew*(Nskew-1)/(4*np.pi):.4e}")
print(f"  Ratio = {wl_mask_clust[0]/(Nskew*(Nskew-1)/(4*np.pi)):.6f}")

# At high ℓ: wl_mask_clust should decay as the mask geometry smooths out
# For ℓ >> ℓ_fund: wl_clust → 0 (no correlation between sightlines at small scales)
print(f"\n  wl_mask_clust profile:")
for ll in [0, 10, 25, 50, 100, 200, 300, 400, 490]:
    print(f"    l={ll:3d}: wl_clust = {wl_mask_clust[ll]:.4e}, "
          f"wl_clust/wl_clust[0] = {wl_mask_clust[ll]/wl_mask_clust[0]:.6f}")

# The angular scale of the mask: theta_mask ≈ L/chi
# So wl_clust should start decaying at ℓ ~ pi/theta_mask = pi*chi/L ≈ 13
# And have oscillations with period ℓ ~ 2*pi*chi/L ≈ 26
print(f"\n  Expected oscillation period: delta_ell ≈ {2*np.pi*chi_bar/L:.1f}")
