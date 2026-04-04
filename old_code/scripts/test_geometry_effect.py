#!/usr/bin/env python
"""
Investigate the ell-dependent ~3-4% residual at chi_0.

ROOT CAUSE HYPOTHESIS: Sightlines live on a FLAT PLANE at x=chi_0, not on
a spherical shell. The 3D position of sightline j is:
    r_j = (chi_0, y_j, z_j)
The distance from origin varies:
    |r_j| = sqrt(chi_0^2 + y_j^2 + z_j^2)
This ranges from chi_0 (center) to chi_0*sqrt(1+2(L/chi_0)^2) (corners).

Different sightlines are at different effective distances, breaking the 
simple Limber relation C_true = P(ell/chi)/chi^2 with a single chi.

Tests:
  1. Distribution of r_j for the actual sightlines
  2. C_true_eff(ell) = <P(ell/r_j) / (L r_j^2)> averaged over sightlines
  3. Per-pair average: C_true_pair(ell) using pair-weighted r_eff
  4. Sub-shell approach: break LOS into chunks with different effective chi
  5. Best overall approach
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# ================================================================== #
# Load data and setup                                                 #
# ================================================================== #
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
del GRF_tmp; gc.collect()

dchi = L / N
chi_0 = 5000.0
chi_bar = chi_0 + L / 2.0

# Floor and diag_cl (same as before)
W_floor = N**2 * Nskew / (4*np.pi)

kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2 = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2 / (4*np.pi)

# ================================================================== #
# 1. Get exact sightline positions and distances                      #
# ================================================================== #
print("="*80)
print("1. SIGHTLINE GEOMETRY")
print("="*80)

# Reproduce the exact sightline positions
coords_grid = np.linspace(0, L, N)
np.random.seed(100)
inds = np.unique(np.random.randint(0, N, size=(10000, 2)), axis=0)
Nskew_check = len(inds)
print(f"Nskew = {Nskew_check} (should be {Nskew})")

# Sightline transverse positions in the box
y_sightlines = coords_grid[inds[:, 1]]  # coords[1] = y
z_sightlines = coords_grid[inds[:, 0]]  # coords[0] = x -> z after swap

# In process_skewers: all_x = tmp_all_z = coords[2] + shift (radial)
# all_y = tmp_all_y = coords[1] (transverse)
# all_z = tmp_all_x = coords[0] (transverse)
# compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])
# i.e., (x=chi_0, y=y_sightlines, z=z_sightlines)

x_at_first_pixel = chi_0  # all_x[:,0] = chi_0 for all sightlines

# Distance from origin for each sightline
r_j = np.sqrt(chi_0**2 + y_sightlines**2 + z_sightlines**2)

print(f"\n  chi_0 = {chi_0:.1f}")
print(f"  chi_bar = {chi_bar:.2f}")
print(f"  L = {L:.2f}")
print(f"  L/chi_0 = {L/chi_0:.4f} (NOT thin!)")
print(f"\n  Sightline distances from origin:")
print(f"    min(r_j) = {r_j.min():.1f}  (sightlines near y=z=0)")
print(f"    max(r_j) = {r_j.max():.1f}  (sightlines near y=z=L)")
print(f"    mean(r_j) = {np.mean(r_j):.1f}")
print(f"    median(r_j) = {np.median(r_j):.1f}")
print(f"    rms(r_j) = {np.sqrt(np.mean(r_j**2)):.1f}")
print(f"    <1/r_j^2>^{-1/2} = {1/np.sqrt(np.mean(1/r_j**2)):.1f}")

# Distribution
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    print(f"    {p}th percentile: {np.percentile(r_j, p):.1f}")

# Theoretical estimates for uniform distribution on [0,L]^2:
print(f"\n  Theoretical for uniform on [0,L]^2:")
print(f"    <y^2> = L^2/3 = {L**2/3:.0f}")
print(f"    <y^2+z^2> = 2L^2/3 = {2*L**2/3:.0f}")
print(f"    sqrt(chi_0^2 + 2L^2/3) = {np.sqrt(chi_0**2 + 2*L**2/3):.1f}")

# ================================================================== #
# Build coupling matrix                                               #
# ================================================================== #
Nl_large = 2000
wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

print(f"\nBuilding coupling matrix (Nl_large={Nl_large})...")
t0 = time.time()
couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()
print(f"  Done in {time.time()-t0:.1f}s")
del couple; gc.collect()

# ================================================================== #
# 2. C_true with sightline-averaged geometry                         #
# ================================================================== #
print(f"\n{'='*80}")
print("2. SIGHTLINE-AVERAGED C_true")
print("   C_true_eff(ell) = b1^2 <P(ell/r_j) / (L r_j^2)>_j")
print("   vs. C_true_chi0 = b1^2 P(ell/chi_0) / (L chi_0^2)")
print(f"{'='*80}")

ells_ext = np.arange(Nl_large, dtype=float)
NperBin = 32
n_bins = Nl // NperBin
mask = np.ones(Nl, dtype=bool)
mask[:NperBin] = False
mask[Nl-NperBin:] = False

# C_true at chi_0 (current)
C_true_chi0 = b1**2 * plin((ells_ext + 0.5) / chi_0) / (L * chi_0**2)

# C_true averaged over sightline distances
# For each ell, average P((ell+0.5)/r_j) / r_j^2 over all sightlines
C_true_avg = np.zeros(Nl_large)
for el_idx in range(Nl_large):
    ell_val = el_idx + 0.5
    k_perp_j = ell_val / r_j  # different k for each sightline
    P_j = plin(k_perp_j)
    C_true_avg[el_idx] = b1**2 * np.mean(P_j / r_j**2) / L

# Ratio: C_true_avg / C_true_chi0
ratio_ctrue = C_true_avg[:Nl] / C_true_chi0[:Nl]
print(f"\n  C_true_avg/C_true_chi0 at ell=50:  {ratio_ctrue[50]:.5f}")
print(f"  C_true_avg/C_true_chi0 at ell=100: {ratio_ctrue[100]:.5f}")
print(f"  C_true_avg/C_true_chi0 at ell=250: {ratio_ctrue[250]:.5f}")
print(f"  C_true_avg/C_true_chi0 at ell=450: {ratio_ctrue[450]:.5f}")

# Forward model with each C_true
theory_chi0 = (M_clust @ C_true_chi0)[:Nl] + diag_cl
theory_avg  = (M_clust @ C_true_avg)[:Nl] + diag_cl

print(f"\n  Forward model ratios (data/theory):")
print(f"  {'ell':>6s}  {'r(chi_0)':>10s}  {'r(avg_rj)':>10s}  {'C_ratio':>10s}")
for b in range(n_bins):
    lo = b * NperBin
    hi = (b + 1) * NperBin
    ell_c = (lo + hi - 1) / 2.0
    d_avg = np.mean(cl_mean[lo:hi])
    r0 = d_avg / np.mean(theory_chi0[lo:hi])
    ra = d_avg / np.mean(theory_avg[lo:hi])
    cr = np.mean(ratio_ctrue[lo:hi])
    print(f"  {ell_c:6.1f}  {r0:10.5f}  {ra:10.5f}  {cr:10.5f}")

ratios_chi0 = []
ratios_avg = []
for b in range(1, n_bins - 1):
    lo = b * NperBin
    hi = (b + 1) * NperBin
    ratios_chi0.append(np.mean(cl_mean[lo:hi]) / np.mean(theory_chi0[lo:hi]))
    ratios_avg.append(np.mean(cl_mean[lo:hi]) / np.mean(theory_avg[lo:hi]))

print(f"\n  Summary (excl first/last bin):")
print(f"    chi_0:  mean r = {np.mean(ratios_chi0):.5f} ± {np.std(ratios_chi0):.5f}")
print(f"    avg_rj: mean r = {np.mean(ratios_avg):.5f} ± {np.std(ratios_avg):.5f}")

# ================================================================== #
# 3. PAIR-WEIGHTED effective chi                                      #
# ================================================================== #
print(f"\n{'='*80}")
print("3. PAIR-WEIGHTED EFFECTIVE CHI")
print("   For pair (j,k), the angular separation depends on (r_j, r_k).")
print("   The effective distance for the pair is r_eff ~ (r_j + r_k)/2")
print("   or geometric mean sqrt(r_j * r_k).")
print(f"{'='*80}")

# Compute pair-weighted averages (subsample for speed)
np.random.seed(42)
n_sample = 2000  # subsample of sightlines for pair computation
idx = np.random.choice(len(r_j), n_sample, replace=False)
r_sub = r_j[idx]

# All pairs: r_eff = geometric mean of (r_j, r_k)
r_i_grid, r_k_grid = np.meshgrid(r_sub, r_sub)
r_geom = np.sqrt(r_i_grid * r_k_grid)  # geometric mean
r_arith = (r_i_grid + r_k_grid) / 2.0   # arithmetic mean
r_harm = 2.0 / (1.0/r_i_grid + 1.0/r_k_grid)  # harmonic mean

# Flatten (exclude diagonal)
mask_pairs = ~np.eye(n_sample, dtype=bool)
r_geom_flat = r_geom[mask_pairs]
r_arith_flat = r_arith[mask_pairs]
r_harm_flat = r_harm[mask_pairs]

print(f"\n  Pair statistics ({n_sample} sightlines, {len(r_geom_flat)} pairs):")
print(f"    <r_geom_pair> = {np.mean(r_geom_flat):.1f}")
print(f"    <r_arith_pair> = {np.mean(r_arith_flat):.1f}")
print(f"    <r_harm_pair> = {np.mean(r_harm_flat):.1f}")

# C_true with pair-weighted average
# C_true_pair(ell) = b1^2 <P(ell/r_pair) / (L r_pair^2)>_pairs
C_true_pair_geom = np.zeros(Nl_large)
for el_idx in range(0, Nl_large, 10):  # every 10th ell for speed
    ell_val = el_idx + 0.5
    k_vals = ell_val / r_geom_flat
    P_vals = plin(k_vals)
    C_true_pair_geom[el_idx] = b1**2 * np.mean(P_vals / r_geom_flat**2) / L

# Interpolate gaps
from scipy.interpolate import interp1d
idx_computed = np.arange(0, Nl_large, 10)
interp = interp1d(idx_computed, C_true_pair_geom[idx_computed], 
                   kind='linear', fill_value='extrapolate')
C_true_pair_geom = interp(np.arange(Nl_large))

theory_pair = (M_clust @ C_true_pair_geom)[:Nl] + diag_cl

ratios_pair = []
for b in range(1, n_bins - 1):
    lo = b * NperBin
    hi = (b + 1) * NperBin
    ratios_pair.append(np.mean(cl_mean[lo:hi]) / np.mean(theory_pair[lo:hi]))
print(f"\n  pair_geom: mean r = {np.mean(ratios_pair):.5f} ± {np.std(ratios_pair):.5f}")

# ================================================================== #
# 4. EFFECTIVE chi for different statistics                           #
# ================================================================== #
print(f"\n{'='*80}")
print("4. EFFECTIVE chi for different averages")
print(f"{'='*80}")

# Try different chi values
chi_candidates = {
    'chi_0                ': chi_0,
    'chi_bar              ': chi_bar,
    'mean(r_j)            ': np.mean(r_j),
    'median(r_j)          ': np.median(r_j),
    'rms(r_j)             ': np.sqrt(np.mean(r_j**2)),
    '<1/r^2>^{-1/2}       ': 1.0/np.sqrt(np.mean(1.0/r_j**2)),
    '<1/r^4>^{-1/4}       ': (np.mean(1.0/r_j**4))**(-0.25),
    'chi_0 + L/3          ': chi_0 + L/3.0,
    'chi_0 + 2L^2/(6*chi0)': chi_0 + 2*L**2/(6*chi_0),  # second-order correction
}

for label, chi in chi_candidates.items():
    C_true_test = b1**2 * plin((ells_ext + 0.5) / chi) / (L * chi**2)
    theory_test = (M_clust @ C_true_test)[:Nl] + diag_cl
    ratios_test = []
    ell_ctrs = []
    for b in range(1, n_bins - 1):
        lo = b * NperBin
        hi = (b + 1) * NperBin
        ratios_test.append(np.mean(cl_mean[lo:hi]) / np.mean(theory_test[lo:hi]))
        ell_ctrs.append((lo + hi - 1) / 2.0)
    slope = np.polyfit(ell_ctrs, ratios_test, 1)[0]
    print(f"  chi={chi:8.1f} ({label}): r = {np.mean(ratios_test):.5f} ± {np.std(ratios_test):.5f}  slope={slope:+.2e}")

# ================================================================== #
# 5. ELL-DEPENDENT effective chi                                      #
# ================================================================== #
print(f"\n{'='*80}")
print("5. ELL-DEPENDENT EFFECTIVE CHI from sightline averaging")
print("   chi_eff(ell) such that P(ell/chi_eff)/chi_eff^2 = <P(ell/r_j)/r_j^2>")
print(f"{'='*80}")

# For each ell, find chi_eff
chi_eff_per_ell = np.zeros(Nl)
for el_idx in range(Nl):
    ell_val = el_idx + 0.5
    target = C_true_avg[el_idx]
    
    # Binary search for chi_eff
    chi_lo, chi_hi = 4800, 6000
    for _ in range(50):
        chi_mid = (chi_lo + chi_hi) / 2.0
        val = b1**2 * plin(ell_val / chi_mid) / (L * chi_mid**2)
        if val > target:
            chi_lo = chi_mid
        else:
            chi_hi = chi_mid
    chi_eff_per_ell[el_idx] = (chi_lo + chi_hi) / 2.0

# Print chi_eff at selected ell values
for el in [10, 50, 100, 200, 300, 400, 490]:
    print(f"  ell={el:3d}: chi_eff = {chi_eff_per_ell[el]:.1f}  "
          f"(chi_eff/chi_0 = {chi_eff_per_ell[el]/chi_0:.4f})")

print(f"\n  Mean chi_eff = {np.mean(chi_eff_per_ell[50:450]):.1f}")
print(f"  chi_eff is CONSTANT across ell? "
      f"std/mean = {np.std(chi_eff_per_ell[50:450])/np.mean(chi_eff_per_ell[50:450]):.5f}")

# ================================================================== #
# 6. THE KEY INSIGHT: exact pair-sum vs MASTER                        #
# ================================================================== #
print(f"\n{'='*80}")
print("6. EXACT PAIR SUM (brute force, small subsample)")
print("   Compute Sum_jk xi_2D(Delta_r_jk) * P_ell(cos gamma_jk) directly")
print("   Compare with the MASTER forward model.")
print(f"{'='*80}")

# Compute 2D correlation function from P_lin
from scipy.integrate import quad
from scipy.special import j0 as J0

# xi_2D(r) = b1^2 / L * integral dk k/(2pi) P_lin(k) J0(k*r)
# We'll tabulate this and interpolate
r_tab = np.geomspace(0.1, 3000, 1000)

def xi_2D_integrand(k, r):
    return k / (2*np.pi) * plin(k) * J0(k * r)

# Use discrete sum instead of integral (faster, exact for the box)
xi_2D_tab = np.zeros_like(r_tab)
kvals_1d = np.fft.fftfreq(N, d=L/N) * 2*np.pi  # 1D k-values
kx2, ky2 = np.meshgrid(kvals_1d, kvals_1d)
k_flat = np.sqrt(kx2**2 + ky2**2).ravel()
Pk_vals = plin(np.where(k_flat > 0, k_flat, 1e-10))
Pk_vals[k_flat == 0] = 0

print("  Computing 2D correlation function from discrete modes...")
t0 = time.time()
for ir, r in enumerate(r_tab):
    xi_2D_tab[ir] = b1**2 * N**2 / L**3 * np.sum(Pk_vals * J0(k_flat * r))
print(f"  Done in {time.time()-t0:.1f}s")

xi_2D_interp = interp1d(r_tab, xi_2D_tab, kind='cubic', 
                          fill_value=0, bounds_error=False)

# Use a small subsample for the pair sum
n_sub = 500
np.random.seed(123)
idx_sub = np.random.choice(Nskew, n_sub, replace=False)
y_sub = y_sightlines[idx_sub]
z_sub = z_sightlines[idx_sub]
r_sub2 = r_j[idx_sub]

# Angular positions (reproduce the exact computation)
x_sub = np.full(n_sub, chi_0)  # all at chi_0
phi_sub = np.arctan2(y_sub, x_sub)
s_sub = np.sqrt(x_sub**2 + y_sub**2)
theta_sub = np.arctan2(s_sub, z_sub)

# Compute cos(gamma_jk) for all pairs
from itertools import combinations

# Unit vectors
nhat = np.column_stack([np.sin(theta_sub)*np.cos(phi_sub),
                         np.sin(theta_sub)*np.sin(phi_sub),
                         np.cos(theta_sub)])
cos_gamma = nhat @ nhat.T  # (n_sub, n_sub)

# Physical transverse separations
Delta_r = np.sqrt((y_sub[:,None]-y_sub[None,:])**2 + 
                   (z_sub[:,None]-z_sub[None,:])**2)

# Exact pair-counting Cl
from numpy.polynomial.legendre import legval
print(f"\n  Computing exact pair sum for {n_sub} sightlines ({n_sub**2} pairs)...")
t0 = time.time()

xi_vals = xi_2D_interp(Delta_r)

exact_cl = np.zeros(Nl)
# Use Legendre polynomial recurrence for efficiency
P_prev = np.ones_like(cos_gamma)   # P_0
P_curr = cos_gamma.copy()           # P_1
exact_cl[0] = np.sum(xi_vals * P_prev) / (4*np.pi)
exact_cl[1] = np.sum(xi_vals * P_curr) / (4*np.pi)

for el in range(2, Nl):
    P_next = ((2*el-1)*cos_gamma*P_curr - (el-1)*P_prev) / el
    exact_cl[el] = np.sum(xi_vals * P_next) / (4*np.pi)
    P_prev = P_curr
    P_curr = P_next

print(f"  Done in {time.time()-t0:.1f}s")

# Also compute the MASTER prediction for the same subsample's geometry
# But we can't easily rebuild the coupling matrix for the subsample.
# Instead, compare the exact pair-sum with the full MASTER theory
# (scaled by (n_sub/Nskew)^2 for the different number of sightlines)
scale = (n_sub / Nskew)**2

# Note: the exact pair sum includes BOTH diagonal and off-diagonal.
# The diagonal is: (1/4pi) Sum_j xi_2D(0)
# xi_2D(0) = <w_j^2> = b1^2 N^2/L^3 * Sum_k P(k)
xi_0 = b1**2 * N**2 / L**3 * np.sum(Pk_vals)
diag_exact = n_sub * xi_0 / (4*np.pi)
print(f"\n  xi_2D(0) = {xi_0:.4e}")
print(f"  diag contribution (subsample) = {diag_exact:.4e}")
print(f"  diag_cl (full, expected) = {diag_cl:.4e}")
print(f"  Scale factor (n_sub/Nskew)^2 = {scale:.6f}")

# ================================================================== #
# 7. COMPARISON: exact pair sum vs forward model                      #
# ================================================================== #
print(f"\n{'='*80}")
print("7. COMPARISON: exact pair-sum vs MASTER forward model")
print("   (The pair sum is exact -- no Limber, no MASTER, no approximations)")
print(f"{'='*80}")

# The forward model at full Nskew: theory = M_clust @ C_true + diag_cl
# For the subsample, we expect: exact_cl ≈ theory * scale ... approximately
# (the coupling matrix depends on the exact positions, so this is a rough comparison)

# Better comparison: use the data mean (full 100 sims), and compare
# ratio(data/exact_pair_scaled) vs ratio(data/forward_model)
exact_scaled = exact_cl / scale  # scale up to full Nskew (approximate)

print(f"\n  {'ell':>6s} {'data':>12s} {'fwd(chi0)':>12s} {'exact_pair':>12s} "
      f"{'r(fwd)':>8s} {'r(exact)':>8s}")
for b in range(n_bins):
    lo = b * NperBin
    hi = (b + 1) * NperBin
    ell_c = (lo + hi - 1) / 2.0
    d_avg = np.mean(cl_mean[lo:hi])
    f_avg = np.mean(theory_chi0[lo:hi])
    e_avg = np.mean(exact_scaled[lo:hi])
    r_fwd = d_avg / f_avg if f_avg > 0 else 0
    r_exact = d_avg / e_avg if e_avg > 0 else 0
    print(f"  {ell_c:6.1f} {d_avg:12.4e} {f_avg:12.4e} {e_avg:12.4e} "
          f"{r_fwd:8.4f} {r_exact:8.4f}")

# ================================================================== #
# 8. SUMMARY TABLE: all approaches                                    #
# ================================================================== #
print(f"\n{'='*80}")
print("8. SUMMARY: mean ratio (data/theory) for each approach, ell 48-464")
print(f"{'='*80}")

approaches = {}

# a) chi_0
approaches['chi_0 = 5000'] = theory_chi0

# b) chi_bar
C_true_chibar = b1**2 * plin((ells_ext + 0.5) / chi_bar) / (L * chi_bar**2)
theory_chibar = (M_clust @ C_true_chibar)[:Nl] + diag_cl
approaches['chi_bar = 5691'] = theory_chibar

# c) Sightline avg
approaches['<P/r_j^2>_sightlines'] = theory_avg

# d) Pair avg
approaches['<P/r_pair^2>_pairs'] = theory_pair

# e) Mean r_j
chi_mean_r = np.mean(r_j)
C_true_mean_r = b1**2 * plin((ells_ext + 0.5) / chi_mean_r) / (L * chi_mean_r**2)
theory_mean_r = (M_clust @ C_true_mean_r)[:Nl] + diag_cl
approaches[f'<r_j> = {chi_mean_r:.0f}'] = theory_mean_r

# f) Harmonic mean of 1/r^2
chi_harm = 1.0/np.sqrt(np.mean(1.0/r_j**2))
C_true_harm = b1**2 * plin((ells_ext + 0.5) / chi_harm) / (L * chi_harm**2)
theory_harm = (M_clust @ C_true_harm)[:Nl] + diag_cl
approaches[f'<1/r^2>^-1/2 = {chi_harm:.0f}'] = theory_harm

for name, theory in approaches.items():
    ratios_test = []
    ell_ctrs = []
    for b in range(1, n_bins - 1):
        lo = b * NperBin
        hi = (b + 1) * NperBin
        ratios_test.append(np.mean(cl_mean[lo:hi]) / np.mean(theory[lo:hi]))
        ell_ctrs.append((lo + hi - 1) / 2.0)
    slope = np.polyfit(ell_ctrs, ratios_test, 1)[0]
    print(f"  {name:35s}: r = {np.mean(ratios_test):.5f} ± {np.std(ratios_test):.5f}  slope={slope:+.2e}")
