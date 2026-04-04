#!/usr/bin/env python
"""
Diagnose the ~7% offset in the forward model.

Key insight: angular positions (theta, phi) are computed from the FIRST pixel
of each sightline, at chi_0 = 5000 Mpc/h. But the theory uses chi_bar = 5691.

For the k_z=0 DFT mode, the sum along the LOS projects out the n_z=0 Fourier
slice. This 2D field has fixed physical transverse separations Delta_r.
The angular separations are gamma_jk = Delta_r / chi_0 (not chi_bar !).
So C_true should use chi_0, not chi_bar.

This script tests:
  1. chi_0 vs chi_bar vs chi_mid vs chi_eff (Limber-averaged)
  2. ell vs ell+0.5 in the Limber argument
  3. Integral constraint (monopole removal)
  4. Discrete 2D modes vs continuous P(k)
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# ================================================================== #
# Load data                                                           #
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
chi_0 = 5000.0                      # start of sightline (angular positions!)
chi_bar = chi_0 + L / 2.0           # midpoint
chi_end = chi_0 + L                 # end of sightline
print(f"Box: N={N}, L={L:.2f}, dchi={dchi:.4f}")
print(f"chi_0={chi_0:.1f}, chi_bar={chi_bar:.2f}, chi_end={chi_end:.2f}")
print(f"L/chi_0 = {L/chi_0:.4f}  (NOT a thin shell!)")
print(f"b1={b1:.4f}, Nskew={Nskew}")

# ================================================================== #
# Floor and diagonal (same as before)                                 #
# ================================================================== #
W_floor = N**2 * Nskew / (4*np.pi)

kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2 = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2 / (4*np.pi)

print(f"\ndiag_cl = {diag_cl:.4e}")
print(f"cl_mean[100] = {cl_mean[100]:.4e}")
print(f"diag_cl / cl_mean[100] = {diag_cl/cl_mean[100]:.4f}")

# ================================================================== #
# Build floor-subtracted coupling matrix at Nl_large=2000            #
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
# Test 1: Different chi values                                        #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 1: Forward model with different chi values")
print(f"  C_true(chi) = b1^2 P((ell+0.5)/chi) / (L chi^2)")
print(f"  theory = M_clust @ C_true + diag_cl")
print(f"{'='*80}")

NperBin = 32
ells_arr = np.arange(Nl, dtype=float)
mask = np.ones(Nl, dtype=bool)
mask[:NperBin] = False  # skip first bin
mask[Nl-NperBin:] = False  # skip last bin

chi_values = {
    'chi_0 (start)': chi_0,
    'chi_bar (mid) ': chi_bar,
    'chi_end (end) ': chi_end,
    'chi_0 + L/3   ': chi_0 + L/3.0,
}

ells_ext = np.arange(Nl_large, dtype=float)

for label, chi in chi_values.items():
    C_true = b1**2 * plin((ells_ext + 0.5)/chi) / (L * chi**2)
    theory_clust = (M_clust @ C_true)[:Nl]
    theory = theory_clust + diag_cl

    # Per-bin ratios
    n_bins = Nl // NperBin
    ratios = []
    ell_ctrs = []
    for b in range(1, n_bins - 1):
        lo = b * NperBin
        hi = (b + 1) * NperBin
        r = np.mean(cl_mean[lo:hi]) / np.mean(theory[lo:hi])
        ratios.append(r)
        ell_ctrs.append((lo + hi - 1) / 2.0)

    mean_r = np.mean(ratios)
    std_r = np.std(ratios)
    # Also check ell-slope of ratio
    slope = np.polyfit(ell_ctrs, ratios, 1)[0]
    print(f"  chi={chi:8.1f} ({label}): mean r = {mean_r:.5f} ± {std_r:.5f}  "
          f"slope = {slope:.2e}/ell")

# ================================================================== #
# Test 2: ell vs ell+0.5 vs ell+1                                    #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 2: Limber argument (ell vs ell+0.5 vs ell+1)")
print(f"{'='*80}")

for ell_shift_label, ell_shift in [('ell', 0.0), ('ell+0.5', 0.5), ('ell+1', 1.0)]:
    for chi_label, chi in [('chi_0', chi_0), ('chi_bar', chi_bar)]:
        C_true = b1**2 * plin((ells_ext + ell_shift)/chi) / (L * chi**2)
        theory = (M_clust @ C_true)[:Nl] + diag_cl
        ratios = []
        for b in range(1, n_bins - 1):
            lo = b * NperBin
            hi = (b + 1) * NperBin
            ratios.append(np.mean(cl_mean[lo:hi]) / np.mean(theory[lo:hi]))
        print(f"  {ell_shift_label:6s} at {chi_label:8s}: "
              f"mean r = {np.mean(ratios):.5f} ± {np.std(ratios):.5f}")

# ================================================================== #
# Test 3: Discrete 2D k-modes vs smooth P(ell/chi)                   #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 3: Discrete 2D modes vs continuous C_true")
print("  The 2D box has modes at k = 2*pi*n/L, n=0,1,2,...")
print("  Mapping to ell via ell = k * chi gives ell_n = 2*pi*n*chi/L")
print(f"{'='*80}")

# Compute the discrete 2D power spectrum
kvals_pos = np.fft.fftfreq(N, d=L/N)  # in 1/Mpc (physical freq)
kvals_pos = kvals_pos * 2 * np.pi      # convert to k in h/Mpc
# All 2D k-modes
kx2d, ky2d = np.meshgrid(kvals_pos, kvals_pos)
k_perp_all = np.sqrt(kx2d**2 + ky2d**2).ravel()

for chi_label, chi in [('chi_0', chi_0), ('chi_bar', chi_bar)]:
    # Map each 2D k-mode to ell
    ell_modes = k_perp_all * chi  # ell = k_perp * chi
    
    # Build discrete C_true by binning modes
    C_true_discrete = np.zeros(Nl_large)
    for L_idx in range(Nl_large):
        # Count modes in [L-0.5, L+0.5)
        in_bin = (ell_modes >= L_idx - 0.5) & (ell_modes < L_idx + 0.5)
        if np.sum(in_bin) > 0:
            pk_in_bin = plin(np.where(k_perp_all[in_bin] > 0,
                                       k_perp_all[in_bin], 1e-10))
            pk_in_bin[k_perp_all[in_bin] == 0] = 0
            C_true_discrete[L_idx] = b1**2 * np.mean(pk_in_bin) / (L * chi**2)

    # Smooth (Limber) C_true
    C_true_smooth = b1**2 * plin((np.arange(Nl_large) + 0.5)/chi) / (L * chi**2)

    # Forward models
    th_smooth = (M_clust @ C_true_smooth)[:Nl] + diag_cl
    th_discrete = (M_clust @ C_true_discrete)[:Nl] + diag_cl

    ratios_s, ratios_d = [], []
    for b in range(1, n_bins - 1):
        lo = b * NperBin
        hi = (b + 1) * NperBin
        ratios_s.append(np.mean(cl_mean[lo:hi]) / np.mean(th_smooth[lo:hi]))
        if np.mean(th_discrete[lo:hi]) > 0:
            ratios_d.append(np.mean(cl_mean[lo:hi]) / np.mean(th_discrete[lo:hi]))

    print(f"  {chi_label}: smooth  r = {np.mean(ratios_s):.5f} ± {np.std(ratios_s):.5f}")
    if ratios_d:
        print(f"  {chi_label}: discrete r = {np.mean(ratios_d):.5f} ± {np.std(ratios_d):.5f}")

# ================================================================== #
# Test 4: Integral constraint — does k_perp=0 mode matter?           #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 4: Integral constraint")
print("  The k_perp=0 mode contributes only to ell=0 (monopole).")
print("  Check: fraction of sigma^2 from k_perp < k_fund")
print(f"{'='*80}")

k_fund = 2*np.pi/L
print(f"  k_fundamental = 2pi/L = {k_fund:.5f} h/Mpc")
print(f"  ell_fund(chi_0) = k_fund * chi_0 = {k_fund*chi_0:.1f}")
print(f"  ell_fund(chi_bar) = k_fund * chi_bar = {k_fund*chi_bar:.1f}")

# How many 2D modes below k_fund?
n_below = np.sum(k_perp_all[(k_perp_all > 0) & (k_perp_all < k_fund)])
print(f"  Number of 2D modes with 0 < k < k_fund: {np.sum((k_perp_all > 0) & (k_perp_all < k_fund))}")
power_kperp0 = plin(1e-10) * 0  # k_perp=0 mode has zero power if we exclude it
print(f"  The k_perp=0 (DC) mode is excluded by D-R subtraction: no integral constraint effect at ell>0")

# ================================================================== #
# Test 5: Pixel window function                                       #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 5: Pixel window function")
print("  DirectSHT: exact a_lm = sum_j w_j Y_lm(n_j)")
print("  -> No pixel window. The sightlines are point sources.")
print("  -> NO HEALPix involved (no nside, no pixelization).")
print("  -> Pixel window is UNITY. No correction needed.")
print(f"{'='*80}")

# ================================================================== #
# Test 6: Best-fit chi (directly from data)                          #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 6: Find the best-fit chi that minimizes |data - theory|")
print(f"{'='*80}")

chi_scan = np.linspace(chi_0 - 200, chi_end, 100)
residuals = []
for chi in chi_scan:
    C_true = b1**2 * plin((ells_ext + 0.5)/chi) / (L * chi**2)
    theory = (M_clust @ C_true)[:Nl] + diag_cl
    # Chi-squared (inverse-variance would be better but this is a quick scan)
    resid = np.sum((cl_mean[mask] - theory[mask])**2/theory[mask]**2)
    residuals.append(resid)

best_idx = np.argmin(residuals)
chi_best = chi_scan[best_idx]
print(f"  Best-fit chi = {chi_best:.1f}")
print(f"  chi_0 = {chi_0:.1f}, chi_bar = {chi_bar:.1f}")
print(f"  chi_best / chi_0 = {chi_best/chi_0:.4f}")
print(f"  chi_best / chi_bar = {chi_best/chi_bar:.4f}")
print(f"  chi_best - chi_0 = {chi_best - chi_0:.1f}")

# Show the ratio at chi_best
C_true = b1**2 * plin((ells_ext + 0.5)/chi_best) / (L * chi_best**2)
theory = (M_clust @ C_true)[:Nl] + diag_cl
ratios = []
for b in range(1, n_bins - 1):
    lo = b * NperBin
    hi = (b + 1) * NperBin
    ratios.append(np.mean(cl_mean[lo:hi]) / np.mean(theory[lo:hi]))
print(f"  At chi_best: mean r = {np.mean(ratios):.5f} ± {np.std(ratios):.5f}")

# ================================================================== #
# Test 7: Per-bin detail at chi_0 vs chi_bar                         #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 7: Per-bin ratio at chi_0 vs chi_bar")
print(f"{'ell':>6s}  {'r(chi_0)':>10s}  {'r(chi_bar)':>10s}  {'r(chi_best)':>10s}")
print(f"{'='*80}")

for chi_label, chi in [('chi_0', chi_0), ('chi_bar', chi_bar), ('chi_best', chi_best)]:
    _ = 0  # placeholder

theories_test = {}
for label, chi in [('chi_0', chi_0), ('chi_bar', chi_bar), ('chi_best', chi_best)]:
    C_true = b1**2 * plin((ells_ext + 0.5)/chi) / (L * chi**2)
    theories_test[label] = (M_clust @ C_true)[:Nl] + diag_cl

for b in range(n_bins):
    lo = b * NperBin
    hi = (b + 1) * NperBin
    ell_c = (lo + hi - 1) / 2.0
    d_avg = np.mean(cl_mean[lo:hi])
    r0 = d_avg / np.mean(theories_test['chi_0'][lo:hi])
    rb = d_avg / np.mean(theories_test['chi_bar'][lo:hi])
    rB = d_avg / np.mean(theories_test['chi_best'][lo:hi])
    print(f"{ell_c:6.1f}  {r0:10.5f}  {rb:10.5f}  {rB:10.5f}")

# ================================================================== #
# Test 8: Verify angular positions use chi_0                         #
# ================================================================== #
print(f"\n{'='*80}")
print("TEST 8: Verify angular positions are at chi_0")
print("  process_skewers: all_x = tmp_all_z = coords[2] + shift")
print("  coords[2] from linspace(0, L, N) -> first pixel at z=0")
print("  all_x[:,0] = 0 + 5000 = 5000 = chi_0")
print("  compute_theta_phi_skewer_start uses (all_x[:,0], all_y[:,0], all_z[:,0])")
print("  -> angles are at chi_0, not chi_bar!")
print(f"{'='*80}")

# Compute actual chi_0 from the code
coords_z = np.linspace(0, L, N)
chi_grid = coords_z + 5000.0
print(f"  chi_grid[0] = {chi_grid[0]:.1f}")
print(f"  chi_grid[-1] = {chi_grid[-1]:.2f}")
print(f"  chi_bar from code: {(chi_grid.max() + chi_grid.min())/2:.2f}")

# What chi does the angular patch correspond to?
# Patch size = L, at chi_0 = 5000 -> angular_size = L/chi_0
print(f"\n  Patch angular size: {np.degrees(L/chi_0):.2f} deg (at chi_0)")
print(f"  Patch angular size: {np.degrees(L/chi_bar):.2f} deg (at chi_bar)")

# For the PAIR SEPARATIONS to map correctly, we need chi_0:
# gamma_jk = Delta_r / chi_0 (since theta, phi computed at chi_0)
# C_ell = P_2D(ell/chi_0) / chi_0^2

# ================================================================== #
# Summary                                                             #
# ================================================================== #
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"  The angular positions are computed at chi_0 = {chi_0:.0f}")
print(f"  The current theory uses chi_bar = {chi_bar:.1f}")
print(f"  For the k_z=0 DFT mode:")
print(f"    w_j = sum_alpha delta(n_j, chi_alpha) -> projects out kz=0 2D field")
print(f"    This 2D field has FIXED physical transverse separations")
print(f"    angular gamma_jk = Delta_r / chi_0 (positions at chi_0)")
print(f"    -> C_true should use chi_0, not chi_bar")
print(f"  Best-fit chi = {chi_best:.0f}")
