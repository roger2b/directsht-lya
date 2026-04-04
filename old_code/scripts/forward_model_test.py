#!/usr/bin/env python
"""
Forward-model test: compute M_clust @ C_true + floor and compare directly with data.

This avoids the deconvolution entirely and just checks the forward model.
The deconvolution might have amplified errors because M_clust is not perfectly diagonal.

Approach:
 1. cl_data = cl_mean  (from 100-sim average)
 2. theory = M_clust @ C_true + diag_cl
 3. Check if theory matches cl_data per ell-bin
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
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2)
K_flat = K_perp.ravel()
Pk_flat = plin(np.where(K_flat > 0, K_flat, 1e-10))
Pk_flat[K_flat == 0] = 0
w2 = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2 / (4*np.pi)

print(f"N={N}, L={L:.2f}, chi_bar={chi_bar:.2f}, b1={b1:.4f}")
print(f"diag_cl = {diag_cl:.4e}")
print(f"cl_mean[100] = {cl_mean[100]:.4e}")
print(f"diag_cl / cl_mean[100] = {diag_cl/cl_mean[100]:.4f}")

# Build M_clust for different Nl_large, compute forward model
print(f"\n{'='*80}")
print(f"Forward model: theory = M_clust @ C_true + diag_cl")
print(f"  C_true = b1^2 P(l/chi) / (L chi^2)")
print(f"{'='*80}")

for Nl_large in [500, 1000, 2000, 3000]:
    t0 = time.time()
    
    ells_ext = np.arange(Nl_large, dtype=float)
    C_true = b1**2 * plin((ells_ext + 0.5)/chi_bar) / (L * chi_bar**2)
    
    wl_needed = 2 * Nl_large - 1
    wl_raw = np.zeros(wl_needed)
    n_avail = min(wl_needed, len(wl_ext))
    wl_raw[:n_avail] = wl_ext[:n_avail]
    wl_raw[n_avail:] = W_floor
    wl_clust = wl_raw - W_floor
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
    M_clust = couple.compute_matrix()
    
    theory_clust = (M_clust @ C_true)[:Nl]
    theory = theory_clust + diag_cl
    
    # Per-bin ratios
    NperBin = 32
    ells_arr = np.arange(Nl, dtype=float)
    n_bins = Nl // NperBin
    ratios = []
    for b in range(1, n_bins):
        lo = b * NperBin
        hi = (b+1) * NperBin
        r = np.mean(cl_mean[lo:hi]) / np.mean(theory[lo:hi])
        ratios.append(r)
    
    mean_r = np.mean(ratios)
    std_r = np.std(ratios)
    print(f"  Nl_large={Nl_large:5d}: mean r(data/theory) = {mean_r:.4f} ± {std_r:.4f}  [{time.time()-t0:.1f}s]")
    
    if Nl_large == 2000:
        print(f"\n  Per-bin detail (Nl_large={Nl_large}):")
        for b in range(n_bins):
            lo = b * NperBin
            hi = (b+1) * NperBin
            ell_c = (lo + hi - 1) / 2.0
            d_avg = np.mean(cl_mean[lo:hi])
            t_avg = np.mean(theory[lo:hi])
            tc_avg = np.mean(theory_clust[lo:hi])
            print(f"    ell={ell_c:5.1f}: data={d_avg:.4e}, theory={t_avg:.4e}, "
                  f"clust={tc_avg:.4e}, floor={diag_cl:.4e}, r={d_avg/t_avg:.4f}")
    
    del couple, M_clust; gc.collect()

# Now try with different X values
print(f"\n{'='*80}")
print(f"Scan X values: theory = M_clust @ [b1^2 P/(X chi^2)] + diag_cl")
print(f"{'='*80}")

Nl_large = 2000
ells_ext = np.arange(Nl_large, dtype=float)
P_over_chi2 = b1**2 * plin((ells_ext + 0.5)/chi_bar) / chi_bar**2

wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

theory_clust_unnorm = (M_clust @ P_over_chi2)[:Nl]

mask = np.ones(Nl, dtype=bool)
mask[:32] = False
mask[480:] = False

for X_label, X_val in [("L", L), ("32pi3", 32*np.pi**3), 
                         ("1265", 1265.0), ("L*0.971", L*0.9715),
                         ("L*0.886 (=1225)", L*0.886)]:
    theory = theory_clust_unnorm / X_val + diag_cl
    r_mean = np.mean(cl_mean[mask] / theory[mask])
    r_std = np.std(cl_mean[mask] / theory[mask])
    # Also try to find the BEST X by least-squares
    print(f"  X={X_label:15s} ({X_val:.2f}): mean r = {r_mean:.4f} ± {r_std:.4f}")

# Find best X analytically:
# cl_data = theory_clust/X + diag_cl
# cl_data - diag_cl = theory_clust/X  => X = theory_clust / (cl_data - diag_cl)
cl_clust = cl_mean - diag_cl
X_per_ell = theory_clust_unnorm / np.where(cl_clust > 0, cl_clust, 1e-30)
X_best = np.mean(X_per_ell[mask])
print(f"\n  X_best (mean over ell) = {X_best:.2f}")
print(f"  X_best/L = {X_best/L:.4f}")
print(f"  X_best/32pi3 = {X_best/(32*np.pi**3):.4f}")

# Also: weighted mean (by theory_clust)
weights = theory_clust_unnorm[mask]
X_best_wt = np.sum(weights * X_per_ell[mask]) / np.sum(weights)
print(f"  X_best_weighted = {X_best_wt:.2f} (X/L = {X_best_wt/L:.4f})")

# And: global fit (single X)
# minimize SUM [cl_data - theory_clust/X - diag_cl]^2
# d/dX = SUM 2 [cl_data - tc/X - dc] * tc/X^2 = 0
# => SUM (cl_data-dc)*tc/X^2 = SUM tc^2/X^3
# => X = SUM tc^2 / SUM (cl_data-dc)*tc
tc = theory_clust_unnorm[mask]
dc = cl_clust[mask]
X_lsq = np.sum(tc**2) / np.sum(dc * tc)
print(f"  X_lsq (least squares) = {X_lsq:.2f} (X/L = {X_lsq/L:.4f})")
