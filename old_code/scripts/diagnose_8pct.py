#!/usr/bin/env python
"""
Separate the ell-dependent (clustered) and ell-independent (floor) contributions.

The data Cl = clustered_part(ell) + diagonal_constant
The theory:  M_clust @ C_true + W_floor * sigma^2(C_true)

Since diagonal_constant = Nskew * <w^2> / (4*pi) is known from discrete modes,
we can subtract it:
  Cl_off_diag = Cl_data - diag_cl
  
And check: Cl_off_diag =? M_clust @ C_true (for C_true = b1^2 P / (L chi^2))

But we can't measure the diagonal from data (it's constant in ell, 
mixed with ell-dependent terms). Instead, let's use the fact that
the monopole (ell=0) is dominated by the diagonal.

Actually, let's use the MEASURED cl_mean at DIFFERENT ell values.
The ell-independent part is the same for all ell, so ratio variations
tell us about the clustered part.

Better approach: compute the clustered theory per ell-bin AND the floor,
and check if the ell-dependent shape matches data.
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

# Discrete variance
kvals = np.fft.fftfreq(N, d=L/N) * (2*np.pi)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk = np.zeros_like(K_perp)
Pk[K_perp > 0] = plin(K_perp[K_perp > 0])
w2_discrete = b1**2 * N**2 / L**3 * np.sum(Pk)
diag_cl = Nskew * w2_discrete / (4*np.pi)

print(f"diag_cl = {diag_cl:.4e}")
print(f"cl_mean[100] = {cl_mean[100]:.4e}")
print(f"diag_cl is {diag_cl/cl_mean[100]*100:.1f}% of signal at ell=100")

# Compute clustered theory at Nl_large=2000
Nl_large = 2000
ells_ext = np.arange(Nl_large, dtype=float)
plin_vals = plin((ells_ext + 0.5)/chi_bar)

wl_needed = 2*Nl_large - 1
wl_raw = np.zeros(wl_needed)
na = min(wl_needed, len(wl_ext))
wl_raw[:na] = wl_ext[:na]
wl_raw[na:] = W_floor
wl_clust = wl_raw - W_floor

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

# Try different X values
for X_name, X_val in [("L", L), ("32pi3", 32*np.pi**3), ("fit1265", 1265.0)]:
    cl_true = b1**2 * plin_vals / (X_val * chi_bar**2)
    theory_clust = (M_clust @ cl_true)[:Nl]
    theory_total = theory_clust + diag_cl
    
    print(f"\n=== X = {X_name} ({X_val:.2f}) ===")
    print(f"{'ell':>6s} {'data':>12s} {'th_clust':>12s} {'diag':>10s} {'th_total':>12s} {'ratio':>8s}")
    NperBin = 50
    for i_start in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        i_end = min(i_start + NperBin, Nl)
        d_bin = np.mean(cl_mean[i_start:i_end])
        tc_bin = np.mean(theory_clust[i_start:i_end])
        tt_bin = np.mean(theory_total[i_start:i_end])
        ratio = d_bin / tt_bin if tt_bin > 0 else np.nan
        print(f"  {(i_start+i_end)//2:4d}  {d_bin:12.4e} {tc_bin:12.4e} {diag_cl:10.4e} {tt_bin:12.4e} {ratio:8.4f}")

# Now let me also try this: what if the 8% error is in the LIMBER C_true shape?
# I.e., C_true should not be P_lin(l/chi)/chi^2 but something with corrections.
# Let me fit X per ell-bin:
print(f"\n=== Per-ell-bin X fit ===")
for i_start in range(30, 450, 50):
    i_end = i_start + 50
    d_bin = np.mean(cl_mean[i_start:i_end])
    cl_unnorm = b1**2 * plin_vals / chi_bar**2  # C_true without 1/X
    tc_unnorm = np.mean((M_clust @ cl_unnorm)[:Nl][i_start:i_end])
    X_bin = (tc_unnorm + W_floor * w2_discrete / N**2) / (d_bin)
    # Actually: data = tc_unnorm/X + diag_cl
    # (d_bin - diag_cl) = tc_unnorm/X
    # X = tc_unnorm / (d_bin - diag_cl)
    offdiag_data = d_bin - diag_cl
    X_from_offdiag = tc_unnorm / offdiag_data if offdiag_data > 0 else np.nan
    print(f"  ell={i_start+25:4d}: X_total={X_bin:.2f}, X_offdiag={X_from_offdiag:.2f}, "
          f"offdiag_frac={offdiag_data/d_bin:.3f}")

# Key question: does X_offdiag vary with ell?
# If it's constant = L, then C_true = P/(L chi^2) is correct for the clustered part.
# If it varies, the Limber shape is wrong.
