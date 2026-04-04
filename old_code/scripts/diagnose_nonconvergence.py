#!/usr/bin/env python
"""
Diagnose WHY the MASTER sum doesn't converge: the shot-noise coupling
mixes power from ALL ell' into each measured ell, and C_true computed
from the *continuous* P_lin(k) never stops — but the *box* has no modes
beyond k_Nyquist = pi/Dx, so C_true should be zero there.

Strategy:
  1. Decompose M = M_signal + M_shot (signal: narrow coupling; shot: flat)
  2. Signal coupling converges fast (wl_signal decays within ~50 multipoles)
  3. Shot-noise contribution is analytic: wl_shot/(4pi) * SUM (2ell'+1) C_true[ell']
  4. Apply Nyquist cutoff to C_true: no box modes beyond k_Nyq
  5. Total = signal + shot → should match measured data

Also show the cumulative shot-noise sum to demonstrate convergence.
"""
import sys, os, gc, time
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

Nl = 500
chi_shift = 5000
add_rsd_ = False

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# ---- Load cache ----
datafile = os.path.join(root, "notebooks", "data",
                        "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
d = np.load(datafile)
cl_k_all = d['cl_k']
N = int(d['Nk']); L_box = float(d['L']); Nskew = int(d['Nskew'])
wl_ref = d['wl_k'][0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)

PLKjKk_full = np.load(os.path.join(root, "notebooks", "data", "PLKjKk_lambda2000.npy"))
wl_full = PLKjKk_full / (4 * np.pi)
lambda_max_data = len(PLKjKk_full)
print(f"Loaded {cl_k_all.shape[0]} sims, N={N}, Ns={Nskew}, L_box={L_box:.1f}")

# ---- Cosmology ----
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin; b1 = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

chi_bar = chi_shift + L_box / 2.0
Dx = L_box / N
k_Nyq = np.pi / Dx
ell_Nyq = int(k_Nyq * chi_bar)
SN = N**2 * Nskew
wl_poisson = SN / (4 * np.pi)

print(f"chi_bar={chi_bar:.1f}, Dx={Dx:.2f} Mpc/h")
print(f"k_Nyq = pi/Dx = {k_Nyq:.4f} h/Mpc")
print(f"ell_Nyq = k_Nyq * chi_bar = {ell_Nyq}")
print(f"SN = N^2*Ns = {SN:.4e}, wl_poisson = {wl_poisson:.4e}")

# ================================================================== #
# Step 1: Show WHY the naive MASTER sum diverges                      #
# ================================================================== #
print("\n=== Step 1: Why the sum diverges ===")
# C_true from continuous P_lin
ells_10k = np.arange(10001, dtype=float)
k_perp_10k = (ells_10k + 0.5) / chi_bar
cl_true_10k = b1**2 * plin_ref(k_perp_10k) / (32 * np.pi**3 * chi_bar**2)

# Shot-noise contribution: wl_poisson/(4pi) * SUM (2ell'+1) C_true[ell']
cumsum_nocut = np.cumsum((2 * ells_10k + 1) * cl_true_10k)
shot_nocut = wl_poisson / (4 * np.pi) * cumsum_nocut

# With Nyquist cutoff
cl_true_nyq = cl_true_10k.copy()
cl_true_nyq[k_perp_10k > k_Nyq] = 0.0
cumsum_nyq = np.cumsum((2 * ells_10k + 1) * cl_true_nyq)
shot_nyq = wl_poisson / (4 * np.pi) * cumsum_nyq

print(f"\nCumulative sum of (2ell'+1)*C_true  (NO cutoff vs Nyquist cutoff):")
for ell_max in [500, 1000, 1500, 2000, 3000, 5000, 6600, 8000, 10000]:
    idx = min(ell_max, 10000)
    print(f"  ell'_max={ell_max:6d}: no_cut={cumsum_nocut[idx]:.4e}  "
          f"nyq_cut={cumsum_nyq[idx]:.4e}  "
          f"shot_nocut={shot_nocut[idx]:.2e}  shot_nyq={shot_nyq[idx]:.2e}")

print(f"\n  Shot noise from NO cutoff keeps growing with ell'_max")
print(f"  Shot noise from Nyquist cutoff saturates at ell' ~ {ell_Nyq}")

# ================================================================== #
# Step 2: Signal + shot decomposition with Nyquist cutoff              #
# ================================================================== #
print("\n=== Step 2: Signal + shot noise decomposition ===")

# Signal window: wl_signal = wl_actual - wl_poisson
# For lambda < 2000, use data - Poisson level
# For lambda >= 2000, signal assumed 0
Nl_signal = 1000   # plenty for the narrow signal coupling
wl_sig_arr = np.zeros(2 * Nl_signal - 1)  # 1999 elements
n = min(lambda_max_data, 2 * Nl_signal - 1)
wl_sig_arr[:n] = wl_full[:n] - wl_poisson  # can be negative!

print(f"  wl_signal stats (lambda < {n}):")
print(f"    min = {wl_sig_arr[:n].min():.4e}")
print(f"    max = {wl_sig_arr[:n].max():.4e}")
print(f"    mean(lambda>500) = {np.mean(wl_sig_arr[500:n]):.4e}")

# C_true for signal coupling (Nl_signal elements, WITH Nyquist cutoff)
ells_sig = np.arange(Nl_signal, dtype=float)
k_sig = (ells_sig + 0.5) / chi_bar
cl_true_sig = b1**2 * plin_ref(k_sig) / (32 * np.pi**3 * chi_bar**2)
cl_true_sig[k_sig > k_Nyq] = 0.0  # Nyquist cutoff

# Signal coupling
t0 = time.time()
M_sig = Wigner3j.CoupleMat(Nl_signal, wl_sig_arr).compute_matrix()
pcl_signal = (M_sig @ cl_true_sig)[:Nl]
print(f"  Signal coupling computed in {time.time()-t0:.1f}s")

# Check signal convergence: also at Nl_signal=800
Nl_signal2 = 800
wl_sig_arr2 = np.zeros(2 * Nl_signal2 - 1)
n2 = min(lambda_max_data, 2 * Nl_signal2 - 1)
wl_sig_arr2[:n2] = wl_full[:n2] - wl_poisson
ells_sig2 = np.arange(Nl_signal2, dtype=float)
k_sig2 = (ells_sig2 + 0.5) / chi_bar
cl_true_sig2 = b1**2 * plin_ref(k_sig2) / (32 * np.pi**3 * chi_bar**2)
cl_true_sig2[k_sig2 > k_Nyq] = 0.0
M_sig2 = Wigner3j.CoupleMat(Nl_signal2, wl_sig_arr2).compute_matrix()
pcl_signal2 = (M_sig2 @ cl_true_sig2)[:Nl]

print(f"\n  Signal convergence (Nl_sig=800 vs 1000):")
for ell in [10, 50, 100, 200, 300, 400]:
    frac_diff = (pcl_signal[ell] - pcl_signal2[ell]) / pcl_signal[ell]
    print(f"    ell={ell:3d}: Nl=800={pcl_signal2[ell]:.2e}  "
          f"Nl=1000={pcl_signal[ell]:.2e}  diff={frac_diff:+.4f}")

# Shot noise contribution (analytical, with Nyquist cutoff)
# M_shot[ell,ell'] = (2ell'+1) * wl_poisson / (4pi) from 3j orthogonality
# C_shot[ell] = wl_poisson/(4pi) * SUM_{ell'=0}^{ell_Nyq} (2ell'+1) C_true[ell']
ells_nyq = np.arange(ell_Nyq + 1, dtype=float)
k_nyq_arr = (ells_nyq + 0.5) / chi_bar
cl_true_full = b1**2 * plin_ref(k_nyq_arr) / (32 * np.pi**3 * chi_bar**2)
cl_true_full[k_nyq_arr > k_Nyq] = 0.0
total_variance_nyq = np.sum((2 * ells_nyq + 1) * cl_true_full)
pcl_shot = wl_poisson / (4 * np.pi) * total_variance_nyq  # constant for all ell

print(f"\n  Shot noise (with Nyquist cutoff, ell'_max={ell_Nyq}):")
print(f"    total angular variance = {total_variance_nyq:.4e}")
print(f"    pcl_shot = {pcl_shot:.4e} (constant for all ell)")

# Total theory
pcl_total = pcl_signal + pcl_shot

# Compare to data
print(f"\n  {'ell':>6s} {'signal':>12s} {'shot':>12s} {'total':>12s} "
      f"{'data':>12s} {'tot/data':>10s}")
print(f"  {'-'*66}")
for ell in [10, 50, 100, 200, 300, 400, 490]:
    s = pcl_signal[ell]; n = pcl_shot; t = s + n; d = cl_mean[ell]
    print(f"  {ell:6d} {s:12.2e} {n:12.2e} {t:12.2e} {d:12.2e} {t/d:10.4f}")

mean_ratio = np.mean((pcl_total / cl_mean)[2:])
print(f"\n  Mean total/data (ell>=2): {mean_ratio:.4f}")

# ================================================================== #
# Step 3: Compare with naive MASTER (no cutoff)                       #
# ================================================================== #
print("\n=== Step 3: Naive MASTER vs signal+shot (Nyquist cutoff) ===")
# Naive MASTER at several Nl_large values
Nl_test = [500, 1000, 1200, 1500]
for Nl_large in Nl_test:
    wl_needed = 2 * Nl_large - 1
    wl_ext = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_ext[:n_avail] = wl_full[:n_avail]
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext = b1**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
    M = Wigner3j.CoupleMat(Nl_large, wl_ext).compute_matrix()
    pcl_naive = (M @ cl_true_ext)[:Nl]
    r = np.mean((pcl_naive / cl_mean)[2:])
    print(f"  Nl_large={Nl_large:5d}: naive mean ratio = {r:.4f}")
    del M; gc.collect()

print(f"  signal+shot (Nyquist cutoff): mean ratio = {mean_ratio:.4f}")

# ================================================================== #
# Step 4: Does Nyquist cutoff matter? Show no-cutoff version           #
# ================================================================== #
print("\n=== Step 4: Effect of Nyquist cutoff ===")
# Shot noise WITHOUT cutoff to ell' = 10000
total_variance_nocut = float(cumsum_nocut[-1])  # ell'_max = 10000
pcl_shot_nocut = wl_poisson / (4 * np.pi) * total_variance_nocut

pcl_total_nocut = pcl_signal + pcl_shot_nocut
mean_ratio_nocut = np.mean((pcl_total_nocut / cl_mean)[2:])

print(f"  Shot with Nyquist cutoff   (ell'<{ell_Nyq}): {pcl_shot:.4e}")
print(f"  Shot without cutoff (ell'<10001): {pcl_shot_nocut:.4e}")
print(f"  Ratio: {pcl_shot_nocut/pcl_shot:.4f}")
print(f"\n  signal+shot(Nyq)    / data: {mean_ratio:.4f}")
print(f"  signal+shot(no cut) / data: {mean_ratio_nocut:.4f}")

# ================================================================== #
# Step 5: Sensitivity to wl_poisson (actual vs Poisson)               #
# ================================================================== #
print("\n=== Step 5: Sensitivity to shot-noise level ===")
# Actual mean of wl at high lambda
wl_actual_high = np.mean(wl_full[1000:])
ratio_actual_poisson = wl_actual_high / wl_poisson
print(f"  wl(actual high-lambda mean) = {wl_actual_high:.4e}")
print(f"  wl(Poisson)                 = {wl_poisson:.4e}")
print(f"  ratio actual/Poisson        = {ratio_actual_poisson:.4f}")

# Using actual mean instead of Poisson
pcl_shot_actual = wl_actual_high / (4 * np.pi) * total_variance_nyq
pcl_total_actual = pcl_signal + pcl_shot_actual
# But signal was computed with wl_signal = wl - wl_poisson
# If we use wl_actual, signal should use wl_signal = wl - wl_actual
# Let's recompute for consistency

# Redo with actual mean:
wl_sig_arr_v2 = np.zeros(2 * Nl_signal - 1)
n_v2 = min(lambda_max_data, 2 * Nl_signal - 1)
wl_sig_arr_v2[:n_v2] = wl_full[:n_v2] - wl_actual_high
M_sig_v2 = Wigner3j.CoupleMat(Nl_signal, wl_sig_arr_v2).compute_matrix()
# C_true same (with Nyq cutoff)
pcl_signal_v2 = (M_sig_v2 @ cl_true_sig)[:Nl]
pcl_shot_v2 = wl_actual_high / (4 * np.pi) * total_variance_nyq
pcl_total_v2 = pcl_signal_v2 + pcl_shot_v2

mean_ratio_v2 = np.mean((pcl_total_v2 / cl_mean)[2:])
print(f"\n  With Poisson shot noise:        mean ratio = {mean_ratio:.4f}")
print(f"  With actual high-lam shot noise: mean ratio = {mean_ratio_v2:.4f}")

print(f"\n  {'ell':>6s} {'Poisson':>10s} {'actual':>10s} {'data':>10s}")
for ell in [10, 50, 100, 200, 300, 400]:
    r1 = pcl_total[ell] / cl_mean[ell]
    r2 = pcl_total_v2[ell] / cl_mean[ell]
    print(f"  {ell:6d} {r1:10.4f} {r2:10.4f} {cl_mean[ell]:10.2e}")

# ================================================================== #
# Step 6: Cumulative convergence curves                               #
# ================================================================== #
print("\n=== Step 6: Cumulative shot-noise convergence at specific ells ===")
print(f"  Showing shot_contribution(ell'_max) / data for ell=10,100,400")
cum_shot = wl_poisson / (4 * np.pi) * cumsum_nyq
cum_shot_full = wl_poisson / (4 * np.pi) * cumsum_nocut

for ell_check in [10, 100, 400]:
    print(f"\n  ell = {ell_check}:")
    sig_val = pcl_signal[ell_check]
    dat_val = cl_mean[ell_check]
    for ell_max in [500, 1000, 2000, 3000, 5000, ell_Nyq, 8000, 10000]:
        idx = min(ell_max, 10000)
        total_nyq = sig_val + cum_shot[min(idx, ell_Nyq)]
        total_nocut = sig_val + cum_shot_full[idx]
        print(f"    ell'_max={ell_max:6d}: "
              f"nyq_tot/data={total_nyq/dat_val:.4f}  "
              f"nocut_tot/data={total_nocut/dat_val:.4f}")

print("\nDone!")
