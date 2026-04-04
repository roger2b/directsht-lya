#!/usr/bin/env python
"""
Test MASTER convergence using the ACTUAL PLKjKk values (not shot-noise extrapolation).
PLKjKk was computed to lambda=2000, so we have actual wl up to lambda=1999.
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
N = int(d['Nk'])
L_box = float(d['L'])
Nskew = int(d['Nskew'])
wl_ref = d['wl_k'][0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)
print(f"Loaded {cl_k_all.shape[0]} sims, N={N}, Ns={Nskew}")

# ---- Load PLKjKk ----
PLKjKk_file = os.path.join(root, "notebooks", "data", "PLKjKk_lambda2000.npy")
PLKjKk_full = np.load(PLKjKk_file)
lambda_max_data = len(PLKjKk_full)
print(f"Loaded PLKjKk with {lambda_max_data} multipoles")

# Convert to wl: PLKjKk = 4*pi * wl
wl_full = PLKjKk_full / (4 * np.pi)

# Verify consistency with cached wl_ref
ratio_check = wl_full[:Nl] / wl_ref
print(f"wl_full[:500] vs wl_ref: mean ratio = {np.mean(ratio_check):.6f}, "
      f"max dev = {np.max(np.abs(ratio_check - 1)):.6f}")

# ---- Cosmology ----
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

chi_bar = chi_shift + L_box / 2.0
SN = N**2 * Nskew

# ================================================================== #
# Convergence test: MASTER at increasing Nl_large with ACTUAL wl      #
# ================================================================== #
print("\n=== MASTER convergence with ACTUAL wl ===")
# Max wl lambda available: 1999 (from PLKjKk_full)
# For MASTER at Nl_large, inner sum needs wl up to 2*(Nl_large-1).
# So max Nl_large = (lambda_max_data + 1) / 2 = 1000
# For Nl_large > 1000, the inner sum is incomplete for some (ell, ell') pairs.
# But for the MEASURED rows (ell < 500), inner sum needs wl up to ell + (Nl_large-1).
# For ell=499, wl needed up to 499 + Nl_large - 1. Available up to 1999.
# So Nl_large can go up to 1999 - 499 + 1 = 1501 for all measured rows.

Nl_large_list = [500, 600, 700, 800, 1000, 1200, 1500]
test_ells = [10, 50, 100, 200, 300, 400]

print(f"  (Using ACTUAL wl from PLKjKk, lambda_max_data={lambda_max_data})")
print(f"\n{'Nl_large':>8s}", end="")
for ell in test_ells:
    print(f" {'ell='+str(ell):>10s}", end="")
print(f" {'mean(2:Nl)':>11s} {'time':>6s}")
print("-" * (8 + 11 * len(test_ells) + 13 + 7))

for Nl_large in Nl_large_list:
    t0 = time.time()
    
    # Use ACTUAL wl values
    wl_needed = 2 * Nl_large - 1  # max lambda index for the coupling matrix
    wl_for_coupling = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_for_coupling[:n_avail] = wl_full[:n_avail]
    # For lambda >= lambda_max_data, use wl=0 (very conservative)
    
    # C_true at extended ell' range
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
    
    # Coupling matrix
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
    M = couple.compute_matrix()
    
    # Pseudo-Cl
    n_rows = min(Nl, Nl_large)
    pseudo_cl = (M @ cl_true_ext)[:n_rows]
    
    dt = time.time() - t0
    ratios = pseudo_cl / cl_mean[:n_rows]
    
    print(f"{Nl_large:8d}", end="")
    for ell in test_ells:
        if ell < n_rows:
            print(f" {ratios[ell]:10.4f}", end="")
        else:
            print(f" {'N/A':>10s}", end="")
    mean_r = np.mean(ratios[2:]) if n_rows > 2 else 0.0
    print(f" {mean_r:11.4f} {dt:6.1f}s")
    
    del couple, M
    gc.collect()

# ================================================================== #
# Also show the SHOT-NOISE extrapolation for comparison               #
# ================================================================== #
print("\n=== Comparison: shot-noise extrapolation ===")
wl_shot = SN / (4 * np.pi)

print(f"\n{'Nl_large':>8s}", end="")
for ell in test_ells:
    print(f" {'ell='+str(ell):>10s}", end="")
print(f" {'mean(2:Nl)':>11s}")
print("-" * (8 + 11 * len(test_ells) + 13))

for Nl_large in [500, 1000, 1500]:
    # Shot noise extrapolation
    wl_sn_ext = np.full(2 * Nl_large - 1, wl_shot)
    wl_sn_ext[:Nl] = wl_ref
    
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
    
    couple = Wigner3j.CoupleMat(Nl_large, wl_sn_ext)
    M = couple.compute_matrix()
    pseudo_cl = (M @ cl_true_ext)[:Nl]
    
    ratios = pseudo_cl / cl_mean
    print(f"{Nl_large:8d}", end="")
    for ell in test_ells:
        print(f" {ratios[ell]:10.4f}", end="")
    print(f" {np.mean(ratios[2:]):11.4f}")
    
    del couple, M; gc.collect()

# ================================================================== #
# PLKjKk behavior plot data                                          #
# ================================================================== #
print("\n=== PLKjKk / SN at selected lambdas ===")
lambdas = np.arange(lambda_max_data)
for lam in range(0, lambda_max_data, 100):
    print(f"  lambda={lam:4d}: PLKjKk/SN = {PLKjKk_full[lam]/SN:.4f}")

print(f"\nMean PLKjKk/SN for lambda=500..1999: {np.mean(PLKjKk_full[500:]/SN):.4f}")
print(f"Mean PLKjKk/SN for lambda=1000..1999: {np.mean(PLKjKk_full[1000:]/SN):.4f}")

print("\nDone!")
