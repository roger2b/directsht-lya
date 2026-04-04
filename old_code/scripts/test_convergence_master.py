#!/usr/bin/env python
"""
Test convergence of the MASTER approach by increasing Nl_large
(the size of the coupling matrix), using the shot-noise extrapolation
for wl at lambda > 500.

The MASTER formula is: <pseudo_Cl_ell> = SUM_{ell'=0}^{Nl_large-1} M[ell,ell'] C_true[ell']

If this converges to the measured data, the non-convergence issue is solved.
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
wl_k = d['wl_k']
N = int(d['Nk'])
L_box = float(d['L'])
Nskew = int(d['Nskew'])
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)
print(f"Loaded {cl_k_all.shape[0]} sims, N={N}, Ns={Nskew}")

# ---- Cosmology ----
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

chi_bar = chi_shift + L_box / 2.0

# Shot noise level
wl_shot = N**2 * Nskew / (4 * np.pi)
print(f"chi_bar={chi_bar:.1f}, wl_shot={wl_shot:.4e}")

# ================================================================== #
# Convergence test: MASTER at increasing Nl_large                     #
# ================================================================== #
Nl_large_list = [500, 600, 700, 800, 1000, 1500, 2000, 3000]
test_ells = [10, 50, 100, 200, 300, 400]

print(f"\n{'Nl_large':>8s}", end="")
for ell in test_ells:
    print(f" {'ell='+str(ell):>10s}", end="")
print(f" {'mean(2:Nl)':>11s} {'time':>6s}")
print("-" * (8 + 11 * len(test_ells) + 13 + 7))

for Nl_large in Nl_large_list:
    t0 = time.time()
    
    # Extend wl: use cached values for lambda < 500, shot noise for lambda >= 500
    wl_for_coupling = np.full(2 * Nl_large - 1, wl_shot)  # all shot noise
    n_cached = min(Nl, Nl_large)
    wl_for_coupling[:n_cached] = wl_ref[:n_cached]  # replace with actual values where available
    
    # C_true at extended ell' range
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
    
    # Coupling matrix
    couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
    M = couple.compute_matrix()
    
    # Pseudo-Cl = M @ C_true, but only keep first Nl rows
    n_rows = min(Nl, Nl_large)
    pseudo_cl = (M @ cl_true_ext)[:n_rows]
    
    dt = time.time() - t0
    
    # Ratios
    ratios = pseudo_cl / cl_mean[:n_rows]
    print(f"{Nl_large:8d}", end="")
    for ell in test_ells:
        if ell < n_rows:
            print(f" {ratios[ell]:10.4f}", end="")
        else:
            print(f" {'N/A':>10s}", end="")
    mean_r = np.mean(ratios[2:]) if n_rows > 2 else 0.0
    print(f" {mean_r:11.4f} {dt:6.1f}s")
    
    del couple, M, cl_true_ext, ells_ext, wl_for_coupling
    gc.collect()

# ================================================================== #
# Check: what fraction does extending wl_signal vs extending ell'     #
# contribute?                                                         #
# ================================================================== #
print("\n=== Decomposition at Nl_large=1000 ===")
Nl_large = 1000
ells_ext = np.arange(Nl_large, dtype=float)
cl_true_ext2 = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)

# Full window (signal + shot)
wl_full = np.full(2 * Nl_large - 1, wl_shot)
wl_full[:Nl] = wl_ref
c_full = Wigner3j.CoupleMat(Nl_large, wl_full)
M_full = c_full.compute_matrix()
pcl_full = (M_full @ cl_true_ext2)[:Nl]

# Signal window only (for convergence comparison)  
wl_sig = np.zeros(2 * Nl_large - 1)
wl_sig[:Nl] = wl_ref - wl_shot  # signal part of cached values
# For lambda >= 500, wl_signal assumed 0 (approx)
c_sig = Wigner3j.CoupleMat(Nl_large, wl_sig)
M_sig = c_sig.compute_matrix()
pcl_sig = (M_sig @ cl_true_ext2)[:Nl]

# Shot noise part
wl_sn = np.full(2 * Nl_large - 1, wl_shot)
c_sn = Wigner3j.CoupleMat(Nl_large, wl_sn)
M_sn = c_sn.compute_matrix()
pcl_sn = (M_sn @ cl_true_ext2)[:Nl]

print(f"  Full   / data: mean = {np.mean(pcl_full[2:]/cl_mean[2:]):.4f}")
print(f"  Signal / data: mean = {np.mean(pcl_sig[2:]/cl_mean[2:]):.4f}")
print(f"  Shot   / data: mean = {np.mean(pcl_sn[2:]/cl_mean[2:]):.4f}")
print(f"  Sig+SN / data: mean = {np.mean((pcl_sig+pcl_sn)[2:]/cl_mean[2:]):.4f}")

print(f"\n  {'ell':>6s} {'full/data':>10s} {'sig/data':>10s} {'sn/data':>10s}")
for ell in test_ells:
    rf = pcl_full[ell] / cl_mean[ell]
    rs = pcl_sig[ell] / cl_mean[ell]
    rn = pcl_sn[ell] / cl_mean[ell]
    print(f"  {ell:6d} {rf:10.4f} {rs:10.4f} {rn:10.4f}")

print("\nDone!")
