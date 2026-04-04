#!/usr/bin/env python
"""
Diagnostic: check the consistency of the direct pair-counting theory
vs the MASTER window-convolution theory, and trace the convergence
with lambda_max.
"""
import sys, os, gc, time
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

Nl = 500
chi_shift = 5000
num_qso = 9797
add_rsd_ = False

import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# ---- Load cache ----
datafile = os.path.join(root, "notebooks", "data",
                        "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
d = np.load(datafile)
cl_k_all = d['cl_k']
wl_k = d['wl_k']
N = int(d['Nk'])
L_box = float(d['L'])
Nskew = int(d['Nskew'])
num_sim = cl_k_all.shape[0]
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)

# ---- Cosmology ----
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

dchi = L_box / N
chi_bar = chi_shift + L_box / 2.0

# ---- Sightline positions ----
GRF_pos = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=1000)
all_x, all_y, all_z, _, _, _ = GRF_pos.process_skewers(Nskew=num_qso, shift=chi_shift)
theta_ref, phi_ref = GRF_pos.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
del GRF_pos, all_x, all_y, all_z; gc.collect()

nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
del nhat; gc.collect()

# ================================================================ #
# TEST 1: Verify PLKjKk = 4*pi * wl_ref  for lambda < Nl          #
# ================================================================ #
print("=== TEST 1: PLKjKk vs 4*pi*wl_ref ===")
PLKjKk_500 = sht_lya.legendre_polynomials_sum(Nl, cos_theta, KjKk)[:Nl]
ratio_PW = PLKjKk_500 / (4 * np.pi * wl_ref)
print(f"  PLKjKk[:10] = {PLKjKk_500[:10]}")
print(f"  4pi*wl[:10] = {4*np.pi*wl_ref[:10]}")
print(f"  ratio[:10]  = {ratio_PW[:10]}")
print(f"  ratio mean  = {np.mean(ratio_PW):.6f}")
print(f"  ratio std   = {np.std(ratio_PW):.6f}")
print(f"  ratio range = [{np.min(ratio_PW):.6f}, {np.max(ratio_PW):.6f}]")

# ================================================================ #
# TEST 2: MASTER convolution vs Direct approach at lambda_max=Nl   #
# ================================================================ #
print("\n=== TEST 2: MASTER vs Direct at lambda_max = Nl = 500 ===")
ells = np.arange(Nl, dtype=float)

# -- MASTER approach: Mll @ cl_true
# Build coupling matrix from window (wl_ref)
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()  # this IS M_{ell,ell'} in MASTER convention
cl_true = b1_ref**2 * plin_ref((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
pseudo_cl_master = coupling_wl @ cl_true  # M @ C_true = <tilde{C}_ell>

# -- Direct approach at lambda_max = 500
L_max_500 = Nl + Nl - 1  # = 999 (for triangle ineq)
L_range_500 = np.arange(L_max_500, dtype=float)
pk_L_500 = b1_ref**2 * plin_ref((L_range_500 + 0.5) / chi_bar)
couple_pk_500 = Wigner3j.CoupleMat(Nl, pk_L_500)
coupling_pk_500 = couple_pk_500.compute_matrix()
C_theory_direct_500 = coupling_pk_500 @ PLKjKk_500 / (4 * np.pi) / (2 * np.pi * chi_bar**2)
C_theory_direct_500 /= (4 * np.pi)**2  # "plot" convention

# Compare
ratios_master_direct = pseudo_cl_master / C_theory_direct_500
mask = (pseudo_cl_master > 0) & (C_theory_direct_500 > 0)
print(f"  ratio MASTER/Direct  mean (ell>1) = {np.mean(ratios_master_direct[2:]):.6f}")
print(f"  ratio MASTER/Direct  std  (ell>1) = {np.std(ratios_master_direct[2:]):.6f}")
print(f"  ratio range = [{np.min(ratios_master_direct[mask]):.6f}, {np.max(ratios_master_direct[mask]):.6f}]")

# Also compare with measured cl_mean
ratio_master_meas = pseudo_cl_master / cl_mean
ratio_direct_meas = C_theory_direct_500 / cl_mean
print(f"\n  MASTER  / measured: mean(ell>1) = {np.mean(ratio_master_meas[2:]):.4f}")
print(f"  Direct  / measured: mean(ell>1) = {np.mean(ratio_direct_meas[2:]):.4f}")

# Check specific ells
for ell in [10, 50, 100, 200, 400]:
    print(f"  ell={ell}: MASTER={pseudo_cl_master[ell]:.2e}, "
          f"Direct={C_theory_direct_500[ell]:.2e}, "
          f"Measured={cl_mean[ell]:.2e}, "
          f"M/D={pseudo_cl_master[ell]/C_theory_direct_500[ell]:.6f}")

# ================================================================ #
# TEST 3: Convergence with lambda_max                              #
# ================================================================ #
print("\n=== TEST 3: Convergence with lambda_max ===")
lambda_max_vals = [100, 200, 300, 500, 750, 1000, 1500, 2000]
max_lmax = max(lambda_max_vals)
print(f"Computing PLKjKk to lambda_max={max_lmax}...", flush=True)
t0 = time.time()
PLKjKk_full = sht_lya.legendre_polynomials_sum(max_lmax, cos_theta, KjKk)[:max_lmax]
print(f"  Done in {time.time()-t0:.1f}s")

# For specific ells, compute the cumulative theory sum
test_ells = [10, 50, 100, 200, 300, 400]

print(f"\n{'lmax':>6s}", end="")
for ell in test_ells:
    print(f" {'ell='+str(ell):>12s}", end="")
print(f"  {'mean(2:Nl)':>12s}")
print("-" * (8 + 13 * (len(test_ells) + 1)))

for lmax in lambda_max_vals:
    PLKjKk = PLKjKk_full[:lmax]
    L_max_needed = Nl + lmax - 1
    L_range = np.arange(L_max_needed, dtype=float)
    pk_L = b1_ref**2 * plin_ref((L_range + 0.5) / chi_bar)
    couple_pk = Wigner3j.CoupleMat(lmax, pk_L)
    coupling_pk = couple_pk.compute_matrix()
    C_theory = coupling_pk @ PLKjKk / (4 * np.pi) / (2 * np.pi * chi_bar**2)
    C_theory_plot = np.zeros(Nl)
    n = min(lmax, Nl)
    C_theory_plot[:n] = C_theory[:n] / (4 * np.pi)**2
    
    ratios = np.where(cl_mean > 0, C_theory_plot / cl_mean, 0.0)
    
    print(f"{lmax:6d}", end="")
    for ell in test_ells:
        print(f" {ratios[ell]:12.6f}", end="")
    print(f"  {np.mean(ratios[2:]):12.6f}")

# Also print the MASTER result for comparison
print(f"{'MASTER':>6s}", end="")
for ell in test_ells:
    print(f" {pseudo_cl_master[ell]/cl_mean[ell]:12.6f}", end="")
print(f"  {np.mean(pseudo_cl_master[2:]/cl_mean[2:]):12.6f}")

# ================================================================ #
# TEST 4: Check PLKjKk behavior at high lambda                    #
# ================================================================ #
print("\n=== TEST 4: PLKjKk at high lambda ===")
# PLKjKk[0] should be N^2 * Ns^2
Ns = Nskew
expected_P0 = N**2 * Ns**2
print(f"  PLKjKk[0]      = {PLKjKk_full[0]:.4e}")
print(f"  N^2 * Ns^2     = {expected_P0:.4e}")
print(f"  ratio          = {PLKjKk_full[0]/expected_P0:.6f}")

# Check the shot-noise level (Poisson expectation: N^2 * Ns for self-pairs)
shot_noise_level = N**2 * Ns
print(f"\n  N^2 * Ns (shot) = {shot_noise_level:.4e}")
for lam in [0, 10, 50, 100, 200, 500, 1000, 1500, 1999]:
    print(f"  PLKjKk[{lam:4d}]    = {PLKjKk_full[lam]:12.4e}  "
          f"(ratio to shot = {PLKjKk_full[lam]/shot_noise_level:.4f})")

print("\nDone!")
