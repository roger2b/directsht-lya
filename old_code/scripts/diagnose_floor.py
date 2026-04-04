#!/usr/bin/env python
"""
Compare diag_cl (pair-counting diagonal) vs the MASTER-consistent floor.

The floor-subtracted MASTER uses:
  theory = M_clust @ C_true + diag_cl

But the correct decomposition is:
  theory = M_clust @ C_true + floor_master

where floor_master = W_floor/(4pi) * sum_L (2L+1) C_true(L), which is
the ℓ-independent floor contribution from M_floor @ C_true.

The mismatch diag_cl != floor_master causes the first bin to be off.
"""
import sys, os, gc
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

Nl = 500
NperBin = 32
chi_shift = 5000
num_qso = 9797
add_rsd_ = False

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Load data
datafile = os.path.join(root, "notebooks", "data",
                        "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
d = np.load(datafile)
cl_k_all = d['cl_k']
wl_k = d['wl_k']
Nskew = int(d['Nskew'])
N = int(d['Nk'])
L_box = float(d['L'])
num_sim = cl_k_all.shape[0]

wl_ref = wl_k[0, :Nl]

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

chi_0 = float(chi_shift)
np.random.seed(100)
_inds = np.unique(np.random.randint(0, N, size=(num_qso, 2)), axis=0)
_coords_grid = np.linspace(0, L_box, N)
_r_j = np.sqrt(chi_0**2 + _coords_grid[_inds[:,1]]**2 + _coords_grid[_inds[:,0]]**2)
chi_eff = np.mean(_r_j)
del _inds, _coords_grid, _r_j

# Load PLKjKk
PLKjKk_full = np.load(os.path.join(root, "notebooks", "data", "PLKjKk_lambda4000.npy"))
wl_full = PLKjKk_full / (4 * np.pi)
lambda_max_data = len(PLKjKk_full)

# Floor
W_floor = N**2 * Nskew / (4 * np.pi)

# diag_cl from pair-counting
kvals = np.fft.fftfreq(N, d=1.0) * (2 * np.pi * N / L_box)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin_ref(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2_theory = b1_ref**2 * N**2 / L_box**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2_theory / (4 * np.pi)
print(f"diag_cl (pair-counting) = {diag_cl:.4e}")

# MASTER-consistent floor for different Nl_large
print(f"\n{'Nl_large':>10s} {'floor_master':>14s} {'diag_cl':>14s} {'ratio':>8s} {'diff':>14s}")
print("-" * 65)
for Nl_large in [500, 1000, 2000, 3000, 5000, 10000, 20000, 50000]:
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_eff) / (L_box * chi_eff**2)
    floor_master = W_floor / (4 * np.pi) * np.sum((2 * ells_ext + 1) * cl_true_ext)
    ratio = floor_master / diag_cl
    diff = diag_cl - floor_master
    print(f"{Nl_large:10d} {floor_master:14.4e} {diag_cl:14.4e} {ratio:8.4f} {diff:14.4e}")

# Try also with chi_0 instead of chi_eff
print(f"\nWith chi_0={chi_0} instead of chi_eff={chi_eff:.1f}:")
for Nl_large in [2000, 5000, 10000, 50000]:
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext_0 = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_0) / (L_box * chi_0**2)
    floor_master_0 = W_floor / (4 * np.pi) * np.sum((2 * ells_ext + 1) * cl_true_ext_0)
    ratio_0 = floor_master_0 / diag_cl
    print(f"  Nl_large={Nl_large}: floor_master={floor_master_0:.4e}, ratio={ratio_0:.4f}")

# ---- Now compute theory with floor_master instead of diag_cl ----
print(f"\n\n=== Theory with MASTER-consistent floor vs diag_cl ===")
Nl_large = 2000

wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, lambda_max_data)
wl_raw[:n_avail] = wl_full[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

ells_ext = np.arange(Nl_large, dtype=float)
cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_eff) / (L_box * chi_eff**2)

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

floor_master_2000 = W_floor / (4 * np.pi) * np.sum((2 * ells_ext + 1) * cl_true_ext)

theory_with_diag = (M_clust @ cl_true_ext)[:Nl] + diag_cl
theory_with_floor = (M_clust @ cl_true_ext)[:Nl] + floor_master_2000

diff_const = diag_cl - floor_master_2000
print(f"diag_cl = {diag_cl:.4e}")
print(f"floor_master (Nl_large={Nl_large}) = {floor_master_2000:.4e}")
print(f"Difference (diag_cl - floor) = {diff_const:.4e}")
print(f"Fraction of diag_cl: {diff_const/diag_cl:.4f}")

# Now also compute the floor with a LARGE Nl_large for convergence
ells_big = np.arange(50000, dtype=float)
cl_true_big = b1_ref**2 * plin_ref((ells_big + 0.5) / chi_eff) / (L_box * chi_eff**2)
floor_converged = W_floor / (4 * np.pi) * np.sum((2 * ells_big + 1) * cl_true_big)
print(f"\nfloor_master (Nl_large=50000, converged) = {floor_converged:.4e}")
print(f"diag_cl / floor_converged = {diag_cl / floor_converged:.6f}")

# Bin and compare
cl_mean = np.mean(cl_k_all, axis=0)
cl_std = np.std(cl_k_all, axis=0) / np.sqrt(num_sim)

couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

binned_raw = bins @ cl_mean
binned_diag = bins @ theory_with_diag
binned_floor = bins @ theory_with_floor

print(f"\n{'ell':>8s} {'data':>12s} {'diag_th':>12s} {'floor_th':>12s} "
      f"{'r_diag':>8s} {'r_floor':>8s}")
print("-" * 70)
for i in range(len(binned_ells)):
    rd = binned_raw[i] / binned_diag[i] if binned_diag[i] > 0 else 0
    rf = binned_raw[i] / binned_floor[i] if binned_floor[i] > 0 else 0
    print(f"{binned_ells[i]:8.1f} {binned_raw[i]:12.4e} {binned_diag[i]:12.4e} "
          f"{binned_floor[i]:12.4e} {rd:8.4f} {rf:8.4f}")

r_diag_all = binned_raw / binned_diag
r_floor_all = binned_raw / binned_floor
print(f"\nWith diag_cl:    mean ratio = {np.mean(r_diag_all[1:]):.4f} ± {np.std(r_diag_all[1:]):.4f}")
print(f"With floor_master: mean ratio = {np.mean(r_floor_all[1:]):.4f} ± {np.std(r_floor_all[1:]):.4f}")
print(f"With diag_cl:    ALL bins mean = {np.mean(r_diag_all):.4f}")
print(f"With floor_master: ALL bins mean = {np.mean(r_floor_all):.4f}")

# Try also with floor_converged
theory_with_fc = (M_clust @ cl_true_ext)[:Nl] + floor_converged
binned_fc = bins @ theory_with_fc
r_fc_all = binned_raw / binned_fc
print(f"\nWith converged floor (Nl=50000): mean ratio = {np.mean(r_fc_all[1:]):.4f} ± {np.std(r_fc_all[1:]):.4f}")
print(f"                         ALL bins = {np.mean(r_fc_all):.4f}")

# What chi makes floor_master equal diag_cl?
from scipy.optimize import brentq

def floor_diff(chi):
    ells_big = np.arange(10000, dtype=float)
    ct = b1_ref**2 * plin_ref((ells_big + 0.5) / chi) / (L_box * chi**2)
    fm = W_floor / (4 * np.pi) * np.sum((2 * ells_big + 1) * ct)
    return fm - diag_cl

try:
    chi_match = brentq(floor_diff, 4000, 6500)
    print(f"\nchi that makes floor_master = diag_cl: {chi_match:.1f}")
    print(f"  (chi_0={chi_0}, chi_eff={chi_eff:.1f})")
except:
    print(f"\nNo chi found that matches")
