#!/usr/bin/env python
"""
Verify that K_tilde(k!=0) = 0 for periodic box, and get the normalization
right at k=0 using self-consistent cosmology (regenerating GRFs).

Key physics:
  - Mask weights w_j = 1 for all pixels in periodic box
  - FT(w_j)[k=0] = N,  FT(w_j)[k!=0] = 0  (DFT of constant)
  - So the window C_l(k=0) is non-trivial, but C_l(k!=0) = 0
  - For k=0: mode coupling from angular window exists
  - For k!=0: NO radial window contribution

  - Data weights = w * delta_F = delta_F
  - pseudo-Cl(k) = |a_lm^data(k)|^2 / (2l+1)
  - NO D-R subtraction needed: delta_F already has mean subtracted

GRF field amplitude includes bias factor:
  amplitudes = my_bias * sqrt(P_lin/2) * (1 + beta*mu^2) * random
  => field variance = my_bias^2 * (1 + beta*mu^2)^2 * P_lin

Theory must include b^2 even when add_rsd=False (beta=0):
  P_theory = b^2 * P_lin  (since Kaiser with beta=0)
"""
import sys, os, time
import numpy as np
import healpy as hp

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j

# ═══════════════════════════════════════════════════════════════════ #
# Settings  (tuned for ~4 GB RAM MacBook)                             #
# ═══════════════════════════════════════════════════════════════════ #
import gc

chi_shift  = 5000
Nl         = 100          # reduced from 200
lambda_max = 100          # reduced from 200
num_qso    = 2000         # reduced from 5000
num_sim    = 5
add_rsd_   = False
NperBin    = 16           # reduced from 32
seed0      = 1000

sht_eng = DirectSHT(Nl, 2*Nl, 0.75)

# ═══════════════════════════════════════════════════════════════════ #
# Part 1: Verify K_tilde(k!=0) = 0 for periodic box                  #
# ═══════════════════════════════════════════════════════════════════ #
print("=" * 60)
print("Part 1: Verify K_tilde(k!=0) = 0 for periodic box")
print("=" * 60)

GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed0)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew=num_qso, shift=chi_shift)
chi_grid = all_x[0, :]
delta_F = all_w_gal - 1.0
N = chi_grid.size
dchi = chi_grid[1] - chi_grid[0]
L_box = N * dchi
chi_bar = 0.5 * (chi_grid.min() + chi_grid.max())

# Compute DFT
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)

# Save plin interpolator and bias before freeing GRF
plin_func = GRF.plin
my_bias = GRF.my_bias
my_beta = GRF.my_beta

# Compute theta,phi from first sim's sightlines (for pair counting)
all_theta, all_phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])

print(f"N = {N}, L_box = {L_box:.1f}, dchi = {dchi:.4f}")
print(f"Nskew = {Nskew}, chi_bar = {chi_bar:.1f}")
print(f"\nFT_mask (should be N={N} at k=0, 0 elsewhere):")
print(f"  k=0: FT_mask[0,0] = {FT_mask[0, 0]:.6f}")
print(f"  k=1: max|FT_mask[:,1]| = {np.max(np.abs(FT_mask[:, 1])):.2e}")
print(f"  k=2: max|FT_mask[:,2]| = {np.max(np.abs(FT_mask[:, 2])):.2e}")
print(f"  all k>0: max|FT_mask[:,1:]| = {np.max(np.abs(FT_mask[:, 1:])):.2e}")
print(f"\nFT_delta (should be non-zero at all k):")
print(f"  k=0: mean|FT_delta[:,0]| = {np.mean(np.abs(FT_delta[:, 0])):.4f}")
print(f"  k=1: mean|FT_delta[:,1]| = {np.mean(np.abs(FT_delta[:, 1])):.4f}")
print(f"  k=5: mean|FT_delta[:,5]| = {np.mean(np.abs(FT_delta[:, 5])):.4f}")

# Verify: for k>0 window is exactly zero
assert np.max(np.abs(FT_mask[:, 1:])) < 1e-10, \
    f"FT_mask at k>0 is NOT zero! max = {np.max(np.abs(FT_mask[:, 1:]))}"
print("\n✓ CONFIRMED: K_tilde(k>0) = 0 for periodic box")

# ═══════════════════════════════════════════════════════════════════ #
# Part 2: Check GRF field power includes bias factor                  #
# ═══════════════════════════════════════════════════════════════════ #
print("\n" + "=" * 60)
print("Part 2: GRF field bias and cosmology check")
print("=" * 60)
print(f"my_bias = {my_bias}")
print(f"my_bias^2 = {my_bias**2:.6f}")
print(f"my_beta = {my_beta}")

# Free the large GRF arrays now that we have what we need
del GRF, all_w_rand, all_w_gal, delta_F, FT_mask, FT_delta
del all_x, all_y, all_z
gc.collect()

# ═══════════════════════════════════════════════════════════════════ #
# Part 3: Self-consistent test at k=0 with b^2 in theory             #
# ═══════════════════════════════════════════════════════════════════ #
print("\n" + "=" * 60)
print("Part 3: Self-consistent test at k=0")
print("=" * 60)

# Measure pseudo-Cl at k=0 over multiple sims
cl_stack = []
wl_ref = None
for sim_idx in range(num_sim):
    seed = seed0 + sim_idx
    G = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
    ax, ay, az, wr, wg, ns = G.process_skewers(Nskew=num_qso, shift=chi_shift)
    at, ap = G.compute_theta_phi_skewer_start(ax[:, 0], ay[:, 0], az[:, 0])
    dF = wg - 1.0
    chi = ax[0, :]
    _, fm, fd = sht_lya.compute_dft(chi, wr, dF)
    del G; gc.collect()  # free the ~2 GB GRF immediately

    # SHT at k=0: pseudo-Cl = |alm(FT_delta)|^2, NO D-R subtraction
    hdat = sht_eng(at, ap, fd[:, 0])
    cl = hp.alm2cl(hdat)[:Nl]
    cl_stack.append(cl)
    print(f"  sim {sim_idx}: cl[5]={cl[5]:.4e}")

    if sim_idx == 0:
        hran = sht_eng(at, ap, fm[:, 0])
        wl_ref = hp.alm2cl(hran)[:Nl]

cl_stack = np.array(cl_stack)
cl_mean = np.mean(cl_stack, axis=0)

# Theory: pair-counting approach (original code)
# Power spectrum of the field: b^2 * (1+beta*mu^2)^2 * P_lin
# With add_rsd=False (beta=0): P_field = b^2 * P_lin
L_range = np.arange(lambda_max, dtype=float)

# Theory WITH b^2 factor
pk_L_with_b2 = my_bias**2 * plin_func(L_range / chi_bar)
# Theory WITHOUT b^2 factor (wrong)
pk_L_no_b2 = plin_func(L_range / chi_bar)

# Pair counting
nhat = sht_lya.compute_nhat(all_theta, all_phi)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2  # periodic

print("Computing Legendre pair-counting...", end="", flush=True)
t0 = time.time()
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f"done ({time.time()-t0:.1f}s)")

# Coupling matrices
couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L_with_b2)
coupling_pk = couple_pk.compute_matrix()

couple_pk_nob = Wigner3j.CoupleMat(lambda_max, pk_L_no_b2)
coupling_pk_nob = couple_pk_nob.compute_matrix()

# Theory pseudo-Cl (original formula)
C_ell_with_b2 = coupling_pk @ PLKjKk / (4*np.pi) / (2*np.pi*chi_bar**2)
C_ell_no_b2   = coupling_pk_nob @ PLKjKk / (4*np.pi) / (2*np.pi*chi_bar**2)

# The original code plots: C_theory / (4pi)^2
C_plot_with_b2 = C_ell_with_b2 / (4*np.pi)**2
C_plot_no_b2   = C_ell_no_b2 / (4*np.pi)**2

# Binning
couple_win = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_win = couple_win.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_win)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
bn_ells = bins @ ells

bn_data = bins @ cl_mean
bn_std = bins @ (np.std(cl_stack, axis=0) / np.sqrt(num_sim))
bn_with_b2 = bins @ C_plot_with_b2[:Nl]
bn_no_b2 = bins @ C_plot_no_b2[:Nl]

print(f"\n{'ell':>6s} {'data':>12s} {'theory(b2)':>12s} {'ratio(b2)':>10s} "
      f"{'theory(no b)':>12s} {'ratio(no b)':>10s}")
print("-" * 70)
for i in range(min(6, len(bn_ells))):
    r1 = bn_data[i] / bn_with_b2[i] if bn_with_b2[i] > 0 else np.inf
    r2 = bn_data[i] / bn_no_b2[i] if bn_no_b2[i] > 0 else np.inf
    print(f"{bn_ells[i]:6.0f} {bn_data[i]:12.4e} {bn_with_b2[i]:12.4e} {r1:10.4f} "
          f"{bn_no_b2[i]:12.4e} {r2:10.4f}")

print(f"\nExpected: ratio(with b^2) ≈ 1.0, ratio(no b) ≈ 1/b^2 = {1/my_bias**2:.1f}")

# ═══════════════════════════════════════════════════════════════════ #
# Part 4: Also test with my_bias=1 GRFs for clean comparison         #
# ═══════════════════════════════════════════════════════════════════ #
print("\nDone!")
