#!/usr/bin/env python
"""
Normalization test v3: verify the Wigner coupling identity and find alpha.

Key identity to verify:
  coupling_pk @ PLKjKk = 4π × Mll @ pk  (where pk[L] = P_lin(L/chi_bar))

Then: C_ell_orig = coupling_pk @ PLKjKk / (4π) / (2π chi²) 
                 = Mll @ pk / (2π chi²)
Plotted: C_ell_orig / (4π)² = Mll @ pk / (32 π³ chi²)

So: <pseudo-Cl> = Mll @ pk / (32 π³ chi²) 
i.e.: alpha = 1/(32 π³ chi²) ... but that has dimensions! Need 1/L factor.

This test verifies everything numerically.
"""
import sys, os
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
from sht.theory_lya import theory_cl_k, compute_chi_bar_from_grid

# ---- settings ---- #
chi_shift = 5000
Nl = 100  # smaller for speed
seed = 1000
num_qso = 5000
add_rsd_ = False

sht_eng = DirectSHT(Nl, 2*Nl, 0.75)

# ---- GRF ---- #
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew=num_qso, shift=chi_shift)
all_theta, all_phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
chi_grid = all_x[0, :]
all_w_gal = all_w_gal - 1.0
dchi = chi_grid[1] - chi_grid[0]
L_box = chi_grid.size * dchi
N = chi_grid.size
chi_bar = compute_chi_bar_from_grid(chi_grid)
print(f"Nskew={Nskew}, N={N}, L_box={L_box:.1f}, dchi={dchi:.4f}, chi_bar={chi_bar:.1f}")

# ---- Measure ---- #
k_arr_orig, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)
hdat = sht_eng(all_theta, all_phi, FT_delta[:, 0])
hran = sht_eng(all_theta, all_phi, FT_mask[:, 0])
cl_data = hp.alm2cl(hdat)[:Nl]
cl_rand = hp.alm2cl(hran)[:Nl]

# ---- Build coupling matrices ---- #
L_range = np.arange(Nl, dtype=float)
pk_L = GRF.plin(L_range / chi_bar)  # P_lin(ell/chi_bar)
pk_L[0] = GRF.plin(0.5/chi_bar)  # avoid P(0) divergence

# Coupling from P(k): CoupleMat(Nl, pk_L) → M_pk[l,L'] = (2L'+1)/(4π) Σ_λ (2λ+1) pk[λ] 3j²
couple_pk = Wigner3j.CoupleMat(Nl, pk_L)
coupling_pk = couple_pk.compute_matrix()

# MaskDeconv: Mll from window = cl_rand → Mll[l,L] = (2L+1)/(4π) Σ_λ (2λ+1) cl_rand[λ] 3j²  
couple_win = Wigner3j.CoupleMat(Nl, cl_rand)
coupling_win = couple_win.compute_matrix()
MD = MaskDeconvolution(Nl, cl_rand, precomputed_Wigner=coupling_win)

# ---- Pair counting ---- #
nhat = sht_lya.compute_nhat(all_theta, all_phi)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
print("Computing pair-counting...", end="", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(Nl, cos_theta, KjKk)[:Nl]
print("done")

# ---- VERIFY IDENTITY: coupling_pk @ PLKjKk = 4π × Mll @ pk_L ---- #
lhs = coupling_pk @ PLKjKk
rhs = 4.0 * np.pi * MD.Mll @ pk_L

print("\n=== IDENTITY CHECK: coupling_pk @ PLKjKk = 4π × Mll @ pk ===")
for ell in [2, 5, 10, 20, 50]:
    if ell < Nl:
        print(f"  ell={ell}: LHS={lhs[ell]:.6e}, RHS={rhs[ell]:.6e}, ratio={lhs[ell]/rhs[ell]:.6f}")

# Check if PLKjKk[l] = 4π × cl_rand[l] 
print("\n=== PLKjKk vs 4π × cl_rand ===")
for ell in [0, 2, 5, 10, 20, 50]:
    if ell < Nl:
        ratio = PLKjKk[ell] / (4*np.pi * cl_rand[ell]) if cl_rand[ell] > 0 else np.inf  
        print(f"  ell={ell}: PLKjKk={PLKjKk[ell]:.4e}, 4π*cl_rand={4*np.pi*cl_rand[ell]:.4e}, ratio={ratio:.6f}")

# ---- Original code theory ---- #
C_ell_orig = coupling_pk @ PLKjKk / (4.0 * np.pi) / (2.0 * np.pi * chi_bar**2)
C_ell_orig_plotted = C_ell_orig / (4.0 * np.pi)**2

# ---- Substituting the identity ---- #
# C_ell_orig = [4π × Mll @ pk] / (4π × 2π χ²) = Mll @ pk / (2π χ²)
# C_ell_orig / (4π)² = Mll @ pk / (2π χ² × (4π)²) = Mll @ pk / (32π³ χ²)
c_ell_via_identity = MD.Mll @ pk_L / (2.0 * np.pi * chi_bar**2)
c_ell_plotted_ident = c_ell_via_identity / (4.0 * np.pi)**2

print("\n=== Original theory vs identity-based theory ===")
for ell in [2, 5, 10, 20, 50]:
    if ell < Nl:
        r = C_ell_orig_plotted[ell] / c_ell_plotted_ident[ell] if c_ell_plotted_ident[ell] > 0 else np.inf
        print(f"  ell={ell}: orig={C_ell_orig_plotted[ell]:.4e}, ident={c_ell_plotted_ident[ell]:.4e}, ratio={r:.6f}")

# ---- What is the measured pseudo-Cl? ---- #
print("\n=== Measured vs theory ===")
for ell in [2, 5, 10, 20, 50]:
    if ell < Nl:
        r1 = cl_data[ell] / C_ell_orig_plotted[ell] if C_ell_orig_plotted[ell] != 0 else np.inf
        print(f"  ell={ell}: data={cl_data[ell]:.4e}, theory_plotted={C_ell_orig_plotted[ell]:.4e}, ratio={r1:.4f}")

print(f"\nNow: <pseudo> should = Mll @ pk / (32 π³ χ²)")
print(f"  32π³ = {32*np.pi**3:.4f}")
print(f"  32π³χ² = {32*np.pi**3*chi_bar**2:.4e}")

# Let's compute the mean ratio across bins to see if it's constant:
NperBin = 16
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
bnells = bins @ ells

# Binned comparison
bn_data = bins @ cl_data
bn_theory_plotted = bins @ C_ell_orig_plotted[:Nl]

print(f"\n--- Binned ratios (data / theory_plotted) ---")
ratios = []
for i in range(len(bnells)):
    if bn_theory_plotted[i] > 0:
        r = bn_data[i] / bn_theory_plotted[i]
        ratios.append(r)
        print(f"  ell={bnells[i]:.0f}: ratio = {r:.4f}")

mean_ratio = np.mean(ratios[1:])  # skip first bin (monopole)
print(f"\nMean ratio (skipping monopole) = {mean_ratio:.6f}")
print(f"1/mean_ratio = {1/mean_ratio:.2f}")

# ---- Now: is the theory supposed to be multiplied by Nskew? ---- #
# From my derivation: the discrete approximation gives
#   <Cl_pseudo> = (1/Nskew^2) × SUM formula  or  similar
# But Nskew enters through the pair counting, not as a separate factor.

# The issue might be simpler: looking at the original code's plots for 
# MULTIPLE simulations, the theory matches. But for a SINGLE sim, there's 
# cosmic variance of order 1/Nskew or something.

# Let's run multiple sims and average:
print(f"\n=== Multiple realizations ===")
cl_stack = []
for sim_seed in range(1000, 1010):
    G = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=sim_seed)
    ax, ay, az, wr, wg, ns = G.process_skewers(Nskew=num_qso, shift=chi_shift)
    at, ap = G.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
    wg = wg - 1.0
    chi = ax[0,:]
    _, _, ftd = sht_lya.compute_dft(chi, wr, wg)
    h = sht_eng(at, ap, ftd[:, 0])
    cl = hp.alm2cl(h)[:Nl]
    cl_stack.append(cl)
    
cl_mean = np.mean(cl_stack, axis=0)
cl_std = np.std(cl_stack, axis=0)

bn_mean = bins @ cl_mean
bn_std = bins @ cl_std / np.sqrt(len(cl_stack))

print(f"\n--- Binned ratios (10-sim mean / theory_plotted) ---")
ratios_avg = []
for i in range(len(bnells)):
    if bn_theory_plotted[i] > 0:
        r = bn_mean[i] / bn_theory_plotted[i]
        ratios_avg.append(r)
        snr = bn_mean[i] / bn_std[i] if bn_std[i] > 0 else np.inf
        print(f"  ell={bnells[i]:.0f}: ratio = {r:.4f}  (SNR={snr:.1f})")

mean_ratio_avg = np.mean(ratios_avg[1:])
print(f"\nMean ratio (10 sims) = {mean_ratio_avg:.6f}")
print(f"1/mean_ratio = {1/mean_ratio_avg:.2f}")

# This should tell us if there's a fixed normalization offset or if 
# it's just cosmic variance.
