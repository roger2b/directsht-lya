#!/usr/bin/env python
"""
Self-consistent normalization test.

Regenerates GRFs from scratch (no saved data).  Tests the complete pipeline
at k=0 (the only k mode where the mask FT is non-zero for a periodic box).

For a periodic box with uniform pixel weights K_j = 1:
  - LOS DFT of mask: K_tilde_j(k=0) = N (sum of ones), K_tilde_j(k≠0) = 0
  - The angular window comes from sightline POSITIONS (k-independent)
  - wl_ref = hp.alm2cl(SHT(θ, φ, N × ones))

Comparison:
  theory_plotted = C_ell_orig / (4π)²
  where C_ell_orig = M_pk @ PLKjKk / (4π × 2π × χ̄²)

  <pseudo-Cl>  = hp.alm2cl( SHT(θ, φ, FT_delta_real[:,0]) )

The ratio <pseudo-Cl> / theory_plotted should → 1 with enough sims.
"""
import sys, os, gc, time
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
from sht.theory_lya import compute_chi_bar_from_grid

# =================================================================== #
# Parameters                                                           #
# =================================================================== #
chi_shift  = 5000
Nl         = 500
lambda_max = 500
num_qso    = 9797
num_sim    = 5
k_idx      = 0          # k=0 is the only mode with non-zero mask FT
add_rsd_   = False       # match original saved results
NperBin    = 32

Nx   = 2 * Nl
xmax = 0.75
sht_eng = DirectSHT(Nl, Nx, xmax)
print(f"DirectSHT: Nl={Nl}, Nx={Nx}, xmax={xmax}")

# =================================================================== #
# PART 1: Measure pseudo-Cl at k=0 from multiple GRF realizations     #
# =================================================================== #
print(f"\n--- Generating {num_sim} GRF realizations ---")
cl_stack = []  # measured pseudo-Cl(k=0)
wl_ref = None  # angular window from first sim

for sim_idx in range(num_sim):
    seed = 1000 + sim_idx
    t0 = time.time()

    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
        Nskew=num_qso, shift=chi_shift)
    all_theta, all_phi = GRF.compute_theta_phi_skewer_start(
        all_x[:, 0], all_y[:, 0], all_z[:, 0])
    chi_grid = all_x[0, :]
    delta_F = all_w_gal - 1.0

    # LOS DFT (original convention: unnormalized, real-only output)
    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)

    # Verify: at k=0, FT_mask = N for all sightlines (periodic box)
    N = chi_grid.size
    if sim_idx == 0:
        assert np.allclose(FT_mask[:, 0], N), \
            f"FT_mask[:,0] should be N={N}, got {FT_mask[:5,0]}"
        # And at k≠0: FT_mask = 0
        assert np.allclose(FT_mask[:, 1], 0, atol=1e-10), \
            f"FT_mask[:,1] should be 0, got max={np.max(np.abs(FT_mask[:,1]))}"
        print(f"  ✓ FT_mask[:,0] = {N} (all sightlines)")
        print(f"  ✓ FT_mask[:,1] = 0 (periodic box, k≠0)")

    # SHT on data at k=0 (real-only since FT at k=0 is real for real field)
    hdat = sht_eng(all_theta, all_phi, FT_delta[:, k_idx])
    cl = hp.alm2cl(hdat)[:Nl]
    cl_stack.append(cl)

    # Window from first realization (same sightlines for all sims due to
    # hardcoded np.random.seed(100) in process_skewers)
    if sim_idx == 0:
        hran = sht_eng(all_theta, all_phi, FT_mask[:, k_idx])
        wl_ref = hp.alm2cl(hran)[:Nl]
        chi_grid_ref = chi_grid
        theta_ref, phi_ref = all_theta, all_phi
        Nskew_ref = Nskew
        plin_ref = GRF.plin  # save interpolator
        b1_ref = GRF.my_bias

    # Free memory
    del GRF, all_x, all_y, all_z, all_w_rand, all_w_gal, delta_F
    del FT_mask, FT_delta
    gc.collect()

    dt = time.time() - t0
    print(f"  sim {sim_idx}: seed={seed}, Nskew={Nskew}, dt={dt:.1f}s")

cl_stack = np.array(cl_stack)
cl_mean = np.mean(cl_stack, axis=0)

dchi = chi_grid_ref[1] - chi_grid_ref[0]
L_box = chi_grid_ref.size * dchi
N = chi_grid_ref.size
chi_bar = compute_chi_bar_from_grid(chi_grid_ref)
print(f"\nNskew={Nskew_ref}, N={N}, L_box={L_box:.1f}, dchi={dchi:.4f}, chi_bar={chi_bar:.1f}")
print(f"b1 (bias in GRF) = {b1_ref}")

# =================================================================== #
# PART 2: Pair-counting theory (original code approach)                #
# =================================================================== #
print(f"\n--- Computing pair-counting theory ---")

# Pair counting: PLKjKk[λ] = KjKk × Σ_{j,k} P_λ(cos θ_{jk})
nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2  # periodic box: FT_mask[:,0] = N for all sightlines
del nhat; gc.collect()

t0 = time.time()
print("  Computing Legendre sums...", end="", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f"done ({time.time()-t0:.1f}s)")
del cos_theta; gc.collect()

# Verify identity: PLKjKk[λ] = 4π × wl_ref[λ]
ratio_check = PLKjKk[:min(10,Nl)] / (4 * np.pi * wl_ref[:min(10,Nl)])
print(f"  ✓ PLKjKk / (4π × wl_ref) = {ratio_check[:5]} (should be 1.0)")

# Power spectrum at k=0: P_lin(L/chi_bar, k_par=0)
L_range = np.arange(lambda_max, dtype=float)
# k_arr is in cycles/(Mpc/h), k_par = k_arr[0] = 0
pk_L = plin_ref(L_range / chi_bar)

# With add_rsd=False, GRF field = b1 * delta_m
# We need b1^2 * P_lin in the theory since the measured Cl includes b1^2
pk_L_with_bias = b1_ref**2 * pk_L

# Coupling matrix: M_pk[l,λ] = (2λ+1)/(4π) Σ_L (2L+1) pk[L] (3j)²
t0 = time.time()
couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L_with_bias)
coupling_pk = couple_pk.compute_matrix()
print(f"  Coupling matrix computed in {time.time()-t0:.1f}s")

# Theory (original formula):
C_theory = coupling_pk @ PLKjKk / (4 * np.pi) / (2 * np.pi * chi_bar**2)
C_theory_plotted = C_theory / (4 * np.pi)**2

# =================================================================== #
# PART 3: Binning and comparison                                       #
# =================================================================== #
MD = MaskDeconvolution(Nl, wl_ref)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

binned_theory = bins @ C_theory_plotted[:Nl]
binned_mean = bins @ cl_mean
binned_std = bins @ (np.std(cl_stack, axis=0) / np.sqrt(num_sim))

print(f"\n{'='*70}")
print(f"RESULTS: k=0, {num_sim} sims, Nl={Nl}, Nskew~{Nskew_ref}")
print(f"  add_rsd={add_rsd_}, b1={b1_ref}")
print(f"{'='*70}")
print(f"{'ell':>8s} {'theory':>12s} {'measured':>12s} {'ratio':>8s} {'SNR':>6s}")
print(f"{'-'*50}")

ratios = []
for i in range(min(15, len(binned_ells))):
    if binned_theory[i] > 0:
        r = binned_mean[i] / binned_theory[i]
        snr = binned_mean[i] / binned_std[i] if binned_std[i] > 0 else np.inf
        ratios.append(r)
        print(f"{binned_ells[i]:8.1f} {binned_theory[i]:12.4e} "
              f"{binned_mean[i]:12.4e} {r:8.4f} {snr:6.1f}")

mean_ratio = np.mean(ratios[1:])  # skip monopole
print(f"\nMean ratio (excluding monopole) = {mean_ratio:.4f}")
print(f"Expected: ≈1.0 with scatter from {num_sim} sims")

# =================================================================== #
# PART 4: Verify MaskDeconvolution-based theory gives same result      #
# =================================================================== #
print(f"\n--- MaskDeconvolution-based theory comparison ---")

# C_true for MaskDeconvolution: P_F / (32 π³ χ²)
# With add_rsd=False: P_F = b1^2 × P_lin
cl_true_md = b1_ref**2 * plin_ref(ells / chi_bar) / (32 * np.pi**3 * chi_bar**2)

# Forward convolve and deconvolve (= what convolve_theory_Cls does)
ells_dec, theory_dec = MD.convolve_theory_Cls(cl_true_md, bins)
# Deconvolve the measurement
ells_meas, meas_dec = MD(cl_mean, bins)

print(f"{'ell':>8s} {'th_dec':>12s} {'meas_dec':>12s} {'ratio':>8s}")
print(f"{'-'*44}")
ratios_md = []
for i in range(min(15, len(ells_dec))):
    if theory_dec[i] > 0:
        r = meas_dec[i] / theory_dec[i]
        ratios_md.append(r)
        print(f"{ells_dec[i]:8.1f} {theory_dec[i]:12.4e} "
              f"{meas_dec[i]:12.4e} {r:8.4f}")

mean_ratio_md = np.mean(ratios_md[1:])
print(f"\nMean ratio (MaskDeconv, excl. monopole) = {mean_ratio_md:.4f}")
print(f"Should match pair-counting ratio above: {mean_ratio:.4f}")

print(f"\nDone!")
