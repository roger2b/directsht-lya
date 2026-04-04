#!/usr/bin/env python
"""
Final definitive test: verify both the raw and deconvolved comparisons.

This test:
1. Generates GRFs with ORIGINAL parameters matching saved Nq=9797 data
2. Compares raw binned pseudo-Cl vs raw binned theory (pair-counting)
3. Compares deconvolved pseudo-Cl vs deconvolved theory (MaskDeconvolution)
4. Tests both at k=0
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

# ---- Production parameters ---- #
chi_shift  = 5000
Nl         = 500
lambda_max = 500
num_qso    = 9797
num_sim    = 5
NperBin    = 32

sht_eng = DirectSHT(Nl, 2*Nl, 0.75)

# ---- Generate and measure ---- #
print("=== Generating GRFs (original cosmology, b1=1, add_rsd=False) ===")
cl_stack = []

for i in range(num_sim):
    t0 = time.time()
    # Use ORIGINAL main-branch cosmology
    G = my_GRF.PowerSpectrumGenerator(
        h=0.6770, Omega_b=0.04904, Omega_m=0.3147, ns=0.96824, As=2.10732e-9,
        add_rsd=False, seed=1000+i, my_bias=1.0, my_beta=1.5)
    ax, ay, az, wr, wg, ns_eff = G.process_skewers(Nskew=num_qso, shift=chi_shift)
    at, ap = G.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
    chi = ax[0,:]
    dF = wg - 1.0
    k_arr, fm, fd = sht_lya.compute_dft(chi, wr, dF)
    
    hdat = sht_eng(at, ap, fd[:, 0])
    cl = hp.alm2cl(hdat)[:Nl]
    cl_stack.append(cl)
    
    if i == 0:
        hran = sht_eng(at, ap, fm[:, 0])
        wl_ref = hp.alm2cl(hran)[:Nl]
        plin = G.plin
        theta_ref, phi_ref = at, ap
        chi_ref = chi
        N = chi.size
    
    del G, ax, ay, az, wr, wg, fm, fd; gc.collect()
    print(f"  sim {i}: Nskew={ns_eff}, dt={time.time()-t0:.0f}s")

cl_stack = np.array(cl_stack)
cl_mean = np.mean(cl_stack, axis=0)
cl_std = np.std(cl_stack, axis=0) / np.sqrt(num_sim)
dchi = chi_ref[1] - chi_ref[0]
chi_bar = 0.5*(chi_ref.min() + chi_ref.max())
Nskew = len(theta_ref)
print(f"Nskew={Nskew}, N={N}, chi_bar={chi_bar:.1f}")

# ---- Pair-counting theory ---- #
print("\n=== Computing pair-counting theory ===")
nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
del nhat; gc.collect()
KjKk = N**2

print("Legendre sums...", end="", flush=True)
t0 = time.time()
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f" {time.time()-t0:.0f}s")
del cos_theta; gc.collect()

L_range = np.arange(lambda_max, dtype=float)
pk_L = plin(L_range / chi_bar)
pk_L[0] = plin(0.5/chi_bar)

couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
M_pk = couple_pk.compute_matrix()

C_theory_raw = M_pk @ PLKjKk / (4*np.pi) / (2*np.pi*chi_bar**2)
C_theory_plotted = C_theory_raw / (4*np.pi)**2

# ---- MaskDeconvolution setup ---- #
couple_win = Wigner3j.CoupleMat(Nl, wl_ref)
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=couple_win.compute_matrix())
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
bn_ells = bins @ ells

# ==================================================================== #
# TEST 1: Raw binned comparison (original code approach)                #
# ==================================================================== #
print(f"\n{'='*60}")
print("TEST 1: Raw binned pseudo-Cl vs binned C_theory/(4π)²")
print(f"{'='*60}")
bn_data = bins @ cl_mean
bn_theory = bins @ C_theory_plotted[:Nl]
bn_std = bins @ cl_std

print(f"{'ell':>8s} {'data':>12s} {'theory':>12s} {'ratio':>8s} {'SNR':>6s}")
print("-"*50)
ratios1 = []
for i in range(min(15, len(bn_ells))):
    r = bn_data[i] / bn_theory[i] if bn_theory[i] > 0 else np.inf
    snr = bn_data[i] / bn_std[i] if bn_std[i] > 0 else np.inf
    ratios1.append(r)
    print(f"{bn_ells[i]:8.1f} {bn_data[i]:12.4e} {bn_theory[i]:12.4e} {r:8.4f} {snr:6.1f}")

mr1 = np.mean(ratios1[1:])
print(f"\nMean ratio (excl monopole): {mr1:.4f}")

# ==================================================================== #
# TEST 2: Deconvolved comparison using MaskDeconvolution                #
# ==================================================================== #
print(f"\n{'='*60}")
print("TEST 2: Deconvolved pseudo-Cl vs deconvolved theory")
print(f"{'='*60}")

# Deconvolve the measured pseudo-Cl
bn_ells_dec, cl_deconv = MD(cl_mean, bins)

# For the theory: use C_Limber = P(l/chi)/chi² as the "true" Cl
# Then convolve_theory_Cls applies Mbb^-1 @ bins @ Mll @ C_true
C_limber = pk_L[:Nl] / chi_bar**2

# Test 2a: Use bare Limber C_l
bn_ells_t, theory_deconv_limber = MD.convolve_theory_Cls(C_limber, bins)

print(f"\nTest 2a: C_true = P(l/χ) / χ²  [Limber]")
print(f"{'ell':>8s} {'deconv_data':>12s} {'deconv_theo':>12s} {'ratio':>8s}")
print("-"*44)
for i in range(min(10, len(bn_ells_dec))):
    r = cl_deconv[i] / theory_deconv_limber[i] if theory_deconv_limber[i] > 0 else np.inf
    print(f"{bn_ells_dec[i]:8.1f} {cl_deconv[i]:12.4e} {theory_deconv_limber[i]:12.4e} {r:8.4f}")

# Test 2b: Scale C_true by various factors
print(f"\nTest 2b: Find scaling factor alpha such that deconvolved ratio ≈ 1")
# alpha_fit = ΣI(data_i * model_i) / ΣI(model_i²)
alpha_fit = np.sum(cl_deconv[1:] * theory_deconv_limber[1:]) / np.sum(theory_deconv_limber[1:]**2)
print(f"  Best-fit alpha = {alpha_fit:.6e}")
print(f"  1/(32π³) = {1/(32*np.pi**3):.6e}")
print(f"  alpha/(1/32π³) = {alpha_fit * 32*np.pi**3:.4f}")

# Test 2c: Verify using C_true = P/(32π³χ²)
C_true_fixed = pk_L[:Nl] / (32 * np.pi**3 * chi_bar**2)
_, theory_fixed = MD.convolve_theory_Cls(C_true_fixed, bins)

print(f"\nTest 2c: C_true = P / (32π³χ²)")
print(f"{'ell':>8s} {'deconv_data':>12s} {'deconv_theo':>12s} {'ratio':>8s}")
print("-"*44)
ratios_fixed = []
for i in range(min(15, len(bn_ells_dec))):
    r = cl_deconv[i] / theory_fixed[i] if theory_fixed[i] > 0 else np.inf
    ratios_fixed.append(r)
    print(f"{bn_ells_dec[i]:8.1f} {cl_deconv[i]:12.4e} {theory_fixed[i]:12.4e} {r:8.4f}")

mr_fixed = np.mean(ratios_fixed[1:])
print(f"\nMean ratio (excl monopole): {mr_fixed:.4f}")

# Test 2d: The window matrix approach
# Wbl = MD.window_matrix(bins) converts C_true → decoupled pseudo-Cl
Wbl = MD.window_matrix(bins)
deconv_from_wbl = Wbl @ C_limber

print(f"\nTest 2d: Wbl @ C_Limber vs deconv_data")
print(f"{'ell':>8s} {'deconv_data':>12s} {'Wbl@C_Lim':>12s} {'ratio':>8s}")
print("-"*44)
for i in range(min(10, len(bn_ells_dec))):
    r = cl_deconv[i] / deconv_from_wbl[i] if deconv_from_wbl[i] > 0 else np.inf
    print(f"{bn_ells_dec[i]:8.1f} {cl_deconv[i]:12.4e} {deconv_from_wbl[i]:12.4e} {r:8.4f}")

print(f"\n(Note: Wbl = Mbb^-1 @ bins @ Mll, so Wbl@C = Mbb^-1@bins@Mll@C)")
print(f"This SHOULD equal the deconvolved measurement if C_true is correct")

# ==================================================================== #
# TEST 3: Cross-check — the saved 20-sim data                          #
# ==================================================================== #
print(f"\n{'='*60}")
print("TEST 3: Cross-check with saved 20-sim data")
print(f"{'='*60}")

try:
    saved = np.load(os.path.join(root, 'notebooks', 'data',
                    'Cell_GRF_L1380_N512_Nq9797_Nl500_sims20.npz'))
    theory_saved = saved['theory_cl']
    theory_saved_plotted = theory_saved / (4*np.pi)**2
    saved_ells = saved['binned_ells']
    saved_meas = np.mean(saved['measured_cl'], axis=0)
    
    # The saved window-convolved theory (if it exists)
    if 'window_conv_theory_cl' in saved:
        wct = saved['window_conv_theory_cl']
        print(f"window_conv_theory_cl[0:3] = {wct[:3]}")
        print(f"theory_saved[0:3]          = {theory_saved[:3]}")
        print(f"ratio wct/theory           = {wct[:3]/theory_saved[:3]}")
    
    print(f"\nSaved 20-sim: mean measured / theory_plotted:")
    for i in range(min(10, len(saved_ells))):
        r = saved_meas[i] / theory_saved_plotted[i]
        print(f"  ell={saved_ells[i]:5.1f}: ratio={r:.4f}")
        
except Exception as e:
    print(f"Could not load saved data: {e}")

print(f"\nDone!")
