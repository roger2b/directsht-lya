#!/usr/bin/env python
"""
Lean normalization test: measure pseudo-Cl at k=0, compare to pair-counting
theory. Memory efficient: one GRF at a time, immediately freed.
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

# ---- Settings ---- #
chi_shift  = 5000
Nl         = 100
lambda_max = 100
num_qso    = 2000
num_sim    = 20     # more sims for tighter mean
add_rsd_   = False
NperBin    = 16
seed0      = 1000
k_idx      = 0     # ONLY test k=0

sht_eng = DirectSHT(Nl, 2*Nl, 0.75)

# ---- First sim: get geometry + window ---- #
t0 = time.time()
GRF0 = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed0)
ax, ay, az, wr, wg, ns = GRF0.process_skewers(Nskew=num_qso, shift=chi_shift)
Nskew = ns
chi_grid = ax[0, :]
N = chi_grid.size
dchi = chi_grid[1] - chi_grid[0]
L_box = N * dchi
chi_bar = 0.5 * (chi_grid.min() + chi_grid.max())

# theta/phi (same for all sims due to hardcoded seed in process_skewers)
all_theta, all_phi = GRF0.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
plin_func = GRF0.plin
my_bias = GRF0.my_bias

dF0 = wg - 1.0
_, fm0, fd0 = sht_lya.compute_dft(chi_grid, wr, dF0)

# Window from k=0 randoms SHT
hran = sht_eng(all_theta, all_phi, fm0[:, k_idx])
wl_ref = hp.alm2cl(hran)[:Nl]

# First data point
hdat = sht_eng(all_theta, all_phi, fd0[:, k_idx])
cl_stack = [hp.alm2cl(hdat)[:Nl]]

del GRF0, ax, ay, az, wr, wg, dF0, fm0, fd0; gc.collect()
print(f"Nskew={Nskew}, N={N}, L={L_box:.1f}, chi_bar={chi_bar:.1f}, b={my_bias}, b^2={my_bias**2:.6f}")
print(f"  sim 0: cl[5]={cl_stack[0][5]:.4e}")

# ---- Remaining sims ---- #
for sim_idx in range(1, num_sim):
    seed = seed0 + sim_idx
    G = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
    ax, ay, az, wr, wg, ns = G.process_skewers(Nskew=num_qso, shift=chi_shift)
    at, ap = G.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
    dF = wg - 1.0
    _, _, fd = sht_lya.compute_dft(ax[0,:], wr, dF)
    del G, ax, ay, az, wr, wg, dF; gc.collect()

    hdat = sht_eng(at, ap, fd[:, k_idx])
    cl_stack.append(hp.alm2cl(hdat)[:Nl])
    print(f"  sim {sim_idx}: cl[5]={cl_stack[-1][5]:.4e}")
    del fd; gc.collect()

cl_stack = np.array(cl_stack)
cl_mean = np.mean(cl_stack, axis=0)
cl_std = np.std(cl_stack, axis=0) / np.sqrt(num_sim)
print(f"\n{num_sim} sims done in {time.time()-t0:.0f}s")

# ---- Theory: pair-counting ---- #
L_range = np.arange(lambda_max, dtype=float)
pk_L = my_bias**2 * plin_func(L_range / chi_bar)  # P_field = b^2 * P_lin

nhat = sht_lya.compute_nhat(all_theta, all_phi)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2

print("Pair counting...", end="", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print("done")

couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
coupling_pk = couple_pk.compute_matrix()
C_ell_theory = coupling_pk @ PLKjKk / (4*np.pi) / (2*np.pi*chi_bar**2)
C_plot = C_ell_theory / (4*np.pi)**2

# Binning
couple_win = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_win = couple_win.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_win)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
bn_ells = bins @ ells

bn_data = bins @ cl_mean
bn_err = bins @ cl_std
bn_theory = bins @ C_plot[:Nl]

print(f"\n{'ell':>6s} {'data':>12s} {'theory':>12s} {'ratio':>8s} {'SNR':>6s}")
print("-" * 50)
ratios = []
for i in range(len(bn_ells)):
    r = bn_data[i] / bn_theory[i] if bn_theory[i] > 0 else np.inf
    snr = bn_data[i] / bn_err[i] if bn_err[i] > 0 else np.inf
    ratios.append(r)
    print(f"{bn_ells[i]:6.0f} {bn_data[i]:12.4e} {bn_theory[i]:12.4e} {r:8.4f} {snr:6.1f}")

print(f"\nMean ratio (all bins): {np.mean(ratios):.4f}")
print(f"Mean ratio (skip first): {np.mean(ratios[1:]):.4f}")
