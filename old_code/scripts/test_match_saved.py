#!/usr/bin/env python
"""
Final normalization test: match EXACT production settings.
Use original defaults (b1=1.0, beta=1.5, add_rsd=False → beta=0 internally).
Use Nq=9797 for more sightlines. Nl=500 like saved data.
Run 1 sim only, check ratio within cosmic variance.
Then run with saved parameters and compare to saved file.
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

# ---- Use ORIGINAL defaults from main branch ---- #
chi_shift  = 5000
Nl         = 500
lambda_max = 500
num_qso    = 9797
NperBin    = 32

# ORIGINAL DEFAULTS from main branch
my_bias = 1.0
my_beta = 1.5
# With add_rsd=False, beta gets set to 0 internally,
# But the amplitudes still get multiplied by my_bias.

sht_eng = DirectSHT(Nl, 2*Nl, 0.75)
print(f"Settings: Nl={Nl}, Nq={num_qso}, bias={my_bias}, beta={my_beta}")

# ---- Generate 1 sim (like saved data seed=1000) ---- #
print("\n=== Generating GRF with ORIGINAL parameters ===")
t0 = time.time()
G = my_GRF.PowerSpectrumGenerator(
    h=0.6770, Omega_b=0.04904, Omega_m=0.3147, ns=0.96824, As=2.10732e-9,
    add_rsd=False, seed=1000, my_bias=my_bias, my_beta=my_beta)
ax, ay, az, wr, wg, ns_eff = G.process_skewers(Nskew=num_qso, shift=chi_shift)
at, ap = G.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
chi = ax[0,:]
dF = wg - 1.0
N = chi.size
dchi = chi[1] - chi[0]
chi_bar = 0.5*(chi.min()+chi.max())
plin = G.plin
print(f"Nskew={ns_eff}, N={N}, chi_bar={chi_bar:.1f}, dt={time.time()-t0:.0f}s")

# DFT
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi, wr, dF)

# Release big arrays
del G, ax, ay, az, wr, wg; gc.collect()

# SHT at k=0
hdat = sht_eng(at, ap, FT_delta[:, 0])
hran = sht_eng(at, ap, FT_mask[:, 0])
cl_data = hp.alm2cl(hdat)[:Nl]
wl_ref = hp.alm2cl(hran)[:Nl]

print(f"cl_data[:5] = {cl_data[:5]}")
print(f"wl_ref[0] = {wl_ref[0]:.4e}")

# ---- Compare to saved data ---- #
try:
    saved = np.load(os.path.join(root, 'notebooks', 'data',
                    'Cell_GRF_L1380_N512_Nq9797_Nl500_sims20.npz'))
    cl_saved = saved['cl_k'][0, :]  # first sim, all ells
    wl_saved = saved['wl_k'][0, :]  # first sim
    print(f"\nSaved data comparison:")
    print(f"  cl_saved[:5]  = {cl_saved[:5]}")
    print(f"  cl_data[:5]   = {cl_data[:5]}")
    print(f"  Match? ratios = {cl_data[:5]/cl_saved[:5]}")
    print(f"  wl_saved[0]   = {wl_saved[0]:.4e}")
    print(f"  wl_ref[0]     = {wl_ref[0]:.4e}")
    print(f"  wl match?     = {wl_ref[0]/wl_saved[0]:.6f}")
except Exception as e:
    print(f"Could not load saved data: {e}")

# ---- Theory: pair-counting ---- #
print("\n=== Computing theory ===")
nhat = sht_lya.compute_nhat(at, ap)
cos_theta = np.dot(nhat, nhat.T)
del nhat; gc.collect()

KjKk = N**2
print("Computing Legendre sums...", end="", flush=True)
t0 = time.time()
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f" done ({time.time()-t0:.1f}s)")
del cos_theta; gc.collect()

L_range = np.arange(lambda_max, dtype=float)
pk_L = plin(L_range / chi_bar)
pk_L[0] = plin(0.5/chi_bar)

couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
M_pk = couple_pk.compute_matrix()

C_theory = M_pk @ PLKjKk / (4*np.pi) / (2*np.pi*chi_bar**2)
C_plot = C_theory / (4*np.pi)**2

# Binning
couple_win = Wigner3j.CoupleMat(Nl, wl_ref)
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=couple_win.compute_matrix())
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
bn_ells = bins @ ells
bn_data = bins @ cl_data
bn_theory = bins @ C_plot[:Nl]

# Compare to saved theory
try:
    theory_saved = saved['theory_cl']  # already = binned C_theory (not / (4π)²!)
    theory_saved_plotted = theory_saved / (4*np.pi)**2
    print(f"\nSaved theory comparison:")
    print(f"  theory_saved_plotted[:5] = {theory_saved_plotted[:5]}")
    print(f"  bn_theory[:5]           = {bn_theory[:5]}")
    print(f"  Ratio                   = {bn_theory[:5]/theory_saved_plotted[:5]}")
except:
    pass

print(f"\n{'ell':>8s} {'data':>12s} {'theory':>12s} {'ratio':>8s}")
print("-"*44)
ratios = []
for i in range(min(15, len(bn_ells))):
    r = bn_data[i] / bn_theory[i] if bn_theory[i] > 0 else np.inf
    ratios.append(r)
    print(f"{bn_ells[i]:8.1f} {bn_data[i]:12.4e} {bn_theory[i]:12.4e} {r:8.4f}")

mr = np.mean(ratios[1:])
print(f"\nMean ratio (excl monopole): {mr:.4f}")
print(f"Expected ~1.0 for correctly normalized theory")

# Now compare saved measured with saved theory
try:
    meas_saved = saved['measured_cl']  # (20, 15) binned measurements
    mean_meas_saved = np.mean(meas_saved, axis=0)
    ells_saved = saved['binned_ells']
    print(f"\nSaved file: mean measured / theory_plotted:")
    for i in range(min(10, len(ells_saved))):
        if theory_saved_plotted[i] > 0:
            r = mean_meas_saved[i] / theory_saved_plotted[i]
            print(f"  ell={ells_saved[i]:5.1f}: ratio={r:.4f}")
except:
    pass
