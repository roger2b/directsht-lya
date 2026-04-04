#!/usr/bin/env python
"""
Quick test: does the ratio depend on bias value?
Run the same test with my_bias=1.0 (old default) and my_bias=-0.1521 (new).
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

chi_shift = 5000
Nl = 100
lambda_max = 100
num_qso = 2000
num_sim = 10
NperBin = 16
seed0 = 1000
sht_eng = DirectSHT(Nl, 2*Nl, 0.75)

def run_test(bias_val, label):
    print(f"\n{'='*60}")
    print(f"Testing with my_bias={bias_val} ({label})")
    print(f"{'='*60}")
    
    cl_stack = []
    wl_ref = None
    plin_func = None
    all_theta = all_phi = None
    chi_grid = None
    N = None
    
    for sim_idx in range(num_sim):
        seed = seed0 + sim_idx
        G = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=seed, my_bias=bias_val)
        ax, ay, az, wr, wg, ns = G.process_skewers(Nskew=num_qso, shift=chi_shift)
        at, ap = G.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
        dF = wg - 1.0
        chi = ax[0,:]
        _, fm, fd = sht_lya.compute_dft(chi, wr, dF)
        
        if sim_idx == 0:
            plin_func = G.plin
            all_theta, all_phi = at, ap
            chi_grid = chi
            N = chi.size
            hran = sht_eng(at, ap, fm[:, 0])
            wl_ref = hp.alm2cl(hran)[:Nl]
        
        del G; gc.collect()
        hdat = sht_eng(at, ap, fd[:, 0])
        cl_stack.append(hp.alm2cl(hdat)[:Nl])
        del fd; gc.collect()
    
    Nskew = len(all_theta)
    cl_mean = np.mean(cl_stack, axis=0)
    chi_bar = 0.5 * (chi_grid.min() + chi_grid.max())
    
    # Theory
    L_range = np.arange(lambda_max, dtype=float)
    pk_L = bias_val**2 * plin_func(L_range / chi_bar)
    
    nhat = sht_lya.compute_nhat(all_theta, all_phi)
    cos_theta = np.dot(nhat, nhat.T)
    KjKk = N**2
    PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
    
    couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
    coupling_pk = couple_pk.compute_matrix()
    C_ell_raw = coupling_pk @ PLKjKk / (4*np.pi) / (2*np.pi*chi_bar**2)
    C_plot = C_ell_raw / (4*np.pi)**2
    
    couple_win = Wigner3j.CoupleMat(Nl, wl_ref)
    MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=couple_win.compute_matrix())
    bins = MD.binning_matrix('linear', 0, NperBin)
    ells = np.arange(Nl, dtype=float)
    bn_ells = bins @ ells
    bn_data = bins @ cl_mean
    bn_theory = bins @ C_plot[:Nl]
    
    print(f"Nskew={Nskew}, N={N}, b={bias_val}, b^2={bias_val**2:.6f}")
    print(f"{'ell':>6s} {'data':>12s} {'theory':>12s} {'ratio':>8s}")
    print("-"*42)
    ratios = []
    for i in range(len(bn_ells)):
        r = bn_data[i] / bn_theory[i] if bn_theory[i] > 0 else np.inf
        ratios.append(r)
        print(f"{bn_ells[i]:6.0f} {bn_data[i]:12.4e} {bn_theory[i]:12.4e} {r:8.4f}")
    print(f"Mean ratio: {np.mean(ratios[1:]):.4f}")
    
    del cl_stack, nhat, cos_theta, PLKjKk; gc.collect()

t0 = time.time()
run_test(1.0, "old default")
run_test(-0.1521, "new Lya bias")
print(f"\nTotal time: {time.time()-t0:.0f}s")
