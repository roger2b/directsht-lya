"""Quick end-to-end test of the sFB pipeline."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks'))
import numpy as np

from sht.sht import DirectSHT
from sht.lya_sfb import LyaSFB, _alm2cl_complex
from sht.theory_lya import P_flux, theory_cl_k, compute_chi_bar_from_grid
from notebooks.GRF_class import PowerSpectrumGenerator

Nl = 100
Nx = 500

print('=== Full Pipeline Test ===')

# 1) GRF
print('1. GRF generation...')
t0 = time.time()
GRF = PowerSpectrumGenerator(N=512, L=1380.0, add_rsd=True,
                              my_bias=-0.1521, my_beta=0.2298,
                              z=2.33, seed=1000)
print(f'   GRF took {time.time()-t0:.1f}s')

# 2) Extract sightlines
print('2. Extracting sightlines...')
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = \
    GRF.process_skewers(Nskew=5000, shift=5000.0)
theta, phi = GRF.compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])
chi_grid = all_x[0, :]
chi_bar = compute_chi_bar_from_grid(chi_grid)
delta_F = all_w_gal - 1.0
print(f'   Nskew={Nskew}, chi_bar={chi_bar:.1f}')
print(f'   theta range: [{np.degrees(theta.min()):.2f}, {np.degrees(theta.max()):.2f}] deg')

# 3) LOS FT
print('3. LOS Fourier transform...')
t0 = time.time()
sht_engine = DirectSHT(Nl, Nx)
sfb = LyaSFB(sht_engine, Nl)
k_arr, delta_2d, K_tilde = sfb.compute_los_ft(chi_grid, delta_F)
print(f'   LOS FT took {time.time()-t0:.2f}s')
print(f'   k_arr shape: {k_arr.shape}, delta_2d shape: {delta_2d.shape}')

# 4) SHT for a few k-bins
for k_idx in [1, 5, 10]:
    print(f'4. SHT at k_idx={k_idx}, k={k_arr[k_idx]:.5f} h/Mpc...')
    t0 = time.time()
    alm_d, alm_r = sfb.sht_per_k(theta, phi, delta_2d[:, k_idx], K_tilde[:, k_idx])
    dt = time.time()-t0
    cl, N_norm = sfb.pseudo_cl(alm_d, alm_r, Nl)
    wl = _alm2cl_complex(alm_r, Nl)
    
    # Theory
    ell_arr = np.arange(Nl)
    cl_th = theory_cl_k(ell_arr, k_arr[k_idx], chi_bar, GRF.plin, 
                        b1=-0.1521, beta=0.2298)
    
    # Compare at ell=5,10,20
    for ell_check in [5, 10, 20]:
        if cl_th[ell_check] > 0:
            ratio = cl[ell_check] / cl_th[ell_check]
            print(f'   ell={ell_check}: measured={cl[ell_check]:.4e}, '
                  f'theory={cl_th[ell_check]:.4e}, ratio={ratio:.3f}')
    print(f'   SHT took {dt:.2f}s')

print('\n=== Pipeline test COMPLETE ===')
