#!/usr/bin/env python
"""
Final money plot: pair-counting theory vs measured pseudo-Cl(k=0).
20 sims, Nl=500, lambda_max=1000, b1=-0.1521, no RSD.

DO NOT subtract shot noise — it's already in the pair-counting theory.
"""
import sys, os, gc, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp

sys.path.insert(0, '/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya')
sys.path.insert(0, '/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks')

from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
from sht.theory_lya import compute_chi_bar_from_grid
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j

plt.rcParams.update({'font.size': 14, 'figure.figsize': (10, 7)})

# Parameters
add_rsd_   = False
num_qso    = 9797
chi_shift  = 5000.0
Nl         = 500
lambda_max = 1000
Nx, xmax   = 2*Nl, 3./4.
NperBin    = 32
k_idx      = 0
num_sim    = 20

sht_eng = DirectSHT(Nl, Nx, xmax)

# ===== Simulation loop =====
cl_stack = []
wl_ref   = None

for sim_idx in range(num_sim):
    seed = 1000 + sim_idx
    t0 = time.time()
    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
    ax, ay, az, wr, wg, Nskew = GRF.process_skewers(Nskew=num_qso, shift=chi_shift)
    at, ap = GRF.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
    chi_grid = ax[0,:]
    dF = wg - 1.0

    k_arr, fm, fd = sht_lya.compute_dft(chi_grid, wr, dF)
    N = chi_grid.size

    hdat = sht_eng(at, ap, fd[:, k_idx])
    cl   = hp.alm2cl(hdat)[:Nl]
    cl_stack.append(cl)

    if sim_idx == 0:
        hran    = sht_eng(at, ap, fm[:, k_idx])
        wl_ref  = hp.alm2cl(hran)[:Nl]
        chi0    = chi_grid.copy()
        t0_, p0_ = at.copy(), ap.copy()
        Nskew0  = Nskew
        plin    = GRF.plin
        b1      = GRF.my_bias

    del GRF, ax, ay, az, wr, wg, dF, fm, fd
    gc.collect()
    print(f'  sim {sim_idx}: dt={time.time()-t0:.1f}s', flush=True)

cl_stack = np.array(cl_stack)
cl_mean  = np.mean(cl_stack, axis=0)
cl_std   = np.std(cl_stack, axis=0)

dchi    = chi0[1] - chi0[0]
L_box   = N * dchi
chi_bar = compute_chi_bar_from_grid(chi0)
ells    = np.arange(Nl, dtype=float)

print(f'Nskew={Nskew0}, N={N}, L_box={L_box:.1f}, chi_bar={chi_bar:.1f}, b1={b1}')

# ===== Pair-counting theory =====
nhat = sht_lya.compute_nhat(t0_, p0_)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
del nhat; gc.collect()

t0 = time.time()
print('Computing Legendre sums...', end='', flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f'done ({time.time()-t0:.1f}s)', flush=True)
del cos_theta; gc.collect()

L_range = np.arange(lambda_max, dtype=float)
pk_L = b1**2 * plin(L_range / chi_bar)

couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
coupling_pk = couple_pk.compute_matrix()

C_theory      = coupling_pk @ PLKjKk / (4*np.pi) / (2*np.pi * chi_bar**2)
C_theory_plot = C_theory / (4*np.pi)**2

# ===== Binning =====
MD   = MaskDeconvolution(Nl, wl_ref)
bins = MD.binning_matrix('linear', 0, NperBin)
bn_ells  = bins @ ells
bn_theo  = bins @ C_theory_plot[:Nl]
bn_mean  = bins @ cl_mean
bn_std   = bins @ (cl_std / np.sqrt(num_sim))

# Ratios
ratios = bn_mean / bn_theo
idx_low = np.where(bn_ells < 288)[0]
idx_low = idx_low[1:]  # skip monopole bin
mean_low = np.mean(ratios[idx_low])
mean_all = np.mean(ratios[1:])  # skip monopole

print(f'\nMean ratio (ell < 288): {mean_low:.4f}')
print(f'Mean ratio (all bins):  {mean_all:.4f}')

# ===== MaskDeconvolution =====
cl_true_md = b1**2 * plin(ells / chi_bar) / (32 * np.pi**3 * chi_bar**2)
ells_dec, theo_dec = MD.convolve_theory_Cls(cl_true_md, bins)
_, meas_dec        = MD(cl_mean, bins)
ratios_dec = meas_dec / theo_dec
mean_dec = np.mean(ratios_dec[1:])
print(f'MaskDeconv mean ratio:  {mean_dec:.4f}')

# ===== Plots =====
outdir = '/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks/plots'

# --- Plot 1: Pair-counting money plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={'hspace': 0.05})

ax1.set_title(f'Money plot: {num_sim} sims, $N_\\ell$={Nl}, '
              f'$\\lambda_{{\\max}}$={lambda_max}, $N_{{\\rm skew}}$={Nskew0}, '
              f'$b_1$={b1}')

for i in range(num_sim):
    ax1.plot(bn_ells, bins @ cl_stack[i], 'C0-', alpha=0.12, lw=0.8)
ax1.plot(bn_ells, bn_theo, 'k.--', lw=2.5, ms=6, zorder=10,
         label=r'Theory: $C_\ell^{\rm th}/(4\pi)^2$')
ax1.errorbar(bn_ells, bn_mean, yerr=bn_std, fmt='C3o', ms=5, capsize=3,
             zorder=11, label=f'Mean of {num_sim} sims')
ax1.set_ylabel(r'binned pseudo-$C_\ell(k{=}0)$')
ax1.legend(fontsize=12, loc='upper right')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax2.axhline(1.0, color='k', ls='--', lw=1)
ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ax2.errorbar(bn_ells, ratios, yerr=bn_std/bn_theo, fmt='C3o', ms=5, capsize=3)
ax2.set_xlabel(r'multipole $\ell$')
ax2.set_ylabel('data / theory')
ax2.set_ylim(0.7, 1.5)
ax2.text(0.02, 0.92, f'low-$\\ell$ mean = {mean_low:.3f}\nall-bin mean = {mean_all:.3f}',
         transform=ax2.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig(f'{outdir}/money_plot_final.png', bbox_inches='tight', dpi=150)
plt.savefig(f'{outdir}/money_plot_final.pdf', bbox_inches='tight')
print('Saved money_plot_final.png/pdf')
plt.close()

# --- Plot 2: MaskDeconvolution ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={'hspace': 0.05})

ax1.set_title(f'MaskDeconvolution: {num_sim} sims, $\\lambda_{{\\max}}$={lambda_max}')
ax1.plot(ells_dec, theo_dec, 'k.--', lw=2.5, ms=6, label='Theory (deconvolved)')
ax1.plot(ells_dec, meas_dec, 'C3o', ms=6, label=f'Measured (deconvolved, {num_sim} sims)')
ax1.set_ylabel(r'deconvolved $\hat{C}_b(k{=}0)$')
ax1.legend(fontsize=12)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax2.axhline(1.0, color='k', ls='--', lw=1)
ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ax2.plot(ells_dec, ratios_dec, 'C3o', ms=5)
ax2.set_xlabel(r'multipole $\ell$')
ax2.set_ylabel('data / theory')
ax2.set_ylim(0.7, 1.5)
ax2.text(0.02, 0.92, f'mean ratio = {mean_dec:.3f}',
         transform=ax2.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig(f'{outdir}/money_plot_deconv_final.png', bbox_inches='tight', dpi=150)
plt.savefig(f'{outdir}/money_plot_deconv_final.pdf', bbox_inches='tight')
print('Saved money_plot_deconv_final.png/pdf')
plt.close()

print('\n=== DONE ===')
print(f'Shot noise is NOT subtracted — it is already included in the theory.')
print(f'The pair-counting PLKjKk includes j=k diagonal terms (ℓ-independent floor).')
print(f'The Mll mode-coupling includes the window shot noise floor.')
