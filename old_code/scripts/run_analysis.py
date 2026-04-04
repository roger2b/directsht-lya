"""
Re-run ONLY the analysis/plot cells from master_periodic using the
simulation data saved in the executed notebook.

Note: No shot-noise subtraction. For Ly-α with fixed sightline positions,
Σw²/(4π) is the j=k pair-counting diagonal — cosmological signal, not
Poisson noise. The MASTER framework already accounts for this.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import sys, os

sys.path.insert(0, '/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya')
sys.path.insert(0, '/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks')

from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
from sht.theory_lya import theory_cl_for_deconvolution, compute_chi_bar_from_grid
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
import gc, time

plt.rcParams.update({'font.size': 13, 'figure.figsize': (10, 6)})

# Parameters (matching notebook)
add_rsd_   = False
num_qso    = 9797
chi_shift  = 5000.0
Nl         = 500
lambda_max = 1000
Nx         = 2 * Nl
xmax       = 3./4.
NperBin    = 32
k_idx      = 0
num_sim    = 20

sht_eng = DirectSHT(Nl, Nx, xmax)
print(f'DirectSHT: Nl={Nl}, Nx={Nx}, xmax={xmax}')

# ===== Sim loop (same as notebook) =====
cl_stack = []
wl_ref   = None

for sim_idx in range(num_sim):
    seed = 1000 + sim_idx
    t0 = time.time()

    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed)
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
        Nskew=num_qso, shift=chi_shift)
    all_theta, all_phi = GRF.compute_theta_phi_skewer_start(
        all_x[:, 0], all_y[:, 0], all_z[:, 0])
    chi_grid = all_x[0, :]
    delta_F  = all_w_gal - 1.0

    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)
    N = chi_grid.size

    w_data = FT_delta[:, k_idx]
    hdat = sht_eng(all_theta, all_phi, w_data)
    cl   = hp.alm2cl(hdat)[:Nl]
    cl_stack.append(cl)

    if sim_idx == 0:
        hran       = sht_eng(all_theta, all_phi, FT_mask[:, k_idx])
        wl_ref     = hp.alm2cl(hran)[:Nl]
        chi_grid_0 = chi_grid
        theta_0, phi_0 = all_theta, all_phi
        Nskew_0    = Nskew
        plin_ref   = GRF.plin
        b1_ref     = GRF.my_bias

    del GRF, all_x, all_y, all_z, all_w_rand, all_w_gal, delta_F, FT_mask, FT_delta
    gc.collect()
    print(f'  sim {sim_idx}: seed={seed}, Nskew={Nskew}, dt={time.time()-t0:.1f}s')

cl_stack = np.array(cl_stack)
cl_mean  = np.mean(cl_stack, axis=0)
cl_std   = np.std(cl_stack, axis=0)

dchi    = chi_grid_0[1] - chi_grid_0[0]
L_box   = N * dchi
chi_bar = compute_chi_bar_from_grid(chi_grid_0)
print(f'\nNskew={Nskew_0}, N={N}, L_box={L_box:.1f}, dchi={dchi:.4f}, chi_bar={chi_bar:.1f}')
print(f'b1={b1_ref}')

# ===== Pair-counting theory =====
nhat = sht_lya.compute_nhat(theta_0, phi_0)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
del nhat; gc.collect()

t0 = time.time()
print('\nComputing pair-counting Legendre sums...', end='', flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f'done ({time.time()-t0:.1f}s)')
del cos_theta; gc.collect()

L_range = np.arange(lambda_max, dtype=float)
pk_L = b1_ref**2 * plin_ref(L_range / chi_bar)

t0 = time.time()
couple_pk  = Wigner3j.CoupleMat(lambda_max, pk_L)
coupling_pk = couple_pk.compute_matrix()
print(f'Coupling matrix computed in {time.time()-t0:.1f}s')

C_theory      = coupling_pk @ PLKjKk / (4*np.pi) / (2*np.pi * chi_bar**2)
C_theory_plot = C_theory / (4*np.pi)**2

# ===== Binning & comparison =====
MD   = MaskDeconvolution(Nl, wl_ref)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

binned_theory   = bins @ C_theory_plot[:Nl]
binned_mean     = bins @ cl_mean
binned_std      = bins @ (cl_std / np.sqrt(num_sim))

print(f'\n{"ell":>8s} {"theory":>12s} {"measured":>12s} {"ratio":>10s}')
print('-' * 46)

ratios_raw = []
for i in range(min(15, len(binned_ells))):
    if binned_theory[i] > 0:
        r_raw = binned_mean[i] / binned_theory[i]
        ratios_raw.append(r_raw)
        print(f'{binned_ells[i]:8.1f} {binned_theory[i]:12.4e} '
              f'{binned_mean[i]:12.4e} {r_raw:10.4f}')

mean_ratio_raw_low = np.mean([r for r, e in zip(ratios_raw[1:9], binned_ells[1:9])])
mean_ratio_raw_all = np.mean(ratios_raw[1:])
print(f'\nMean ratio (ell < 288) = {mean_ratio_raw_low:.4f}')
print(f'Mean ratio (all bins)  = {mean_ratio_raw_all:.4f}')

# ===== Money plots =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(binned_ells, binned_theory, 'k.--', lw=2,
        label=r'Theory: $M_{\ell L}\,C_L^{\rm true}$')
for i in range(num_sim):
    ax.plot(binned_ells, bins @ cl_stack[i], 'C0-', alpha=0.15)
ax.errorbar(binned_ells, binned_mean, yerr=binned_std,
            fmt='C3o', ms=5, capsize=3, label=f'Measured mean ({num_sim} sims)')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'$C_\ell(k{=}0)$')
ax.set_title('Pseudo-$C_\\ell$ (pair-counting theory)')
ax.legend(fontsize=10)

ax = axes[1]
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.errorbar(binned_ells[:len(ratios_raw)], np.array(ratios_raw),
            yerr=binned_std[:len(ratios_raw)] / binned_theory[:len(ratios_raw)],
            fmt='C3o', ms=5, capsize=3, label='Measured / theory')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'measured / theory')
ax.set_ylim(0.5, 2.0)
ax.set_title(f'Ratio (low-$\\ell$ mean = {mean_ratio_raw_low:.3f})')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks/plots/money_plot_k0.pdf',
            bbox_inches='tight', dpi=150)
plt.show()
print('Saved money_plot_k0.pdf')

# ===== MaskDeconvolution =====
cl_true_md = b1_ref**2 * plin_ref(ells / chi_bar) / (32 * np.pi**3 * chi_bar**2)

ells_th, theory_dec = MD.convolve_theory_Cls(cl_true_md, bins)
ells_meas, meas_dec = MD(cl_mean, bins)

# Per-realisation scatter for error bars
meas_dec_stack = []
for i in range(num_sim):
    _, md_i = MD(cl_stack[i], bins)
    meas_dec_stack.append(md_i)
meas_dec_stack = np.array(meas_dec_stack)
meas_dec_std = np.std(meas_dec_stack, axis=0) / np.sqrt(num_sim)

print(f'\n{"ell":>8s} {"theory_dec":>12s} {"meas_dec":>12s} {"ratio":>8s}')
print('-' * 44)
ratios_md = []
for i in range(min(15, len(ells_th))):
    if theory_dec[i] > 0:
        r = meas_dec[i] / theory_dec[i]
        ratios_md.append(r)
        print(f'{ells_th[i]:8.1f} {theory_dec[i]:12.4e} '
              f'{meas_dec[i]:12.4e} {r:8.4f}')

print(f'\nDeconvolved ratio (excl. monopole) = {np.mean(ratios_md[1:]):.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(ells_th, theory_dec, 'k.--', lw=2, label='Theory (deconvolved)')
ax.errorbar(ells_meas, meas_dec, yerr=meas_dec_std,
            fmt='C1o', ms=5, capsize=3, label=f'Deconvolved ({num_sim} sims)')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'$\hat{C}_b(k{=}0)$')
ax.set_title('MaskDeconvolution')
ax.legend(fontsize=10)

ax = axes[1]
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.errorbar(ells_th[:len(ratios_md)], ratios_md,
            yerr=meas_dec_std[:len(ratios_md)] / theory_dec[:len(ratios_md)],
            fmt='C1o', ms=5, capsize=3, label='Deconvolved')
ax.plot(binned_ells[:len(ratios_raw)], np.array(ratios_raw),
        'C3s', ms=5, alpha=0.7, label='Pair-counting')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel('measured / theory')
ax.set_ylim(0.5, 2.0)
ax.set_title(f'Ratio (deconv mean = {np.mean(ratios_md[1:]):.3f})')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks/plots/money_plot_deconv_k0.pdf',
            bbox_inches='tight', dpi=150)
plt.show()
print('Saved money_plot_deconv_k0.pdf')

# ===== Summary =====
print('\n=== Pipeline Summary ===')
print(f'Multipoles: ell = 0..{Nl-1}')
print(f'Sightlines: {Nskew_0}')
print(f'Box: N={N}, L={L_box:.1f} Mpc/h, dchi={dchi:.4f} Mpc/h')
print(f'chi_bar = {chi_bar:.1f} Mpc/h')
print(f'Bias: b1 = {b1_ref}, add_rsd = {add_rsd_}')
print(f'Simulations: {num_sim}')
print(f'lambda_max: {lambda_max}')
print(f'\nPair-counting ratio:')
print(f'  low-ell mean: {mean_ratio_raw_low:.4f}')
print(f'  all bins:     {mean_ratio_raw_all:.4f}')
print(f'\nMaskDeconvolution ratio:')
print(f'  excl. monopole: {np.mean(ratios_md[1:]):.4f}')
print(f'\nTheory: C_true = b1^2 * P_lin(ell/chi_bar) / (32 pi^3 chi_bar^2)')
print(f'  32 pi^3 = {32*np.pi**3:.2f} = 2pi x 4pi x 4pi')
print(f'\nNote: No shot-noise subtraction — Sigma w_j^2 / (4 pi) is the')
print(f'j=k diagonal in the pair sum, which is cosmological signal.')
