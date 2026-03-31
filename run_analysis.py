"""
Re-run ONLY the analysis/plot cells from master_periodic using the
simulation data saved in the executed notebook.

Extracts cl_stack, sn_stack, wl_ref etc from the executed notebook's
global kernel state by re-running the sim loop output.
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
sn_stack = []
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

    sn = np.sum(w_data**2) / (4.0 * np.pi)
    sn_stack.append(sn)

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
    print(f'  sim {sim_idx}: seed={seed}, Nskew={Nskew}, SN={sn:.2e}, dt={time.time()-t0:.1f}s')

cl_stack = np.array(cl_stack)
sn_stack = np.array(sn_stack)
sn_mean  = np.mean(sn_stack)

cl_sub_stack = cl_stack - sn_stack[:, None]
cl_mean     = np.mean(cl_stack, axis=0)
cl_std      = np.std(cl_stack, axis=0)
cl_sub_mean = np.mean(cl_sub_stack, axis=0)
cl_sub_std  = np.std(cl_sub_stack, axis=0)

dchi    = chi_grid_0[1] - chi_grid_0[0]
L_box   = N * dchi
chi_bar = compute_chi_bar_from_grid(chi_grid_0)
print(f'\nNskew={Nskew_0}, N={N}, L_box={L_box:.1f}, dchi={dchi:.4f}, chi_bar={chi_bar:.1f}')
print(f'b1={b1_ref}')
print(f'Mean shot noise = {sn_mean:.4e}')
print(f'SN / cl_mean[5]   = {sn_mean / cl_mean[5]:.1%}')
print(f'SN / cl_mean[200] = {sn_mean / cl_mean[200]:.1%}')
print(f'SN / cl_mean[400] = {sn_mean / cl_mean[min(400,Nl-1)]:.1%}')

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
binned_sub_mean = bins @ cl_sub_mean
binned_std      = bins @ (cl_std / np.sqrt(num_sim))
binned_sub_std  = bins @ (cl_sub_std / np.sqrt(num_sim))

print(f'\n{"ell":>8s} {"theory":>12s} {"raw":>12s} {"ratio_raw":>10s} {"SN-sub":>12s} {"ratio_sub":>10s}')
print('-' * 70)

ratios_raw = []
ratios_sub = []
for i in range(min(15, len(binned_ells))):
    if binned_theory[i] > 0:
        r_raw = binned_mean[i] / binned_theory[i]
        r_sub = binned_sub_mean[i] / binned_theory[i]
        ratios_raw.append(r_raw)
        ratios_sub.append(r_sub)
        print(f'{binned_ells[i]:8.1f} {binned_theory[i]:12.4e} '
              f'{binned_mean[i]:12.4e} {r_raw:10.4f} '
              f'{binned_sub_mean[i]:12.4e} {r_sub:10.4f}')

mean_ratio_raw_low = np.mean([r for r, e in zip(ratios_raw[1:9], binned_ells[1:9])])
mean_ratio_sub_low = np.mean([r for r, e in zip(ratios_sub[1:9], binned_ells[1:9])])
mean_ratio_raw_all = np.mean(ratios_raw[1:])
mean_ratio_sub_all = np.mean(ratios_sub[1:])
print(f'\nRaw:    mean ratio (ell < 288) = {mean_ratio_raw_low:.4f}, all = {mean_ratio_raw_all:.4f}')
print(f'SN-sub: mean ratio (ell < 288) = {mean_ratio_sub_low:.4f}, all = {mean_ratio_sub_all:.4f}')
print(f'Shot noise per bin: {sn_mean:.4e}')
print(f'Theory at ell=48: {binned_theory[1]:.4e}')
print(f'SN / Theory at ell=48: {sn_mean / binned_theory[1]:.1%}')
print(f'SN / Theory at ell=464: {sn_mean / binned_theory[-1]:.1%}')

# ===== Money plots =====
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

ax = axes[0, 0]
ax.plot(binned_ells, binned_theory, 'k.--', lw=2,
        label=r'Theory: $C_\ell^{\rm th}/(4\pi)^2$')
for i in range(num_sim):
    ax.plot(binned_ells, bins @ cl_stack[i], 'C0-', alpha=0.15)
ax.errorbar(binned_ells, binned_mean, yerr=binned_std,
            fmt='C3o', ms=5, capsize=3, label=f'Raw mean ({num_sim} sims)')
ax.axhline(sn_mean, color='C2', ls=':', lw=2, label=f'Shot noise = {sn_mean:.2e}')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'$C_\ell(k{=}0)$')
ax.set_title('Raw pseudo-$C_\\ell$')
ax.legend(fontsize=10)

ax = axes[0, 1]
ax.plot(binned_ells, binned_theory, 'k.--', lw=2,
        label=r'Theory: $C_\ell^{\rm th}/(4\pi)^2$')
for i in range(num_sim):
    ax.plot(binned_ells, bins @ cl_sub_stack[i], 'C0-', alpha=0.15)
ax.errorbar(binned_ells, binned_sub_mean, yerr=binned_sub_std,
            fmt='C1o', ms=5, capsize=3, label=f'SN-subtracted mean ({num_sim} sims)')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'$C_\ell(k{=}0) - N_\ell$')
ax.set_title('Shot-noise subtracted')
ax.legend(fontsize=10)

ax = axes[1, 0]
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.errorbar(binned_ells[:len(ratios_raw)], np.array(ratios_raw),
            yerr=binned_std[:len(ratios_raw)] / binned_theory[:len(ratios_raw)],
            fmt='C3o', ms=5, capsize=3, label='Raw')
ax.errorbar(binned_ells[:len(ratios_sub)], np.array(ratios_sub),
            yerr=binned_sub_std[:len(ratios_sub)] / binned_theory[:len(ratios_sub)],
            fmt='C1s', ms=5, capsize=3, label='SN-subtracted')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'measured / theory')
ax.set_ylim(0.5, 2.0)
ax.set_title(f'Raw mean={mean_ratio_raw_low:.3f}, SN-sub mean={mean_ratio_sub_low:.3f} (low $\\ell$)')
ax.legend()

ax = axes[1, 1]
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.errorbar(binned_ells[:len(ratios_raw)], np.array(ratios_raw),
            yerr=binned_std[:len(ratios_raw)] / binned_theory[:len(ratios_raw)],
            fmt='C3o', ms=5, capsize=3, label='Raw')
ax.errorbar(binned_ells[:len(ratios_sub)], np.array(ratios_sub),
            yerr=binned_sub_std[:len(ratios_sub)] / binned_theory[:len(ratios_sub)],
            fmt='C1s', ms=5, capsize=3, label='SN-subtracted')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'measured / theory')
ax.set_ylim(0.8, 1.5)
ax.set_title(f'Zoomed: all-bin raw={mean_ratio_raw_all:.3f}, SN-sub={mean_ratio_sub_all:.3f}')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks/plots/money_plot_k0.pdf',
            bbox_inches='tight', dpi=150)
plt.show()
print('Saved money_plot_k0.pdf')

# ===== MaskDeconvolution =====
cl_true_md = b1_ref**2 * plin_ref(ells / chi_bar) / (32 * np.pi**3 * chi_bar**2)

# Raw
ells_raw, theory_raw_dec = MD.convolve_theory_Cls(cl_true_md, bins)
ells_meas_raw, meas_raw_dec = MD(cl_mean, bins)

# SN-subtracted
ells_sub, theory_sub_dec = MD.convolve_theory_Cls(cl_true_md, bins)
ells_meas_sub, meas_sub_dec = MD(cl_sub_mean, bins)

# Per-realisation scatter for error bars (SN-subtracted)
meas_dec_stack = []
for i in range(num_sim):
    _, md_i = MD(cl_sub_stack[i], bins)
    meas_dec_stack.append(md_i)
meas_dec_stack = np.array(meas_dec_stack)
meas_dec_std = np.std(meas_dec_stack, axis=0) / np.sqrt(num_sim)

print(f'\n{"ell":>8s} {"theory_dec":>12s} {"raw_dec":>12s} {"r_raw":>8s} {"sub_dec":>12s} {"r_sub":>8s}')
print('-' * 66)
ratios_md_raw = []
ratios_md_sub = []
for i in range(min(15, len(ells_sub))):
    if theory_sub_dec[i] > 0:
        r_raw = meas_raw_dec[i] / theory_raw_dec[i]
        r_sub = meas_sub_dec[i] / theory_sub_dec[i]
        ratios_md_raw.append(r_raw)
        ratios_md_sub.append(r_sub)
        print(f'{ells_sub[i]:8.1f} {theory_sub_dec[i]:12.4e} '
              f'{meas_raw_dec[i]:12.4e} {r_raw:8.4f} '
              f'{meas_sub_dec[i]:12.4e} {r_sub:8.4f}')

print(f'\nRaw deconv:    mean ratio (excl. monopole) = {np.mean(ratios_md_raw[1:]):.4f}')
print(f'SN-sub deconv: mean ratio (excl. monopole) = {np.mean(ratios_md_sub[1:]):.4f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(ells_sub, theory_sub_dec, 'k.--', lw=2, label='Theory (deconvolved)')
ax.errorbar(ells_meas_sub, meas_sub_dec, yerr=meas_dec_std,
            fmt='C1o', ms=5, capsize=3, label=f'SN-sub deconvolved ({num_sim} sims)')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel(r'$\hat{C}_b(k{=}0)$')
ax.set_title('MaskDeconvolution (SN-subtracted)')
ax.legend(fontsize=10)

ax = axes[1]
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.errorbar(ells_sub[:len(ratios_md_raw)], ratios_md_raw,
            fmt='C3s', ms=5, capsize=3, alpha=0.7, label='Raw')
ax.errorbar(ells_sub[:len(ratios_md_sub)], ratios_md_sub,
            yerr=meas_dec_std[:len(ratios_md_sub)] / theory_sub_dec[:len(ratios_md_sub)],
            fmt='C1o', ms=5, capsize=3, label='SN-subtracted')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel('measured / theory')
ax.set_ylim(0.5, 2.0)
ax.set_title('Deconvolved ratios')
ax.legend()

ax = axes[2]
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.plot(binned_ells[:len(ratios_raw)], np.array(ratios_raw), 'C3s', ms=5, alpha=0.7, label='Raw (pair-counting)')
ax.plot(binned_ells[:len(ratios_sub)], np.array(ratios_sub), 'C1o', ms=5, label='SN-sub (pair-counting)')
ax.set_xlabel(r'multipole $\ell$')
ax.set_ylabel('measured / theory')
ax.set_ylim(0.5, 2.0)
ax.set_title('Shot noise impact')
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
print(f'Mean shot noise: {sn_mean:.4e}')
print(f'\nPair-counting theory ratio:')
print(f'  Raw    (low ell):  {mean_ratio_raw_low:.4f}')
print(f'  SN-sub (low ell):  {mean_ratio_sub_low:.4f}')
print(f'  Raw    (all bins): {mean_ratio_raw_all:.4f}')
print(f'  SN-sub (all bins): {mean_ratio_sub_all:.4f}')
print(f'\nMaskDeconvolution ratio:')
print(f'  Raw    (excl. monopole): {np.mean(ratios_md_raw[1:]):.4f}')
print(f'  SN-sub (excl. monopole): {np.mean(ratios_md_sub[1:]):.4f}')
print(f'\nTheory: C_true = b1^2 * P_lin(ell/chi_bar) / (32 pi^3 chi_bar^2)')
print(f'  32 pi^3 = {32*np.pi**3:.2f} = 2pi x 4pi x 4pi')
