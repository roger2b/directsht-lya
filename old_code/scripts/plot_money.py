#!/usr/bin/env python
"""
Post-processing: read the 100-sim cache and make money_plot_final.pdf.

Uses the **floor-subtracted MASTER** approach:
  1. Split wl = wl_clust + W_floor, where W_floor = N^2 Nskew/(4pi)
     is the white-noise floor from point-source mask.
  2. Build M_clust from wl_clust only (converges with Nl_large).
  3. Add the diagonal (floor) contribution analytically:
     diag_cl = Nskew * <w^2> / (4pi)  [ell-independent].
  4. C_true = b1^2 P_lin(ell/chi) / (L chi^2)   [from 2D P_2D = b1^2 N^2 P/L].

This resolves the non-convergence of the naive MASTER sum for point-source masks
(where wl does not decay to zero at high lambda).

LoVerde & Afshordi (2008), PRD 78, 123506.
"""
import sys, os, gc, time
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

# ---- Parameters ---- #
Nl              = 500
Nl_large_best   = 2000   # converged: floor-subtracted M doesn't grow
Nl_large_list   = [500, 1000, 2000, 3000]  # show convergence
NperBin         = 32
chi_shift       = 5000
num_qso         = 9797
add_rsd_        = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# ================================================================== #
# Load cached data                                                    #
# ================================================================== #
datafile = os.path.join(root, "notebooks", "data",
                        "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
if not os.path.exists(datafile):
    print(f"Cache not found: {datafile}")
    print("Falling back to 20-sim cache...")
    datafile = os.path.join(root, "notebooks", "data",
                            "Cell_GRF_L1380_N512_Nq9797_Nl500_sims20.npz")

d = np.load(datafile)
cl_k_all = d['cl_k']
wl_k     = d['wl_k']
Nskew    = int(d['Nskew'])
N        = int(d['Nk'])
L_box    = float(d['L'])
num_sim  = cl_k_all.shape[0]
print(f"Loaded {num_sim} sims from {os.path.basename(datafile)}")

wl_ref = wl_k[0, :Nl] if wl_k.shape[1] >= Nl else wl_k[0]

# ================================================================== #
# Cosmology                                                           #
# ================================================================== #
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
beta_ref = GRF_tmp.my_beta
del GRF_tmp; gc.collect()

dchi = L_box / N
# Angular positions (theta, phi) are computed from the FIRST pixel of each
# sightline, at chi_0 = chi_shift.  But sightlines lie on a FLAT PLANE at
# x=chi_0, so each sightline is at distance r_j = sqrt(chi_0^2 + y_j^2 + z_j^2).
# The effective chi for the Limber mapping is <r_j> ≈ sqrt(chi_0^2 + 2L^2/3).
chi_0   = float(chi_shift)
chi_bar = chi_shift + L_box / 2.0       # kept for reference / labeling

# Compute effective chi from sightline geometry
np.random.seed(100)
_inds = np.unique(np.random.randint(0, N, size=(num_qso, 2)), axis=0)
_coords_grid = np.linspace(0, L_box, N)
_r_j = np.sqrt(chi_0**2 + _coords_grid[_inds[:,1]]**2 + _coords_grid[_inds[:,0]]**2)
chi_eff = np.mean(_r_j)
del _inds, _coords_grid, _r_j
print(f"chi_0={chi_0:.1f}, chi_eff=<r_j>={chi_eff:.1f}, chi_bar={chi_bar:.1f}")
print(f"b1={b1_ref:.4f}, beta={beta_ref:.4f}")

# ================================================================== #
# Load precomputed extended angular window (PLKjKk to lambda=4000)   #
# ================================================================== #
PLKjKk_file = os.path.join(root, "notebooks", "data", "PLKjKk_lambda4000.npy")
if not os.path.exists(PLKjKk_file):
    PLKjKk_file = os.path.join(root, "notebooks", "data", "PLKjKk_lambda2000.npy")
if os.path.exists(PLKjKk_file):
    PLKjKk_full = np.load(PLKjKk_file)
    wl_full = PLKjKk_full / (4 * np.pi)
    lambda_max_data = len(PLKjKk_full)
    print(f"  Loaded PLKjKk ({lambda_max_data} multipoles) from cache")
else:
    print(f"  PLKjKk not found; run compute_PLKjKk.py first")
    sys.exit(1)

# ================================================================== #
# White-noise floor and diagonal contribution                        #
# ================================================================== #
W_floor = N**2 * Nskew / (4 * np.pi)
print(f"  W_floor = N^2 Nskew/(4pi) = {W_floor:.4e}")

# Compute the MASTER-consistent floor contribution analytically:
#   floor_cl = W_floor/(4pi) * sum_L (2L+1) C_true(L)
# This is the ℓ-independent constant from the white-noise part of wl.
# We truncate C_true at the box Nyquist scale k_Nyq = pi*N/L to ensure
# convergence (the box has no power beyond k_Nyq).
k_Nyq = np.pi * N / L_box
L_Nyq = k_Nyq * chi_eff
_L_floor = 8000  # large enough for convergence (L_Nyq ~ 5962)
_ells_floor = np.arange(_L_floor, dtype=float)
_kperp_floor = (_ells_floor + 0.5) / chi_eff
_cl_floor = np.where(_kperp_floor < k_Nyq,
                     b1_ref**2 * plin_ref(_kperp_floor) / (L_box * chi_eff**2),
                     0.0)
floor_cl = W_floor / (4 * np.pi) * np.sum((2 * _ells_floor + 1) * _cl_floor)
print(f"  floor_cl (MASTER, L_Nyq={L_Nyq:.0f}) = {floor_cl:.4e}")
print(f"  Fraction of cl_mean[100]: {floor_cl / np.mean(cl_k_all[:, 100]):.3f}")
del _ells_floor, _kperp_floor, _cl_floor

# ================================================================== #
# Theory: Floor-subtracted MASTER approach                           #
# ================================================================== #
# Standard MASTER: <pseudo_Cl> = M @ C_true
# For point-source masks, wl reaches a white-noise floor W_floor at
# high lambda, so the naive MASTER sum DOES NOT converge.
#
# Solution: decompose wl = wl_clust + W_floor, where wl_clust -> 0 at high lambda.
#   <pseudo_Cl> = M_clust @ C_true + floor_cl
# where floor_cl = W_floor/(4pi) * SUM (2L+1) C_true(L), computed
# analytically with C_true truncated at the Nyquist scale.
#
# C_true = b1^2 * P_lin(ell/chi) / (L * chi^2)
# derived from verified P_2D = b1^2 N^2 P/L.
theories = {}  # key: Nl_large -> theory_cl[:Nl]

for Nl_large in Nl_large_list:
    t1 = time.time()
    
    # Build wl_clust for the coupling matrix
    wl_needed = 2 * Nl_large - 1
    wl_raw = np.zeros(wl_needed)
    n_avail = min(wl_needed, lambda_max_data)
    wl_raw[:n_avail] = wl_full[:n_avail]
    wl_raw[n_avail:] = W_floor  # fill beyond data with floor
    wl_clust = wl_raw - W_floor
    
    # C_true at extended ell' range (use chi_eff = <r_j> for flat-plane geometry)
    ells_ext = np.arange(Nl_large, dtype=float)
    cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_eff) \
                  / (L_box * chi_eff**2)
    
    # Coupling matrix from wl_clust (floor-subtracted)
    couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
    M_clust = couple.compute_matrix()
    
    # Theory = M_clust @ C_true + floor_cl (analytical floor)
    theories[Nl_large] = (M_clust @ cl_true_ext)[:Nl] + floor_cl
    
    print(f"  Nl_large={Nl_large}: coupling in {time.time()-t1:.1f}s")
    del couple, M_clust; gc.collect()

print(f"Theory computed for Nl_large = {Nl_large_list}")

# ================================================================== #
# Binning & statistics                                                #
# ================================================================== #
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

cl_mean = np.mean(cl_k_all, axis=0)
cl_std  = np.std(cl_k_all, axis=0) / np.sqrt(num_sim)

binned_raw = bins @ cl_mean
binned_std = bins @ cl_std

# Compute ratios for each Nl_large
all_ratios = {}   # key: Nl_large -> (ratio_ells, ratios, ratio_errs, binned_theory)
for Nl_large in Nl_large_list:
    bt = bins @ theories[Nl_large]
    r_ells, rats, r_errs = [], [], []
    for i in range(len(binned_ells)):
        if bt[i] > 0:
            r_ells.append(binned_ells[i])
            rats.append(binned_raw[i] / bt[i])
            r_errs.append(binned_std[i] / bt[i])
    all_ratios[Nl_large] = (np.array(r_ells), np.array(rats),
                            np.array(r_errs), bt)

# Print summary table
print(f"\n{'ell':>6s}", end="")
for Nl_large in Nl_large_list:
    print(f"  {'r('+str(Nl_large)+')':>10s}", end="")
print(f"  {'raw':>12s}")
print("-" * (6 + 12 * len(Nl_large_list) + 14))

for i in range(len(all_ratios[Nl_large_list[0]][0])):
    e = all_ratios[Nl_large_list[0]][0][i]
    print(f"{e:6.0f}", end="")
    for Nl_large in Nl_large_list:
        r = all_ratios[Nl_large][1][i]
        print(f"  {r:10.4f}", end="")
    print(f"  {binned_raw[i]:12.4e}")

for Nl_large in Nl_large_list:
    rr = all_ratios[Nl_large][1]
    re = all_ratios[Nl_large][0]
    print(f"\nNl_large={Nl_large}: mean ratio (excl mono) = {np.mean(rr[1:]):.4f} ± {np.std(rr[1:]):.4f}")

# ---- Floor-subtracted MaskDeconvolution ---- #
# Deconvolve the clustered part (cl_mean - diag_cl) with M_clust,
# then compare with C_true = b1^2 P(ell/chi) / (L chi^2).
wl_clust_md = np.zeros(2*Nl-1)
n_av = min(2*Nl-1, lambda_max_data)
wl_raw_md = np.zeros(2*Nl-1)
wl_raw_md[:n_av] = wl_full[:n_av]
wl_raw_md[n_av:] = W_floor
wl_clust_md = wl_raw_md - W_floor

couple_clust_md = Wigner3j.CoupleMat(Nl, wl_clust_md)
M_clust_md = couple_clust_md.compute_matrix()
MD_clust = MaskDeconvolution(Nl, wl_clust_md, precomputed_Wigner=M_clust_md)

# Deconvolve each sim individually for error bars
dec_all = np.zeros((num_sim, len(bins)))
for isim in range(num_sim):
    cl_clust_i = cl_k_all[isim] - floor_cl
    _, dec_all[isim] = MD_clust(cl_clust_i, bins)

meas_dec = np.mean(dec_all, axis=0)
meas_dec_std = np.std(dec_all, axis=0) / np.sqrt(num_sim)
ells_dec = MD_clust(cl_k_all[0] - floor_cl, bins)[0]

cl_true_limber = b1_ref**2 * plin_ref((ells + 0.5) / chi_eff) / (L_box * chi_eff**2)
_, theory_dec = MD_clust.convolve_theory_Cls(cl_true_limber, bins)

ratios_md = []
ratios_md_err = []
print(f"\n--- Floor-subtracted MaskDeconvolution ---")
print(f"{'ell':>6s} {'theory':>12s} {'meas':>12s} {'err':>12s} {'ratio':>8s}")
print(f"{'-'*52}")
for i in range(len(ells_dec)):
    if theory_dec[i] > 0:
        r = meas_dec[i] / theory_dec[i]
        re = meas_dec_std[i] / theory_dec[i]
        ratios_md.append(r)
        ratios_md_err.append(re)
        print(f"{ells_dec[i]:6.0f} {theory_dec[i]:12.4e} "
              f"{meas_dec[i]:12.4e} {meas_dec_std[i]:12.4e} {r:8.4f}")
print(f"\nFloor-sub MaskDeconv: mean ratio = {np.mean(ratios_md[1:]):.4f}")

# ================================================================== #
# PLOT 1: Money plot — floor-subtracted MASTER convergence           #
# ================================================================== #
plotdir = os.path.join(root, "notebooks", "plots")
os.makedirs(plotdir, exist_ok=True)

colors_nl  = {500: 'C1', 1000: 'C2', 2000: 'C4', 3000: 'C3'}
markers_nl = {500: 'v',  1000: 's',  2000: 'D',  3000: '^'}

fig, axes = plt.subplots(2, 1, figsize=(11, 9),
                         gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
ax1, ax2 = axes
fig.subplots_adjust(hspace=0.05)

# Top panel: pseudo-Cl with individual sims
for i in range(min(num_sim, 100)):
    bn = bins @ cl_k_all[i]
    ax1.plot(binned_ells, bn, 'k-', alpha=0.05, lw=0.3)

ax1.errorbar(binned_ells, binned_raw, yerr=binned_std,
             fmt='o', color='C0', ms=5, capsize=3, zorder=10,
             label=f'Measured mean ({num_sim} sims)')

for Nl_large in Nl_large_list:
    bt = all_ratios[Nl_large][3]
    ax1.plot(binned_ells, bt, f'{markers_nl[Nl_large]}--',
             color=colors_nl[Nl_large], lw=1.5, ms=4,
             label=rf'MASTER $N_{{\ell\prime}}$={Nl_large}')

ax1.set_ylabel(r'binned pseudo-$C_\ell(k{=}0)$', fontsize=14)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_title(f'Floor-subtracted MASTER: {num_sim} sims, $N_\\ell$={Nl}, '
              f'$N_{{\\rm skew}}$={Nskew}, $b_1$={b1_ref:.4f}',
              fontsize=13)
ax1.tick_params(labelbottom=False)

# Bottom panel: ratios for each Nl_large
ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15)

for idx, Nl_large in enumerate(Nl_large_list):
    re, rr, rerr, _ = all_ratios[Nl_large]
    offset = idx * 1.5
    ax2.errorbar(re + offset, rr, yerr=rerr,
                 fmt=markers_nl[Nl_large], color=colors_nl[Nl_large], ms=4, capsize=2,
                 label=rf'$N_{{\ell\prime}}$={Nl_large}')

ax2.set_xlabel(r'multipole $\ell$', fontsize=14)
ax2.set_ylabel('measured / theory', fontsize=14)
ax2.set_ylim(0.85, 1.25)
ax2.legend(fontsize=9, loc='upper left', ncol=len(Nl_large_list))

plt.savefig(os.path.join(plotdir, "money_plot_final.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(plotdir, "money_plot_final.png"), bbox_inches='tight', dpi=150)
print(f"\nSaved money_plot_final.pdf/.png to {plotdir}")
plt.close()

# ================================================================== #
# PLOT 2: Floor-subtracted MaskDeconvolution                         #
# ================================================================== #
fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8),
                           gridspec_kw={'height_ratios': [3, 1]},
                           sharex=True)
ax5, ax6 = axes3
fig3.subplots_adjust(hspace=0.05)

ax5.plot(ells_dec, theory_dec, 'k.--', lw=2,
         label=r'Theory $C_\ell = b_1^2 P(\ell/\bar\chi) / (L\bar\chi^2)$')
ax5.errorbar(ells_dec, meas_dec, yerr=meas_dec_std, fmt='o', color='C0', ms=5,
             capsize=3, label='Measured (deconvolved, floor-subtracted)')
ax5.set_ylabel(r'deconvolved $C_\ell(k{=}0)$', fontsize=14)
ax5.legend(fontsize=10)
ax5.set_title(f'Floor-subtracted MaskDeconvolution ($N_\\ell$={Nl}): {num_sim} sims', fontsize=13)
ax5.tick_params(labelbottom=False)

ax6.axhline(1, color='k', ls='--', lw=0.8)
ax6.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ratios_md_arr = np.array(ratios_md)
ratios_md_err_arr = np.array(ratios_md_err)
ax6.errorbar(ells_dec[:len(ratios_md_arr)], ratios_md_arr, yerr=ratios_md_err_arr,
             fmt='o', color='C0', ms=5, capsize=2)
ax6.set_xlabel(r'multipole $\ell$', fontsize=14)
ax6.set_ylabel('measured / theory', fontsize=14)
ax6.set_ylim(0.85, 1.35)

plt.savefig(os.path.join(plotdir, "money_plot_deconv_final.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(plotdir, "money_plot_deconv_final.png"), bbox_inches='tight', dpi=150)
print(f"Saved money_plot_deconv_final.pdf/.png")
plt.close()

print("\nDone!")
