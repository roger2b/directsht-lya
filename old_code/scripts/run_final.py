#!/usr/bin/env python
"""
Money plots using cached 20-sim data.
Uses the existing Cell_GRF...sims20.npz for cl_k and wl_k.

Note: No shot-noise subtraction. For Ly-alpha with fixed sightline positions,
Sigma w_j^2 / (4 pi) is the j=k pair-counting diagonal -- cosmological signal,
not Poisson noise. The MASTER framework already accounts for this.
"""
import sys, os, gc, time, subprocess, json
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

# ---- Parameters ---- #
Nl         = 500
lambda_max = 1000
num_sim    = 20
k_idx      = 0
add_rsd_   = False
NperBin    = 32
chi_shift  = 5000
num_qso    = 9797
sn_cache   = os.path.join(root, "_sn_cache.npy")
datafile   = os.path.join(root, "notebooks", "data",
                          "Cell_GRF_L1380_N512_Nq9797_Nl500_sims20.npz")

# ================================================================== #
# PHASE 1: Load cached data                                          #
# ================================================================== #
print("Loading cached 20-sim data...")
d = np.load(datafile)
cl_k   = d['cl_k']      # (20, 500) raw pseudo-Cl per sim
wl_k   = d['wl_k']      # (20, 564) window per sim (all same)
Nskew  = int(d['Nskew'])
N      = int(d['Nk'])
L_box  = float(d['L'])
binned_ells_ref = d['binned_ells']
print(f"  {cl_k.shape[0]} sims, Nskew={Nskew}, N={N}, L_box={L_box}")

wl_ref = wl_k[0, :Nl]  # all sims have same window

# ================================================================== #
# PHASE 1b: Compute shot noise per sim (in subprocess to avoid OOM)  #
# ================================================================== #
if os.path.exists(sn_cache):
    print(f"Loading SN cache from {sn_cache}")
    sn_stack = np.load(sn_cache)
    print(f"  SN values: {sn_stack}")
else:
    print(f"\n--- Computing shot noise per sim (via subprocess) ---")
    # Write a small helper script that generates one sim, computes SN, exits
    helper = os.path.join(root, "_sn_helper.py")
    with open(helper, 'w') as f:
        f.write(f'''#!/usr/bin/env python
import sys, os, gc
sys.path.insert(0, "{root}")
sys.path.insert(0, os.path.join("{root}", "notebooks"))
import numpy as np
import GRF_class as my_GRF
import SHT_lya as sht_lya

seed = int(sys.argv[1])
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew={num_qso}, shift={chi_shift})
chi_grid = all_x[0, :]
delta_F = all_w_gal - 1.0
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)
w_j = FT_delta[:, 0]
sn = float(np.sum(w_j**2) / (4.0 * np.pi))
print(sn, flush=True)
''')

    sn_stack = np.zeros(num_sim)
    for i in range(num_sim):
        seed = 1000 + i
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, helper, str(seed)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  sim {i} FAILED: {result.stderr[:200]}")
            sys.exit(1)
        # Parse last line (skip JAX warnings etc)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        sn_stack[i] = float(lines[-1])
        dt = time.time() - t0
        print(f"  sim {i:2d}: seed={seed}, SN={sn_stack[i]:.4e}, dt={dt:.1f}s")

    np.save(sn_cache, sn_stack)
    os.remove(helper)
    print(f"Saved SN cache to {sn_cache}")

# ================================================================== #
# PHASE 2: Theory (pair-counting with lambda_max=1000)                #
# ================================================================== #
print(f"\n--- Phase 2: Theory (lambda_max={lambda_max}) ---")

import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Get cosmology params
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

dchi = L_box / N
chi_bar = chi_shift + L_box / 2.0
print(f"  dchi={dchi:.4f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# Sightline positions (same for all sims — seed(100))
GRF_pos = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=1000)
all_x, all_y, all_z, _, _, _ = GRF_pos.process_skewers(Nskew=num_qso, shift=chi_shift)
theta_ref, phi_ref = GRF_pos.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
del GRF_pos, all_x, all_y, all_z; gc.collect()

nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
del nhat; gc.collect()
print(f"  cos_theta: {cos_theta.shape}, {cos_theta.nbytes/1e9:.1f} GB")

t0 = time.time()
print(f"  Legendre sums (lambda_max={lambda_max})...", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f"  Done in {time.time()-t0:.1f}s")
del cos_theta; gc.collect()

L_range = np.arange(lambda_max, dtype=float)
pk_L = b1_ref**2 * plin_ref(L_range / chi_bar)

t0 = time.time()
print("  Wigner coupling matrix...", flush=True)
couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
coupling_pk = couple_pk.compute_matrix()
print(f"  Done in {time.time()-t0:.1f}s")

C_theory = coupling_pk @ PLKjKk / (4 * np.pi) / (2 * np.pi * chi_bar**2)
C_theory_plotted = C_theory / (4 * np.pi)**2

# ================================================================== #
# PHASE 3: Binning and comparison                                    #
# ================================================================== #
print(f"\n--- Phase 3: Binning & comparison ---")

cl_mean = np.mean(cl_k, axis=0)
cl_sub_stack = cl_k - sn_stack[:, None]
cl_sub_mean = np.mean(cl_sub_stack, axis=0)
cl_sub_std = np.std(cl_sub_stack, axis=0)
sn_mean = np.mean(sn_stack)

print(f"b1={b1_ref:.4f}")

MD = MaskDeconvolution(Nl, wl_ref)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

binned_theory = bins @ C_theory_plotted[:Nl]
binned_raw = bins @ cl_mean
binned_sub = bins @ cl_sub_mean
binned_sn = sn_mean * np.sum(bins, axis=1)
binned_sub_std = bins @ cl_sub_std / np.sqrt(num_sim)

print(f"\n{'='*80}")
print(f"RESULTS: k=0, {num_sim} sims, Nl={Nl}, lambda_max={lambda_max}, Nskew={Nskew}")
print(f"{'='*80}")
print(f"{'ell':>6s} {'theory':>12s} {'raw':>12s} {'SN-sub':>12s} "
      f"{'SN%':>7s} {'r(raw)':>8s} {'r(sub)':>8s} {'SNR':>6s}")
print(f"{'-'*76}")

ratios_raw, ratios_sub = [], []
for i in range(len(binned_ells)):
    if binned_theory[i] > 0:
        r_raw = binned_raw[i] / binned_theory[i]
        r_sub = binned_sub[i] / binned_theory[i]
        snr = binned_sub[i] / binned_sub_std[i] if binned_sub_std[i] > 0 else np.inf
        frac_sn = binned_sn[i] / binned_raw[i] * 100 if binned_raw[i] > 0 else 0
        ratios_raw.append(r_raw)
        ratios_sub.append(r_sub)
        print(f"{binned_ells[i]:6.0f} {binned_theory[i]:12.4e} {binned_raw[i]:12.4e} "
              f"{binned_sub[i]:12.4e} {frac_sn:6.1f}% {r_raw:8.4f} {r_sub:8.4f} {snr:6.1f}")

mean_raw = np.mean(ratios_raw[1:])
mean_sub = np.mean(ratios_sub[1:])
mean_raw_low = np.mean([r for r, e in zip(ratios_raw[1:], binned_ells[1:]) if e < 288])
mean_sub_low = np.mean([r for r, e in zip(ratios_sub[1:], binned_ells[1:]) if e < 288])

print(f"\nMean ratio (excl mono): all={mean_raw:.4f}, low-ell(<288)={mean_raw_low:.4f}")
print(f"(SN-sub for reference only: all={mean_sub:.4f}, low-ell={mean_sub_low:.4f})")

# ---- MaskDeconvolution approach ---- #
print(f"\n--- MaskDeconvolution (deconvolved) ---")
cl_true_md = b1_ref**2 * plin_ref(ells / chi_bar) / (32 * np.pi**3 * chi_bar**2)
ells_dec, theory_dec = MD.convolve_theory_Cls(cl_true_md, bins)
_, meas_raw_dec = MD(cl_mean, bins)
_, meas_sub_dec = MD(cl_sub_mean, bins)

print(f"{'ell':>6s} {'th_dec':>12s} {'raw_dec':>12s} {'sub_dec':>12s} "
      f"{'r(raw)':>8s} {'r(sub)':>8s}")
print(f"{'-'*56}")
ratios_md_raw, ratios_md_sub = [], []
for i in range(len(ells_dec)):
    if theory_dec[i] > 0:
        r_raw = meas_raw_dec[i] / theory_dec[i]
        r_sub = meas_sub_dec[i] / theory_dec[i]
        ratios_md_raw.append(r_raw)
        ratios_md_sub.append(r_sub)
        print(f"{ells_dec[i]:6.0f} {theory_dec[i]:12.4e} {meas_raw_dec[i]:12.4e} "
              f"{meas_sub_dec[i]:12.4e} {r_raw:8.4f} {r_sub:8.4f}")

print(f"\nMaskDeconv mean ratio: {np.mean(ratios_md_raw[1:]):.4f}")
print(f"(SN-sub for reference: {np.mean(ratios_md_sub[1:]):.4f})")

# ================================================================== #
# PHASE 4: Plots                                                     #
# ================================================================== #
print(f"\n--- Generating plots ---")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plotdir = os.path.join(root, "notebooks", "plots")
os.makedirs(plotdir, exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1, ax2 = axes

for i in range(num_sim):
    bn = bins @ cl_sub_stack[i]
    ax1.plot(binned_ells, bn, 'k-', alpha=0.12, lw=0.5)

ax1.errorbar(binned_ells, binned_sub, yerr=binned_sub_std,
             fmt='o', color='C3', ms=5, capsize=3, label=f'SN-subtracted mean ({num_sim} sims)')
ax1.errorbar(binned_ells + 1, binned_raw, fmt='s', color='C0', ms=4, alpha=0.4,
             label='Raw mean (no SN sub)')
ax1.plot(binned_ells, binned_theory, 'k.--', lw=2, label=r'Theory: $C_\ell^{\rm th}/(4\pi)^2$')
ax1.set_ylabel(r'binned pseudo-$C_\ell(k{=}0)$', fontsize=14)
ax1.legend(fontsize=11)
ax1.set_title(f'Money plot: {num_sim} sims, Nl={Nl}, $\\lambda_{{\\rm max}}$={lambda_max}, '
              f'Nskew={Nskew}, $b_1$={b1_ref:.4f}', fontsize=13)

ax2.axhline(1, color='k', ls='--', lw=0.5)
ax2.errorbar(binned_ells, np.array(ratios_sub),
             yerr=binned_sub_std / binned_theory, fmt='o', color='C3', ms=5, capsize=3,
             label='SN-subtracted')
ax2.plot(binned_ells, ratios_raw, 's', color='C0', ms=4, alpha=0.4, label='Raw')
ax2.set_xlabel(r'multipole $\ell$', fontsize=14)
ax2.set_ylabel('data / theory', fontsize=14)
ax2.set_ylim(0.5, 1.8)
ax2.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plotdir, "money_plot_SN.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(plotdir, "money_plot_SN.png"), bbox_inches='tight', dpi=150)
print("  Saved money_plot_SN.pdf/.png")
plt.close()

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
ax3, ax4 = axes2

ax3.plot(ells_dec, theory_dec, 'k.--', lw=2, label='Theory (deconvolved)')
ax3.plot(ells_dec, meas_sub_dec, 'o', color='C3', ms=5, label='SN-sub measured (deconv)')
ax3.plot(ells_dec, meas_raw_dec, 's', color='C0', ms=4, alpha=0.4, label='Raw measured (deconv)')
ax3.set_ylabel(r'deconvolved $C_\ell(k{=}0)$', fontsize=14)
ax3.legend(fontsize=11)
ax3.set_title(f'MaskDeconvolution: {num_sim} sims, $\\lambda_{{\\rm max}}$={lambda_max}', fontsize=13)

ax4.axhline(1, color='k', ls='--', lw=0.5)
ax4.plot(ells_dec, ratios_md_sub, 'o', color='C3', ms=5, label='SN-subtracted')
ax4.plot(ells_dec, ratios_md_raw, 's', color='C0', ms=4, alpha=0.4, label='Raw')
ax4.set_xlabel(r'multipole $\ell$', fontsize=14)
ax4.set_ylabel('data / theory', fontsize=14)
ax4.set_ylim(0.0, 2.5)
ax4.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plotdir, "money_plot_deconv_SN.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(plotdir, "money_plot_deconv_SN.png"), bbox_inches='tight', dpi=150)
print("  Saved money_plot_deconv_SN.pdf/.png")
plt.close()

print("\nDone!")
