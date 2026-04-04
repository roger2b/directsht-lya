#!/usr/bin/env python
"""
100-sim production run for money_plot_final.pdf.

Strategy:
  - Reuse the 20 cached sims from Cell_GRF_L1380_N512_Nq9797_Nl500_sims20.npz
  - Run 80 more sims (seeds 1020-1099) via subprocess-per-sim for memory safety
  - Combine all 100 sims, compute theory, produce money_plot_final.pdf

No shot-noise subtraction (it's cosmological signal, not Poisson noise).
"""
import sys, os, gc, time, subprocess, tempfile
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

# ---- Parameters ---- #
Nl         = 500
lambda_max = 1000
num_sim    = 100
k_idx      = 0
add_rsd_   = False
NperBin    = 32
chi_shift  = 5000
num_qso    = 9797

datafile_20 = os.path.join(root, "notebooks", "data",
                           "Cell_GRF_L1380_N512_Nq9797_Nl500_sims20.npz")
outfile     = os.path.join(root, "notebooks", "data",
                           "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
plotdir     = os.path.join(root, "notebooks", "plots")
os.makedirs(plotdir, exist_ok=True)

# ================================================================== #
# PHASE 1: Load the 20 cached sims                                   #
# ================================================================== #
print("="*80)
print("Loading cached 20-sim data...")
d = np.load(datafile_20)
cl_k_20 = d['cl_k']       # (20, 500)
wl_k_20 = d['wl_k']       # (20, 564) — all identical
Nskew   = int(d['Nskew'])
N       = int(d['Nk'])
L_box   = float(d['L'])
print(f"  {cl_k_20.shape[0]} sims loaded, Nskew={Nskew}, N={N}, L_box={L_box}")

wl_ref = wl_k_20[0, :Nl]

# ================================================================== #
# PHASE 2: Run 80 more sims via subprocess                           #
# ================================================================== #
num_extra = num_sim - cl_k_20.shape[0]
seed_start = 1000 + cl_k_20.shape[0]  # seeds 1020-1099
print(f"\n{'='*80}")
print(f"Running {num_extra} extra sims (seeds {seed_start}-{seed_start+num_extra-1})...")

# Write worker script to a temp file
worker_code = f'''#!/usr/bin/env python
import sys, os
sys.path.insert(0, "{root}")
sys.path.insert(0, os.path.join("{root}", "notebooks"))
import numpy as np
from sht.sht import DirectSHT
from sht.lya_sfb import LyaSFB
import GRF_class as my_GRF
import SHT_lya as sht_lya

seed = int(sys.argv[1])
Nl = {Nl}
chi_shift = {chi_shift}
num_qso = {num_qso}

# Generate GRF
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew=num_qso, shift=chi_shift)

chi_grid = all_x[0, :]
delta_F = all_w_gal - 1.0

# DFT (unnormalized, real-part only — matching original convention)
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)

# Sightline directions
theta, phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])

# SHT at k=0
sht_engine = DirectSHT(Nl, 2*Nl, 0.75)
w_j = FT_delta[:, 0]         # data weights at k=0
alm_data = sht_engine(theta, phi, w_j)

from sht.lya_sfb import _alm2cl_complex
cl = _alm2cl_complex(alm_data, Nl)

# Print as JSON-style line for reliable parsing
import json
print("RESULT:" + json.dumps(cl.tolist()), flush=True)
'''

worker_file = os.path.join(root, "_worker_100.py")
with open(worker_file, 'w') as f:
    f.write(worker_code)

cl_k_extra = np.zeros((num_extra, Nl))
t_total = time.time()

for i in range(num_extra):
    seed = seed_start + i
    sim_idx = cl_k_20.shape[0] + i
    t0 = time.time()
    
    result = subprocess.run(
        [sys.executable, worker_file, str(seed)],
        capture_output=True, text=True, timeout=600
    )
    
    if result.returncode != 0:
        print(f"  sim {sim_idx} (seed={seed}) FAILED:")
        print(result.stderr[:500])
        sys.exit(1)
    
    # Parse result line
    for line in result.stdout.strip().split('\n'):
        if line.startswith("RESULT:"):
            import json
            cl_k_extra[i] = np.array(json.loads(line[7:]))
            break
    else:
        print(f"  sim {sim_idx}: No RESULT line found in output!")
        print(result.stdout[-500:])
        sys.exit(1)
    
    dt = time.time() - t0
    elapsed = time.time() - t_total
    eta = elapsed / (i+1) * (num_extra - i - 1)
    print(f"  sim {sim_idx:3d} (seed={seed}): cl[1]={cl_k_extra[i,1]:.4e}, "
          f"dt={dt:.0f}s, ETA={eta/60:.1f}min", flush=True)

os.remove(worker_file)
dt_total = time.time() - t_total
print(f"\nAll {num_extra} sims done in {dt_total/60:.1f} minutes")

# ================================================================== #
# PHASE 3: Combine and save                                          #
# ================================================================== #
cl_k_all = np.vstack([cl_k_20, cl_k_extra])  # (100, 500)
print(f"\nCombined cl_k shape: {cl_k_all.shape}")

np.savez(outfile,
         cl_k=cl_k_all, wl_k=wl_k_20[:1],
         Nskew=Nskew, Nk=N, L=L_box)
print(f"Saved to {outfile}")

# ================================================================== #
# PHASE 4: Theory (pair-counting with lambda_max=1000)                #
# ================================================================== #
print(f"\n{'='*80}")
print(f"Computing theory (lambda_max={lambda_max})...")

import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

dchi = L_box / N
chi_bar = chi_shift + L_box / 2.0
print(f"  dchi={dchi:.4f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# Sightline positions (deterministic — seed(100) inside GRF_class)
GRF_pos = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=1000)
all_x, all_y, all_z, _, _, _ = GRF_pos.process_skewers(Nskew=num_qso, shift=chi_shift)
theta_ref, phi_ref = GRF_pos.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
del GRF_pos, all_x, all_y, all_z; gc.collect()

nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
del nhat; gc.collect()

t0 = time.time()
print(f"  Legendre sums (lambda_max={lambda_max})...", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f"  Done in {time.time()-t0:.1f}s")
del cos_theta; gc.collect()

L_range = np.arange(lambda_max, dtype=float)

# --- Standard Limber: k_perp = L / chi_bar ---
pk_L = b1_ref**2 * plin_ref(L_range / chi_bar)

t0 = time.time()
print("  Wigner coupling matrix (standard Limber)...", flush=True)
couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
coupling_pk = couple_pk.compute_matrix()
print(f"  Done in {time.time()-t0:.1f}s")

C_theory = coupling_pk @ PLKjKk / (4 * np.pi) / (2 * np.pi * chi_bar**2)
C_theory_plotted = C_theory / (4 * np.pi)**2

# --- Extended Limber: k_perp = (L + 1/2) / chi_bar  [LoVerde & Afshordi 2008] ---
pk_L_ext = b1_ref**2 * plin_ref((L_range + 0.5) / chi_bar)

t0 = time.time()
print("  Wigner coupling matrix (extended Limber)...", flush=True)
couple_pk_ext = Wigner3j.CoupleMat(lambda_max, pk_L_ext)
coupling_pk_ext = couple_pk_ext.compute_matrix()
print(f"  Done in {time.time()-t0:.1f}s")

C_theory_ext = coupling_pk_ext @ PLKjKk / (4 * np.pi) / (2 * np.pi * chi_bar**2)
C_theory_ext_plotted = C_theory_ext / (4 * np.pi)**2

# ================================================================== #
# PHASE 5: Binning, statistics, and comparison                       #
# ================================================================== #
print(f"\n{'='*80}")
print(f"Binning & statistics...")

MD = MaskDeconvolution(Nl, wl_ref)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

cl_mean = np.mean(cl_k_all, axis=0)
cl_std  = np.std(cl_k_all, axis=0) / np.sqrt(num_sim)

binned_theory     = bins @ C_theory_plotted[:Nl]
binned_theory_ext = bins @ C_theory_ext_plotted[:Nl]
binned_raw        = bins @ cl_mean
binned_std        = bins @ cl_std

print(f"\n{'ell':>6s} {'th(L)':>12s} {'th(L+½)':>12s} {'raw':>12s} "
      f"{'r(L)':>8s} {'r(L+½)':>8s} {'err':>12s}")
print(f"{'-'*72}")
ratios = []
ratios_ext = []
for i in range(len(binned_ells)):
    if binned_theory[i] > 0:
        r     = binned_raw[i] / binned_theory[i]
        r_ext = binned_raw[i] / binned_theory_ext[i]
        ratios.append(r)
        ratios_ext.append(r_ext)
        print(f"{binned_ells[i]:6.0f} {binned_theory[i]:12.4e} {binned_theory_ext[i]:12.4e} "
              f"{binned_raw[i]:12.4e} {r:8.4f} {r_ext:8.4f} {binned_std[i]:12.4e}")

mean_ratio = np.mean(ratios[1:])
mean_ratio_ext = np.mean(ratios_ext[1:])
mean_ratio_low = np.mean([r for r, e in zip(ratios[1:], binned_ells[1:]) if e < 288])
mean_ratio_ext_low = np.mean([r for r, e in zip(ratios_ext[1:], binned_ells[1:]) if e < 288])
print(f"\nStandard Limber  (L/chi): all={mean_ratio:.4f}, low-ell(<288)={mean_ratio_low:.4f}")
print(f"Extended Limber (L+½/chi): all={mean_ratio_ext:.4f}, low-ell(<288)={mean_ratio_ext_low:.4f}")

# ---- MaskDeconvolution ---- #
print(f"\n--- MaskDeconvolution (deconvolved) ---")
cl_true_md     = b1_ref**2 * plin_ref(ells / chi_bar) / (32 * np.pi**3 * chi_bar**2)
cl_true_md_ext = b1_ref**2 * plin_ref((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
ells_dec, theory_dec     = MD.convolve_theory_Cls(cl_true_md, bins)
_,        theory_dec_ext = MD.convolve_theory_Cls(cl_true_md_ext, bins)
_, meas_dec = MD(cl_mean, bins)

ratios_md = []
ratios_md_ext = []
print(f"{'ell':>6s} {'th(L)':>12s} {'th(L+½)':>12s} {'meas':>12s} {'r(L)':>8s} {'r(L+½)':>8s}")
print(f"{'-'*62}")
for i in range(len(ells_dec)):
    if theory_dec[i] > 0:
        r     = meas_dec[i] / theory_dec[i]
        r_ext = meas_dec[i] / theory_dec_ext[i]
        ratios_md.append(r)
        ratios_md_ext.append(r_ext)
        print(f"{ells_dec[i]:6.0f} {theory_dec[i]:12.4e} {theory_dec_ext[i]:12.4e} "
              f"{meas_dec[i]:12.4e} {r:8.4f} {r_ext:8.4f}")
print(f"\nMaskDeconv standard Limber: {np.mean(ratios_md[1:]):.4f}")
print(f"MaskDeconv extended Limber: {np.mean(ratios_md_ext[1:]):.4f}")

# ================================================================== #
# PHASE 6: Money plot                                                 #
# ================================================================== #
print(f"\n{'='*80}")
print("Generating money_plot_final.pdf...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                         gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
ax1, ax2 = axes
fig.subplots_adjust(hspace=0.05)

# --- Top panel: pseudo-Cl ---
# Individual sims (thin lines)
for i in range(min(num_sim, 100)):
    bn = bins @ cl_k_all[i]
    ax1.plot(binned_ells, bn, 'k-', alpha=0.07, lw=0.4)

ax1.errorbar(binned_ells, binned_raw, yerr=binned_std,
             fmt='o', color='C0', ms=5, capsize=3, zorder=10,
             label=f'Measured mean ({num_sim} sims)')
ax1.plot(binned_ells, binned_theory, 'k.--', lw=2,
         label=r'Theory $\ell/\bar\chi$ (standard Limber)')
ax1.plot(binned_ells, binned_theory_ext, 'r.:', lw=2,
         label=r'Theory $(\ell{+}1/2)/\bar\chi$ (extended Limber)')
ax1.set_ylabel(r'binned pseudo-$C_\ell(k{=}0)$', fontsize=14)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_title(f'{num_sim} sims, $N_\\ell$={Nl}, $\\lambda_{{\\rm max}}$={lambda_max}, '
              f'$N_{{\\rm skew}}$={Nskew}, $b_1$={b1_ref:.4f}', fontsize=13)
ax1.tick_params(labelbottom=False)

# --- Bottom panel: ratio ---
ratio_err = binned_std / binned_theory
ratio_err_ext = binned_std / binned_theory_ext
ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ax2.errorbar(binned_ells, ratios, yerr=ratio_err[1:] if len(ratio_err)>1 else None,
             fmt='o', color='C0', ms=5, capsize=3, label=r'data / theory($\ell/\bar\chi$)')
ax2.errorbar(binned_ells+1.5, ratios_ext,
             yerr=ratio_err_ext[1:] if len(ratio_err_ext)>1 else None,
             fmt='s', color='C3', ms=4, capsize=3, alpha=0.8,
             label=r'data / theory($(\ell{+}1/2)/\bar\chi$)')
ax2.set_xlabel(r'multipole $\ell$', fontsize=14)
ax2.set_ylabel('measured / theory', fontsize=14)
ax2.set_ylim(0.6, 1.5)
ax2.legend(fontsize=10, loc='upper right')

plt.savefig(os.path.join(plotdir, "money_plot_final.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(plotdir, "money_plot_final.png"), bbox_inches='tight', dpi=150)
print(f"  Saved money_plot_final.pdf/.png to {plotdir}")
plt.close()

# ---- Deconvolution plot ---- #
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8),
                           gridspec_kw={'height_ratios': [3, 1]},
                           sharex=True)
ax3, ax4 = axes2
fig2.subplots_adjust(hspace=0.05)

ax3.plot(ells_dec, theory_dec, 'k.--', lw=2, label=r'Theory $\ell/\bar\chi$')
ax3.plot(ells_dec, theory_dec_ext, 'r.:', lw=2, label=r'Theory $(\ell{+}1/2)/\bar\chi$')
ax3.plot(ells_dec, meas_dec, 'o', color='C0', ms=5, label='Measured (deconvolved)')
ax3.set_ylabel(r'deconvolved $C_\ell(k{=}0)$', fontsize=14)
ax3.legend(fontsize=10)
ax3.set_title(f'MaskDeconvolution: {num_sim} sims', fontsize=13)
ax3.tick_params(labelbottom=False)

ax4.axhline(1, color='k', ls='--', lw=0.8)
ax4.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ax4.plot(ells_dec, ratios_md, 'o', color='C0', ms=5,
         label=r'data/theory($\ell/\bar\chi$)')
ax4.plot(np.array(ells_dec)+1.5, ratios_md_ext, 's', color='C3', ms=4, alpha=0.8,
         label=r'data/theory($(\ell{+}1/2)/\bar\chi$)')
ax4.set_xlabel(r'multipole $\ell$', fontsize=14)
ax4.set_ylabel('measured / theory', fontsize=14)
ax4.set_ylim(0.0, 2.5)
ax4.legend(fontsize=10)

plt.savefig(os.path.join(plotdir, "money_plot_deconv_final.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(plotdir, "money_plot_deconv_final.png"), bbox_inches='tight', dpi=150)
print(f"  Saved money_plot_deconv_final.pdf/.png")
plt.close()

print(f"\n{'='*80}")
print("Done!")
