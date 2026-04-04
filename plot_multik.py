#!/usr/bin/env python
"""
plot_multik.py — Money plots for multi-k C_ell(k) results.

Reads the output of compute_theory_multik.py and produces:
  1. Multi-panel pseudo-Cl plot (one per k_par, with theory + data)
  2. Ratio plot (meas/theory vs ell for all k_par values)
  3. Deconvolved C_ell vs theory for each k_par

Usage:
    python plot_multik.py --theoryfile results_multik/Cell_multik_*_theory.npz
"""
import argparse, sys, os
import numpy as np

parser = argparse.ArgumentParser(description="Plot multi-k C_ell(k) results")
parser.add_argument("--theoryfile", type=str, required=True,
                    help="Path to *_theory.npz from compute_theory_multik.py")
parser.add_argument("--plotdir", type=str, default=None,
                    help="Output plot directory (default: same as theoryfile)")
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, "notebooks"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib_params_file  # noqa: F401
from matplotlib.gridspec import GridSpec

# ================================================================== #
# Load
# ================================================================== #
d = np.load(args.theoryfile)
k_par           = d['k_par']
ells            = d['ells']
binned_ells     = d['binned_ells']
NperBin         = int(d['NperBin'])
cl_k_all        = d['cl_k_all']        # (Nsims, Nk, Nl)
cl_mean_all     = d['cl_mean_all']      # (Nk, Nl)
theory_pseudo_all = d['theory_pseudo_all']  # (Nk, Nl)
floor_cl_all    = d['floor_cl_all']
cl_true_all     = d['cl_true_all']
ells_dec        = d['ells_dec']
dec_mean        = d['dec_mean']
dec_std         = d['dec_std']
theory_dec_all  = d['theory_dec_all']
chi_eff         = float(d['chi_eff'])
Nl              = int(d['Nl'])
Nl_large        = int(d['Nl_large'])
Nskew           = int(d['Nskew'])
bias            = float(d['bias'])
Nsims           = int(d['Nsims'])
sigma_c         = float(d['sigma_c']) if 'sigma_c' in d else 0.0
N_noise         = float(d['N_noise']) if 'N_noise' in d else 0.0

Nk = len(k_par)
Nbins = len(binned_ells)

plotdir = args.plotdir or os.path.dirname(args.theoryfile)
os.makedirs(plotdir, exist_ok=True)

print(f"Loaded {args.theoryfile}")
print(f"  {Nsims} sims, Nk={Nk}, Nl={Nl}, Nskew={Nskew}")
print(f"  k_par = {k_par}")
print(f"  sigma_c = {sigma_c:.4f}, N_noise = {N_noise:.4e}")

# ================================================================== #
# Binning helper
# ================================================================== #
bins_mat = np.zeros((Nbins, Nl))
for i in range(Nbins):
    lo = i * NperBin
    hi = min(lo + NperBin, Nl)
    bins_mat[i, lo:hi] = 1.0 / (hi - lo)

# ================================================================== #
# PLOT 1: Multi-panel pseudo-Cl  (nrows x ncols subplots, one per k)
# ================================================================== #
ncols = min(3, Nk)
nrows = (Nk + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows),
                         squeeze=False, sharex=True)
fig.subplots_adjust(hspace=0.25, wspace=0.3)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, Nk))

for ik in range(Nk):
    ir, ic = divmod(ik, ncols)
    ax = axes[ir, ic]

    binned_raw = bins_mat @ cl_mean_all[ik]
    binned_std = bins_mat @ (np.std(cl_k_all[:, ik, :], axis=0) / np.sqrt(Nsims))
    binned_theory = bins_mat @ theory_pseudo_all[ik]

    ax.errorbar(binned_ells, binned_raw, yerr=binned_std,
                fmt='o', color=colors[ik], ms=4, capsize=2,
                label=f'Measured ({Nsims} sims)')
    ax.plot(binned_ells, binned_theory, 'k--', lw=1.5, label='Theory')

    ax.set_title(f'$k_\\parallel = {k_par[ik]:.4f}$ h/Mpc', fontsize=14)
    if ir == nrows - 1:
        ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'pseudo-$C_\ell(k)$')
    ax.legend(fontsize=10, loc='upper right')

# Hide empty subplots
for ik in range(Nk, nrows * ncols):
    ir, ic = divmod(ik, ncols)
    axes[ir, ic].set_visible(False)

noise_str = f', $\\sigma_c={sigma_c:.2f}$' if sigma_c > 0 else ''
fig.suptitle(f'Pseudo-$C_\\ell(k)$: {Nsims} sims, $N_\\ell$={Nl}, '
             f'$N_{{\\rm skew}}$={Nskew}{noise_str}', fontsize=16, y=1.02)

outname = os.path.join(plotdir, "multik_pseudo_cl.pdf")
plt.savefig(outname, bbox_inches='tight')
plt.savefig(outname.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print(f"Saved {outname}")
plt.close()

# ================================================================== #
# PLOT 2: Ratio plot — all k on one figure, two panels
# ================================================================== #
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

for ik in range(Nk):
    binned_raw = bins_mat @ cl_mean_all[ik]
    binned_std = bins_mat @ (np.std(cl_k_all[:, ik, :], axis=0) / np.sqrt(Nsims))
    binned_theory = bins_mat @ theory_pseudo_all[ik]

    offset = ik * 1.0  # small horizontal offset for clarity
    ax1.errorbar(binned_ells + offset, binned_raw, yerr=binned_std,
                 fmt='o', color=colors[ik], ms=3, capsize=2,
                 label=f'$k_\\parallel$={k_par[ik]:.3f}')

    # Ratio (bottom panel)
    mask = binned_theory > 0
    ratio = np.where(mask, binned_raw / binned_theory, np.nan)
    ratio_err = np.where(mask, binned_std / binned_theory, np.nan)
    ax2.errorbar(binned_ells[mask] + offset, ratio[mask],
                 yerr=ratio_err[mask],
                 fmt='o', color=colors[ik], ms=3, capsize=1)

# Theory line in top panel (just k=0 as reference)
binned_theory_k0 = bins_mat @ theory_pseudo_all[0]
ax1.plot(binned_ells, binned_theory_k0, 'k--', lw=2, label='Theory (k=0)')

ax1.set_ylabel(r'binned pseudo-$C_\ell(k)$')
ax1.legend(fontsize=9, ncol=min(4, Nk+1), loc='upper right')
ax1.set_title(f'Multi-$k$ pseudo-$C_\\ell$: {Nsims} sims{noise_str}')
ax1.tick_params(labelbottom=False)

ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ax2.set_xlabel(r'multipole $\ell$')
ax2.set_ylabel('meas / theory')
ax2.set_ylim(0.7, 1.3)

outname = os.path.join(plotdir, "multik_ratio.pdf")
plt.savefig(outname, bbox_inches='tight')
plt.savefig(outname.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print(f"Saved {outname}")
plt.close()

# ================================================================== #
# PLOT 3: Deconvolved C_ell for each k
# ================================================================== #
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

for ik in range(Nk):
    offset = ik * 1.0
    ax1.errorbar(ells_dec + offset, dec_mean[ik], yerr=dec_std[ik],
                 fmt='o', color=colors[ik], ms=3, capsize=2,
                 label=f'$k_\\parallel$={k_par[ik]:.3f}')
    ax1.plot(ells_dec, theory_dec_all[ik], '--', color=colors[ik], lw=1.5)

    # Ratio
    mask = theory_dec_all[ik] > 0
    ratio = np.where(mask, dec_mean[ik] / theory_dec_all[ik], np.nan)
    ratio_err = np.where(mask, dec_std[ik] / theory_dec_all[ik], np.nan)
    ax2.errorbar(ells_dec[mask] + offset, ratio[mask],
                 yerr=ratio_err[mask],
                 fmt='o', color=colors[ik], ms=3, capsize=1)

ax1.set_ylabel(r'deconvolved $C_\ell(k)$')
ax1.legend(fontsize=9, ncol=min(4, Nk), loc='upper right')
ax1.set_title(f'Floor-subtracted deconvolved $C_\\ell(k)$: '
              f'{Nsims} sims{noise_str}')
ax1.tick_params(labelbottom=False)

ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15)
ax2.set_xlabel(r'multipole $\ell$')
ax2.set_ylabel('meas / theory')
ax2.set_ylim(0.7, 1.3)

outname = os.path.join(plotdir, "multik_deconv.pdf")
plt.savefig(outname, bbox_inches='tight')
plt.savefig(outname.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print(f"Saved {outname}")
plt.close()

# ================================================================== #
# Summary table
# ================================================================== #
print(f"\n{'='*72}")
print(f"Summary: mean ratio (excl first bin) at each k_par")
print(f"{'k_par':>10s} {'pseudo ratio':>14s} {'deconv ratio':>14s}")
print("-" * 40)
for ik in range(Nk):
    binned_raw = bins_mat @ cl_mean_all[ik]
    binned_theory = bins_mat @ theory_pseudo_all[ik]
    mask_p = binned_theory > 0
    ratios_p = binned_raw[mask_p] / binned_theory[mask_p]

    mask_d = theory_dec_all[ik] > 0
    ratios_d = dec_mean[ik][mask_d] / theory_dec_all[ik][mask_d]

    rp = np.mean(ratios_p[1:]) if len(ratios_p) > 1 else np.nan
    rd = np.mean(ratios_d[1:]) if len(ratios_d) > 1 else np.nan
    print(f"{k_par[ik]:10.5f} {rp:14.4f} {rd:14.4f}")

print(f"\nDone!")
