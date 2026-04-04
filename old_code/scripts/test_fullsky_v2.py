#!/usr/bin/env python -u
"""
Definitive normalization test: vary angular coverage to isolate shot noise.

Key insight: the box covers angular extent ~ L/chi_shift on the sky.
By reducing chi_shift, we increase angular coverage → window becomes more
diagonal → mode coupling simplifies → normalization check is cleaner.

Tests TWO chi_shift values:
  - chi_shift = 5000: small patch (~14°), strong mode coupling
  - chi_shift = 800:  large patch (~60°), weak mode coupling

For each, compares measured pseudo-Cl to pair-counting theory (which we
KNOW works from 20-sim production tests).

Also demonstrates that SN = Σw²/(4π) is NOT a separate noise term —
it's part of the cosmological signal in the MASTER framework.
"""
import sys, os, gc
import numpy as np
import healpy as hp

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
from sht.theory_lya import theory_cl_k, compute_chi_bar_from_grid
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================================================================== #
#  Settings                                                               #
# ====================================================================== #
N_box    = 64
L_box    = 1380.0
Nl       = 50
num_sim  = 10
add_rsd  = False
seed0    = 3000
NperBin  = 8

# Two chi_shift values: small patch vs large patch
chi_shifts = [5000.0, 800.0]

def extract_all_sightlines(GRF, shift):
    """Extract ALL N^2 sightlines."""
    N, L = GRF.N, GRF.L
    coords = np.meshgrid(*[np.linspace(0, L, N) for _ in range(3)])
    iy, ix = np.meshgrid(np.arange(N), np.arange(N))
    inds = np.column_stack([ix.ravel(), iy.ravel()])
    Nskew = len(inds)
    dens_lya = GRF.dens[inds[:, 0], inds[:, 1], :] + 1.0
    tmp_x = coords[0][inds[:, 0], inds[:, 1], :]
    tmp_y = coords[1][inds[:, 0], inds[:, 1], :]
    tmp_z = coords[2][inds[:, 0], inds[:, 1], :] + shift
    all_x, all_y, all_z = tmp_z.copy(), tmp_y.copy(), tmp_x.copy()
    return all_x, all_y, all_z, np.ones_like(all_x), dens_lya - 1.0, Nskew


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, chi_shift in enumerate(chi_shifts):
    print(f"\n{'='*60}")
    print(f"CHI_SHIFT = {chi_shift}")
    print(f"{'='*60}")

    sht = DirectSHT(Nl, 2*Nl, 0.75)

    # Reference sightline positions from first sim
    GRF0 = my_GRF.PowerSpectrumGenerator(N=N_box, L=L_box, add_rsd=add_rsd,
                                           seed=seed0, verbose=False)
    allx, ally, allz, _, _, Nskew = extract_all_sightlines(GRF0, chi_shift)
    chi_grid = allx[0, :]
    theta, phi = GRF0.compute_theta_phi_skewer_start(allx[:, 0], ally[:, 0], allz[:, 0])
    dchi = chi_grid[1] - chi_grid[0]
    N = chi_grid.size
    chi_bar = compute_chi_bar_from_grid(chi_grid)
    plin = GRF0.plin
    b1 = GRF0.my_bias

    ang_extent = np.degrees(L_box / chi_bar)
    print(f"Nskew={Nskew}, N={N}, chi_bar={chi_bar:.0f}, angular extent={ang_extent:.1f}°")

    # Window spectrum from randoms at k=0
    hran = sht(theta, phi, N * np.ones(Nskew))
    wl_ref = hp.alm2cl(hran)[:Nl]

    # SN from unit weights
    sn_rand = N**2 * Nskew / (4.0 * np.pi)
    print(f"wl_ref[0]={wl_ref[0]:.4e}, (N*Nskew)^2/(4pi)={(N*Nskew)**2/(4*np.pi):.4e}")

    del GRF0; gc.collect()

    # ---- Measure pseudo-Cl from multiple sims ---- #
    cl_stack, sn_stack = [], []
    for isim in range(num_sim):
        GRF = my_GRF.PowerSpectrumGenerator(N=N_box, L=L_box, add_rsd=add_rsd,
                                              seed=seed0+isim, verbose=False)
        _, _, _, _, wg, _ = extract_all_sightlines(GRF, chi_shift)
        _, _, ftd = sht_lya.compute_dft(chi_grid, np.ones_like(wg), wg)
        w = ftd[:, 0]
        h = sht(theta, phi, w)
        cl = hp.alm2cl(h)[:Nl]
        sn = np.sum(w**2) / (4.0 * np.pi)
        cl_stack.append(cl)
        sn_stack.append(sn)
        del GRF, wg, ftd; gc.collect()

    cl_mean = np.mean(cl_stack, axis=0)
    cl_std = np.std(cl_stack, axis=0)
    sn_mean = np.mean(sn_stack)
    print(f"Mean SN = {sn_mean:.4e}")

    # ---- Pair-counting theory (the KNOWN-CORRECT approach) ---- #
    nhat = sht_lya.compute_nhat(theta, phi)
    cos_theta = np.dot(nhat, nhat.T)
    KjKk = N**2
    print("Computing pair-counting Legendre sums...", end="", flush=True)
    PLKjKk = sht_lya.legendre_polynomials_sum(Nl, cos_theta, KjKk)[:Nl]
    print("done")

    # Coupling matrix with P(k) as weights
    ells = np.arange(Nl, dtype=float)
    pk_L = b1**2 * plin(ells / chi_bar)  # P_F = b1^2 * P_lin at k_par=0, beta=0
    pk_L[0] = b1**2 * plin(0.5 / chi_bar)

    couple_pk = Wigner3j.CoupleMat(Nl, pk_L)
    M_pk = couple_pk.compute_matrix()

    C_theory_orig = M_pk @ PLKjKk / (4.0 * np.pi) / (2.0 * np.pi * chi_bar**2)
    C_theory_plotted = C_theory_orig / (4.0 * np.pi)**2

    # ---- MaskDeconvolution approach ---- #
    MD = MaskDeconvolution(Nl, wl_ref)
    Mll = MD.Mll
    cl_limber = b1**2 * plin(ells / chi_bar) / chi_bar**2
    cl_limber[0] = b1**2 * plin(0.5 / chi_bar) / chi_bar**2
    cl_32pi3 = cl_limber / (32.0 * np.pi**3)
    cl_convolved_32 = Mll @ cl_32pi3

    # ---- Verify identity: C_theory_plotted = Mll @ pk / (32π³ χ²) ---- #
    cl_identity = Mll @ pk_L / (32.0 * np.pi**3 * chi_bar**2)
    print(f"Identity check: C_pair[5]={C_theory_plotted[5]:.4e}, "
          f"Mll@pk/(32π³χ²)[5]={cl_identity[5]:.4e}, "
          f"ratio={C_theory_plotted[5]/cl_identity[5]:.6f}")

    # ---- Binning ---- #
    bins = MD.binning_matrix('linear', 0, NperBin)
    bnells = bins @ ells

    bn_data = bins @ cl_mean
    bn_std = bins @ cl_std / np.sqrt(num_sim)
    bn_theory_pair = bins @ C_theory_plotted[:Nl]
    bn_theory_md = bins @ cl_convolved_32
    bn_sn = sn_mean * np.ones_like(bn_data)  # binned SN = SN (flat)

    # ---- Print ratios ---- #
    print(f"\n{'ell':>6} {'data':>12} {'pair_th':>12} {'md_th':>12} "
          f"{'SN':>12} {'r_pair':>8} {'r_md':>8} {'r_pair_SN':>8}")
    for i in range(len(bnells)):
        rp = bn_data[i] / bn_theory_pair[i] if bn_theory_pair[i] > 0 else np.inf
        rm = bn_data[i] / bn_theory_md[i] if bn_theory_md[i] > 0 else np.inf
        rps = (bn_data[i] - sn_mean) / bn_theory_pair[i] if bn_theory_pair[i] > 0 else np.inf
        print(f"{bnells[i]:6.0f} {bn_data[i]:12.4e} {bn_theory_pair[i]:12.4e} "
              f"{bn_theory_md[i]:12.4e} {sn_mean:12.4e} {rp:8.4f} {rm:8.4f} {rps:8.4f}")

    # Ratios excluding edge bins
    r_pair_all = [bn_data[i] / bn_theory_pair[i] for i in range(1, len(bnells)-1)
                  if bn_theory_pair[i] > 0]
    r_pair_sn = [(bn_data[i] - sn_mean) / bn_theory_pair[i] for i in range(1, len(bnells)-1)
                 if bn_theory_pair[i] > 0]
    print(f"\nMean ratios (excl. first/last bin):")
    print(f"  Raw:         {np.mean(r_pair_all):.4f} ± {np.std(r_pair_all)/np.sqrt(len(r_pair_all)):.4f}")
    print(f"  SN-sub:      {np.mean(r_pair_sn):.4f} ± {np.std(r_pair_sn)/np.sqrt(len(r_pair_sn)):.4f}")
    print(f"  SN/data_mid: {sn_mean/bn_data[len(bnells)//2]:.4f}")

    # ---- Decompose: what fraction of theory comes from diagonal? ---- #
    # Theory = off-diagonal (signal) + diagonal (j=k contribution)
    # For pair-counting: PLKjKk_diag[λ] = KjKk × Nskew (since P_l(1)=1)
    # PLKjKk_offdiag = PLKjKk - PLKjKk_diag
    PLdiag = KjKk * Nskew * np.ones(Nl)  # Σ_j KjKk * P_l(cos 0) = KjKk * Nskew
    PLoffdiag = PLKjKk - PLdiag
    C_diag = M_pk @ PLdiag / (4*np.pi) / (2*np.pi*chi_bar**2) / (4*np.pi)**2
    C_offdiag = M_pk @ PLoffdiag / (4*np.pi) / (2*np.pi*chi_bar**2) / (4*np.pi)**2
    print(f"\n  Theory decomposition at ℓ=Nl//2:")
    l_mid = Nl // 2
    print(f"    C_total[{l_mid}]   = {C_theory_plotted[l_mid]:.4e}")
    print(f"    C_diag[{l_mid}]    = {C_diag[l_mid]:.4e}")
    print(f"    C_offdiag[{l_mid}] = {C_offdiag[l_mid]:.4e}")
    print(f"    diag/total      = {C_diag[l_mid]/C_theory_plotted[l_mid]:.4f}")
    print(f"    SN_data         = {sn_mean:.4e}")
    print(f"    SN / data[{l_mid}]  = {sn_mean / cl_mean[l_mid]:.4f}")

    # ---- Plot ---- #
    ax = axes[0, col]
    ax.semilogy(bnells, bn_data, 'C0o-', lw=1.5, ms=4, label=f'Data ({num_sim} sims)')
    ax.semilogy(bnells, bn_data - sn_mean, 'C1s--', lw=1, ms=3, label='Data − SN')
    ax.semilogy(bnells, bn_theory_pair, 'k^--', lw=2, ms=5, label='Pair-counting theory')
    ax.semilogy(bnells, bn_theory_md, 'rv--', lw=1.5, ms=4, label=r'$M_{\ell\ell}\cdot C^{32\pi^3}$')
    ax.axhline(sn_mean, color='gray', ls=':', lw=1, label=f'SN = {sn_mean:.1e}')
    ax.set_ylabel(r'$C_\ell(k=0)$')
    ax.set_title(f'$\\chi_{{shift}}={chi_shift:.0f}$, patch={ang_extent:.0f}°, Nskew={Nskew}')
    ax.legend(fontsize=7, loc='best')

    axr = axes[1, col]
    axr.plot(bnells, bn_data / bn_theory_pair, 'C0o-', lw=1.5, ms=4, label='Raw / pair theory')
    axr.plot(bnells, (bn_data - sn_mean) / bn_theory_pair, 'C1s--', lw=1, ms=3, label='(Data−SN) / pair theory')
    axr.axhline(1.0, color='k', ls=':', lw=1)
    axr.set_xlabel(r'Multipole $\ell$')
    axr.set_ylabel('Ratio')
    axr.set_ylim(0, 2.5)
    axr.legend(fontsize=8)

plt.tight_layout()
outdir = os.path.join(root, 'notebooks', 'plots')
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, 'normalization_definitive.png'), dpi=150)
plt.savefig(os.path.join(outdir, 'normalization_definitive.pdf'))
print(f"\nPlots saved to notebooks/plots/normalization_definitive.*")
print(f"\n{'='*60}")
print(f"CONCLUSION")
print(f"{'='*60}")
print(f"If raw ratio ≈ 1 and SN-sub ratio < 1:")
print(f"  → SN is already in the theory (pair-counting diagonal terms)")
print(f"  → Do NOT subtract SN separately")
print(f"  → The 32π³ normalization is correct")
