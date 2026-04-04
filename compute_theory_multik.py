#!/usr/bin/env python
"""
compute_theory_multik.py — Theory prediction for C_ell(k) at multiple k_par.

For the periodic box with FFT frequencies, the LOS modes are orthogonal
(no k-k' coupling), so the theory at each k_par is:

    <C~_ell(k_n)> = Sum_L M_ell_L^ang x C_true(L, k_n) + floor_cl(k_n)

where M^ang is the SAME angular MASTER coupling as for k=0, and

    C_true(ell, k_par) = b^2 (1+beta mu^2)^2 P_lin(|k|) / (L_box chi_eff^2)

The angular coupling matrix is built from the MEASURED angular window wl
saved by the simulation (at k=0), ensuring consistency with the actual
sightline positions.

Usage:
    python compute_theory_multik.py --simfile results_multik/Cell_multik_*.npz
"""
import argparse, sys, os, gc, time
import numpy as np

parser = argparse.ArgumentParser(description="Multi-k theory for C_ell(k)")
parser.add_argument("--simfile",    type=str, required=True, help="Path to sim .npz")
parser.add_argument("--Nl_large",   type=int, default=2000,  help="Extended ell for coupling matrix")
parser.add_argument("--NperBin",    type=int, default=32,    help="Multipoles per bin")
parser.add_argument("--outdir",     type=str, default=None)
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# ================================================================== #
# Load simulation data                                                #
# ================================================================== #
d = np.load(args.simfile)
cl_k_all  = d['cl_k']           # (Nsims, Nk, Nl)
wl_k      = d['wl_k']           # (Nk, Nl)
k_par     = d['k_par']          # (Nk,)
Nskew     = int(d['Nskew'])
N         = int(d['Nk_pix'])
L_box     = float(d['L'])
chi_shift = float(d['chi_shift'])
Lbox_3d   = float(d['Lbox'])
Ncell     = int(d['Ncell'])
bias      = float(d['bias'])
add_rsd   = bool(d['add_rsd'])
beta      = float(d['beta']) if add_rsd else 0.0

# Noise parameters (may be absent for noiseless runs)
sigma_c    = float(d['sigma_c']) if 'sigma_c' in d else 0.0
noise_frac = float(d['noise_frac']) if 'noise_frac' in d else 0.0

Nsims, Nk, Nl = cl_k_all.shape

# Noise bias: white in ell, independent of k
N_noise = Nskew * sigma_c**2 * N / (4 * np.pi) if sigma_c > 0 else 0.0

# Sightline geometry
all_x0 = d['all_x0']
all_y0 = d['all_y0']
all_z0 = d['all_z0']
r_j = np.sqrt(all_x0**2 + all_y0**2 + all_z0**2)
chi_eff = np.mean(r_j)

print(f"Loaded {args.simfile}")
print(f"  {Nsims} sims, Nk={Nk}, Nl={Nl}, Nskew={Nskew}, N={N}")
print(f"  L_box={L_box:.2f}, chi_eff={chi_eff:.1f}")
print(f"  k_par = {k_par}")
print(f"  bias={bias:.4f}, beta={beta:.4f}, add_rsd={add_rsd}")
print(f"  sigma_c={sigma_c:.4f}, noise_frac={noise_frac:.2f}, N_noise={N_noise:.4e}")

# ================================================================== #
# Cosmology
# ================================================================== #
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd, seed=0,
                                         my_bias=bias, my_beta=beta,
                                         N=Ncell, L=Lbox_3d)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

# ================================================================== #
# C_true(ell, k_par) with RSD Kaiser factor
# ================================================================== #
ells = np.arange(Nl, dtype=float)

def compute_cl_true(ells, k_par_val, b1, beta, plin, chi_eff, L_box):
    k_perp = (ells + 0.5) / chi_eff
    k_abs = np.sqrt(k_perp**2 + k_par_val**2)
    if k_par_val == 0:
        mu2 = np.zeros_like(ells)
    else:
        mu2 = k_par_val**2 / (k_perp**2 + k_par_val**2)
    kaiser = (1.0 + beta * mu2)**2
    return b1**2 * kaiser * plin(k_abs) / (L_box * chi_eff**2)

cl_true_all = np.zeros((Nk, Nl))
for ik in range(Nk):
    cl_true_all[ik] = compute_cl_true(ells, k_par[ik], b1, beta, plin,
                                       chi_eff, L_box)

# ================================================================== #
# Angular window -- from MEASURED wl at k=0                           #
# ================================================================== #
W_floor = N**2 * Nskew / (4 * np.pi)
k_Nyq = np.pi * N / L_box

wl_from_sim = wl_k[0, :]  # (Nl,)
print(f"\nUsing measured wl from simulation ({Nl} multipoles)")
print(f"  wl[0]={wl_from_sim[0]:.4e}, W_floor={W_floor:.4e}")

# ================================================================== #
# Floor-subtracted coupling matrix (angular, k-independent)          #
# ================================================================== #
def compute_floor_cl(k_par_val):
    L_Nyq = k_Nyq * chi_eff
    _L_floor = max(int(2 * L_Nyq), 8000)
    _ells = np.arange(_L_floor, dtype=float)
    _cl = compute_cl_true(_ells, k_par_val, b1, beta, plin, chi_eff, L_box)
    _k_perp = (_ells + 0.5) / chi_eff
    _k_abs = np.sqrt(_k_perp**2 + k_par_val**2)
    mask = _k_abs < k_Nyq
    return W_floor / (4 * np.pi) * np.sum((2 * _ells + 1) * _cl * mask)

Nl_large = args.Nl_large
wl_needed = 2 * Nl_large - 1

# Extend measured wl beyond Nl with the floor
wl_extended = np.full(wl_needed, W_floor)
wl_extended[:min(Nl, wl_needed)] = wl_from_sim[:min(Nl, wl_needed)]
wl_clust = wl_extended - W_floor

print(f"\nBuilding coupling matrix (Nl_large={Nl_large})...")
t0 = time.time()
couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()
print(f"  Done in {time.time()-t0:.1f}s")
del couple; gc.collect()

# ================================================================== #
# Convolved theory for each k                                        #
# ================================================================== #
NperBin = args.NperBin
theory_pseudo_all = np.zeros((Nk, Nl))
floor_cl_all = np.zeros(Nk)

# Binning
wl_ref_k0 = wl_from_sim[:Nl]
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref_k0)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref_k0, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, NperBin)
binned_ells = bins @ ells

for ik in range(Nk):
    k_val = k_par[ik]
    cl_true_ext = compute_cl_true(np.arange(Nl_large, dtype=float),
                                   k_val, b1, beta, plin, chi_eff, L_box)
    floor_cl = compute_floor_cl(k_val)
    floor_cl_all[ik] = floor_cl
    theory_pseudo_all[ik] = (M_clust @ cl_true_ext)[:Nl] + floor_cl
    print(f"  k={k_val:.5f}: floor_cl={floor_cl:.4e}, "
          f"theory[100]={theory_pseudo_all[ik, min(100, Nl-1)]:.4e}")

# ================================================================== #
# Subtract noise bias from measurements                               #
# ================================================================== #
if N_noise > 0:
    print(f"\nSubtracting noise bias N_noise={N_noise:.4e} from all C_ell(k)")
    cl_k_all = cl_k_all - N_noise  # broadcast over (Nsims, Nk, Nl)

# ================================================================== #
# Statistics and ratios                                               #
# ================================================================== #
cl_mean_all = np.mean(cl_k_all, axis=0)
cl_std_all  = np.std(cl_k_all, axis=0) / np.sqrt(Nsims)

print(f"\n{'='*72}")
print(f"Pseudo-Cl ratios (binned data/theory) for each k:")
ratios_all = np.zeros((Nk, len(binned_ells)))
for ik in range(Nk):
    binned_raw = bins @ cl_mean_all[ik]
    binned_theory = bins @ theory_pseudo_all[ik]
    for ib in range(len(binned_ells)):
        if binned_theory[ib] > 0:
            ratios_all[ik, ib] = binned_raw[ib] / binned_theory[ib]
    valid = ratios_all[ik, 1:]
    valid = valid[valid > 0]
    print(f"  k={k_par[ik]:.5f}: mean ratio (excl 1st)="
          f"{np.mean(valid):.4f} +/- {np.std(valid):.4f}")

# ================================================================== #
# Floor-subtracted deconvolution per k                               #
# ================================================================== #
wl_clust_md = np.full(2 * Nl - 1, W_floor)
wl_clust_md[:min(Nl, 2*Nl-1)] = wl_from_sim[:min(Nl, 2*Nl-1)]
wl_clust_md = wl_clust_md - W_floor

couple_clust_md = Wigner3j.CoupleMat(Nl, wl_clust_md)
M_clust_md = couple_clust_md.compute_matrix()
MD_clust = MaskDeconvolution(Nl, wl_clust_md, precomputed_Wigner=M_clust_md)

dec_all = np.zeros((Nk, Nsims, len(bins)))
dec_mean = np.zeros((Nk, len(bins)))
dec_std = np.zeros((Nk, len(bins)))
theory_dec_all = np.zeros((Nk, len(bins)))

for ik in range(Nk):
    for isim in range(Nsims):
        cl_clust_i = cl_k_all[isim, ik] - floor_cl_all[ik]
        _, dec_all[ik, isim] = MD_clust(cl_clust_i, bins)
    dec_mean[ik] = np.mean(dec_all[ik], axis=0)
    dec_std[ik]  = np.std(dec_all[ik], axis=0) / np.sqrt(Nsims)
    _, theory_dec_all[ik] = MD_clust.convolve_theory_Cls(cl_true_all[ik], bins)

ells_dec = MD_clust(cl_k_all[0, 0] - floor_cl_all[0], bins)[0]

# ================================================================== #
# Save                                                                #
# ================================================================== #
outdir = args.outdir or os.path.dirname(args.simfile)
os.makedirs(outdir, exist_ok=True)
simbase = os.path.splitext(os.path.basename(args.simfile))[0]
outfile = os.path.join(outdir, f"{simbase}_theory.npz")

np.savez(outfile,
         k_par=k_par, ells=ells, binned_ells=binned_ells, NperBin=NperBin,
         cl_k_all=cl_k_all, cl_mean_all=cl_mean_all, cl_std_all=cl_std_all,
         cl_true_all=cl_true_all, theory_pseudo_all=theory_pseudo_all,
         floor_cl_all=floor_cl_all,
         ells_dec=ells_dec, dec_mean=dec_mean, dec_std=dec_std,
         theory_dec_all=theory_dec_all,
         chi_eff=chi_eff, chi_shift=chi_shift,
         Nl=Nl, Nl_large=Nl_large, Nskew=Nskew, N=N, L_box=L_box,
         Lbox_3d=Lbox_3d, Ncell=Ncell,
         bias=bias, beta=beta, add_rsd=add_rsd,
         Nsims=Nsims, W_floor=W_floor,
         sigma_c=sigma_c, noise_frac=noise_frac, N_noise=N_noise)

print(f"\nSaved {outfile}")
