#!/usr/bin/env python
"""
compute_theory.py — Compute the floor-subtracted MASTER theory prediction
for a given simulation output (.npz from run_sims.py).

Reads the sim output (pseudo-Cl's, angular window, sightline positions) and
computes the convolved theory + deconvolved theory at the same binning as
used in the money plot.

Usage:
    python compute_theory.py --simfile results/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz \
                             --Nl_large 2000 --NperBin 32 --outdir results

Outputs results/<tag>_theory.npz containing everything needed for plotting.
"""
import argparse, sys, os, gc, time
import numpy as np

parser = argparse.ArgumentParser(description="Compute floor-subtracted MASTER theory")
parser.add_argument("--simfile",   type=str, required=True, help="Path to sim .npz")
parser.add_argument("--Nl_large",  type=int,   default=2000,  help="Extended ell range for coupling matrix")
parser.add_argument("--NperBin",   type=int,   default=32,    help="Multipoles per bin")
parser.add_argument("--lambda_max", type=int,  default=4000,  help="Max multipole for PLKjKk angular window")
parser.add_argument("--outdir",    type=str,   default=None,  help="Output directory (default: same as simfile)")
parser.add_argument("--recompute_PLKjKk", action="store_true",
                    help="Force recomputation of pair-counting angular window")
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# ================================================================== #
# Load simulation data                                                #
# ================================================================== #
d = np.load(args.simfile)
cl_k_all  = d['cl_k']           # (N_sims, Nl)
wl_k      = d['wl_k']           # (1, Nl) or (N_sims, Nl)
Nskew     = int(d['Nskew'])
N         = int(d['Nk'])         # number of LOS pixels
L_box     = float(d['L'])        # LOS box length
chi_shift = float(d['chi_shift'])
Lbox_3d   = float(d['Lbox'])     # 3D box side
Ncell     = int(d['Ncell'])
bias      = float(d['bias'])
add_rsd   = bool(d['add_rsd'])
beta      = float(d['beta']) if add_rsd else 0.0

Nl = cl_k_all.shape[1]
num_sim = cl_k_all.shape[0]

# Angular window (first sim only — deterministic sightline positions)
wl_ref = wl_k[0, :Nl] if wl_k.ndim == 2 else wl_k[:Nl]

# Sightline positions for chi_eff and PLKjKk
if 'all_x0' in d:
    all_x0 = d['all_x0']
    all_y0 = d['all_y0']
    all_z0 = d['all_z0']
    r_j = np.sqrt(all_x0**2 + all_y0**2 + all_z0**2)
    chi_eff = np.mean(r_j)
    theta_ref = d['theta']
    phi_ref   = d['phi']
else:
    # Fallback: reconstruct from grid
    chi_0 = chi_shift
    nqso_stored = int(d.get('nqso', Nskew))
    np.random.seed(100)
    _inds = np.unique(np.random.randint(0, Ncell, size=(nqso_stored, 2)), axis=0)
    _coords = np.linspace(0, Lbox_3d, Ncell)
    r_j = np.sqrt(chi_0**2 + _coords[_inds[:, 1]]**2 + _coords[_inds[:, 0]]**2)
    chi_eff = np.mean(r_j)
    # Reconstruct theta, phi
    GRF_pos = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd, seed=0, N=Ncell, L=Lbox_3d)
    _ax, _ay, _az, _, _, _ = GRF_pos.process_skewers(Nskew=nqso_stored, shift=chi_shift)
    theta_ref, phi_ref = GRF_pos.compute_theta_phi_skewer_start(_ax[:, 0], _ay[:, 0], _az[:, 0])
    del GRF_pos, _ax, _ay, _az

print(f"Loaded {args.simfile}")
print(f"  {num_sim} sims, Nl={Nl}, Nskew={Nskew}, N={N}, L_box={L_box:.2f}")
print(f"  chi_shift={chi_shift:.1f}, chi_eff=<r_j>={chi_eff:.1f}")
print(f"  bias={bias:.4f}, beta={beta:.4f}, add_rsd={add_rsd}")

# ================================================================== #
# Cosmology — P_lin                                                   #
# ================================================================== #
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd, seed=0,
                                         my_bias=bias, my_beta=beta,
                                         N=Ncell, L=Lbox_3d)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

# ================================================================== #
# Pair-counting angular window (PLKjKk)                               #
# ================================================================== #
tag = f"L{int(Lbox_3d)}_N{Ncell}_Nq{Nskew}"
PLKjKk_file = os.path.join(root, "notebooks", "data",
                           f"PLKjKk_{tag}_lambda{args.lambda_max}.npy")

if os.path.exists(PLKjKk_file) and not args.recompute_PLKjKk:
    print(f"Loading cached PLKjKk from {PLKjKk_file}")
    PLKjKk_full = np.load(PLKjKk_file)
else:
    print(f"Computing PLKjKk to lambda_max={args.lambda_max} ...")
    nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
    cos_theta = np.dot(nhat, nhat.T)
    KjKk = N**2
    t0 = time.time()
    PLKjKk_full = sht_lya.legendre_polynomials_sum(
        args.lambda_max, cos_theta, KjKk)[:args.lambda_max]
    print(f"  PLKjKk computed in {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(PLKjKk_file), exist_ok=True)
    np.save(PLKjKk_file, PLKjKk_full)
    print(f"  Saved to {PLKjKk_file}")
    del nhat, cos_theta; gc.collect()

wl_full = PLKjKk_full / (4 * np.pi)
lambda_max_data = len(PLKjKk_full)

# ================================================================== #
# Floor-subtracted MASTER theory                                      #
# ================================================================== #
W_floor = N**2 * Nskew / (4 * np.pi)
dchi = L_box / N

# Compute floor_cl analytically with Nyquist cutoff
k_Nyq = np.pi * N / L_box
L_Nyq = k_Nyq * chi_eff
_L_floor = max(int(2 * L_Nyq), 8000)
_ells_floor = np.arange(_L_floor, dtype=float)
_kperp_floor = (_ells_floor + 0.5) / chi_eff
_cl_floor = np.where(_kperp_floor < k_Nyq,
                     b1**2 * plin(_kperp_floor) / (L_box * chi_eff**2),
                     0.0)
floor_cl = W_floor / (4 * np.pi) * np.sum((2 * _ells_floor + 1) * _cl_floor)
print(f"\nW_floor = {W_floor:.4e}")
print(f"floor_cl = {floor_cl:.4e} (L_Nyq={L_Nyq:.0f})")
del _ells_floor, _kperp_floor, _cl_floor

# ---- Build coupling matrix from floor-subtracted wl ---- #
Nl_large = args.Nl_large
wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, lambda_max_data)
wl_raw[:n_avail] = wl_full[:n_avail]
wl_raw[n_avail:] = W_floor  # fill beyond data
wl_clust = wl_raw - W_floor

ells_ext = np.arange(Nl_large, dtype=float)
cl_true_ext = b1**2 * plin((ells_ext + 0.5) / chi_eff) / (L_box * chi_eff**2)

print(f"\nBuilding coupling matrix (Nl_large={Nl_large})...")
t0 = time.time()
couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()
theory_pseudo = (M_clust @ cl_true_ext)[:Nl] + floor_cl
print(f"  Done in {time.time()-t0:.1f}s")
del couple, M_clust; gc.collect()

# ---- C_true at Nl resolution ---- #
ells = np.arange(Nl, dtype=float)
cl_true = b1**2 * plin((ells + 0.5) / chi_eff) / (L_box * chi_eff**2)

# ================================================================== #
# Binning and pseudo-Cl statistics                                    #
# ================================================================== #
NperBin = args.NperBin
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, NperBin)
binned_ells = bins @ ells

# Pseudo-Cl stats
cl_mean = np.mean(cl_k_all, axis=0)
cl_std  = np.std(cl_k_all, axis=0) / np.sqrt(num_sim)
binned_raw = bins @ cl_mean
binned_std = bins @ cl_std
binned_theory = bins @ theory_pseudo

# ================================================================== #
# Floor-subtracted deconvolution (per-sim for error bars)            #
# ================================================================== #
wl_clust_md = np.zeros(2 * Nl - 1)
n_av = min(2 * Nl - 1, lambda_max_data)
wl_raw_md = np.zeros(2 * Nl - 1)
wl_raw_md[:n_av] = wl_full[:n_av]
wl_raw_md[n_av:] = W_floor
wl_clust_md = wl_raw_md - W_floor

couple_clust_md = Wigner3j.CoupleMat(Nl, wl_clust_md)
M_clust_md = couple_clust_md.compute_matrix()
MD_clust = MaskDeconvolution(Nl, wl_clust_md, precomputed_Wigner=M_clust_md)

dec_all = np.zeros((num_sim, len(bins)))
for isim in range(num_sim):
    cl_clust_i = cl_k_all[isim] - floor_cl
    _, dec_all[isim] = MD_clust(cl_clust_i, bins)

meas_dec = np.mean(dec_all, axis=0)
meas_dec_std = np.std(dec_all, axis=0) / np.sqrt(num_sim)
ells_dec = MD_clust(cl_k_all[0] - floor_cl, bins)[0]

# Theory bandpower window
_, theory_dec = MD_clust.convolve_theory_Cls(cl_true, bins)

# ================================================================== #
# Print summary                                                       #
# ================================================================== #
print(f"\n{'='*60}")
print(f"Pseudo-Cl ratios (binned data / theory):")
print(f"{'ell':>6s} {'theory':>12s} {'meas':>12s} {'ratio':>8s}")
for i in range(len(binned_ells)):
    if binned_theory[i] > 0:
        r = binned_raw[i] / binned_theory[i]
        print(f"{binned_ells[i]:6.0f} {binned_theory[i]:12.4e} "
              f"{binned_raw[i]:12.4e} {r:8.4f}")

ratios = binned_raw[1:] / binned_theory[1:]
print(f"\nMean ratio (excl first bin): {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

print(f"\nDeconvolved ratios:")
print(f"{'ell':>6s} {'theory':>12s} {'meas':>12s} {'ratio':>8s}")
for i in range(len(ells_dec)):
    if theory_dec[i] > 0:
        r = meas_dec[i] / theory_dec[i]
        print(f"{ells_dec[i]:6.0f} {theory_dec[i]:12.4e} "
              f"{meas_dec[i]:12.4e} {r:8.4f}")

# ================================================================== #
# Save                                                                #
# ================================================================== #
outdir = args.outdir or os.path.dirname(args.simfile)
os.makedirs(outdir, exist_ok=True)
simbase = os.path.splitext(os.path.basename(args.simfile))[0]
outfile = os.path.join(outdir, f"{simbase}_theory.npz")

np.savez(outfile,
         # Binning
         ells=ells,
         binned_ells=binned_ells,
         NperBin=NperBin,
         # Pseudo-Cl (binned)
         binned_raw=binned_raw,
         binned_std=binned_std,
         binned_theory=binned_theory,
         # Pseudo-Cl (unbinned, all sims)
         cl_k_all=cl_k_all,
         cl_mean=cl_mean,
         # Theory
         cl_true=cl_true,
         theory_pseudo=theory_pseudo,
         floor_cl=floor_cl,
         W_floor=W_floor,
         # Deconvolved
         ells_dec=ells_dec,
         meas_dec=meas_dec,
         meas_dec_std=meas_dec_std,
         theory_dec=theory_dec,
         dec_all=dec_all,
         # Parameters
         chi_eff=chi_eff,
         chi_shift=chi_shift,
         Nl=Nl,
         Nl_large=Nl_large,
         Nskew=Nskew,
         N=N,
         L_box=L_box,
         Lbox_3d=Lbox_3d,
         Ncell=Ncell,
         bias=bias,
         beta=beta,
         num_sim=num_sim)

print(f"\nSaved theory to {outfile}")
