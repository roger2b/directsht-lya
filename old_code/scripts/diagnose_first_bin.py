#!/usr/bin/env python
"""
Diagnose why the first ℓ-bin (ℓ~16) is low in the money plot.
Key question: does the floor decomposition break down at low ℓ?
"""
import sys, os, gc
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

Nl = 500
NperBin = 32
chi_shift = 5000
num_qso = 9797
add_rsd_ = False

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j
from sht.mask_deconvolution import MaskDeconvolution

# Load data
datafile = os.path.join(root, "notebooks", "data",
                        "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
d = np.load(datafile)
cl_k_all = d['cl_k']
wl_k = d['wl_k']
Nskew = int(d['Nskew'])
N = int(d['Nk'])
L_box = float(d['L'])
num_sim = cl_k_all.shape[0]
print(f"Loaded {num_sim} sims, Nskew={Nskew}, N={N}, L={L_box:.2f}")

wl_ref = wl_k[0, :Nl]

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

dchi = L_box / N
chi_0 = float(chi_shift)

# chi_eff
np.random.seed(100)
_inds = np.unique(np.random.randint(0, N, size=(num_qso, 2)), axis=0)
_coords_grid = np.linspace(0, L_box, N)
_r_j = np.sqrt(chi_0**2 + _coords_grid[_inds[:,1]]**2 + _coords_grid[_inds[:,0]]**2)
chi_eff = np.mean(_r_j)
del _inds, _coords_grid, _r_j

# Load PLKjKk
PLKjKk_file = os.path.join(root, "notebooks", "data", "PLKjKk_lambda4000.npy")
PLKjKk_full = np.load(PLKjKk_file)
wl_full = PLKjKk_full / (4 * np.pi)
lambda_max_data = len(PLKjKk_full)

# Floor
W_floor = N**2 * Nskew / (4 * np.pi)
print(f"\nW_floor = N^2 Nskew/(4pi) = {W_floor:.4e}")

# ---- Examine wl at low ℓ ----
print(f"\n{'ell':>5s} {'wl':>14s} {'W_floor':>14s} {'wl-Wfloor':>14s} {'(wl-Wf)/Wf':>14s}")
print("-" * 65)
for ell in range(20):
    wl_val = wl_full[ell]
    diff = wl_val - W_floor
    frac = diff / W_floor
    print(f"{ell:5d} {wl_val:14.4e} {W_floor:14.4e} {diff:14.4e} {frac:14.6f}")

print(f"\n... higher ℓ:")
for ell in [30, 50, 100, 200, 499]:
    wl_val = wl_full[ell]
    diff = wl_val - W_floor
    frac = diff / W_floor
    print(f"{ell:5d} {wl_val:14.4e} {W_floor:14.4e} {diff:14.4e} {frac:14.6f}")

# ---- Approach 1: pair-counting theory (no floor decomposition) ----
# <Cl> = (1/(4pi)) * sum_L M^(p)_ell_L * PLKjKk_L / (2pi chi^2)
# where M^(p) couples P(lambda/chi) into ell-L space
print("\n\n=== Approach 1: Direct pair-counting theory (no floor decomposition) ===")

Nl_large = 2000
kperp = (np.arange(Nl_large) + 0.5) / chi_eff
pk_L = b1_ref**2 * plin_ref(kperp)

# Build coupling matrix with P(k) as input spectrum
couple_pk = Wigner3j.CoupleMat(Nl_large, pk_L)
M_pk = couple_pk.compute_matrix()

# PLKjKk truncated/extended to 2*Nl_large - 1
PLKjKk_ext = np.zeros(Nl_large)
n_av = min(Nl_large, lambda_max_data)
PLKjKk_ext[:n_av] = PLKjKk_full[:n_av]
# For ℓ beyond data, use analytical floor: PLKjKk = 4pi * W_floor
PLKjKk_ext[n_av:] = 4 * np.pi * W_floor

theory_pair = (M_pk @ PLKjKk_ext)[:Nl] / (4 * np.pi * 2 * np.pi * chi_eff**2)

# ---- Approach 2: Floor-subtracted MASTER (current) ----
print("\n=== Approach 2: Floor-subtracted MASTER (current) ===")

wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, lambda_max_data)
wl_raw[:n_avail] = wl_full[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

cl_true_ext = b1_ref**2 * plin_ref((np.arange(Nl_large) + 0.5) / chi_eff) / (L_box * chi_eff**2)

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

# diag_cl
kvals = np.fft.fftfreq(N, d=1.0) * (2 * np.pi * N / L_box)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin_ref(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2_theory = b1_ref**2 * N**2 / L_box**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2_theory / (4 * np.pi)

theory_master = (M_clust @ cl_true_ext)[:Nl] + diag_cl

# ---- Approach 3: Full MASTER (no floor subtraction), but with enough multipoles ----
print("\n=== Approach 3: Full MASTER with Nl_large=2000 (no floor subtraction) ===")
couple_full = Wigner3j.CoupleMat(Nl_large, wl_raw)
M_full = couple_full.compute_matrix()
theory_full = (M_full @ cl_true_ext)[:Nl]

# ---- Compare in first few bins ----
print("\n" + "=" * 90)
print("COMPARISON: all three approaches")
print("=" * 90)

cl_mean = np.mean(cl_k_all, axis=0)
cl_std = np.std(cl_k_all, axis=0) / np.sqrt(num_sim)

# Bin
couple_wl_bin = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl_bin = couple_wl_bin.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl_bin)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

binned_raw = bins @ cl_mean
binned_std = bins @ cl_std
binned_pair = bins @ theory_pair
binned_master = bins @ theory_master
binned_full = bins @ theory_full

print(f"\n{'bin_ell':>8s} {'data':>12s} {'pair':>12s} {'floor-sub':>12s} {'full_M':>12s}"
      f" {'r_pair':>8s} {'r_fsub':>8s} {'r_full':>8s}")
print("-" * 100)
for i in range(len(binned_ells)):
    e = binned_ells[i]
    d_val = binned_raw[i]
    p_val = binned_pair[i]
    m_val = binned_master[i]
    f_val = binned_full[i]
    rp = d_val/p_val if p_val > 0 else 0
    rm = d_val/m_val if m_val > 0 else 0
    rf = d_val/f_val if f_val > 0 else 0
    print(f"{e:8.1f} {d_val:12.4e} {p_val:12.4e} {m_val:12.4e} {f_val:12.4e}"
          f" {rp:8.4f} {rm:8.4f} {rf:8.4f}")

# Print means excluding first bin
for name, ratios_arr in [("pair", binned_raw / binned_pair),
                         ("floor-sub", binned_raw / binned_master),
                         ("full_M", binned_raw / binned_full)]:
    vals = ratios_arr[1:]  # exclude first bin
    print(f"\n{name}: mean(excl bin0) = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    vals_all = ratios_arr
    print(f"{name}: mean(all bins)  = {np.mean(vals_all):.4f} ± {np.std(vals_all):.4f}")

# ---- Also check: what does the first bin contain? ----
print("\n\n=== First bin details ===")
print(f"First bin: ℓ = 0..{NperBin-1}, center {binned_ells[0]:.1f}")
print(f"  Binning matrix row 0:")
bm_row0 = bins[0]
nonzero = np.where(bm_row0 > 0)[0]
print(f"  Non-zero ℓ: {nonzero[0]}..{nonzero[-1]}")
print(f"  bin weights: {bm_row0[nonzero[0]:nonzero[-1]+1]}")

print(f"\n  Per-ℓ theory comparison (ℓ=0..{min(40, Nl)-1}):")
print(f"  {'ell':>4s} {'data':>12s} {'pair':>12s} {'fsub':>12s} {'r_pair':>8s} {'r_fsub':>8s}")
for ell in range(min(40, Nl)):
    d_val = cl_mean[ell]
    p_val = theory_pair[ell]
    m_val = theory_master[ell]
    rp = d_val / p_val if p_val > 0 else 0
    rm = d_val / m_val if m_val > 0 else 0
    print(f"  {ell:4d} {d_val:12.4e} {p_val:12.4e} {m_val:12.4e} {rp:8.4f} {rm:8.4f}")

# ---- Check ℓ=0 monopole contribution ----
print(f"\n=== Monopole (ℓ=0) check ===")
print(f"  cl_mean[0] = {cl_mean[0]:.4e}")
print(f"  theory_pair[0] = {theory_pair[0]:.4e}")
print(f"  theory_master[0] = {theory_master[0]:.4e}")
print(f"  wl_full[0] = {wl_full[0]:.4e}")
print(f"  W_floor = {W_floor:.4e}")
print(f"  wl_clust[0] = {wl_full[0] - W_floor:.4e}")
print(f"  PLKjKk[0] = {PLKjKk_full[0]:.4e}")
print(f"  For reference: wl[0] = sum_j,k 1 = Nskew^2 = {Nskew**2:.4e}")
print(f"    -> wl[0]/Nskew^2 = {wl_full[0]/(Nskew**2):.6f}")
print(f"    -> PLKjKk[0] = 4pi*wl[0] = {4*np.pi*wl_full[0]:.4e}")
