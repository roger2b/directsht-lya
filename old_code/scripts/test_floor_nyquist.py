#!/usr/bin/env python
"""
Test: zero C_true beyond L_Nyq where k_perp = L/chi > pi*N/L_box (the box Nyquist).
This makes the MASTER floor sum converge and match the physical diag_cl.
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

wl_ref = wl_k[0, :Nl]

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

chi_0 = float(chi_shift)
np.random.seed(100)
_inds = np.unique(np.random.randint(0, N, size=(num_qso, 2)), axis=0)
_coords = np.linspace(0, L_box, N)
_r_j = np.sqrt(chi_0**2 + _coords[_inds[:,1]]**2 + _coords[_inds[:,0]]**2)
chi_eff = np.mean(_r_j)
del _inds, _coords, _r_j

# Load PLKjKk
PLKjKk_full = np.load(os.path.join(root, "notebooks", "data", "PLKjKk_lambda4000.npy"))
wl_full = PLKjKk_full / (4 * np.pi)
lambda_max_data = len(PLKjKk_full)

# Physical parameters
W_floor = N**2 * Nskew / (4 * np.pi)
k_Nyq = np.pi * N / L_box
L_Nyq = k_Nyq * chi_eff
print(f"k_Nyq = pi*N/L = {k_Nyq:.4f}")
print(f"L_Nyq = k_Nyq * chi_eff = {L_Nyq:.1f}")
print(f"chi_eff = {chi_eff:.1f}")

# diag_cl from discrete 2D modes (physical)
kvals = np.fft.fftfreq(N, d=1.0) * (2 * np.pi * N / L_box)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin_ref(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2_theory = b1_ref**2 * N**2 / L_box**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2_theory / (4 * np.pi)
print(f"\ndiag_cl (physical) = {diag_cl:.4e}")

# ---- Floor sum WITH Nyquist cutoff ----
print(f"\n{'L_max':>8s} {'floor_sum':>14s} {'ratio to diag':>16s}")
print("-" * 42)
for L_max in [500, 1000, 2000, 3000, 4000, 5000, 5500, 5900, 5960, 6000, 6500, 7000, 8000, 10000]:
    ells_ext = np.arange(L_max, dtype=float)
    kperp = (ells_ext + 0.5) / chi_eff
    cl_true = np.where(kperp < k_Nyq,
                       b1_ref**2 * plin_ref(kperp) / (L_box * chi_eff**2),
                       0.0)
    floor_sum = W_floor / (4 * np.pi) * np.sum((2 * ells_ext + 1) * cl_true)
    print(f"{L_max:8d} {floor_sum:14.4e} {floor_sum/diag_cl:16.6f}")

# ---- Full test: M_clust @ C_true_cut + floor_sum_converged ----
print(f"\n\n=== Full theory with Nyquist-cutoff C_true ===")
Nl_large = 2000

wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, lambda_max_data)
wl_raw[:n_avail] = wl_full[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

ells_ext = np.arange(Nl_large, dtype=float)
kperp_ext = (ells_ext + 0.5) / chi_eff
cl_true_ext = b1_ref**2 * plin_ref(kperp_ext) / (L_box * chi_eff**2)
# No cutoff needed at Nl_large=2000 since 2000 < L_Nyq

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

# floor_converged: compute at large L_max with cutoff
L_max_converged = 8000
ells_big = np.arange(L_max_converged, dtype=float)
kperp_big = (ells_big + 0.5) / chi_eff
cl_true_big = np.where(kperp_big < k_Nyq,
                       b1_ref**2 * plin_ref(kperp_big) / (L_box * chi_eff**2),
                       0.0)
floor_converged = W_floor / (4 * np.pi) * np.sum((2 * ells_big + 1) * cl_true_big)
print(f"floor_converged (L_max={L_max_converged}, cutoff at L_Nyq={L_Nyq:.0f}) = {floor_converged:.4e}")
print(f"diag_cl = {diag_cl:.4e}")
print(f"floor_converged / diag_cl = {floor_converged/diag_cl:.6f}")

# Theory: M_clust @ C_true + floor_converged
theory_floor_fix = (M_clust @ cl_true_ext)[:Nl] + floor_converged
# Compare with theory using diag_cl
theory_diag = (M_clust @ cl_true_ext)[:Nl] + diag_cl

# Binning
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

cl_mean = np.mean(cl_k_all, axis=0)
binned_raw = bins @ cl_mean
binned_fix = bins @ theory_floor_fix
binned_diag = bins @ theory_diag

print(f"\n{'ell':>8s} {'data':>12s} {'fix_th':>12s} {'diag_th':>12s} "
      f"{'r_fix':>8s} {'r_diag':>8s}")
print("-" * 70)
for i in range(len(binned_ells)):
    rf = binned_raw[i] / binned_fix[i] if binned_fix[i] > 0 else 0
    rd = binned_raw[i] / binned_diag[i] if binned_diag[i] > 0 else 0
    print(f"{binned_ells[i]:8.1f} {binned_raw[i]:12.4e} {binned_fix[i]:12.4e} "
          f"{binned_diag[i]:12.4e} {rf:8.4f} {rd:8.4f}")

r_fix = binned_raw / binned_fix
r_diag = binned_raw / binned_diag
print(f"\nWith floor_converged: mean(excl bin0) = {np.mean(r_fix[1:]):.4f} ± {np.std(r_fix[1:]):.4f}")
print(f"With diag_cl:         mean(excl bin0) = {np.mean(r_diag[1:]):.4f} ± {np.std(r_diag[1:]):.4f}")
print(f"With floor_converged: ALL bins = {np.mean(r_fix):.4f} ± {np.std(r_fix):.4f}")
print(f"With diag_cl:         ALL bins = {np.mean(r_diag):.4f} ± {np.std(r_diag):.4f}")

# ---- Also check with the EXACT discrete 2D sum for the floor ----
# The floor sum SHOULD equal diag_cl if computed from the discrete modes
# Instead of integral, use sum over 2D k-modes weighted by the angular mapping
print(f"\n\n=== Understanding the mismatch ===")
print(f"W_floor = N^2 Nskew/(4pi) = {W_floor:.4e}")

# Compare continuous and discrete integrals
# Continuous: int dk P(k) 2pi k = (2pi)^2/L^2 * sum_{kx,ky} P(k) 
# gives the 2D field variance
# Angular: sum_L (2L+1) C_L / (4pi) = angular field variance
#
# The angular mapping is: k_perp = L/chi, so dL = chi dk.
# sum_L (2L+1) C_L ≈ ∫ 2L C_L dL = ∫ 2(chi k)(b1^2 P(k)/(L chi^2)) chi dk  
#                   = 2 b1^2/L * ∫ k P(k) dk  (note: L here is L_box)
# 
# Discrete 2D sum: <w^2> = b1^2 N^2/L^3 * sum P(k_perp)
#   = b1^2 N^2/L^3 * (L^2/(2pi)^2 * ∫ 2pi k P(k) dk)
#   = b1^2 N^2/(2pi L) * ∫ k P(k) dk
#
# So: diag_cl = Nskew * <w^2>/(4pi) = Nskew b1^2 N^2 / (4pi * 2pi L) * ∫ k P(k) dk
#     floor_sum = W_floor/(4pi) * sum(2L+1)C_L ≈ N^2 Nskew/(4pi)^2 * 2 b1^2/L * ∫ k P(k) dk
#              = Nskew b1^2 N^2 / (8 pi^2 L) * ∫ k P(k) dk
#
# Ratio: diag_cl / floor_sum = (8 pi^2) / (4pi * 2pi) = 8pi^2 / 8pi^2 = 1
# Wait, they should be equal!

# Let me be more careful:
# floor_sum = W_floor/(4pi) * sum(2L+1) b1^2 P((L+0.5)/chi)/(L_box chi^2)
# Continuum limit: sum_L (2L+1)f(L) -> ∫ 2L f(L) dL for large L
# ≈ (W_floor/(4pi)) * ∫_0^infty 2 b1^2 P(L/chi)/(L_box chi^2) dL
# = (W_floor/(4pi)) * 2 b1^2/(L_box chi) * ∫_0^infty P(k) dk   [with k=L/chi]
# 
# diag_cl = Nskew * <w^2>/(4pi)
# <w^2> = b1^2 N^2/L_box^3 * sum_{k_x,k_y} P(k_perp)
# Continuum: sum -> (L_box^2/(2pi)^2) * ∫ 2pi k P(k) dk
# <w^2> ≈ b1^2 N^2/(2pi L_box) * ∫ k P(k) dk
# 
# So floor_sum ≈ (N^2 Nskew/(4pi)^2) * 2 b1^2/(L_box chi) * ∫ P dk
#    diag_cl ≈ Nskew/(4pi) * b1^2 N^2/(2pi L_box) * ∫ k P dk
#
# These are different integrals! ∫ P dk vs ∫ k P dk.
# With CDM: P ~ k at small k (n_s≈1), so ∫_0 P dk diverges LESS fast than ∫_0 k P dk.
# At large k: P ~ k^{-3}, so ∫ P dk ~ ∫ k^{-3} dk converges,
#                            ∫ k P dk ~ ∫ k^{-2} dk also converges.

# But wait, the sum in floor_sum (2L+1)≈2L, and C_L = b1^2 P(L/chi)/(L_box chi^2)
# so (2L) C_L = 2 b1^2 P(L/chi)/(L_box chi^2)
# integral: ∫_0^inf 2 b1^2 P(L/chi)/(L_box chi^2) dL = 2 b1^2/(L_box chi) * ∫ P(k) dk

# And the discrete sum <w^2> involves:
# sum_{kx,ky} P(kperp) -> (L^2/(2pi)^2) ∫ 2pi k P(k) dk = L^2/(2pi) ∫ k P(k) dk
# <w^2> = b1^2 N^2/L^3 * L^2/(2pi) ∫ k P(k) dk = b1^2 N^2/(2pi L) ∫ k P(k) dk

# So the two integrals ARE different: floor uses ∫ P(k) dk, diag uses ∫ k P(k) dk
# This is a 1D vs 2D integral issue!

# Let me verify numerically
dk = 0.001
k_arr = np.arange(dk, k_Nyq, dk)
P_arr = plin_ref(k_arr)
int_P = np.sum(P_arr) * dk         # ∫ P dk
int_kP = np.sum(k_arr * P_arr) * dk  # ∫ k P dk

print(f"∫ P(k) dk (to k_Nyq={k_Nyq:.2f}) = {int_P:.4e}")
print(f"∫ k P(k) dk (to k_Nyq={k_Nyq:.2f}) = {int_kP:.4e}")
print(f"Ratio ∫kP/∫P = {int_kP/int_P:.4f}")

# Predicted floor_sum (continuous):
floor_pred = N**2 * Nskew / (4*np.pi)**2 * 2 * b1_ref**2 / (L_box * chi_eff) * int_P
# Predicted diag_cl (continuous):
diag_pred = Nskew / (4*np.pi) * b1_ref**2 * N**2 / (2*np.pi*L_box) * int_kP
print(f"\nfloor_pred  (continuous) = {floor_pred:.4e}")
print(f"diag_pred   (continuous) = {diag_pred:.4e}")
print(f"diag_cl     (discrete)   = {diag_cl:.4e}")
print(f"floor_conv  (discrete sum) = {floor_converged:.4e}")
print(f"\nfloor_pred / diag_pred = {floor_pred/diag_pred:.4f}")
