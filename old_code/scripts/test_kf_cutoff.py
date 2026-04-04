#!/usr/bin/env python
"""
Test: apply BOTH the k_fundamental cutoff (no modes below 2pi/L) and the
Nyquist cutoff (no modes above pi*N/L) to C_true, then recompute M_clust theory.

The box has DISCRETE 2D k-modes: k_perp = 2*pi*sqrt(nx^2+ny^2)/L for integer nx,ny.
Below k_f = 2pi/L, there are no modes.  Above k_Nyq = pi*N/L, there are no modes.
The Limber C_true(L) = b1^2 P(L/chi)/(L chi^2) is wrong outside this range.
"""
import sys, os, gc, time
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

PLKjKk_full = np.load(os.path.join(root, "notebooks", "data", "PLKjKk_lambda4000.npy"))
wl_full = PLKjKk_full / (4 * np.pi)
lambda_max_data = len(PLKjKk_full)

W_floor = N**2 * Nskew / (4 * np.pi)

# Physical scales
k_f = 2 * np.pi / L_box
k_Nyq = np.pi * N / L_box
L_f = k_f * chi_eff
L_Nyq = k_Nyq * chi_eff
print(f"k_f = 2pi/L = {k_f:.6f}, L_f = k_f*chi = {L_f:.1f}")
print(f"k_Nyq = pi*N/L = {k_Nyq:.4f}, L_Nyq = k_Nyq*chi = {L_Nyq:.1f}")
print(f"chi_eff = {chi_eff:.1f}")
print(f"First bin: ℓ = 0..{NperBin-1}")

# diag_cl
kvals = np.fft.fftfreq(N, d=1.0) * (2 * np.pi * N / L_box)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin_ref(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2_theory = b1_ref**2 * N**2 / L_box**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2_theory / (4 * np.pi)
print(f"\ndiag_cl = {diag_cl:.4e}")

# ---- Build theories with different C_true prescriptions ----
Nl_large = 2000

wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, lambda_max_data)
wl_raw[:n_avail] = wl_full[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()

ells_ext = np.arange(Nl_large, dtype=float)
kperp_ext = (ells_ext + 0.5) / chi_eff

# Version A: No cutoff (current)
cl_A = b1_ref**2 * plin_ref(kperp_ext) / (L_box * chi_eff**2)

# Version B: k_fundamental cutoff only
cl_B = cl_A.copy()
cl_B[kperp_ext < k_f] = 0.0

# Version C: Both k_f and k_Nyq cutoffs
cl_C = cl_A.copy()
cl_C[kperp_ext < k_f] = 0.0
cl_C[kperp_ext > k_Nyq] = 0.0

# Version D: Smooth transition at k_f (use sinc² window to model the discrete grid)
# P_discrete ≈ P_lin * sinc²(k_x L/2piN) * sinc²(k_y L/2piN) ≈ P * sinc⁴(kperp L/2piN)
# For isotropic average, roughly P * sinc²(kperp/k_Nyq * pi/2)
# Actually, this is the pixel window for the grid. But for the 2D DFT on a grid,
# the modes are at discrete k values, not continuous.
# Let's try a different approach: compute the EXACT discrete C_true from the 2D mode sum

# Version E: Exact C_true from discrete 2D mode counting
# C_L should be computed as an average of P(k_perp) over all k-modes in a ring L±0.5
# This naturally accounts for the discrete grid
print("\nComputing exact discrete C_true from 2D mode counting...")
kvals_1d = np.fft.fftfreq(N, d=L_box/(2*np.pi*N))  # k values in h/Mpc
kx2d, ky2d = np.meshgrid(kvals_1d, kvals_1d)
kperp_grid = np.sqrt(kx2d**2 + ky2d**2).ravel()
# Angular multipole for each 2D mode
ell_mode = kperp_grid * chi_eff  # L = k_perp * chi_eff

# For each ell bin [L, L+1), count modes and sum P(k_perp)
cl_discrete = np.zeros(Nl_large)
mode_count = np.zeros(Nl_large)
for i in range(len(kperp_grid)):
    L_idx = int(ell_mode[i])
    if 0 < L_idx < Nl_large and kperp_grid[i] > 0:
        cl_discrete[L_idx] += plin_ref(kperp_grid[i])
        mode_count[L_idx] += 1

# Average P in each ell bin, then convert to C_true
# C_true(L) = b1^2 * <P(k_perp)>_ring / (L_box * chi_eff^2)
# But we want the TOTAL contribution, weighted by mode count
# Actually, in the Limber framework: sum over modes in the ring
# The 2D power in the ring [L, L+1) is: (b1^2 / L_box^3) * N^2 * sum_modes P(k)
# And C_true(L) = (1/chi^2) * (1/(2pi*L)) * this  ??? No...

# Let me think more carefully. The 2D field on the grid is:
# delta_2D(x,y) = b1 * (1/N) sum_alpha delta_3D(x,y,chi_alpha)  [N equal-weight average]
# Hmm, actually w_j = sum_alpha delta_F(j,alpha), no normalization.
# So P_2D(k_perp) = <|w_k|^2> / L_box^2 = b1^2 * N^2 * P_3D(k) / L_box^3 * L_box^2 ??
# 
# Actually, <|w_k|^2> = b1^2 * N^2 P_3D(k_perp, k_z=0) / V  [for unnormalized DFT on grid]
# where V = L_box^3. And |w_k|^2 summed over k gives <w^2> * N_sightlines... 
# 
# Let me just take a simpler approach.

# From C_true = b1^2 P(L/chi) / (L chi^2), the continuous integral:
# sum_L (2L+1) C_true(L) = sum_L (2L+1) b1^2 P(L/chi)/(L chi^2) ≈ 2 b1^2/chi ∫ P(k)dk
#
# The discrete version should use the actual modes:
# Replace the integral ∫ P(k) dk with the sum 1/Δk_L Σ_modes P(k_mode)
# where Δk_L (the spacing in k_perp corresponding to ΔL=1) is 1/chi.

# So the discrete C_true(L) is:
# C_discrete(L) = b1^2 / (chi * L_box * chi) * Σ_{modes in ring L} P(k_mode) * (1/N_ring_continuous)
# where N_ring_continuous = 2L+1 is the number of modes in the ring if continuous.
# Hmm, this isn't right either.

# Let me go back to basics.
# <w_j w_k> = Σ_L C_true(L) P_L(cos gamma_{jk})
# 
# The off-diagonal part:
# <w_j w_k>_{j≠k} = (b1^2 N^2 / L^3) Σ_{k_perp} P(k_perp) exp(i k_perp · theta_{jk} * chi)
#
# The angular correlation function:
# = (b1^2 N^2 / L^3) Σ_{k_perp} P(k_perp) J_0(k_perp theta chi)  [2D isotropic]
#
# But in the MASTER framework, this is also:
# = Σ_L C_true(L) P_L(cos gamma) ≈ Σ_L C_true(L) J_0(L gamma)
#
# Comparing: C_true(L) = (b1^2 N^2 / L^3) × (density of states at k = L/chi)
# The 2D density of states at k in the discrete grid: 
# n(k) dk = Σ_{nx,ny} delta(k - |k_{nx,ny}|) dk ≈ (L^2/(2pi)^2) 2pi k dk  (continuous limit)
# = (L^2 k / (2pi)) dk

# So C_true(L) dL = (b1^2 N^2 / L^3) × P(L/chi) × (L^2/(2pi)) × (L/chi) × (1/chi) dL
# Wait, k = L/chi, dk = dL/chi.
# C_true(L) = b1^2 N^2 / L^3 × P(L/chi) × L^2/(2pi) × (L/chi) × (1/chi)
# = b1^2 N^2 P(L/chi) × L / (2pi L chi^2)
# = b1^2 N^2 P(L/chi) / (2pi L_box chi^2)   [using N^2/L^3 × L^2 × L/(chi^2) = ...]

# OK this is getting complicated. Let me just compute numerically.

# More direct approach: compute the pair-counting theory at each ℓ bin
# For j≠k pairs: <Cl>_offdiag = (1/(4pi)) Σ_{j≠k} <w_j w_k> P_l(cos gamma)
# For j=j pairs: <Cl>_diag = diag_cl
# 
# And <w_j w_k> = (b1^2 N^2 / L^3) Σ_{kx,ky} P(kperp) exp(i k·(yj-yk, zj-zk))
#
# This can be rewritten as:
# Σ_{j≠k} <w_j w_k> P_l(cos gamma) = (b1^2 N^2 / L^3) Σ_k P(kperp) 
#     × Σ_{j≠k} exp(i k·Δr_jk) P_l(cos gamma_jk)
# 
# = (b1^2 N^2 / L^3) Σ_k P(kperp) [|Σ_j exp(i k·r_j) Yl_m(nj)|^2 - Σ_j]  ← too complex

# Let me just do the simplest thing: zero C_true below L_f and above L_Nyq.

# First, check what the mode count looks like
print(f"\nMode count in first few L-bins:")
print(f"{'L':>5s} {'modes':>8s} {'continuous':>12s} {'ratio':>8s}")
for L in range(35):
    cont_modes = 2 * np.pi * (L + 0.5) / (chi_eff * k_f)  # rough expected count
    m = mode_count[L]
    print(f"{L:5d} {m:8.0f} {cont_modes:12.1f} {m/cont_modes if cont_modes>0 else 0:8.3f}")

# Build theories
theories_list = [
    ("A: Limber (current)", cl_A),
    ("B: k_f cutoff", cl_B),
    ("C: k_f + k_Nyq cutoff", cl_C),
]

# Compute all theories
print("\n" + "=" * 100)
results = {}
for label, cl_true in theories_list:
    theory = (M_clust @ cl_true)[:Nl] + diag_cl
    results[label] = theory

# Also compute with floor_converged instead of diag_cl, for C version
L_max_conv = 8000
ells_big = np.arange(L_max_conv, dtype=float)
kperp_big = (ells_big + 0.5) / chi_eff
cl_true_cut = np.where((kperp_big >= k_f) & (kperp_big <= k_Nyq),
                       b1_ref**2 * plin_ref(kperp_big) / (L_box * chi_eff**2),
                       0.0)
floor_conv = W_floor / (4 * np.pi) * np.sum((2 * ells_big + 1) * cl_true_cut)
print(f"floor_converged (with cutoffs) = {floor_conv:.4e}")
print(f"diag_cl = {diag_cl:.4e}")
print(f"ratio = {floor_conv/diag_cl:.6f}")

theory_best = (M_clust @ cl_C)[:Nl] + floor_conv
results["E: k_f+k_Nyq + floor_conv"] = theory_best

# Bin and compare
couple_wl = Wigner3j.CoupleMat(Nl, wl_ref)
coupling_wl = couple_wl.compute_matrix()
MD = MaskDeconvolution(Nl, wl_ref, precomputed_Wigner=coupling_wl)
bins = MD.binning_matrix('linear', 0, NperBin)
ells = np.arange(Nl, dtype=float)
binned_ells = bins @ ells

cl_mean = np.mean(cl_k_all, axis=0)
binned_raw = bins @ cl_mean

print(f"\n{'ell':>8s}", end="")
for label in results:
    short = label[:20]
    print(f" {short:>12s}", end="")
print()
print("-" * (8 + 13 * len(results)))

for i in range(len(binned_ells)):
    print(f"{binned_ells[i]:8.1f}", end="")
    for label in results:
        bt = bins @ results[label]
        r = binned_raw[i] / bt[i] if bt[i] > 0 else 0
        print(f" {r:12.4f}", end="")
    print()

for label in results:
    bt = bins @ results[label]
    ratios = binned_raw / bt
    print(f"\n{label}:")
    print(f"  mean(excl bin0) = {np.mean(ratios[1:]):.4f} ± {np.std(ratios[1:]):.4f}")
    print(f"  mean(all bins)  = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    print(f"  bin0 ratio      = {ratios[0]:.4f}")
