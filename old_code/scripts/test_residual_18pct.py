#!/usr/bin/env python
"""
Find the remaining ~1.8% after geometry correction.

Two known effects:
 1. Flat-plane geometry: r_j > chi_0 -> need <P(ell/r_j)/r_j^2> (+2.5% fix)
 2. Limber discrete sum: angular sum != continuous integral (~3% deficit of sigma^2 )

Can we combine them to get ratio -> 1.000?

Also test: is the diag_cl exactly right?
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_mean = np.mean(d['cl_k'], axis=0)
cl_all = d['cl_k']
N = int(d['Nk'])
L = float(d['L'])
Nl = 500
Nskew = int(d['Nskew'])
num_sim = cl_all.shape[0]

PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
b1 = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

chi_0 = 5000.0
W_floor = N**2 * Nskew / (4*np.pi)

# Exact <w^2> from discrete 2D modes
kvals = np.fft.fftfreq(N, d=1.0) * (2*np.pi*N/L)
kx, ky = np.meshgrid(kvals, kvals)
K_perp = np.sqrt(kx**2 + ky**2).ravel()
Pk_flat = plin(np.where(K_perp > 0, K_perp, 1e-10))
Pk_flat[K_perp == 0] = 0
w2 = b1**2 * N**2 / L**3 * np.sum(Pk_flat)
diag_cl = Nskew * w2 / (4*np.pi)

# Exact sightline positions for Nskew=9600
np.random.seed(100)
inds = np.unique(np.random.randint(0, N, size=(9797, 2)), axis=0)
coords_grid = np.linspace(0, L, N)
y_sl = coords_grid[inds[:, 1]]
z_sl = coords_grid[inds[:, 0]]
r_j = np.sqrt(chi_0**2 + y_sl**2 + z_sl**2)
chi_eff = np.mean(r_j)  # ~5125

# Coupling matrix
Nl_large = 2000
wl_needed = 2 * Nl_large - 1
wl_raw = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_ext))
wl_raw[:n_avail] = wl_ext[:n_avail]
wl_raw[n_avail:] = W_floor
wl_clust = wl_raw - W_floor

couple = Wigner3j.CoupleMat(Nl_large, wl_clust)
M_clust = couple.compute_matrix()
del couple; gc.collect()

NperBin = 32
n_bins = Nl // NperBin
ells_ext = np.arange(Nl_large, dtype=float)

def compute_ratio(theory_cl, label=""):
    """Compute binned ratio (data/theory) for ell=48..464"""
    ratios = []
    for b in range(1, n_bins - 1):
        lo = b * NperBin
        hi = (b + 1) * NperBin
        ratios.append(np.mean(cl_mean[lo:hi]) / np.mean(theory_cl[lo:hi]))
    ell_ctrs = [(b*NperBin + (b+1)*NperBin - 1)/2.0 for b in range(1, n_bins-1)]
    slope = np.polyfit(ell_ctrs, ratios, 1)[0]
    m, s = np.mean(ratios), np.std(ratios)
    print(f"  {label:50s}: r = {m:.5f} ± {s:.5f}  slope={slope:+.2e}")
    return m, s

# ================================================================== #
print("="*80)
print("A) Baseline approaches")
print("="*80)

# A1: chi_0
C_true = b1**2 * plin((ells_ext+0.5)/chi_0) / (L * chi_0**2)
th = (M_clust @ C_true)[:Nl] + diag_cl
compute_ratio(th, f"chi_0={chi_0:.0f}")

# A2: <r_j>
C_true = b1**2 * plin((ells_ext+0.5)/chi_eff) / (L * chi_eff**2)
th = (M_clust @ C_true)[:Nl] + diag_cl
compute_ratio(th, f"<r_j>={chi_eff:.0f}")

# A3: Sightline-averaged C_true
C_true_avg = np.zeros(Nl_large)
for el_idx in range(Nl_large):
    k_j = (el_idx + 0.5) / r_j
    C_true_avg[el_idx] = b1**2 * np.mean(plin(k_j) / r_j**2) / L
th_avg = (M_clust @ C_true_avg)[:Nl] + diag_cl
compute_ratio(th_avg, "<P(ell/r_j)/r_j^2>")

# ================================================================== #
print(f"\n{'='*80}")
print("B) Check diag_cl from the simulation data directly")
print("="*80)

# The diagonal term diag_cl = (1/4pi) Sum_j <w_j^2>
# For 100 sims, we can estimate this from the high-ell plateau
# If the off-diagonal decays to zero at high ell, then cl -> diag_cl
print(f"  Theoretical diag_cl = {diag_cl:.4e}")

# High-ell average from data (should approach diag_cl if off-diag is small)
for ell_range in [(400,500), (450,500), (480,500)]:
    avg_cl_high = np.mean(cl_mean[ell_range[0]:ell_range[1]])
    print(f"  cl_mean[{ell_range[0]}:{ell_range[1]}] = {avg_cl_high:.4e}  "
          f"(ratio to diag_cl: {avg_cl_high/diag_cl:.4f})")

# The actual <w_j^2> from the simulations should be stored as wl[0]/Nskew * 4pi
# or equivalently: diag_cl_sim = cl_data at ell where off-diag is negligible
# But off-diag is NOT negligible even at ell=500 because the patch has power there.

# ================================================================== #
print(f"\n{'='*80}")
print("C) Sightline-averaged C_true + test different diag_cl values")
print("="*80)

# Try adjusting diag_cl up/down by a few percent
for diag_factor, label in [(0.95, 'diag*0.95'), (0.97, 'diag*0.97'),
                             (1.00, 'diag*1.00'), (1.03, 'diag*1.03'),
                             (1.05, 'diag*1.05')]:
    th = (M_clust @ C_true_avg)[:Nl] + diag_cl * diag_factor
    compute_ratio(th, f"<P/r^2> + {label}")

# ================================================================== #
print(f"\n{'='*80}")
print("D) Cross-check: is diag_cl consistent with the data's ell-average?")
print("   diag_cl = off-diagonal floor + sightline variance")
print("   Theory: diag_cl = Nskew * <w^2> / (4pi)")
print(f"{'='*80}")

# The total pseudo-Cl is: <Cl> = Cl_offdiag(ell) + diag_cl
# At ell=0: Cl_offdiag should be maximal (all pairs correlated)
# At ell>>1 much larger than patch size: Cl_offdiag -> 0

# But our wl_clust at ell=500 is NOT zero (wl_clust[499] < 0 typically
# because we subtract W_floor from a noisy wl)
print(f"  wl_ext[0] = {wl_ext[0]:.4e}")
print(f"  wl_ext[499] = {wl_ext[499]:.4e}")
print(f"  W_floor  = {W_floor:.4e}")
print(f"  wl_ext[499] / W_floor = {wl_ext[499]/W_floor:.6f}")

# The M_clust @ C_true part at each ell
theory_clust_only = (M_clust @ C_true_avg)[:Nl]
print(f"\n  theory_clust (off-diag coupling) at ell=50:  {theory_clust_only[50]:.4e}")
print(f"  theory_clust at ell=250: {theory_clust_only[250]:.4e}")
print(f"  theory_clust at ell=450: {theory_clust_only[450]:.4e}")
print(f"  diag_cl             =    {diag_cl:.4e}")
print(f"  fraction of total at ell=50:  {diag_cl/(theory_clust_only[50]+diag_cl):.3f}")
print(f"  fraction of total at ell=250: {diag_cl/(theory_clust_only[250]+diag_cl):.3f}")
print(f"  fraction of total at ell=450: {diag_cl/(theory_clust_only[450]+diag_cl):.3f}")

# ================================================================== #
print(f"\n{'='*80}")
print("E) Direct pair-counting theory at chi_eff (no MASTER)")
print("   Use M^(pk) @ PLKjKk / (4pi * 2pi * chi^2 * (4pi)^2)")
print(f"{'='*80}")

# Build M^(pk): coupling matrix with P_F as input spectrum
for chi_label, chi in [('chi_0', chi_0), ('<r_j>', chi_eff)]:
    pk_at_chi = b1**2 * plin(np.arange(Nl, dtype=float) / chi)
    couple_pk = Wigner3j.CoupleMat(Nl, pk_at_chi)
    M_pk = couple_pk.compute_matrix()
    
    PLKjKk_Nl = PLKjKk[:Nl]
    theory_pair = (M_pk @ PLKjKk_Nl) / (4*np.pi * 2*np.pi * chi**2 * (4*np.pi)**2)
    
    compute_ratio(theory_pair, f"Pair-counting at {chi_label}={chi:.0f}")
    del couple_pk, M_pk; gc.collect()

# ================================================================== #
print(f"\n{'='*80}")
print("F) Try: C_true from exact discrete 2D modes mapped to ell via r_j")
print("   Instead of P_lin(ell/r), bin the exact discrete modes")
print(f"{'='*80}")

# For each 2D mode (kx, ky), it maps to ell = k_perp * r_j for each sightline
# Averaged over sightlines, the effective ell for each mode is <k_perp * r_j>

# Simpler: for each ell, find which discrete modes contribute
# ell = k * r_j -> k = ell/r_j
# With r_j varying from 5000 to 5367, the range of k for ell=100 is
# k in [100/5367, 100/5000] = [0.01863, 0.020]

# The number of 2D modes per unit dk is ~ 2pi k L^2 / (2pi)^2 = k L^2 / (2pi)
# This should be well-sampled.

# Discrete C_true: for each ell, average P_F(ell/r_j)/r_j^2 over sightlines,
# but using the discrete P evaluated at the exact k=ell/r_j...
# Actually, for the GRF generated in the box, between the discrete modes
# the power is zero. The expectation of |delta_k|^2 is only nonzero at
# the grid k-modes.
#
# But the Limber formula uses a SMOOTH P(k), which is the input for the GRF.
# Since the GRF amplitudes are drawn from N(0, sqrt(P(k)/2)), the expectation
# is exactly P_lin(k) at each mode. So in the 100-sim average, the smooth
# P_lin should be the right input for C_true. No discreteness correction needed.

print("  The GRF is drawn from smooth P_lin(k). The 100-sim average")
print("  converges to the expectation, so P_lin is exact for C_true.")
print("  The discrete mode spacing doesn't affect the EXPECTATION,")
print("  only the variance (sample variance from finite number of modes).")

# ================================================================== #
print(f"\n{'='*80}")
print("G) Best fit: combine geometry + diag_cl adjustment")
print("   Find (chi, alpha) such that M@C_true(chi) + alpha*diag_cl best fits")
print(f"{'='*80}")

from scipy.optimize import minimize

def chi2_func(params):
    chi, alpha = params
    C = b1**2 * plin((ells_ext+0.5)/chi) / (L * chi**2)
    th = (M_clust @ C)[:Nl] + alpha * diag_cl
    mask = np.ones(Nl, dtype=bool)
    mask[:32] = False
    mask[480:] = False
    return np.sum(((cl_mean[mask] - th[mask])/th[mask])**2)

result = minimize(chi2_func, [5125, 1.0], method='Nelder-Mead')
chi_best, alpha_best = result.x
print(f"  Best fit: chi = {chi_best:.1f}, diag_cl_factor = {alpha_best:.5f}")
print(f"  chi_best / chi_0 = {chi_best/chi_0:.4f}")
print(f"  chi_best / <r_j> = {chi_best/chi_eff:.4f}")

C = b1**2 * plin((ells_ext+0.5)/chi_best) / (L * chi_best**2)
th = (M_clust @ C)[:Nl] + alpha_best * diag_cl
compute_ratio(th, f"Best fit: chi={chi_best:.0f}, a={alpha_best:.4f}")

# Also try with geometry-averaged C_true + free alpha
def chi2_func2(alpha):
    th = (M_clust @ C_true_avg)[:Nl] + alpha * diag_cl
    mask = np.ones(Nl, dtype=bool)
    mask[:32] = False
    mask[480:] = False
    return np.sum(((cl_mean[mask] - th[mask])/th[mask])**2)

from scipy.optimize import minimize_scalar
result2 = minimize_scalar(chi2_func2, bounds=(0.8, 1.2), method='bounded')
alpha2 = result2.x
print(f"\n  With <P/r^2>: best diag_factor = {alpha2:.5f}")
th2 = (M_clust @ C_true_avg)[:Nl] + alpha2 * diag_cl
compute_ratio(th2, f"<P/r^2> + diag*{alpha2:.4f}")
