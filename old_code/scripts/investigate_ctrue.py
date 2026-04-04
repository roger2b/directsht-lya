#!/usr/bin/env python
"""
Investigate whether C_true is correct.

The MASTER formula gives theory ~30% too high when we include all l' up to Nyquist.
This means either:
1. The C_true formula is wrong (wrong normalization or missing cutoff)
2. The decomposition SN approach is wrong (PLKjKk != SN at high lambda)
3. There's a subtlety about the box geometry we're missing

Let's check option 1: compare C_true with the ACTUAL power spectrum from the sims.
We can get the actual Cl from 100 sims by inverting the window:
  C_measured = (wl)^{-1} * pseudo_Cl
where wl is the window function (pair-counting normalization).

But actually, the measured pseudo-Cl at ell=0..499 IS what we're trying to match.
The question is whether C_true = Pl(k=ell/chi) / normalization is correct.

Actually, let me reconsider. The simulation generates a 3D GRF in a box with P(k).
The theoretical 2D angular power spectrum C_true[l] comes from projecting P(k):
  C_true[l] = b1^2 * P_lin(k_perp) / (32*pi^3*chi^2)  where k_perp = (l+0.5)/chi
This is the Limber-flat-sky result. But our box has finite extent:
  - The transverse box size L_perp ~27 deg at chi_bar. So small L modes are discrete.
  - The box has a discrete k-grid: k = 2*pi*n/L for integer n.
  - P_lin(k) in the simulation is evaluated on this grid and aliased by the FFT.
  
The key: at high ell (high k_perp), the discrete k-grid matters.
k_perp = (l+0.5)/chi, and the discrete grid spacing dk = 2*pi/L.
The number of modes per dl is roughly dk_perp/dk = L/chi * dl.
For L=1383, chi=5691: dk_perp for dl=1 is 1/chi ~ 1.8e-4 h/Mpc, while dk=2*pi/1383=4.5e-3 h/Mpc.
So dl for one grid step is dk*chi = 4.5e-3*5691 = 25.7.
This means each k-mode maps to about 26 ells. The continuous C_true formula may not
account for this discreteness.

BUT: the GRF is generated in 3D, and the SHT measures the projected 2D field.
The C_true formula should still be correct for the EXPECTATION of the pseudo-Cl,
as long as the projection (sum along LOS) is done correctly.

Let me instead check: is the problem simply that PLKjKk != SN at high lambda?
If PLKjKk < SN at high lambda (which it is: mean(PLKjKk[1500:]) = 2.43e9 < 2.52e9),
then the actual shot noise coupling is LESS than Poisson.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# Load cache
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_k_all = d['cl_k']
wl_k = d['wl_k']
N = int(d['Nk'])
L = float(d['L'])
Nskew = int(d['Nskew'])
Nl = 500
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)

# Load PLKjKk
PLKjKk = np.load('notebooks/data/PLKjKk_lambda2000.npy')
wl_full = PLKjKk / (4*np.pi)

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, Nskew={Nskew}, L={L:.1f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# ---- What if we DEFINE the effective shot noise level from the constraint ----
# If the theory should equal the data:
# <pseudo_Cl> = M_direct(Nl_large=1000) @ C_true + SN_eff * (sigma_sq_full - sigma_sq_1000)
# Set this equal to cl_mean and solve for SN_eff:
Nl_large = 1000
wl_needed = 2 * Nl_large - 1
wl_for_coupling = np.zeros(wl_needed)
n_avail = min(wl_needed, len(wl_full))
wl_for_coupling[:n_avail] = wl_full[:n_avail]

ells_ext = np.arange(Nl_large, dtype=float)
C_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)

couple = Wigner3j.CoupleMat(Nl_large, wl_for_coupling)
M = couple.compute_matrix()
theory_master_1000 = (M @ C_true_ext)[:Nl]
del couple, M; gc.collect()

# sigma^2 contributions
kNy = np.pi / (L/N)
ell_Ny = int(kNy * chi_bar)
ells_high = np.arange(ell_Ny + 1)
C_true_high = b1_ref**2 * plin_ref((ells_high + 0.5) / chi_bar) / (32*np.pi**3 * chi_bar**2)
sigma_sq_full = np.sum((2*ells_high+1)/(4*np.pi) * C_true_high)
sigma_sq_1000 = np.sum((2*ells_ext+1)/(4*np.pi) * C_true_ext)
delta_sigma_sq = sigma_sq_full - sigma_sq_1000

# Solve: cl_mean ≈ theory_master_1000 + SN_eff_4pi * delta_sigma_sq
# This is overdetermined (500 equations, 1 unknown).
# Least squares:
residual = cl_mean - theory_master_1000
SN_eff_4pi = np.mean(residual[10:]) / delta_sigma_sq
SN_eff = SN_eff_4pi * 4*np.pi
print(f"\nEffective SN from data:")
print(f"  SN_eff = {SN_eff:.4e}")
print(f"  SN_Poisson = {float(N**2 * Nskew):.4e}")
print(f"  Ratio: {SN_eff / (N**2 * Nskew):.4f}")
print(f"  mean(PLKjKk[100:])  = {np.mean(PLKjKk[100:]):.4e}")
print(f"  mean(PLKjKk[500:])  = {np.mean(PLKjKk[500:]):.4e}")
print(f"  mean(PLKjKk[1000:]) = {np.mean(PLKjKk[1000:]):.4e}")
print(f"  mean(PLKjKk[1500:]) = {np.mean(PLKjKk[1500:]):.4e}")

# ---- What does the PLKjKk actually look like at high lambda? ----
print(f"\n---- PLKjKk structure ----")
print(f"PLKjKk[0] = {PLKjKk[0]:.4e}")
SN_poisson = float(N**2 * Nskew)
for lam in range(0, 2000, 100):
    chunk = PLKjKk[lam:lam+100]
    print(f"  lambda {lam:4d}-{lam+99:4d}: mean={np.mean(chunk):.4e}, "
          f"std={np.std(chunk):.4e}, ratio to SN={np.mean(chunk)/SN_poisson:.4f}")

# ---- Key diagnostic: what SN level gives the right answer? ----
# theory_corrected[l] = theory_master_1000[l] + SN_trial/(4pi) * delta_sigma_sq
# We want mean(cl_mean[10:] / theory_corrected[10:]) = 1.0
print(f"\n---- Scanning SN level ----")
for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    SN_trial = frac * SN_poisson
    theory_trial = theory_master_1000 + SN_trial/(4*np.pi) * delta_sigma_sq
    ratio_trial = cl_mean[10:] / theory_trial[10:]
    print(f"  SN = {frac:.1f}*SN_Poisson: mean data/theory = {np.mean(ratio_trial):.4f}")

# ---- Maybe the issue is that C_true is wrong, not SN ----
# The Limber formula C_true = b1^2 * Plin(k_perp)/... may not be quite right.
# In the periodic box, the SHT gives:
# alm = SUM_j delta_j Y*_lm(nhat_j) where delta_j = (1/N) SUM_{kz} b1*delta_k3D(k_perp, kz) exp(ikz*zj)
# For the monopole k-bin (kz=0 or the full LOS average):
# delta_j^{k=0} = (1/N) SUM_{kz} b1*delta_k3D * exp(ikz*zj)   [all kz modes contribute]
# Wait -- what does cl_k_all store? Is it the k=0 mode only?
# From the summary: cl_k stores pseudo-Cl for the k=0 (monopole) radial mode.
# But that's the k-parallel = 0 mode -- the LOS-averaged field.
# For k_parallel=0: delta_j = (1/N) SUM_{z} delta(x_j, y_j, z)
# This is the mean of the 3D field along the LOS.
# P_2D(k_perp) = P_3D(k_perp, k_par=0) * L_LOS ... hmm, need to think about this.

# Actually looking at the normalization: C_true = b1^2 Plin(k)/32pi^3/chi^2
# Let me check what 32pi^3 comes from.
# 32pi^3 = 2pi * (4pi)^2 = 2pi * 16pi^2
# Or: 32pi^3 = (2pi)^3 * (4/pi) ... no.
# 32pi^3 = 2 * (4pi)^3 / (4pi) = ...
# More carefully: C_true comes from the sFB formalism. 
# For a periodic box with LOS length L_par:
# C_l(k) = (b1^2 / chi_bar^2) * (1/(2*pi*L_par)) * Integral over k_par of P_3D(k_perp, k_par)
# For k-mode k_par = 0: contribution is P_3D(k_perp, 0) / (2*pi*L_par)
# But the discrete k_par spacing is dk_par = 2*pi/L_par
# So the k=0 mode has: C_true = P_3D(k_perp, 0) * dk_par / (2*pi*L_par * chi_bar^2) ???

# Actually I think the normalization has been validated already.
# The issue must be something else. Let me check what happens if we use a LOWER C_true.
# If the 30% excess comes from double-counting: C_true is for ALL k_par modes,
# but the data is only the k_par=0 mode.

# How many k_par modes are there? N=512 modes.
# If the data shows one k-bin, and the theory sums over all k_par, the theory overcounts by N?
# No, the C_true formula already accounts for this. Let me re-derive.

# The field measured by the SHT for k-bin k_n is:
# delta_j^{k_n} = SUM_z w(z) * delta(x_j, y_j, z) * exp(-i*k_n*z)
# For k_n=0 (monopole): delta_j = SUM_z delta(x_j, y_j, z) * 1 = N * <delta>_LOS(j)

# Then pseudo-Cl = (1/(2l+1)) SUM_m |alm|^2 where alm = SUM_j delta_j^{k_n} Y*_lm(nhat_j)
# <pseudo-Cl> = SUM_{j,k} <delta_j^{k_n} delta_k^{k_n}*> Pl(cos_jk)/(4pi)

# For the GRF in periodic box:
# <delta_j^{k_n} delta_k^{k_n}*> = SUM_{k_par} |SUM_z exp(i(k_par-k_n)*z_z)|^2 * b1^2 * Plin(k3D) / L^3
# For k_n=0: = SUM_{k_par} N^2 delta_{k_par,0} * b1^2 * Plin(k_perp) / L^3
# Wait, that gives only k_par=0 contributing, which is what we want.
# <delta_j^{k=0} delta_k^{k=0}> = N^2 * b1^2 * Plin(k_perp=|rj-rk|_perp) / L^3 ??
# No, this is in Fourier space. Let me be more careful.

# delta_j^{k=0} = SUM_{z=0}^{N-1} delta(x_j, y_j, z_z)
# where delta(x,y,z) is the real-space density field.
# In terms of Fourier modes: delta(r) = (1/L^3) SUM_k delta_tilde(k) exp(ik.r) with L^3 = (N*dx)^3
# But actually the GRF generates amplitudes and inverse-FFTs.
# Let me just check numerically: compute C_true * wl_theory vs measured.

# What if the normalization was calibrated but only for Nl=500 MASTER (which underestimates)?
# The ratio = 0.975 at Nl_large=1000 was ACCIDENTALLY good because we're missing the
# shot-noise coupling from l'>1000.

# Let me be VERY concrete. 
# FACT 1: Direct MASTER at Nl_large=1000 gives theory/data = 0.969 (theory slightly low)
# FACT 2: Adding analyical SN from l'>1000 gives theory/data = 1.29 (theory way too high)
# FACT 3: The SN at high lambda (Poisson = 2.52e9) is the issue.
#
# If the SN were ZERO at high lambda (no shot noise), then theory=MASTER(1000), ratio=0.969.
# We need a small correction (~3%) from l'>1000.
# The question is: what is the TRUE shot noise level that couples l'>1000 to our measurements?
#
# The shot noise in PLKjKk comes from the pair-counting of sightline positions.
# For random positions: PLKjKk[lambda>>0] -> N^2 * Ns  (Poisson)
# For correlated positions: PLKjKk[lambda] depends on the angular distribution.
# Our sightlines are on a regular grid (golden spiral? uniform?).
# Let me check what the sightline distribution is.

# The SN_eff from fitting = SN_Poisson * some fraction:
SN_eff_ratio = SN_eff / SN_poisson
print(f"\nSN_eff / SN_Poisson = {SN_eff_ratio:.4f}")
print(f"\nThis means the effective SN at high lambda is {100*SN_eff_ratio:.1f}% of Poisson")
print(f"Alternatively, if C_true is wrong by a factor:")
print(f"  factor = cl_mean / (theory_master_1000 + SN_Poisson/(4pi)*delta_sigma_sq)")
factor = np.mean(cl_mean[10:]) / np.mean(theory_master_1000[10:] + SN_poisson/(4*np.pi)*delta_sigma_sq)
print(f"  factor = {factor:.4f}")
print(f"  If C_true is ~{factor:.2f}x what we compute, the theory would match.")
