#!/usr/bin/env python
"""
Direct numerical test of C_true.

Instead of trying to converge the MASTER sum, compute the THEORY prediction
from first principles for a SINGLE realization, using the actual discrete
2D modes, and compare with the measured pseudo-Cl.

For a single realization with Nskew sightlines:
  w_j = (N^3/L^{3/2}) * SUM_{k_perp} a(k_perp,0)/N^3 * N * exp(2pi*i*k_perp.n_j/N)
      = (1/L^{3/2}) * SUM_{k_perp} a(k_perp,0) * N * exp(...)

  a_lm = SUM_j w_j Y_lm^*(hat{n}_j)
  Cl = |a_lm|^2 / (2l+1)

Instead of ensemble-averaging, I'll compute the EXPECTED pseudo-Cl from
the known amplitudes a(k_perp, 0) for one realization. Each k-mode contributes
independently, so:

<Cl>_theory = SUM_{k_perp} |a(k_perp,0)|^2 * |SUM_j Y_lm^*(n_j) exp(ik_perp.r_j)|^2 / (2l+1) * (N/L^{3/2})^2

But this is complicated. Let me instead do a simpler test:

For a SINGLE realization, compute w_j from the known amplitudes, then run SHT,
and check the normalization using a known constant mode.

Actually, the simplest test: inject a KNOWN signal w_j = const, run SHT, check Cl.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
from sht.sht import DirectSHT
from sht.lya_sfb import _alm2cl_complex
import fast_Wigner3j as Wigner3j

# Load the 100-sim cache to get the same sightline geometry
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_k_all = d['cl_k']
wl_k = d['wl_k']
Nskew_data = int(d['Nskew'])
N = int(d['Nk'])
L = float(d['L'])
Nl = 500
cl_mean = np.mean(cl_k_all, axis=0)
wl_ref = wl_k[0, :Nl]

print(f"N={N}, L={L}, Nskew={Nskew_data}, Nl={Nl}")
print(f"cl_mean[0:5] = {cl_mean[:5]}")
print(f"wl_ref[0:5] = {wl_ref[:5]}")

# ---- Step 1: What are the typical values? ----
# cl_mean is the average pseudo-Cl from the 100 simulations.
# wl_ref is the angular window from the random catalog.
# The theory prediction is: <pseudo_Cl> = SUM_l' M[l,l'] * C_true[l']

# Load cosmology
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=42, verbose=False)
plin = GRF.plin
b1 = GRF.my_bias
chi_shift = 5000
chi_bar = chi_shift + L / 2.0

# ---- Step 2: What IS C_true? ----
# From the DENSITY field normalization:
# density(r) = ifftn(amp) * L^{3/2} / (L/N)^3  [from density_field()]
# <|amp(k)|^2> = b1^2 * Plin(k)
# w_j = SUM_z density(x_j, y_j, z) = (N/L^{3/2}) * SUM_{k_perp} b1*a(k_perp,0) * exp(ik.r_j)
# Wait, I keep going back and forth. Let me be super precise.

# density_field: dens = real(ifftn(amplitudes)) * boxvol^{1/2} / pix
# where amplitudes = b1 * a(k), <|a(k)|^2> = P(k)
# boxvol = L^3, pix = (L/N)^3
# dens = ifftn(b1*a) * L^{3/2} / (L/N)^3 = ifftn(b1*a) * N^3/L^{3/2}
# ifftn has 1/N^3 factor, so:
# dens(r) = (N^3/L^{3/2}) * (1/N^3) * SUM_k b1*a(k)*exp(2pi*i*k.n/N) = (b1/L^{3/2}) * SUM_k a(k)*exp(...)

# w_j = SUM_{nz} dens(nx_j, ny_j, nz) = (b1/L^{3/2}) * SUM_k SUM_nz a(k)*exp(2pi*i*k.n/N)
# SUM_nz exp(2pi*i*kz*nz/N) = N * delta_{kz,0}
# w_j = (b1*N/L^{3/2}) * SUM_{k_perp} a(k_perp,0) * exp(2pi*i*k_perp.n_j/N)

# For the SHT with Nskew sightlines:
# alm = SUM_j w_j Y_lm^*(n_j)
# Cl = |alm|^2 / (2l+1)

# <Cl> = SUM_{j,k} <w_j w_k*> Y_lm^*(n_j) Y_lm(n_k) / (2l+1)
# = (b1^2 * N^2 / L^3) * SUM_{k_perp} P(k_perp) * |SUM_j Y_lm^*(n_j) exp(ik.r_j)|^2 / (2l+1)
# = (b1^2 * N^2 / L^3) * SUM_{k_perp} P(k_perp) * F_l(k_perp)

# where F_l(k_perp) = (1/(2l+1)) SUM_m |SUM_j Y_lm^*(n_j) exp(ik_perp.r_j)|^2

# In the MASTER framework, this is rewritten as:
# <Cl> = SUM_l' M[l,l'] * S_l'
# where S_l' incorporates the signal power at multipole l'.

# The angular power of the field w(n) (if defined continuously) would be:
# C_l^{ww} = <|a_lm^w|^2>/(2l+1) where a_lm^w = integral w(n) Y_lm^*(n) dOmega
# But w(n) is only defined at discrete sightline positions.

# The WHOLE POINT of the MASTER approach is:
# a_lm^{pseudo} = SUM_j w_j Y_lm^*(n_j)  [discrete sum, no dOmega]
# <pseudo Cl> = SUM_l' M[l,l'] * C_true[l']
# where M[l,l'] = (2l'+1)/(4pi) SUM_lambda (2lambda+1) W[lambda] 3j^2
# and W[lambda] = <|a_lm^W|^2>/(2lambda+1) = wl
# with a_lm^W = SUM_j 1 * Y_lm^*(n_j)

# And C_true[l'] is the TRUE angular power spectrum of the CONTINUOUS field
# s(nhat) ≡ w(nhat) evaluated at any direction nhat on the sky
# (not just at the sightline positions).

# What is this continuous field? 
# s(nhat) = (b1*N/L^{3/2}) * SUM_{k_perp} a(k_perp,0) * exp(ik_perp . r_perp(nhat))
# where r_perp(nhat) is the transverse position on the box face for direction nhat.

# For small angle from the box center: r_perp = chi_bar * theta_nhat
# More precisely: r_perp depends on the geometry. The box face is at some distance
# and the sightlines map to transverse positions.

# The angular power of s(nhat):
# a_lm^s = integral s(nhat) Y_lm^*(nhat) dOmega
# This is a Fourier-Bessel type integral.

# For a small-angle patch at distance chi_bar, the flat-sky approximation gives:
# a_lm^s ≈ integral s(theta) e^{-i*l*theta} d^2theta  [schematic]
# C_l^s = P_2D(l) * [some normalization involving chi_bar and the transverse geometry]

# The 2D angular power spectrum P_2D(l/chi):
# s(nhat) = (b1*N/L^{3/2}) * SUM_{k_perp} a(k_perp,0) * exp(ik_perp . chi_bar * theta)
# This is a 2D Fourier series with modes at k_perp * chi_bar in angular space.
# The angular power is:
# P_2D^{ang}(ell) = (b1^2*N^2/L^3) * <|a(k)|^2> * (volume factor for mode density)

# In the continuous limit:
# SUM_{k_perp} -> (L/2pi)^2 integral d^2k_perp
# C_l = (b1^2*N^2/L^3) * (2pi/chi_bar^2) * P(l/chi_bar) * ... 

# I keep getting confused. Let me just do the numerical test.

# ---- Step 3: Numerical verification ----
# Take ONE realization. The EXACT Cl is:
# Cl = (1/(2l+1)) SUM_m |SUM_j w_j Y_lm^*(n_j)|^2

# For the theory, compute:
# Cl_theory = SUM_l' M[l,l'] * C_true[l']
# where C_true is parameterized as alpha * b1^2 * P((l+0.5)/chi) / chi^2
# and I want to find alpha.

# From the 100-sim data, <Cl> / [M @ (b1^2 * P / chi^2)] should give alpha.

print("\n---- Step 3: Determine alpha from data ----")
# Use a small Nl_large first, acknowledging truncation
Nl_test = 500
ells = np.arange(Nl_test, dtype=float)
# Unnormalized C_true (without alpha):
cl_unnorm = b1**2 * plin((ells + 0.5) / chi_bar) / chi_bar**2

# Coupling matrix with the cached wl (500 multipoles)
wl_for_couple = np.zeros(2*Nl_test - 1)
wl_for_couple[:Nl_test] = wl_ref[:Nl_test]
couple = Wigner3j.CoupleMat(Nl_test, wl_for_couple)
M = couple.compute_matrix()

cl_theory_unnorm = M @ cl_unnorm

# Fit alpha: cl_mean = alpha * cl_theory_unnorm
# Use ell range 30..400 to avoid boundaries
mask = np.ones(Nl_test, dtype=bool)
mask[:30] = False
mask[450:] = False

alpha_fit = np.mean(cl_mean[mask]) / np.mean(cl_theory_unnorm[mask])
print(f"alpha_fit = {alpha_fit:.6e}")
print(f"1/(32*pi^3) = {1/(32*np.pi**3):.6e}")
print(f"N^2/(2*pi*L) = {N**2/(2*np.pi*L):.6e}")
print(f"alpha_fit / [1/(32*pi^3)] = {alpha_fit / (1/(32*np.pi**3)):.4f}")
print(f"alpha_fit / [N^2/(2*pi*L)] = {alpha_fit / (N**2/(2*np.pi*L)):.4f}")

# Also check various simple scalings:
print(f"\nalpha_fit * chi^2 = {alpha_fit * chi_bar**2:.4f}")  # should show P-scaling
print(f"alpha_fit * chi^2 * L = {alpha_fit * chi_bar**2 * L:.4f}")
print(f"alpha_fit * chi^2 * L / N^2 = {alpha_fit * chi_bar**2 * L / N**2:.6e}")
print(f"alpha_fit * chi^2 * L * 2*pi / N^2 = {alpha_fit * chi_bar**2 * L * 2*np.pi / N**2:.6e}")
print(f"alpha_fit * chi^2 * 4*pi^2 = {alpha_fit * chi_bar**2 * 4*np.pi**2:.6e}")
print(f"alpha_fit * 32*pi^3*chi^2 = {alpha_fit * 32*np.pi**3 * chi_bar**2:.6e}")

# key: what factor, in combination with b1^2*P/chi^2, gives the data?
# C_true = alpha * b1^2 * P / chi^2
# The notes say alpha = 1/(32*pi^3)
# My derivation suggests alpha = N^2/(2*pi*L)

# But this depends on the TRUNCATION of the coupling sum.
# Let me also try with different ell ranges to see if alpha is stable.
print(f"\nalpha stability:")
for lo, hi in [(10,100), (30,200), (50,300), (100,400), (200,450)]:
    m = np.zeros(Nl_test, dtype=bool)
    m[lo:hi] = True
    a = np.mean(cl_mean[m]) / np.mean(cl_theory_unnorm[m])
    print(f"  ell={lo:3d}-{hi:3d}: alpha = {a:.6e}")

# ---- Step 4: What about using wl to 2000 for better convergence? ----
print("\n---- Step 4: With extended PLKjKk ----")
PLKjKk = np.load('notebooks/data/PLKjKk_lambda2000.npy')
wl_ext = PLKjKk / (4*np.pi)

Nl_large = 1000
ells_ext = np.arange(Nl_large, dtype=float)
cl_unnorm_ext = b1**2 * plin((ells_ext + 0.5) / chi_bar) / chi_bar**2

wl_for_couple_ext = np.zeros(2*Nl_large - 1)
n_avail = min(2*Nl_large-1, len(wl_ext))
wl_for_couple_ext[:n_avail] = wl_ext[:n_avail]
couple_ext = Wigner3j.CoupleMat(Nl_large, wl_for_couple_ext)
M_ext = couple_ext.compute_matrix()

cl_theory_ext = (M_ext @ cl_unnorm_ext)[:Nl]

mask2 = np.ones(Nl, dtype=bool)
mask2[:30] = False
mask2[450:] = False
alpha_fit_ext = np.mean(cl_mean[mask2]) / np.mean(cl_theory_ext[mask2])
print(f"alpha_fit (Nl_large=1000, wl to 2000) = {alpha_fit_ext:.6e}")
print(f"alpha_fit_ext / [1/(32*pi^3)] = {alpha_fit_ext / (1/(32*np.pi**3)):.4f}")

# Also try Nl_large=1500
Nl_large2 = 1500
ells_ext2 = np.arange(Nl_large2, dtype=float)
cl_unnorm_ext2 = b1**2 * plin((ells_ext2 + 0.5) / chi_bar) / chi_bar**2
wl_for_couple_ext2 = np.zeros(2*Nl_large2 - 1)
n_avail2 = min(2*Nl_large2-1, len(wl_ext))
wl_for_couple_ext2[:n_avail2] = wl_ext[:n_avail2]
couple_ext2 = Wigner3j.CoupleMat(Nl_large2, wl_for_couple_ext2)
M_ext2 = couple_ext2.compute_matrix()
cl_theory_ext2 = (M_ext2 @ cl_unnorm_ext2)[:Nl]
alpha_fit_ext2 = np.mean(cl_mean[mask2]) / np.mean(cl_theory_ext2[mask2])
print(f"alpha_fit (Nl_large=1500, wl to 2000) = {alpha_fit_ext2:.6e}")
print(f"alpha_fit_ext2 / [1/(32*pi^3)] = {alpha_fit_ext2 / (1/(32*np.pi**3)):.4f}")

del GRF; gc.collect()
