#!/usr/bin/env python
"""
Check what P_lin looks like at the relevant scales and understand why
the MASTER sum isn't converging.
"""
import sys, os, gc
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF

GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin = GRF_tmp.plin
L = 1382.7
chi_bar = 5000 + L/2.0

# Check P_lin at k = ell/chi
print(f"{'ell':>6s} {'k (h/Mpc)':>10s} {'P_lin':>12s} {'(2l+1)*P/chi^2':>16s}")
print("-" * 50)
for ell in [100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000]:
    k = ell / chi_bar
    P = plin(k)
    contrib = (2*ell+1) * P / chi_bar**2
    print(f"  {ell:5d} {k:10.4f} {P:12.4f} {contrib:16.6e}")

# The Nyquist wavenumber of the grid:
N = 512
k_Ny = np.pi * N / L
ell_Ny = k_Ny * chi_bar
print(f"\nk_Ny = {k_Ny:.4f} h/Mpc")
print(f"ell_Ny = {ell_Ny:.0f}")

# P_lin at k_Ny
print(f"P_lin(k_Ny) = {plin(k_Ny):.6f}")

# The GRF amplitudes are only defined up to k_Ny. Above that, the true signal is zero.
# But the MASTER sum runs C_true to infinity, and plin is still > 0 there!
# So the correct C_true should be ZERO above ell_Ny.
# (Because the GRF has no power above k_Ny.)

print(f"\n=== KEY INSIGHT ===")
print(f"The GRF has no power above k_Ny = {k_Ny:.4f}, i.e., ell > {ell_Ny:.0f}")
print(f"So C_true(ell) should be 0 for ell > {ell_Ny:.0f}")
print(f"But ell_Ny = {ell_Ny:.0f} >> 3500, so this doesn't explain non-convergence at 3500.")

# Let me check: what's the actual P_2D of our discrete grid?
# P_2D(k_perp) = b1^2 * N^2 * P_lin(k_perp) / L  [verified]
# But this is only for k_perp values on the grid: k_perp = 2*pi*(nx,ny)/L
# The discrete sum SUM_{k_perp} has N^2 terms.
# In the Limber approximation, ell corresponds to k_perp = ell/chi
# The grid spacing in k is dk = 2*pi/L, corresponding to delta_ell = dk * chi = 2*pi*chi/L

delta_ell = 2*np.pi * chi_bar / L
print(f"\nGrid spacing in ell: delta_ell = 2*pi*chi/L = {delta_ell:.1f}")
print(f"This means the true C_l has 'comb' structure — it's nonzero only at")
print(f"ell values that correspond to grid k-modes, spaced by {delta_ell:.1f}")
print(f"A smooth Limber C_true OVERESTIMATES the power because it fills in between modes.")

# Number of k-modes per unit ell:
# In 2D, number of modes in ring [k, k+dk] = 2*pi*k*dk * (L/(2pi))^2 = L^2*k*dk/(2pi)
# Converting to ell: k = ell/chi, dk = dell/chi
# N_modes(ell)*dell = L^2 * ell/(chi * 2pi) * dell/chi = L^2 * ell * dell / (2 pi chi^2)

# The smooth assumption: C_true(ell) = P_2D(ell/chi)/chi^2 implies a "density" of modes.
# But the actual mode density is discrete.
# 
# Actually, wait. The Limber approximation says:
# C_l = integral dk_par ... which for us becomes a sum over kz modes.
# At k_par = 0, there's exactly ONE kz mode contributing (the k=0 mode).
# So C_l is NOT the integral over the LOS — it's just the kz=0 contribution.
#
# The "density" effect matters for the TRANSVERSE modes, which are summed in the 
# pair-counting formula. The smooth approximation replaces SUM_k with integral dk,
# which is fine when the mode spacing is much smaller than the scales of variation.
# mode spacing ~ 2*pi/L ~ 0.0045 h/Mpc, while P_lin varies on scale ~ 0.1 h/Mpc.
# So the smooth approximation should be good.

# Let me look at the actual cumulative contribution more carefully.
# The contribution from ell' = L to the theory at ell = l:
# M[l,L] * C_true[L] = (2L+1)/(4pi) * SUM_lam (2lam+1) wl[lam] 3j^2 * b1^2*P(L/chi)/(X*chi^2)

# For the white-noise part (wl = W_floor):
# (2L+1)/(4pi) * W_floor * 1 * b1^2*P/(X*chi^2)
# = (2L+1) * W_floor * b1^2 * P(L/chi) / (4*pi * X * chi^2)

# Let me compute this properly:
b1 = GRF_tmp.my_bias
W_floor = N**2 * 9600 / (4*np.pi)  # N^2 * Nskew / (4*pi)

print(f"\nW_floor = {W_floor:.4e}")
print(f"\nPer-L-bin contribution (2L+1)*W_floor*b1^2*P(L/chi)/(4pi*chi^2):")
total = 0
for ell_p in range(1, 10001):
    k = ell_p / chi_bar
    P = plin(k)
    contrib = (2*ell_p+1) * W_floor * b1**2 * P / (4*np.pi * chi_bar**2)
    total += contrib
    if ell_p in [500, 1000, 2000, 3000, 5000, 7000, 10000]:
        print(f"  L={ell_p}: per-bin={contrib:.4e}, cumsum={total:.4e}")

print(f"\nTotal up to L=10000: {total:.4e}")
# The measured cl at, say, ell=100: ~7.4e4
# The off-diagonal sum (from the white noise floor) should be ADDED to the diagonal peak.
# This tells us how much the theory grows from the white-noise tail.
print(f"cl_mean[100] = {np.mean(np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')['cl_k'], axis=0)[100]:.4e}")
