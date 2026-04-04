#!/usr/bin/env python
"""
Trace the EXACT normalization chain from GRF amplitudes -> density -> w_j -> C_l.

Key findings from reading GRF_class.py:
  amplitudes(k) = b1 * a(k)  where <|a(k)|^2> = P_lin(|k|)  [NOT P*L^3!]
  density(r) = ifftn(amplitudes) * L^{3/2} / (L/N)^3 = ifftn(amplitudes) * N^3/L^{3/2}

So density(r) = (N^3/L^{3/2}) * (1/N^3) * SUM_k amp(k) exp(2pi*i*k.n/N)
             = (1/L^{3/2}) * SUM_k amp(k) exp(ik.r)

And w_j = SUM_alpha density(x_j, y_j, z_alpha) 
       = (1/L^{3/2}) * SUM_{k_perp} amp(k_perp,0) * N * exp(ik_perp.r_j)
       ... wait, need to be careful with the kz=0 selection.

Let me just compute it numerically and compare.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF

# Create a GRF
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=42, verbose=False)
N = GRF.N
L = GRF.L
b1 = GRF.my_bias
plin = GRF.plin
k_f = 2 * np.pi / L

print(f"N={N}, L={L:.1f}, b1={b1:.4f}, k_f={k_f:.6f}")
print(f"boxvol^(1/2) = {L**(3/2):.1f}")
print(f"pix = (L/N)^3 = {(L/N)**3:.6f}")
print(f"N^3/L^(3/2) = {N**3/L**(3/2):.4f}")

# ---- Check: amplitudes normalization ----
amps = GRF.amplitudes  # = b1 * a(k)
print(f"\nAmplitudes: shape={amps.shape}, dtype={amps.dtype}")
print(f"  mean(|amp|^2) = {np.mean(np.abs(amps)**2):.6e}")

# Expected: <|amp(k)|^2> = b1^2 * P_lin(|k|) where k is in physical units
kfft = GRF.kfft
KX, KY, KZ = np.meshgrid(kfft, kfft, kfft, indexing='ij')
K_3d = np.sqrt(KX**2 + KY**2 + KZ**2)
Pk_grid = plin(K_3d.ravel()).reshape(K_3d.shape)
print(f"  mean(b1^2 * Plin(k)) = {b1**2 * np.mean(Pk_grid):.6e}")
print(f"  ratio = {np.mean(np.abs(amps)**2) / (b1**2 * np.mean(Pk_grid)):.4f}")

# ---- Check: density normalization ----
dens = GRF.dens
print(f"\nDensity: mean={np.mean(dens):.4e}, std={np.std(dens):.4e}")

# Manual: dens = real(ifftn(amp)) * L^{3/2} / (L/N)^3
pix = (L / N) ** 3
boxvol = L ** 3
dens_check = np.real(np.fft.ifftn(amps)) * boxvol**(0.5) / pix
print(f"Manual density: max|diff| = {np.max(np.abs(dens - dens_check)):.4e}")

# ---- Variance of density ----
# <delta(r)^2> = (N^3/L^{3/2})^2 * (1/N^3)^2 * SUM_k <|amp(k)|^2>
#              = (1/L^3) * SUM_k b1^2 * Plin(k)
#              = (b1^2/L^3) * N^3 * <Plin>   [N^3 modes in total]
# Let's check:
var_expected = (b1**2 / L**3) * N**3 * np.mean(Pk_grid)
print(f"\n<delta^2> expected = {var_expected:.6e}")
print(f"<delta^2> measured = {np.var(dens):.6e}")
print(f"Ratio: {np.var(dens) / var_expected:.4f}")

# ---- w_j = SUM_z density(x,y,z) ----
# For ALL sightlines on the grid:
w_all = np.sum(dens, axis=2)  # shape (N, N)
print(f"\nw_all: shape={w_all.shape}, mean={np.mean(w_all):.4e}, std={np.std(w_all):.4e}")

# Theoretical: 
# w_j = SUM_alpha density(x_j, y_j, z_alpha)
# density = (1/L^{3/2}) * SUM_k amp(k) * exp(ik.r) 
# Hmm wait, that's not right. Let me redo:
# density = ifftn(amp) * L^{3/2} / (L/N)^3
# = (1/N^3) * SUM_k amp(k) exp(2pi*i*k.n/N) * L^{3/2} / (L/N)^3
# = (1/N^3) * SUM_k amp(k) exp(2pi*i*k.n/N) * N^3 / L^{3/2}
# = (1/L^{3/2}) * SUM_k amp(k) exp(2pi*i*k.n/N)

# SUM_nz density(nx,ny,nz) = (1/L^{3/2}) * SUM_{kx,ky} amp(kx,ky,0) * N * exp(2pi*i*(kx*nx+ky*ny)/N)
# = (N/L^{3/2}) * [2D IFFT of amp(:,:,0) * N^2]
# Actually: 2D IFFT in numpy = (1/N^2) SUM exp(...)
# So SUM_nz density = (N/L^{3/2}) * N^2 * ifft2(amp(:,:,0))
# = N^3/L^{3/2} * ifft2(amp(:,:,0))

w_check = (N**3 / L**(1.5)) * np.real(np.fft.ifft2(amps[:, :, 0]))
print(f"Check: max|w_all - formula| = {np.max(np.abs(w_all - w_check)):.4e}")

# ---- Variance of w_j ----
# <w_j^2> = (N^3/L^{3/2})^2 * (1/N^4) * SUM_{k_perp} <|amp(k_perp,0)|^2>
# = (N^6/L^3) * (1/N^4) * SUM_{k_perp} b1^2 * Plin(k_perp)
# = (N^2 * b1^2 / L^3) * SUM_{k_perp} Plin(k_perp)

# k_perp sum: there are N^2 modes in the kz=0 plane
k_perp_mag = np.sqrt(KX[:, :, 0]**2 + KY[:, :, 0]**2)
Pk_2d = plin(k_perp_mag.ravel()).reshape(k_perp_mag.shape)
sum_Pk_2d = np.sum(Pk_2d)

var_w_expected = (N**2 * b1**2 / L**3) * sum_Pk_2d
print(f"\n<w^2> expected = {var_w_expected:.6e}")
print(f"<w^2> measured = {np.var(w_all):.6e}")
print(f"Ratio: {np.var(w_all) / var_w_expected:.4f}")

# ---- Now: the angular pseudo-Cl ----
# The pseudo-Cl from the SHT is:
# chat_Cl = (1/(2l+1)) SUM_m |a_lm|^2
# where a_lm = SUM_j w_j * Y_lm^*(hat{n}_j) / (4pi)
# Wait, what normalization does the DirectSHT use?
# From the DirectSHT code: a_lm = SUM weights * Y_lm^* * w_j
# And a_lm is in HEALPix convention.
# The data is computed as cl_k = hp.alm2cl(hdat) where hdat is the DirectSHT output.

# For a FULL-SKY survey:
# a_lm = integral w(n_hat) Y_lm^*(n_hat) dOmega
# hat{C}_l = (1/(2l+1)) SUM_m |a_lm|^2

# For a discrete set of sightlines with unit weights:
# a_lm = SUM_j w_j * Y_lm^*(n_hat_j) * Delta_Omega_j
# where Delta_Omega_j is the effective solid angle per sightline.
# But if the DirectSHT just sums without any solid angle weight, then:
# a_lm = SUM_j w_j * Y_lm^*(n_hat_j)

# This is the "pseudo-alm" with no pixelization correction.
# <hat{C}_l> = SUM_{j,k} <w_j w_k> Y_lm(n_j) Y_lm^*(n_k) / (2l+1)...

# The key connection to the coupling matrix approach:
# <hat{C}_l> = SUM_{l'} M[l,l'] * C_true[l']
# where C_true[l'] is the angular power spectrum of the field w(n).

# What IS C_true for the field w(n)?
# w at a given direction n_hat: only defined at the Nskew sightline positions.
# But in the MASTER framework, we treat w(n) as:
# w(n) = W(n) * s(n) + noise
# where W is the window (survey mask) and s(n) is the underlying continuous field.

# The continuous field s(n_hat) at angular position n_hat that passes through
# position r_perp on the box face (at distance chi_bar) is:
# s(n_hat) = SUM_alpha density(r_perp(n_hat), z_alpha)

# This is defined EVERYWHERE on the box face (for all N^2 positions),
# not just at the Nskew positions.

# So C_true[l] is the angular Cl of the field s(n_hat) = SUM_alpha density(r_perp, z_alpha)
# evaluated at all angular positions corresponding to the box face.

# <s(n1) s(n2)> = <w(r1) w(r2)> = (N^2*b1^2/L^3) * SUM_{k_perp} Plin(k_perp) exp(ik_perp.Delta_r)

# Now converting to a continuous formula:
# SUM_{k_perp} -> (L/(2pi))^2 * integral d^2k
# <s(n1) s(n2)> = (N^2*b1^2/L^3) * (L/(2pi))^2 * integral Plin(k_perp) exp(ik.Dr) d^2k
# = (N^2*b1^2) / (4*pi^2*L) * integral Plin(k) exp(ik.Dr) d^2k

# Using the plane-wave expansion and the Limber approximation:
# integral Plin(k) exp(ik.Dr) d^2k = integral dk k * integral dtheta Plin(k) exp(ik*Dr*cos(theta))
# = 2*pi * integral dk k Plin(k) J0(k*Dr)
# And using the Limber approx: Dr = chi_bar * theta_12
# C_true[l] = integral <s(n1) s(n2)> Pl(cos_12) dOmega_2 / (4pi)

# Actually, for the flat-sky approximation (which is appropriate for a small box):
# C_true[l] = (2pi) * integral theta * <s(theta)s(0)> * J0(l*theta) dtheta
# = (2pi) * (N^2*b1^2)/(4pi^2*L) * integral dtheta theta * [2pi integral dk k Plin(k) J0(k*chi*theta)] * J0(l*theta)
# = (N^2*b1^2)/(2pi*L) * [integral dk k Plin(k)] * [integral dtheta theta J0(k*chi*theta) J0(l*theta)]
# = (N^2*b1^2)/(2pi*L) * integral dk k Plin(k) * delta(k*chi - l) / (l)  [orthogonality]
# = (N^2*b1^2)/(2pi*L*chi^2) * Plin(l/chi)

# So C_true[l] = N^2 * b1^2 * Plin(l/chi) / (2*pi*L*chi^2)

# But the notes say C_true = b1^2 * P_F(l/chi) / (32*pi^3*chi^2)
# where P_F = b1^2 * Plin [for no alias, no RSD]
# So C_true_notes = b1^4 * Plin / (32*pi^3*chi^2)   # Wait, that has b1^4?!
# Hmm, the notes have b1 already inside P_F: P_F = b1^2 * Plin
# So C_true = b1^2 * Plin / (32*pi^3*chi^2)

# My derivation: C_true = N^2 * b1^2 * Plin / (2*pi*L*chi^2)

# The ratio: (N^2) / (2*pi*L) vs 1/(32*pi^3)
# = (N^2 * 32*pi^3) / (2*pi*L) = N^2 * 16*pi^2 / L

# With N=512, L=1382.7:
print(f"\n---- Normalization comparison ----")
print(f"N^2 / (2*pi*L) = {N**2 / (2*np.pi*L):.4f}")
print(f"1 / (32*pi^3) = {1 / (32*np.pi**3):.6e}")
print(f"Ratio: N^2/(2*pi*L) / [1/(32*pi^3)] = {(N**2 / (2*np.pi*L)) / (1/(32*np.pi**3)):.4e}")
print(f"= N^2 * 16*pi^2 / L = {N**2 * 16*np.pi**2 / L:.4e}")

# Hmm, these are VERY different. The notes formula gives C_true ~ 1e-6 level.
# My derivation gives C_true ~ 30 * Plin/chi^2.

# But wait -- I need to be more careful about what w_j IS in the simulation.
# The SHT code takes the DFT of the skewers at a specific k_parallel mode.
# Not just the sum! Let me re-check.

# In the cached data: cl_k is the pseudo-Cl at some specific k_parallel.
# From the lya_GRFs_directSHT_loop.py code:
# FT_delta = sht_lya.compute_dft(chi_grid, mask_ones, delta_skewers[j], ...) 
# This computes the DFT along the LOS and picks out specific k-modes.
# Then alm = SHT(FT_delta_at_k) for each k-mode.

# So the field is NOT the SUM of density along LOS, but the DFT at a specific k_par!
# Let me check what compute_dft does and which k_par mode is stored.

print("\n---- This changes everything! ----")
print("The field is a specific k_par DFT mode, not the sum!")
print("Need to check compute_dft in SHT_lya.py and the k-mode selection.")
