#!/usr/bin/env python
"""
Direct numerical computation of C_true from the density field correlation.

The measured pseudo-Cl is: 
  <hat{C}_l> = SUM_{l'} M[l,l'] C_true[l']

I want to compute C_true[l] numerically by computing the angular correlation
function of the CONTINUOUS field w(n_hat) = SUM_z density(r_perp(n_hat), z)
and comparing with the Limber formula.

Strategy:
1. Generate ONE realization.  
2. Compute w(n_hat) on a dense angular grid (using the FULL N^2 transverse grid).
3. Compute the angular power spectrum of this w-field using a HEALPix map.
4. Average over several realizations.
5. This gives C_true[l] numerically, which we compare with the analytic formula.

Actually, the full N^2 grid gives w on a SQUARE patch, not the full sky.
So I can't use HEALPix directly. But I CAN compute the pseudo-Cl from ALL N^2 
sightlines and then DECONVOLVE the coupling matrix (from ALL sightlines' window).

Let me instead do something simpler:
- Compute the 2D power spectrum of w_all (on the grid) using FFT.
- Convert to C_true using the flat-sky relation: C_l ≈ P_2D(l/chi) / chi^2.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF

N = 512
chi_shift = 5000

print("Computing 2D power spectrum of the LOS-summed density field...")

Pk2d_all = []
for seed in range(42, 52):
    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=seed, verbose=False)
    L = GRF.L
    b1 = GRF.my_bias
    plin = GRF.plin
    dens = GRF.dens
    
    # w_all = SUM_z density on the N^2 grid
    w_all = np.sum(dens, axis=2)  # (N, N)
    
    # 2D power spectrum via FFT
    # w_FT(k_perp) = FT(w_all) using numpy's fft2
    # P_2D(k) = |w_FT(k)|^2 / (N^2)  ... or some normalization
    w_ft = np.fft.fft2(w_all)
    # In numpy's convention: w_ft[nx, ny] = SUM_{ix,iy} w[ix,iy] exp(-2pi*i*(nx*ix+ny*iy)/N)
    # And the 2D power spectrum: P(k) = <|w_ft|^2> / N^2^2 * L^2  [for proper continuum normalization]
    # Actually, let me think about this differently.
    
    # w[ix,iy] is defined on a grid of N×N.
    # The continuous field on the box face: w(r_perp) = w[ix,iy] where r_perp = (ix,iy)*dx, dx=L/N
    # The Fourier transform: w_cont(k) = ∫ w(r) exp(-ik.r) d^2r ≈ dx^2 * SUM w[ix,iy] exp(-ik.r[ix,iy])
    # = dx^2 * w_ft[n] where k = (2pi/L)*n
    # So w_cont(k) = (L/N)^2 * w_ft[n]
    # P_2D(k) = <|w_cont(k)|^2> / (L^2)  [dividing by area to get power per unit area]
    # = (L/N)^4 * <|w_ft|^2> / L^2
    # = (L/N)^4 / L^2 * |w_ft|^2
    # = L^2/N^4 * |w_ft|^2
    
    Pk_2d = L**2 / N**4 * np.abs(w_ft)**2
    Pk2d_all.append(Pk_2d)
    del GRF, dens; gc.collect()

Pk2d_mean = np.mean(Pk2d_all, axis=0)
chi_bar = chi_shift + L/2

print(f"N={N}, L={L:.4f}, b1={b1:.4f}, chi_bar={chi_bar:.1f}")
print(f"Mean 2D power computed from {len(Pk2d_all)} realizations")

# Bin P_2D(k_perp) in annuli
k_f = 2*np.pi/L
kfft = np.fft.fftfreq(N) * 2*np.pi * N / L
KX, KY = np.meshgrid(kfft, kfft, indexing='ij')
k_perp = np.sqrt(KX**2 + KY**2)

# Theory: what should P_2D(k_perp) be?
# w(r_perp) = SUM_z density(r_perp, z) [sum of N values along LOS]
# density = real(ifftn(amp)) * L^{3/2} / (L/N)^3
# w = N * density_at_kz0 (in Fourier) → see earlier derivation
# w_cont(k_perp) = integral w(r) exp(-ik.r) d^2r
# Since w is on a discrete grid with spacing dx = L/N:
# w_cont(k_perp) ≈ dx^2 * SUM_{ix,iy} w[ix,iy] exp(-ik.r)

# Using the identity: 
# w(r) = (b1/L^{3/2}) * SUM_{k'} a(k') N * delta(k'_z,0) exp(ik'.r)
# Let me evaluate w_cont(k_perp):
# w_cont(k) = integral w(r) e^{-ik.r} d^2r
# For a finite box of size L×L:
# = L^2 * delta_{k, k_perp} * (b1*N/L^{3/2}) * a(k_perp, 0) / N^2
# Wait, this is discrete. The FT of a periodic function:
# w_cont(k_{nx,ny}) = L^2 * c_{nx,ny} where c are the Fourier coefficients.

# The coefficient c_{nx,ny} = (1/L^2) * integral w(r) e^{-ik.r} d^2r / (1)?
# With numpy convention: w_ft[nx,ny] = SUM w[ix,iy] exp(-2pi*i*(nx*ix+ny*iy)/N)
# And k = 2pi*(nx,ny)/L.
# So w_cont(k) = (L/N)^2 * w_ft[n]

# P_2D(k) = <|w_cont(k)|^2> / L^2 = (L/N)^4 |w_ft|^2 / L^2 = (L^2/N^4) |w_ft|^2

# From the known amplitudes:
# w(r_perp) = (b1*N/L^{3/2}) * SUM_{k'_perp} a(k'_perp,0) * exp(ik'.r_perp)
# This is a Fourier series with period L.
# The Fourier coefficient at k = (2pi/L)(nx,ny) is:
# w_hat[nx,ny] = (b1*N/L^{3/2}) * a(nx,ny,0)
# And w_cont(k) = L^2 * w_hat[n] / L^2 = w_hat[n]? 
# No: w(r) = SUM_n w_hat[n] exp(ik_n.r), so FT: w_cont(k_n) = L^2 * w_hat[n].
# Wait, for a periodic function: w(r) = SUM_n c_n exp(ik_n.r)
# FT: integral w(r) exp(-ik_m.r) d^2r = L^2 * c_m
# So w_cont(k) = L^2 * c(k) where c(k) = (b1*N/L^{3/2}) * a(k_perp,0)

# Therefore: P_2D(k) = <|w_cont(k)|^2> / L^2 = L^4 * (b1^2*N^2/L^3) * P_lin(|k|) / L^2
# = b1^2 * N^2 * L^{4-3-2} * P_lin(|k|) [wait]
# = L^2 * b1^2 * N^2 / L^3 * P_lin(|k|)
# = b1^2 * N^2 * P_lin(|k|) / L

# But I defined P_2D = (L^2/N^4) * |w_ft|^2, and |w_ft|^2 = N^4/L^4 * |w_cont|^2:
# (actually w_ft = (N/L)^2 * w_cont in the discrete→continuous conversion)
# So |w_ft|^2 = (N/L)^4 * |w_cont|^2
# P_2D = (L^2/N^4) * (N/L)^4 * |w_cont|^2 = |w_cont|^2/L^2
# And |w_cont|^2 = L^4 * b1^2*N^2/L^3 * P_lin = b1^2*N^2*L * P_lin
# So P_2D = b1^2*N^2*L/L^2 * P_lin = b1^2*N^2*P_lin/L

# Let me verify:
Pk_theory = b1**2 * N**2 * plin(k_perp.ravel()).reshape(k_perp.shape) / L
Pk_ratio = Pk2d_mean / Pk_theory

# Bin and compare
k_edges = np.linspace(k_f, 1.0, 40)
print(f"\nBinned P_2D comparison:")
print(f"{'k_mid':>8} {'P_meas':>12} {'P_theory':>12} {'ratio':>8}")
for i in range(len(k_edges)-1):
    m = (k_perp >= k_edges[i]) & (k_perp < k_edges[i+1])
    if np.sum(m) > 5:
        pm = np.mean(Pk2d_mean[m])
        pt = np.mean(Pk_theory[m])
        r = pm/pt
        km = (k_edges[i]+k_edges[i+1])/2
        if i < 5 or i % 5 == 0:
            print(f"{km:8.4f} {pm:12.4e} {pt:12.4e} {r:8.4f}")

# Overall ratio
m_all = k_perp > k_f  # exclude k=0
print(f"\nOverall ratio P_meas/P_theory: {np.mean(Pk2d_mean[m_all]) / np.mean(Pk_theory[m_all]):.6f}")

# Now: the ANGULAR power spectrum C_l = P_2D(l/chi) / chi^2 (flat-sky Limber)
# = b1^2 * N^2 * P_lin(l/chi) / (L * chi^2)
# Compare with notes: C_true = b1^2 * P_lin(l/chi) / (32*pi^3 * chi^2)
# Ratio: N^2 / L vs 1/(32*pi^3)
# N^2/L = 512^2/1380 = 189.9
# 1/(32*pi^3) = 1.008e-3
# Ratio = 189.9 / 1.008e-3 = 1.88e5

# This is WAY off. So my derivation of P_2D is wrong, or the flat-sky relation is wrong,
# or the C_true in the notes is correct and the flat-sky formula needs extra factors.

# Actually: the flat-sky C_l = P_2D(l/chi) / chi^2 is for a CONTINUOUS field on the full sky.
# But here, w(r_perp) lives on a PATCH of size L×L at distance chi_bar.
# The solid angle subtended is Omega_box = L^2/chi_bar^2.
# For a partial-sky power spectrum: C_l^{full-sky} = P_2D / (chi^2 * f_sky)?
# No, that doesn't make sense either. C_l doesn't scale with f_sky.

# The issue is that P_2D as I computed it is the power per unit AREA on the box face,
# in [(Mpc/h)^3] units (since w is dimensionless but summed over N pixels).
# Wait, w_j = SUM_z delta(x_j, y_j, z) is dimensionless (delta is dimensionless).
# So P_2D has units of [(Mpc/h)^2] (area from the Fourier transform).

# Actually, w is just a number (sum of dimensionless delta). P_2D(k) has units of area.
# The angular Cl for a field on a patch:
# We define: s(nhat) = w(r_perp(nhat)) for directions nhat pointing to the box face.
# a_lm^s = integral s(nhat) Y_lm^*(nhat) dOmega
# For small angles: dOmega ≈ d^2theta, and r_perp = chi * theta:
# a_lm^s ≈ integral w(chi*theta) Y_lm^*(theta) d^2theta
# In flat-sky: Y_lm → exp(il.theta)/sqrt(A_sky) and d^2theta = d^2r_perp/chi^2:
# a_l^s ≈ (1/chi^2) * w_cont(l/chi)

# Then: C_l = <|a_l^s|^2> = (1/chi^4) * <|w_cont(l/chi)|^2>
# And P_2D(k) = <|w_cont(k)|^2> / L^2:
# So <|w_cont(k)|^2> = P_2D(k) * L^2
# C_l = P_2D(l/chi) * L^2 / chi^4

# Hmm, but that gives C_l with units of area^2/distance^4 which is dimensionless. Good.
# C_l = P_2D(l/chi) * L^2 / chi^4
# = [b1^2 * N^2 * P_lin(l/chi) / L] * L^2 / chi^4
# = b1^2 * N^2 * P_lin(l/chi) * L / chi^4

# With the notes: C_true = b1^2 * P_lin / (32*pi^3 * chi^2)
# Ratio: (N^2 * L / chi^4) / (1/(32*pi^3*chi^2)) = 32*pi^3 * N^2 * L / chi^2

# That makes it even worse. Something is seriously wrong with my P_2D derivation.

# Let me check empirically: P_2D(k) from the FFT measurements.
# Then compare with the expected C_l from MASTER.

print(f"\n---- Cross-check ----")
print(f"P_2D at k=0.1: {np.mean(Pk2d_mean[(k_perp>0.09)&(k_perp<0.11)]):.4e}")
print(f"b1^2 * N^2 * P_lin(0.1) / L = {b1**2 * N**2 * plin(0.1) / L:.4e}")
print(f"b1^2 * P_lin(0.1) = {b1**2 * plin(0.1):.4e}")
print(f"b1^2 * P_lin(0.1) * L = {b1**2 * plin(0.1) * L:.4e}")
print(f"ell at k=0.1: {0.1 * chi_bar:.0f}")
