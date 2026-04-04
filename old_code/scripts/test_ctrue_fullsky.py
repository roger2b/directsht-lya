#!/usr/bin/env python
"""
Compute C_true NUMERICALLY from the discrete 2D power spectrum on the box grid,
without any Limber approximation. Compare with the analytic C_true formula.

The key insight is that w_j is defined at DISCRETE angular positions on the sky,
and the angular power of this field can be computed exactly from the 2D modes.

Strategy:
1. Generate one realization with ALL N^2 sightlines (no mask).
2. Compute w_j = SUM_z density(x_j, y_j, z) for all j.
3. Run DirectSHT on these w_j to get FULL-SKY pseudo-Cl.
4. Compare with the analytic MASTER prediction.

If data matches theory with ALL sightlines, then the C_true formula is correct
and the window function is causing the discrepancy.
If data > theory (or vice versa) even with ALL sightlines, C_true is wrong.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import SHT_lya as sht_lya
from sht.sht import DirectSHT
from sht.lya_sfb import _alm2cl_complex
import fast_Wigner3j as Wigner3j

N = 512
L = 1380.0
Nl = 200  # smaller for speed
chi_shift = 5000
chi_bar = chi_shift + L / 2.0

print(f"N={N}, L={L}, Nl={Nl}, chi_bar={chi_bar}")

# ---- Step 1: Generate density and compute w_j for ALL sightlines ----
print("\n---- Step 1: Generate density field ----")
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=42, verbose=False)
dens = GRF.dens  # (N, N, N)
plin = GRF.plin
b1 = GRF.my_bias

# w_j for all N^2 sightlines
w_all = np.sum(dens, axis=2)  # (N, N) — sum along z-axis
print(f"w_all: shape={w_all.shape}, std={np.std(w_all):.4e}")

# ---- Step 2: Compute angular positions for ALL sightlines ----
print("\n---- Step 2: Compute angular positions ----")
coords = np.meshgrid(*[np.linspace(0, L, N) for _ in range(3)])
# All sightlines: ix, iy ∈ 0..N-1
ix_all, iy_all = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
ix_flat = ix_all.ravel()
iy_flat = iy_all.ravel()

# 3D position of the START of each sightline (at z=chi_shift)
x_start = coords[0][ix_flat, iy_flat, 0]
y_start = coords[1][ix_flat, iy_flat, 0]
z_start = coords[2][ix_flat, iy_flat, 0] + chi_shift

# After the coordinate swap in process_skewers, x=tmp_z, y=tmp_y, z=tmp_x.
# So the actual coordinates used for (theta, phi) are:
# all_x = z + shift, all_y = y, all_z = x
# The skewer start is (all_x[:,0], all_y[:,0], all_z[:,0]).
# theta_phi uses (x,y,z) -> (theta,phi) via standard spherical coords:
# theta = arccos(z/r), phi = arctan2(y,x)
# With swap: r = sqrt(x_start^2 + y_start^2 + z_start^2)

# Let me reproduce the swap exactly:
tmp_x = coords[0][ix_flat, iy_flat, 0]  # original x-coord
tmp_y = coords[1][ix_flat, iy_flat, 0]  # original y-coord
tmp_z = coords[2][ix_flat, iy_flat, 0] + chi_shift  # z + shift

# Swap: all_x = tmp_z, all_y = tmp_y, all_z = tmp_x
all_x = tmp_z  # this is the LOS direction (radial ~ chi)
all_y = tmp_y
all_z = tmp_x

theta, phi = GRF.compute_theta_phi_skewer_start(all_x, all_y, all_z)
print(f"theta: min={np.min(theta):.4f}, max={np.max(theta):.4f}")
print(f"phi: min={np.min(phi):.4f}, max={np.max(phi):.4f}")

w_flat = w_all.ravel()
print(f"N_sightlines = {len(w_flat)}")

# ---- Step 3: Run DirectSHT ----
print("\n---- Step 3: DirectSHT ----")
sht_engine = DirectSHT(Nl, 2*Nl, 0.75)
t1 = time.time()
alm = sht_engine(theta, phi, w_flat)
print(f"SHT done in {time.time()-t1:.1f}s")

cl_full = _alm2cl_complex(alm, Nl)

# Also compute the window function: w_j = 1 for all sightlines
alm_ones = sht_engine(theta, phi, np.ones(len(w_flat)))
wl_full_sky = _alm2cl_complex(alm_ones, Nl)

print(f"cl_full[0:5] = {cl_full[:5]}")
print(f"wl_full[0:5] = {wl_full_sky[:5]}")
print(f"wl_full[0] / N^4 = {wl_full_sky[0] / N**4:.6f}")

# ---- Step 4: Compare ----
print("\n---- Step 4: Theory prediction ----")
ells = np.arange(Nl, dtype=float)

# Formula from notes: C_true = b1^2 * Plin(l/chi) / (32*pi^3*chi^2)
cl_true_notes = b1**2 * plin((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)

# My derivation: C_true = N^2 * b1^2 * Plin(l/chi) / (2*pi*L*chi^2)  
cl_true_mine = N**2 * b1**2 * plin((ells + 0.5) / chi_bar) / (2 * np.pi * L * chi_bar**2)

print(f"C_true (notes) [ell=10] = {cl_true_notes[10]:.6e}")
print(f"C_true (mine)  [ell=10] = {cl_true_mine[10]:.6e}")
print(f"Ratio mine/notes = {cl_true_mine[10] / cl_true_notes[10]:.2f}")

# For the full-sky (all N^2 sightlines), the MASTER formula still applies:
# <pseudo_Cl> = SUM_{l'} M[l,l'] C_true[l']
# where M is built from wl_full_sky (not the subset wl).

# But if ALL sightlines are used and they tile the full box face,
# the window is nearly uniform, and M ≈ delta_ll' (up to boundary effects).

# Simple check: cl_full should be close to C_true * wl_full_sky[0] / (4pi)?
# No — for a uniform window, M[l,l'] ~ delta_ll' * (N^2)^2/(4pi) * ... 

# Actually, let me just compare directly.
# The pseudo-Cl from ALL sightlines (no coupling needed if window is uniform):
# If the sightlines perfectly tile the sphere, cl_full = C_true directly.
# But they don't tile the sphere — they tile a small patch of the sky.
# So we still need the coupling matrix.

# Let me use MASTER with the full-sky window wl:
print("\n---- Step 5: MASTER with full-sky window ----")
wl_for_couple = np.zeros(2*Nl - 1)
n_copy = min(2*Nl-1, Nl)
wl_for_couple[:n_copy] = wl_full_sky[:n_copy]

couple = Wigner3j.CoupleMat(Nl, wl_for_couple)
M = couple.compute_matrix()

cl_theory_notes = M @ cl_true_notes
cl_theory_mine = M @ cl_true_mine

# Compare at ell=10..Nl-10 to avoid edge effects
ell_range = slice(10, Nl-10)
ratio_notes = np.mean(cl_full[ell_range]) / np.mean(cl_theory_notes[ell_range])
ratio_mine = np.mean(cl_full[ell_range]) / np.mean(cl_theory_mine[ell_range])

print(f"\nRatio (data/theory_notes) = {ratio_notes:.4f}")
print(f"Ratio (data/theory_mine)  = {ratio_mine:.4f}")

# Print in bins for more detail
print("\nBinned comparison:")
for ell_lo in range(10, Nl-10, 20):
    ell_hi = min(ell_lo + 20, Nl-10)
    d = np.mean(cl_full[ell_lo:ell_hi])
    tn = np.mean(cl_theory_notes[ell_lo:ell_hi])
    tm = np.mean(cl_theory_mine[ell_lo:ell_hi])
    print(f"  ell={ell_lo:3d}-{ell_hi:3d}: data/notes={d/tn:.4f}  data/mine={d/tm:.4f}")

# ---- Step 6: Also try without coupling (in case window is near-uniform) ----
# If w_j is very uniform (all N^2 sightlines from a patch), then PLKjKk should
# be very peaked, and the coupling should be dominated by the diagonal.
print(f"\n---- Step 6: Coupling matrix diagnostics ----")
print(f"M diagonal dominance: M[50,50]/sum(M[50,:]) = {M[50,50]/np.sum(M[50,:]):.4f}")
print(f"wl_full[0:10] = {wl_full_sky[:10]}")
print(f"wl_full ratios: wl[1]/wl[0]={wl_full_sky[1]/wl_full_sky[0]:.4f}, wl[2]/wl[0]={wl_full_sky[2]/wl_full_sky[0]:.4f}")

del GRF, dens; gc.collect()
