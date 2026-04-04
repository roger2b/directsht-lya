#!/usr/bin/env python -u
"""
Full-sampling diagnostic: use ALL N^2 sightlines from a small box.

With full transverse sampling (every grid cell → a sightline):
- Angular window ≈ full sky → mode coupling M_ll ≈ diagonal
- pseudo-Cl ≈ C_true directly — no window complications
- Shot noise role becomes unambiguous

This isolates the normalization question from the window function question.
"""
import sys, os, gc
import numpy as np
import healpy as hp

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

from sht.sht import DirectSHT
from sht.theory_lya import theory_cl_k, compute_chi_bar_from_grid
import GRF_class as my_GRF

# ====================================================================== #
#  Settings: small box, full sampling                                     #
# ====================================================================== #
N_box    = 64        # small grid → N^2 = 4096 sightlines (all of them)
L_box    = 1380.0    # same physical size → same P(k) scales
Nl       = 50        # modest multipoles — enough to see the shape
chi_shift = 5000.0
num_sim  = 5
add_rsd  = False
seed0    = 2000      # different seed series from production

Nx   = 2 * Nl
xmax = 0.75
sht  = DirectSHT(Nl, Nx, xmax)
print(f"DirectSHT: Nl={Nl}, Nx={Nx}, xmax={xmax}")

# ====================================================================== #
#  Helper: extract ALL sightlines from a GRF                              #
# ====================================================================== #
def extract_all_sightlines(GRF, shift):
    """Extract ALL N^2 sightlines — full transverse sampling."""
    N = GRF.N
    L = GRF.L
    coords = np.meshgrid(*[np.linspace(0, L, N) for _ in range(3)])

    # ALL transverse indices (no random subsampling)
    inds_y, inds_x = np.meshgrid(np.arange(N), np.arange(N))
    inds = np.column_stack([inds_x.ravel(), inds_y.ravel()])
    Nskew = len(inds)  # = N^2

    # Density skewers
    dens_lya = GRF.dens[inds[:, 0], inds[:, 1], :] + 1.0  # 1 + delta

    # Coordinates (swap: x→LOS, z→transverse, as in original code)
    tmp_x = coords[0][inds[:, 0], inds[:, 1], :]
    tmp_y = coords[1][inds[:, 0], inds[:, 1], :]
    tmp_z = coords[2][inds[:, 0], inds[:, 1], :] + shift

    all_x = tmp_z.copy()  # LOS direction (shifted)
    all_y = tmp_y.copy()
    all_z = tmp_x.copy()  # transverse

    # delta_F = (1+delta) - 1 = delta
    all_w_gal = dens_lya - 1.0
    all_w_rand = np.ones_like(all_x, dtype=float)

    return all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew


# ====================================================================== #
#  Phase 1: Generate sims, measure pseudo-Cl at k=0                       #
# ====================================================================== #
print(f"\n{'='*60}")
print(f"FULL-SAMPLING TEST: N={N_box}, L={L_box}, Nl={Nl}")
print(f"Nskew = N^2 = {N_box**2}")
print(f"{'='*60}")

# We need chi_grid, theta, phi from first sim (same grid for all)
GRF0 = my_GRF.PowerSpectrumGenerator(N=N_box, L=L_box, add_rsd=add_rsd,
                                       seed=seed0, verbose=False)
allx, ally, allz, w_rand, w_gal, Nskew = extract_all_sightlines(GRF0, chi_shift)
chi_grid = allx[0, :]  # same for all sightlines
theta, phi = GRF0.compute_theta_phi_skewer_start(allx[:, 0], ally[:, 0], allz[:, 0])
dchi = chi_grid[1] - chi_grid[0]
N = chi_grid.size  # = N_box
chi_bar = compute_chi_bar_from_grid(chi_grid)

print(f"Nskew = {Nskew}, Npix = {N}, dchi = {dchi:.4f} Mpc/h")
print(f"chi_bar = {chi_bar:.2f} Mpc/h")
print(f"L_los = {N * dchi:.2f} Mpc/h")

# Compute DFT for first sim to understand the FT convention
import SHT_lya as sht_lya
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, w_rand, w_gal)
print(f"\nFT_mask[:3, 0] = {FT_mask[:3, 0]}  (should be {N})")
print(f"FT_mask[:3, 1] = {FT_mask[:3, 1]}  (should be ~0 for periodic)")
print(f"|FT_mask[:, 1]|_max = {np.max(np.abs(FT_mask[:, 1])):.2e}")

# SHT of data and randoms at k=0
hdat = sht(theta, phi, FT_delta[:, 0])
hran = sht(theta, phi, FT_mask[:, 0])
cl_data_0 = hp.alm2cl(hdat)[:Nl]
cl_rand_0 = hp.alm2cl(hran)[:Nl]

# Shot noise: sum of w_j^2 / (4pi)
w_j = FT_delta[:, 0]  # weights at k=0
SN = np.sum(w_j**2) / (4.0 * np.pi)
w_j_rand = FT_mask[:, 0]  # = N for all j
SN_rand = np.sum(w_j_rand**2) / (4.0 * np.pi)

print(f"\n--- Shot noise ---")
print(f"SN_data = Σ w_j² / (4π) = {SN:.4e}")
print(f"SN_rand = Σ (N)² / (4π) = N²×Nskew/(4π) = {N**2 * Nskew / (4*np.pi):.4e} = {SN_rand:.4e}")
print(f"cl_data[ℓ=Nl-1] = {cl_data_0[Nl-1]:.4e} (should be ≈ SN + signal)")
print(f"cl_rand[0]       = {cl_rand_0[0]:.4e}")
print(f"(N×Nskew)²/(4π)  = {(N*Nskew)**2/(4*np.pi):.4e}")

del GRF0; gc.collect()

# Now run multiple sims
print(f"\n--- Running {num_sim} simulations ---")
cl_stack = []
sn_stack = []
for isim in range(num_sim):
    seed = seed0 + isim
    GRF = my_GRF.PowerSpectrumGenerator(N=N_box, L=L_box, add_rsd=add_rsd,
                                          seed=seed, verbose=False)
    _, _, _, _, wg, _ = extract_all_sightlines(GRF, chi_shift)
    plin = GRF.plin
    b1 = GRF.my_bias
    beta = GRF.my_beta

    # FT at k=0
    _, _, ftd = sht_lya.compute_dft(chi_grid, np.ones_like(wg), wg)
    w = ftd[:, 0]  # real weights at k=0
    h = sht(theta, phi, w)
    cl = hp.alm2cl(h)[:Nl]
    sn = np.sum(w**2) / (4.0 * np.pi)

    cl_stack.append(cl)
    sn_stack.append(sn)
    print(f"  sim {isim}: cl[5]={cl[5]:.4e}, SN={sn:.4e}")
    del GRF, wg, ftd; gc.collect()

cl_stack = np.array(cl_stack)
sn_stack = np.array(sn_stack)
cl_mean = np.mean(cl_stack, axis=0)
cl_std = np.std(cl_stack, axis=0)
sn_mean = np.mean(sn_stack)

print(f"\nMean SN = {sn_mean:.4e}")
print(f"Mean cl[5] = {cl_mean[5]:.4e}")
print(f"Mean cl[{Nl-1}] = {cl_mean[Nl-1]:.4e}")

# ====================================================================== #
#  Phase 2: Theory predictions                                            #
# ====================================================================== #
print(f"\n{'='*60}")
print(f"THEORY COMPARISON")
print(f"{'='*60}")

ells = np.arange(Nl, dtype=float)

# Theory 1: raw Limber C_true = P_F / chi_bar^2
cl_limber = theory_cl_k(ells, 0.0, chi_bar, plin, b1=b1, beta=beta)
print(f"\nTheory (Limber): P_F(ℓ/χ̄, k=0) / χ̄²")
print(f"  cl_limber[5] = {cl_limber[5]:.4e}")

# Theory 2: the 32π³ version
cl_32pi3 = cl_limber / (32.0 * np.pi**3)
print(f"\nTheory (32π³): P_F / (32π³ χ̄²)")
print(f"  cl_32pi3[5] = {cl_32pi3[5]:.4e}")

# Theory 3: with mode coupling (should be ≈ no coupling for full sampling)
from sht.mask_deconvolution import MaskDeconvolution
MD = MaskDeconvolution(Nl, cl_rand_0)
Mll = MD.Mll
cl_convolved_limber = Mll @ cl_limber
cl_convolved_32pi3 = Mll @ cl_32pi3

# ====================================================================== #
#  Phase 3: Compare — what normalization is correct?                      #
# ====================================================================== #
print(f"\n{'='*60}")
print(f"RATIO ANALYSIS")
print(f"{'='*60}")

print(f"\n--- Raw ratios (no SN subtraction) ---")
print(f"{'ell':>4} {'cl_data':>12} {'Limber':>12} {'32π³':>12} "
      f"{'Mll@Limber':>12} {'Mll@32π³':>12} "
      f"{'r(Lim)':>10} {'r(32π³)':>10} {'r(M@L)':>10} {'r(M@32)':>10}")
for l in [0, 2, 5, 10, 20, 30, 40, 49]:
    if l < Nl:
        rL = cl_mean[l] / cl_limber[l] if cl_limber[l] > 0 else np.inf
        r32 = cl_mean[l] / cl_32pi3[l] if cl_32pi3[l] > 0 else np.inf
        rML = cl_mean[l] / cl_convolved_limber[l] if cl_convolved_limber[l] > 0 else np.inf
        rM32 = cl_mean[l] / cl_convolved_32pi3[l] if cl_convolved_32pi3[l] > 0 else np.inf
        print(f"{l:4d} {cl_mean[l]:12.4e} {cl_limber[l]:12.4e} {cl_32pi3[l]:12.4e} "
              f"{cl_convolved_limber[l]:12.4e} {cl_convolved_32pi3[l]:12.4e} "
              f"{rL:10.4f} {r32:10.4f} {rML:10.4f} {rM32:10.4f}")

print(f"\n--- SN-subtracted ratios ---")
print(f"{'ell':>4} {'cl-SN':>12} {'r(Lim)':>10} {'r(32π³)':>10} {'r(M@L)':>10} {'r(M@32)':>10}")
for l in [0, 2, 5, 10, 20, 30, 40, 49]:
    if l < Nl:
        cl_sub = cl_mean[l] - sn_mean
        rL = cl_sub / cl_limber[l] if cl_limber[l] > 0 else np.inf
        r32 = cl_sub / cl_32pi3[l] if cl_32pi3[l] > 0 else np.inf
        rML = cl_sub / cl_convolved_limber[l] if cl_convolved_limber[l] > 0 else np.inf
        rM32 = cl_sub / cl_convolved_32pi3[l] if cl_convolved_32pi3[l] > 0 else np.inf
        print(f"{l:4d} {cl_sub:12.4e} {rL:10.4f} {r32:10.4f} {rML:10.4f} {rM32:10.4f}")

# ====================================================================== #
#  Phase 4: What is the correct answer from first principles?             #
# ====================================================================== #
print(f"\n{'='*60}")
print(f"FIRST-PRINCIPLES DERIVATION")
print(f"{'='*60}")

# The pseudo-Cl estimator for data weights w_j = FT_delta[j, k=0]:
#
#   pseudo-Cl = (1/(2l+1)) Σ_m |a_lm|² 
#   where a_lm = Σ_j w_j Y*_lm(n_j)
#
# So: <pseudoCl> = (1/(2l+1)) Σ_m Σ_{j,k} <w_j w_k> Y*_lm(n_j) Y_lm(n_k)
#                = Σ_{j,k} <w_j w_k> P_l(cos θ_jk) / (4π)    [addition thm]
#
# Now w_j = Σ_n δ_F(j, n)  (DFT at k=0 = sum over LOS pixels)
# So <w_j w_k> = Σ_{n,n'} <δ_F(j,n) δ_F(k,n')>
#
# For a periodic GRF:
#   <δ_F(j,n) δ_F(k,n')> = (1/V) Σ_K P_F(K) e^{iK·(r_jn - r_kn')}
# where V = L^3 is the box volume.
#
# For full transverse sampling, the sum over (j,k) pairs at fixed angular
# separation θ samples the transverse plane uniformly. The LOS sum at k=0
# projects out the k_par=0 mode.
#
# Let me compute <w_j w_k> directly and check:

print("\nDirect <w_j w_k> check:")

# Generate one GRF and compute the empirical covariance
GRF_check = my_GRF.PowerSpectrumGenerator(N=N_box, L=L_box, add_rsd=add_rsd,
                                            seed=seed0, verbose=False)
_, _, _, _, wg_check, _ = extract_all_sightlines(GRF_check, chi_shift)
_, _, ftd_check = sht_lya.compute_dft(chi_grid, np.ones_like(wg_check), wg_check)
w_check = ftd_check[:, 0]  # (Nskew,) real weights at k=0

# Empirical <w_j w_k> along diagonal and off-diagonal
print(f"  <w_j²> (diagonal mean) = {np.mean(w_check**2):.4e}")
print(f"  <w_j>  (mean weight)   = {np.mean(w_check):.4e}")

# Compare diagonal mean to SN:
# SN = Σ w_j² / (4π) = Nskew × <w_j²> / (4π)
print(f"  Nskew × <w_j²> / (4π) = {Nskew * np.mean(w_check**2) / (4*np.pi):.4e}")
print(f"  SN (computed)          = {np.sum(w_check**2) / (4*np.pi):.4e}")

# The key diagnostic: what fraction of pseudo-Cl is from diagonal vs off-diagonal?
# At high ℓ (large angular separation), off-diagonal <w_j w_k> → 0 (uncorrelated)
# So pseudo-Cl → SN (diagonal)
print(f"\n  cl_mean[{Nl-1}] = {cl_mean[Nl-1]:.4e}  (highest ℓ)")
print(f"  SN_mean       = {sn_mean:.4e}")
print(f"  cl/SN at high ℓ = {cl_mean[Nl-1] / sn_mean:.4f}")
print(f"  (cl - SN)[{Nl-1}] = {cl_mean[Nl-1] - sn_mean:.4e}")

# Theory signal at high ℓ:
print(f"  cl_limber[{Nl-1}] = {cl_limber[Nl-1]:.4e}")
print(f"  Mll@cl_limber[{Nl-1}] = {cl_convolved_limber[Nl-1]:.4e}")

# ====================================================================== #
#  Phase 5: Brute-force normalization scan                                #
# ====================================================================== #
print(f"\n{'='*60}")
print(f"BRUTE-FORCE NORMALIZATION")
print(f"{'='*60}")

# For the forward model <pseudo-Cl> = Mll @ C_true:
# Find alpha such that cl_mean ≈ Mll @ (alpha × cl_limber)
alpha_fit_raw = np.sum(cl_mean[2:] * cl_convolved_limber[2:]) / np.sum(cl_convolved_limber[2:]**2)
alpha_fit_sn = np.sum((cl_mean[2:] - sn_mean) * cl_convolved_limber[2:]) / np.sum(cl_convolved_limber[2:]**2)

print(f"Best alpha (raw):        {alpha_fit_raw:.6e}")
print(f"Best alpha (SN-sub):     {alpha_fit_sn:.6e}")
print(f"1/(32π³) =               {1/(32*np.pi**3):.6e}")
print(f"alpha_raw / (1/32π³) =   {alpha_fit_raw / (1/(32*np.pi**3)):.4f}")
print(f"alpha_sn / (1/32π³) =    {alpha_fit_sn / (1/(32*np.pi**3)):.4f}")

# Also check: does the window approach full-sky?
print(f"\n--- Window analysis ---")
print(f"cl_rand[0] = {cl_rand_0[0]:.4e}")
print(f"(N × Nskew)² / (4π) = {(N * Nskew)**2 / (4*np.pi):.4e}")
print(f"cl_rand[1] = {cl_rand_0[1]:.4e}")
print(f"cl_rand[Nl-1] = {cl_rand_0[Nl-1]:.4e}")
print(f"Full-sky Mll should be ≈ cl_rand[0] × δ_ll")
print(f"Mll[5,5] = {Mll[5,5]:.4e}")
print(f"cl_rand[0] × (2×5+1)/(4π) = {cl_rand_0[0] * 11 / (4*np.pi):.4e}")

# Check if mode coupling is approximately diagonal
diag = np.diag(Mll)
offdiag = Mll - np.diag(diag)
print(f"\nMll diagonal dominance:")
print(f"  |diag| mean     = {np.mean(np.abs(diag[2:])):.4e}")
print(f"  |offdiag| mean  = {np.mean(np.abs(offdiag[2:, 2:])):.4e}")
print(f"  ratio            = {np.mean(np.abs(offdiag[2:, 2:])) / np.mean(np.abs(diag[2:])):.4e}")

# If Mll is nearly diagonal, then <pseudoCl> ≈ Mll[l,l] × C_true[l]
# So C_true = cl_mean / Mll[l,l]
print(f"\nDirect C_true extraction (no window, just divide by Mll_ll):")
print(f"{'ell':>4} {'cl_data':>12} {'Mll_ll':>12} {'cl/Mll':>12} {'cl_limber':>12} {'ratio':>10}")
for l in [2, 5, 10, 20, 30, 40]:
    if l < Nl:
        c_ext = cl_mean[l] / diag[l]
        r = c_ext / cl_limber[l] if cl_limber[l] > 0 else np.inf
        print(f"{l:4d} {cl_mean[l]:12.4e} {diag[l]:12.4e} {c_ext:12.4e} {cl_limber[l]:12.4e} {r:10.6f}")

print(f"\nSN-subtracted extraction:")
print(f"{'ell':>4} {'(cl-SN)/Mll':>12} {'cl_limber':>12} {'ratio':>10}")
for l in [2, 5, 10, 20, 30, 40]:
    if l < Nl:
        c_ext = (cl_mean[l] - sn_mean) / diag[l]
        r = c_ext / cl_limber[l] if cl_limber[l] > 0 else np.inf
        print(f"{l:4d} {c_ext:12.4e} {cl_limber[l]:12.4e} {r:10.6f}")

# ====================================================================== #
#  Phase 6: Plot                                                          #
# ====================================================================== #
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)
ax, axr = axes

ells_plot = ells[2:]
cl_plot = cl_mean[2:]
sn_line = np.full_like(cl_plot, sn_mean)

# Theory predictions
ax.semilogy(ells_plot, cl_plot, 'C0-', lw=1.5, label=f'Measured (mean {num_sim} sims)')
ax.semilogy(ells_plot, cl_plot - sn_mean, 'C1--', lw=1.5, label='Measured − SN')
ax.semilogy(ells_plot, cl_convolved_limber[2:], 'k--', lw=2, label=r'$M_{\ell\ell} \cdot C^{\rm Limber}_\ell$')
ax.semilogy(ells_plot, cl_convolved_32pi3[2:], 'r--', lw=2, label=r'$M_{\ell\ell} \cdot C^{32\pi^3}_\ell$')
ax.axhline(sn_mean, color='gray', ls=':', label=f'SN = {sn_mean:.2e}')
ax.set_ylabel(r'$C_\ell(k=0)$')
ax.set_title(f'Full-sampling test: N={N_box}, Nskew=N²={Nskew}, Nl={Nl}, {num_sim} sims')
ax.legend(fontsize=9, loc='upper right')

# Ratio panel
for label, thy, ls in [
    ('raw / Mll@Limber', cl_convolved_limber, 'C0-'),
    ('(raw-SN) / Mll@Limber', cl_convolved_limber, 'C1--'),
]:
    if 'SN' in label:
        r = (cl_mean[2:] - sn_mean) / thy[2:]
    else:
        r = cl_mean[2:] / thy[2:]
    axr.plot(ells_plot, r, ls, lw=1.5, label=label)

axr.axhline(1.0, color='k', ls=':')
axr.set_xlabel(r'Multipole $\ell$')
axr.set_ylabel('Ratio')
axr.set_ylim(0, 3)
axr.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(root, 'notebooks', 'plots', 'fullsky_diagnostic.png'), dpi=150)
plt.savefig(os.path.join(root, 'notebooks', 'plots', 'fullsky_diagnostic.pdf'))
print(f"\nPlots saved to notebooks/plots/fullsky_diagnostic.*")

del GRF_check; gc.collect()
print("\nDone!")
