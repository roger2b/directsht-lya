#!/usr/bin/env python
"""
Minimal apples-to-apples window function test.

Goal: verify that measured binned pseudo-Cl matches binned C_theory from the
pair-counting formula, using EXACTLY the same paths as the original code.

Key insight from user: for Ly-α, weights = w*δ_F, so pseudo-Cl = |a_lm(data)|².
No D-R subtraction. The window function enters only through the theory's
coupling matrix and pair-counting.

Uses my_bias=1.0 to match the saved data that showed ratio ≈ 1.
Small Nl=100 and Nskew~2000 to save memory.
"""
import sys, os, gc, time
import numpy as np
import healpy as hp

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution
import GRF_class as my_GRF
import SHT_lya as sht_lya
import fast_Wigner3j as Wigner3j

# ---- Settings (small for memory) ---- #
chi_shift  = 5000
Nl         = 100
lambda_max = Nl
num_qso    = 2000     # → ~1991 unique sightlines
num_sim    = 5
NperBin    = 16
my_bias    = 1.0      # old defaults → theory should match
my_beta    = 1.5

sht_eng = DirectSHT(Nl, 2*Nl, 0.75)
print(f"DirectSHT: Nl={Nl}")

# ====================================================================== #
# STEP 1: Measure pseudo-Cl at k=0, average over sims                    #
# ====================================================================== #
print("\n=== STEP 1: Measure pseudo-Cl ===")
cl_stack, wl_ref = [], None
plin_ref, theta_ref, phi_ref, chi_ref, N_ref, Nskew_ref = [None]*6

for i in range(num_sim):
    t0 = time.time()
    G = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=1000+i,
                                       my_bias=my_bias, my_beta=my_beta)
    ax, ay, az, wr, wg, ns = G.process_skewers(Nskew=num_qso, shift=chi_shift)
    at, ap = G.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
    chi = ax[0, :]
    dF = wg - 1.0   # δ_F = (1+δ) - 1 = δ

    # DFT: same convention as original (scipy.linalg.dft, returns real)
    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi, wr, dF)

    # Verify: at k=0, FT_mask = N for all sightlines
    if i == 0:
        print(f"  FT_mask[0:3, 0] = {FT_mask[:3, 0]}  (should all be {chi.size})")
        print(f"  FT_mask[0, 1]   = {FT_mask[0, 1]:.1e}  (should be ~0 for k≠0)")

    # SHT at k=0
    hdat = sht_eng(at, ap, FT_delta[:, 0])
    cl = hp.alm2cl(hdat)[:Nl]
    cl_stack.append(cl)

    if i == 0:
        hran = sht_eng(at, ap, FT_mask[:, 0])
        wl_ref = hp.alm2cl(hran)[:Nl]
        plin_ref = G.plin
        theta_ref, phi_ref = at, ap
        chi_ref = chi
        N_ref = chi.size
        Nskew_ref = ns

    del G, ax, ay, az, wr, wg, FT_mask, FT_delta; gc.collect()
    print(f"  sim {i}: Nskew={ns}, {time.time()-t0:.1f}s")

cl_stack = np.array(cl_stack)
cl_mean = np.mean(cl_stack, axis=0)

dchi = chi_ref[1] - chi_ref[0]
chi_bar = 0.5 * (chi_ref.min() + chi_ref.max())
print(f"\nN={N_ref}, Nskew={Nskew_ref}, chi_bar={chi_bar:.1f}, dchi={dchi:.4f}")

# ====================================================================== #
# STEP 2: Theory - pair-counting (EXACT original formula)                 #
# ====================================================================== #
print("\n=== STEP 2: Theory (pair-counting approach) ===")

# a) Pair-counting angular window
nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
del nhat; gc.collect()

KjKk = N_ref**2  # periodic: all FT_mask[:,0] = N
print(f"KjKk = N^2 = {KjKk}")

t0 = time.time()
print("Computing Legendre pair-counting...", end="", flush=True)
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
print(f" done ({time.time()-t0:.1f}s)")
del cos_theta; gc.collect()

# b) Power spectrum at k=0 (no RSD since add_rsd=False → beta=0)
# With my_bias=1.0 and add_rsd=False: field is 1.0 * δ_m, theory is P_lin
# The key: compute_amplitudes3d does: my_bias * (1+beta*mu^2) * δ
# With add_rsd=False, beta is set to 0, so amplitude = my_bias * δ_m
# The field power is thus my_bias^2 * P_lin(k)
# 
# BUT: the original Power_spectrum function uses:
#   Kaiser_factor = 1 (when add_rsd=False)
#   return P_lin(k) — NOT my_bias^2 * P_lin
#
# With my_bias=1.0, this doesn't matter: b^2 = 1.
# With my_bias=-0.1521, this DOES matter: field has b^2*P_lin but theory uses P_lin.

L_range = np.arange(lambda_max, dtype=float)
pk_L = plin_ref(L_range / chi_bar)  # P_lin(ell/chi_bar) at k_par=0
pk_L[0] = plin_ref(0.5 / chi_bar)  # avoid P(0)

# c) Coupling matrix with P(k) as weights
t0 = time.time()
couple_pk = Wigner3j.CoupleMat(lambda_max, pk_L)
M_pk = couple_pk.compute_matrix()
print(f"Coupling matrix computed in {time.time()-t0:.1f}s")

# d) Original theory formula
C_theory = M_pk @ PLKjKk / (4*np.pi) / (2*np.pi * chi_bar**2)
C_theory_plotted = C_theory / (4*np.pi)**2

# ====================================================================== #
# STEP 3: Compare binned quantities                                       #
# ====================================================================== #
print("\n=== STEP 3: Binned comparison ===")

bins_mat = np.zeros((Nl // NperBin, Nl))
for b in range(bins_mat.shape[0]):
    bins_mat[b, b*NperBin:(b+1)*NperBin] = 1.0 / NperBin

ells = np.arange(Nl, dtype=float)
bn_ells = bins_mat @ ells
bn_data = bins_mat @ cl_mean
bn_theory = bins_mat @ C_theory_plotted[:Nl]

print(f"\nComparing: binned <pseudo-Cl>  vs  binned C_theory/(4π)²")
print(f"{'ell':>6s} {'data':>12s} {'theory':>12s} {'ratio':>8s}")
print("-"*42)
ratios = []
for i in range(len(bn_ells)):
    r = bn_data[i] / bn_theory[i] if bn_theory[i] > 0 else np.inf
    ratios.append(r)
    print(f"{bn_ells[i]:6.0f} {bn_data[i]:12.4e} {bn_theory[i]:12.4e} {r:8.4f}")

mr = np.mean(ratios[1:])
print(f"\nMean ratio (excl monopole): {mr:.4f}")
print(f"  -> If ~1.0: original pair-counting theory is correct")
print(f"  -> If ~b²={my_bias**2:.4f}: theory is missing b² factor")

# ====================================================================== #
# STEP 4: Alternative approach — use MaskDeconvolution forward model      #
# ====================================================================== #
print("\n=== STEP 4: MaskDeconvolution forward model ===")
print("Here: <pseudo-Cl[l]> = Mll[l,L] @ C_true[L]")
print("where Mll is built from wl_ref (the cl_rand window)")
print("and C_true[L] = P_F(L/chi_bar, k=0) / chi_bar²")

MD = MaskDeconvolution(Nl, wl_ref)
Mll = MD.Mll  # (Nl, Nl)

# Theory true Cl  
C_true_l = pk_L[:Nl] / chi_bar**2

# Forward-modeled pseudo-Cl
pseudo_predicted = Mll @ C_true_l

bn_forward = bins_mat @ pseudo_predicted

print(f"\nComparing: binned <pseudo-Cl>  vs  binned Mll@C_true")
print(f"{'ell':>6s} {'data':>12s} {'Mll forward':>12s} {'ratio':>8s}")
print("-"*42)
ratios_fwd = []
for i in range(len(bn_ells)):
    r = bn_data[i] / bn_forward[i] if bn_forward[i] > 0 else np.inf
    ratios_fwd.append(r)
    print(f"{bn_ells[i]:6.0f} {bn_data[i]:12.4e} {bn_forward[i]:12.4e} {r:8.4f}")

mrf = np.mean(ratios_fwd[1:])
print(f"\nMean ratio (excl monopole): {mrf:.4f}")

# ====================================================================== #
# STEP 5: Verify the identity: M_pk @ PLKjKk = 4π × Mll @ pk_L          #
# ====================================================================== #
print("\n=== STEP 5: Verify M_pk @ PLKjKk == 4π × Mll @ pk_L ===")
lhs = M_pk @ PLKjKk
rhs = 4.0 * np.pi * Mll @ pk_L[:Nl]
# Note: Mll is Nl x Nl, but M_pk is lambda_max x lambda_max
# pad pk_L if needed
for ell in [2, 5, 10, 20, 50]:
    if ell < Nl:
        print(f"  ell={ell}: LHS={lhs[ell]:.6e}, 4π*Mll@pk={rhs[ell]:.6e}, "
              f"ratio={lhs[ell]/rhs[ell]:.6f}")

# ====================================================================== #
# STEP 6: What's the relationship?                                        #
# ====================================================================== #
print("\n=== STEP 6: Algebraic relationship ===")
print("Original code plots C_theory/(4π)² where:")
print("  C_theory = M_pk @ PLKjKk / (4π × 2π × χ²)")
print("Using identity: M_pk @ PLKjKk = 4π × Mll @ pk_L")
print("  C_theory = 4π × Mll @ pk_L / (4π × 2π × χ²)")
print("           = Mll @ pk_L / (2π × χ²)")
print("  C_theory/(4π)² = Mll @ pk_L / (2π × χ² × 16π²)")
print("                 = Mll @ pk_L / (32π³ × χ²)")
print("                 = Mll @ [pk_L / (32π³ × χ²)]")
print()

# So: C_theory_plotted[l] = Mll[l,L] × pk_L[L] / (32π³ χ²)
# Meanwhile: <pseudo-Cl[l]> = Mll[l,L] × C_true[L]
# For these to match: C_true[L] = pk_L[L] / (32π³ χ²)
#
# But from LIMBER: C_L = P_F(L/χ, k) / χ²
# So the difference is a factor of 32π³ ≈ 992.2

factor = 32 * np.pi**3
print(f"If C_true[L] = pk_L / χ² (Limber), then:")
print(f"  ratio = C_theory_plotted / (Mll @ C_Limber)")
print(f"        = 1 / (32π³) = {1/factor:.6e}")
print(f"So C_theory_plotted = Mll @ C_Limber / {factor:.1f}")
print()

# Let's check: is the measured pseudo-Cl ≈ Mll @ C_Limber?
pseudo_limber = Mll @ C_true_l  # C_true_l = pk_L/chi^2
bn_limber = bins_mat @ pseudo_limber

print(f"Measured BINNED pseudo-Cl[1] = {bn_data[1]:.6e}")
print(f"Mll @ (pk/chi²) BINNED [1]  = {bn_limber[1]:.6e}")
print(f"Ratio                        = {bn_data[1]/bn_limber[1]:.6f}")
print(f"C_theory_plotted BINNED [1]  = {bn_theory[1]:.6e}")
print(f"Mll@(pk/chi²) / (32π³)  [1] = {bn_limber[1]/factor:.6e}")
print(f"Ratio theory/Mll-based       = {bn_theory[1]/(bn_limber[1]/factor):.6f}")
print()

# The conclusion: the original pair-counting theory ALREADY includes the window.
# It predicts: C_theory_plotted[l] = (1/32π³χ²) × ΣL Mll[l,L] pk_L[L]
# But this is NOT equal to the pseudo-Cl in general.
# 
# The question is: what EXACT normalization makes <pseudo-Cl> match the theory?
# From STEP 3, the ratio tells us.

# ====================================================================== #
# STEP 7: Direct derivation of <pseudo-Cl>                               #
# ====================================================================== #
print("=== STEP 7: First-principles derivation ===")
print("""
Measured: a_lm = Σ_j w_j Y*_lm(n_j)
where w_j = Σ_n δ_F(j,n) [DFT at k=0, unnormalized]

<pseudo-Cl> = (1/(2l+1)) Σ_m <|a_lm|²>
            = (1/(2l+1)) Σ_m Σ_{j,k} <w_j w_k> Y*_lm(nj) Y_lm(nk)
            = Σ_{j,k} <w_j w_k> P_l(cos θ_jk) / (4π)

Now: w_j = Σ_n δ_F(j,n)
<w_j w_k> = Σ_{n,n'} <δ(j,n) δ(k,n')>

For GRF in periodic box:
  δ(x) = Σ_K δ̂_K e^{iK·x}  with <|δ̂_K|²> = P(K)/V_box
  
  Field = b₁ × δ_m (since beta=0, add_rsd=False)
  <|δ̂_K|²> = b₁² × P_lin(K) / V_box

For sightline j at (y_j, z_j) with LOS pixels at x_n:
  δ_F(j,n) = b₁ × Σ_K δ̂_K e^{i(K_x x_n + K_y y_j + K_z z_j)}

  w_j = Σ_n Σ_K δ̂_K e^{i(K_y y_j + K_z z_j)} e^{iK_x x_n}
      = Σ_K δ̂_K e^{i(K_y y_j + K_z z_j)} [N δ_{K_x,0}]  
      = N × Σ_{K_y,K_z} δ̂_{0,K_y,K_z} e^{i(K_y y_j + K_z z_j)}

  Note: b₁ is already absorbed into δ̂_K.

<w_j w_k> = N² × Σ_{K_y,K_z} <|δ̂_{0,K_y,K_z}|²> e^{iK_⊥·Δr_⊥}
          = N² × Σ_{K_⊥} P_F(K_⊥, K_x=0) / V_box × e^{iK_⊥·Δr}

where P_F = b₁² P_lin (with add_rsd=False).

Now <pseudo-Cl> = Σ_{j,k} <w_j w_k> P_l(cos θ_jk) / (4π)

Substituting:
<pseudo-Cl> = (N² / V_box) Σ_{j,k} Σ_{K_⊥} P_F(K_⊥,0) e^{iK_⊥·Δr} P_l(cosθ) / (4π)

This can be written as a sum over transverse modes:
<pseudo-Cl> = (N²/V) Σ_{K_⊥} P_F(K_⊥,0) × W_l(K_⊥)

where W_l(K_⊥) = Σ_{j,k} e^{iK_⊥·Δr_jk} P_l(cosθ_jk) / (4π)
""")

# Let me compute this numerically for a few ell values
print("Now computing numerical verification...")

# V_box = L^3 but the sightlines span L in transverse and L_box in LOS
# Actually V_box = L^3 for the GRF
L = N_ref * dchi  # same as GRF.L
V_box = L**3

print(f"L = {L:.1f} Mpc/h = N*dchi = {N_ref}*{dchi:.4f}")
print(f"V_box = L^3 = {V_box:.2e}")
print(f"N²/V = {N_ref**2/V_box:.6e}")

# For the Limber-like approach (continuum limit):
# Σ_{K_⊥} → (L/(2π))² ∫ d²K_⊥
# = (L/2π)² ∫ K_⊥ dK_⊥ dφ = (L/2π)² × 2π ∫ K_⊥ dK_⊥ [azimuthal symmetry]
# So:
# <pseudo-Cl> ≈ (N²/V) × (L/2π)² × 2π × ∫ K_⊥ dK_⊥ P_F(K_⊥,0) W_l(K_⊥)
#             = N²/(L × (2π)) ∫ K_⊥ dK_⊥ P_F(K_⊥,0) W_l(K_⊥)

# If W_l(K_⊥) is sharply peaked at K_⊥ = l/χ_bar (Limber)...
# Actually W_l(K_⊥) involves the pair counting, which encodes the geometry.
# The standard result is that for a distant survey:
#   W_l(K_⊥) → (something) × δ(K_⊥ - l/χ_bar) × 2π/K_⊥
# This is the flat-sky/Limber approximation.

# For our DISCRETE sightlines at distance χ_bar from the observer, 
# the transverse separation r_⊥ relates to the angle θ by r_⊥ ≈ χ_bar θ.
# The Limber approximation gives:
# <pseudo-Cl> = (N²/V) × (L²/(2π)) × P_F(l/χ_bar, 0) / χ_bar² × [angular window factor]

# Let's just check the numerical ratio and see what factor we need:
factor_needed = bn_data[1] / bn_theory[1]
print(f"\nRatio data/theory_plotted at bin 1: {factor_needed:.6f}")

# Compute: N²/V × L² / (2π)
nv_l2_2pi = N_ref**2 / V_box * L**2 / (2*np.pi)
print(f"N²/V × L²/(2π) = {nv_l2_2pi:.6e}")

# And the original code's normalization 1/(32π³χ²):
code_norm = 1.0 / (32 * np.pi**3 * chi_bar**2)
print(f"1/(32π³χ²) = {code_norm:.6e}")

# The additional factor to make them match:
print(f"Needed extra factor: {factor_needed:.6e}")
print(f"  × (32π³χ²) = {factor_needed * 32*np.pi**3*chi_bar**2:.6e}")
