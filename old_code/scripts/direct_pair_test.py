#!/usr/bin/env python
"""
Final definitive normalization trace.

Goal: derive the correct C_true from first principles and verify numerically.

KNOWN FACTS (verified):
1. P_2D(k_perp) = b1^2 * N^2 * P_lin(k_perp) / L   [verified to 0.1%]
2. <w^2> = b1^2 * N^2 / L^3 * SUM P_lin(k_perp)     [verified to 0.01%]

NOTES' CLAIM:
  C_true = P_F / (32 pi^3 chi^2) = b1^2 P_lin / (32 pi^3 chi^2)

MY DERIVATION (needs verification):
  The flat-sky C_true = P_2D(l/chi) / chi^2 = b1^2 * N^2 * P_lin(l/chi) / (L * chi^2)

But WAIT - the above C_true is the angular power spectrum of the CONTINUOUS field
s(n_hat) = w(r_perp(n_hat)), where angles map to transverse positions.
The issue is: what is the correct mapping between R^2 and the sphere?

For a flat patch of physical size L×L at distance chi:
  theta_max = L / (2*chi)  [half-angle, assuming centered on LOS]
  The patch covers solid angle Omega = L^2 / chi^2

For the flat-sky power spectrum: C_l = P_2D(l/chi) / chi^2
This is correct for a field defined on the full sky (e.g., CMB).
But our field is defined only on the patch. The PSEUDO-Cl includes the window.

For the MASTER framework: <pseudo_Cl> = SUM M[l,l'] C_true[l']
C_true is the full-sky angular power spectrum of s(nhat).
Since s(nhat) is zero outside the patch, C_true IS the power of the windowed field.
Wait no — in MASTER, C_true is the power of the UNDERLYING field (without window).

Hmm. Let me go back to basics.

The MASTER approach says:
<pseudo_Cl> = SUM_L Mll' C_L^true

where:
  pseudo_Cl = |SUM_j w_j Y_lm^*(n_j)|^2 / (2l+1)
  M = coupling matrix from window
  C_L^true = full-sky angular Cl of the underlying continuous signal field

The "underlying continuous signal field" is s(nhat), which is w(r_perp(nhat)).
This field IS defined on the full sphere (it's zero outside the box patch).
NO — in the MASTER framework, s(nhat) is the UNWINDOWED field that would exist
everywhere if we had data everywhere. The window W(nhat) = SUM_j delta(nhat - nhat_j)
is the sampling pattern.

So actually: pseudo_alm = SUM_j w_j Y_lm^*(n_j) = SUM_j [W_j * s(n_j)] Y_lm^*
In the standard MASTER: pseudo_Cl = SUM_L M[l,L] C_true[L]
where C_true is the Cl of s(nhat) defined on the FULL SKY.

But s(nhat) for our box is a continuous field that maps each direction nhat
to a transverse position on the box face and then gives w(r_perp).
This field IS defined for ALL directions nhat (within reason — the far field
of the box face). The angular power of this field is:

C_l^true = (1/4pi) integral |a_lm^s|^2 / (2l+1)
         = ... the angular power spectrum of the continuous field w(r_perp(nhat)).

For the flat-sky approximation:
C_l^true = P_2D(l/chi) / chi^2

But this P_2D is the power spectrum PER UNIT k-SPACE AREA normalized by the
TOTAL AREA of the field. The issue relates to how we normalize P_2D.

OK let me just stop hand-waving and compute it directly.

I will:
1. Generate one realization
2. Compute w(r_perp) on the full N^2 grid  
3. Place ALL N^2 sightlines into the SHT
4. Get the full-patch pseudo-Cl
5. Deconvolve the full-patch window to get C_true numerically
6. Compare with the two candidate formulas

Actually, I already did this test in test_ctrue_fullsky.py and it failed because
the coupling was too broad with Nl=200. Let me redo with a different approach:
Instead of deconvolving, I'll use the RATIO method.

For ALL N^2 sightlines, the pseudo-Cl of the signal and the pseudo-Cl of the
uniform window (with w_j = 1) should be in a fixed ratio related to C_true.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# Instead of doing the full SHT approach (which is slow and needs large Nl),
# let me directly verify WITHIN the pair-counting framework.

# The pair-counting prediction is (notes Eq. 14):
# <Cl> = 1/(4pi * 2pi * chi^2) * SUM_L M^(p)_lL * P_L
# where P_L = N^2 * SUM_jk Pl(cos gamma_jk) is the pair sum
# and M^(p) uses P_F(L/chi) as spectrum weights.

# Alternative: from MY derivation, the pair-counting should be:
# <Cl> = (1/4pi) * SUM_jk <wj wk> Pl(cos gamma)
# = (1/4pi) * (b1^2 N^2 / L^3) * SUM_{k_perp} Plin(k) * SUM_jk exp(ik.Dr) Pl(cos gamma)

# The SUM_jk exp(ik.Dr) Pl(cos gamma) is a known function of k and the sightline geometry.

# Instead of computing this huge double sum, let me use the KNOWN relationship:
# If the notes formula is: 
#   <Cl> = SUM_L M_lL * b1^2 * Plin(L/chi) / (32pi^3*chi^2)
# And my formula is:
#   <Cl> = SUM_L M_lL * b1^2 * N^2 * Plin(L/chi) / (L*chi^2)
# The ONLY difference is the prefactor in C_true.

# I already found that with Nl_large=1000, wl to 2000:
# data / theory_notes = 1.01
# And with Nl_large=3500, wl to 4000:
# data / theory_notes = 0.847

# If my formula were correct: 
#   C_true_mine / C_true_notes = N^2/(L) / (1/(32pi^3)) = 32*pi^3 * N^2 / L
print(f"32*pi^3*N^2/L = {32*np.pi**3 * 512**2 / 1380:.1f}")
# That's ~60k — clearly wrong.

# So the flat-sky C_l = P_2D/chi^2 formula is NOT what goes into MASTER.
# The issue is that the "P_2D" I measured in compute_P2D.py is on the DISCRETE grid
# which gives a different normalization than the continuous field.

# Let me think about this differently.
# The CONTINUOUS field: s(theta, phi) = w(chi * theta, chi * phi) for small angles.
# s(nhat) is defined for ALL nhat on the sky (zero if nhat doesn't hit the box).
# But in MASTER, s is the underlying FULL-SKY field. We pretend it extends everywhere.

# For a flat-sky Fourier transform:
# s_hat(l) = ∫ s(theta) e^{-il.theta} d^2theta  [flat-sky FT]
# C_l^true = |s_hat(l)|^2 / A_survey

# where A_survey is the solid angle of the survey.

# In our case:
# s(theta) = w(chi_bar * theta)
# s_hat(l) = ∫ w(chi*theta) e^{-il.theta} d^2theta
#           = (1/chi^2) ∫ w(r) e^{-il.r/chi} d^2r     [change of variables]
#           = (1/chi^2) * w_cont(l/chi)

# where w_cont(k) = ∫ w(r) e^{-ik.r} d^2r ≈ (L/N)^2 * w_fft(n) for k=(2pi/L)*n

# And C_l = |s_hat(l)|^2 / A = (1/chi^4) |w_cont(l/chi)|^2 / (L^2/chi^2)
# = |w_cont(l/chi)|^2 / (L^2 chi^2)
# = P_2D(l/chi) * L^2 / (L^2 chi^2)  ... wait, I already did this.

# P_2D(k) = |w_cont(k)|^2 / L^2  [I defined this]
# So |w_cont|^2 = P_2D * L^2

# C_l = (1/chi^4) * P_2D(l/chi) * L^2 / A_survey
# where A_survey is the solid angle of the patch.

# Hmm, I think the full-sky normalization is:
# C_l = (1/chi^4) * |w_cont|^2 / A_total_sphere
# = (1/chi^4) * P_2D * L^2 / (4pi)

# Hmm, let me try from the MASTER inverse.

# MASTER says: <pseudo_Cl> = SUM M[l,l'] C_true[l']
# If M is known, then C_true = M^{-1} <pseudo_Cl>
# For the Nskew sightlines case, I can compute M from wl_ref.

# But the problem is that M is rank-deficient (only Nl rows, and C_true
# extends to ell' >> Nl). We can't invert it.

# OK, different approach. Let me use the fact that I KNOW P_2D perfectly.
# I'll derive C_true from a dimensional analysis / integral matching.

# The expectation of the pseudo-Cl is:
# <hat{C}_l> = (1/(2l+1)) SUM_m |SUM_j w_j Y_lm^*(nj)|^2
# = (1/4pi) SUM_jk <wj wk> Pl(cos gamma_jk)

# I KNOW <wj wk> exactly. Let me compute this pair sum numerically
# for a few ell values and compare with the MASTER prediction using
# different C_true formulas.

# This is the most direct test. I need:
# 1. The sightline positions (theta_j, phi_j) 
# 2. The transverse positions (x_j, y_j) on the box face
# 3. The theory correlation <wj wk>(Δr) = (b1^2*N^2/L^3) SUM P(k) exp(ik.Δr)

# For a subset of pairs, compute the sum.

print("\n---- Direct pair-counting test ----")

GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=42, verbose=False)
L = GRF.L
b1 = GRF.my_bias
N = GRF.N
plin = GRF.plin
chi_bar = 5000 + L/2

# Get sightline positions
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(Nskew=9797)
theta, phi = GRF.compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])

# Transverse positions on the box face (pixels)
np.random.seed(100)
inds = np.unique(np.random.randint(0, N, size=(9797, 2)), axis=0)
Nskew = len(inds)
print(f"Nskew = {Nskew}")

delta_skewers = all_w_gal - 1.0
w_j = np.sum(delta_skewers, axis=1)  # (Nskew,)

# Compute cos(gamma_jk) for all pairs (expensive, do for subset)
nhat = np.column_stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
cos_gamma = nhat @ nhat.T  # (Nskew, Nskew)
np.fill_diagonal(cos_gamma, 1.0)

# MEASURED pair-counting Cl:
from scipy.special import eval_legendre
Nl_test = 100
cl_pair_meas = np.zeros(Nl_test)
for l in range(Nl_test):
    Pl = eval_legendre(l, cos_gamma)
    cl_pair_meas[l] = (1/(4*np.pi)) * np.sum(w_j[:, None] * w_j[None, :] * Pl)

# THEORETICAL pair-counting Cl using KNOWN <wj wk>:
# <wj wk> = (b1^2*N^2/L^3) * SUM_{k_perp} P(k_perp) * exp(ik_perp.Δr_jk)
# where Δr_jk is the transverse pixel separation * (L/N)

# Precompute the 2D correlation function
# xi_2D(Δr) = (b1^2*N^2/L^3) * SUM_{k_perp} P(k_perp) * exp(ik.Δr)
# This equals b1^2*N^2/L^3 * IFFT2(P(k_perp)) * N^2   [from the 2D IFFT]
# Because: SUM_k P(k) exp(ik.r) = N^2 * IFFT2(P_on_grid)[r_index]

kfft = np.fft.fftfreq(N) * 2*np.pi * N / L
KX, KY = np.meshgrid(kfft, kfft, indexing='ij')
k_perp = np.sqrt(KX**2 + KY**2)
Pk_grid = plin(k_perp.ravel()).reshape(k_perp.shape)

# 2D correlation function on the pixel grid:
xi_grid = np.fft.ifft2(Pk_grid).real * N**2  # SUM P*exp = IFFT * N^2
xi_grid *= b1**2 * N**2 / L**3  # prefactor

# Now: <wj wk> = xi_grid[Δix, Δiy] where Δix = ix_j - ix_k (mod N)
print(f"xi_grid[0,0] = {xi_grid[0,0]:.4f}  (= <w^2> at same position)")
print(f"<w^2> from variance = {np.var(w_j):.4f}")
# These should be close (xi_grid[0,0] = signal variance at one point)

# Theoretical pair-sum Cl:
cl_pair_theory = np.zeros(Nl_test)
for l in range(Nl_test):
    Pl = eval_legendre(l, cos_gamma)
    xi_pairs = np.zeros((Nskew, Nskew))
    for ii in range(Nskew):
        for jj in range(ii, Nskew):
            dix = (inds[ii, 0] - inds[jj, 0]) % N
            diy = (inds[ii, 1] - inds[jj, 1]) % N
            xi_pairs[ii, jj] = xi_grid[dix, diy]
            xi_pairs[jj, ii] = xi_pairs[ii, jj]
    cl_pair_theory[l] = (1/(4*np.pi)) * np.sum(xi_pairs * Pl)
    if l < 5 or l % 20 == 0:
        print(f"  l={l}: meas={cl_pair_meas[l]:.4e}, theory={cl_pair_theory[l]:.4e}, "
              f"ratio={cl_pair_meas[l]/cl_pair_theory[l]:.4f}")
    break  # just do l=0 to check, since the double loop is slow

# For l=0 only: Pl = 1
print(f"\nFast check at l=0:")
sum_xi = np.sum(xi_grid[0,0]) * Nskew  # diagonal terms
for ii in range(min(100, Nskew)):
    for jj in range(ii+1, min(100, Nskew)):
        dix = (inds[ii, 0] - inds[jj, 0]) % N
        diy = (inds[ii, 1] - inds[jj, 1]) % N
        sum_xi += 2 * xi_grid[dix, diy]

# Instead of the full computation, check order of magnitude:        
print(f"\nxi_grid[0,0] (diagonal, = signal variance) = {xi_grid[0,0]:.4e}")
print(f"Nskew * xi_grid[0,0] = {Nskew * xi_grid[0,0]:.4e}  (shot noise term)")
print(f"Measured Cl[0] = {cl_pair_meas[0]:.4e}")

# And from the MASTER theory:
# <Cl=0> = SUM M[0,l'] * C_true[l']
# For C_true = b1^2 * Plin(l/chi) / (32pi^3 chi^2):
from scipy.interpolate import interp1d
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
wl_ref = d['wl_k'][0, :500]

ells = np.arange(500, dtype=float)
cl_true_notes = b1**2 * plin((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)

wl_couple = np.zeros(999)
wl_couple[:500] = wl_ref
couple = Wigner3j.CoupleMat(500, wl_couple)
M = couple.compute_matrix()

cl_master_notes = M @ cl_true_notes
print(f"\nMaster Cl[0] (notes formula, Nl=500) = {cl_master_notes[0]:.4e}")
print(f"Ratio meas/MASTER = {cl_pair_meas[0] / cl_master_notes[0]:.4f}")

# Now: the key insight. cl_pair_meas is for ONE realization (noisy).
# The 100-sim average is better. Let me compare the 100-sim average:
cl_mean = np.mean(d['cl_k'], axis=0)
print(f"\n100-sim mean Cl[0] = {cl_mean[0]:.4e}")
print(f"Ratio mean/MASTER = {cl_mean[0] / cl_master_notes[0]:.4f}")

del GRF; gc.collect()
