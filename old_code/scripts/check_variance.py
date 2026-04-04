#!/usr/bin/env python
"""
Compute the expected pseudo-Cl DIRECTLY from the discrete 2D modes.

For the field w_j = sums along LOS at sightline positions:
  <w_j w_k> = (b1^2 * N^2 / L^3) * SUM_{k_perp} P_lin(|k_perp|) * exp(ik_perp . (r_j - r_k))

where k_perp = (2pi/L) * (nx, ny) with nx, ny = -N/2+1, ..., N/2.

The pseudo-Cl is:
  <Cl> = SUM_{j,k} <w_j w_k> * (1/(2l+1)) SUM_m Y_lm^*(n_j) Y_lm(n_k)
       = (2l+1)/(4pi) * SUM_{j,k} <w_j w_k> * P_l(cos theta_{jk})  [addition theorem]

Wait, that's the addition theorem in the wrong direction. Let me be careful:
  SUM_m Y_lm^*(n_j) Y_lm(n_k) = (2l+1)/(4pi) * P_l(cos theta_{jk})

So:
  <Cl> = (1/(2l+1)) * (2l+1)/(4pi) * SUM_{j,k} <w_j w_k> P_l(cos theta_{jk})
       = (1/(4pi)) * SUM_{j,k} <w_j w_k> P_l(cos theta_{jk})

But this mixes up the signal and window. In the PSEUDO-Cl framework:
  <pseudo Cl> = SUM_l' M[l,l'] C_true[l']

The M matrix involves the WINDOW only, and C_true is the signal.

Actually, let me think differently. Instead of MASTER, compute:
  pseudo_Cl = (1/(2l+1)) SUM_m |a_lm|^2
  a_lm = SUM_j w_j Y_lm^*(n_j)

So: <pseudo_Cl> = (1/(2l+1)) SUM_m SUM_{j,k} <w_j w_k> Y_lm^*(n_j) Y_lm(n_k)

Substituting:
  <w_j w_k> = prefactor * SUM_{k_perp} P(k_perp) exp(ik_perp . Delta_r_{jk})

where prefactor = b1^2 * N^2 / L^3, and Delta_r is the PHYSICAL transverse separation.

Wait, but the exp(ik.Dr) term uses GRID indices. Let me re-express in physical units.

Actually, let me go back to basics and just compute the ensemble average by averaging
the theory prediction over k_perp modes numerically.

For EACH k_perp mode, the contribution to the field is:
  w_j^{(k)} = (b1*N/L^{3/2}) * a(k_perp, 0) * exp(ik_perp . r_j)
where r_j is the physical transverse position: r_j = n_j * (L/N)

The SHT of this single-mode field:
  a_lm^{(k)} = (b1*N/L^{3/2}) * a(k_perp, 0) * SUM_j exp(ik_perp . r_j) * Y_lm^*(n_hat_j)

And the contribution to <Cl> from this mode:
  <Cl>^{(k)} = (b1^2*N^2/L^3) * P(k_perp) * |SUM_j exp(ik.r_j) Y_lm^*(n_j)|^2 / (2l+1)

Summing over all k_perp modes gives the total <pseudo Cl>.

This is exact but requires computing |SUM_j exp(ik.r_j) Y_lm^*(n_j)|^2 for each k_perp,
which is expensive (N^2 sightlines * N^2 k-modes * Nl multipoles).

Instead, let me use the PAIR-COUNTING approach which is equivalent but faster:
  PLKjKk_signal[l] = SUM_{j,k} P_lin(|Delta_r_{jk} * k_f|) * P_l(cos_jk) 
                     ... no, this doesn't work because P_lin depends on k_perp, not r.

OK, let me try a fundamentally different approach. Instead of the angular MASTER,
compute what the simulation ACTUALLY gives as Cl by averaging the discrete modes.

The fastest approach: for one realization, w(n_hat) is known → run SHT → get pseudo_Cl.
Average over 100 realizations → compare with theory.
We HAVE this: it's cl_mean from the cache.

So the question is: what theory should match cl_mean?

Let me try to compute the theory prediction from the DISCRETE sum over k-modes
rather than the continuous Limber integral.

Each k-mode (nx, ny) contributes:
  c_l^{(nx,ny)} = (b1^2*N^2/L^3) * P_lin(k_f*sqrt(nx^2+ny^2)) * F_l(k_perp)

where F_l(k_perp) = |SUM_j Y_lm^*(n_j) exp(ik_perp.r_j)|^2 / (2l+1) summed over m.

= (1/(2l+1)) * SUM_{j,k} cos(k_perp.(r_j-r_k)) * SUM_m Y_lm^*(n_j) Y_lm(n_k)
= (1/(4pi)) * SUM_{j,k} cos(k_perp.Delta_r) * P_l(cos theta_{jk})

Hmm, this still involves the full pair sum.

OK, let me just try the simplest possible thing: 
compare the RATIO of <w^2> measured vs theory.
If this differs from 1, there's a clear normalization issue.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF

# We need multiple realizations to measure <w^2>
# The cached data has 100 sims — but only the PSEUDO-CL, not w_j directly.
# Let me generate a few sims and compute <w_j^2> vs theory.

N = 512
num_qso = 9797  # not int(1e4) because process_skewers uses unique random indices

print(f"N={N}, num_qso target={num_qso}")

var_w_measured = []
var_w_theory = []

for seed in range(100, 105):
    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=seed, verbose=False)
    L = GRF.L
    plin = GRF.plin
    b1 = GRF.my_bias
    dens = GRF.dens
    
    # Get the skewer positions (same random seed=100 in process_skewers)
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew_actual = GRF.process_skewers(Nskew=num_qso)
    
    # Compute w_j for the actual sightlines from the DENSITY field
    # all_w_gal = dens + 1, so delta = all_w_gal - 1
    # w_j = SUM_z delta_j = SUM_z (all_w_gal[j, :] - 1)
    # But this is for Nskew sightlines only.
    delta_skewers = all_w_gal - 1.0  # (Nskew, Nz)
    w_j = np.sum(delta_skewers, axis=1)  # (Nskew,)
    
    var_w_measured.append(np.var(w_j))
    
    del GRF; gc.collect()

print(f"Nskew_actual = {Nskew_actual}")
print(f"L = {L:.4f}, b1 = {b1:.4f}")

# Theoretical variance:
# <w_j^2> = (b1^2 * N^2 / L^3) * SUM_{k_perp} P_lin(|k_perp|)
# where sum is over the 2D FFT grid at kz=0.
k_f = 2*np.pi / L
kfft = np.fft.fftfreq(N) * 2*np.pi * N / L
KX, KY = np.meshgrid(kfft, kfft)
k_perp_mag = np.sqrt(KX**2 + KY**2)
Pk_2d = plin(k_perp_mag.ravel()).reshape(k_perp_mag.shape)
sum_Pk = np.sum(Pk_2d)

var_w_theory_val = b1**2 * N**2 / L**3 * sum_Pk

print(f"\n<w^2> theory  = {var_w_theory_val:.4e}")
print(f"<w^2> measured = {np.mean(var_w_measured):.4e} (±{np.std(var_w_measured):.4e})")
print(f"Ratio measured/theory = {np.mean(var_w_measured) / var_w_theory_val:.4f}")
print(f"Individual ratios: {[f'{v/var_w_theory_val:.4f}' for v in var_w_measured]}")

# Now check the ANGULAR POWER content.
# If we sum the pseudo-Cl weighted by (2l+1)/(4pi):
# SUM_l (2l+1)/(4pi) * <Cl> = SUM_l (2l+1)/(4pi) * SUM_l' M[l,l'] C_true[l']

# For a point source:
# SUM_l (2l+1)^2/(4pi) * |Y_lm|^2 → delta function
# So SUM_l (2l+1) * pseudo_Cl / (4pi) = SUM_j w_j^2 / (4pi)?

# Actually, Parseval's theorem for pseudo-alm:
# SUM_l (2l+1) Cl = SUM_l SUM_m |a_lm|^2 = SUM_j,k w_j w_k SUM_l SUM_m Y_lm^*(j) Y_lm(k)
# = SUM_j,k w_j w_k * delta(n_j, n_k) [orthogonality, but not for discrete points!]
# For discrete points at DIFFERENT positions:
# SUM_{l=0}^{Nl-1} SUM_m Y_lm^*(n_j) Y_lm(n_k) ≠ delta_{jk}
# So this doesn't simplify cleanly.

# Instead, let me just check: does the C_true formula give the right TOTAL power?
# Total <pseudo_Cl> weighted by (2l+1): 
# T = SUM_l (2l+1) <Cl> / (4pi)
# From data:
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_mean = np.mean(d['cl_k'], axis=0)
Nl = 500
ells = np.arange(Nl, dtype=float)
T_data = np.sum((2*ells+1) * cl_mean) / (4*np.pi)
print(f"\nTotal power SUM (2l+1)*Cl/(4pi) from data = {T_data:.4e}")

# From theory with notes formula:
chi_bar = 5000 + L/2
cl_true_notes = b1**2 * plin((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
T_theory_notes = np.sum((2*ells+1) * cl_true_notes) / (4*np.pi)
print(f"Total power from C_true (notes) to Nl=500 = {T_theory_notes:.4e}")
print(f"Ratio data/theory_notes = {T_data/T_theory_notes:.4e}")

# This ratio should equal wl[0] if M is dominated by the shot noise term.
# Because SUM_l (2l+1) M[l,l'] C[l'] = wl[0] * SUM_l' C[l'] * (2l'+1)/(4pi)?
# Not exactly.

# Actually, let me check: what is SUM_l (2l+1) M[l,l']?
# M[l,l'] = (2l'+1)/(4pi) SUM_lambda (2lambda+1) wl[lambda] (3j000)^2
# SUM_l (2l+1) M[l,l'] = (2l'+1)/(4pi) SUM_lambda (2lambda+1) wl[lambda] SUM_l (2l+1) (3j000)^2
# By 3j orthogonality: SUM_l (2l+1) (3j(l,l',lambda;0,0,0))^2 = 1
# So SUM_l (2l+1) M[l,l'] = (2l'+1)/(4pi) SUM_lambda (2lambda+1) wl[lambda] * 1
# Hmm, that's SUM_lambda (2lambda+1) wl[lambda]. But this sum equals:
# SUM_lambda (2lambda+1) wl[lambda] = SUM_j 1^2 * SUM_lambda (2lambda+1) = 4pi * SUM_j 1 = 4pi * Nskew
# Wait, SUM_lambda (2lambda+1) Cl = 4pi * f(0) where f is the 2PCF at theta=0.
# For wl = |SUM_j Y_lm^*(j)|^2 / (2l+1): SUM_l (2l+1) wl = SUM_j,k SUM_l SUM_m Y_lm^*(j) Y_lm(k)
# = SUM_j,k delta(n_j, n_k) [if infinite l sum]
# = Nskew [only j=k terms if all positions distinct]

# Hmm, but the l sum is truncated at Nl.
# For perfect completeness: SUM_l=0^inf (2l+1) wl = Nskew (for unit weights)
# But we only have Nl=500 terms.

# Let me just compute SUM (2l+1) wl:
wl_ref = d['wl_k'][0, :Nl]
total_wl = np.sum((2*ells+1) * wl_ref) / (4*np.pi)
print(f"\nSUM (2l+1) wl / (4pi) = {total_wl:.4e}")
print(f"Nskew^2 = {Nskew_actual**2:.4e}")
print(f"Ratio = {total_wl / Nskew_actual**2:.4f}")

# And: SUM_j w_j^2 vs Nskew * <w^2>:
print(f"\nNskew * <w^2> = {Nskew_actual * var_w_theory_val:.4e}")
