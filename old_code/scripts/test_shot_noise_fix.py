#!/usr/bin/env python
"""
Diagnose the convergence issue: show that PLKjKk has a shot-noise floor
at high L, and that subtracting it makes the theory sum converge fast.

The shot-noise floor is:
    PLKjKk_shot = N^2 * Ns  (from diagonal j=k self-pairs)

After subtraction, PLKjKk_signal = PLKjKk - SN decays at large L,
so the coupling sum converges quickly.

The shot-noise contribution is then added analytically:
    C_shot[ell] = SN * row_sum_of_coupling_pk[ell] / normalization
    
Row sum of coupling_pk[ell,:] = (1/4pi) sum_{lambda even w/ ell} (2*lambda+1) pk[lambda]
by the 3j orthogonality sum_L (2L+1) (3j(ell,L,lam))^2 = 1.
"""
import sys, os, gc, time
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

Nl = 500
chi_shift = 5000
add_rsd_ = False

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# ---- Load cache ----
datafile = os.path.join(root, "notebooks", "data",
                        "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
d = np.load(datafile)
cl_k_all = d['cl_k']
wl_k = d['wl_k']
N = int(d['Nk'])
L_box = float(d['L'])
Nskew = int(d['Nskew'])
num_sim = cl_k_all.shape[0]
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)
print(f"Loaded {num_sim} sims, N={N}, Ns={Nskew}, L_box={L_box:.1f}")

# ---- Cosmology ----
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=0)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
del GRF_tmp; gc.collect()

dchi = L_box / N
chi_bar = chi_shift + L_box / 2.0
ells = np.arange(Nl, dtype=float)

# ==================================================================
# Step 1: Check the shot-noise level in PLKjKk
# ==================================================================
print("\n=== Step 1: Shot noise in PLKjKk ===")
PLKjKk_500 = 4 * np.pi * wl_ref  # verified identity
SN_expected = N**2 * Nskew
print(f"  N^2 * Ns = {SN_expected:.4e}")
print(f"  PLKjKk[0]  = {PLKjKk_500[0]:.4e}  (should be (N*Ns)^2 = {(N*Nskew)**2:.4e})")
print(f"  PLKjKk[10] = {PLKjKk_500[10]:.4e}")
print(f"  PLKjKk[100] = {PLKjKk_500[100]:.4e}")
print(f"  PLKjKk[200] = {PLKjKk_500[200]:.4e}")
print(f"  PLKjKk[400] = {PLKjKk_500[400]:.4e}")
print(f"  PLKjKk[499] = {PLKjKk_500[499]:.4e}")
print(f"  Ratio PLKjKk[499] / SN = {PLKjKk_500[499] / SN_expected:.4f}")
print(f"  Ratio PLKjKk[400] / SN = {PLKjKk_500[400] / SN_expected:.4f}")
print(f"  Ratio PLKjKk[200] / SN = {PLKjKk_500[200] / SN_expected:.4f}")

# ==================================================================
# Step 2: MASTER with full wl vs truncated wl
# ==================================================================
print("\n=== Step 2: MASTER with truncated vs extended wl ===")

# Truncated wl (standard: wl has 500 elements, padded to 999)
couple_wl_500 = Wigner3j.CoupleMat(Nl, wl_ref)
M_500 = couple_wl_500.compute_matrix()
cl_true = b1_ref**2 * plin_ref((ells + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)
pseudo_cl_trunc = M_500 @ cl_true

# For the "full" inner sum, we need wl up to lambda = 2*Nl - 2 = 998
# We only have 500 wl values. Elements 500..998 are ZERO (padded).
# The shot noise level is wl_shot = N^2 * Ns / (4*pi)
wl_shot = SN_expected / (4 * np.pi)
print(f"  wl_shot = Ns*N^2/(4pi) = {wl_shot:.4e}")
print(f"  wl[499] = {wl_ref[499]:.4e}")
print(f"  wl[499]/wl_shot = {wl_ref[499]/wl_shot:.4f}")

# Extend wl with the shot-noise floor for lambda >= 500
wl_extended = np.zeros(2 * Nl - 1)  # 999 elements
wl_extended[:Nl] = wl_ref
wl_extended[Nl:] = wl_shot  # assume shot noise continues flat

couple_wl_ext = Wigner3j.CoupleMat(Nl, wl_extended)
M_ext = couple_wl_ext.compute_matrix()
pseudo_cl_ext = M_ext @ cl_true

print(f"\n  MASTER (truncated wl) / data: mean = {np.mean(pseudo_cl_trunc[2:]/cl_mean[2:]):.4f}")
print(f"  MASTER (extended wl)  / data: mean = {np.mean(pseudo_cl_ext[2:]/cl_mean[2:]):.4f}")

# Check specific ells
print(f"\n  {'ell':>6s} {'trunc/data':>12s} {'ext/data':>12s} {'ext/trunc':>12s}")
for ell in [10, 50, 100, 200, 300, 400]:
    rt = pseudo_cl_trunc[ell] / cl_mean[ell]
    re = pseudo_cl_ext[ell] / cl_mean[ell]
    print(f"  {ell:6d} {rt:12.6f} {re:12.6f} {re/rt:12.6f}")

# ==================================================================
# Step 3: MASTER with extended wl (signal part) + shot noise analytically
# ==================================================================
print("\n=== Step 3: Signal + shot noise decomposition ===")

# Signal part: wl_signal = wl - wl_shot (decays at large lambda)
wl_signal = wl_ref.copy()
wl_signal -= wl_shot
# For lambda >= Nl, wl_signal = 0 (no data, but also expected to be ~0)
print(f"  wl_signal[0]   = {wl_signal[0]:.4e}  (wl_ref[0] - shot)")
print(f"  wl_signal[100] = {wl_signal[100]:.4e}")
print(f"  wl_signal[200] = {wl_signal[200]:.4e}")
print(f"  wl_signal[400] = {wl_signal[400]:.4e}")
print(f"  wl_signal[499] = {wl_signal[499]:.4e}")

# MASTER with signal part only (should converge better)
couple_signal = Wigner3j.CoupleMat(Nl, wl_signal)
M_signal = couple_signal.compute_matrix()
pseudo_cl_signal = M_signal @ cl_true

# Shot noise contribution: M_shot[ell,ell'] = (2ell'+1) * wl_shot / (4pi) * 1/(2ell+1)
# (only for even ell+ell', using 3j orthogonality)
# So C_shot[ell] = SUM_{ell'} M_shot[ell,ell'] * C_true[ell']
#                = wl_shot / (4pi) * SUM_{ell': ell+ell' even} (2ell'+1) * C_true[ell'] / (2ell+1)
# But the completeness relation sum_{lambda} (2lambda+1) (3j)^2 = 1 
# is only exact when the lambda sum goes to infinity.
# With truncation at lambda=998 (enough for ell,ell' < 500), it's exact.
# Let me just compute it numerically with the shot noise wl.
wl_shot_arr = np.full(2 * Nl - 1, wl_shot)  # constant wl_shot for all lambda
couple_shot = Wigner3j.CoupleMat(Nl, wl_shot_arr)
M_shot = couple_shot.compute_matrix()
pseudo_cl_shot = M_shot @ cl_true

pseudo_cl_decomp = pseudo_cl_signal + pseudo_cl_shot

print(f"\n  Signal / data: mean = {np.mean(pseudo_cl_signal[2:]/cl_mean[2:]):.4f}")
print(f"  Shot   / data: mean = {np.mean(pseudo_cl_shot[2:]/cl_mean[2:]):.4f}")
print(f"  Total (S+SN) / data: mean = {np.mean(pseudo_cl_decomp[2:]/cl_mean[2:]):.4f}")
print(f"  Extended wl  / data: mean = {np.mean(pseudo_cl_ext[2:]/cl_mean[2:]):.4f}")

print(f"\n  {'ell':>6s} {'signal':>10s} {'shot':>10s} {'total':>10s} {'data':>10s} {'tot/data':>10s}")
for ell in [10, 50, 100, 200, 300, 400]:
    s = pseudo_cl_signal[ell]
    n = pseudo_cl_shot[ell]
    t = s + n
    d = cl_mean[ell]
    print(f"  {ell:6d} {s:10.2e} {n:10.2e} {t:10.2e} {d:10.2e} {t/d:10.4f}")

# ==================================================================
# Step 4: How much does EXTENDING the ell' sum help?
# ==================================================================
print("\n=== Step 4: Extending the ell' sum to higher Nl ===")
# The MASTER sum goes over ell' = 0..Nl-1. 
# At ell'=499, C_true is P_lin(500/chi_bar) / (32*pi^3*chi^2).
# k_perp = 500/5691 ~ 0.088 h/Mpc -> P_lin non-negligible!
k_at_499 = (499 + 0.5) / chi_bar
print(f"  k_perp at ell'=499: {k_at_499:.4f} h/Mpc")
print(f"  P_lin(k) = {plin_ref(k_at_499):.2f} (Mpc/h)^3")
print(f"  C_true[499] = {cl_true[499]:.4e}")
print(f"  C_true[0] = {cl_true[0]:.4e}")
print(f"  C_true[499]/C_true[10] = {cl_true[499]/cl_true[10]:.4f}")

# Extend C_true to higher ell'
Nl_ext = 8000  # extend past k_max*chi_bar ~ 6600, ensuring full convergence
ells_ext = np.arange(Nl_ext, dtype=float)
cl_true_ext = b1_ref**2 * plin_ref((ells_ext + 0.5) / chi_bar) / (32 * np.pi**3 * chi_bar**2)

# For the shot-noise coupling, we need M_shot[ell, ell'] for ell' up to Nl_ext.
# M_shot[ell, ell'] = wl_shot * (2ell'+1) / (4pi) * 1/(2ell+1) [analytic, for ell+ell' even]
# But computing the full matrix is expensive. Instead use the analytic formula.
# C_shot_ell = (wl_shot/(4pi)) * 1/(2ell+1) * SUM_{ell': even} (2ell'+1) cl_true_ext[ell']
sum_even = np.sum((2 * ells_ext[::2] + 1) * cl_true_ext[::2])
sum_odd = np.sum((2 * ells_ext[1::2] + 1) * cl_true_ext[1::2])
print(f"\n  Total angular variance 4pi*sigma^2 = sum (2ell+1) C_ell:")
print(f"    With Nl=500:  {np.sum((2*ells+1)*cl_true):.4e}")
print(f"    With Nl=2000: {np.sum((2*ells_ext+1)*cl_true_ext):.4e}")
print(f"    Even ell' sum: {sum_even:.4e}")
print(f"    Odd ell' sum:  {sum_odd:.4e}")

# Analytical shot noise at extended Nl
# M_shot[ell,ell'] = (2ell'+1) wl_shot/(4pi) for ALL ell,ell'
# (3j completeness: sum_lambda (2lambda+1)(3j)^2 = 1, no ell dependence)
# So C_shot = wl_shot/(4pi) * sum_{ell'} (2ell'+1) C_true[ell']  = CONSTANT
sum_total_ext = np.sum((2 * ells_ext + 1) * cl_true_ext)
pseudo_cl_shot_analytic = np.full(Nl, wl_shot / (4 * np.pi) * sum_total_ext)

# Compare with numerical shot noise at Nl=500
print(f"\n  Shot (numerical Nl=500) / Shot (analytic Nl=2000) at ell=10: "
      f"{pseudo_cl_shot[10]/pseudo_cl_shot_analytic[10]:.6f}")
print(f"  Shot (numerical Nl=500) / Shot (analytic Nl=2000) at ell=100: "
      f"{pseudo_cl_shot[100]/pseudo_cl_shot_analytic[100]:.6f}")

# ALSO need to extend the signal coupling to higher ell' 
# For the SIGNAL part: M_signal has a NARROW coupling range (because wl_signal decays fast)
# So the ell' = 0..499 sum should be well converged IF the window width is << 500.
# The window width depends on the field of view ~ L_box/chi_bar radians.
# FOV ~ 1383/5691 ~ 0.24 rad -> ell_fov ~ pi/0.24 ~ 13. That's narrow!
# But the shot noise makes the window much broader. With shot noise subtracted,
# the "signal" window should be narrow.

# Let's check by extending the signal coupling matrix
# Need CoupleMat of size Nl_ext x Nl_ext with wl_signal... too expensive.
# Instead, check if the signal part at Nl=500 is already converged.

# Total prediction: signal(Nl=500) + shot(analytic, Nl=2000)
pseudo_cl_fixed = pseudo_cl_signal + pseudo_cl_shot_analytic
print(f"\n  Signal   (Nl=500)  / data: mean = {np.mean(pseudo_cl_signal[2:]/cl_mean[2:]):.4f}")
print(f"  Shot(500)          / data: mean = {np.mean(pseudo_cl_shot[2:]/cl_mean[2:]):.4f}")
print(f"  Shot(2000,analytic)/ data: mean = {np.mean(pseudo_cl_shot_analytic[2:]/cl_mean[2:]):.4f}")
print(f"  Signal+Shot(500)   / data: mean = {np.mean(pseudo_cl_decomp[2:]/cl_mean[2:]):.4f}")
print(f"  Signal+Shot(2000)  / data: mean = {np.mean(pseudo_cl_fixed[2:]/cl_mean[2:]):.4f}")
print(f"  Extended wl only   / data: mean = {np.mean(pseudo_cl_ext[2:]/cl_mean[2:]):.4f}")

print(f"\n  {'ell':>6s} {'sig+sn500':>12s} {'sig+sn2000':>12s} {'extwl':>12s} {'data':>10s}")
for ell in [10, 50, 100, 200, 300, 400]:
    v1 = pseudo_cl_decomp[ell]
    v2 = pseudo_cl_fixed[ell]
    v3 = pseudo_cl_ext[ell]
    d = cl_mean[ell]
    print(f"  {ell:6d} {v1/d:12.6f} {v2/d:12.6f} {v3/d:12.6f} {d:10.2e}")

# ==================================================================
# Step 5: Full fix — extend signal coupling to higher ell' too
# ==================================================================
print("\n=== Step 5: Checking if signal coupling needs higher Nl ===")
# The coupling M_signal[ell, ell'] should decay for |ell - ell'| >> ell_FOV ~ 13
# So even at ell'=499, the coupling to ell=0 should be tiny.
# Check: M_signal row sums should be ~ constant (dominated by nearby ell')
print(f"  M_signal row sums:")
row_sums = np.sum(M_signal, axis=1)
print(f"    ell=10:  {row_sums[10]:.6f}")
print(f"    ell=100: {row_sums[100]:.6f}")
print(f"    ell=250: {row_sums[250]:.6f}")
print(f"    ell=400: {row_sums[400]:.6f}")
print(f"    ell=490: {row_sums[490]:.6f}")

# Check: max off-diagonal coupling range
for ell in [10, 250, 490]:
    row = M_signal[ell]
    significant = np.where(np.abs(row) > 0.001 * np.max(np.abs(row)))[0]
    print(f"    ell={ell}: significant coupling ell' in [{significant[0]}, {significant[-1]}]")

print("\nDone!")
