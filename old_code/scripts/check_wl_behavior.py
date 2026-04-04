#!/usr/bin/env python
"""
Check the behavior of wl (angular window) at high lambda.
For a set of point sources, wl = (1/(2l+1)) SUM_m |SUM_j Y_lm(n_j)|^2
This should NOT decay to zero for point sources — it approaches a 
white noise floor ~ N_s / (4*pi).

Also check what the MASTER sum M @ C_true converges to.
"""
import sys, os, gc
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

# Extended PLKjKk
PLKjKk = np.load('notebooks/data/PLKjKk_lambda4000.npy')
wl_ext = PLKjKk / (4*np.pi)

print("Behavior of wl = PLKjKk / (4*pi) at high lambda:")
print(f"{'lambda':>8s} {'wl':>14s} {'(2l+1)*wl':>14s} {'wl/wl[0]':>12s}")
print("-" * 55)
for lam in [0, 10, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500]:
    if lam < len(wl_ext):
        print(f"  {lam:5d}  {wl_ext[lam]:14.4e} {(2*lam+1)*wl_ext[lam]:14.4e} {wl_ext[lam]/wl_ext[0]:12.6f}")

# Check: for point sources, the white-noise floor should be
# wl ~ N_s * N^2 / (4*pi) (since u_lm = N * SUM Y_lm, and <|SUM Y_lm|^2> ~ N_s/(4pi))
# Wait: <|SUM_j Y_lm(n_j)|^2> = SUM_j |Y_lm(n_j)|^2 + SUM_{j!=k} Y_lm(n_j) Y*_lm(n_k)
# For random positions, the off-diagonal averages to zero, so
# <|SUM Y_lm|^2> ~ SUM_j |Y_lm|^2 ~ N_s * <|Y_lm|^2>_sky ~ N_s/(4pi)
# So wl ~ N^2 * N_s / (4*pi) for random mask
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
N = int(d['Nk'])
L = float(d['L'])
Nskew = int(d['Nskew'])
print(f"\nN={N}, Nskew={Nskew}")
wl_floor = N**2 * Nskew / (4*np.pi)
print(f"Expected white-noise floor: N^2 * Nskew / (4*pi) = {wl_floor:.4e}")
print(f"Actual wl at high lambda:")
print(f"  wl[2000] = {wl_ext[2000]:.4e}, ratio = {wl_ext[2000]/wl_floor:.4f}")
print(f"  wl[3000] = {wl_ext[3000]:.4e}, ratio = {wl_ext[3000]/wl_floor:.4f}")
print(f"  wl[3500] = {wl_ext[3500]:.4e}, ratio = {wl_ext[3500]/wl_floor:.4f}")

# Now check: the MASTER sum M @ C_true. If wl -> const, then 
# (M @ C_true)[l] = SUM_L (2L+1)/(4pi) SUM_lam (2lam+1) wl[lam] 3j^2 C_true[L]
# This double sum might diverge because wl doesn't decay.
# The inner sum SUM_lam (2lam+1) wl[lam] 3j^2:
#   For 3j(l,L,lam; 0,0,0), triangle rule requires |l-L| <= lam <= l+L
#   and 3j^2 ~ 1/(pi*sqrt(lam)) for large lam
#   So (2lam+1) * const * 1/(pi*sqrt(lam)) ~ 2*lam^{1/2} / pi
#   SUM from |l-L| to l+L: ~(l+L - |l-L|) * O(sqrt(l+L)) ~ O(min(l,L)^{3/2})
# This sum converges! Because the triangle rule limits the lambda range.
# 
# But the outer SUM_L also runs to Nl_large.
# C_true falls like P_lin(L/chi)/chi^2 ~ L^{n_s-4} for large L
# (2L+1)*C_true ~ L^{n_s-3} ~ L^{-2} for n_s~1
# The coupling from the 3j sums adds ~ L^{1/2} from the triangle width
# So the outer sum goes as SUM_L L^{-2} * L^{1/2} = SUM L^{-3/2}
# This converges! So the MASTER sum SHOULD converge.
#
# But numerically it doesn't seem to. Let me check what's happening.

# Check if wl grows (instead of staying flat)
print(f"\nwl growth check:")
print(f"  wl[100] = {wl_ext[100]:.4e}")
print(f"  wl[1000] = {wl_ext[1000]:.4e}")
print(f"  wl[2000] = {wl_ext[2000]:.4e}")
print(f"  wl[3500] = {wl_ext[3500]:.4e}")
print(f"  wl[100]/(2*100+1) = {wl_ext[100]/201:.4e}")
print(f"  wl[1000]/(2*1000+1) = {wl_ext[1000]/2001:.4e}")
print(f"  wl[2000]/(2*2000+1) = {wl_ext[2000]/4001:.4e}")
print(f"  wl[3500]/(2*3500+1) = {wl_ext[3500]/7001:.4e}")
