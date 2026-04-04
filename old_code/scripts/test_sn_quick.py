"""Quick shot noise estimate without allocating a full GRF."""
import numpy as np

# From the executed notebook outputs, extract the key numbers.
# The last run (20 sims, Nl=500, Nskew~9600) stored cl_stack.
# Let me estimate from the theory instead.

# Typical values:
Nskew = 9600
N = 512  # pixels per sightline
# delta_F is O(b1 * sigma_lin) where b1 = -0.1521 and sigma_lin ~ 3 at z=2.3
# w_j = sum_n delta_F(j,n) (unnormalized DFT at k=0)
# For Gaussian field: var(w_j) = N * var(delta_F) = N * sigma^2
# sigma^2 ~ b1^2 * sigma_lin^2 / N (per pixel) so var(w_j) ~ b1^2 * sigma_lin^2
# Actually more carefully: w_j = sum_n delta(j,n), var(w_j) = N * var(delta) + correlations
# For the purpose of estimation, we just need sum(w^2):

# From the original saved data: cl[5] ~ 2e6 for Nq=9797, cl ~ 2e6 at low l
# Shot noise = sum(w^2)/(4pi), and we need sum(w^2) = Nskew * <w^2>
# <w^2> = var(w) + <w>^2 = var(w) (since <delta>=0 so <w>=0)

# Alternatively: the white noise floor should be visible in the Cl at high l.
# For SHT of discrete points with weights w_j:
# pseudo-Cl at high l -> sum_j w_j^2 / (4pi)
# This is because at high l, the angular modes are uncorrelated and
# each sightline contributes independently.

# From the notebook's ratio plot: at l~450, ratio is ~1.3
# If the true signal drops but the noise is flat, then:
#   (signal + noise) / signal = 1.3 => noise = 0.3 * signal
# At l~450, signal ~ 1e6 (from the theory), noise ~ 3e5

# But more precisely, let me compute from the measurement:
# The pseudo-Cl at high-l asymptotes to the shot noise.
# From the notebook output: cl[400] ~ ? 

# Let me just compute it analytically:
# For the unnormalized DFT at k=0: w_j = sum_n delta_F[j,n]
# <sum_j w_j^2> = sum_j sum_{n,n'} <delta[j,n] delta[j,n']>
#               = sum_j sum_n var(delta) + sum_j sum_{n!=n'} C_delta(|n-n'|)
# The second term depends on LOS correlation function.

# For practical purposes, let me find cl at large l from the existing data.
# In the executed notebook, the binned_mean was printed. Let me analyze
# what SN must be to explain the high-l excess.

# From test_selfconsistent.py output (lambda_max=1000, 20 sims):
# l~464: ratio = 1.29. l~300 ratio ~ 1.05
# So the high-l excess is ~O(20-30%).

# Shot noise for pseudo-Cl with discrete SHT:
# SN_ell = (1/(2l+1)) sum_m |sum_j w_j Y*_lm(j)|^2 - signal
# At high l (l >> 1), from adding theorem, the sum_j w_j Y_lm are approximately 
# independent across l,m, and SN -> sum_j w_j^2 / (4pi)

# For the signal part: the window-convolved theory is
#   <pseudo-Cl> = Mll @ C_true + SN
# where SN is the shot noise floor.

# For uniform sightline density: SN = sum(w^2) / (4pi)

# For OUR analysis: we should subtract SN from pseudo-Cl BEFORE mode coupling.

# This is the key formula for the Lya forest SHT estimator:
# C_l(k) = |a_lm|^2/(2l+1) - N_l
# where N_l = sum_j |w_j(k)|^2 / (4pi) is the shot noise

print("Shot noise formula for Ly-alpha SHT:")
print("  N_ell = sum_j w_j(k)^2 / (4 pi)")
print("  This is l-independent (white noise)")
print()
print("For periodic box at k=0:")
print(f"  w_j = sum_n delta_F(j,n) for j=1..{Nskew}")
print(f"  Each w_j ~ O(1-10) (fluctuating sum of {N} pixels)")
print(f"  Typical sum(w^2) ~ Nskew * O(10) ~ {Nskew * 10:.0e}")
print(f"  N_ell ~ {Nskew * 10 / (4*np.pi):.0e}")
print()
print("Expected signal (pair-counting theory) ~ 2e6 at l~100")
print(f"  Fractional shot noise ~ {Nskew * 10 / (4*np.pi) / 2e6:.1%}")
print()
print("Conclusion: shot noise is ~1% of signal at low l, but")
print("at high l where signal drops, it becomes significant.")
print("It's a flat additive bias that must be subtracted.")
