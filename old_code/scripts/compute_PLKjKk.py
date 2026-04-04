#!/usr/bin/env python
"""
Compute PLKjKk (pair-counting angular window) to lambda_max=2000
and save to disk. This takes ~240s for the Legendre sum.
"""
import sys, os, gc, time
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

chi_shift = 5000
num_qso = 9797
add_rsd_ = False
lambda_max = 2000

import GRF_class as my_GRF
import SHT_lya as sht_lya

# Load N from cache for KjKk
d = np.load(os.path.join(root, "notebooks", "data",
            "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz"))
N = int(d['Nk'])
Nskew = int(d['Nskew'])
del d

# Sightline positions (must match the cache)
GRF_pos = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=1000)
all_x, all_y, all_z, _, _, _ = GRF_pos.process_skewers(Nskew=num_qso, shift=chi_shift)
theta_ref, phi_ref = GRF_pos.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
del GRF_pos, all_x, all_y, all_z; gc.collect()

nhat = sht_lya.compute_nhat(theta_ref, phi_ref)
cos_theta = np.dot(nhat, nhat.T)
KjKk = N**2
del nhat; gc.collect()

print(f"Computing PLKjKk to lambda_max={lambda_max} "
      f"(cos_theta: {cos_theta.shape}, {cos_theta.nbytes/1e9:.2f} GB)...", flush=True)
t0 = time.time()
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta, KjKk)[:lambda_max]
dt = time.time() - t0
print(f"Done in {dt:.1f}s")

# Save
outfile = os.path.join(root, "notebooks", "data", f"PLKjKk_lambda{lambda_max}.npy")
np.save(outfile, PLKjKk)
print(f"Saved to {outfile}")

# Quick diagnostics
SN = N**2 * Nskew
print(f"\nPLKjKk[0] = {PLKjKk[0]:.4e}  (expected (N*Ns)^2 = {(N*Nskew)**2:.4e})")
for lam in [100, 200, 500, 1000, 1500, 1999]:
    print(f"PLKjKk[{lam:4d}] = {PLKjKk[lam]:.4e}  "
          f"(ratio to SN = {PLKjKk[lam]/SN:.4f})")
