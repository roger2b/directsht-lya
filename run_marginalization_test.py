#!/usr/bin/env python
"""
run_marginalization_test.py — Demonstrate k_parallel=0 marginalization.

Continuum-fitting systematics act as additive distortions that are constant
along each line of sight, i.e. they live entirely in the k_parallel=0 mode.
The standard mitigation is to "project out" this mode by subtracting the
mean fluctuation along each LOS:

    δ_F  →  δ_F  -  <δ_F>_LOS

This is equivalent to zeroing the k_par=0 Fourier coefficient per sightline,
so:
  - C_ℓ(k=0) is destroyed (by construction),
  - C_ℓ(k≠0) is UNCHANGED (the mean subtraction is orthogonal to all k≠0
    Fourier modes on the FFT grid).

This script proves the equivalence by running a batch of simulations
WITH and WITHOUT the mean subtraction, and comparing C_ℓ(k) at each k.

Similarly, k_perp≈0 modes (the angular monopole ℓ=0) can be marginalized
by removing ℓ=0 from the measured pseudo-C_ℓ(k), e.g. excluding the first
bin or projecting out a constant angular template.  See the companion
notebook for a full discussion.

Usage:
    python run_marginalization_test.py --Nsims 5 --Nk 5 --Nl 200
"""
import argparse, sys, os, gc, time, json, subprocess
import numpy as np

parser = argparse.ArgumentParser(description="k_par=0 marginalization test")
parser.add_argument("--Nsims",      type=int,   default=5)
parser.add_argument("--Lbox",       type=float, default=1380.0)
parser.add_argument("--Ncell",      type=int,   default=512)
parser.add_argument("--Nskew",      type=int,   default=9800)
parser.add_argument("--chi_shift",  type=float, default=5000.0)
parser.add_argument("--Nl",         type=int,   default=200)
parser.add_argument("--Nk",         type=int,   default=5)
parser.add_argument("--noise_frac", type=float, default=0.0)
parser.add_argument("--seed0",      type=int,   default=2000)
parser.add_argument("--outdir",     type=str,   default="results_marginalization")
parser.add_argument("--add_rsd",    action="store_true")
parser.add_argument("--bias",       type=float, default=-0.1521)
parser.add_argument("--beta",       type=float, default=0.2298)
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))
os.makedirs(args.outdir, exist_ok=True)

num_qso_target = args.Nskew

print("=" * 72)
print("Marginalization test:  δ  vs  δ − <δ>_LOS")
print("=" * 72)
print(f"  Nsims={args.Nsims}, Nk={args.Nk}, Nl={args.Nl}")
print(f"  Nskew={num_qso_target}, noise_frac={args.noise_frac}")
print(f"  add_rsd={args.add_rsd}")
print("=" * 72, flush=True)

# ---------------------------------------------------------------------------
# Worker: runs BOTH standard and mean-subtracted in one subprocess
# ---------------------------------------------------------------------------
worker_code = f'''#!/usr/bin/env python
import sys, os, json
import numpy as np
sys.path.insert(0, "{root}")
sys.path.insert(0, os.path.join("{root}", "notebooks"))

from sht.sht import DirectSHT
from sht.lya_sfb import LyaSFB
import GRF_class as my_GRF

seed       = int(sys.argv[1])
compute_wl = int(sys.argv[2])
noise_frac = float(sys.argv[3])

Nl         = {args.Nl}
Nk         = {args.Nk}
chi_shift  = {args.chi_shift}
num_qso    = {num_qso_target}
add_rsd    = {args.add_rsd}
my_bias    = {args.bias}
my_beta    = {args.beta}

# Generate GRF
GRF = my_GRF.PowerSpectrumGenerator(
    N={args.Ncell}, L={args.Lbox},
    add_rsd=add_rsd, my_bias=my_bias, my_beta=my_beta, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = \\
    GRF.process_skewers(Nskew=num_qso, shift=chi_shift)

chi_grid = all_x[0, :]
delta_F  = all_w_gal - 1.0

# Noise
sigma_c = 0.0
if noise_frac > 0:
    sigma_c = noise_frac
    rng = np.random.default_rng(seed + 999999)
    noise = rng.normal(0, sigma_c, size=delta_F.shape)
    delta_F = delta_F + noise

theta, phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])

# Mean-subtracted version: δ → δ − <δ>_LOS
delta_F_meansub = delta_F - np.mean(delta_F, axis=1, keepdims=True)

# SHT engine
sht_engine = DirectSHT(Nl, 2*Nl, 0.75)
sfb = LyaSFB(sht_engine, Nl)
k_indices = list(range(Nk))
dchi = chi_grid[1] - chi_grid[0]
N_pix = len(chi_grid)

# --- Standard: C_ell(k) ---
k_arr, cl_k_std, wl_k, _ = sfb.compute_all_cl_k(
    theta, phi, chi_grid, delta_F, K_j=None, k_arr=None,
    k_indices=k_indices)

# --- Mean-subtracted: C_ell(k) ---
_, cl_k_msub, _, _ = sfb.compute_all_cl_k(
    theta, phi, chi_grid, delta_F_meansub, K_j=None, k_arr=None,
    k_indices=k_indices)

k_selected = k_arr[k_indices]

out = {{
    "cl_k_std":  cl_k_std.tolist(),
    "cl_k_msub": cl_k_msub.tolist(),
    "k_par":     k_selected.tolist(),
    "Nskew":     Nskew,
    "Nk_pix":    N_pix,
    "L":         float(chi_grid[-1] - chi_grid[0] + dchi),
    "dchi":      float(dchi),
    "sigma_c":   sigma_c,
}}

if compute_wl:
    out["wl_k"]   = wl_k.tolist()
    out["theta"]  = theta.tolist()
    out["phi"]    = phi.tolist()
    out["all_x0"] = all_x[:, 0].tolist()
    out["all_y0"] = all_y[:, 0].tolist()
    out["all_z0"] = all_z[:, 0].tolist()

print("RESULT:" + json.dumps(out), flush=True)
'''

worker_file = os.path.join(args.outdir, "_worker_marg.py")
with open(worker_file, 'w') as f:
    f.write(worker_code)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cl_k_std_all  = np.zeros((args.Nsims, args.Nk, args.Nl))
cl_k_msub_all = np.zeros((args.Nsims, args.Nk, args.Nl))
wl_k_ref = None
k_par = None
meta = {}
t_total = time.time()

for i in range(args.Nsims):
    seed = args.seed0 + i
    compute_wl_flag = 1 if (i == 0) else 0
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, worker_file, str(seed),
         str(compute_wl_flag), str(args.noise_frac)],
        capture_output=True, text=True, timeout=1800
    )

    if result.returncode != 0:
        print(f"  sim {i} (seed={seed}) FAILED:\n{result.stderr[:800]}")
        sys.exit(1)

    for line in result.stdout.strip().split('\n'):
        if line.startswith("RESULT:"):
            out = json.loads(line[7:])
            break
    else:
        print(f"  sim {i}: no RESULT line!\n{result.stdout[-500:]}")
        sys.exit(1)

    cl_k_std_all[i]  = np.array(out["cl_k_std"])
    cl_k_msub_all[i] = np.array(out["cl_k_msub"])

    if i == 0:
        wl_k_ref = np.array(out["wl_k"])
        k_par = np.array(out["k_par"])
        meta["Nskew"] = out["Nskew"]
        meta["Nk_pix"] = out["Nk_pix"]
        meta["L"] = out["L"]
        meta["dchi"] = out["dchi"]
        meta["sigma_c"] = out["sigma_c"]
        meta["theta"]  = np.array(out["theta"])
        meta["phi"]    = np.array(out["phi"])
        meta["all_x0"] = np.array(out["all_x0"])
        meta["all_y0"] = np.array(out["all_y0"])
        meta["all_z0"] = np.array(out["all_z0"])

    dt = time.time() - t0
    # Quick comparison at k=1
    if args.Nk > 1:
        ratio_k1 = cl_k_msub_all[i, 1, 10] / cl_k_std_all[i, 1, 10]
    else:
        ratio_k1 = np.nan
    print(f"  sim {i:3d}/{args.Nsims} (seed={seed}): "
          f"msub/std at k=1,ℓ=10: {ratio_k1:.6f}, dt={dt:.0f}s",
          flush=True)

os.remove(worker_file)
dt_total = time.time() - t_total
print(f"\nAll {args.Nsims} sims done in {dt_total/60:.1f} minutes")

# ---------------------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("Comparison: standard vs mean-subtracted C_ℓ(k)")
print(f"{'='*72}")

for ik in range(len(k_par)):
    std_mean  = np.mean(cl_k_std_all[:, ik, :], axis=0)
    msub_mean = np.mean(cl_k_msub_all[:, ik, :], axis=0)

    # Fractional difference (excluding ℓ=0)
    valid = std_mean[1:] > 0
    if np.any(valid):
        frac_diff = np.abs(msub_mean[1:][valid] / std_mean[1:][valid] - 1)
        max_diff = np.max(frac_diff)
        mean_diff = np.mean(frac_diff)
    else:
        max_diff = mean_diff = np.nan

    k_label = f"k={k_par[ik]:.5f}"
    if ik == 0:
        ratio_l0 = msub_mean[0] / std_mean[0] if std_mean[0] > 0 else 0.0
        print(f"  {k_label} (MARGINALIZED): C_ℓ=0 ratio = {ratio_l0:.2e}"
              f"  (should be ~0 for perfect removal)")
        print(f"    → k_par=0 C_ℓ(k=0) reduced by "
              f"{(1 - ratio_l0)*100:.1f}%")
    else:
        print(f"  {k_label}: max |frac diff| = {max_diff:.2e}, "
              f"mean = {mean_diff:.2e}  (should be ~machine eps)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
tag = (f"L{int(args.Lbox)}_N{args.Ncell}_Nq{meta['Nskew']}"
       f"_Nl{args.Nl}_Nk{len(k_par)}_sims{args.Nsims}")
if args.add_rsd:
    tag += "_rsd"
outfile = os.path.join(args.outdir, f"marginalization_test_{tag}.npz")

np.savez(outfile,
         cl_k_std  = cl_k_std_all,
         cl_k_msub = cl_k_msub_all,
         wl_k      = wl_k_ref,
         k_par     = k_par,
         Nskew     = meta["Nskew"],
         Nk_pix    = meta["Nk_pix"],
         L         = meta["L"],
         dchi      = meta["dchi"],
         sigma_c   = meta.get("sigma_c", 0.0),
         noise_frac = args.noise_frac,
         theta     = meta["theta"],
         phi       = meta["phi"],
         all_x0    = meta["all_x0"],
         all_y0    = meta["all_y0"],
         all_z0    = meta["all_z0"],
         chi_shift = args.chi_shift,
         Lbox      = args.Lbox,
         Ncell     = args.Ncell,
         add_rsd   = args.add_rsd,
         bias      = args.bias,
         beta      = args.beta)

print(f"\nSaved {outfile}")
print(f"  cl_k_std/msub shape: {cl_k_std_all.shape}")
