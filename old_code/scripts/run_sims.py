#!/usr/bin/env python
"""
run_sims.py — Run N GRF simulations and measure pseudo-C_ell(k=0).

Configurable by L_box, N_cell, n_QSO (per deg^2), chi_shift, N_sims, etc.
Outputs a single .npz file with all pseudo-C_ell's and the angular window.

Usage:
    python run_sims.py --Nsims 100 --Lbox 1380 --Ncell 512 --nqso 60 \
                       --chi_shift 5000 --Nl 500 --outdir results

For NERSC:
    srun -n 1 -c 64 python run_sims.py --Nsims 200 --Lbox 2000 --Ncell 512 ...
"""
import argparse, sys, os, gc, time, json, subprocess, tempfile
import numpy as np

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ly-alpha pseudo-Cl GRF simulations")
parser.add_argument("--Nsims",     type=int,   default=100,  help="Number of GRF realisations")
parser.add_argument("--Lbox",      type=float, default=1380.0, help="Box side length [Mpc/h]")
parser.add_argument("--Ncell",     type=int,   default=512,  help="Grid cells per dimension")
parser.add_argument("--nqso",      type=float, default=60.0, help="Quasar density [deg^-2]")
parser.add_argument("--chi_shift", type=float, default=5000.0, help="Comoving distance to near face [Mpc/h]")
parser.add_argument("--Nl",        type=int,   default=500,  help="Number of multipoles")
parser.add_argument("--seed0",     type=int,   default=1000, help="Starting random seed")
parser.add_argument("--outdir",    type=str,   default="results", help="Output directory")
parser.add_argument("--add_rsd",   action="store_true",      help="Include RSD (beta != 0)")
parser.add_argument("--bias",      type=float, default=-0.1521, help="Ly-alpha bias b1")
parser.add_argument("--beta",      type=float, default=0.2298, help="RSD beta (only if --add_rsd)")
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

os.makedirs(args.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
patch_area_sr  = (args.Lbox / args.chi_shift)**2           # steradians
patch_area_deg = patch_area_sr * (180.0 / np.pi)**2        # deg^2
num_qso_target = int(args.nqso * patch_area_deg)

print("=" * 72)
print("Ly-alpha pseudo-Cl simulation run")
print("=" * 72)
print(f"  Lbox       = {args.Lbox:.1f} Mpc/h")
print(f"  Ncell      = {args.Ncell}")
print(f"  chi_shift  = {args.chi_shift:.1f} Mpc/h")
print(f"  n_QSO      = {args.nqso:.1f} deg^-2")
print(f"  patch area = {patch_area_deg:.2f} deg^2")
print(f"  N_skewers  = {num_qso_target}")
print(f"  Nl         = {args.Nl}")
print(f"  N_sims     = {args.Nsims}")
print(f"  seed range = {args.seed0} .. {args.seed0 + args.Nsims - 1}")
print(f"  add_rsd    = {args.add_rsd}")
print(f"  outdir     = {args.outdir}")
print("=" * 72, flush=True)

# ---------------------------------------------------------------------------
# Write a per-sim worker script (subprocess for memory safety)
# ---------------------------------------------------------------------------
worker_code = f'''#!/usr/bin/env python
import sys, os, json
import numpy as np
sys.path.insert(0, "{root}")
sys.path.insert(0, os.path.join("{root}", "notebooks"))

from sht.sht import DirectSHT
from sht.lya_sfb import _alm2cl_complex
import GRF_class as my_GRF
import SHT_lya as sht_lya

seed       = int(sys.argv[1])
compute_wl = int(sys.argv[2])  # 1 = also compute angular window

Nl         = {args.Nl}
chi_shift  = {args.chi_shift}
num_qso    = {num_qso_target}
add_rsd    = {args.add_rsd}
my_bias    = {args.bias}
my_beta    = {args.beta}

GRF = my_GRF.PowerSpectrumGenerator(
    N={args.Ncell}, L={args.Lbox},
    add_rsd=add_rsd, my_bias=my_bias, my_beta=my_beta, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = \\
    GRF.process_skewers(Nskew=num_qso, shift=chi_shift)

chi_grid = all_x[0, :]
delta_F  = all_w_gal - 1.0

k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)

theta, phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])

sht_engine = DirectSHT(Nl, 2*Nl, 0.75)

# Pseudo-Cl from data at k=0
w_j = FT_delta[:, 0]
alm_data = sht_engine(theta, phi, w_j)
cl = _alm2cl_complex(alm_data, Nl)

out = {{"cl": cl.tolist(), "Nskew": Nskew,
       "Nk": len(chi_grid), "L": float(chi_grid[-1] - chi_grid[0] + chi_grid[1] - chi_grid[0])}}

if compute_wl:
    w_rand_k0 = FT_mask[:, 0]
    alm_rand = sht_engine(theta, phi, w_rand_k0)
    wl = _alm2cl_complex(alm_rand, Nl)
    out["wl"] = wl.tolist()
    # Also save sightline positions for chi_eff
    out["theta"] = theta.tolist()
    out["phi"]   = phi.tolist()
    out["all_x0"] = all_x[:, 0].tolist()
    out["all_y0"] = all_y[:, 0].tolist()
    out["all_z0"] = all_z[:, 0].tolist()

print("RESULT:" + json.dumps(out), flush=True)
'''

worker_file = os.path.join(args.outdir, "_worker_sim.py")
with open(worker_file, 'w') as f:
    f.write(worker_code)

# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
cl_k_all = np.zeros((args.Nsims, args.Nl))
wl_ref   = None
meta     = {}
t_total  = time.time()

for i in range(args.Nsims):
    seed = args.seed0 + i
    compute_wl_flag = 1 if (i == 0) else 0
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, worker_file, str(seed), str(compute_wl_flag)],
        capture_output=True, text=True, timeout=1200
    )

    if result.returncode != 0:
        print(f"  sim {i} (seed={seed}) FAILED:\n{result.stderr[:500]}")
        sys.exit(1)

    for line in result.stdout.strip().split('\n'):
        if line.startswith("RESULT:"):
            out = json.loads(line[7:])
            break
    else:
        print(f"  sim {i}: no RESULT line!\n{result.stdout[-500:]}")
        sys.exit(1)

    cl_k_all[i] = np.array(out["cl"])

    if i == 0:
        wl_ref = np.array(out["wl"])
        meta["Nskew"] = out["Nskew"]
        meta["Nk"]    = out["Nk"]
        meta["L"]     = out["L"]
        meta["theta"] = np.array(out["theta"])
        meta["phi"]   = np.array(out["phi"])
        meta["all_x0"] = np.array(out["all_x0"])
        meta["all_y0"] = np.array(out["all_y0"])
        meta["all_z0"] = np.array(out["all_z0"])

    dt = time.time() - t0
    elapsed = time.time() - t_total
    eta = elapsed / (i + 1) * (args.Nsims - i - 1)
    print(f"  sim {i:4d}/{args.Nsims} (seed={seed}): "
          f"cl[1]={cl_k_all[i,1]:.4e}, dt={dt:.0f}s, ETA={eta/60:.1f}min",
          flush=True)

os.remove(worker_file)
dt_total = time.time() - t_total
print(f"\nAll {args.Nsims} sims done in {dt_total/60:.1f} minutes")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
tag = (f"L{int(args.Lbox)}_N{args.Ncell}_Nq{meta['Nskew']}"
       f"_Nl{args.Nl}_sims{args.Nsims}")
outfile = os.path.join(args.outdir, f"Cell_GRF_{tag}.npz")

np.savez(outfile,
         cl_k      = cl_k_all,
         wl_k      = wl_ref[np.newaxis, :],
         Nskew     = meta["Nskew"],
         Nk        = meta["Nk"],
         L         = meta["L"],
         theta     = meta["theta"],
         phi       = meta["phi"],
         all_x0    = meta["all_x0"],
         all_y0    = meta["all_y0"],
         all_z0    = meta["all_z0"],
         chi_shift = args.chi_shift,
         Lbox      = args.Lbox,
         Ncell     = args.Ncell,
         nqso      = args.nqso,
         add_rsd   = args.add_rsd,
         bias      = args.bias,
         beta      = args.beta)

print(f"\nSaved {outfile}")
print(f"  cl_k shape: {cl_k_all.shape}")
print(f"  Nskew: {meta['Nskew']}, Nk: {meta['Nk']}, L: {meta['L']:.2f}")
