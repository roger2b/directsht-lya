#!/usr/bin/env python
"""
run_sims_multik.py — Measure pseudo-C_ell(k) at multiple k_parallel values
from GRF simulations, with optional per-pixel Gaussian noise.

Uses the full complex e^{ikχ} weighting (non-periodic LOS FT) via the
LyaSFB class, with Re/Im split SHTs.

Noise model:
    Each pixel of each sightline receives additive noise ~ N(0, σ_c²).
    σ_c = noise_frac (default 0.10 = 10% of mean flux).
    This contributes a white (ell-independent, k-independent) noise bias:
        N_ell = N_pix * σ_c² * N_skew / (4π)
    which is analytically subtractable.

Usage:
    python run_sims_multik.py --Nsims 20 --Nk 10 --Nl 200 --outdir results_multik
    python run_sims_multik.py --Nsims 20 --Nk 10 --noise_frac 0.10 --outdir results_noisy
"""
import argparse, sys, os, gc, time, json, subprocess
import numpy as np

parser = argparse.ArgumentParser(description="Multi-k Ly-alpha pseudo-Cl simulations")
parser.add_argument("--Nsims",      type=int,   default=20,    help="Number of GRF realisations")
parser.add_argument("--Lbox",       type=float, default=1380.0, help="Box side length [Mpc/h]")
parser.add_argument("--Ncell",      type=int,   default=512,   help="Grid cells per dimension")
parser.add_argument("--nqso",       type=float, default=60.0,  help="QSO density [deg^-2]")
parser.add_argument("--Nskew",      type=int,   default=None,
                    help="Override num skewers passed to process_skewers (before np.unique dedup)")
parser.add_argument("--chi_shift",  type=float, default=5000.0, help="Comoving dist to near face [Mpc/h]")
parser.add_argument("--Nl",         type=int,   default=200,   help="Number of multipoles")
parser.add_argument("--Nk",         type=int,   default=10,    help="Number of k_par bins (0..Nk-1)")
parser.add_argument("--noise_frac", type=float, default=0.0,
                    help="Pixel noise RMS relative to mean (0.10 = 10%%)")
parser.add_argument("--seed0",      type=int,   default=1000,  help="Starting random seed")
parser.add_argument("--outdir",     type=str,   default="results_multik", help="Output directory")
parser.add_argument("--add_rsd",    action="store_true",       help="Include RSD")
parser.add_argument("--bias",       type=float, default=-0.1521, help="Ly-alpha bias b1")
parser.add_argument("--beta",       type=float, default=0.2298, help="RSD beta (only if --add_rsd)")
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))
os.makedirs(args.outdir, exist_ok=True)

# Derived quantities
patch_area_sr  = (args.Lbox / args.chi_shift)**2
patch_area_deg = patch_area_sr * (180.0 / np.pi)**2
if args.Nskew is not None:
    num_qso_target = args.Nskew
else:
    num_qso_target = int(args.nqso * patch_area_deg)

print("=" * 72)
print("Multi-k Ly-alpha pseudo-Cl simulation")
print("=" * 72)
print(f"  Lbox        = {args.Lbox:.1f} Mpc/h")
print(f"  Ncell       = {args.Ncell}")
print(f"  chi_shift   = {args.chi_shift:.1f} Mpc/h")
print(f"  nqso        = {args.nqso:.1f} deg^-2  =>  Nskew ~ {num_qso_target}")
print(f"  Nl          = {args.Nl}")
print(f"  Nk          = {args.Nk} (k bins 0..{args.Nk-1})")
print(f"  noise_frac  = {args.noise_frac}")
print(f"  Nsims       = {args.Nsims}")
print(f"  add_rsd     = {args.add_rsd}")
print("=" * 72, flush=True)

# ---------------------------------------------------------------------------
# Worker script (subprocess for memory safety)
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
delta_F  = all_w_gal - 1.0   # delta_F = (rho/rho_bar) - 1

# ---- Add per-pixel Gaussian noise ----
sigma_c = 0.0
if noise_frac > 0:
    sigma_c = noise_frac  # relative to mean flux = 1
    rng = np.random.default_rng(seed + 999999)
    noise = rng.normal(0, sigma_c, size=delta_F.shape)
    delta_F = delta_F + noise

theta, phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])

# Set up SHT engine and LyaSFB
sht_engine = DirectSHT(Nl, 2*Nl, 0.75)
sfb = LyaSFB(sht_engine, Nl)

# k_par values: FFT grid (first Nk non-negative frequencies)
dchi = chi_grid[1] - chi_grid[0]
N_pix = len(chi_grid)
k_indices = list(range(Nk))  # indices into fftfreq array

# Compute C_ell(k) using explicit e^{{ikchi}} weighting
k_arr, cl_k, wl_k = sfb.compute_all_cl_k(
    theta, phi, chi_grid, delta_F,
    K_j=None,    # uniform weights (K_j = 1)
    k_arr=None,  # use FFT grid
    k_indices=k_indices)

k_selected = k_arr[k_indices]

out = {{
    "cl_k": cl_k.tolist(),      # (Nk, Nl)
    "k_par": k_selected.tolist(),
    "Nskew": Nskew,
    "Nk_pix": N_pix,
    "L": float(chi_grid[-1] - chi_grid[0] + dchi),
    "dchi": float(dchi),
    "sigma_c": sigma_c,
}}

if compute_wl:
    out["wl_k"] = wl_k.tolist()  # (Nk, Nl)
    out["theta"] = theta.tolist()
    out["phi"]   = phi.tolist()
    out["all_x0"] = all_x[:, 0].tolist()
    out["all_y0"] = all_y[:, 0].tolist()
    out["all_z0"] = all_z[:, 0].tolist()

print("RESULT:" + json.dumps(out), flush=True)
'''

worker_file = os.path.join(args.outdir, "_worker_multik.py")
with open(worker_file, 'w') as f:
    f.write(worker_code)

# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
cl_k_all = np.zeros((args.Nsims, args.Nk, args.Nl))
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

    cl_k_all[i] = np.array(out["cl_k"])  # (Nk, Nl)

    if i == 0:
        wl_k_ref = np.array(out["wl_k"])  # (Nk, Nl)
        k_par = np.array(out["k_par"])
        meta["Nskew"] = out["Nskew"]
        meta["Nk_pix"] = out["Nk_pix"]
        meta["L"] = out["L"]
        meta["dchi"] = out["dchi"]
        meta["sigma_c"] = out["sigma_c"]
        meta["theta"] = np.array(out["theta"])
        meta["phi"]   = np.array(out["phi"])
        meta["all_x0"] = np.array(out["all_x0"])
        meta["all_y0"] = np.array(out["all_y0"])
        meta["all_z0"] = np.array(out["all_z0"])

    dt = time.time() - t0
    elapsed = time.time() - t_total
    eta = elapsed / (i + 1) * (args.Nsims - i - 1)
    print(f"  sim {i:4d}/{args.Nsims} (seed={seed}): "
          f"cl[k=0,ell=1]={cl_k_all[i,0,1]:.4e}, dt={dt:.0f}s, "
          f"ETA={eta/60:.1f}min", flush=True)

os.remove(worker_file)
dt_total = time.time() - t_total
print(f"\nAll {args.Nsims} sims done in {dt_total/60:.1f} minutes")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
tag = (f"L{int(args.Lbox)}_N{args.Ncell}_Nq{meta['Nskew']}"
       f"_Nl{args.Nl}_Nk{args.Nk}_sims{args.Nsims}")
if args.noise_frac > 0:
    tag += f"_noise{args.noise_frac:.2f}"
if args.add_rsd:
    tag += "_rsd"
outfile = os.path.join(args.outdir, f"Cell_multik_{tag}.npz")

np.savez(outfile,
         cl_k      = cl_k_all,       # (Nsims, Nk, Nl)
         wl_k      = wl_k_ref,       # (Nk, Nl)
         k_par     = k_par,           # (Nk,)
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
         nqso      = args.nqso,
         add_rsd   = args.add_rsd,
         bias      = args.bias,
         beta      = args.beta)

print(f"\nSaved {outfile}")
print(f"  cl_k shape: {cl_k_all.shape}")
print(f"  k_par: {k_par[:5]} ... (Nk={len(k_par)})")
print(f"  Nskew: {meta['Nskew']}, Nk_pix: {meta['Nk_pix']}")
if args.noise_frac > 0:
    N_noise = meta["Nk_pix"] * meta["sigma_c"]**2 * meta["Nskew"] / (4 * np.pi)
    print(f"  sigma_c: {meta['sigma_c']:.4f}, noise bias N_ell: {N_noise:.4e}")
