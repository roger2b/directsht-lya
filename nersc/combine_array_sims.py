#!/usr/bin/env python
"""
combine_array_sims.py — Combine per-task outputs from SLURM array jobs
into a single .npz file identical to what run_sims_phase2.py produces.

Usage:
    python nersc/combine_array_sims.py \
        --indir results_phase2_hires \
        --outdir results_phase2_hires
"""
import argparse, os, glob
import numpy as np

parser = argparse.ArgumentParser(description="Combine array-job sim outputs")
parser.add_argument("--indir",  type=str, required=True)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--prefix", type=str, default="Cell_phase2",
                    help="Filename prefix (Cell_phase2 or Cell_multik)")
args = parser.parse_args()

outdir = args.outdir or args.indir
os.makedirs(outdir, exist_ok=True)

# Find per-sim directories (sim_0, sim_1, ...)
sim_dirs = sorted(glob.glob(os.path.join(args.indir, "sim_*")))
if not sim_dirs:
    print(f"No sim_* directories found in {args.indir}")
    exit(1)

print(f"Found {len(sim_dirs)} sim directories")

# Load first sim to get shapes and metadata
first_files = glob.glob(os.path.join(sim_dirs[0], f"{args.prefix}_*.npz"))
if not first_files:
    print(f"No {args.prefix}_*.npz found in {sim_dirs[0]}")
    exit(1)

d0 = np.load(first_files[0])
cl_k_0 = d0['cl_k']             # (1, Nk, Nl) for single-sim output
_, Nk, Nl = cl_k_0.shape
Nsims = len(sim_dirs)

print(f"  Nk={Nk}, Nl={Nl}, Nsims={Nsims}")

# Allocate combined arrays
cl_k_all = np.zeros((Nsims, Nk, Nl))
cl_k_all[0] = cl_k_0[0]

# Load remaining sims
for i, sdir in enumerate(sim_dirs[1:], start=1):
    files = glob.glob(os.path.join(sdir, f"{args.prefix}_*.npz"))
    if not files:
        print(f"  WARNING: no .npz in {sdir}, skipping")
        continue
    di = np.load(files[0])
    cl_k_all[i] = di['cl_k'][0]

# Build output filename
tag = (f"L{int(d0['Lbox'])}_N{int(d0['Ncell'])}_Nq{int(d0['Nskew'])}"
       f"_Nl{Nl}_Nk{Nk}_sims{Nsims}")
if float(d0.get('noise_frac', 0)) > 0:
    tag += f"_noise{float(d0['noise_frac']):.2f}"
if bool(d0.get('add_rsd', False)):
    tag += "_rsd"
if 'coverage_min' in d0:
    tag += f"_cov{float(d0['coverage_min']):.2f}"
elif 'mask_frac' in d0:
    tag += f"_mask{float(d0['mask_frac']):.2f}"
if bool(d0.get('marginalize', False)):
    tag += "_marg"

outfile = os.path.join(outdir, f"{args.prefix}_{tag}.npz")

save_dict = dict(
    cl_k=cl_k_all,
    wl_k=d0['wl_k'],
    k_par=d0['k_par'],
    Nskew=d0['Nskew'],
    Nk_pix=d0['Nk_pix'],
    L=d0['L'],
    dchi=d0['dchi'],
    sigma_c=d0.get('sigma_c', 0.0),
    noise_frac=d0.get('noise_frac', 0.0),
    theta=d0['theta'],
    phi=d0['phi'],
    all_x0=d0['all_x0'],
    all_y0=d0['all_y0'],
    all_z0=d0['all_z0'],
    chi_shift=d0['chi_shift'],
    Lbox=d0['Lbox'],
    Ncell=d0['Ncell'],
    nqso=d0['nqso'],
    add_rsd=d0['add_rsd'],
    bias=d0['bias'],
    beta=d0['beta'],
    marginalize=d0.get('marginalize', False),
    radial=d0.get('radial', False))

# Phase 2 partial coverage fields
if 'wfloor_k' in d0:
    save_dict['wfloor_k'] = d0['wfloor_k']
if 'n_pix_per_sight' in d0:
    save_dict['n_pix_per_sight'] = d0['n_pix_per_sight']
if 'coverage_min' in d0:
    save_dict['coverage_min'] = d0['coverage_min']
elif 'mask_frac' in d0:
    save_dict['mask_frac'] = d0['mask_frac']

np.savez(outfile, **save_dict)

print(f"\nSaved {outfile}")
print(f"  cl_k.shape = {cl_k_all.shape}")
