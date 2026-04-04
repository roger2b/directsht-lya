#!/usr/bin/env python
"""
run_final_test.py — Full end-to-end test of the sFB C_ℓ(k_∥) estimator.

Runs three stages:
  1. Simulation:  Generate GRFs, measure pseudo-C_ℓ(k) at multiple k_∥
  2. Theory:      Compute MASTER-convolved theory predictions + deconvolution
  3. Marginalization: Verify δ→δ−<δ>_LOS kills k_∥=0 without affecting k≠0

Default settings match the production configuration from the paper:
  - N_ℓ = 500, N_k = 10, N_skew ≈ 9600, RSD on, 20 sims
  - Optional 10% Gaussian pixel noise

Usage (quick test, ~3 min):
    python run_final_test.py --Nsims 2 --Nk 5 --Nl 200

Usage (production, ~35 min):
    python run_final_test.py --Nsims 20 --Nk 10 --Nl 500

Usage (with noise):
    python run_final_test.py --Nsims 20 --Nk 10 --Nl 500 --noise_frac 0.10
"""
import argparse, sys, os, time
import subprocess

parser = argparse.ArgumentParser(description="Full sFB C_ℓ(k) test pipeline")
parser.add_argument("--Nsims",      type=int,   default=20)
parser.add_argument("--Lbox",       type=float, default=1380.0)
parser.add_argument("--Ncell",      type=int,   default=512)
parser.add_argument("--Nskew",      type=int,   default=9800)
parser.add_argument("--chi_shift",  type=float, default=5000.0)
parser.add_argument("--Nl",         type=int,   default=500)
parser.add_argument("--Nk",         type=int,   default=10)
parser.add_argument("--Nl_large",   type=int,   default=1000)
parser.add_argument("--noise_frac", type=float, default=0.0)
parser.add_argument("--seed0",      type=int,   default=1000)
parser.add_argument("--outdir",     type=str,   default="results_final_test")
parser.add_argument("--skip_marg",  action="store_true",
                    help="Skip the marginalization test (saves time)")
args = parser.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
py = sys.executable


def run_step(label, cmd):
    """Run a subprocess step and check for errors."""
    print(f"\n{'='*72}")
    print(f"  STAGE: {label}")
    print(f"{'='*72}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print(f"\n*** FAILED: {label} (exit code {result.returncode}) ***")
        sys.exit(1)
    print(f"\n  [{label}] completed in {(time.time()-t0)/60:.1f} min")
    return result


# ================================================================== #
# Stage 1: Run simulations                                           #
# ================================================================== #
sim_cmd = [
    py, "run_sims_multik.py",
    "--Nsims", str(args.Nsims),
    "--Nk", str(args.Nk),
    "--Nl", str(args.Nl),
    "--Nskew", str(args.Nskew),
    "--chi_shift", str(args.chi_shift),
    "--Lbox", str(args.Lbox),
    "--Ncell", str(args.Ncell),
    "--seed0", str(args.seed0),
    "--outdir", args.outdir,
    "--add_rsd",
    "--bias", str(-0.1521),
    "--beta", str(0.2298),
]
if args.noise_frac > 0:
    sim_cmd += ["--noise_frac", str(args.noise_frac)]

run_step("Simulations", sim_cmd)

# Find the output file
import glob
sim_files = sorted(glob.glob(os.path.join(args.outdir, "Cell_multik_*.npz")))
sim_files = [f for f in sim_files if "theory" not in f]
if not sim_files:
    print("No simulation output found!")
    sys.exit(1)
simfile = sim_files[-1]
print(f"  Sim file: {simfile}")

# ================================================================== #
# Stage 2: Compute theory                                            #
# ================================================================== #
theory_cmd = [
    py, "compute_theory_multik.py",
    "--simfile", simfile,
    "--Nl_large", str(args.Nl_large),
    "--outdir", args.outdir,
]
run_step("Theory computation", theory_cmd)

theoryfile = simfile.replace(".npz", "_theory.npz")
print(f"  Theory file: {theoryfile}")

# ================================================================== #
# Stage 3: Generate plots                                            #
# ================================================================== #
plot_cmd = [
    py, "plot_multik.py",
    "--theoryfile", theoryfile,
    "--plotdir", args.outdir,
]
run_step("Plotting", plot_cmd)

# ================================================================== #
# Stage 4: Marginalization test (optional)                           #
# ================================================================== #
if not args.skip_marg:
    marg_cmd = [
        py, "run_marginalization_test.py",
        "--Nsims", str(min(args.Nsims, 5)),
        "--Nk", str(min(args.Nk, 5)),
        "--Nl", str(min(args.Nl, 200)),
        "--Nskew", str(args.Nskew),
        "--outdir", args.outdir,
        "--add_rsd",
    ]
    if args.noise_frac > 0:
        marg_cmd += ["--noise_frac", str(args.noise_frac)]

    run_step("Marginalization test", marg_cmd)

# ================================================================== #
# Summary                                                             #
# ================================================================== #
print(f"\n{'='*72}")
print("  ALL STAGES COMPLETE")
print(f"{'='*72}")
print(f"  Output directory: {args.outdir}/")
print(f"  Sim file:    {os.path.basename(simfile)}")
print(f"  Theory file: {os.path.basename(theoryfile)}")
print(f"  Plots:       multik_pseudo_cl.pdf, multik_ratio.pdf, multik_deconv.pdf")
if not args.skip_marg:
    marg_files = glob.glob(os.path.join(args.outdir, "marginalization_test_*.npz"))
    if marg_files:
        print(f"  Marg test:   {os.path.basename(marg_files[-1])}")
print()
print("  Open the companion notebook to explore results interactively:")
print("    notebooks/plot_results_multik.ipynb")
