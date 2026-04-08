#!/bin/bash
#SBATCH --job-name=sfb-phase2
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase2_%A_%a.out
#SBATCH --error=logs/phase2_%A_%a.err

# ============================================================================
# Phase 2: High-resolution sFB C_ell(k) on NERSC Perlmutter (CPU nodes)
# Partial radial coverage: each sightline has [COVERAGE_MIN*Npix, Npix] pixels.
#
# Settings: L=1380 Mpc/h, Ncell=512, lmax=1200, kpar_max≈0.91 h/Mpc (Nk=200)
#
# Usage:
#   # Single job (100 sims serially on one node):
#   sbatch submit_phase2.sh
#
#   # Or with SLURM array (1 sim per job, faster):
#   sbatch --array=0-99 submit_phase2.sh
#
#   # Then run theory + plots:
#   sbatch submit_theory.sh
# ============================================================================

set -euo pipefail

# --- Configuration ---
NSIMS=100
NL=1200               # lmax
NK=200                # k_parallel modes (dk ≈ 0.00455 h/Mpc → k_max ≈ 0.91 h/Mpc)
LBOX=1380.0
NCELL=512
NQSO=60.0
CHI_SHIFT=5000.0
COVERAGE_MIN=0.50
NOISE_FRAC=0.0        # Set to e.g. 0.10 for noisy run
BIAS=-0.15
BETA=1.61
SEED0=1000

CODEDIR=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}
OUTDIR=${CODEDIR}/results_phase2_hires

mkdir -p "${OUTDIR}" logs

# --- Load environment ---
module load python
source activate desi  # Adjust to your conda env name

cd "${CODEDIR}"

# --- Check if running as array job ---
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Array job: each task runs 1 sim
    SIM_ID=${SLURM_ARRAY_TASK_ID}
    echo "Array task ${SIM_ID}: running 1 sim (seed=$((SEED0 + SIM_ID)))"

    python run_sims_phase2.py \
        --Nsims 1 \
        --Nk ${NK} --Nl ${NL} \
        --Lbox ${LBOX} --Ncell ${NCELL} --nqso ${NQSO} \
        --chi_shift ${CHI_SHIFT} \
        --bias ${BIAS} --beta ${BETA} \
        --coverage_min ${COVERAGE_MIN} \
        --noise_frac ${NOISE_FRAC} \
        --seed0 $((SEED0 + SIM_ID)) \
        --add_rsd \
        --outdir "${OUTDIR}/sim_${SIM_ID}"
else
    # Single job: run all sims serially
    echo "Running ${NSIMS} sims serially on 1 node"

    ARGS="--Nsims ${NSIMS} --Nk ${NK} --Nl ${NL}"
    ARGS+=" --Lbox ${LBOX} --Ncell ${NCELL} --nqso ${NQSO}"
    ARGS+=" --chi_shift ${CHI_SHIFT}"
    ARGS+=" --bias ${BIAS} --beta ${BETA}"
    ARGS+=" --coverage_min ${COVERAGE_MIN} --noise_frac ${NOISE_FRAC}"
    ARGS+=" --seed0 ${SEED0} --add_rsd"
    ARGS+=" --outdir ${OUTDIR}"

    python run_sims_phase2.py ${ARGS}
fi

echo "Done: $(date)"
