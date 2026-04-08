#!/bin/bash
#SBATCH --job-name=sfb-phase1
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase1_%A_%a.out
#SBATCH --error=logs/phase1_%A_%a.err

# ============================================================================
# Phase 1: Full-sky sFB C_ell(k) on NERSC Perlmutter (CPU nodes)
#
# Settings: L=1380 Mpc/h, Ncell=512, lmax=1200, Nk=200 (kmax≈0.91 h/Mpc)
# No angular mask — periodic box validation.
#
# Usage:
#   # Array mode (1 sim per job, recommended):
#   sbatch --array=0-99 nersc/submit_phase1.sh
#
#   # Serial mode (all sims on one node):
#   sbatch nersc/submit_phase1.sh
#
#   # Then run theory:
#   sbatch nersc/submit_theory_phase1.sh
# ============================================================================

set -euo pipefail

# --- Configuration ---
NSIMS=100
NL=1200               # lmax
NK=220                # k_parallel modes (kmax ≈ 1.0 h/Mpc)
LBOX=1380.0
NCELL=512
NQSO=60.0
CHI_SHIFT=5000.0
NOISE_FRAC=0.0
BIAS=-0.15
BETA=1.61
SEED0=1000

CODEDIR=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}
OUTDIR=${CODEDIR}/results_phase1_hires

mkdir -p "${OUTDIR}" logs

# --- Load environment ---
module load python
source activate desi  # Adjust to your conda env name

cd "${CODEDIR}"

# --- Check if running as array job ---
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SIM_ID=${SLURM_ARRAY_TASK_ID}
    echo "Array task ${SIM_ID}: running 1 sim (seed=$((SEED0 + SIM_ID)))"

    python run_sims_multik.py \
        --Nsims 1 \
        --Nk ${NK} --Nl ${NL} \
        --Lbox ${LBOX} --Ncell ${NCELL} --nqso ${NQSO} \
        --chi_shift ${CHI_SHIFT} \
        --bias ${BIAS} --beta ${BETA} \
        --noise_frac ${NOISE_FRAC} \
        --seed0 $((SEED0 + SIM_ID)) \
        --add_rsd \
        --outdir "${OUTDIR}/sim_${SIM_ID}"
else
    echo "Running ${NSIMS} sims serially on 1 node"

    ARGS="--Nsims ${NSIMS} --Nk ${NK} --Nl ${NL}"
    ARGS+=" --Lbox ${LBOX} --Ncell ${NCELL} --nqso ${NQSO}"
    ARGS+=" --chi_shift ${CHI_SHIFT}"
    ARGS+=" --bias ${BIAS} --beta ${BETA}"
    ARGS+=" --noise_frac ${NOISE_FRAC}"
    ARGS+=" --seed0 ${SEED0} --add_rsd"
    ARGS+=" --outdir ${OUTDIR}"

    python run_sims_multik.py ${ARGS}
fi

echo "Done: $(date)"
