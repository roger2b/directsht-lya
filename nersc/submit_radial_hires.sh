#!/bin/bash
#SBATCH --job-name=sfb-radial
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/radial_%A_%a.out
#SBATCH --error=logs/radial_%A_%a.err

# ============================================================================
# High-resolution radial-sightline sFB C_ell(k) on NERSC Perlmutter (CPU)
#
# Settings: L=1380 Mpc/h, Ncell=1024, lmax=500, Nk=20
#           nqso=60 deg^-2, observer at origin, box at chi(z=2.33)
#           K_j = 1/L normalization (K_tilde(k=0) = 1)
#
# Usage:
#   # Array job: 500 sims (1 per task, ~20 min each):
#   sbatch --array=0-499 nersc/submit_radial_hires.sh
#
#   # Or serial (all 500 on one node, ~7 days):
#   sbatch nersc/submit_radial_hires.sh
#
#   # After completion, combine + theory + plots:
#   python nersc/combine_array_sims.py --indir results_radial_hires --prefix Cell_multik
#   sbatch nersc/submit_theory_radial.sh
# ============================================================================

set -euo pipefail

# --- Configuration ---
NSIMS=500
NL=500                 # lmax (higher than laptop Nl=200)
NK=20                  # k_parallel modes (dk ≈ 0.00455 h/Mpc → k_max ≈ 0.091 h/Mpc)
LBOX=1380.0
NCELL=1024             # double resolution (dchi ≈ 1.35 Mpc/h vs 2.70)
NQSO=60.0
NOISE_FRAC=0.0
BIAS=-0.1521
BETA=0.2298
SEED0=2000

CODEDIR=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}
OUTDIR=${CODEDIR}/results_radial_hires

mkdir -p "${OUTDIR}" logs

# --- Load environment ---
module load python
source activate desi

cd "${CODEDIR}"

# --- Check if running as array job ---
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SIM_ID=${SLURM_ARRAY_TASK_ID}
    echo "Array task ${SIM_ID}: running 1 sim (seed=$((SEED0 + SIM_ID)))"

    python run_sims_multik.py \
        --Nsims 1 \
        --Nk ${NK} --Nl ${NL} \
        --Lbox ${LBOX} --Ncell ${NCELL} --nqso ${NQSO} \
        --bias ${BIAS} --beta ${BETA} \
        --noise_frac ${NOISE_FRAC} \
        --seed0 $((SEED0 + SIM_ID)) \
        --add_rsd --radial \
        --outdir "${OUTDIR}/sim_${SIM_ID}"
else
    echo "Running ${NSIMS} sims serially on 1 node"

    python run_sims_multik.py \
        --Nsims ${NSIMS} \
        --Nk ${NK} --Nl ${NL} \
        --Lbox ${LBOX} --Ncell ${NCELL} --nqso ${NQSO} \
        --bias ${BIAS} --beta ${BETA} \
        --noise_frac ${NOISE_FRAC} \
        --seed0 ${SEED0} \
        --add_rsd --radial \
        --outdir "${OUTDIR}"
fi

echo "Done: $(date)"
