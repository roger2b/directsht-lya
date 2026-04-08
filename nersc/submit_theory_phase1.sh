#!/bin/bash
#SBATCH --job-name=sfb-theory-p1
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/theory_p1_%j.out
#SBATCH --error=logs/theory_p1_%j.err

# ============================================================================
# Post-processing: theory + plots for Phase 1 high-res results.
#
# If sims were run as array jobs, first combine them:
#   python nersc/combine_array_sims.py --indir results_phase1_hires \
#       --outdir results_phase1_hires --prefix Cell_multik
#
# Then submit this script.
#
# Usage:
#   sbatch nersc/submit_theory_phase1.sh
# ============================================================================

set -euo pipefail

CODEDIR=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}
OUTDIR=${CODEDIR}/results_phase1_hires

module load python
source activate desi

cd "${CODEDIR}"

# --- Find the combined simulation file ---
SIMFILE=$(ls -1t ${OUTDIR}/Cell_multik_*.npz 2>/dev/null | grep -v theory | head -1)
if [[ -z "${SIMFILE}" ]]; then
    echo "ERROR: No simulation .npz found in ${OUTDIR}"
    exit 1
fi
echo "Using simulation file: ${SIMFILE}"

# --- Theory (Nl_large=2500) ---
python compute_theory_multik.py \
    --simfile "${SIMFILE}" \
    --Nl_large 2500 \
    --NperBin 48 \
    --outdir "${OUTDIR}"

echo "Done: $(date)"
