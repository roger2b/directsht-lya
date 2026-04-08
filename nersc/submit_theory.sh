#!/bin/bash
#SBATCH --job-name=sfb-theory
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/theory_%j.out
#SBATCH --error=logs/theory_%j.err

# ============================================================================
# Post-processing: theory + plots for Phase 2 high-res results.
#
# If sims were run as array jobs, first combine them:
#   python nersc/combine_array_sims.py --indir results_phase2_hires --outdir results_phase2_hires
#
# Then submit this script to compute theory and make plots.
#
# Usage:
#   sbatch nersc/submit_theory.sh
# ============================================================================

set -euo pipefail

CODEDIR=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}
OUTDIR=${CODEDIR}/results_phase2_hires

module load python
source activate desi

cd "${CODEDIR}"

# --- Find the combined simulation file ---
SIMFILE=$(ls -1t ${OUTDIR}/Cell_phase2_*.npz 2>/dev/null | grep -v theory | head -1)
if [[ -z "${SIMFILE}" ]]; then
    echo "ERROR: No simulation .npz found in ${OUTDIR}"
    exit 1
fi
echo "Using simulation file: ${SIMFILE}"

# --- Theory (Nl_large=2500 for high-res) ---
python compute_theory_phase2.py \
    --simfile "${SIMFILE}" \
    --Nl_large 2500 \
    --NperBin 48 \
    --outdir "${OUTDIR}"

# --- Plots ---
THEORYFILE="${SIMFILE%.npz}_theory.npz"
python plot_phase2.py \
    --theoryfile "${THEORYFILE}" \
    --plotdir "${OUTDIR}/plots"

echo "Done: $(date)"
