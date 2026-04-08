#!/bin/bash
#SBATCH --job-name=sfb-theory-rad
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/theory_radial_%j.out
#SBATCH --error=logs/theory_radial_%j.err

# ============================================================================
# Post-processing: theory + plots for radial high-res results.
#
# If sims were run as array jobs, first combine them:
#   python nersc/combine_array_sims.py \
#       --indir results_radial_hires --prefix Cell_multik
#
# Then submit this script:
#   sbatch nersc/submit_theory_radial.sh
# ============================================================================

set -euo pipefail

CODEDIR=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}
OUTDIR=${CODEDIR}/results_radial_hires

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

# --- Theory (Nl_large=3000 for high-res Nl=500) ---
python compute_theory_multik.py \
    --simfile "${SIMFILE}" \
    --Nl_large 3000 \
    --NperBin 32 \
    --outdir "${OUTDIR}"

# --- Plots ---
THEORYFILE="${SIMFILE%.npz}_theory.npz"
python plot_multik.py \
    --theoryfile "${THEORYFILE}" \
    --plotdir "${OUTDIR}/plots"

echo "Done: $(date)"
