# NERSC Scripts

SLURM batch scripts for running the sFB C_ℓ(k) pipeline on NERSC Perlmutter.

## High-resolution settings

| Parameter | Value |
|---|---|
| L_box | 1380 Mpc/h |
| N_cell | 512 |
| ℓ_max | 1200 |
| N_k | 200 (k_∥ max ≈ 0.91 h/Mpc) |
| b_Lyα | −0.15 |
| β | 1.61 |
| N_sims | 100 |
| Nl_large | 2500 |

## Quick start

### Phase 1 (full sky — no mask)

```bash
# 1. Run 100 sims as a SLURM array
sbatch --array=0-99 nersc/submit_phase1.sh

# 2. Combine per-sim outputs
python nersc/combine_array_sims.py \
    --indir results_phase1_hires \
    --outdir results_phase1_hires \
    --prefix Cell_multik

# 3. Compute theory
sbatch nersc/submit_theory_phase1.sh
```

### Phase 2 (30% angular mask)

```bash
# 1. Run 100 sims as a SLURM array
sbatch --array=0-99 nersc/submit_phase2.sh

# 2. Combine per-sim outputs
python nersc/combine_array_sims.py \
    --indir results_phase2_hires \
    --outdir results_phase2_hires \
    --prefix Cell_phase2

# 3. Compute theory + plots
sbatch nersc/submit_theory.sh
```

## Noisy run

Edit `submit_phase2.sh` and set `NOISE_FRAC=0.10`, then change `OUTDIR`.

## Files

| Script | Purpose |
|---|---|
| `submit_phase1.sh` | SLURM batch: Phase 1 sims (full sky) |
| `submit_phase2.sh` | SLURM batch: Phase 2 sims (30% mask) |
| `submit_theory_phase1.sh` | Theory + deconvolution (Phase 1) |
| `submit_theory.sh` | Theory + deconvolution + plots (Phase 2) |
| `combine_array_sims.py` | Merge per-task `.npz` into single file |

## Resource estimates (per sim, Perlmutter CPU node)

| Setting | ℓ_max=500, Nk=10 | ℓ_max=1200, Nk=200 |
|---|---|---|
| Wall time | ~1 min | ~20-40 min |
| Memory | ~4 GB | ~10 GB |
| Theory (Nl_large=2500) | ~1 min | ~20 min |
