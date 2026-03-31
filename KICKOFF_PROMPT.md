# Claude Code Kickoff Prompt — sFB Ly-α Forest Estimator

Copy everything below the `---` line into Claude Code from within the project directory:

```bash
cd /Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya
```

---

You are working on the `directsht-lya` project — extending a galaxy spherical harmonic transform code to compute $C_\ell(k)$ for the Lyman-α forest using the spherical Fourier-Bessel (sFB) framework. 

## Orientation

1. **Read `PLANNING_SFB_LYA.md`** at the project root — this is the master planning document. Follow it precisely. It contains all theory, equations, agent team assignments, milestones, and validation checks.
2. **Read `CHANGELOG.md`** if it exists — it tracks what's done and what's next.
3. Check the last commit: `git log --oneline -5`

## Git Setup (do this FIRST, before any code changes)

```bash
# Create and switch to a dev branch
git checkout -b dev/sfb-estimator

# Verify you're on the right branch
git branch
```

Do NOT use `find /`, `find /Users`, or scan outside the project directory. The filesystem is large and these commands will hang. Use `grep -r` within `.` or relative paths only.

## Environment

- Conda env: `desi` (already configured, all deps installed)
- No JAX needed — use numpy/numba only
- Python packages available: numpy, scipy, healpy, numba, matplotlib, astropy, camb or CLASS for P(k)

## What to do

Follow **Phase 1 (Periodic Box)** from the planning document, milestones 1.1 through 1.8. Spawn agent teams as described in §6 of the planning doc. The teams are:

### Team A — Theory & Equations (spawn 2 subagents)
- A1: Derive the periodic-box sFB equations. Write `docs/theory_equations.md`.
- A2: Cross-check equations against the attached PDF (in `PLANNING_SFB_LYA.md`) and the galaxy paper. Verify conventions.

### Team B — Codebase Analysis (spawn 2 subagents)  
- B1: Read `sht/sht.py` and `sht/mask_deconvolution.py` — document the DirectSHT and MaskDeconvolution APIs. Read `notebooks/analyzing_mocks.ipynb` if present — trace the galaxy pipeline. Write `docs/codebase_analysis.md`.
- B2: Read `sht/GRF_class.py` and `notebooks/lya_GRFs_directSHT_loop_26062024.ipynb` — document the current Ly-α pipeline and identify gaps. Append findings to `docs/codebase_analysis.md`.

### Team C — GRF Simulation Audit (spawn 2 subagents)
- C1: Audit `sht/GRF_class.py` against the parameters in §5.1 of the planning doc (z=2.33, b1=-0.1521, beta=0.2298, L=1380 Mpc/h, Ncell=512, Planck 2018 cosmology). Fix any issues. Validate that the 1D power spectrum from extracted sightlines matches theory.
- C2: Verify the coordinate mapping from box (x,y,z) to sky (theta, phi, chi). Verify that chi_bar is correctly computed for z=2.33.

### Team D — sFB Estimator Core (spawn 2 subagents)
- D1: Implement the LOS Fourier transform in `sht/lya_sfb.py`. For each sightline j and k-mode: `delta_2d[j,k] = sum_alpha K_j[alpha] * delta_F[j,alpha] * exp(i*k*chi[alpha]) * dchi`. Handle real/imaginary split for DirectSHT.
- D2: Implement the SHT-per-k-bin and pseudo-Cℓ(k) computation in `sht/lya_sfb.py`. Call DirectSHT with Re and Im weights separately, combine, compute |a_lm^f - N*w_lm|^2 / (2l+1).

### Team E — Window Function & Theory (spawn 2 subagents)
- E1: Implement k-dependent window W_lambda(k), noise floor N_w(k), and mode-coupling M_ll'(k) in `sht/mask_deconvolution_lya.py`. Reuse 3j symbol code from `mask_deconvolution.py`.
- E2: Implement theory C_ℓ(k) = P_F(ℓ/chi_bar, k) / chi_bar^2 in `sht/theory_lya.py`. Apply window convolution. This produces the comparison curve.

### Team F — Master Notebook (spawn 2 subagents)
- F1: Write `notebooks/master_periodic.ipynb` following the outline in §8 of the planning doc. Detailed markdown, all steps, diagnostic plots.
- F2: Run consistency checks: k=0 limit, ℓ=0 limit, Re/Im consistency, multiple realizations.

## Execution Strategy

1. **Start with Teams A+B+C in parallel** — they are read-only / documentation tasks and GRF audit. These produce the foundation.
2. **Then Teams D+E** — implement the core estimator and window machinery. These depend on B's output (API understanding) and C's output (working GRF).
3. **Finally Team F** — assemble the master notebook. Depends on D+E.
4. **Commit at each milestone** with a descriptive message. Push to the `dev/sfb-estimator` branch.
5. **Update `CHANGELOG.md`** after each milestone.

## Critical Implementation Details (from the planning doc)

### Weight Convention (Ly-α ≠ galaxies!)
- `weights_data = K_j(chi) * delta_F(chi, n_hat_j) * exp(i*k*chi)` — the flux times the Fourier phase
- `weights_rand = K_j(chi) * exp(i*k*chi)` — just the window times the Fourier phase  
- For GRF mocks: K_j(chi) = 1 (uniform weights), so `weights_rand = exp(i*k*chi)`
- The DirectSHT takes REAL weights, so split into Re and Im parts, run SHT twice per k-bin, recombine as a_lm = a_lm_Re + i*a_lm_Im

### Key Equations (periodic box, flat-sky limit)
```
Theory:  C_ℓ(k) ≈ P_F(k_perp = ℓ/chi_bar, k_parallel = k) / chi_bar^2

Where:   P_F(k_perp, k_par) = b1^2 * (1 + beta * mu^2)^2 * P_lin(k)
         mu = k_par / sqrt(k_perp^2 + k_par^2)
         b1 = -0.1521, beta = 0.2298, z = 2.33

Measured: Ĉ_ℓ(k) = (1/(2ℓ+1)) * sum_m |a_lm^f(k) - N*w_lm(k)|^2

Window:  ⟨Ĉ_ℓ(k)⟩ = sum_L M_ℓL(k) * C_L(k)
```

### File Outputs
```
sht/lya_sfb.py                 # Core sFB class
sht/mask_deconvolution_lya.py  # k-dependent mode coupling  
sht/theory_lya.py              # Theory C_ℓ(k)
docs/theory_equations.md        # Equation reference
docs/codebase_analysis.md       # API docs and gap analysis
notebooks/master_periodic.ipynb # THE deliverable
CHANGELOG.md                    # Session state
```

## Commit Protocol
- `git add -A && git commit -m "milestone X.Y: description"` 
- `git push origin dev/sfb-estimator`
- Update CHANGELOG.md before each push

## DO NOTs
- Do NOT use JAX — numpy/numba only
- Do NOT use `find /` or scan outside `.`
- Do NOT modify `sht/sht.py` or `sht/mask_deconvolution.py` — these are the upstream galaxy code; create new files instead
- Do NOT skip the theory validation — the money plot is measured vs. window-convolved theory C_ℓ(k)

Start now. Read the planning document, set up the branch, then spawn Teams A+B+C.
