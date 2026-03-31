# CHANGELOG — directsht-lya sFB Estimator

## Session: 2026-03-31

### Milestone 1.1 — GRF Audit & Fix (Team C)
- **Fixed** `notebooks/GRF_class.py`:
  - Cosmological parameters → Planck 2018 (Ω_m=0.3111, h=0.6766, n_s=0.9665)
  - Redshift → z=2.33 (was hardcoded at 2.4)
  - Bias → b1=-0.1521 (was 1.0)
  - Beta → β=0.2298 (was 1.5)
  - Added `z` as constructor parameter (no longer hardcoded)

### Milestone 1.2 — LOS Fourier Transform (Team D1)
- **Created** `sht/lya_sfb.py`:
  - `LyaSFB.compute_los_ft()` — complex LOS FT via matrix multiply
  - Handles uniform weights K_j=1 (periodic box)
  - Returns complex delta_2d and K_tilde

### Milestone 1.3 — SHT per k-bin (Team D2)
- **Created** `sht/lya_sfb.py`:
  - `LyaSFB.sht_per_k()` — Re/Im split for DirectSHT (real-valued weights)
  - `LyaSFB.pseudo_cl()` — pseudo-C_l(k) with monopole normalization
  - `LyaSFB.compute_all_cl_k()` — full pipeline loop over k-bins

### Milestone 1.4 — k-dependent Window (Team E1)
- **Created** `sht/mask_deconvolution_lya.py`:
  - `MaskDeconvolutionLya` — precomputes Wigner 3j symbols once
  - `get_M(W_l)` — mode-coupling matrix from window spectrum
  - `noise_floor_w/f()` — Wolz et al. noise floor
  - `decouple()` and `convolve_theory()` — binning and decoupling

### Milestone 1.5 — Theory Prediction (Teams A+E2)
- **Created** `sht/theory_lya.py`:
  - `P_flux()` — anisotropic Kaiser flux power spectrum
  - `theory_cl_k()` — flat-sky C_l(k) = P_F(l/chi_bar, k) / chi_bar^2

### Documentation (Teams A+B)
- **Created** `docs/theory_equations.md` — complete equation reference
- **Created** `docs/codebase_analysis.md` — DirectSHT/MaskDeconvolution API docs

### Milestone 1.6–1.8 — Master Notebook (Team F)
- **Created** `notebooks/master_periodic.ipynb`:
  - Sections 0–9: full pipeline from GRF → comparison plots
  - Includes diagnostic plots at each step
  - Consistency checks: k=0, ell=0, Re/Im swap

### Status
- Phase 1 code complete, ready for execution and validation
- Next: Run notebook, verify money plots, iterate on any mismatches
