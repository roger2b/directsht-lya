# Spherical Fourier-Bessel Clustering Analysis of the Lyman-α Forest

## Master Planning Document

**Project:** Extend `directsht` (galaxy harmonic analysis) → `directsht-lya` (Ly-α forest sFB analysis)  
**Goal:** Compute $C_\ell(k)$ for the Ly-α forest flux fluctuations using the spherical Fourier-Bessel framework  
**Repos:**
- Galaxy code: https://github.com/martinjameswhite/directsht/tree/main
- Ly-α fork: https://github.com/roger2b/directsht-lya/  (which is in the folder directsht-lya)
**Key papers:**
- Baleato Lizancos & White 2024 (arXiv:2312.12285) — galaxy harmonic analysis  
- de Belsunce, Baleato Lizancos & White (attached PDF) — Ly-α sFB theory  
- Wolz, Alonso & Nicola 2024 (arXiv:2407.21013) — catalog-based pseudo-$C_\ell$ with noise floor  

---

## Table of Contents

1. [Theoretical Foundation: Galaxies → Ly-α Forest](#1-theoretical-foundation)
2. [The sFB Estimator Equations](#2-the-sfb-estimator-equations)
3. [Noise Floor and Catalog-Based Corrections](#3-noise-floor)
4. [Codebase Architecture](#4-codebase-architecture)
5. [GRF Simulation Setup](#5-grf-simulation-setup)
6. [Agent Team Assignments](#6-agent-team-assignments)
7. [Implementation Milestones](#7-implementation-milestones)
8. [Master Notebook Outline](#8-master-notebook-outline)

---

## 1. Theoretical Foundation: Galaxies → Ly-α Forest

### 1.1 Galaxy Case (Baleato Lizancos & White 2024)

For galaxies, the pseudo-$C_\ell$ method works as follows:

**Step 1: Define the FKP-style overdensity field.** Rather than dividing data by randoms ($\delta = n_g/n_r - 1$), the FKP approach uses the *linear* difference:

$$
\bar{n}\,\delta(\hat{n}) \propto n_g(\hat{n}) - \alpha\, n_r(\hat{n})
$$

where $\alpha = \sum_i \omega_i^{(d)} / \sum_j \omega_j^{(r)}$ normalizes randoms to match data.

**Step 2: Direct SHT.** The spherical harmonic coefficients of data and (scaled) randoms are computed *directly* from point positions, without pixelization:

$$
a_{\ell m}^{(d)} = \sum_i \omega_i^{(d)} Y_{\ell m}^*(\hat{n}_i), \quad
w_{\ell m} = \sum_j \omega_j^{(r)} Y_{\ell m}^*(\hat{n}_j)
$$

with normalization $w_{00} = a_{00}^{(d)}$.

**Step 3: Pseudo-$C_\ell$.** Square the difference and average over $m$:

$$
\hat{C}_\ell = \frac{1}{2\ell+1} \sum_m |a_{\ell m}^{(d)} - w_{\ell m}|^2
$$

**Step 4: Mode coupling.** The expectation value relates to the true spectrum via the mode-coupling matrix:

$$
\langle \hat{C}_\ell \rangle = \sum_{\ell'} M_{\ell\ell'} C_{\ell'}, \quad
M_{\ell\ell'} = \frac{2\ell'+1}{4\pi} \sum_\lambda (2\lambda+1) \begin{pmatrix} \ell & \ell' & \lambda \\ 0 & 0 & 0 \end{pmatrix}^2 W_\lambda
$$

where $W_\lambda = \frac{1}{2\lambda+1}\sum_m |w_{\lambda m}|^2$ is the window power spectrum.

**Step 5: Bandpower decoupling.** Bin $\hat{C}_\ell$ into bandpowers, compute the binned coupling matrix $M_{bb'}$, invert, and report the decoupled spectrum $\tilde{C}_b$.

### 1.2 Ly-α Forest Extension: Adding the Line-of-Sight

The **key difference** for the Ly-α forest is that each quasar sightline $j$ at position $\hat{n}_j$ provides a *spectrum* $\delta_F(\chi)$ of flux fluctuations sampled at many comoving distances $\chi$ along the line-of-sight. This is in contrast to galaxies, which are point-like tracers projected onto the sky.

The Ly-α forest naturally admits a **radial-angular decomposition**: a Fourier transform along the line-of-sight (in $\chi$) combined with a spherical harmonic transform on the sky. This yields $C_\ell(k)$, the angular power spectrum as a function of line-of-sight wavenumber $k$.

**From 2D → 3D:** The galaxy case computes $C_\ell$. The Ly-α case computes $C_\ell(k)$, where the extra dimension $k$ comes from the Fourier transform along each sightline. The angular part (SHT) is identical in structure to the galaxy case; the new ingredient is the line-of-sight Fourier transform.

### 1.3 Physical Picture

Consider the 3D flux field $\delta_F(\chi\hat{n})$ observed along sparse sightlines through the IGM. We define the projected 2D field:

$$
\delta_{2\mathrm{D}}(\hat{n}; k) = \int d\chi\, K(\hat{n}, \chi; k)\, \delta_F(\chi\hat{n})
$$

where the kernel $K$ encodes both the survey mask *and* the Fourier phase factor $e^{ik\chi}$:

$$
K(\hat{n}, \chi; k) = \sum_j \delta(\hat{n} - \hat{n}_j)\, K_j(\chi)\, e^{ik\chi}
$$

Here $K_j(\chi)$ is the weight function (inverse noise variance) for sightline $j$, and the delta function restricts the field to be sampled only at quasar positions $\hat{n}_j$.

**This means:** For each $k$-mode, we first Fourier-transform each sightline to get $\delta_{2\mathrm{D}}(\hat{n}_j; k)$, then perform a *standard* direct SHT at the quasar positions with these as weights. This is exactly the galaxy `directsht` pipeline, but run once per $k$-bin.

---

## 2. The sFB Estimator Equations

### 2.1 Line-of-Sight Fourier Transform (per sightline)

For each quasar sightline $j$, compute the Fourier-weighted flux:

$$
\delta_{2\mathrm{D}}(\hat{n}_j; k) = \sum_\alpha K_j(\chi_\alpha)\, \delta_F(\chi_\alpha\hat{n}_j)\, e^{ik\chi_\alpha}\, \Delta\chi
$$

and the Fourier-weighted window:

$$
\tilde{K}_j(k) = \sum_\alpha K_j(\chi_\alpha)\, e^{ik\chi_\alpha}\, \Delta\chi
$$

where $\alpha$ indexes the pixels along sightline $j$.

**Implementation note:** Since DESI quasar spectra share a common wavelength grid, the matrix $e^{ik\chi_\alpha}$ can be computed once and applied to all sightlines. This is either a matrix-vector multiply or, if the grid is uniform, an FFT (though masked pixels argue for the explicit sum to avoid aliasing).

### 2.2 Angular Pseudo-Power Spectrum (per $k$-bin)

For each $k$-mode, use $\delta_{2\mathrm{D}}(\hat{n}_j; k)$ as weights and perform a direct SHT:

$$
a_{\ell m}^{(f)}(k) = \sum_j \delta_{2\mathrm{D}}(\hat{n}_j; k)\, Y_{\ell m}^*(\hat{n}_j)
$$

Similarly, for the window (randoms):

$$
w_{\ell m}(k) = \sum_j \tilde{K}_j(k)\, Y_{\ell m}^*(\hat{n}_j)
$$

The pseudo-$C_\ell(k)$ is then:

$$
\hat{C}_\ell(k) = \frac{1}{2\ell+1} \sum_m |a_{\ell m}^{(f)}(k) - \mathcal{N}\, w_{\ell m}(k)|^2
$$

where $\mathcal{N}$ is a normalization ensuring the zero-mode matches (analogous to $\alpha$ in the galaxy case). **Key subtlety:** For the Ly-α forest with GRF mocks, the "data" weights are $w_j \cdot \delta_F$ and the "random" weights are just $w_j$ (unity for GRF mocks). This is fundamentally different from galaxies where both data and randoms have the same type of weights.

### 2.3 Mode-Coupling Matrix

The mode-coupling matrix for the Ly-α case has the same angular structure as the galaxy case but depends on $k$:

$$
\hat{C}_\ell(k) = \sum_L M_{\ell L}(k)\, C_L(k) + N_\ell(k)
$$

where:

$$
M_{\ell L}(k) = \frac{2L+1}{4\pi} \sum_\lambda (2\lambda+1) \begin{pmatrix} \ell & L & \lambda \\ 0 & 0 & 0 \end{pmatrix}^2 W_\lambda(k)
$$

and the window spectrum is:

$$
W_\lambda(k) = \frac{1}{2\lambda+1} \sum_m |w_{\lambda m}(k)|^2
$$

**Critical difference from galaxies:** $W_\lambda(k)$ depends on $k$ because the Fourier weights $\tilde{K}_j(k)$ vary with $k$. This means the mode-coupling matrix must be recomputed for each $k$-bin. However, the 3$j$ symbols are $k$-independent and can be cached.

### 2.4 Theory Prediction (Flat-Sky Limit)

In the flat-sky limit ($L \gg 1$), the theory angular power spectrum relates to the 3D power spectrum as:

$$
C_L(k) \approx \frac{1}{\bar{\chi}^2} \int \frac{dk'}{2\pi}\, P\!\left(k_\perp = L/\bar{\chi},\, k_\parallel = k'\right)\, W(k - k')
$$

where $\bar{\chi}$ is the mean comoving distance, $W(k-k')$ is the line-of-sight window function, and $P(k_\perp, k_\parallel)$ is the anisotropic 3D power spectrum.

**For the periodic box (GRF test):** The window is a sinc function (Fourier transform of the top-hat), and the theory simplifies considerably because there is no survey geometry.

### 2.5 Periodic Case vs. Survey Geometry

**Periodic case (Phase 1):**
- Box of size $L_\text{box}$ with periodic boundary conditions
- Sightlines sample the full box uniformly
- Window function $K_j(\chi)$ = top-hat over the box length
- $\tilde{K}_j(k)$ = sinc-like function
- No Alcock-Paczyński effect
- Theory: $C_\ell(k) = P(k_\perp = \ell/\bar{\chi}, k_\parallel = k) / \bar{\chi}^2$ convolved with the $k$-space window

**Non-periodic case (Phase 2):**
- Realistic survey footprint with edges and varying depth
- Non-trivial selection function
- Window function $K_j(\chi)$ includes inverse-variance weighting
- Requires full mode-coupling matrix treatment
- Must marginalize/deproject problematic modes ($k_\parallel = 0$, $k_\perp = 0$)

---

## 3. Noise Floor and Catalog-Based Corrections

### 3.1 The Wolz et al. (2024) Noise Floor

For discretely-sampled fields, the mask pseudo-$C_\ell$ has a shot-noise floor $\tilde{N}^w$ that causes mode coupling between distant multipoles. Following Wolz et al. (arXiv:2407.21013):

**Mask shot noise:**
$$
\tilde{N}^w(k) = \frac{1}{4\pi} \sum_j |\tilde{K}_j(k)|^2
$$

This is the self-pair contribution to the window spectrum. For galaxies this is just $\frac{1}{4\pi}\sum_i w_i^2$. For the Ly-α forest, it acquires $k$-dependence through $\tilde{K}_j(k)$.

**Field noise:**
$$
\tilde{N}^f(k) = \frac{1}{4\pi} \sum_j |\delta_{2\mathrm{D}}(\hat{n}_j; k)|^2
$$

### 3.2 The Unbiased Estimator Recipe

Following Wolz et al., the corrected procedure is:

1. Compute $\hat{C}_\ell^w(k) = W_\ell(k)$ and $\hat{C}_\ell^f(k)$ from the SHTs.
2. Estimate $\tilde{N}^w(k)$ and $\tilde{N}^f(k)$ directly from the data.
3. Subtract: $\tilde{S}_\ell^w(k) = \hat{C}_\ell^w(k) - \tilde{N}^w(k)$ and $\tilde{S}_\ell^f(k) = \hat{C}_\ell^f(k) - \tilde{N}^f(k)$.
4. Compute the mode-coupling matrix using $\tilde{S}_\ell^w(k)$ instead of $\hat{C}_\ell^w(k)$.
5. Apply mode decoupling to $\tilde{S}_\ell^f(k)$.

### 3.3 When Is This Needed?

For GRF simulations with a dense sampling ($\sim 60$ sightlines/deg², typical DESI-like density), the shot noise floor may be subdominant. However:

- It becomes important at high $\ell$ where the signal $W_\ell$ drops.
- It is essential for sparse catalogs or when the survey geometry is complex.
- For the **periodic box test**, the noise floor may not be critical, but implementing it from the start ensures correctness when moving to real data.

**Recommendation:** Implement the noise-subtracted version from the start. Check if $\tilde{N}^w(k) / \hat{C}_\ell^w(k)$ is small for the GRF mocks; if so, verify that both versions give consistent results.

### 3.4 Self-Skewer Contributions

The "auto-spectrum" or "self-skewer" contribution corresponds to the $j = k$ (same sightline) terms in the pair sums. For galaxies, this is the shot-noise term. For the Ly-α forest, the self-skewer contribution contains additional information — it is related to the 1D power spectrum $P_{1\mathrm{D}}(k)$.

**For GRF simulations:** Include self-skewers. The theory prediction naturally includes them.

**For real data:** Self-skewers may be contaminated by continuum fitting systematics. The decision to include/exclude them should be deferred to the real-data analysis.

---

## 4. Codebase Architecture

### 4.1 Existing `directsht` (Galaxy Code)

```
directsht/
├── sht/
│   ├── sht.py           # DirectSHT class: core SHT engine
│   ├── mask_deconvolution.py  # MaskDeconvolution class: mode-coupling, bandpower decoupling
│   ├── lognormal_mocks.py     # LogNormalMocks: mock galaxy catalog generator
│   └── utils.py
├── notebooks/
│   ├── analyzing_mocks.ipynb  # KEY: full pseudo-Cℓ pipeline for galaxies
│   ├── basic_example.ipynb
│   └── ...
└── setup.py
```

**Key classes:**
- `DirectSHT(Nl, Nx)` — Computes $a_{\ell m} = \sum_i \omega_i Y_{\ell m}^*(\theta_i, \phi_i)$ using Hermite spline interpolation of $P_\ell^m(\cos\theta)$.
- `MaskDeconvolution` — Computes $W_\ell$, $M_{\ell\ell'}$, bandpower binning, and decoupling. Takes `wlm` as input.

### 4.2 Existing `directsht-lya` (Ly-α Fork)

```
directsht-lya/
├── sht/
│   ├── sht.py                # Same DirectSHT class
│   ├── mask_deconvolution.py  # Same MaskDeconvolution class
│   ├── GRF_class.py           # NEW: Gaussian random field simulations for Ly-α
│   └── ...
├── notebooks/
│   ├── lya_GRFs_directSHT_loop_26062024.ipynb  # KEY: Ly-α GRF + SHT pipeline
│   └── ...
└── setup.py
```

### 4.3 Required New/Modified Modules

| Module | Status | Description |
|--------|--------|-------------|
| `sht/GRF_class.py` | Exists, needs audit | GRF simulation: generates 3D Gaussian field, extracts sightlines, applies bias/RSD |
| `sht/lya_sfb.py` | **NEW** | Core sFB estimator: LOS Fourier transform + SHT per $k$-bin |
| `sht/mask_deconvolution_lya.py` | **NEW** or modified | $k$-dependent mode-coupling matrix, noise floor subtraction |
| `sht/theory_lya.py` | **NEW** | Theory $C_\ell(k)$ prediction from $P(k_\perp, k_\parallel)$ |
| `notebooks/master_periodic.ipynb` | **NEW** | Master notebook: periodic box from start to finish |
| `notebooks/master_survey.ipynb` | **NEW** | Master notebook: non-periodic case with survey geometry |

---

## 5. GRF Simulation Setup

### 5.1 Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Redshift | $z = 2.33$ | Central redshift of Ly-α forest |
| Linear bias | $b_1 = -0.1521$ | Flux bias parameter |
| RSD parameter | $\beta = 0.2298$ | Kaiser RSD parameter ($\beta = f/b_1$ where $f$ is the growth rate) |
| Box size | $L = 1380.0\, h^{-1}\text{Mpc}$ | Comoving box size |
| Grid cells | $N_\text{cell} = 512$ | Number of cells per dimension |
| Cosmology | Planck 2018 best-fit | $\Omega_m = 0.3111$, $\Omega_b = 0.04897$, $h = 0.6766$, $n_s = 0.9665$, $\sigma_8 = 0.8102$ |
| QSO density | $\sim 60\, \text{deg}^{-2}$ | DESI-like quasar number density |
| Fundamental mode | $k_F = 2\pi/L \approx 0.00455\, h\,\text{Mpc}^{-1}$ | |
| Nyquist frequency | $k_\text{Ny} = \pi N_\text{cell}/L \approx 1.17\, h\,\text{Mpc}^{-1}$ | |

### 5.2 The 3D Flux Power Spectrum

The Ly-α forest flux power spectrum in the Kaiser approximation is:

$$
P_F(k_\perp, k_\parallel) = b_1^2 (1 + \beta\, \mu^2)^2\, P_\text{lin}(k)
$$

where $\mu = k_\parallel / k$, $k = \sqrt{k_\perp^2 + k_\parallel^2}$, and $P_\text{lin}(k)$ is the linear matter power spectrum at $z = 2.33$.

### 5.3 GRF Generation Procedure

1. Generate a 3D Gaussian random field on a $512^3$ grid with power spectrum $P_F(k_\perp, k_\parallel)$.
2. Extract $N_s$ sightlines along one box axis (say $z$-axis), at random transverse positions.
3. For the periodic case: sightlines span the full box.
4. Apply weights $K_j(\chi) = 1$ (uniform) for the periodic case.
5. Optionally add white noise: $\sigma_c = 0.1 \times \langle |F| \rangle$ (10% RMS noise).

### 5.4 Consistency Check: GRF_class.py Audit Points

The following must be verified in `GRF_class.py`:

- [ ] Cosmological parameters match Planck 2018 ($\Omega_m$, $h$, $n_s$, $\sigma_8$ as above)
- [ ] Box size $L = 1380.0\, h^{-1}\text{Mpc}$, $N_\text{cell} = 512$
- [ ] Bias parameters: $b_1 = -0.1521$, $\beta = 0.2298$
- [ ] Kaiser formula correctly applied: $(1 + \beta\mu^2)^2$, not $(b + f\mu^2)^2$ — need to verify convention
- [ ] Redshift $z = 2.33$ used for $P_\text{lin}(k)$
- [ ] Sightline sampling: random positions, $\sim 60\,\text{deg}^{-2}$ equivalent
- [ ] Angular positions $(\theta, \phi)$ correctly computed from box transverse coordinates to sky coordinates
- [ ] Comoving distance $\chi$ correctly computed from box $z$-axis coordinate

---

## 6. Agent Team Assignments

### Team A: Theory & Equations (2 agents)

**Agent A1: Theory Derivation**
- Derive the complete sFB estimator equations from scratch
- Write down the periodic-box case explicitly
- Verify that setting $K_j(\chi) = \text{const}$ and integrating over the full box recovers the expected convolution
- Derive the theory prediction $C_\ell^\text{theory}(k)$ from $P_F(k_\perp, k_\parallel)$
- Document: "How does $C_\ell(k)$ reduce to $C_\ell$ when there is no LOS structure?"

**Agent A2: Theory Cross-Check**
- Read the attached PDF equations (2.1)–(2.20) carefully
- Compare to the galaxy equations in Baleato Lizancos & White (2024), Eqs. (2.1)–(2.9)
- Identify and document all sign/normalization convention differences
- Verify the flat-sky limit: $C_L(k) \approx P(L/\bar{\chi}, k)/\bar{\chi}^2$ 
- Cross-check normalization: what does $C_\ell(k) = 1$ correspond to physically?

**Deliverables:** `docs/theory_equations.md` — comprehensive equation reference

### Team B: Codebase Analysis (2 agents)

**Agent B1: Galaxy Code (`directsht`)**
- Read `sht/sht.py` — understand `DirectSHT.__call__` signature, return format
- Read `sht/mask_deconvolution.py` — understand `MaskDeconvolution` API:
  - How is $W_\ell$ computed?
  - How is $M_{\ell\ell'}$ computed?
  - How is bandpower decoupling done?
  - What normalization convention is used?
- Read `notebooks/analyzing_mocks.ipynb` — trace the full pipeline:
  1. Generate mock data and randoms
  2. Compute `alm_data` and `alm_rand` via `DirectSHT`
  3. Normalize randoms
  4. Compute pseudo-$C_\ell$ and window
  5. Mode-decouple or normalize
- Document the exact function calls and their arguments

**Agent B2: Ly-α Code (`directsht-lya`)**
- Read `sht/GRF_class.py` — understand:
  - How is the 3D field generated?
  - How are sightlines extracted?
  - What coordinate system is used? (box coordinates → sky coordinates?)
  - What bias model is implemented?
- Read `notebooks/lya_GRFs_directSHT_loop_26062024.ipynb`:
  - What is the current pipeline?
  - Where does it stop working / what is incomplete?
  - What normalization is used?
- Document gaps between current code and the target pipeline

**Deliverables:** `docs/codebase_analysis.md` — detailed API docs and gap analysis

### Team C: GRF Simulations (2 agents)

**Agent C1: GRF Validation**
- Audit `GRF_class.py` against the parameter table in §5.1
- Check: Is the power spectrum correctly generated in Fourier space?
- Check: Are the sightline positions correctly sampled?
- Check: Is the bias/RSD model correct? ($b_1 = -0.1521$ means flux *anti*-correlates with matter)
- Run the GRF code, extract sightlines, and verify the 1D power spectrum matches theory

**Agent C2: Coordinate System & Geometry**
- Verify the mapping from box coordinates $(x, y, z)$ to angular sky coordinates $(\theta, \phi)$
- For the periodic box: what is $\bar{\chi}$? (The box center in comoving distance)
- Verify that the angular separation between sightlines matches $\ell = k_\perp \bar{\chi}$
- Check edge effects: does the box size correspond to the stated comoving extent at $z = 2.33$?

**Deliverables:** Updated `GRF_class.py`, `docs/grf_validation.md`

### Team D: sFB Estimator Implementation (2 agents)

**Agent D1: LOS Fourier Transform**
- Implement the line-of-sight Fourier transform for each sightline:
  ```python
  delta_2d[j, k] = sum_alpha K_j[alpha] * delta_F[j, alpha] * exp(i*k*chi[alpha]) * dchi
  K_tilde[j, k] = sum_alpha K_j[alpha] * exp(i*k*chi[alpha]) * dchi
  ```
- Handle the $k$-grid: fundamental mode $k_F = 2\pi/L_\text{LOS}$, up to $k_\text{max}$
- Separate real and imaginary parts for the SHT (which operates on real-valued weights)
- Decide: FFT vs. explicit matrix multiply (FFT is faster but can't handle masked pixels cleanly)

**Agent D2: Angular SHT per $k$-bin + Pseudo-$C_\ell$**
- For each $k$-bin, call `DirectSHT` with:
  - `weights_gal = Re[delta_2d[:, k]]` and `Im[delta_2d[:, k]]` separately
  - `weights_rand = Re[K_tilde[:, k]]` and `Im[K_tilde[:, k]]` separately
- Combine: $a_{\ell m}^{(f)}(k) = a_{\ell m}^{(\text{Re})}(k) + i\, a_{\ell m}^{(\text{Im})}(k)$
- Compute $\hat{C}_\ell(k) = \frac{1}{2\ell+1}\sum_m |a_{\ell m}^{(f)} - \mathcal{N}\, w_{\ell m}|^2$
- **Normalization $\mathcal{N}$:** For galaxies, $a_{00}^{(r)} = a_{00}^{(d)}$. For Ly-α, the analogous condition is to match the monopole of the random and data SHTs. Need to think about whether this is done per-$k$ or globally.

**Deliverables:** `sht/lya_sfb.py` — core sFB estimator class

### Team E: Window Function & Decoupling (2 agents)

**Agent E1: $k$-dependent Window**
- Compute $W_\lambda(k) = \frac{1}{2\lambda+1}\sum_m |w_{\lambda m}(k)|^2$ for each $k$
- Implement noise floor subtraction: $\tilde{S}_\lambda^w(k) = W_\lambda(k) - \tilde{N}^w(k)$
- Implement $M_{\ell L}(k)$ using the $\tilde{S}_\lambda^w(k)$ (reuse 3$j$ symbol cache from `MaskDeconvolution`)

**Agent E2: Theory Convolution & Comparison**
- Implement the theory prediction: $C_\ell^\text{theory}(k) = P_F(\ell/\bar{\chi}, k)/\bar{\chi}^2$
- Apply the window: $C_\ell^\text{windowed}(k) = \sum_L M_{\ell L}(k)\, C_L^\text{theory}(k)$
- Compare to the measured $\hat{C}_\ell(k)$ — this is the key validation plot

**Deliverables:** Modified `sht/mask_deconvolution_lya.py`, `sht/theory_lya.py`

### Team F: Notebook & Testing (2 agents)

**Agent F1: Master Notebook Writer**
- Structure the notebook with detailed markdown cells explaining each step
- Include: parameter setup → GRF generation → sightline extraction → LOS FT → SHT → pseudo-$C_\ell$ → window → theory comparison
- Add diagnostic plots at each step (1D power spectrum, angular positions, etc.)

**Agent F2: Debugging & Consistency Checks**
- Verify that for $k = 0$, the estimator reduces to something sensible (related to the mean flux)
- Verify that for $\ell = 0$, the estimator gives $P_{1\mathrm{D}}(k)$ (or something related)
- Cross-check against a brute-force pair-counting estimate for small $N_s$
- Check that swapping real/imaginary parts doesn't change the power spectrum

**Deliverables:** `notebooks/master_periodic.ipynb`

---

## 7. Implementation Milestones

### Phase 1: Periodic Box (Priority)

| # | Milestone | Teams | Tests |
|---|-----------|-------|-------|
| 1.1 | Audit & fix `GRF_class.py` | C1, C2 | 1D P(k) matches theory |
| 1.2 | Implement LOS Fourier transform | D1 | FT of uniform field → delta function at k=0 |
| 1.3 | Implement SHT per k-bin | D2 | For single k-mode, pseudo-Cℓ is smooth |
| 1.4 | Compute k-dependent window | E1 | Wλ(k) decays with λ for uniform sampling |
| 1.5 | Theory prediction (flat-sky) | A1, E2 | Cℓ(k) matches P(ℓ/χ̄, k)/χ̄² |
| 1.6 | Window-convolved theory vs. measured Ĉℓ(k) | E2, F1 | Agreement within statistical noise |
| 1.7 | Mode-decoupled Cℓ(k) vs. theory | E1, E2 | Clean recovery of input spectrum |
| 1.8 | Master notebook complete | F1, F2 | All cells run, all plots match |

### Phase 2: Non-Periodic (After Phase 1 passes)

| # | Milestone | Teams | Tests |
|---|-----------|-------|-------|
| 2.1 | Non-trivial survey mask | C1 | Remove periodicity, add edges |
| 2.2 | Noise floor subtraction | E1 | Nw(k) correctly estimated |
| 2.3 | Full mode-coupling with noise correction | E1, E2 | Stable Mℓℓ' at high ℓ |
| 2.4 | Add white noise | C1 | Noise bias correctly subtracted |
| 2.5 | Mode deprojection (k∥=0, k⊥=0) | A1 | Contaminated modes removed |
| 2.6 | Master notebook for survey case | F1 | End-to-end pipeline |

---

## 8. Master Notebook Outline

### `notebooks/master_periodic.ipynb`

```
# Spherical Fourier-Bessel C_ℓ(k) for the Ly-α Forest: Periodic Box Test

## 0. Setup
- Import packages, set up cosmology
- Define physical parameters (Table from §5.1)

## 1. From Galaxies to the Ly-α Forest
- **Text:** Explain the conceptual bridge
- **Key equation:** The galaxy Cℓ is a 2D angular spectrum; 
  the Ly-α Cℓ(k) adds a LOS Fourier dimension
- **Diagram:** Show the decomposition: 3D field → LOS FT → 2D fields per k → SHT per k

## 2. Generate Gaussian Random Field
- Call GRF_class to generate 3D flux field
- Plot a slice of the 3D field
- Extract sightlines, plot their positions on the sky
- Verify: compute 1D P(k) from sightlines, compare to theory

## 3. Line-of-Sight Fourier Transform
- For each sightline: compute delta_2D(n̂_j; k) and K̃_j(k)
- Plot: |delta_2D|² vs k for a few sightlines
- Plot: |K̃_j(k)|² (should be sinc² for top-hat window)

## 4. Angular SHT per k-bin
- Initialize DirectSHT
- For each k-bin: run SHT with flux weights and window weights
- Compute pseudo-Cℓ(k)
- Plot: Ĉℓ(k) as a 2D image (ℓ vs k)

## 5. Window Function
- Compute Wλ(k) for each k
- Compute noise floor Ñ^w(k) and Ñ^f(k)
- Plot: Wλ(k) vs λ for several k values
- Plot: noise floor relative to Wλ

## 6. Mode-Coupling Matrix
- Compute Mℓℓ'(k) for each k
- Plot: rows of Mℓℓ' for several (ℓ, k) values

## 7. Theory Prediction
- Compute C_L^theory(k) = P_F(L/χ̄, k) / χ̄²
- Apply window: C_ℓ^windowed(k) = Σ_L M_ℓL(k) C_L^theory(k)
- Plot: theory vs measured pseudo-Cℓ(k)

## 8. Mode Decoupling
- Bin in ℓ, compute binned coupling matrix, invert
- Plot: decoupled C̃_b(k) vs. theory C_b^theory(k)
- **This is the money plot**

## 9. Consistency Checks
- Check ℓ → 0 limit (P1D-like)
- Check k → 0 limit (projected angular spectrum)
- Check that Re/Im SHTs combine correctly
- Multiple GRF realizations: check scatter matches expected variance
```

---

## 9. Key Conventions & Pitfalls

### 9.1 Weight Conventions (Galaxy vs. Ly-α)

| Quantity | Galaxy | Ly-α Forest |
|----------|--------|-------------|
| Data weights | $\omega_i$ (FKP weights) | $K_j(\chi)\, \delta_F(\chi\hat{n}_j)\, e^{ik\chi}$ |
| Random weights | $\alpha\, \omega_i^{(r)}$ | $K_j(\chi)\, e^{ik\chi}$ |
| Shot noise | $\frac{1}{4\pi}\sum_i \omega_i^2$ | $\frac{1}{4\pi}\sum_j |\tilde{K}_j(k)|^2$ |
| Overdensity | $n_g - \alpha n_r$ | $\sum_j [w_j \delta_F - w_j \cdot 0] \cdot e^{ik\chi} Y_{\ell m}^*$ |

### 9.2 Normalization Conventions

- **Galaxy code** (`analyzing_mocks.ipynb`): Normalizes $a_{00}^{(r)} = a_{00}^{(d)}$, then the mode-decoupled spectrum is properly normalized.
- **Ly-α case:** The FKP normalization for $C_\ell(k)$ should be chosen so that a shot-noise-like spectrum has $C_\ell(k) = \text{const}$. The paper proposes normalizing by $\mathcal{N}$ defined via Eq. (2.17)–(2.20), but notes issues with self-pairs. **For GRF mocks, start with the simpler normalization matching $a_{00}^{(r)} = a_{00}^{(d)}$ per $k$-bin.**

### 9.3 Real vs. Complex SHT

The `DirectSHT` class operates on real-valued weights. Since $\delta_{2\mathrm{D}}(\hat{n}_j; k)$ is complex (due to $e^{ik\chi}$), we must:

1. Run SHT on $\text{Re}[\delta_{2\mathrm{D}}(\hat{n}_j; k)]$ → get $a_{\ell m}^{(\text{Re})}$
2. Run SHT on $\text{Im}[\delta_{2\mathrm{D}}(\hat{n}_j; k)]$ → get $a_{\ell m}^{(\text{Im})}$
3. Combine: $a_{\ell m}(k) = a_{\ell m}^{(\text{Re})} + i\, a_{\ell m}^{(\text{Im})}$

Same for the window SHT. This doubles the number of SHT calls but is straightforward.

### 9.4 $k$-Grid

For a box of length $L$ along the LOS:
- Fundamental mode: $k_F = 2\pi / L$
- Maximum useful $k$: limited by pixel spacing $\Delta\chi = L/N_\text{cell}$, so $k_\text{Ny} = \pi/\Delta\chi = \pi N_\text{cell}/L$
- For $L = 1380\, h^{-1}\text{Mpc}$, $N_\text{cell} = 512$: $k_F \approx 0.00455$, $k_\text{Ny} \approx 1.17\, h\,\text{Mpc}^{-1}$
- Number of $k$-bins depends on science case; start with $\sim 50$–100 linearly spaced bins

### 9.5 $\ell$-Range

- Minimum $\ell$: set by angular extent of survey footprint. For periodic box covering the full sky equivalent, $\ell_\text{min} \sim 1$.
- Maximum $\ell$: set by mean angular separation between sightlines. For $\sim 60\,\text{deg}^{-2}$, typical separation $\sim 0.13°$, so $\ell_\text{max} \sim 180°/0.13° \approx 1400$. Start with $\ell_\text{max} = 500$–1000.

---

## 10. Environment & Workflow

### 10.1 Conda Environment

```bash
conda activate desi
```

Required packages: `numpy`, `scipy`, `healpy`, `numba`, `matplotlib`, `astropy`
Optional (GPU acceleration): `jax`, `jaxlib`

### 10.2 Git Workflow

```bash
cd directsht-lya
git checkout -b dev/sfb-estimator
# Work on feature branches, merge to dev
# Push at every meaningful checkpoint
```

### 10.3 File Organization

```
directsht-lya/
├── sht/
│   ├── sht.py                     # [UNCHANGED] DirectSHT class
│   ├── mask_deconvolution.py      # [UNCHANGED] Galaxy MaskDeconvolution
│   ├── GRF_class.py               # [AUDIT & UPDATE] Ly-α GRF simulations
│   ├── lya_sfb.py                 # [NEW] Core sFB estimator
│   ├── mask_deconvolution_lya.py  # [NEW] k-dependent mode coupling
│   └── theory_lya.py              # [NEW] Theory C_ℓ(k)
├── notebooks/
│   ├── master_periodic.ipynb      # [NEW] Phase 1 master notebook
│   ├── master_survey.ipynb        # [NEW] Phase 2 master notebook
│   └── lya_GRFs_directSHT_loop_26062024.ipynb  # [EXISTING] Reference
├── docs/
│   ├── theory_equations.md        # [NEW] Equation reference
│   └── codebase_analysis.md       # [NEW] API documentation
├── CHANGELOG.md                   # [NEW] Agent orientation file
└── README.md                      # [UPDATE] Add Ly-α description
```

---

## 11. Validation Checklist

### Phase 1: Periodic Box

- [ ] GRF 1D power spectrum matches theory $P_F(k)$ averaged over angles
- [ ] LOS Fourier transform of constant field gives delta at $k=0$
- [ ] For a single $k$-mode, angular distribution looks like a projected 2D field
- [ ] Window spectrum $W_\lambda(k)$ is concentrated at low $\lambda$ for uniform sampling
- [ ] Pseudo-$C_\ell(k)$ matches window-convolved theory
- [ ] Mode-decoupled $C_\ell(k)$ matches input $P_F(\ell/\bar{\chi}, k)/\bar{\chi}^2$
- [ ] Multiple realizations: scatter consistent with expected cosmic variance
- [ ] Code runs in < 5 minutes on a MacBook for the fiducial setup

### Phase 2: Survey Geometry

- [ ] Non-periodic mask correctly captured in window
- [ ] Noise floor subtraction stabilizes mode-coupling at high $\ell$
- [ ] White noise correctly subtracted / modeled
- [ ] Deprojection of $k_\parallel = 0$ modes removes continuum-fitting-like contamination
- [ ] Deprojection of $k_\perp = 0$ modes removes slowly-varying systematics
- [ ] Agreement between window-convolved theory and measurement

---

## 12. Session Orientation Protocol

When starting a new session:

1. **Read `CHANGELOG.md`** — see what's done, what failed, what's next
2. **Check last commit** — `git log --oneline -5`
3. **Run the master notebook** — verify current state works
4. **Pick next milestone** from §7
5. **Spawn agent teams** as described in §6
6. **Push at every checkpoint** — meaningful commits with descriptive messages
7. **Update `CHANGELOG.md`** before ending the session
