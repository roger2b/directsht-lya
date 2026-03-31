# Codebase Analysis: directsht-lya

## 1. Galaxy Pipeline (`DirectSHT` + `MaskDeconvolution`)

### 1.1 DirectSHT API

**Constructor:** `DirectSHT(Nell, Nx, xmax=0.75, dflt_type='float64', null_unphysical=True)`
- `Nell`: Number of multipoles (ℓ from 0 to Nell-1)
- `Nx`: Number of cos(θ) grid points for interpolation
- Precomputes Legendre function tables on a grid of x = cos(θ)

**Main `__call__(t, p, w)`:**
- `t`: theta array (radians), shape `(Npoints,)`
- `p`: phi array (radians), shape `(Npoints,)`
- `w`: real-valued weights, shape `(Npoints,)`
- Returns: `alm` complex array, shape `(Nlm,)`, Healpix convention
- Index: `indx(ℓ, m) = m*(2*Nell-1-m)//2 + ℓ`

**Key constraint:** Weights must be **real-valued**. Complex weights require
two SHT calls (Re and Im parts).

**`alm2cl(alm)`:**
- Returns `Cl[ℓ] = (1/(2ℓ+1)) * [|a_{ℓ0}|² + 2*Σ_{m>0} |a_{ℓm}|²]`
- Compatible with healpy `hp.alm2cl()`

### 1.2 MaskDeconvolution API

**Constructor:** `MaskDeconvolution(Nl, W_l, precomputed_Wigner=None)`
- `Nl`: Number of multipoles
- `W_l`: Window power spectrum (1D array)
- Computes Wigner 3j symbols and mode-coupling matrix M_ℓℓ'

**Main `__call__(Cl, bins, mode='deconvolution')`:**
- `Cl`: Per-ℓ pseudo-power spectrum
- `bins`: Binning matrix (Nbin × Nell)
- `mode`: 'deconvolution' (MASTER) or 'normalization'
- Returns: `(binned_ells, decoupled_bandpowers)`

**`convolve_theory_Cls(Clt, bins, mode)`:**
- Applies window function to theory Cl
- Returns `(binned_ells, convolved_bandpowers)`

**`window_matrix(bins, mode)`:**
- Returns the window matrix W_bl such that `C_b = W_bl @ C_l^theory`

**`binning_matrix(type, start, step)`:**
- Returns a binning matrix for uniform ('linear') or sqrt binning
- `step=32` gives 32-multipole bins

### 1.3 Galaxy Pipeline Steps (from `analyzing_mocks.ipynb`)

```python
# 1. Initialize SHT
sht = DirectSHT(Nl, Nx)

# 2. Compute alm from data and randoms
alm_data = sht(theta_data, phi_data, weights_data)
alm_rand = sht(theta_rand, phi_rand, weights_rand)

# 3. Normalize: match a00
alm_rand *= alm_data[0].real / alm_rand[0].real

# 4. Compute pseudo-Cl and window
Cl_diff = hp.alm2cl(alm_data - alm_rand)
Wl = hp.alm2cl(alm_rand) - shot_noise

# 5. Mode decouple
MD = MaskDeconvolution(Nl, Wl)
bins = MD.binning_matrix('linear', start=0, step=32)
binned_ells, Cl_decoupled = MD(Cl_diff, bins)

# 6. Window-convolved theory
binned_ells, Cl_theory_conv = MD.convolve_theory_Cls(Cl_theory, bins)
```

---

## 2. Ly-α Code: `GRF_class.py`

### 2.1 PowerSpectrumGenerator API

**Location:** `notebooks/GRF_class.py`

**Constructor defaults (NEED UPDATING):**
```python
PowerSpectrumGenerator(
    h=0.6770, Omega_b=0.04904, Omega_m=0.3147,
    ns=0.96824, As=2.10732e-9, mnu=0.0,
    N=512, L=1380.0, bins=30,
    add_rsd=True, my_bias=1.0, my_beta=1.5,
    seed=1000, verbose=False
)
```

**Key methods:**
- `process_skewers(Nskew, shift)` → extracts sightlines from 3D field
- `compute_theta_phi_skewer_start(x, y, z)` → converts to sky angles
- `compute_amplitudes3d()` → generates Fourier-space amplitudes with Kaiser RSD
- `density_field()` → inverse FFT to real-space density
- `get_linear_matter_power_spectrum(z=[2.4])` → CAMB P_lin(k)

### 2.2 Coordinate Mapping (Box → Sky)

Sightlines along the simulation z-axis are extracted at transverse positions.
After extraction, coordinates are swapped:
- `all_x` ← `z_box + shift` (becomes LOS distance χ)
- `all_y` ← `y_box` (transverse)
- `all_z` ← `x_box` (transverse)

Sky angles computed from the *start* of each sightline:
- `theta = arctan2(sqrt(y² + z²), x)` — angle from LOS axis
- `phi = arctan2(y, z)` — azimuthal angle

### 2.3 Issues Found (to be fixed)

| Issue | Current | Target | Status |
|-------|---------|--------|--------|
| Cosmology | Ω_m=0.3147, h=0.6770 | Ω_m=0.3111, h=0.6766 | FIXED |
| Redshift | z=2.4 (hardcoded) | z=2.33 | FIXED |
| Bias | b1=1.0 | b1=-0.1521 | FIXED |
| Beta | β=1.5 | β=0.2298 | FIXED |
| RSD | add_rsd=False in loop script | add_rsd=True | FIXED |

---

## 3. Current Ly-α Pipeline (`lya_GRFs_directSHT_loop_26062024.py`)

### 3.1 Existing Steps

1. Generate GRF with `PowerSpectrumGenerator`
2. Extract sightlines via `process_skewers`
3. Compute (θ, φ) for sightline start positions
4. DFT along LOS via `SHT_lya.compute_dft` — **returns only real part!**
5. DirectSHT per k-mode (one at a time)
6. Compute pseudo-Cl(k) via `hp.alm2cl`
7. Legendre window multipoles via pair counting
8. Theory from flat-sky: P(ℓ/χ̄, k) / χ̄²
9. Wigner 3j mode coupling
10. Binning and comparison plots

### 3.2 Gaps for Full sFB Estimator

| Gap | Description | Solution |
|-----|-------------|----------|
| Complex FT | `compute_dft` drops imaginary part | Keep both Re and Im |
| Multi-k loop | Only processes one k at a time | Loop over k-bins |
| Normalization | No N·w_lm subtraction | Implement per-k matching |
| Noise floor | Not implemented | Add Wolz et al. correction |
| No SHT for randoms | Window computed via pair-counting | Run SHT on random weights too |
| Code structure | All in notebook/script | Modularize into `sht/lya_sfb.py` |

---

## 4. New Modules Required

### 4.1 `sht/lya_sfb.py` — Core sFB estimator
- LOS Fourier transform (complex, per-sightline)
- SHT per k-bin (Re/Im split)
- Pseudo-Cl(k) computation
- Normalization

### 4.2 `sht/mask_deconvolution_lya.py` — k-dependent mode coupling
- Compute W_λ(k) per k-bin
- Noise floor subtraction
- Mode-coupling matrix M_ℓL(k) using cached 3j symbols
- Bandpower decoupling per k-bin

### 4.3 `sht/theory_lya.py` — Theory prediction
- P_F(k_perp, k_par) from CAMB P_lin
- C_ℓ^theory(k) = P_F(ℓ/χ̄, k) / χ̄²
- Window convolution interface
