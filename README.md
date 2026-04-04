# directsht-lya

**Spherical Fourier-Bessel (sFB) power spectrum estimator for the Lyman-α forest.**

This code measures the angular power spectrum as a function of line-of-sight wavenumber, $C_\ell(k_\parallel)$, from Ly-α forest sightlines using a direct (pixelization-free) spherical harmonic transform combined with a line-of-sight Fourier transform. It extends the galaxy pseudo-$C_\ell$ estimator of [Baleato Lizancos & White (2024)](http://arxiv.org/abs/2312.12285) to the 3D Ly-α case.

**Key references:**
- Baleato Lizancos & White 2024 ([arXiv:2312.12285](http://arxiv.org/abs/2312.12285)) — Direct harmonic analysis of discrete tracers
- de Belsunce, Baleato Lizancos & White (in preparation) — Ly-α sFB theory
- Wolz, Alonso & Nicola 2024 ([arXiv:2407.21013](http://arxiv.org/abs/2407.21013)) — Catalog-based pseudo-$C_\ell$ with noise

---

## Method Overview

The estimator works in three steps for each $k_\parallel$-mode:

1. **Line-of-sight Fourier transform** — For each quasar sightline $j$ at sky position $\hat{n}_j$, compute the Fourier-weighted flux fluctuation:

$$\delta_{\mathrm{2D}}(\hat{n}_j; k) = \sum_\alpha K_j(\chi_\alpha)\, \delta_F(\chi_\alpha, \hat{n}_j)\, e^{ik\chi_\alpha}\, \Delta\chi$$

2. **Angular SHT** — Perform a direct spherical harmonic transform of the projected 2D field at quasar positions (no pixelization):

$$a_{\ell m}(k) = \sum_j \delta_{\mathrm{2D}}(\hat{n}_j; k)\, Y_{\ell m}^*(\hat{n}_j)$$

3. **Pseudo-$C_\ell(k)$** — Form the angular power spectrum at each $k$:

$$\hat{C}_\ell(k) = \frac{1}{2\ell+1} \sum_m |a_{\ell m}^{\mathrm{data}}(k) - \alpha\, w_{\ell m}(k)|^2$$

The mode-coupling induced by the angular survey mask is deconvolved via the MASTER algorithm using Wigner 3j symbols. Redshift-space distortions enter through the Kaiser factor $(1 + \beta\mu^2)^2$ in the theory prediction $C_\ell^{\mathrm{true}}(k)$.

---

## Installation

```bash
git clone https://github.com/roger2b/directsht-lya.git
cd directsht-lya
pip install -e .
```

**Requirements:** `numpy`, `scipy`, `numba`, `healpy`, `camb`

Optional: If **JAX** is available, the SHT computation is automatically GPU-accelerated. The code falls back to NumPy/Numba if JAX is not present.

---

## Repository Structure

```
directsht-lya/
├── run_sims_multik.py          # Stage 1: Run GRF sims → measure pseudo-Cℓ(k)
├── compute_theory_multik.py    # Stage 2: MASTER theory + deconvolution
├── plot_multik.py              # Stage 3: Publication-quality plots
├── run_marginalization_test.py # Test: k∥=0 marginalization demo
├── run_final_test.py           # Orchestrator: chains all stages
│
├── sht/                        # Core library
│   ├── sht.py                  #   Direct SHT (DirectSHT class)
│   ├── lya_sfb.py              #   sFB Cℓ(k) estimator (LyaSFB class)
│   ├── theory_lya.py           #   Theory Cℓ(k) with Kaiser RSD
│   ├── mask_deconvolution.py   #   MASTER mode-coupling deconvolution
│   ├── mask_deconvolution_lya.py  # k-dependent extension for Lyα
│   ├── threej000.py            #   Wigner 3j symbol computation
│   ├── mocks.py                #   Mock catalog utilities
│   └── ...                     #   Interpolation, Legendre, utilities
│
├── notebooks/
│   ├── plot_results_multik.ipynb  # Pedagogical results notebook
│   ├── GRF_class.py              # GRF simulation generator (CAMB + Kaiser)
│   ├── fast_Wigner3j.py          # Fast Wigner 3j computation
│   └── matplotlib_params_file.py # Plot style configuration
│
├── docs/                       # Theory documentation
│   ├── theory_equations.md     #   Full derivation of the estimator
│   ├── method.md               #   Method summary
│   └── notes_normalization.tex #   Normalization notes
│
├── old_code/                   # Archived development code
├── setup.py
└── main.pdf                    # Reference paper draft
```

---

## Quick Start

### Run a quick validation (2 sims, 5 k-modes, 200 multipoles)

```bash
python run_final_test.py --Nsims 2 --Nk 5 --Nl 200 --outdir results_quick
```

This chains all four stages automatically: simulations → theory → plots → marginalization test.

### Run a production measurement (20 sims, 10 k-modes, 500 multipoles, with RSD)

```bash
# Stage 1: Simulate and measure Cℓ(k)
python run_sims_multik.py \
    --Nsims 20 --Nk 10 --Nl 500 --Nskew 9800 --add_rsd \
    --outdir results_prod

# Stage 2: Compute theory + deconvolve
python compute_theory_multik.py \
    --simfile results_prod/Cell_multik_*.npz \
    --Nl_large 1000 --outdir results_prod

# Stage 3: Plot
python plot_multik.py \
    --theoryfile results_prod/*_theory.npz \
    --plotdir results_prod
```

### Run with noise

```bash
python run_sims_multik.py \
    --Nsims 20 --Nk 10 --Nl 500 --Nskew 9800 --add_rsd \
    --noise_frac 0.10 --outdir results_noisy
```

The theory pipeline automatically subtracts the noise bias when `sigma_c > 0` is detected in the simulation output.

---

## Pipeline Scripts

| Script | Purpose | Key Output |
|--------|---------|------------|
| `run_sims_multik.py` | Generate GRF Ly-α sightlines on a periodic box, measure pseudo-$C_\ell(k)$ at multiple $k_\parallel$ values | `Cell_multik_*.npz` |
| `compute_theory_multik.py` | Compute MASTER-convolved theory, binned deconvolution, noise floor correction | `*_theory.npz` |
| `plot_multik.py` | Multi-panel pseudo-$C_\ell$, ratio, and deconvolved plots (PDF + PNG) | `multik_*.pdf` |
| `run_marginalization_test.py` | Verify that $\delta \to \delta - \langle\delta\rangle_{\rm LOS}$ kills $k_\parallel=0$ exactly | `marginalization_test_*.npz` |
| `run_final_test.py` | End-to-end orchestrator (calls all four scripts in sequence) | All of the above |

---

## The `sht` Library

The core `sht/` package provides the building blocks:

- **`DirectSHT`** (`sht.py`) — Pixel-free spherical harmonic transform for point sets on the sphere. Supports JAX GPU acceleration.
- **`LyaSFB`** (`lya_sfb.py`) — The sFB $C_\ell(k)$ estimator: LOS Fourier transform → angular SHT → pseudo-power spectrum.
- **`MaskDeconvolution`** (`mask_deconvolution.py`) — MASTER mode-coupling matrix from Wigner 3j symbols; binned bandpower deconvolution.
- **`MaskDeconvolutionLya`** (`mask_deconvolution_lya.py`) — $k$-dependent extension with noise floor handling.
- **`theory_lya`** (`theory_lya.py`) — Theory $C_\ell(k)$ with anisotropic Kaiser RSD: $P_F(k) = b^2(1+\beta\mu^2)^2 P_{\rm lin}(k)$.

### Basic Usage (SHT only)

```python
from sht.sht import DirectSHT

# Create an SHT instance (Nl multipoles, Nx spline points)
sht = DirectSHT(Nl=500, Nx=1024)

# Compute alm from point positions (theta, phi in radians) and weights
alms = sht(thetas, phis, weights)
```

### Basic Usage (sFB Estimator)

```python
from sht.lya_sfb import LyaSFB

# Initialize with sightline geometry
sfb = LyaSFB(theta, phi, chi_grid, Nl=500, k_par=k_array)

# Feed in flux fluctuations (Nsightlines × Npixels)
sfb.set_data(delta_F)

# Compute Cℓ(k) for all k-modes
cl_k = sfb.compute_all_cl_k()  # shape: (Nk, Nl)
```

---

## Validation Results

With 20 GRF simulations (Lbox = 1380 Mpc/h, Ncell = 512, ~9600 sightlines, 500 multipoles, 10 k-bins, Kaiser RSD with β = 0.23):

- **Noiseless**: Mean ratio ⟨data/theory⟩ = 1.00 ± 0.02 at all 10 $k_\parallel$ values
- **Noisy (σ_c = 0.10)**: After noise bias subtraction, ratios identical to noiseless within scatter
- **Marginalization**: Subtracting the LOS mean kills $k_\parallel = 0$ exactly (ratio = $10^{-29}$) while leaving all $k_\parallel \neq 0$ modes unchanged to machine precision ($10^{-14}$)

---

## Marginalization Recipe

To marginalize out systematics that are degenerate with specific $k$ modes:

- **$k_\parallel \approx 0$** (e.g., continuum fitting): subtract the per-sightline mean $\delta \to \delta - \langle\delta\rangle_{\rm LOS}$. This is a Fourier-space projection that zeroes $k_\parallel = 0$ exactly and leaves all other modes untouched.
- **$k_\perp \approx 0$** (e.g., large-scale angular systematics): exclude the $\ell = 0$ bin, since $k_\perp = (\ell + 1/2)/\chi_{\rm eff}$.
- **General template**: for a LOS template $t(\chi)$, project $\delta \to \delta - \frac{\langle\delta \cdot t\rangle}{\langle t \cdot t\rangle} t$.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Authors

Roger de Belsunce, Andreu Baleato Lizancos & Martin White

We give several examples of how to compute alms for different sets of points,
do a pseudo-spectrum calculation for mock galaxies (generated by the LogNormalMocks
class and using the MaskDeconvolution class to handle the mode-coupling matrices
and window functions) and look at how the code performs in Jupyter notebooks within
the `notebooks` directory.  Please look there for further information.


