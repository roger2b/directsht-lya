# Pseudo-$C_\ell(k)$ Estimator for the Ly-$\alpha$ Forest

## Overview

We measure the angular power spectrum $C_\ell(k_\parallel)$ of the Ly-$\alpha$
forest from sightlines placed inside a periodic Gaussian random field (GRF) box.
The pipeline has two independent parts:

1. **Measurement** (per simulation): GRF → skewers → LOS DFT → DirectSHT → pseudo-$C_\ell$
2. **Theory prediction**: Limber $C_\ell^{\rm true}$ convolved with the survey
   window via a floor-subtracted MASTER framework.

---

## 1. Measurement Pipeline

### 1.1 GRF box

A 3-D Gaussian random field is drawn on an $N_{\rm cell}^3$ grid of side
$L_{\rm box}$ (Mpc/$h$) at redshift $z = 2.33$ using Planck 2018 cosmology
(via CAMB).  Fourier amplitudes are drawn as

$$
a(\mathbf{k}) = b_1 \bigl(1 + \beta\,\mu^2\bigr)
\bigl[\mathcal{N}(0, \sqrt{P_{\rm lin}(k)/2}) + i\,\mathcal{N}(0, \sqrt{P_{\rm lin}(k)/2})\bigr],
$$

with Hermitian symmetry enforced.  The density field is obtained by inverse FFT.
Parameters: $b_1 = -0.1521$, $\beta = 0$ (RSD off for validation).

### 1.2 Sightline extraction

$N_{\rm skew}$ sightlines are placed at random transverse grid positions
$(i_y, i_z)$ (drawn once with `np.random.seed(100)`, shared across all sims).
Each sightline runs along the full $x$-axis of the box ($N_{\rm cell}$ pixels).

The box is shifted so that the near face sits at comoving distance
$\chi_0$ (= `chi_shift`, e.g. 5000 Mpc/$h$).  Each sightline $j$ has angular
coordinates $(\theta_j, \phi_j)$ computed from the position of its **first
pixel** at $(\chi_0,\, y_j,\, z_j)$.

The number of sightlines is set by a target quasar density:

$$
N_{\rm skew} = n_{\rm QSO}\;(\text{deg}^{-2}) \times \Omega_{\rm patch}\;(\text{deg}^2),
\qquad
\Omega_{\rm patch} = \left(\frac{L_{\rm box}}{\chi_0}\right)^2 \times \left(\frac{180}{\pi}\right)^2.
$$

### 1.3 Line-of-sight DFT

For each sightline $j$, an unnormalized DFT is computed:

$$
w_j(k) = \mathrm{Re}\!\left[\sum_{\alpha=0}^{N-1} \delta_F(\hat{n}_j, \chi_\alpha)\; e^{-2\pi i\, n\, \alpha/N}\right],
$$

where $k = 2\pi n / (N\Delta\chi)$.  We use `scipy.linalg.dft(N)` and keep the
**real part only** (inherited convention).  For the $k = 0$ (DC) mode, this is
simply $w_j = \sum_\alpha \delta_F$.

### 1.4 Spherical harmonic transform

The DirectSHT engine computes

$$
a_{\ell m}(k) = \sum_{j=1}^{N_{\rm skew}} w_j(k)\; Y_{\ell m}(\theta_j, \phi_j)
$$

using interpolated $P_\ell^m(\cos\theta)$ tables (no HEALPix pixelisation).

### 1.5 Pseudo-$C_\ell$

$$
\hat{C}_\ell(k) = \frac{1}{2\ell+1} \sum_{m=-\ell}^{\ell} |a_{\ell m}(k)|^2.
$$

**No shot-noise subtraction** — for Ly-$\alpha$ with fixed sightline positions,
the diagonal ($j = k$) pairs carry cosmological signal, not Poisson noise.

---

## 2. Theory Prediction (Floor-Subtracted MASTER)

### 2.1 Angular window

The SHT window function is computed from a "randoms" SHT using uniform weights
$\tilde{K}_j(k{=}0) = N_{\rm cell}$ for each sightline:

$$
W_\lambda = \frac{1}{2\lambda+1} \sum_m |u_{\lambda m}|^2,
\qquad
u_{\lambda m} = N_{\rm cell} \sum_j Y_{\lambda m}(\hat{n}_j).
$$

For point-source masks, $W_\lambda$ reaches a **white-noise floor** at high
$\lambda$:

$$
W_{\rm floor} = \frac{N_{\rm cell}^2\; N_{\rm skew}}{4\pi}.
$$

Because this floor never decays, the standard MASTER sum $\sum_{L'} M_{\ell L'} C_{L'}^{\rm true}$ does not converge.

### 2.2 Floor subtraction

We decompose the window: $W_\lambda = W_\lambda^{\rm clust} + W_{\rm floor}$,
where $W_\lambda^{\rm clust} \to 0$ at high $\lambda$.  The MASTER prediction
becomes

$$
\langle \hat{C}_\ell \rangle = \sum_{L'} M_{\ell L'}^{\rm clust}\; C_{L'}^{\rm true}
\;+\; \underbrace{\frac{W_{\rm floor}}{4\pi} \sum_{L'} (2L'+1)\; C_{L'}^{\rm true}}_{\text{floor}_{\,\ell}\;\text{(}{\ell}\text{-independent)}},
$$

where $M^{\rm clust}$ is built from $W^{\rm clust}$ only.  The floor sum is
truncated at the box Nyquist scale $k_{\rm Nyq} = \pi N_{\rm cell}/L_{\rm box}$
(i.e. $L_{\rm Nyq} = k_{\rm Nyq}\, \chi_{\rm eff}$) to ensure convergence.

### 2.3 True angular power spectrum

$$
C_\ell^{\rm true} = \frac{b_1^2\; P_{\rm lin}\bigl((\ell + \tfrac{1}{2})/\chi_{\rm eff}\bigr)}{L_{\rm box}\; \chi_{\rm eff}^2},
$$

where $\chi_{\rm eff} = \langle r_j \rangle$ is the mean sightline distance.
Because sightlines lie on a **flat plane** at $x = \chi_0$ (not on a spherical
shell), each sightline sits at distance

$$
r_j = \sqrt{\chi_0^2 + y_j^2 + z_j^2},
$$

and $\chi_{\rm eff} = \langle r_j \rangle \approx \sqrt{\chi_0^2 + 2L_{\rm box}^2/3}$.
This is distinct from $\chi_0$ (near face) or $\bar\chi = \chi_0 + L_{\rm box}/2$
(box midpoint).

### 2.4 Mode-coupling matrix

The MASTER mode-coupling matrix is

$$
M_{\ell L}^{\rm clust} = \frac{2L+1}{4\pi} \sum_{\lambda} (2\lambda+1)\; W_\lambda^{\rm clust}\;
\begin{pmatrix} \ell & L & \lambda \\ 0 & 0 & 0 \end{pmatrix}^{\!2},
$$

computed via `fast_Wigner3j.CoupleMat(Nl_large, wl_clust)`.

### 2.5 Binning and deconvolution

Multipoles are averaged in linear bins of width $\Delta\ell$.  The binned
theory is $\langle \hat{C}_b \rangle = B_b \cdot (M^{\rm clust} C^{\rm true} + \text{floor}_\ell)$
where $B$ is the binning matrix.

Alternatively, the clustering MASTER can be inverted (deconvolution):
subtract floor\_$\ell$ from the data, then solve $M_{bb}^{-1}\, \tilde{C}_b$
to recover $C_\ell^{\rm true}$ directly.

---

## 3. Key Approximations and Their Effects

| Approximation | Effect on ratio | Status |
|---|---|---|
| $\bar\chi$ (box midpoint) as distance | +7.5% bias, ℓ-tilted | **Fixed** → use $\chi_{\rm eff}$ |
| `diag_cl` from discrete 2D-mode sum | −1.5% offset | **Fixed** → MASTER-consistent floor |
| Limber $C_\ell^{\rm true}$ below $L_f = 2\pi\chi/L$ | First bin (ℓ < $L_f$) unreliable | **Known** — discard |
| Pixel window (HEALPix) | Not applicable | DirectSHT is exact |
| Integral constraint ($k_\perp = 0$) | No effect at ℓ > 0 | N/A |

### Current accuracy

With 100 sims, $N_\ell = 500$, $N_{\ell'} = 2000$:

- **Mean ratio** (measured / theory, excluding first bin): $0.996 \pm 0.018$
- **First bin** (ℓ ≈ 16): ratio ∼ 0.88 (below $L_f \approx 23$, unreliable)
- All $N_{\ell'}$ values converge (floor subtraction eliminates truncation sensitivity)

---

## 4. Code Organisation

| File | Purpose |
|---|---|
| `run_sims.py` | Run N GRF sims → pseudo-$C_\ell$ cache (`.npz`) |
| `compute_theory.py` | Angular window + floor-subtracted MASTER theory (`.npz`) |
| `notebooks/GRF_class.py` | GRF generation (CAMB + Fourier draw) |
| `notebooks/SHT_lya.py` | DFT, pair-counting, Legendre sums |
| `notebooks/fast_Wigner3j.py` | Wigner 3$j$ coupling matrices |
| `sht/sht.py` | DirectSHT engine |
| `sht/lya_sfb.py` | `_alm2cl_complex` |
| `sht/mask_deconvolution.py` | MASTER binning & deconvolution |


## 5. Run on NERSC
`python run_sims.py --Nsims 100 --Lbox 1380 --Ncell 512 --nqso 60 --outdir results`

`python compute_theory.py --simfile results/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz --outdir results`

Then open `notebooks/plot_money_results.ipynb`