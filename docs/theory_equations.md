# Spherical Fourier-Bessel Estimator: Theory & Equations

## 1. Overview

This document provides a self-contained derivation of the $C_\ell(k)$ estimator
for the Ly-α forest, extending the galaxy pseudo-$C_\ell$ method (Baleato Lizancos
& White 2024) to include a line-of-sight Fourier dimension.

**Key papers:**
- Baleato Lizancos & White 2024 (arXiv:2312.12285) — galaxy harmonic analysis
- de Belsunce, Baleato Lizancos & White — Ly-α sFB theory (`main.pdf`)
- Wolz, Alonso & Nicola 2024 (arXiv:2407.21013) — catalog-based pseudo-$C_\ell$

---

## 2. Galaxy Case (Review)

For galaxies, the pseudo-$C_\ell$ estimator is:

### 2.1 Direct SHT
$$
a_{\ell m}^{(d)} = \sum_i \omega_i^{(d)} Y_{\ell m}^*(\hat{n}_i), \quad
w_{\ell m} = \sum_j \omega_j^{(r)} Y_{\ell m}^*(\hat{n}_j)
$$

### 2.2 Pseudo-$C_\ell$
$$
\hat{C}_\ell = \frac{1}{2\ell+1} \sum_m |a_{\ell m}^{(d)} - w_{\ell m}|^2
$$

### 2.3 Mode-Coupling Matrix
$$
\langle \hat{C}_\ell \rangle = \sum_{\ell'} M_{\ell\ell'} C_{\ell'}, \quad
M_{\ell\ell'} = \frac{2\ell'+1}{4\pi} \sum_\lambda (2\lambda+1)
\begin{pmatrix} \ell & \ell' & \lambda \\ 0 & 0 & 0 \end{pmatrix}^2 W_\lambda
$$

where $W_\lambda = \frac{1}{2\lambda+1}\sum_m |w_{\lambda m}|^2$.

---

## 3. Ly-α Extension: From $C_\ell$ to $C_\ell(k)$

### 3.1 Physical Picture

Each quasar sightline $j$ at sky position $\hat{n}_j$ provides a spectrum
$\delta_F(\chi, \hat{n}_j)$ of flux fluctuations sampled at comoving distances
$\chi_\alpha$ along the line-of-sight.

The key insight: decompose the 3D field into angular (SHT) and radial (Fourier)
modes. This yields $C_\ell(k)$ — the angular power spectrum as a function of
line-of-sight wavenumber $k$.

### 3.2 Line-of-Sight Fourier Transform

For each sightline $j$, define the Fourier-weighted flux at mode $k$:

$$
\delta_{2\mathrm{D}}(\hat{n}_j; k) = \sum_\alpha K_j(\chi_\alpha)\,
\delta_F(\chi_\alpha, \hat{n}_j)\, e^{ik\chi_\alpha}\, \Delta\chi
$$

and the Fourier-weighted window:

$$
\tilde{K}_j(k) = \sum_\alpha K_j(\chi_\alpha)\, e^{ik\chi_\alpha}\, \Delta\chi
$$

where $K_j(\chi)$ is the weight function for sightline $j$ (inverse noise variance).

**For GRF periodic box:** $K_j(\chi) = 1$ (uniform weights), so:
- $\delta_{2\mathrm{D}}(\hat{n}_j; k) = \sum_\alpha \delta_F(\chi_\alpha, \hat{n}_j)\, e^{ik\chi_\alpha}\, \Delta\chi$
- $\tilde{K}_j(k) = \sum_\alpha e^{ik\chi_\alpha}\, \Delta\chi$ (sinc-like)

### 3.3 Angular SHT per $k$-bin

For each $k$-mode, the 2D projected field $\delta_{2\mathrm{D}}(\hat{n}_j; k)$
lives on the sky at quasar positions. Apply a direct SHT:

$$
a_{\ell m}^{(f)}(k) = \sum_j \delta_{2\mathrm{D}}(\hat{n}_j; k)\, Y_{\ell m}^*(\hat{n}_j)
$$

$$
w_{\ell m}(k) = \sum_j \tilde{K}_j(k)\, Y_{\ell m}^*(\hat{n}_j)
$$

**Real/Imaginary split:** Since $\delta_{2\mathrm{D}}(\hat{n}_j; k)$ is complex
(due to $e^{ik\chi}$), and `DirectSHT` takes real weights only:

1. $a_{\ell m}^{(\text{Re})}(k) = \mathrm{SHT}[\mathrm{Re}(\delta_{2\mathrm{D}}(\cdot; k))]$
2. $a_{\ell m}^{(\text{Im})}(k) = \mathrm{SHT}[\mathrm{Im}(\delta_{2\mathrm{D}}(\cdot; k))]$
3. $a_{\ell m}^{(f)}(k) = a_{\ell m}^{(\text{Re})}(k) + i\, a_{\ell m}^{(\text{Im})}(k)$

Same procedure for $w_{\ell m}(k)$.

### 3.4 Pseudo-$C_\ell(k)$

$$
\hat{C}_\ell(k) = \frac{1}{2\ell+1} \sum_m |a_{\ell m}^{(f)}(k) - \mathcal{N}\, w_{\ell m}(k)|^2
$$

where $\mathcal{N}$ is chosen to match $a_{00}^{(f)}(k) = \mathcal{N}\, w_{00}(k)$
for each $k$-bin. In the GRF mock context, this ensures the mean flux is correctly
subtracted.

### 3.5 Expanding the Modulus Squared

For complex $a_{\ell m}$ and $w_{\ell m}$, the power spectrum becomes:

$$
|a_{\ell m}^{(f)} - \mathcal{N} w_{\ell m}|^2 =
|a_{\ell m}^{(\text{Re})} - \mathcal{N} w_{\ell m}^{(\text{Re})}|^2 +
|a_{\ell m}^{(\text{Im})} - \mathcal{N} w_{\ell m}^{(\text{Im})}|^2
$$

plus cross terms involving the imaginary parts of the $a_{\ell m}$ themselves.
Since $a_{\ell m}$ from `DirectSHT` are complex (Healpix convention with
$m \geq 0$), the full expansion is:

$$
\hat{C}_\ell(k) = C_\ell^{(\text{Re,Re})}(k) + C_\ell^{(\text{Im,Im})}(k) + 2\,\text{cross terms}
$$

In practice, we compute `hp.alm2cl(alm_re_diff) + hp.alm2cl(alm_im_diff)` where:
- `alm_re_diff = alm_data_re - N * alm_rand_re`
- `alm_im_diff = alm_data_im - N * alm_rand_im`

---

## 4. Mode-Coupling Matrix

### 4.1 $k$-Dependent Mode Coupling

$$
\langle \hat{C}_\ell(k) \rangle = \sum_L M_{\ell L}(k)\, C_L(k)
$$

$$
M_{\ell L}(k) = \frac{2L+1}{4\pi} \sum_\lambda (2\lambda+1)
\begin{pmatrix} \ell & L & \lambda \\ 0 & 0 & 0 \end{pmatrix}^2 W_\lambda(k)
$$

### 4.2 $k$-Dependent Window Spectrum

$$
W_\lambda(k) = \frac{1}{2\lambda+1} \sum_m |w_{\lambda m}(k)|^2
$$

**Critical:** $W_\lambda(k)$ depends on $k$ because the Fourier weights
$\tilde{K}_j(k)$ vary with $k$. The mode-coupling matrix must be recomputed
per $k$-bin, but the Wigner $3j$ symbols are $k$-independent and cached.

### 4.3 Noise Floor (Wolz et al. 2024)

Mask shot noise:
$$
\tilde{N}^w(k) = \frac{1}{4\pi} \sum_j |\tilde{K}_j(k)|^2
$$

Corrected window: $\tilde{S}_\lambda^w(k) = W_\lambda(k) - \tilde{N}^w(k)$

---

## 5. Theory Prediction

### 5.1 Flat-Sky Limit (Periodic Box)

For the periodic box at mean comoving distance $\bar{\chi}$:

$$
C_L^{\text{theory}}(k) = \frac{1}{\bar{\chi}^2}\, P_F\!\left(k_\perp = \frac{L}{\bar{\chi}},\; k_\parallel = k\right)
$$

### 5.2 Ly-α 3D Power Spectrum (Kaiser Approximation)

$$
P_F(k_\perp, k_\parallel) = b_1^2\, (1 + \beta\, \mu^2)^2\, P_{\text{lin}}(k, z)
$$

where:
- $\mu = k_\parallel / \sqrt{k_\perp^2 + k_\parallel^2}$
- $k = \sqrt{k_\perp^2 + k_\parallel^2}$
- $b_1 = -0.1521$ (flux bias, negative = anti-biased)
- $\beta = 0.2298$ (Kaiser RSD parameter)
- $z = 2.33$

### 5.3 Window-Convolved Theory

The expected pseudo-$C_\ell(k)$ is:

$$
\langle \hat{C}_\ell(k) \rangle = \sum_L M_{\ell L}(k)\, C_L^{\text{theory}}(k)
$$

This is the comparison curve for the "money plot".

---

## 6. Periodic Box Specifics

### 6.1 Parameters

| Parameter | Value |
|-----------|-------|
| $z$ | 2.33 |
| $\bar{\chi}$ | Mean comoving distance at $z = 2.33$ |
| $L_{\text{box}}$ | 1380.0 $h^{-1}$ Mpc |
| $N_{\text{cell}}$ | 512 |
| $k_F = 2\pi/L$ | 0.00455 $h$ Mpc$^{-1}$ |
| $k_{\text{Ny}} = \pi N/L$ | 1.165 $h$ Mpc$^{-1}$ |
| $b_1$ | -0.1521 |
| $\beta$ | 0.2298 |
| Cosmology | Planck 2018 |

### 6.2 Uniform Window

For the periodic box with $K_j(\chi) = 1$:
- $\tilde{K}_j(k) = \sum_\alpha e^{ik\chi_\alpha} \Delta\chi$
- This is sinc-like: peaks sharply at $k = 0$ and decays for large $k$.
- The window *varies by sightline* only through the LOS grid $\chi_\alpha$ (which is the same for all sightlines in the periodic case).

### 6.3 Reduction to $C_\ell$ (no LOS Structure)

Setting $k = 0$: the Fourier weight is just $e^{i \cdot 0 \cdot \chi} = 1$,
so $\delta_{2\mathrm{D}}(\hat{n}_j; 0) = \sum_\alpha K_j(\chi_\alpha) \delta_F(\chi_\alpha, \hat{n}_j) \Delta\chi$
— a weighted integral of flux along the LOS. This recovers a 2D angular spectrum
(i.e. the standard $C_\ell$) of the line-of-sight averaged field.

---

## 7. Sign/Normalization Conventions

### 7.1 Fourier Convention
We use $e^{+ik\chi}$ in the LOS Fourier transform (following the paper convention).

### 7.2 Power Spectrum Normalization
$C_\ell(k)$ has dimensions of [power $\times$ length], since it relates to $P(k)$
via division by $\bar{\chi}^2$.

### 7.3 Healpix $a_{\ell m}$ Convention
Only $m \geq 0$ stored. The `alm2cl` formula accounts for the $m = 0$ and $m > 0$
terms as:
$$
C_\ell = \frac{1}{2\ell+1}\left[|a_{\ell 0}|^2 + 2\sum_{m=1}^{\ell} |a_{\ell m}|^2\right]
$$

### 7.4 Bias Convention
The code uses $\text{amplitude} = b_1 \times (1 + \beta \mu^2) \times \mathcal{N}(0, \sqrt{P_{\text{lin}}/2})$
so $P_F = b_1^2 (1 + \beta \mu^2)^2 P_{\text{lin}}$.

With $b_1 = -0.1521 < 0$: the field is anti-correlated with matter density
(absorbed flux is inversely related to neutral hydrogen density). The sign of $b_1$
doesn't affect $P_F$ since it enters quadratically, but it does affect the sign
of $\delta_F$.
