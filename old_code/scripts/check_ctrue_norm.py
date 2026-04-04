#!/usr/bin/env python
"""
Investigate WHY C_true overpredicts the measured pseudo-Cl by ~20-30%.

Key hypotheses:
1. The k_perp = (ell+0.5)/chi approximation is wrong (should be k_perp = ell/chi or something else)
2. The normalization factor 32*pi^3 is wrong
3. There's a missing angular pixel window function or beam that attenuates high-ell power
4. The box geometry / sightline sampling creates an effective pixel window at the angular scale
   corresponding to the mean sightline separation
5. The plin interpolation is inaccurate at high k (near Nyquist)

Let me test each.
"""
import sys, os, gc, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import GRF_class as my_GRF
import fast_Wigner3j as Wigner3j

# Load cache
d = np.load('notebooks/data/Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz')
cl_k_all = d['cl_k']
wl_k = d['wl_k']
N = int(d['Nk'])
L = float(d['L'])
Nskew = int(d['Nskew'])
Nl = 500
wl_ref = wl_k[0, :Nl]
cl_mean = np.mean(cl_k_all, axis=0)

# Cosmology
GRF_tmp = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=0, verbose=False)
plin_ref = GRF_tmp.plin
b1_ref = GRF_tmp.my_bias
chi_bar = 5000 + L/2.0
del GRF_tmp; gc.collect()

print(f"N={N}, Nskew={Nskew}, L={L:.1f}, chi_bar={chi_bar:.1f}, b1={b1_ref:.4f}")

# ---- Let me compute the theory from the 3D power spectrum directly ----
# In the sim, the 3D field is: delta(r) = b1 * sum_k A_k exp(ik.r) / L^{3/2}
# where A_k are the Fourier modes with <|A_k|^2> = P_lin(k)
# (This is what GRF_class generates.)

# The LOS-averaged (k_par=0) field at sightline j is:
# delta_j = sum_{pixels} delta(x_j, y_j, z_alpha) = SUM over z of delta
# In Fourier: delta_j = sum_{k_perp, k_par} A_k exp(ik_perp.r_j) exp(ik_par.z_alpha)
# Summing over alpha (z-pixels):
# delta_j^{k=0} = SUM_alpha delta(x_j, y_j, z_alpha)
#               = SUM_{k_perp} SUM_{k_par} A_k exp(ik_perp.r_j) SUM_alpha exp(ik_par.z_alpha)
# SUM_alpha exp(ik_par.z_alpha) = N * delta_{k_par, 0} (only k_par=0 survives for the DFT at n=0)
# So delta_j^{k=0} = N * SUM_{k_perp} A_{k_perp, k_par=0} * exp(ik_perp . r_perp_j) / L^{3/2}

# Actually, the GRF code generates:
# amplitudes = b1 * complex_gaussian * sqrt(P(k)) / L^{3/2}
# density = real(IFFT(amplitudes)) (uses numpy's IFFT convention)
# numpy IFFT: x(n) = (1/N^3) SUM_k X(k) exp(2pi*i*k*n/N)

# Let me trace through more carefully.
# In GRF_class.py, the density field is:
# dens = np.real(np.fft.ifftn(amplitudes))
# amplitudes = b1 * get_amplitudes3d(L, N, pk_all, beta, seed, mu)
# get_amplitudes3d makes: a(k) = sqrt(pk) * L^{3/2} * gaussian_complex

# Wait, let me check get_amplitudes3d:
import inspect
# Need to check the function source
"""
@jit(nopython=True, cache=True, parallel=True)
def get_amplitudes3d(L, N, pk, beta, seed, mu):
    np.random.seed(seed)
    amplitudes = np.zeros((N, N, N), dtype=np.complex128)
    norm = np.sqrt(pk * L ** 3)
    for ...
        amplitudes[i, j, k] = norm[i, j, k] * (random_real + 1j * random_imag) / np.sqrt(2.0)
"""
# So amplitudes = sqrt(P(k)*L^3) / sqrt(2) * (a + ib)
# Then density = real(IFFT(amplitudes))
# numpy IFFT: dens(r) = (1/N^3) SUM_k amplitudes(k) exp(2pi*i*k*n/N)
# where n is the grid index (r = n*dx, dx = L/N)

# <|amplitudes(k)|^2> = P(k)*L^3/2 * (1^2 + 1^2) / 2 = P(k)*L^3/2
# Wait: let me recalculate. random_real and random_imag are N(0,1) independent.
# amplitudes = sqrt(P*L^3) / sqrt(2) * (a + ib)
# |amplitudes|^2 = P*L^3/2 * (a^2 + b^2)
# <|amplitudes|^2> = P*L^3/2 * 2 = P*L^3

# density = (1/N^3) SUM_k amplitudes(k) exp(2pi*i*k.n/N)
# <|density(r)|^2> = (1/N^6) SUM_k P(k)*L^3 = (L^3/N^6) SUM_k P(k)
# = (1/N^3) * (L/N)^3 * (N^3) * <P> = ... this is getting complicated.

# Let me just numerically check what the sims give.
# The measured pseudo-Cl is at k=0 (monopole DFT bin).
# w_j = SUM_alpha delta(x_j, y_j, z_alpha) [the DFT at k=0]
# = N * real(SUM_{k_par} amplitudes(kx_j, ky_j, k_par) * exp(2pi*i*k_par*alpha/N) / N)
# Wait, this comes from the 3D inverse FFT.
# density(x, y, z) = (1/N^3) SUM_{kx,ky,kz} amplitudes(kx,ky,kz) exp(2pi*i*(kx*ix + ky*iy + kz*iz)/N)
# w_j = SUM_{iz=0}^{N-1} density(ix_j, iy_j, iz)
# = (1/N^3) SUM_{kx,ky,kz} amplitudes * exp(2pi*i*(kx*ix_j+ky*iy_j)/N) * SUM_iz exp(2pi*i*kz*iz/N)
# SUM_iz exp(2pi*i*kz*iz/N) = N * delta_{kz,0}
# So w_j = (1/N^2) SUM_{kx,ky} amplitudes(kx,ky,0) * exp(2pi*i*(kx*ix_j+ky*iy_j)/N)
# = (1/N^2) SUM_{k_perp} a(k_perp, k_par=0) * exp(ik_perp . r_perp_j / (L/N) * 2pi/N)
# = (1/N^2) SUM_{k_perp} a(k_perp, 0) * exp(2pi*i*(kx*ix+ky*iy)/N)

# Note: I'm using gride indices. The actual k = 2*pi*n/L where n is the FFT index.
# And the position r = ix * dx where dx = L/N.
# exp(ik.r) = exp(2*pi*i*n*ix*dx/L) = exp(2*pi*i*n*ix/N)

# So w_j = (1/N^2) * SUM_{kx,ky} a(kx,ky,0) * exp(2*pi*i*(kx*ix_j+ky*iy_j)/N)

# Wait, but the code computes w_j differently. Let me re-read:
# In lya_GRFs_directSHT_loop.py:
# delta_skewers = GRF.process_skewers(Nskew=num_qso)[3]  # shape (Nskew, N)
# Then FT: for each skewer, DFT along z-axis.
# k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, mask_ones, delta_skewers[j], ...)
# FT_delta = Re(DFT(delta_skewers[j]))

# delta_skewers[j] = density field along sightline j = [delta(x_j,y_j,z_0), ..., delta(x_j,y_j,z_{N-1})]
# DFT: FT[n] = SUM_alpha delta(j, z_alpha) * exp(-2*pi*i*n*alpha/N)
# = SUM_alpha density(ix_j, iy_j, iz_alpha) * exp(-2*pi*i*n*alpha/N)
# For n=0: FT[0] = SUM_alpha density(ix_j, iy_j, iz_alpha) = w_j

# But SHT_lya uses scipy.linalg.dft which has:
# dft(N)_{mn} = exp(-2*pi*i*m*n/N)
# So FT = delta . dft(N) and FT[n] = SUM_alpha delta[alpha] * exp(-2*pi*i*n*alpha/N)
# And then Re() is taken.

# For n=0 (our case): FT[0] = SUM_alpha delta[alpha] = the sum of the density along LOS.
# = density_sum = w_j (unnormalized DFT at k=0)

# OK so w_j = SUM_alpha density(x_j, y_j, z_alpha)
# From the IFFT: density(r) = (1/N^3) SUM_k amplitudes(k) exp(2*pi*i*k.r/N)
# w_j = (1/N^3) SUM_k SUM_alpha amplitudes(k) exp(2*pi*i*(kx*ix_j+ky*iy_j+kz*alpha)/N)
# = (1/N^3) SUM_{kx,ky} amplitudes(kx,ky,0) * N * exp(2*pi*i*(kx*ix_j+ky*iy_j)/N)
# = (1/N^2) SUM_{kx,ky} amplitudes(kx,ky,0) * exp(2*pi*i*(kx*ix_j+ky*iy_j)/N)

# Good. So w_j = (1/N^2) SUM_{k_perp} a(k_perp, 0) * exp(ik_perp.r_perp_j)
# where the position r_perp_j is in grid index coordinates and k_perp is also in grid indices.

# Now the angular correlation:
# <w_j w_k> = (1/N^4) SUM_{k_perp} <|a(k_perp, 0)|^2> exp(ik_perp.(r_j-r_k))
# Since <|a(k)|^2> = P(k)*L^3:
# = (L^3/N^4) SUM_{k_perp} P(k_perp, 0) exp(ik_perp.(r_j-r_k))

# The k_perp sum runs over the 2D integer grid: nx=-N/2+1..N/2, ny=-N/2+1..N/2
# But P(k) = P_lin(|k|) for k_par=0: P(nx, ny, 0) = P_lin(k_f*sqrt(nx^2+ny^2))
# where k_f = 2*pi/L.

# Converting to continuous notation:
# k_perp = 2*pi*n_perp/L, dk_perp = (2*pi/L) * dn
# SUM_{k_perp} -> (L/(2pi))^2 * integral dk_perp^2    [in continuous limit]
# <w_j w_k> = (L^3/N^4) * (L/(2pi))^2 * integral P(k_perp) exp(ik_perp.Delta_r) dk_perp^2
# Wait no, the sum is discrete. But equivalently:
# <w_j w_k> = (L^3/N^4) * SUM_{nx,ny} P_lin(k_f*sqrt(nx^2+ny^2)) * exp(2*pi*i*(nx*Dix+ny*Diy)/N)

# For the pseudo-Cl, we need:
# <hat{C}_ell> = SUM_{j,k} <w_j w_k> P_ell(cos_jk) / (4pi)

# Now, the angular position of sightline j involves its actual (theta, phi) on the sky.
# The transverse position of sightline j is at (x_j, y_j) on the box face.
# The angular position is (x_j - L/2 + chi_shift*sin(theta)cos(phi), ...) ...
# Actually, the GRF class computes theta_j, phi_j from the 3D positions at z=chi_shift:
# theta, phi = compute_theta_phi_skewer_start(x[:,0], y[:,0], z[:,0])
# where (x,y,z) are the 3D coordinates of each skewer endpoint.

# The key approximation in C_true: k_perp = ell/chi_bar (Limber).
# This converts the 2D angular correlation into a 3D transverse correlation.
# But the ACTUAL correlation depends on the detailed geometry.

# Instead of investigating the C_true formula further, let me take a different approach:
# Compute the EXPECTED angular correlation function from the known 3D power spectrum,
# and then see what pseudo-Cl it predicts, WITHOUT using the Limber approximation.

# Actually, let me first just check: what overall normalization factor alpha would
# make the theory match the data when fully converged?

# From the extended PLKjKk to lambda=4000 study:
# At Nl_large=3500: theory/data = 1.18
# Still growing. Converged value (to Nyquist) estimated at ~1.25.
# So alpha = 1/1.25 = 0.80 — we need C_true to be 80% of what we compute.

# What could cause a factor of ~0.80?
# 1. Missing 1/N or N factor: N=512, not obvious
# 2. Factor of 2: 0.80 is not 0.5
# 3. P_3D(k_perp, k_par=0) is the kz=0 mode; for a discrete FFT, this includes
#    contributions from all k_par (aliasing). Hmm, but for k_par=0 exactly, only
#    the kpar=0 mode contributes.
# 4. The DISCRETE sum over k_perp should use the DISCRETE power spectrum (which
#    differs from the continuous P_lin at high k due to aliasing and gridding).

# Let me check hypothesis 4: the discrete power spectrum.
# In the simulation, P(k) is evaluated at the discrete grid points BEFORE FFT.
# The FFT perfectly represents the modes at those k-values.
# At high k (near Nyquist), there's no aliasing because the modes are discrete.
# So P(k) at each grid point is exactly P_lin(k).
# Unless... the initial conditions have a cutoff or deconvolution.

# Let me also check the process_skewers function to see if there's a normalization there.
print(f"\n---- Testing normalization with a simple simulation ----")
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=42, verbose=False)

# Get density field
dens = GRF.dens  # The density field, shape (N, N, N)
print(f"Density field: shape={dens.shape}, mean={np.mean(dens):.4e}, std={np.std(dens):.4e}")

# Get amplitudes at kz=0 plane
amps = GRF.amplitudes  # shape (N, N, N)
# The kz=0 slice is amps[:,:,0]
amps_kz0 = amps[:, :, 0]
print(f"Amplitudes at kz=0: mean|a|^2 = {np.mean(np.abs(amps_kz0)**2):.4e}")

# Expected: <|a(k)|^2> = P_lin(k) * L^3
# Mean over all k_perp modes:
k_perp_grid = np.fft.fftfreq(N) * 2*np.pi*N/L
kx = k_perp_grid
ky = k_perp_grid
KX, KY = np.meshgrid(kx, ky)
k_mag = np.sqrt(KX**2 + KY**2)
Pk_expected = plin_ref(k_mag.ravel()).reshape(k_mag.shape)
mean_Pk_L3 = np.mean(Pk_expected) * L**3 * b1_ref**2
print(f"<P(k)*L^3*b1^2> = {mean_Pk_L3:.4e}")
print(f"Ratio |a|^2 / (P*L^3*b1^2) = {np.mean(np.abs(amps_kz0)**2)/mean_Pk_L3:.4f}")

# Now compute the LOS sum: delta_j = SUM_alpha density(ix_j, iy_j, alpha)
# For a complete set of sightlines (all N^2 positions):
wj_all = np.sum(dens, axis=2)  # sum along z-axis, shape (N, N)
print(f"\nw_j (LOS sum): shape={wj_all.shape}, mean={np.mean(wj_all):.4e}, std={np.std(wj_all):.4e}")

# Check: w_j should equal (1/N^2) * SUM_{k_perp} a(k_perp,0) * exp(ik_perp.r_j)
# But density = IFFT(amplitudes), so SUM_z density = N * density_FT_at_kz0
# Wait: IFFT: density(n) = (1/N^3) SUM_k a(k) exp(2pi*i*k.n/N)
# SUM_nz density(nx,ny,nz) = (1/N^3) SUM_kx,ky,kz a(k) exp(2pi*i*kx*nx/N) exp(2pi*i*ky*ny/N) SUM_nz exp(2pi*i*kz*nz/N)
# = (1/N^3) * N * SUM_kx,ky a(kx,ky,0) exp(2pi*i*(kx*nx+ky*ny)/N)
# = (1/N^2) SUM_kx,ky a(kx,ky,0) exp(2pi*i*(kx*nx+ky*ny)/N)

# This is exactly the 2D IFFT of a(kx,ky,0) with a 1/N^2 factor.
# NumPy's IFFT already applies 1/N per dimension.
# So wj_all should equal ifft2(a(kx,ky,0)) * N^2 / N^2 = ... hmm
# Let me just check with the FFT:
wj_fft = np.fft.ifft2(amps_kz0).real * N**2 / N**2  # no extra factor needed?
# Actually: density = np.fft.ifftn(amplitudes).real
# SUM_z ifftn = (1/N^3) SUM exp() * SUM_nz exp() = (1/N^2) * SUM a(kx,ky,0) * exp()
# = ifft2(a(kx,ky,0))   if we use (1/N^2) normalization
# But numpy ifft2 uses 1/N^2 normalization. So SUM_z density = ifft2(a(kx,ky,0))? No.
# density = ifftn(amplitudes) with normalization 1/N^3
# SUM_z density = (1/N^3) SUM_k a(k) exp(ik_perp.r) * N * delta(kz,0)
# = (N/N^3) SUM_{k_perp} a(k_perp,0) exp(ik_perp.r)
# = (1/N^2) SUM_{k_perp} a(k_perp,0) exp(ik_perp.r)
# = ifft2(a(:,:,0)) using numpy convention

# Check:
wj_check = np.fft.ifft2(amps_kz0).real
print(f"Check: max|wj_all - ifft2(amps_kz0)| = {np.max(np.abs(wj_all - wj_check)):.4e}")

# OK so w_j = ifft2(a(kx,ky,0)) = (1/N^2) SUM_{k_perp} a(k_perp,0) exp(ik_perp.r_j)

# Now: <w_j w_k> = (1/N^4) SUM_{k_perp} <|a(k_perp,0)|^2> exp(ik_perp.(r_j-r_k))
# <|a(k_perp,0)|^2> = Pk_lin(k_perp) * L^3 * b1^2
# So <w_j w_k> = (b1^2 * L^3 / N^4) SUM_{k_perp} Pk_lin(k_perp) exp(ik_perp.Delta_r)

# For the angular pseudo-Cl:
# <Cl> = SUM_{j,k} <w_j w_k> Pl(cos_jk) / (4pi)
# = (b1^2 L^3 / N^4) * (1/(4pi)) SUM_{j,k} SUM_{k_perp} Pk(k_perp) exp(ik.Dr) Pl(cos_jk)

# In the continuous limit:
# SUM_{k_perp} -> (L/(2pi))^2 * integral d^2k_perp
# So <Cl> ~ (b1^2 L^3 / N^4) * (L/(2pi))^2 * (1/(4pi)) * [stuff]
#         = (b1^2 L^5 / (4pi^3 N^4)) * [stuff]

# But in the Limber/C_true approach:
# <Cl> = SUM_L M[l,L] * b1^2*Plin(L/chi) / (32*pi^3*chi^2)
# The M matrix has norm ~ (2L+1)/(4pi) * sum lambda (2lam+1) W * 3j^2
# For shot-noise dominated W: M ~ (2L+1)/(4pi) * SN/(4pi)
# And SN = N^2 * Nskew

# Hmm, this is getting complex. Let me just numerically verify with the actual sim.
# I'll compute the pseudo-Cl from the full grid (all N^2 sightlines, not the Nskew subset).

# Actually: the key question is whether the discrete 2D power spectrum
# matches b1^2*Plin(k_perp)*L^3.
# <|a(k_perp,0)|^2> should equal b1^2*Plin(k_perp)*L^3.

# Let me check with the actual amplitudes:
print(f"\n---- Discrete P(k) check ----")
k_f = 2*np.pi/L
k_perp_arr = np.fft.fftfreq(N, d=L/N) * 2*np.pi  # k in physical units
kx_arr = k_perp_arr
ky_arr = k_perp_arr

# Bin |a|^2 by |k_perp|
kx_grid = kx_arr
ky_grid = ky_arr
KX, KY = np.meshgrid(kx_grid, ky_grid)
k_perp_mag = np.sqrt(KX**2 + KY**2)

# For a single realization, |a|^2 is noisy. But we can compare the mean ratio.
Pk_measured_2D = np.abs(amps_kz0)**2
Pk_theory_2D = b1_ref**2 * plin_ref(k_perp_mag.ravel()).reshape(k_perp_mag.shape) * L**3

# Average in k-bins
k_edges = np.linspace(0, np.pi*N/L, 50)
for i in range(len(k_edges)-1):
    mask = (k_perp_mag >= k_edges[i]) & (k_perp_mag < k_edges[i+1])
    if np.sum(mask) > 10:
        ratio = np.mean(Pk_measured_2D[mask]) / np.mean(Pk_theory_2D[mask])
        k_mid = (k_edges[i] + k_edges[i+1]) / 2
        if i < 5 or (i % 10 == 0):
            print(f"  k={k_mid:.4f}: <|a|^2>/<P*L^3*b1^2> = {ratio:.4f} ({np.sum(mask)} modes)")

# The ratio should be ~1.0 on average (noisy for single sim).
# Any systematic deviation would indicate a normalization issue.
print(f"\nOverall mean ratio: {np.mean(Pk_measured_2D) / np.mean(Pk_theory_2D):.4f}")
# For a single sim, this has chi-squared noise but should be ~1.0 on average.
