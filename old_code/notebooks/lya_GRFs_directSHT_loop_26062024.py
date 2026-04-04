import os
import numpy as np
from matplotlib_params_file import *
import multiprocessing as mp
import healpy as hp

# GRFs
import GRF_class as my_GRF

# import function for SHT-lya
import SHT_lya as sht_lya

import sys

sys.path.insert(0, '/Users/rdb/Desktop/research/lya/P3D_Cell/directsht-lya/')
from sht.sht                import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution

try:
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)
except:
    print("No GPU found")

if len(sys.argv) <= 6:
    assert "input format should be: python lya_GRFs_directSHT_loop_26062024.py chi_shift Nl lambda_max L_max num_sim num_qso"

# read in params
chi_shift = int(sys.argv[1])
Nl = int(sys.argv[2])
lambda_max = int(sys.argv[3])
L_max = int(sys.argv[4])
num_sim = int(sys.argv[5])
num_qso = int(sys.argv[6])
# add_rsd_ = bool(sys.argv[6])

# Set up an sht instance.
# Nl   = 500
Nx   = 2*Nl
xmax = 3./4.
#
sht= DirectSHT(Nl,Nx,xmax)
print("For general, Direct SHT has Nl=",sht.Nell,", Nx=",Nx," and xmax=",xmax)

# define GRF settings
# define number of qso drawn from the box
# num_qso = int(1e+4)
# set `add_rsd=True' if you want to add RSD
add_rsd_=False
# shift GRF box
# chi_shift = 5000. # Mpc/h

# define index for calculation 
k_idx = 0

periodic=True

print('compute for k index:', k_idx)
print('compute for chi_shift:', chi_shift)
print('compute for periodic:', periodic)
print('compute for Nl:', Nl)
print('lambda_max:', lambda_max)
print('L_max:', L_max)
print('num_sim:', num_sim)
print('num qso   :', num_qso)
print('add rsd   :', add_rsd_)

rsd_str=''
if add_rsd_: rsd_str='_rsd' 
opt = f'chi{chi_shift}_Nl{Nl}_lambda{lambda_max}_L{L_max}_sim{num_sim}_Nqso{num_qso}{rsd_str}'

cl_k = []
wl_k = []
for idx, my_seed in enumerate(range(num_sim)):
    # compute GRF
    my_seed = 1000 + my_seed
    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=my_seed)
    # Sample skewers
    # Note that we apply a shift along the z axis. This is necessary to ensure that the $\mu$ angle is with respect to the z-axis
    # computes $\mu$ with respect to the combined line-of-sight; $(\vec{r}_i+\vec{r}_j)/2$    
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(Nskew=num_qso, shift=chi_shift)
    all_theta, all_phi = GRF.compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])
    chi_grid = all_x[0,:] # Mpc/h
    # definition of D=w*delta
    all_w_gal=all_w_gal-1.

    # compute discrete Fourier transform of matrix $N_{qso} \times N_{pix}$    
    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)
    tdata,pdata,wdata = all_theta, all_phi, FT_delta.real
    trand,prand,wrand = all_theta, all_phi, FT_mask.real
    print(f'Nskew = {Nskew}, Nk = {wrand.shape[1]}')

    # measure $C_{\ell}(k)$ for different values of $k$
    hdat = sht(tdata,pdata,wdata[:,k_idx])
    # Do the same for the randoms.
    hran = sht(trand,prand,wrand[:,k_idx])
    # normalization for FKP-type weights don't need that here
    # hran*= hdat[0]/hran[0]
    hdif = hp.alm2cl(hdat)
    wl   = hp.alm2cl(hran)
    cl_k.append(hdif) # C_\ell(k) data
    wl_k.append(wl) # C_\ell(k) randoms

cl_k = np.stack(cl_k)
wl_k = np.stack(wl_k)

precompute_Wigner3j = True
if precompute_Wigner3j:
    # Wigner3j code
    import fast_Wigner3j as Wigner3j

    #initialize class
    couple_mat = Wigner3j.CoupleMat(Nl, wl_k[k_idx])
    coupling_matrix = couple_mat.compute_matrix()

    MD = MaskDeconvolution(Nl,wl_k[k_idx],precomputed_Wigner=coupling_matrix)
else:
    # if I want to use SHT for the Wigner3j computation
    print('use SHT native Wigner3j computation')
    MD = MaskDeconvolution(Nl,wl_k[k_idx])
#
# choose binning for Cell's
NperBin = 2**5
bins    = MD.binning_matrix('linear',0,NperBin)
Mbl     = MD.window_matrix(bins)


# Compute nhat
nhat = sht_lya.compute_nhat(tdata, pdata)
# Compute cos(theta_nj_nk)
cos_theta_njnk = np.dot(nhat, nhat.T)

if periodic:
    KjKk = GRF.N**2
else:
    KjKk = np.outer(FT_mask[:, k_idx], FT_mask[:, k_idx])

#test run to initialize numba 
_my_result = sht_lya.legendre_polynomials_sum(1, cos_theta_njnk,KjKk)

print('compute the Legendre polynomials sum times the mask')
PLKjKk = sht_lya.legendre_polynomials_sum(lambda_max, cos_theta_njnk, KjKk)[:-1]
print('done')

def Power_spectrum(kh_perp, kh_par, add_rsd=add_rsd_):
    kh = np.sqrt(kh_par**2 + kh_perp**2)
    pk = GRF.plin(kh)
    Kaiser_factor = 1.
    if add_rsd:
        mu = kh_par / kh
        Kaiser_factor = GRF.my_bias**2 * (1. + GRF.my_beta * mu**2)**2
    # print(kh, pk)
    return Kaiser_factor * pk


L_range = np.arange(0, L_max, 1)

# is chi bar approx correct?
chi_bar = (chi_grid.max()+chi_grid.min())/2
print('compute for k index:', k_idx, 'with chibar: ', chi_bar, f'and 1/(2 pi chi^2)={(1/(2.*np.pi*chi_bar**2)):.2e}')

# Limber approx for power spec
pk_L = Power_spectrum(kh_perp=L_range/chi_bar, kh_par=k_arr[k_idx])


def compute_power_spectrum(lambda_max, pk_L, PLKjKk, chi_bar=chi_bar, bins=bins, Nl=Nl):
    """
    Compute the binned theory power spectrum and the binned measured Cell's.
    """
    couple_mat = Wigner3j.CoupleMat(lambda_max, pk_L)
    coupling_matrix_pk_L = couple_mat.compute_matrix()

    # Compute theory power spectrum
    C_ell = np.dot(coupling_matrix_pk_L, PLKjKk) / (4. * np.pi) / (2. * np.pi * chi_bar**2)
    # Bin the theory power spectrum
    binned_C_ell = np.dot(bins, C_ell[:Nl])
    return binned_C_ell

binned_Cell = compute_power_spectrum(lambda_max, pk_L, PLKjKk, chi_bar=chi_bar, bins=bins, Nl=Nl)

ells = np.arange(Nl)
my_binned_ells = np.dot(bins, ells)

plt.figure(figsize=(9,6))
plt.title(f'{opt}')
plt.plot(my_binned_ells, binned_Cell/(4*np.pi)**2, 'k.--',lw=3, label=r'$C_{\ell}^{\rm th}/(4\pi)^2$')
# plt.plot(my_binned_ells, 2*binned_Cell/(4*np.pi)**2, 'k.--',lw=3, label=r'$2C_{\ell}^{\rm th}/(4\pi)^2$')
# plt.plot(my_binned_ells, 1.5*binned_Cell/(4*np.pi)**2, 'k.--',lw=3, label=r'$1.5C_{\ell}^{\rm th}/(4\pi)^2$')

binned_hdif_mean = []
for i in range(num_sim):
    binned_hdif = bins @ cl_k[i]
    binned_hdif_mean.append(binned_hdif)
    plt.plot(my_binned_ells, binned_hdif,'k--', alpha=0.3)

binned_hdif_mean = np.mean(np.array(binned_hdif_mean), axis=0)
binned_hdif_std = np.std(np.array(binned_hdif_mean), axis=0)
plt.errorbar(my_binned_ells, binned_hdif_mean,yerr=binned_hdif_std/np.sqrt(num_sim), color='C3', label=r'$C_{\ell}^{\rm data}$')

plt.xlabel(r'multipole $\ell$')
plt.ylabel(r'$C_{\ell}(k)$')
plt.legend(ncol=1, fontsize=18, loc='upper right')
plt.savefig(f'plots/Cell_{opt}.pdf', bbox_inches='tight')
