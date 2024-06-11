import os, sys
import numpy as np
from matplotlib_params_file import *
import multiprocessing as mp
import healpy as hp

# GRFs
import GRF_class as my_GRF

# import function for SHT-lya
import SHT_lya as sht_lya

#sys.path.insert(0, '/global/homes/r/rmvd2/lya_Cl/directsht-lya/')
sys.path.insert(0, '/Users/rdb/Desktop/research/lya/P3D_Cell/directsht-lya/')
from sht.sht                import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution

try:
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)
except:
    print("No GPU found")

# Wigner3j code
import fast_Wigner3j as Wigner3j

periodic=True


# define GRF settings

# define number of qso drawn from the box
num_qso = int(1e+4)
num_sim = 20

# set `add_rsd=True' if you want to add RSD
add_rsd_=False

#initialize SHT
# Set up an sht instance.
Nl   = 500
Nx   = 2*Nl
xmax = 5.0/8.0
NperBin = 2**5
lambda_max = 2*Nl
L_max = Nl

sht= DirectSHT(Nl,Nx,xmax)

print("For general, Direct SHT has Nl=",sht.Nell,", Nx=",Nx," and xmax=",xmax)

data_dict = {
    'theory_cl': [],
    'binned_ells': [],
    'measured_cl': [],
    'window_conv_theory_cl': [],
    'cl_k': [],
    'wl_k': [],
    'Nskew': [],
    'Nk': [],
    'L': []
}

def Power_spectrum(kh_perp, kh_par, GRF, add_rsd=add_rsd_):
    kh = np.sqrt(kh_par**2 + kh_perp**2)
    pk = GRF.plin(kh)
    Kaiser_factor = 1.
    if add_rsd:
        mu = kh_par / kh
        Kaiser_factor = GRF.my_bias**2 * (1. + GRF.my_beta * mu**2)**2
    # print(kh, pk)
    return Kaiser_factor * pk


# define index for calculation 
k_idx = 0

for seed_idx in range(num_sim):
    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed_idx, verbose=False)
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(Nskew=num_qso)
    all_theta, all_phi = GRF.compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])
    chi_grid = all_x[0,:]

    if seed_idx > 0: old_tdata, old_pdata, old_wdata = all_theta, all_phi, FT_delta

    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)

    if seed_idx > 0: print('compare theta and phi: ',np.allclose(old_tdata, tdata),np.allclose(old_pdata, pdata))

    tdata,pdata,wdata = all_theta, all_phi, FT_delta
    trand,prand,wrand = all_theta, all_phi, FT_mask
    print(f'Nskew = {Nskew}, Nk = {GRF.N}, L = {GRF.L}')

    # Calculate the angular power spectrum of the randoms for the window function
    hran = sht(trand,prand,wrand[:,k_idx])
    wl = hp.alm2cl(hran)
    # Calculate the angular power spectrum of the difference data-randoms
    hdat = sht(trand,prand,wdata[:,k_idx])
    cl = hp.alm2cl(hdat-hran)
    
    data_dict['wl_k'].append(wl)
    data_dict['cl_k'].append(cl)

    #compute theory power spectrum

#save GRF params
data_dict['Nskew'] = Nskew
data_dict['Nk'] = GRF.N
data_dict['L'] = GRF.L
#save the data
data_dict['cl_k'] = np.stack(data_dict['cl_k'])
data_dict['wl_k'] = np.stack(data_dict['wl_k'])

############## compute theory spectra ###########

#initialize class
couple_mat = Wigner3j.CoupleMat(Nl, wl)
coupling_matrix_window = couple_mat.compute_matrix()

MD = MaskDeconvolution(Nl,wl,precomputed_Wigner=coupling_matrix_window)
# choose binning for Cell's
bins    = MD.binning_matrix('linear',0,NperBin)
#Mbl     = MD.window_matrix(bins)

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

# compute F_{ell L lambda} matrix
couple_mat = Wigner3j.CoupleMat(Nl, PLKjKk)
coupling_matrix_leg_pol = couple_mat.compute_matrix()

# is the L range sufficient?
L_range = np.arange(0, L_max, 1)

# is chi bar approx correct?
chi_bar = (chi_grid.max()+chi_grid.min())/2
print('compute for k index:', k_idx, 'with chibar: ', chi_bar, f'and 1/(2 pi chi^2)={(1/(2.*np.pi*chi_bar**2)):.2e}')

# Limber approx for power spec
pk_L = Power_spectrum(kh_perp=L_range/chi_bar, kh_par=k_arr[k_idx], GRF=GRF)/(2.*np.pi*chi_bar**2)

#theory power spectrum
C_ell = np.dot(coupling_matrix_leg_pol, pk_L)/(4.*np.pi)
# window convolved theory power spectrum
C_ell_hat = np.dot(coupling_matrix_window, C_ell) #[:MD.Mll.shape[1]]

# binning
ells = np.arange(Nl)
my_binned_ells = np.dot(bins, ells)

# bin theory Cell's
binned_C_ell = bins @ C_ell#[:MD.Mll.shape[1]]
# bin theory window func convolved Cell's
binned_C_ell_hat = bins @ C_ell_hat
# bin measured Cell's
binned_hdif = []
for i in range(num_sim):
    binned_hdif.append(bins @ cl_k[i])
binned_hdif = np.stack(binned_hdif)

data_dict['theory_cl'] = binned_C_ell
data_dict['window_conv_theory_cl'] = binned_C_ell_hat

data_dict['binned_ells'] = my_binned_ells
data_dict['measured_cl'] = binned_hdif

np.savez(f'./data/Cell_GRF_L{int(GRF.L):d}_N{int(GRF.N):d}_Nq{int(Nskew):d}_Nl{int(Nl):d}_sims{int(num_sim):d}.npz', **data_dict)