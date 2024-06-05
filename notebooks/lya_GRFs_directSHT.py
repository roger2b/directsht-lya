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

do_plots=False

# define GRF settings

# define number of qso drawn from the box
num_qso = int(1e+4)
num_sim = 5

# set `add_rsd=True' if you want to add RSD
add_rsd_=False

#initialize SHT
# Set up an sht instance.
Nl   = 500
Nx   = 2*Nl
xmax = 5.0/8.0
#
sht= DirectSHT(Nl,Nx,xmax)
#
# Here we don't go to higher lmax in W_l since we DO NOT mode-decouple
buffer_ells = 64
sht_randoms = DirectSHT(Nl+buffer_ells,Nx,xmax)
#
print("For general, Direct SHT has Nl=",sht.Nell,", Nx=",Nx," and xmax=",xmax)
print("For randoms, Direct SHT has Nl=",sht_randoms.Nell,", Nx=",Nx," and xmax=",xmax)

cl_k = []
wl = []
for seed_idx in range(num_sim):
    GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd_, seed=seed_idx, verbose=False)
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(Nskew=num_qso)
    all_theta, all_phi = GRF.compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])
    chi_grid = all_x[0,:]

    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)

    tdata,pdata,wdata = all_theta, all_phi, FT_delta
    trand,prand,wrand = all_theta, all_phi, FT_mask
    print(f'Nskew = {Nskew}, Nk = {GRF.N}, L = {GRF.L}')

    # define index for calculation 
    k_idx = 0
    # To Do: do this in a loop

    # Calculate the angular power spectrum of the randoms for the window function
    hran_for_wl = sht_randoms(trand,prand,wrand[:,k_idx])
    # # For lya we ignore the shot noise contribution and compute directly
    # # the angular power spectrum of the window function
    wl.append(hp.alm2cl(hran_for_wl))

    # Calculate the angular power spectrum of the data-randoms
    hdat = sht(tdata,pdata,wdata[:,k_idx]-wrand[:,k_idx])
    cl_sht = hp.alm2cl(hdat)
    cl_k.append(cl_sht)

cl_k = np.stack(cl_k)
wl = np.stack(wl)

np.save(f'./data/cl_k_GRF_L{int(GRF.L):d}_N{int(GRF.N):d}_Nq{int(Nskew):d}_Nl{int(Nl):d}_sims{int(num_sim):d}.npy', cl_k)

np.save(f'./data/wl_GRF_L{int(GRF.L):d}_N{int(GRF.N):d}_Nq{int(Nskew):d}_Nl{int(Nl):d}_sims{int(num_sim):d}.npy', wl)
