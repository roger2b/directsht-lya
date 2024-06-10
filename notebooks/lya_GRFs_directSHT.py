import os, sys
import numpy as np
import multiprocessing as mp
import healpy as hp

from matplotlib_params_file import *
import GRF_class as my_GRF
import SHT_lya as sht_lya

sys.path.insert(0, '/Users/rdb/Desktop/research/lya/P3D_Cell/directsht-lya/')
from sht.sht import DirectSHT
from sht.mask_deconvolution import MaskDeconvolution

try:
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)
except:
    print("No GPU found")

####################################################

def initialize_sht(Nl, buffer_ells, xmax):
    Nx = 2 * Nl
    sht = DirectSHT(Nl, Nx, xmax)
    sht_randoms = DirectSHT(Nl + buffer_ells, Nx, xmax)
    print(f"For general, Direct SHT has Nl={sht.Nell}, Nx={Nx}, and xmax={xmax}")
    print(f"For randoms, Direct SHT has Nl={sht_randoms.Nell}, Nx={Nx}, and xmax={xmax}")
    return sht, sht_randoms

def process_simulations(GRF, num_qso, sht, sht_randoms):
    all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(Nskew=num_qso)
    all_theta, all_phi = GRF.compute_theta_phi_skewer_start(all_x[:,0], all_y[:,0], all_z[:,0])
    chi_grid = all_x[0,:]

    k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal)

    tdata, pdata, wdata = all_theta, all_phi, FT_delta
    trand, prand, wrand = all_theta, all_phi, FT_mask
    print(f'Nskew = {Nskew}, Nk = {GRF.N}, L = {GRF.L}')

    # Calculate the angular power spectrum of the randoms for the window function
    hran_for_wl = sht_randoms(trand, prand, wrand[:,0])
    wl = hp.alm2cl(hran_for_wl)

    # Calculate the angular power spectrum of the data-randoms
    hdat = sht(tdata, pdata, wdata[:,0] - wrand[:,0])
    cl_sht = hp.alm2cl(hdat)
    return cl_sht, wl, Nskew

def save_results(cl_k, wl,cl_fname, wl_fname):
    cl_k_filename = f'{cl_fname}'
    wl_filename = f'{wl_fname}'
    np.save(cl_k_filename, cl_k)
    np.save(wl_filename, wl)

def main():
    do_plots = False

    # Define GRF settings
    num_qso = int(1e+4)
    num_sim = 2
    add_rsd = False

    # Initialize SHT
    Nl = 500
    buffer_ells = 64
    xmax=5.0/8.0
    sht, sht_randoms = initialize_sht(Nl, buffer_ells=buffer_ells, xmax=xmax)

    # Process simulations
    cl_k = []
    wl = []
    for seed_idx in range(num_sim):
        GRF = my_GRF.PowerSpectrumGenerator(add_rsd=add_rsd, seed=seed_idx, verbose=False)
        cl_sht, wl_sht, Nskew = process_simulations(num_sim, num_qso, GRF, sht, sht_randoms)
        cl_k.append(cl_sht)
        wl.append(wl_sht)
    cl_k = np.stack(cl_k)
    wl = np.stack(wl)

    # Save results
    cl_fname = f'./data/cl_k_GRF_L{int(GRF.L):d}_N{int(GRF.N):d}_Nq{int(Nskew):d}_Nl{int(Nl):d}_sims{int(num_sim):d}.npy'
    wl_fname = f'./data/wl_GRF_L{int(GRF.L):d}_N{int(GRF.N):d}_Nq{int(Nskew):d}_Nl{int(Nl):d}_sims{int(num_sim):d}.npy'
    save_results(cl_k, wl, cl_fname, wl_fname)

if __name__ == "__main__":
    main()
