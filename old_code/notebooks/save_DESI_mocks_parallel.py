
#!/usr/bin/env python

import numpy as np
import sys, os, fitsio, glob
from astropy.io import fits
import matplotlib.pyplot as plt
import multiprocessing as mp
import re 

basedir = '/pscratch/sd/r/rmvd2/eBOSS_DR16/lya/London_mocks'

# eBOSS mocks 
mock_c='eboss-0.2' # contaminated mocks
# mock_c='eboss-0.0'# uncontaminated mocks

# Specify the number of CPUs to be used
num_cpus = 30

###### constants ##########
LYA_CONST = 1215.67


def extract_last_integer(input_string):
    # Use regular expression to find all integers in the string
    integers = re.findall(r'\b\d+\b', input_string)
    
    # Check if any integers are found
    if integers:
        # Return the last integer as an integer
        return int(integers[-1])
    else:
        # Return an appropriate value (e.g., None) if no integer is found
        return None

def split_directory_filename(fsave):
    directory = fsave.rsplit('/', 1)[0]
    filename = fsave.rsplit('/', 1)[1]
    return directory, filename


def create_dict_mocks(ra,dec,z_list, targetid,weight_gal, weight_rand, healpix_pix, save_dir, Nx, subsample=False):
    """
    Create a dictionary of mock data and save it to a file.

    Parameters:
    - ra (array-like): Array of right ascension values in radians.
    - dec (array-like): Array of declination values in radians.
    - z_list (array-like): Array of redshift values.
    - targetid (array-like): Array of target ID values.
    - weight_gal (array-like): Array of galaxy weight values: weight*delta
    - weight_rand (array-like): Array of random weight values: weight
    - healpix_pix (array-like): Array of HEALPIX pixel values.
    - save_dir (str): Path to the file where the data will be saved.
    - Nx (int): Number of data points.
    - subsample (bool, optional): Whether to subsample the data. Defaults to False.

    Returns:
    - bool: True if the data is saved successfully, False otherwise.
    """

    data_type=[('redshift', np.float64),\
               ('targetid', np.float64),\
               ('RA',       np.float64),\
               ('DEC',      np.float64),\
               ('w_gal',    np.float64),\
               ('w_rand',   np.float64),\
               ('HEALPIX',  np.float64)]
    dataout=np.zeros(Nx,dtype=data_type)
    
    if subsample:
        print('before subsampling:', ra.size)
        #choose a random subsample from data
        # % of QSOs
        N = np.int32(np.unique(targetid).size/100*10)
        print(N)
        indexes = np.unique(targetid, return_index=True)[1]
        unique_TID = np.array([targetid[index] for index in sorted(indexes)])

        np.random.seed(10)
        random_indices=np.random.choice(unique_TID,N,replace=False)
        mask_subsample=np.in1d(targetid,random_indices)
        ra            = ra[mask_subsample]
        dec           = dec[mask_subsample]
        z_list        = z_list[mask_subsample]
        targetid      = targetid[mask_subsample]
        weight_gal    = weight_gal[mask_subsample]
        weight_rand   = weight_rand[mask_subsample]
        print('after subsampling:', ra.size)
        if ra.size < 100:
            print('do not save this HEALPIX PIXEL')
            return False
        save_dir = save_dir[:-4]+'_sub.npy'
        dataout=np.zeros(ra.size,dtype=data_type)

    dataout['RA']       = ra
    dataout['DEC']      = dec
    dataout['redshift'] = z_list
    dataout['targetid'] = targetid
    dataout['w_gal']    = weight_gal
    dataout['w_rand']   = weight_rand
    dataout['HEALPIX']  = healpix_pix

    if(os.path.isfile(save_dir)):
        print('file exists: STOP \n %s'%(save_dir))
    else:
#        _dir, _name = split_directory_filename(save_dir)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        print('save \n %s'%(save_dir))
        np.save(save_dir, dataout, allow_pickle=True)


def save_eBOSSDR16_deltas(filelist, save_name, save_file=False):
    if os.path.isfile(save_name):
        print('file exists: STOP \n %s'%(save_name))
        return 0,0,0,0,0,0,0
    else:
        ra = []
        dec = []
        zpix = []
        tid = []
        w_gal = []
        deltas = []
        w_rand = []
        hp_pix = []
        
        for filename in filelist:
            hdul = fitsio.FITS(filename)
            HEALPIX_PIX = extract_last_integer(filename)
            # print(HEALPIX_PIX)
            
            for hdu in hdul[1:]:
                header = hdu.read_header()
                data = hdu.read()
                Npix = header['NAXIS2']
                assert Npix, (np.isfinite(data['DELTA']) & np.isfinite(data['WEIGHT']) ).sum()
                w_gal.append(data['WEIGHT']*data['DELTA'])
                w_rand.append(data['WEIGHT'])
                ra.append(np.repeat(header['RA'],Npix))
                dec.append(np.repeat(header['DEC'],Npix))
                zpix.append(10.**data['LOGLAM']/LYA_CONST-1)
                tid.append(np.repeat(header['THING_ID'],Npix))
                hp_pix.append(np.repeat(HEALPIX_PIX,Npix))

        all_ra    = np.concatenate(ra, axis=0)
        all_dec   = np.concatenate(dec, axis=0)
        all_ra    = np.rad2deg(all_ra)
        all_dec   = np.rad2deg(all_dec)

        all_z     = np.concatenate(zpix, axis=0)
        all_tid   = np.concatenate(tid, axis=0)
        all_w_gal = np.concatenate(w_gal, axis=0)
        all_w_rand= np.concatenate(w_rand, axis=0)
        all_hp_pix= np.concatenate(hp_pix, axis=0)

        #save_name = '/pscratch/sd/r/rmvd2/DESI/mocks/london/{}{}/{}/DESI_mocks_all_lya_data_RA_DEC_Z_TID_WGAL_WRAND_PIX{}.npy'.format(mocks_ver, mocks_name,which_delta,HEALPIX_PIX)
        #save_name = '/pscratch/sd/r/rmvd2/eBOSS_DR16/lya/eBOSS_DR16_all_lya_data_RA_DEC_Z_WGAL_WRAND_PIX{}.npy'.format(HEALPIX_PIX)
        #save_name = '/pscratch/sd/r/rmvd2/eBOSS_DR16/lya/eBOSS_DR16_all_lya_data_RA_DEC_Z_WGAL_WRAND_ALLPIX_FLOAT64.npy'
        # print(save_name)

        if save_file:
                create_dict_mocks(all_ra,
                                all_dec,
                                all_z,
                                all_tid,
                                all_w_gal,
                                all_w_rand, 
                                all_hp_pix,
                                save_dir=save_name, 
                                Nx=all_ra.size,
                                subsample=False)

        return all_ra, all_dec, all_z, all_tid, all_w_gal, all_w_rand, all_hp_pix


def process_mock_batch(idx):
    mock_version=f"v9.0.{idx}"
    dir =f"/global/cfs/projectdirs/desi/science/lya/picca_on_mocks/london/v9.0/{mock_version}/{mock_c}/deltas/"
    filelist = np.sort(glob.glob(dir+'delta-*.fits.gz'))
    print(filelist[0]+'\n')

    fsave=f"{basedir}/{mock_version}/{mock_c}/eBOSS_mock_all_lya_data_RA_DEC_Z_TID_WGAL_WRAND_HPX.npy"
    all_ra, all_dec, all_z, all_tid, all_w_gal, all_w_rand, all_hp_pix = save_eBOSSDR16_deltas(filelist, save_name=fsave, save_file=True)

# run in series
# for idx in range(1,100):
#    process_mock_batch(idx)



# Create a pool of worker processes with the specified number of CPUs
pool = mp.Pool(num_cpus)

# Map the process_mock function to the range of mock versions
pool.map(process_mock_batch, np.arange(0, 100, 1))

# Close the pool to prevent any more tasks from being submitted
pool.close()

# Wait for all worker processes to finish
pool.join()


def convert_coordinates(data, omega_m=0.31, omega_k=0., w_dark_energy=-1.):
    """
    Convert the coordinates from RA, DEC, Z to x, y, z.

    Parameters:
    - data: numpy structured array containing the data with columns 'redshift', 'targetid', 'RA', 'DEC', 'w_gal', 'w_rand', 'HEALPIX'
    - omega_m: float, optional, default=0.31, the matter density parameter
    - omega_k: float, optional, default=0., the curvature density parameter
    - w_dark_energy: float, optional, default=-1., the equation of state parameter for dark energy

    Returns:
    - all_x: numpy array, the x-coordinates in Mpc/h
    - all_y: numpy array, the y-coordinates in Mpc/h
    - all_z: numpy array, the z-coordinates in Mpc/h
    """
    all_redshift = data['redshift']
    all_tid = data['targetid']
    all_ra = data['RA']
    all_dec = data['DEC']
    all_w_gal = data['w_gal']
    all_w_rand = data['w_rand']
    all_healpix = data['HEALPIX']

    # print('convert to x,y,z from RA,DEC')
    from astropy.constants import c as c_light
    import sys
    import astropy.units as u
    sys.path.insert(0, '/global/homes/r/rmvd2/lya_P3D/HIPSTER_20012024/HIPSTER/python/wcdm/')
    import wcdm
    ################### convert to x,y,z #########################
    #print("Converting z to comoving distances:")
    all_comoving_radius = wcdm.coorddist(all_redshift, omega_m, w_dark_energy, omega_k)

    # Convert to Mpc/h
    H_0_h = 100 * u.km / u.s / u.Mpc  # to ensure we get output in Mpc/h units
    H_0_SI = H_0_h.to(1. / u.s)
    comoving_radius_Mpc = ((all_comoving_radius / H_0_SI * c_light).to(u.Mpc)).value

    # Convert to polar coordinates in radians
    all_phi_rad = all_ra * np.pi / 180.
    all_theta_rad = np.pi / 2. - all_dec * np.pi / 180.

    # Now convert to x,y,z coordinates
    all_z = comoving_radius_Mpc * np.cos(all_theta_rad)
    all_x = comoving_radius_Mpc * np.sin(all_theta_rad) * np.cos(all_phi_rad)
    all_y = comoving_radius_Mpc * np.sin(all_theta_rad) * np.sin(all_phi_rad)
    
    return all_x, all_y, all_z, all_w_gal, all_w_rand, all_tid, all_healpix


def process_mock(mock_version):
    fin = f"{basedir}/{mock_version}/{mock_c}/eBOSS_mock_all_lya_data_RA_DEC_Z_TID_WGAL_WRAND_HPX.npy"
    try:
        data = np.load(fin)
        print(fin+'\n')
        all_x, all_y, all_z, all_w_gal, all_w_rand, all_tid, all_healpix = convert_coordinates(data)

        print('size:', all_x.size)

        select_NGC = True
        if select_NGC:
            # for mocks all_x < 0 --> NGC (for data it is the opposite)
            all_y = all_y[all_x<0]
            all_z = all_z[all_x<0]
            all_w_rand = all_w_rand[all_x<0]
            all_w_gal = all_w_gal[all_x<0]
            all_tid = all_tid[all_x<0]
            all_healpix = all_healpix[all_x<0]  
            all_x = all_x[all_x<0]

        print('NGC size: %s \n'%all_x.size)
        fout = f"{basedir}/lya_{mock_version}_{mock_c}_mock_skewers.xyzwrwgth.NGC"

        print("save data \n")
        # Now write to file:
        with open(fout,"w+") as outfile:
            for p in range(all_w_rand.size):
                outfile.write("%.8f %.8f %.8f %.8f %.8f %d %.8f\n" %(all_x[p],
                                                                    all_y[p],
                                                                    all_z[p],
                                                                    all_w_rand[p],
                                                                    (all_w_gal[p]+all_w_rand[p]), # this is w(1+d) !!! 
                                                                    all_tid[p],
                                                                    all_healpix[p]))
        print("Output positions (of length %d) written successfully to %s!"%(all_w_rand.size,fout))
    except:
        print('error for %s \n'%fin)
        pass

# Create a pool of worker processes with the specified number of CPUs
pool = mp.Pool(num_cpus)

# Map the process_mock function to the range of mock versions
pool.map_async(process_mock, [f"v9.0.{idx}" for idx in range(100)])

# Close the pool to prevent any more tasks from being submitted
pool.close()

# Wait for all worker processes to finish
pool.join()
