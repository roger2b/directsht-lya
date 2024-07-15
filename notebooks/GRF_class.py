import camb
from camb import model
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import interp1d

class PowerSpectrumGenerator:
    def __init__(self, 
                 h=0.6770, 
                 Omega_b=0.04904, 
                 Omega_m=0.3147, 
                 ns=0.96824, 
                 As=2.10732e-9, 
                 mnu=0.0, 
                 N=512, 
                 L=1380.0, 
                 bins=30,
                 add_rsd=True,
                 my_bias=1.,
                 my_beta=1.5, 
                 seed=1000,
                 verbose=False):
    
        self.verbose = verbose
        # define cosmology
        self.h = h
        self.Omega_b = Omega_b
        self.Omega_m = Omega_m
        self.ns = ns
        self.As = As
        self.mnu = mnu
        self.H0 = h * 100
        self.ombh2 = Omega_b * h ** 2
        self.omch2 = (Omega_m - Omega_b) * h ** 2
        self.pars = camb.CAMBparams()

        # Fourier modes calculation
        self.N = N
        self.L = L
        self.bins = bins
        self.k_f = 2 * np.pi / L
        if self.verbose: print('Fundamental mode k_f=', self.k_f)

        self.kfft = np.fft.fftfreq(N) * 2.0 * np.pi / self.L * self.N
        self.kminfft = np.amin(np.abs(self.kfft))
        self.kmaxfft = np.amax(np.abs(self.kfft))
        self.kmax = np.sqrt(3) * self.kmaxfft
        if self.verbose: print('max mode k_f=', self.kmax)
        
        self.kmin = self.kminfft
        self.kNy = self.kmaxfft
        if self.verbose: print('Nyquist frequency:', self.kNy)
        
        self.k_bins = np.geomspace(1.e-4, self.kmax, bins + 1)
        self.k_bin_ctrs = (self.k_bins[1:] + self.k_bins[:-1]) / 2.0
        
        if self.verbose: 
            print('ratio of Nyquist to fundamental freq', self.kNy / self.k_f)
            print('ratio of max to Nyquist freq', self.kmax / self.kNy)

        # RSD
        self.add_rsd = add_rsd

        self.my_bias = my_bias
        if self.add_rsd:
            self.my_beta = my_beta
        else: 
            self.my_beta = 0.
        self.seed = seed
        if self.verbose: 
            print('beta', self.my_beta)
            print('bias', self.my_bias)
            print('RSD:', self.add_rsd)
            print('seed:', self.seed)
            
        if self.verbose: print('get power spectrum from CAMB')
        self.kh_lin, self.z_lin, self.pk_lin = self.get_linear_matter_power_spectrum()

        self.plin = interp1d(self.kh_lin, self.pk_lin[0,:], fill_value="extrapolate")
        if self.verbose: print('define k grid')
        all_ks_3d = get_ks3d(self.L, self.N, self.kfft)

        self.pk_all = self.plin(all_ks_3d)
        if self.verbose: print('compute amplitudes')
        self.amplitudes = self.compute_amplitudes3d()

        self.amplitudes_squared = np.real(self.amplitudes*np.conj(self.amplitudes))
        self.dens = self.density_field()

    def compute_power_spectrum(self):
        if self.verbose: print('compute power spectrum')
        k_eff, Pk, Pk2, Pk4, counts, totcounts = compute_Pk(self.N, self.amplitudes_squared, self.bins, self.k_bins, self.kfft)
        return k_eff, Pk, Pk2, Pk4, counts, totcounts

    def set_cosmology(self, z=[2.4]):
        self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu)
        self.pars.InitPower.set_params(ns=self.ns, As=self.As)
        self.pars.set_matter_power(redshifts=z, kmax=50)
        self.pars.NonLinear = model.NonLinear_none

    def get_linear_matter_power_spectrum(self, z=[2.4]):
        self.set_cosmology(z)
        results = camb.get_results(self.pars)
        kh_lin, z_lin, pk_lin = results.get_matter_power_spectrum(minkh=1e-4, 
                                                                  maxkh=10, 
                                                                  npoints=200)
        return kh_lin, z_lin, pk_lin

    def plot_power_spectrum(self, kh_lin, pk_lin, z=[2.4]):
        for i, (redshift, line) in enumerate(zip(z, ['-', '--'])):
            plt.loglog(kh_lin, pk_lin[i, :], ls=line)
        plt.xlabel('k/h Mpc')
        plt.legend([f'lin z={z[0]:1.1f}', 'nbodykit Plin'], loc='lower left')
        plt.title(f'Matter power at z={z}')
        plt.show()

    def compute_amplitudes3d(self):
        if self.verbose: print('Include anisotropies', self.add_rsd)
        # if aniso:
        mu = compute_mu3d(self.L, self.N, self.kfft)
        if self.verbose: 
            print('bias', self.my_bias)
            print('beta', self.my_beta)
        return self.my_bias * get_amplitudes3d(self.L, self.N, self.pk_all, self.my_beta, self.seed, mu)

    def get_multipoles(self):
        p0 = self.my_bias**2 * (1. + 2. * self.my_beta / 3. + self.my_beta**2. / 5.) * self.pk_lin[0,:]
        p2 = self.my_bias**2 * (4. * self.my_beta / 3. + 4. * self.my_beta**2. / 7.) * self.pk_lin[0,:]
        p4 = self.my_bias**2 * (8. * self.my_beta**2. / 35.) * self.pk_lin[0,:]
        return p0, p2, p4

    def density_field(self, d=3):
        if self.verbose: print('Transforming amplitudes to density field')
        boxvol = float(self.L)**d
        pix    = (float(self.L)/float(self.N))**d
        dens   = np.fft.ifftn(self.amplitudes) * boxvol ** (1./2.) / pix
        return np.real(dens)
    
    def process_skewers(self, Nskew, shift = 5e+3):
        """
        Process skewers for a 3D grid and compute related fields.
    
        Parameters:
        N (int): The number of points along each axis of the 3D grid.
        L (float): The length of the side of the 3D box.
        dens (np.array): The density field of the 3D grid.
        shift (float): The displacement value for the z-coordinates.
    
        Returns:
        tuple: Processed coordinates (all_x, all_y, all_z), all_w_rand, all_w_gal, and all_hpx.
        """
        # Generate 3D meshgrid coordinates
        coords = np.meshgrid(*[np.linspace(0, self.L, self.N) for _ in range(3)])
    
        # Generate skewers
        # always extract the same skewers
        np.random.seed(100)
        inds = np.unique(np.random.randint(0, self.N, size=(Nskew, 2)), axis=0)
        Nskew = len(inds)
        if self.verbose: print("N_skew = %d / %d" % (Nskew, self.N**2))
    
        # Compute density skewers and add 1
        dens_lya = self.dens[inds[:,0], inds[:,1], :] 
        skewer_field = dens_lya + 1.
    
        # Displace the box in the z-direction
        print("Displacing box by %.3e" % shift)
        tmp_all_x = coords[0][inds[:,0], inds[:,1], :]  # + shift
        tmp_all_y = coords[1][inds[:,0], inds[:,1], :]  # + shift
        tmp_all_z = coords[2][inds[:,0], inds[:,1], :] + shift
    
        # Swap coordinates
        all_x = tmp_all_z.copy()
        all_y = tmp_all_y.copy()
        all_z = tmp_all_x.copy()
    
        # Define mask (in draft: K_j(chi))
        all_w_rand = np.ones_like(all_x, dtype='float')
        
        # Define delta (in draft: delta_F(chi))
        all_w_gal = np.asarray(skewer_field, dtype='float')
        
        return all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew

    def compute_theta_phi_skewer_start(self, x,y,z):
        # only compute the Theta, Phi angle for the *first* pixel of the Lya skewer

        xsq = x ** 2.
        ysq = y ** 2.

        s = (xsq + ysq) ** 0.5

        # convert to degrees
        phi = np.arctan2(y, x)
        theta = np.arctan2(s, z)
    
        return theta, phi

@jit(nopython=True)
def compute_mu3d(L,n, kfft):
    # print('compute mu = k_los / |k|')
    mu  = np.ones((n,n,n))
    for i in range(n):
        for j in range(n):
            for l in range(n):
                kx = kfft[i]
                ky = kfft[j]
                kz = kfft[l]
                k_sum = np.sqrt(kx**2.0 + ky**2.0 + kz**2.0)  # module of distance of origin to compute P(k)
                mu[i, j, l] = kz / (1e-10 + k_sum)  # angle to z axis
    return mu


@jit(nopython=True)
def get_ks3d(L, n, kfft):
    kk = np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for l in range(n):
                kx = kfft[i]
                ky = kfft[j]
                kz = kfft[l]
                k_sum = np.sqrt(kx**2.0 + ky**2.0 + kz**2.0) # module of distance of origin to compute P(k)
                kk[i,j,l] = k_sum
    return kk

@jit(nopython=True)
def get_amplitudes3d(L, n, Pk, beta, seed, mu):
    np.random.seed(seed)
    areal = np.zeros((n,n,n))
    aim = np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for l in range(np.int32(n/2+1)):
                pk = Pk[i,j,l]
                if (i==0 or i==np.int32(n/2)) and (j==0 or j==np.int32(n/2)) and (l==0 or l==np.int32(n/2)):
                    areal[i,j,l] =  np.random.normal(0., np.sqrt(pk/2.)) * ( 1. + beta * (mu[i,j,l])**2)
                    aim[i,j,l]   = 0
                else:
                    areal[i,j,l] =  np.random.normal(0., np.sqrt(pk/2.)) * ( 1. + beta * (mu[i,j,l])**2)
                    aim[i,j,l]   =  np.random.normal(0., np.sqrt(pk/2.)) * ( 1. + beta * (mu[i,j,l])**2)
                    areal[(n-i)%n][(n-j)%n][(n-l)%n] = areal[i,j,l]
                    aim[(n-i)%n][(n-j)%n][(n-l)%n]   = -aim[i,j,l]
    a = areal + 1.0j*aim
    return a


@jit(nopython=True)
def compute_Pk(n, amplitudes_squared, bins, k_bins, kfft):
    Phat = np.zeros(bins)
    Phat2 = np.zeros(bins)
    Phat4 = np.zeros(bins)
    k_eff = np.zeros(bins)
    counts = np.zeros(bins) # initialize counts of mode
    totcounts=0
    for i in range(n):
        for j in range(n): 
            for l in range(n):
                totcounts+=1
                kx=kfft[i]
                ky=kfft[j]
                kz=kfft[l]
                k_sum=np.sqrt(kx**2.0 + ky**2.0 + kz**2.0) # module of distance of origin to compute P(k)
                mu   = kz/(1e-10+k_sum) # angle to z axis
                l2mu = (3.*mu**2.-1.)/2.
                l4mu = (35.*mu**4.-30.*mu**2. + 3.)/8.

                for m in range(bins):
                    if (k_sum>=k_bins[m] and k_sum<k_bins[m+1]): # check if in bin
                        Phat[m]  += amplitudes_squared[i,j,l] # measure monopole
                        Phat2[m] += amplitudes_squared[i,j,l]*(2.*2.+1.)*l2mu # measure quadrupole
                        Phat4[m] += amplitudes_squared[i,j,l]*(2.*4.+1.)*l4mu # measure hexadecapole
                        k_eff[m] += k_sum
                        counts[m]+= 1.

    # normalization 
    Phat /=counts # P over number of modes
    Phat2/=counts
    Phat4/=counts
    k_eff/=counts

    print(sum(counts))
    print(totcounts)
    print(n**3)
    return k_eff, Phat, Phat2, Phat4, counts, totcounts
