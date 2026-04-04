import numpy as np
import scipy.linalg as spl
from scipy.special import eval_legendre
from numba import njit

#################### FFT functions ####################

def compute_dft(chi_grid, mask, delta, apply_inv_length=False):
    # Discrete FT of data
    # Ensure that chi_grid has a uniform step size
    dchi = chi_grid[1] - chi_grid[0]
    if apply_inv_length:
        mask  /= dchi
        delta /= dchi
    # Number of points
    N = len(chi_grid)

    # Define k-array for Fourier transform
    k_arr = np.fft.fftfreq(N, d=dchi)

    FT_mat = spl.dft(N)
    # Perform the DFT using matrix multiplication
    FT_mask  = np.dot(mask, FT_mat)
    FT_delta = np.dot(delta, FT_mat)
    return k_arr, np.real(FT_mask), np.real(FT_delta)

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

#################### theory module functions ####################

def compute_nhat(tdata, pdata):
    """
    Compute the nhat array given tdata and pdata. [the r_ij's cancel]

    Parameters:
    tdata (np.ndarray): Array of theta data with shape (Nskew,)
    pdata (np.ndarray): Array of phi data with shape (Nskew,)

    Returns:
    np.ndarray: The computed nhat array with shape (Nskew, 3)
    """
    # Compute sin and cos of tdata and pdata
    sin_tdata = np.sin(tdata)
    cos_tdata = np.cos(tdata)
    sin_pdata = np.sin(pdata)
    cos_pdata = np.cos(pdata)

    # Compute nhat
    nhat = np.column_stack((sin_tdata * cos_pdata, sin_tdata * sin_pdata, cos_tdata))
    return nhat

@njit(parallel=True)
def legendre_polynomials_sum(n, x, kk):
    """
    Compute the Legendre polynomials up to the nth order at points x and return their sums.

    Parameters:
    n (int): Order up to which the Legendre polynomials are computed.
    x (array_like): NxN points at which to evaluate the polynomials.
    kk (array_like): matrix from outer product of K_j * K_k

    Returns:
    ndarray: An array where each entry is the sum of the Legendre polynomial of the corresponding order.

    To Do: use symmetry of x to reduce computation time
    """
    x = np.asarray(x)
    P_sum = np.zeros(n + 1)
    
    # Initial polynomials P0 and P1
    P0 = np.ones_like(x)
    P_sum[0] = np.sum(P0 * kk)
    
    if n > 0:
        P1 = x
        P_sum[1] = np.sum(P1 * kk)
    
    # Recurrence relation for higher-order polynomials
    if n > 1:
        Pn_minus_2 = P0
        Pn_minus_1 = P1
        for k in range(2, n + 1):
            Pn = ((2*k - 1) * x * Pn_minus_1 - (k - 1) * Pn_minus_2) / k
            P_sum[k] = np.sum(Pn * kk)
            Pn_minus_2 = Pn_minus_1
            Pn_minus_1 = Pn
    
    return P_sum

def compute_PLKjKk(lambda_idx,cos_theta_njnk, KjKk):
    """
    Compute PLKjKk given input data.

    Parameters:
    lambda_idx (int): Index for the Legendre polynomial
    cos_theta_njnk (np.ndarray): matrix of cos(theta) of all j and k qso pairs with shape (Nskew, Nskew)
    KjKk (np.ndarray): matrix of KjKk values with shape (Nskew, Nskew)
    """
 
    # To Do: use https://numba-special.readthedocs.io/en/latest/index.html to compute the Legendre polynomials quickly
    # TO DO: compute with GPU code

    # Get the upper triangular indices
    upper_tri_indices = np.triu_indices(cos_theta_njnk.shape[0])
    # Compute the Legendre polynomial values for the upper triangular part
    legendre_vals_upper_tri = eval_legendre(lambda_idx, cos_theta_njnk[upper_tri_indices])
    # Initialize the full Legendre values matrix
    legendre_vals = np.zeros_like(cos_theta_njnk)
    # Fill the upper triangular part with computed values
    legendre_vals[upper_tri_indices] = legendre_vals_upper_tri
    # Since the matrix is symmetric, fill the lower triangular part
    legendre_vals = legendre_vals + legendre_vals.T - np.diag(np.diag(legendre_vals))

    return np.sum(legendre_vals * KjKk)

def compute_PLKjKk_parallel(args):
    lambda_idx, cos_theta_njnk, KjKk = args
    return compute_PLKjKk(lambda_idx, cos_theta_njnk, KjKk)

def F_ell_L_lambda(ell, L, lambda_):
    # Compute the prefactor
    prefactor = (2.*L + 1.) * (2.*lambda_ + 1.) / (4. * np.pi)**2
    # Compute the Wigner 3-j symbol
    wigner_3j_value = MD.w3j000(ell, L, lambda_)**2
    return prefactor * wigner_3j_value

def compute_Wigner3j_symmetry_range(j1, j2):
    """
    Compute the range for j3 given j1 and j2: 
    |j1 - j2| <= j3 <= j1 + j2
    """
    return abs(j1 - j2), abs(j1 + j2)
############### theory functions ####################