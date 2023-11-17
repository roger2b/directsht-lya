import numpy as np
import numba as nb

try:
    jax_present = True
    from jax import device_put, vmap, jit
    import jax.numpy as jnp
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp

default_dtype = 'float64'
@nb.njit(parallel=True)
def bin_data(data, bin_indices, Nbins):
    '''
    Bin data into Nbins bins
    :param data: 1D np array of data to bin
    :param bin_indices: 1D np array with the bin index for each data point
    :param Nbins: number of bins
    '''
    bin_sums = np.zeros(Nbins, dtype=default_dtype)
    for i in nb.prange(len(data)):
        bin_sums[bin_indices[i]] += data[i]
    return bin_sums

def get_phi_dep(phi_data_sorted, m, which_part):
    '''
    Calculate the real or imaginary part of the phi dependence of the Ylm's
    :param phi_data_sorted: a 1d numpy array of phi data points (same length as theta_data_sorted)
    :param m: the m index of the Ylm
    :param which_part: 'cos' or 'sin' for the real or imaginary part of the phi dependence
    :return: a 1D numpy array of the real or imaginary part of the phi dependence of the Ylm's at each phi data point
    '''
    fn = function_map.get(which_part)
    # JIT compile the function and vectorize it
    if jax_present:
        vect_fn = vmap(jit(fn), in_axes=(0, None))
        out = np.array(vect_fn(phi_data_sorted, m), dtype=default_dtype)
    else:
        out = fn(phi_data_sorted, m)
    return out

@nb.njit(parallel=True)
def cosmphi_np(phi, m):
    return np.cos(m * phi)
@nb.njit(parallel=True)
def sinmphi_np(phi, m):
    return np.sin(m * phi)
def cosmphi_jax(phi, m):
    return jnp.cos(m * phi)
def sinmphi_jax(phi, m):
    return jnp.sin(m * phi)

# Map functions based on which_part
function_map = {
    'cos': cosmphi_jax if jax_present else cosmphi_np,
    'sin': sinmphi_jax if jax_present else sinmphi_np,
}
def getlm(lmax, szalm, i=None):
    """Get the l and m from index and lmax. From Healpy

    Parameters
    ----------
    lmax : int
      The maximum l defining the alm layout
    szalm : int
      The size of the alm array
    i : int or None
      The index for which to compute the l and m.
      If None, the function return l and m for i=0..Alm.getsize(lmax)
    """
    if i is None:
        i = np.arange(szalm)
    assert (
        np.max(i) < szalm
    ), "Invalid index, it should less than the max alm array length of {}".format(
        szalm
    )

    with np.errstate(all="raise"):
        m = (
            np.ceil(
                ((2 * lmax + 1) - np.sqrt((2 * lmax + 1) ** 2 - 8 * (i - lmax))) / 2
            )
        ).astype(int)
        l = i - m * (2 * lmax + 1 - m) // 2
    return (l, m)
