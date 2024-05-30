# Copyright (C) 2024 Roger de Belsunce & Steven Gratton 
import sys
import math
import numpy as np 
import healpy as hp
from numba import jit, njit, prange


_fastmath = False
_parallel = True
_nopython = False

print('_fastmath {}'.format(_fastmath))
print('_parallel {}'.format(_parallel))
print('_nopython {}'.format(_nopython))

@jit(nopython=_nopython, parallel=_parallel, fastmath=_fastmath)
def _compute_matrix(lmax, lnjp1, one_jp1, lng, g, one_g, wl, m):
    # Precompute logarithmic values
    for i in range(4 * lmax + 1):
        lnjp1[i] = np.log(i + 1)
        one_jp1[i] = 1.0 / (i + 1)

    # Compute logarithmic values
    lng[0] = 0.0
    for i in range(1, 2 * lmax + 1):
        lng[i] = lng[i - 1] + np.log((i - 0.5) / i)

    # Compute exponential values
    for i in range(2 * lmax + 1):
        g[i] = np.exp(lng[i])
        one_g[i] = np.exp(-lng[i])

    # Compute the matrix elements
    for i in prange(lmax + 1):
        for j in range(i, lmax + 1):
            tmp = 0.0
            for l in range(j - i, i + j + 1, 2):
                j_sum = i + j + l
                j_2 = j_sum // 2
                tmp += (2.0 * l + 1.0) * wl[l] * g[j_2 - i] * g[j_2 - j] * g[j_2 - l] * one_g[j_2] * one_jp1[j_sum]
            m[i * (lmax + 1) + j] = tmp * 0.25 / math.pi

    # Symmetrize the matrix
    for i in range(lmax + 1):
        for j in range(i, lmax + 1):
            m[j * (lmax + 1) + i] = m[i * (lmax + 1) + j]

    # Scale the matrix elements
    for i in range(lmax + 1):
        for j in range(lmax + 1):
            m[i * (lmax + 1) + j] *= (2.0 * j + 1.0)

    return m.reshape((lmax+1, lmax+1))

class CoupleMat:
    def __init__(self, 
                 Nl, 
                 wl,
                 verbose=False):
        
        self.Nl   = Nl
        self.lmax = Nl-1
        pad       = max(0,2*Nl-1-wl.size)
        self.wl  = np.pad(wl,(0,pad),'constant',constant_values=0)
        self.verbose = verbose
        
        if self.verbose: print('start computation Wigner 3j symbols')

        # Initialize arrays
        self.lnjp1 = np.zeros(4 * self.lmax + 1)
        self.lng = np.zeros(2 * self.lmax + 1)
        self.g = np.zeros(2 * self.lmax + 1)
        self.one_g = np.zeros(2 * self.lmax + 1)
        self.one_jp1 = np.zeros(4 * self.lmax + 1)
        self.m = np.zeros((self.lmax + 1) * (self.lmax + 1))

    def compute_matrix(self):
        return _compute_matrix(self.lmax, self.lnjp1, self.one_jp1, self.lng, self.g, self.one_g, self.wl, self.m)
    