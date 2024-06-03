import numpy as np

def eval_legendre_numpy(_lambda, x, kk):
    """
    Evaluate the nth order Legendre polynomial at points x.

    Parameters:
    _lambda (int): Order of the Legendre polynomial.
    x (array_like): Points at which to evaluate the polynomial.
    kk (array_like): matrix from outer product of K_j * K_k

    Returns:
    ndarray: Values of the Legendre polynomial of order n at points x.
    """
    Plambda = np.zeros(_lambda)
    for n in range(_lambda):
        if n == 0:
            Pn = np.ones_like(x)
        elif n == 1:
            Pn = x
        else:
            Pn_minus_1 = np.ones_like(x)
            Pn = x
            for i in range(1, n):
                Pn_minus_2 = Pn_minus_1
                Pn_minus_1 = Pn
                Pn = ((2*i + 1) * x * Pn_minus_1 - i * Pn_minus_2) / (i + 1)
        Plambda[n] = np.sum(Pn * kk)
    return Plambda











##########################################################################################

def eval_legendre_numpy(n, x):
    """
    Evaluate the nth order Legendre polynomial at points x.

    Parameters:
    n (int): Order of the Legendre polynomial.
    x (array_like): Points at which to evaluate the polynomial.

    Returns:
    ndarray: Values of the Legendre polynomial of order n at points x.
    """
    if n == 0:
        Pn = np.ones_like(x)
    elif n == 1:
        Pn = x
    else:
        Pn_minus_1 = np.ones_like(x)
        Pn = x
        for i in range(1, n):
            Pn_minus_2 = Pn_minus_1
            Pn_minus_1 = Pn
            Pn = ((2*i + 1) * x * Pn_minus_1 - i * Pn_minus_2) / (i + 1)
    return Pn

