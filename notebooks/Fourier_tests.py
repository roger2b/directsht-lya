# test FFT from numpy re units

import numpy as np

data = all_w_gal[0]
nX = GRF.N
sizeX = GRF.L

lx = np.zeros(nX)
lx[:int(nX/2+1)] = 2.*np.pi/sizeX * np.arange(nX//2+1)
lx[int(nX/2+1):] = 2.*np.pi/sizeX * np.arange(-nX//2+1, 0, 1)
# ly = 2.*np.pi/sizeY * np.arange(nY//2+1)
# lx, ly = np.meshgrid(lx, ly, indexing='ij')

# l = np.sqrt(lx**2 + ly**2)
dataFourier = np.zeros((nX))
dX = float(sizeX)/(nX-1)
x = dX * np.arange(nX)

def fourier(data):
    """Fourier transforms, notmalized such that
    f(k) = int dx e-ikx f(x)
    f(x) = int dk/2pi eikx f(k)
    """
    # use numpy's fft
    result = np.fft.rfftn(data)
    result *= dX
    return result

# Gaussians and tests for Fourier transform conventions
def genGaussian(meanX=0., sigma1d=1.):
    result = np.exp(-0.5*((x-meanX)**2 )/sigma1d**2)
    result /= 2.*np.pi*sigma1d**2
    return result

def genGaussianFourier(meanLX=0., sigma1d=1.):
    result = np.exp(-0.5*((lx-meanLX)**2)/sigma1d**2)
    result /= 2.*np.pi*sigma1d**2
    return result

def testFourierGaussian():
    """tests that the FT of a Gaussian is a Gaussian,
    with correct normalization and variance
    """
    # generate a quarter of a Gaussian
    sigma1d = sizeX / 10.
    data = genGaussian(sigma1d=sigma1d)
    # show it
    plt.plot(data, label='Gaussian');plt.show()

    # fourier transform it
    dataFourier = fourier(data)
    # print(dataFourier.shape)
    # lxLeft = 2.*np.pi/sizeX * (np.arange(-nX//2+1, 1, 1) - 0.5)
    # print(lxLeft.shape)
    # plt.plot(lxLeft, dataFourier[:-1], label='FFT ( gaussian)')

    # computed expected Gaussian
    expectedFourier = genGaussianFourier(sigma1d=1./sigma1d)
    expectedFourier *= 2.*np.pi*(1./sigma1d)**2
    expectedFourier /= 4.   # because only one quadrant in real space

    #plotFourier(data=dataFourier/expectedFourier-1.)

    # compare along one axis
    plt.plot(dataFourier, 'k', label='FFT ( gaussian)')
    plt.plot(expectedFourier, 'r', label='expected Fourier')
    plt.legend()

    plt.show()

testFourierGaussian()