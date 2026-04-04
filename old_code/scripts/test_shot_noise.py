"""Estimate shot noise magnitude vs signal at different multipoles."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))
import numpy as np, healpy as hp
import GRF_class as my_GRF
import SHT_lya as sht_lya
from sht.sht import DirectSHT

Nl = 500
sht_eng = DirectSHT(Nl, 2*Nl, 0.75)
GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=1000)
ax, ay, az, wr, wg, Ns = GRF.process_skewers(Nskew=9797, shift=5000)
at, ap = GRF.compute_theta_phi_skewer_start(ax[:,0], ay[:,0], az[:,0])
chi = ax[0,:]
dF = wg - 1.0
ka, fm, fd = sht_lya.compute_dft(chi, wr, dF)
N = chi.size

w = fd[:, 0]  # weights at k=0
SN = np.sum(w**2) / (4 * np.pi)
cl = hp.alm2cl(sht_eng(at, ap, w))[:Nl]

print(f'Nskew={Ns}, N={N}')
print(f'Shot noise = sum(w^2)/(4pi) = {SN:.4e}')
print()

for ell in [0, 5, 50, 100, 200, 300, 400, 499]:
    frac = SN / cl[ell] * 100
    print(f'  ell={ell:3d}: cl={cl[ell]:.4e}, SN/cl={frac:6.1f}%, cl-SN={cl[ell]-SN:.4e}')
