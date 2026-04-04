#!/usr/bin/env python
"""
Regenerate sims 0-19 (seeds 1000-1019) using the same worker approach as sims 20-99,
then combine into a clean 100-sim cache.
"""
import sys, os, gc, time, subprocess, json
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "notebooks"))

Nl = 500
chi_shift = 5000
num_qso = 9797

# Worker script
worker_code = f'''#!/usr/bin/env python
import sys, os
sys.path.insert(0, "{root}")
sys.path.insert(0, os.path.join("{root}", "notebooks"))
import numpy as np
from sht.sht import DirectSHT
from sht.lya_sfb import _alm2cl_complex
import GRF_class as my_GRF
import SHT_lya as sht_lya

seed = int(sys.argv[1])
Nl = {Nl}

GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=seed)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew={num_qso}, shift={chi_shift})
chi_grid = all_x[0, :]
delta_F = all_w_gal - 1.0
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, delta_F)
theta, phi = GRF.compute_theta_phi_skewer_start(
    all_x[:, 0], all_y[:, 0], all_z[:, 0])
sht_engine = DirectSHT(Nl, 2*Nl, 0.75)
w_j = FT_delta[:, 0]
alm_data = sht_engine(theta, phi, w_j)
cl = _alm2cl_complex(alm_data, Nl)
import json
print("RESULT:" + json.dumps(cl.tolist()), flush=True)
print("NSKEW:" + str(Nskew), flush=True)
'''

worker_file = os.path.join(root, "_worker_redo.py")
with open(worker_file, 'w') as f:
    f.write(worker_code)

# Run sims 0-19
num_redo = 20
cl_k_redo = np.zeros((num_redo, Nl))
t_total = time.time()

for i in range(num_redo):
    seed = 1000 + i
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, worker_file, str(seed)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"  sim {i} (seed={seed}) FAILED: {result.stderr[:300]}")
        sys.exit(1)
    for line in result.stdout.strip().split('\n'):
        if line.startswith("RESULT:"):
            cl_k_redo[i] = np.array(json.loads(line[7:]))
        elif line.startswith("NSKEW:"):
            nskew = int(line[6:])
    dt = time.time() - t0
    print(f"  sim {i:2d} (seed={seed}): cl[1]={cl_k_redo[i,1]:.4e}, Nskew={nskew}, dt={dt:.0f}s")

os.remove(worker_file)
print(f"\nRedone {num_redo} sims in {(time.time()-t_total)/60:.1f} min")

# Load old 100-sim cache and replace first 20
old = np.load(os.path.join(root, "notebooks", "data",
              "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz"))
cl_k_all = old['cl_k'].copy()  # (100, 500)
N = int(old['Nk'])
L_box = float(old['L'])

print(f"\nBefore fix: cl_k[0,1]={cl_k_all[0,1]:.4e}, cl_k[20,1]={cl_k_all[20,1]:.4e}")
cl_k_all[:num_redo] = cl_k_redo
print(f"After  fix: cl_k[0,1]={cl_k_all[0,1]:.4e}, cl_k[20,1]={cl_k_all[20,1]:.4e}")

# Also compute the window (wl) from one sim
result = subprocess.run(
    [sys.executable, '-c', f'''
import sys, os, numpy as np
sys.path.insert(0, "{root}")
sys.path.insert(0, os.path.join("{root}", "notebooks"))
from sht.sht import DirectSHT
from sht.lya_sfb import _alm2cl_complex
import GRF_class as my_GRF
import SHT_lya as sht_lya
import healpy as hp

GRF = my_GRF.PowerSpectrumGenerator(add_rsd=False, seed=1000)
all_x, all_y, all_z, all_w_rand, all_w_gal, Nskew = GRF.process_skewers(
    Nskew={num_qso}, shift={chi_shift})
chi_grid = all_x[0, :]
k_arr, FT_mask, FT_delta = sht_lya.compute_dft(chi_grid, all_w_rand, all_w_gal - 1.0)
theta, phi = GRF.compute_theta_phi_skewer_start(all_x[:, 0], all_y[:, 0], all_z[:, 0])
sht_engine = DirectSHT({Nl}, 2*{Nl}, 0.75)

# Window: SHT of FT_mask at k=0 (= N for all sightlines in periodic box)
w_rand_k0 = FT_mask[:, 0]
alm_rand = sht_engine(theta, phi, w_rand_k0)
wl = _alm2cl_complex(alm_rand, {Nl})

# Also uniform weight window
alm_ones = sht_engine(theta, phi, np.ones(Nskew))
wl_ones = _alm2cl_complex(alm_ones, {Nl})

import json
print("WL:" + json.dumps(wl.tolist()))
print("WL_ONES:" + json.dumps(wl_ones.tolist()))
print("NSKEW:" + str(Nskew))
print("NK:" + str(len(chi_grid)))
print("L:" + str(chi_grid[-1] - chi_grid[0] + (chi_grid[1]-chi_grid[0])))
'''],
    capture_output=True, text=True, timeout=600
)

for line in result.stdout.strip().split('\n'):
    if line.startswith("WL:"):
        wl = np.array(json.loads(line[3:]))
    elif line.startswith("WL_ONES:"):
        wl_ones = np.array(json.loads(line[8:]))
    elif line.startswith("NSKEW:"):
        nskew_final = int(line[6:])
    elif line.startswith("NK:"):
        nk_final = int(line[3:])
    elif line.startswith("L:"):
        L_final = float(line[2:])

print(f"\nNskew={nskew_final}, Nk={nk_final}, L={L_final}")
print(f"wl[0:3] = {wl[:3]}")
print(f"wl_ones[0:3] = {wl_ones[:3]}")

# Save clean cache
outfile = os.path.join(root, "notebooks", "data",
                       "Cell_GRF_L1380_N512_Nq9797_Nl500_sims100.npz")
np.savez(outfile,
         cl_k=cl_k_all, wl_k=wl[np.newaxis, :],
         wl_ones=wl_ones[np.newaxis, :],
         Nskew=nskew_final, Nk=nk_final, L=L_final)
print(f"\nSaved clean 100-sim cache to {outfile}")
print("Done!")
