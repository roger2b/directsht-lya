#!/bin/bash
#
# Create a JAX CPU environment at NERSC.
# date: 05222024
# Need to module load these in the scripts that are calling
# code running under this environment as well.
#
# module load cudatoolkit
# module load cudnn
module load python
# Verify the versions of cudatoolkit and cudnn are compatible with JAX
# module list
#
# Set up the environment, cloning from nersc-mpi4py to get the
# proper MPI environment.
conda create --name jax-env-cpu --clone nersc-mpi4py
conda activate jax-env-cpu
conda update --all -y
conda install -c conda-forge numpy scipy ipykernel -y
python3 -Xfrozen_modules=off -m ipykernel \
        install --user --name jax-env-cpu --display-name JAX-env-gpu
#
conda install -c conda-forge numba healpy camb -y
#
python3 -m pip install --upgrade "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#
#
# test installation
# python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform); import jax.numpy as jnp; print(jnp.exp(2.34))"
#
python3 -m pip install -v git+https://github.com/martinjameswhite/directsht
#
# For testing purposes:
# python
# import jax
# jax.config.update("jax_enable_x64", True)
# from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)

# try:
#     jax_present = True
#     from jax import vmap, jit, devices
#     import jax.numpy as jnp
#     import sht.legendre_jax as legendre
#     import sht.interp_funcs_jax as interp
#     import sht.utils_jax as utils
#     from   sht.utils_jax import move_to_device
# except ImportError:
#     jax_present = False
#     move_to_device = lambda x, **kwargs: x  # Dummy definition for fallback
#     print("JAX not found. Falling back to NumPy.")
#     from numba import njit as jit
#     import sht.legendre_py as legendre
#     import sht.interp_funcs_py as interp
#     import sht.utils_py as utils