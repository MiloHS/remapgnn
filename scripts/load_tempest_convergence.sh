#!/bin/bash

module purge --force 2>/dev/null || module purge

# Older LCRC stack that was used by run_tempest_one_pair.sh
module load gcc/8.5.0-ede2bck
module load openmpi/4.1.1-by6rv67
module load hdf5/1.10.7-zjwj3y2
module load netcdf-c/4.7.4-y6twdyt
module load netcdf-cxx/4.2-gr4zt6w
module load intel-oneapi-mkl/2023.1.0

export PATH=$HOME/remapgnn/software/tempestremap-install/bin:$PATH

export NETCDF_ROOT=$(dirname $(dirname $(which nc-config)))
export NETCDF_CXX_ROOT=/gpfs/fs1/soft/swing/spack-0.16.1/opt/spack/linux-ubuntu20.04-x86_64/gcc-8.5.0/netcdf-cxx-4.2-gr4zt6w
export MKL_LIB=${MKLROOT}/lib/intel64

export LD_LIBRARY_PATH=$HOME/remapgnn/software/tempestremap-install/lib:${NETCDF_ROOT}/lib:${NETCDF_CXX_ROOT}/lib:${MKL_LIB}:${LD_LIBRARY_PATH:-}
