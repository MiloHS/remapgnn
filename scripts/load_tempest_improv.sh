#!/bin/bash

module purge --force
module load gcc/13.2.0
module load openmpi/5.0.2-gcc-13.2.0
module load hdf5/1.14.3-openmpi-5.0.2-gcc-13.2.0
module load netcdf-c/4.9.2-openmpi-5.0.2-gcc-13.2.0
module load netcdf-cxx/4.2-openmpi-5.0.2-gcc-13.2.0
module load intel-oneapi-mkl/2024.2.2-gcc-13.2.0

export AEC_LIB=/gpfs/fs1/soft/improv/software/spack-built/linux-rhel8-zen3/gcc-13.2.0/libaec-1.0.6-l466oxo/lib64
export BZIP2_LIB=/gpfs/fs1/soft/improv/software/spack-built/linux-rhel8-zen3/gcc-13.2.0/bzip2-1.0.8-nn42rty/lib
export BLOSC_LIB=/gpfs/fs1/soft/improv/software/spack-built/linux-rhel8-zen3/gcc-13.2.0/c-blosc-1.21.5-4yepsnc/lib64

export PATH=$HOME/remapgnn/software/tempestremap-install-improv/bin:$PATH
export LD_LIBRARY_PATH=$BLOSC_LIB:$BZIP2_LIB:$AEC_LIB:$HOME/remapgnn/software/tempestremap-install-improv/lib:${LD_LIBRARY_PATH:-}
