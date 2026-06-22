#!/bin/bash
# Generate TempestRemap conservative FV->FV reference maps for ANY topology family pair. Pairs as args
# (SRC_to_TGT), e.g.: bash scripts/gen_maps_topo.sh CS-r32_to_MPAS-r4 CS-r32_to_CSRR-r64_lev2_tr
# Families resolved by mesh_path below: CS/ICOD/RLL (MIRA uniform), ICO (generated triangles),
# MPAS (MIRA Voronoi), CSRR (MIRA regionally-refined CS, key = remainder e.g. r64_lev2_tr).
set -o pipefail
cd "$HOME/remapgnn"
S=/gpfs/fs1/soft/swing/spack-0.16.1/opt/spack/linux-ubuntu20.04-x86_64
MKLDIR=$(find $S -path '*intel64/libmkl_rt.so.2' 2>/dev/null | head -1 | xargs dirname)
GCCDIR=$(find $S -path '*gcc-8.5.0*/lib64/libstdc++.so.6' 2>/dev/null | head -1 | xargs dirname)
export PATH="$HOME/remapgnn/software/tempestremap-install/bin:$PATH"
export LD_LIBRARY_PATH="$S/gcc-8.5.0/netcdf-c-4.7.4-y6twdyt/lib:$S/gcc-8.5.0/hdf5-1.10.7-zjwj3y2/lib:$S/gcc-8.5.0/openmpi-4.1.1-by6rv67/lib:$MKLDIR:$GCCDIR"
export TMPDIR="$HOME/remapgnn/tmp"; mkdir -p tmp maps_medium_improv data/gen_meshes/ICO
U=data/MIRA-Datasets/Meshes/UniformlyRefined
R=data/MIRA-Datasets/Meshes/RegionallyRefined

mesh_path() {
  local name=$1
  case $name in
    CS-r*)   echo "$U/CS/sample_NM16_O10_${name}_TPW_CFR_TPO_A1_A2.nc";;
    ICOD-r*) echo "$U/ICOD/sample_NM16_O10_${name}_TPW_CFR_TPO_A1_A2.nc";;
    RLL-r*)  echo "$U/RLL/sample_NM16_O10_${name}_TPW_CFR_TPO_A1_A2.nc";;
    ICO-r*)  local res=${name#ICO-r}; local f="data/gen_meshes/ICO/${name}.g"
             [ -f "$f" ] || GenerateICOMesh --res "$res" --file "$f" >/dev/null 2>&1; echo "$f";;
    MPAS-r*) echo "data/gen_meshes/MPAS/${name}_unit.nc";;   # pre-normalized to unit sphere (km->unit)
    CSRR-*)  echo "data/gen_meshes/CSRR/${name}_unit.nc";;    # pre-normalized to unit sphere
    HP-n*)   echo "data/gen_meshes/HEALPIX/${name}.nc";;      # pre-generated via healpix_to_scrip.py
    *) echo "";;
  esac
}

for PAIR in "$@"; do
  SRC_NAME=${PAIR%%_to_*}; TGT_NAME=${PAIR#*_to_}
  SRC=$(mesh_path "$SRC_NAME"); TGT=$(mesh_path "$TGT_NAME")
  OV="maps_medium_improv/ov_${PAIR}.nc"; MAP="maps_medium_improv/map_${PAIR}_conserve.nc"
  echo "=== $PAIR ==="; echo "  src=$SRC"; echo "  tgt=$TGT"
  if [ ! -f "$SRC" ] || [ ! -f "$TGT" ]; then echo "  MISSING MESH"; continue; fi
  [ -f "$OV" ]  || GenerateOverlapMesh --a "$SRC" --b "$TGT" --out "$OV" --out_format netcdf4 --method fuzzy >/dev/null 2>&1
  [ -f "$MAP" ] || GenerateOfflineMap --in_mesh "$SRC" --out_mesh "$TGT" --ov_mesh "$OV" \
      --in_type fv --out_type fv --in_np 1 --out_np 1 --correct_areas --out_map "$MAP" >/dev/null 2>&1
  echo "  ov=$(ls -la $OV 2>/dev/null | awk '{print $5}')B  map=$(ls -lh $MAP 2>/dev/null | awk '{print $5}')"
done
echo "GEN_MAPS_TOPO_DONE"
