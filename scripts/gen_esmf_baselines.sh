#!/bin/bash
# Generate ESMF baseline weight files (SCRIP format) for RemapGNN pairs, drop-in for the audit.
# Methods: bilinear, conserve (1st order), conserve2nd (2nd order).
# Usage: bash gen_esmf_baselines.sh PAIR [PAIR ...]
set -o pipefail
cd "$HOME/remapgnn"
PY=/home/mschlittgenli/.conda/envs/remap_gpu/bin/python
P=/gpfs/fs1/soft/swing/manual/anaconda3/2020.11/pkgs
ESMF_BIN=$P/esmf-8.0.1-nompi_he31a43a_2/bin/ESMF_RegridWeightGen
# Prepend the MATCHING hdf5 1.10.6 (+ netcdf + esmf) so they win over the stale 1.10.4 also in cache;
# glob supplies remaining transitive deps. (First match wins -> avoids the HDF5 1.10.4/1.10.6 segfault.)
GLOB=$(ls -d $P/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="$P/hdf5-1.10.6-nompi_h6a2412b_1114/lib:$P/libnetcdf-4.7.4-nompi_h56d31a8_107/lib:$P/netcdf-fortran-4.5.3-nompi_h1a0d97b_101/lib:$P/esmf-8.0.1-nompi_he31a43a_2/lib:$GLOB:${LD_LIBRARY_PATH:-}"
mkdir -p tmp/scrip maps_medium_improv

for PAIR in "$@"; do
  MAP="maps_medium_improv/map_${PAIR}_conserve.nc"
  if [ ! -f "$MAP" ]; then echo "SKIP $PAIR (no np1 map)"; continue; fi
  SRC="tmp/scrip/${PAIR}_a.nc"; TGT="tmp/scrip/${PAIR}_b.nc"
  echo "=== $PAIR : extracting SCRIP grids ==="
  "$PY" scripts/_scrip_from_map.py "$MAP" a "$SRC" 2>&1 | grep -v "command not found" | tail -1
  "$PY" scripts/_scrip_from_map.py "$MAP" b "$TGT" 2>&1 | grep -v "command not found" | tail -1
  for M in bilinear conserve conserve2nd; do
    OUT="maps_medium_improv/map_${PAIR}_esmf_${M}.nc"
    EXTRA="--ignore_degenerate --ignore_unmapped"
    if [ "$M" != "bilinear" ]; then EXTRA="$EXTRA --norm_type dstarea --line_type greatcircle"; fi
    "$ESMF_BIN" --source "$SRC" --destination "$TGT" --weight "$OUT" --method "$M" $EXTRA >/dev/null 2>tmp/scrip/rwg_${PAIR}_${M}.err
    rc=$?
    sz=$(ls -lh "$OUT" 2>/dev/null | awk '{print $5}')
    echo "  esmf ${M}: exit=$rc weight=${sz:-MISSING} $( [ $rc -ne 0 ] && tail -1 tmp/scrip/rwg_${PAIR}_${M}.err )"
  done
done
echo "GEN_ESMF_DONE"
