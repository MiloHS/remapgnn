#!/bin/bash
set -euo pipefail

cd "$HOME/remapgnn"
source "$HOME/remapgnn/scripts/load_tempest_convergence.sh"

mkdir -p maps_medium_improv logs_medium_improv tmp
export TMPDIR="$HOME/remapgnn/tmp"
export TMP="$HOME/remapgnn/tmp"
export TEMP="$HOME/remapgnn/tmp"

LEVELS=("$@")
if [[ ${#LEVELS[@]} -eq 0 ]]; then
  LEVELS=(16 32 64 128)
fi

for R in "${LEVELS[@]}"; do
  SRC_NAME="ICOD-r${R}"
  TGT_NAME="CS-r${R}"

  SRC="$HOME/remapgnn/data/MIRA-Datasets/Meshes/UniformlyRefined/ICOD/sample_NM16_O10_ICOD-r${R}_TPW_CFR_TPO_A1_A2.nc"
  TGT="$HOME/remapgnn/data/MIRA-Datasets/Meshes/UniformlyRefined/CS/sample_NM16_O10_CS-r${R}_TPW_CFR_TPO_A1_A2.nc"

  PAIR="${SRC_NAME}_to_${TGT_NAME}"
  OV="$HOME/remapgnn/maps_medium_improv/ov_${PAIR}.nc"
  MAP="$HOME/remapgnn/maps_medium_improv/map_${PAIR}_conserve.nc"
  LOG="$HOME/remapgnn/logs_medium_improv/${PAIR}_convergence.log"

  echo "================================================================================"
  echo "Pair: $PAIR"
  echo "Source: $SRC"
  echo "Target: $TGT"
  echo "Overlap: $OV"
  echo "Map: $MAP"
  echo "Log: $LOG"
  echo "================================================================================"

  if [[ ! -f "$SRC" ]]; then
    echo "ERROR: missing source mesh: $SRC"
    exit 1
  fi

  if [[ ! -f "$TGT" ]]; then
    echo "ERROR: missing target mesh: $TGT"
    exit 1
  fi

  {
    echo "Started: $(date)"

    if [[ ! -f "$OV" ]]; then
      echo "Generating overlap mesh..."
      GenerateOverlapMesh \
        --a "$SRC" \
        --b "$TGT" \
        --out "$OV" \
        --out_format netcdf4 \
        --method fuzzy
    else
      echo "Overlap already exists; skipping."
    fi

    if [[ ! -f "$MAP" ]]; then
      echo "Generating offline map..."
      GenerateOfflineMap \
        --in_mesh "$SRC" \
        --out_mesh "$TGT" \
        --ov_mesh "$OV" \
        --in_type fv \
        --out_type fv \
        --in_np 1 \
        --out_np 1 \
        --correct_areas \
        --out_map "$MAP"
    else
      echo "Map already exists; skipping."
    fi

    echo "Finished: $(date)"
    ls -lh "$MAP" "$OV"
  } 2>&1 | tee "$LOG"
done
