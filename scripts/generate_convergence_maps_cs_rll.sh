#!/bin/bash
set -euo pipefail

cd "$HOME/remapgnn"
source "$HOME/remapgnn/scripts/load_tempest_convergence.sh"

mkdir -p maps_medium_improv logs_medium_improv tmp
export TMPDIR="$HOME/remapgnn/tmp"
export TMP="$HOME/remapgnn/tmp"
export TEMP="$HOME/remapgnn/tmp"

TRIPLES=(
  "32 90 180"
  "64 180 360"
  "128 360 720"
)

for triple in "${TRIPLES[@]}"; do
  read -r CS_R RLL_LAT RLL_LON <<< "$triple"

  CS_NAME="CS-r${CS_R}"
  RLL_NAME="RLL-r${RLL_LAT}-${RLL_LON}"

  CS_MESH="$HOME/remapgnn/data/MIRA-Datasets/Meshes/UniformlyRefined/CS/sample_NM16_O10_CS-r${CS_R}_TPW_CFR_TPO_A1_A2.nc"
  RLL_MESH="$HOME/remapgnn/data/MIRA-Datasets/Meshes/UniformlyRefined/RLL/sample_NM16_O10_RLL-r${RLL_LAT}-${RLL_LON}_TPW_CFR_TPO_A1_A2.nc"

  for DIR in "CS_to_RLL" "RLL_to_CS"; do
    if [[ "$DIR" == "CS_to_RLL" ]]; then
      SRC_NAME="$CS_NAME"
      TGT_NAME="$RLL_NAME"
      SRC="$CS_MESH"
      TGT="$RLL_MESH"
    else
      SRC_NAME="$RLL_NAME"
      TGT_NAME="$CS_NAME"
      SRC="$RLL_MESH"
      TGT="$CS_MESH"
    fi

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

    [[ -f "$SRC" ]] || { echo "ERROR missing source mesh: $SRC"; exit 1; }
    [[ -f "$TGT" ]] || { echo "ERROR missing target mesh: $TGT"; exit 1; }

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
done
