# Using the RemapGNN model on a new mesh pair

This is the quickstart for **applying the released model** to remap a field from one
spherical mesh to another. For the longer reference (feature details, visualization,
troubleshooting) see [`INFERENCE.md`](INFERENCE.md).

**What you need:** just the **two mesh files** (source and target), each with cell
longitudes/latitudes and, ideally, a cell-area variable. Inference uses **no**
TempestRemap, ESMF, or supermesh — only the mesh geometry and the trained network.
A GPU is optional (it only speeds up the GNN forward pass); **CPU works**.

---

## 1. Install

```bash
git clone https://github.com/MiloHS/remapgnn.git
cd remapgnn
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # torch, numpy, scipy, xarray, pandas, netCDF4, pyarrow
pip install -e .
```
For a CUDA machine, install the matching torch build first:
`pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`.
Tested with Python 3.11, torch 2.6.

## 2. Get the model weights (GitHub Release asset)

The weights are **not** in the git repo (they ship as a Release asset). Download the
two files from the latest [release](https://github.com/MiloHS/remapgnn/releases) and
place the checkpoint where the config expects it:

```bash
mkdir -p models_medium_improv
# from the release page, download:
#   fv_gen_ALL_e400_s0.pt   ->  models_medium_improv/fv_gen_ALL_e400_s0.pt
# the config it pairs with is already in the repo:
#   configs/v20b_base_a3p0_mink8_geom_v12.json
```

`fv_gen_ALL_e400_s0.pt` is the deployable, max-coverage model (trained across all six
mesh families). See [`MODEL_CARD.md`](MODEL_CARD.md).

## 3. Build the candidate graph (source + target meshes → parquet)

```bash
python scripts/build_external_kdist_graph.py \
  --src-mesh /path/to/SRC.nc  --tgt-mesh /path/to/TGT.nc \
  --src-name MYSRC            --tgt-name MYTGT \
  --out analysis_medium_improv/edge_dataset_MYSRC_to_MYTGT_kdist_a3p0_mink8.parquet \
  --alpha 3.0 --min-k 8 --max-k 256
```
`--alpha 3.0 --min-k 8` produce the `kdist_a3p0_mink8` graph the released model
expects. The builder reads `lon`/`lat` (auto-detected names; degrees or radians) and,
if present, a cell-area variable; otherwise it assumes uniform areas and warns.
Follow the `edge_dataset_<SRC>_to_<TGT>_kdist_a3p0_mink8.parquet` naming so the pair
is auto-inferred downstream.

## 4. Build the sparse remap operator

```bash
python scripts/build_remap_operator.py \
  --config configs/v20b_base_a3p0_mink8_geom_v12.json \
  --model  models_medium_improv/fv_gen_ALL_e400_s0.pt \
  --edge-parquet analysis_medium_improv/edge_dataset_MYSRC_to_MYTGT_kdist_a3p0_mink8.parquet \
  --pair MYSRC_to_MYTGT \
  --out-map maps_medium_improv/map_MYSRC_to_MYTGT.npz \
  --n-cg 800 --projection-dtype float64
```
> **Important:** pass `--model models_medium_improv/fv_gen_ALL_e400_s0.pt` explicitly.
> The script's built-in default points at an older development checkpoint; the released
> model is the one above.

This writes an `.npz` with keys `S, src_index, tgt_index, area_src, area_tgt,
metadata_json`. `S` is the per-edge weight, with `(tgt_index, src_index)` the COO
row/column. The operator is **conservative and consistent to solver tolerance**
(residuals ≈ 1e-9; see `metadata_json`). Use `--out-map …​.nc` instead to write a
SCRIP/TempestRemap-style map (1-based `row`/`col`).

## 5. Apply the operator to a field

**Built-in (one-shot build + apply):**
```bash
python scripts/build_remap_operator.py \
  --edge-parquet analysis_medium_improv/edge_dataset_MYSRC_to_MYTGT_kdist_a3p0_mink8.parquet \
  --pair MYSRC_to_MYTGT \
  --out-map maps_medium_improv/map_MYSRC_to_MYTGT.npz \
  --src-field-nc /path/to/SRC.nc --field TotalPrecipWater \
  --out-field maps_medium_improv/remapped_TPW.nc \
  --target-mesh-nc /path/to/TGT.nc
```

**Or apply a saved `.npz` yourself (no repo import needed):**
```python
import numpy as np, xarray as xr

d   = np.load("maps_medium_improv/map_MYSRC_to_MYTGT.npz")
S, col, row = d["S"], d["src_index"], d["tgt_index"]
n_tgt = int(d["area_tgt"].shape[0])

x = xr.open_dataset("SRC.nc")["TotalPrecipWater"].values.reshape(-1).astype(np.float64)  # length n_src
y = np.zeros(n_tgt)
np.add.at(y, row, S * x[col])          # y = remapped field on the target mesh
# equivalently: scipy.sparse.coo_matrix((S,(row,col)),shape=(n_tgt,x.size)).tocsr() @ x
```
(For a `.nc` map, subtract 1 from `row`/`col` first — they are stored 1-based.)

---

Reusing the same operator on many fields is a single sparse mat-vec each; you only
build the operator once per mesh pair.
