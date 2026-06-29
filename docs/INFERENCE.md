# Inference with RemapGNN

This page describes the current research-prototype inference path for using
RemapGNN on a new source/target mesh pair.

Current default:

- model: `v12_geom_base`
- expected weight path:
  `models_medium_improv/highorder_signed_v12_geom_mom1e4.pt`
- config: `configs/v20b_base_a3p0_mink8_geom_v12.json`
- projection: float64, `eps_rel=1e-12`, `n_cg=800`

The old GitHub release contains a v18 corrector model.  That release is useful
history, but the current documented path below expects a new v12 weight release.

## Install

```bash
git clone https://github.com/MiloHS/remapgnn.git
cd remapgnn

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On a CUDA machine, install the matching PyTorch build first if needed; see
`requirements.txt`.

## Download weights

The current v12 model should be distributed as a GitHub Release asset and
extracted so this file exists:

```text
models_medium_improv/highorder_signed_v12_geom_mom1e4.pt
```

Suggested future release tag/name:

```text
v12-geom-base
remapgnn_v12_geom_base_2026-06-29.tar.gz
```

Until that release exists, copy the checkpoint from the Swing workspace into
the path above.

## What “arbitrary mesh” means here

The GNN does not consume raw meshes directly.  The pipeline is:

```text
source mesh + target mesh
        ↓
k-distance candidate graph with geometric features
        ↓
GNN predicts signed edge masses
        ↓
float64 conservative/consistent projection
        ↓
sparse remap operator
        ↓
optional application to source fields
```

So arbitrary-mesh usage currently means: any mesh pair that can be converted
into the expected cell-centered edge parquet with areas, centers, and candidate
source-target edges.

The helper script accepts common mesh variable names:

- longitude: `lon`, `longitude`, `lonCell`, `xlon`
- latitude: `lat`, `latitude`, `latCell`, `ylat`
- area: `cell_area`, `area`, `areaCell`, `cellArea`, `area_cell`
- Cartesian centers, if present: `x/y/z`, `xCell/yCell/zCell`, `cell_x/cell_y/cell_z`

If areas are absent, the graph builder falls back to uniform unit-sphere areas;
that is acceptable only for quick tests.

## 1. Build a candidate graph

```bash
mkdir -p work/graphs outputs

python scripts/build_external_kdist_graph.py \
  --src-mesh my_data/source_mesh.nc \
  --tgt-mesh my_data/target_mesh.nc \
  --src-name MY-SOURCE \
  --tgt-name MY-TARGET \
  --out work/graphs/edge_dataset_MY-SOURCE_to_MY-TARGET_kdist_a2p0_mink8.parquet \
  --alpha 2.0 \
  --min-k 8 \
  --max-k 256 \
  --normalize-area-sums
```

This is supermesh-free: it uses centers, areas, and nearest-neighbor candidate
edges, not polygon overlaps.

Important caveat: every target cell gets at least `min-k` source candidates,
but some source cells can still have zero candidate edges on unusual mesh pairs.
The operator builder reports zero-degree source/target counts because zero-edge
source cells cannot be conserved by any sparse operator on that graph.

## 2. Build the learned remap operator

```bash
python scripts/build_remap_operator.py \
  --config configs/v20b_base_a3p0_mink8_geom_v12.json \
  --model models_medium_improv/highorder_signed_v12_geom_mom1e4.pt \
  --edge-parquet work/graphs/edge_dataset_MY-SOURCE_to_MY-TARGET_kdist_a2p0_mink8.parquet \
  --pair MY-SOURCE_to_MY-TARGET \
  --out-map outputs/MY-SOURCE_to_MY-TARGET_remapgnn_v12.nc \
  --summary-json outputs/MY-SOURCE_to_MY-TARGET_remapgnn_v12_summary.json \
  --projection-dtype float64 \
  --projection-eps-rel 1e-12 \
  --n-cg 800
```

The NetCDF map stores:

- `S`: sparse remap weights
- `row`: 1-based target indices
- `col`: 1-based source indices
- `area_a`: source cell areas
- `area_b`: target cell areas

You can also write a compressed NumPy map:

```bash
--out-map outputs/MY-SOURCE_to_MY-TARGET_remapgnn_v12.npz
```

At the end, the script prints a small audit:

```text
conservation_residual
consistency_residual
zero_degree_source
zero_degree_target
operator build time
```

For the current model, the intended deployable setting is roughly
`conservation_residual ≈ 1e-9` on supported graph pairs.

## 3. Optionally apply the operator to a field

If your source field file has a variable named `temperature`:

```bash
python scripts/build_remap_operator.py \
  --config configs/v20b_base_a3p0_mink8_geom_v12.json \
  --model models_medium_improv/highorder_signed_v12_geom_mom1e4.pt \
  --edge-parquet work/graphs/edge_dataset_MY-SOURCE_to_MY-TARGET_kdist_a2p0_mink8.parquet \
  --pair MY-SOURCE_to_MY-TARGET \
  --out-map outputs/MY-SOURCE_to_MY-TARGET_remapgnn_v12.nc \
  --src-field-nc my_data/source_field.nc \
  --field temperature \
  --target-mesh-nc my_data/target_mesh.nc \
  --out-field outputs/temperature_on_target_remapgnn_v12.nc
```

The field output is a simple target-cell NetCDF file with the remapped variable,
target cell area, and lon/lat metadata when available.

## 4. Summarize or visualize

Basic summary:

```bash
python scripts/summarize_remap_output.py \
  --pred-nc outputs/temperature_on_target_remapgnn_v12.nc \
  --field temperature \
  --target-mesh-nc my_data/target_mesh.nc \
  --source-nc my_data/source_field.nc \
  --source-field temperature \
  --source-mesh-nc my_data/source_mesh.nc \
  --out-csv outputs/temperature_summary.csv
```

Visualization:

```bash
python scripts/visualize_remap_output.py \
  --pred-nc outputs/temperature_on_target_remapgnn_v12.nc \
  --field temperature \
  --target-mesh-nc my_data/target_mesh.nc \
  --out outputs/temperature_on_target_remapgnn_v12.png
```

## Legacy v18 release

The old release asset for `v18_irno_corrector_from_v16_l24_a2p0_mink8` used the
legacy script:

```bash
python scripts/infer_prepared_pair.py ...
```

That path is kept for reproducibility of the old release, but it is not the
current recommended model/tool path.
