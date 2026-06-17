# Inference with trained remapgnn weights

This page describes how to run research inference with the trained `v18_irno_corrector_from_v16_l24_a2p0_mink8` model.

The workflow is:

    clone repo
    download weights
    prepare source mesh, target mesh, and source field
    build candidate source-target graph
    run learned remapping inference
    optionally visualize the output
    optionally compute summary metrics

## 1. Clone the repository

    git clone https://github.com/MiloHS/remapgnn.git
    cd remapgnn

## 2. Install Python dependencies

Using a virtual environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Or using conda:

    conda create -n remapgnn python=3.11 -y
    conda activate remapgnn
    pip install -r requirements.txt

The main dependencies are PyTorch, NumPy, pandas, SciPy, xarray, pyarrow, netCDF4, and matplotlib.

## 3. Download trained weights

The trained v18 weights are given as a GitHub Release asset.

Release page:

    https://github.com/MiloHS/remapgnn/releases/tag/v18-weights

Download:

    remapgnn_v18_weights.tar.gz

Place it in the repository root and extract it:

    tar -xzf remapgnn_v18_weights.tar.gz

## 4. Model summary

The current model is v18:

- frozen v16 gated-hybrid-attention GNN/Sinkhorn base remapper
- iterative learned corrector
- correction stages at `lmax=8`, `lmax=16`, and `lmax=24`
- Sinkhorn balancing after each correction step
- final output is a sparse conservative remapping operator

## 5. Input requirements

You need three files:

    source_mesh.nc
    target_mesh.nc
    source_field.nc

The source and target meshes should be spherical finite-volume or cell-centered meshes.

The mesh files should contain longitude, latitude, and preferably cell area. The helper scripts try common names such as:

    lon, longitude, lonCell, xlon
    lat, latitude, latCell, ylat
    cell_area, area, areaCell, cellArea

## 6. Build a candidate source-target graph

The GNN does not consume raw meshes directly, you need to generate a source-target candidate graph with geometric edge features.

For a new mesh pair, build the graph:

    mkdir -p analysis_medium_improv outputs

    python scripts/build_external_kdist_graph.py \
      --src-mesh my_data/source_mesh.nc \
      --tgt-mesh my_data/target_mesh.nc \
      --src-name MY-SOURCE \
      --tgt-name MY-TARGET \
      --out analysis_medium_improv/edge_dataset_MY-SOURCE_to_MY-TARGET_kdist_a2p0_mink8.parquet \
      --alpha 2.0 \
      --min-k 8 \
      --max-k 256 \
      --normalize-area-sums

This writes:

    analysis_medium_improv/edge_dataset_MY-SOURCE_to_MY-TARGET_kdist_a2p0_mink8.parquet

The `--normalize-area-sums` option is useful when source and target meshes have slightly different total area normalizations.

## 7. Run learned remapping inference

Apply the trained model to a source field.

For a field named `temperature`:

    python scripts/infer_prepared_pair.py \
      --config configs/v18_irno_corrector_from_v16_l24_a2p0_mink8.json \
      --pair MY-SOURCE_to_MY-TARGET \
      --edge-parquet analysis_medium_improv/edge_dataset_MY-SOURCE_to_MY-TARGET_kdist_a2p0_mink8.parquet \
      --src-field-nc my_data/source_field.nc \
      --target-mesh-nc my_data/target_mesh.nc \
      --field temperature \
      --stage lmax24 \
      --balance-iters 2000 \
      --out outputs/temperature_remapped_to_target.nc \
      --out-map outputs/MY-SOURCE_to_MY-TARGET_learned_operator.npz

This writes:

    outputs/temperature_remapped_to_target.nc
    outputs/MY-SOURCE_to_MY-TARGET_learned_operator.npz
    
For a faster test run, use fewer Sinkhorn iterations:

    --balance-iters 300

## 8. Visualize the remapped field

If you only have the prediction:

    python scripts/visualize_remap_output.py \
      --pred-nc outputs/temperature_remapped_to_target.nc \
      --field temperature \
      --target-mesh-nc my_data/target_mesh.nc \
      --out outputs/temperature_remapped_to_target.png

If you also have a target truth/reference field:

    python scripts/visualize_remap_output.py \
      --pred-nc outputs/temperature_remapped_to_target.nc \
      --field temperature \
      --target-mesh-nc my_data/target_mesh.nc \
      --truth-nc my_data/target_truth.nc \
      --truth-field temperature \
      --out outputs/temperature_prediction_truth_error.png

## 9. Compute summary metrics

Without truth, compute basic field statistics:

    python scripts/summarize_remap_output.py \
      --pred-nc outputs/temperature_remapped_to_target.nc \
      --field temperature \
      --target-mesh-nc my_data/target_mesh.nc \
      --out-csv outputs/temperature_summary.csv

If the source field and source mesh are provided, the script also computes global conservation:

    python scripts/summarize_remap_output.py \
      --pred-nc outputs/temperature_remapped_to_target.nc \
      --field temperature \
      --target-mesh-nc my_data/target_mesh.nc \
      --source-nc my_data/source_field.nc \
      --source-field temperature \
      --source-mesh-nc my_data/source_mesh.nc \
      --out-csv outputs/temperature_summary.csv

If target truth is available, compute relative L2 and area-weighted relative L2:

    python scripts/summarize_remap_output.py \
      --pred-nc outputs/temperature_remapped_to_target.nc \
      --field temperature \
      --target-mesh-nc my_data/target_mesh.nc \
      --truth-nc my_data/target_truth.nc \
      --truth-field temperature \
      --source-nc my_data/source_field.nc \
      --source-field temperature \
      --source-mesh-nc my_data/source_mesh.nc \
      --out-csv outputs/temperature_summary.csv
- Users should validate against analytic truth, TempestRemap, or another trusted reference before using outputs scientifically.
- Very large target meshes can be slow because Sinkhorn balancing is run during inference.
- RLL source meshes may show pole-related ambiguity.
- The current timing diagnostics compare learned inference against already-built Tempest maps, not full Tempest map-generation time.

