# Inference with trained remapgnn weights

We describe tthe current worfflow for the trained `v18_irno_corrector_from_v16_l24_a2p0_mink8` model.

## Download weights

The trained weights are given as a GitHub Release.

Release page: https://github.com/MiloHS/remapgnn/releases/tag/v18-weights

Download the release archive, place it in the repository root, and extract it with:

    tar -xzf remapgnn_v18_weights.tar.gz

The archive contains:

    configs/v18_irno_corrector_from_v16_l24_a2p0_mink8.json
    models_medium_improv/bipartite_gnn_sinkhorn_v16_gated_hybridattn_balanced_long_harmonic_l24_kdist_a2p0_mink8.pt
    models_medium_improv/bipartite_gnn_sinkhorn_v18_irno_corrector_from_v16_l24_kdist_a2p0_mink8.pt
    MANIFEST.md
    SHA256SUMS.txt

## Model summary

The current best model is v18:

- frozen v16 gated-hybrid-attention GNN/Sinkhorn base remapper
- iterative learned corrector
- correction stages at `lmax=8`, `lmax=16`, and `lmax=24`
- Sinkhorn balancing after each correction step
- final output is a sparse conservative remapping operator

## Expected inference workflow

For a new source-target mesh pair:

1. Prepare a source spherical finite-volume mesh.
2. Prepare a target spherical finite-volume mesh.
3. Build candidate source-target edges using the same graph rule used in training:
   - k-distance graph
   - `alpha = 2.0`
   - `min_k = 8`
4. Compute the expected geometric edge features.
5. Load the v16 base remapper and v18 corrector weights.
6. Run the base GNN and iterative corrector trajectory.
7. Sinkhorn-balance the predicted sparse edge weights.
8. Apply the learned sparse operator to a source field.

## Current limitation

The model takes a candidate source-target graph with the same feature schema used during training, does not provide a nice interface for arbitrary NetCDF meshes and fields. 

For now, the tested workflow is through the experiment/evaluation scripts used in this repository, using mesh pairs with prepared candidate edge datasets.

## Prepared-graph inference script

The repository includes a research inference script for mesh pairs with a prepared candidate edge graph:

    scripts/infer_prepared_pair.py

Example:

    python scripts/infer_prepared_pair.py \
      --config configs/v18_irno_corrector_from_v16_l24_a2p0_mink8.json \
      --pair CS-r32_to_ICOD-r32 \
      --src-field-nc data/MIRA-Datasets/Meshes/UniformlyRefined/CS/sample_NM16_O10_CS-r32_TPW_CFR_TPO_A1_A2.nc \
      --target-mesh-nc data/MIRA-Datasets/Meshes/UniformlyRefined/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc \
      --field AnalyticalFun1 \
      --stage lmax24 \
      --balance-iters 2000 \
      --out analysis_medium_improv/inference_demo_CS-r32_to_ICOD-r32_AnalyticalFun1.nc \
      --out-map analysis_medium_improv/inference_demo_CS-r32_to_ICOD-r32_lmax24_operator.npz

This script assumes that the edge dataset for the requested pair already exists in the configured analysis directory, for example:

    analysis_medium_improv/edge_dataset_CS-r32_to_ICOD-r32_kdist_a2p0_mink8.parquet

The script writes a target-field NetCDF file and can optionally write the learned sparse operator as a compressed NumPy archive.
