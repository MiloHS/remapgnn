# Inference with trained remapgnn weights

This page describes the current research inference workflow for the trained `v18_irno_corrector_from_v16_l24_a2p0_mink8` model.

## Download weights

The trained weights are distributed separately as a GitHub Release asset, not committed directly to the repository.

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

The current research code does not yet provide a polished one-command interface for arbitrary external NetCDF meshes and fields. The model consumes a candidate source-target graph with the same feature schema used during training. A production-style arbitrary-topology inference script is planned.

For now, the tested workflow is through the experiment/evaluation scripts used in this repository, using mesh pairs with prepared candidate edge datasets.

## Known issues

- Very large target meshes can be slow because Sinkhorn balancing is run during inference.
- RLL meshes may show pole-related ambiguity when used as source meshes.
- Performance on completely unseen topologies should be treated as experimental.
- The current diagnostics compare learned inference against already-built Tempest maps; this is not the same as comparing against full Tempest map-generation time.
