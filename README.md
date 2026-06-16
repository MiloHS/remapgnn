# remapgnn

Learned conservative remapping operators for spherical climate meshes.

This project develops a sparse GNN/Sinkhorn surrogate for TempestRemap-style conservative remapping. The model operates on a fixed candidate source-target bipartite graph. A GNN predicts positive edge scores, and a sparse Sinkhorn balancing step converts those scores into a conservative remapping operator.

## Core idea

For a source mesh and target mesh, we build candidate source-target edges. The model predicts edge scores on this graph. Sinkhorn balancing turns those scores into a sparse mass matrix that satisfies conservative remapping constraints. The remapping weights are then applied to fields on the source mesh.

The goal is not to replace physical simulation. The goal is to learn a fast, sparse, conservative approximation to TempestRemap operators across mesh pairs.

## Current best result

The current best model is `v18_irno_corrector_from_v16_l24_a2p0_mink8`.

It uses:

- a frozen v16 gated-hybrid-attention GNN/Sinkhorn base remapper
- a shared gated-hybrid-attention corrector
- iterative correction steps conditioned on spherical harmonic bands
- Sinkhorn balancing after each correction step

The correction trajectory is:

1. base v16 operator
2. correction with `lmax=8`
3. correction with `lmax=16`
4. correction with `lmax=24`

Across six mesh pairs, v18 reduced average field relative L2 error versus Tempest from about `0.002956` to `0.002715`, and reduced average spherical-harmonic spectral error from about `1.58e-2` to `1.43e-2`.

## Repository structure

- `remapgnn/` — reusable package code
- `scripts/` — training and evaluation scripts
- `configs/` — experiment configurations
- `docs/` — experiment notes, model lineage, and result summaries
- `analysis_medium_improv/github_results/` — curated result CSVs and figures

Large raw datasets, generated maps, edge parquet files, and model checkpoints are intentionally excluded from normal Git tracking.

## Important docs

- `docs/MODEL_LINEAGE.md`
- `docs/RESULTS_SUMMARY.md`

## Main training scripts

- `scripts/train_config.py`
- `scripts/train_config_balanced.py`
- `scripts/train_config_balanced_harmonic.py`
- `scripts/train_config_irno_corrector.py`

## Main evaluation scripts

- `scripts/evaluate_config.py`
- `scripts/evaluate_spectral_harmonics.py`
- `scripts/evaluate_irno_corrector.py`
- `scripts/evaluate_irno_spectral_trajectory.py`

## Model lineage summary

- v10: hybrid attention GNN
- v11: gated hybrid attention GNN
- v16: gated hybrid attention with spherical harmonic loss up to degree 24
- v18: frozen v16 plus iterative conditional corrector
- v19: gentler v18 ablation

## Status

The current best result is v18. v19 is kept as an ablation showing that gentler correction remains stable but does not outperform the main v18 seed123 run.
