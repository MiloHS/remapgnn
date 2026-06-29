# remapgnn

Learned conservative remapping operators for spherical climate meshes.

The project goal is not to beat mature offline remapping packages in every
setting.  The goal is a reusable learned remap operator that is conservative,
reasonably accurate, mesh-flexible, and faster than constructing a new
high-order overlap/supermesh map when a cached map is not already available.

## Current result

The current default is:

- model: `v12_geom_base`
- weights: `models_medium_improv/highorder_signed_v12_geom_mom1e4.pt`
- config: `configs/v20b_base_a3p0_mink8_geom_v12.json`
- inference projection: float64 solve, `eps_rel=1e-12`, `n_cg=800`

Short version:

> `v12_geom_base` with the cleaned projection is conservative to about `2e-9`,
> improves on earlier learned baselines on real fields, and is faster than
> generating new TempestRemap `np2` maps/supermeshes at the tested r32/r64
> resolutions.  It does not beat TempestRemap `np2` on accuracy, and it does
> not beat cached Tempest maps on load/apply time.

See [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) for the numbers and
recommended wording.

## Use on a new mesh pair

The current user-facing path is:

1. build a k-distance candidate graph from source/target mesh files;
2. run the v12 GNN and cleaned conservative projection;
3. write a sparse remap operator;
4. optionally apply it to a source field.

Start with [`docs/INFERENCE.md`](docs/INFERENCE.md).

The main command for the model/projection step is:

```bash
python scripts/build_remap_operator.py \
  --config configs/v20b_base_a3p0_mink8_geom_v12.json \
  --model models_medium_improv/highorder_signed_v12_geom_mom1e4.pt \
  --edge-parquet work/graphs/edge_dataset_SRC_to_TGT_kdist_a3p0_mink8.parquet \
  --pair SRC_to_TGT \
  --out-map outputs/SRC_to_TGT_remapgnn_v12.nc
```

The current model weights should be distributed as a GitHub Release asset; see
[`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md).

## Important artifacts

Primary docs:

- [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md)
- [`docs/INFERENCE.md`](docs/INFERENCE.md)
- [`docs/MODEL_LINEAGE.md`](docs/MODEL_LINEAGE.md)
- [`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md)

Local audit/benchmark outputs used to produce the summary docs:

- `analysis_medium_improv/audits/v12_expanded_realfields_nonico_f64_proj800_eps12/`
- `analysis_medium_improv/audits/projection_sweep_v12_nonico_f64_eps12/`
- `analysis_medium_improv/benchmarks/v12_clean_projection/`
- `analysis_medium_improv/benchmarks/v12_clean_projection_r64/`
- `analysis_medium_improv/benchmarks/tempest_generation_nonico/`
- `analysis_medium_improv/benchmarks/tempest_generation_r64/`

These generated outputs are ignored by default; the important numbers are
copied into [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md).

Primary scripts:

- `scripts/build_external_kdist_graph.py`
- `scripts/build_remap_operator.py`
- `scripts/summarize_remap_output.py`
- `scripts/visualize_remap_output.py`
- `scripts/train_config_highorder.py`
- `scripts/train_config_highorder_corrector.py`
- `scripts/audit_remap_operator.py`
- `scripts/sweep_projection_conservation.py`
- `scripts/benchmark_remap_operator.py`
- `scripts/benchmark_tempest_generation.py`

Cluster-specific jobs, logs, model weights, maps, and generated analysis outputs
are ignored by default.  Model weights should be distributed through GitHub
Releases, not committed to git.

## Repository structure

- `remapgnn/` — package code
- `scripts/` — public inference, training, audit, and benchmarking scripts
- `configs/` — current public configs
- `docs/` — current results, inference instructions, model lineage, and release notes

## Current interpretation

Keep:

- geometric features and moment-aware training from `v12_geom_base`
- the cleaned float64 projection for deployable conservation
- the audit suite over real fields, analytic fields, spectral shells, and
  Cartesian moments
- the benchmark split between learned operator construction, cached-map
  loading, and Tempest map generation

Do not claim:

- that the learned operator is more accurate than TempestRemap `np2`
- that it is faster than using an already-cached Tempest map

Reasonable paper/tool framing:

> A conservative, supermesh-free learned remapping prototype that approaches
> `np2` accuracy while reducing new-operator construction cost at the tested
> resolutions.
