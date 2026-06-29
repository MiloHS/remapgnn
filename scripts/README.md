# Scripts

This directory contains the public scripts kept for the current v12 prototype.

## User-facing inference

- `build_external_kdist_graph.py` — build a source-target candidate graph from
  mesh files.
- `build_remap_operator.py` — build the learned conservative sparse remap
  operator from a prepared graph.
- `summarize_remap_output.py` — summarize a remapped field.
- `visualize_remap_output.py` — quick plotting helper for remapped fields.
- `normalize_mesh_unit.py` — helper for meshes whose Cartesian coordinates are
  not already on the unit sphere.

## Current training / audit / benchmark path

- `train_config_highorder.py`
- `train_config_highorder_corrector.py`
- `audit_remap_operator.py`
- `sweep_projection_conservation.py`
- `benchmark_remap_operator.py`
- `benchmark_tempest_generation.py`
- `check_real_field_coverage.py`

## Retained helpers

- `train_config_balanced_harmonic.py`
- `train_config_irno_corrector.py`
- `evaluate_refinement_convergence.py`

These helpers still provide shared utilities imported by the current audit and
training scripts.  A future cleanup could move those utilities into the
`remapgnn/` package and remove the legacy filenames.
