# Repository cleanup guide

Date: 2026-06-29

This repo contains many exploratory training runs.  The goal of cleanup is not
to erase history; it is to make the current result reproducible and obvious.

## Current result to keep visible

Primary docs:

- [`docs/CURRENT_RESULTS.md`](CURRENT_RESULTS.md)
- [`docs/PROJECT_STATUS_AND_NEXT_STEPS.md`](PROJECT_STATUS_AND_NEXT_STEPS.md)

Primary model/config:

- `models_medium_improv/highorder_signed_v12_geom_mom1e4.pt`
- `configs/v20b_base_a3p0_mink8_geom_v12.json`
- `configs/v20b_base_a3p0_mink8.json`

Primary evaluation scripts:

- `scripts/audit_remap_operator.py`
- `scripts/sweep_projection_conservation.py`
- `scripts/benchmark_remap_operator.py`
- `scripts/benchmark_tempest_generation.py`
- `scripts/check_real_field_coverage.py`
- `scripts/train_config_highorder.py`
- `scripts/train_config_highorder_corrector.py`

Primary reproducibility jobs:

- `jobs_audit_v12_expanded_realfields_f64_proj800_eps12.pbs`
- `jobs_projection_sweep_v12_nonico_f64_eps12.pbs`
- `jobs_benchmark_v12_clean_projection.pbs`
- `jobs_benchmark_tempest_generation_nonico.pbs`
- `jobs_benchmark_v12_r64_scaling.pbs`

Primary result directories:

- `analysis_medium_improv/audits/v12_expanded_realfields_nonico_f64_proj800_eps12/`
- `analysis_medium_improv/audits/projection_sweep_v12_nonico_f64_eps12/`
- `analysis_medium_improv/benchmarks/v12_clean_projection/`
- `analysis_medium_improv/benchmarks/tempest_generation_nonico/`
- `analysis_medium_improv/benchmarks/v12_clean_projection_r64/`
- `analysis_medium_improv/benchmarks/tempest_generation_r64/`

Useful comparison/negative-result directories:

- `analysis_medium_improv/audits/v10b_vs_v10d_6band_a4k16_guardrail/`
- `analysis_medium_improv/audits/v12_geom_guarded_corrector/`
- `analysis_medium_improv/audits/v12_geom_v10b_guardrail/`

## Current conclusions

- Default model: `v12_geom_base + cleaned projection`.
- Optional spectral variant: `v12_geom_v10b`.
- Rejected architecture direction: `v10d` / wide-stencil 6-band corrector.
- Do not present the model as beating cached Tempest maps.
- Do present it as faster than generating a new `np2` Tempest map/supermesh at
  tested r32/r64 cases.

## Archive policy

Top-level PBS files and logs from older exploratory runs are archived under:

- `archive/legacy_jobs_2026-06-29/`
- `archive/legacy_logs_2026-06-29/`

Exploratory one-off configs are archived under:

- `archive/experimental_configs_2026-06-29/`

Archiving means "remove from the root clutter but keep recoverable."  Nothing
is intentionally deleted.

Files left at repo root should either be source/configuration files or current
reproducibility jobs/logs.

Configs left in `configs/` should be either:

- current/reproducible configs used by the active audit/benchmark jobs, or
- tracked historical lineage configs referenced by older docs/scripts.

For details, see `configs/README.md`.

## Do not delete without checking

Do not delete these just because they look large:

- `models_medium_improv/`
- `maps_medium_improv/`
- `analysis_medium_improv/edge_dataset_*`
- `data/MIRA-Datasets/`

These are required for training, audit, and benchmarking.
