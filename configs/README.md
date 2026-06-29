# Configs

This directory is for configs that are either current/reproducible or part of
the tracked model lineage.

## Current default configs

- `v20b_base_a3p0_mink8_geom_v12.json` — current default model/config for
  `v12_geom_base`.
- `v20b_base_a3p0_mink8.json` — current non-geometry baseline config used by
  audits and benchmarks for older learned operators such as `v10b`.

These two are referenced by the current PBS jobs at the repo root.

## Historical lineage configs

The tracked v10–v20 configs are kept here because they document the project
lineage and are still referenced by older docs/scripts:

- v10/v11: early hybrid-attention baselines
- v15/v16/v17: harmonic-loss base models
- v18/v19: IRNO/corrector lineage
- v20a/v20b: topology-holdout/diverse-topology experiments

## Archived exploratory configs

One-off sweeps and rejected/smoke configs were moved to:

- `archive/experimental_configs_2026-06-29/`

Move a config back here only when it becomes part of the current reproducible
story or a paper/table artifact.
