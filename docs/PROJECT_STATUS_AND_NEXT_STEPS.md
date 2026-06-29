# RemapGNN project status and next steps

Date: 2026-06-29

Goal: build a learned, supermesh-free conservative remap operator for climate
fields on different meshes. The model does not need exact formal second-order
behavior everywhere, but it should be conservative, consistent, accurate enough
to be useful, and efficient/deployable as a tool.

## Current thesis

The strongest honest project story is:

> A learned GNN remap operator can use only mesh/candidate-graph geometry, then
> enforce conservation and consistency by projection, to approach second-order
> remapping accuracy without requiring an overlap supermesh at inference.

This should be presented as "similar accuracy with a different efficiency and
deployment profile", not as "strictly beats TempestRemap".

## Current default model

Use this as the main tool/paper candidate:

- Model: `v12_geom_base`
- Pack: `models_medium_improv/highorder_signed_v12_geom_mom1e4.pt`
- Config: `configs/v20b_base_a3p0_mink8_geom_v12.json`
- Projection at inference/audit:
  - `projection_dtype=float64`
  - `projection_eps_rel=1e-12`
  - `n_cg=800`

Why this model:

- best learned model on expanded real-field audit;
- simpler than the corrector;
- better moment behavior than the v12 corrector;
- cleaned projection gives conservation/consistency residuals near `2e-9`.

Optional spectral/analytic variant:

- `v12_geom_v10b`
- Pack: `models_medium_improv/highorder_corrector_v12_geom_v10b.pt`
- It has the best learned analytic/spectral means, but worse real-field and
  moment behavior than `v12_geom_base`.

## Key current results

Primary cleaned audit:

- [`analysis_medium_improv/audits/v12_expanded_realfields_nonico_f64_proj800_eps12/summary.md`](../analysis_medium_improv/audits/v12_expanded_realfields_nonico_f64_proj800_eps12/summary.md)
- 5 non-ICO mesh pairs.
- 25/25 available real-field cases.
- 70 analytic-function cases.
- Spectral shells through degree 48.
- Cleaned projection: `float64`, `eps_rel=1e-12`, `n_cg=800`.

Important accuracy numbers:

| operator | real mean rel-L2 | analytic mean rel-L2 | spectral mean rel-L2 | max conservation residual |
| --- | ---: | ---: | ---: | ---: |
| `np2` | `0.002085` | `0.041290` | `0.055341` | `1.19e-15` |
| `v12_geom_base` | `0.003103` | `0.051071` | `0.070676` | `2.03e-9` |
| `v12_geom_guarded` | `0.003132` | `0.050113` | `0.069610` | `2.03e-9` |
| `v12_geom_v10b` | `0.003215` | `0.047105` | `0.064814` | `2.03e-9` |
| `v10b` | `0.003652` | `0.051952` | `0.071658` | `2.03e-9` |
| `np1` | `0.007816` | `0.096311` | `0.134040` | `1.22e-16` |

Important moment numbers:

| operator | Cartesian order-1 mean rel-L2 | Cartesian order-2 mean rel-L2 |
| --- | ---: | ---: |
| `np2` | `2.10e-4` | `3.42e-4` |
| `v12_geom_base` | `7.28e-4` | `1.09e-3` |
| `v12_geom_guarded` | `7.53e-4` | `1.12e-3` |
| `v12_geom_v10b` | `8.65e-4` | `1.27e-3` |
| `v10b` | `1.10e-3` | `1.61e-3` |

## Efficiency results

Cached-map deployment benchmark:

- [`analysis_medium_improv/benchmarks/v12_clean_projection/summary.md`](../analysis_medium_improv/benchmarks/v12_clean_projection/summary.md)
- [`analysis_medium_improv/benchmarks/v12_clean_projection_r64/summary.md`](../analysis_medium_improv/benchmarks/v12_clean_projection_r64/summary.md)

Tempest generation benchmark:

- [`analysis_medium_improv/benchmarks/tempest_generation_nonico/summary.md`](../analysis_medium_improv/benchmarks/tempest_generation_nonico/summary.md)
- [`analysis_medium_improv/benchmarks/tempest_generation_r64/summary.md`](../analysis_medium_improv/benchmarks/tempest_generation_r64/summary.md)

Summary:

| resolution/pairs | learned build only | learned input+build | Tempest `np2` generation | build-only speedup vs `np2` gen | input+build speedup vs `np2` gen |
| --- | ---: | ---: | ---: | ---: | ---: |
| r32 non-ICO set | `0.244s` | `0.810s` | `1.738s` | `8.1x` | `2.3x` |
| r64 CS↔ICOD | `0.395s` | `2.179s` | `5.301s` | `13.4x` | `2.4x` |

Interpretation:

- Against already-cached Tempest maps, learned is slower.
- Against Tempest `np2` map generation / supermesh construction, learned is faster.
- The main learned-side bottleneck is now input/candidate-feature loading, not
  the GNN/projection build.
- Sparse apply is slower for learned operators because the learned candidate
  stencil currently has about 2.6x as many edges as `np2`.

## Model lineage and keeper status

| model | status | lesson |
| --- | --- | --- |
| `v6a` signed base | useful historical baseline | Signed projected weights were necessary; nonnegative operators were too restrictive. |
| `v10b` safe corrector | reliable older baseline | Corrector helped versus older learned models, but is now not the default. |
| `v10d` 6-band/a4/k16 | reject | More bands/wider stencil degraded clean audit metrics. |
| `v12_geom_base` | default keeper | Best learned real-field result; simpler and structurally cleaner. |
| `v12_geom_v10b` | optional spectral variant | Best learned analytic/spectral mean, but not best real/moment model. |
| `v12_geom_guarded` | evidence, not default | Guardrails helped structure but did not beat base on real fields. |

## Immediate next steps

1. Keep `v12_geom_base + cleaned projection` as the default tool candidate.
2. Optimize input/feature loading; this dominates learned input+build time.
3. Reduce learned stencil size if possible, because apply cost is edge-count
   dominated.
4. If doing another model experiment, prefer a focused quadratic/moment-loss
   training run on the base model, not another broad corrector-band sweep.
5. For paper/tool presentation, use the cleaned audit and generation benchmark
   tables above as the main evidence.

