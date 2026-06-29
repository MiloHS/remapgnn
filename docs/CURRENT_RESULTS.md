# Current RemapGNN results

Date: 2026-06-29

## One-sentence result

`v12_geom_base` with the cleaned projection is the current default: it is not as
accurate as TempestRemap `np2`, but it is conservative to about `2e-9`, clearly
better than earlier learned baselines on real fields, and faster than generating
new `np2` Tempest maps/supermeshes at r32/r64.

## Default model

- Model: `v12_geom_base`
- Pack: `models_medium_improv/highorder_signed_v12_geom_mom1e4.pt`
- Config: `configs/v20b_base_a3p0_mink8_geom_v12.json`
- Inference projection:
  - `projection_dtype=float64`
  - `projection_eps_rel=1e-12`
  - `n_cg=800`

## Accuracy/conservation headline

Source audit: `analysis_medium_improv/audits/v12_expanded_realfields_nonico_f64_proj800_eps12/`.
Generated audit outputs are not tracked in the cleaned GitHub repo; this table
copies the key public numbers.

| operator | real mean rel-L2 | analytic mean rel-L2 | spectral mean rel-L2 | max conservation residual |
| --- | ---: | ---: | ---: | ---: |
| `np2` | `0.002085` | `0.041290` | `0.055341` | `1.19e-15` |
| `v12_geom_base` | `0.003103` | `0.051071` | `0.070676` | `2.03e-9` |
| `v12_geom_v10b` | `0.003215` | `0.047105` | `0.064814` | `2.03e-9` |
| `v10b` | `0.003652` | `0.051952` | `0.071658` | `2.03e-9` |
| `np1` | `0.007816` | `0.096311` | `0.134040` | `1.22e-16` |

Interpretation:

- `np2` is still better on accuracy.
- `v12_geom_base` is the best learned model on real fields.
- `v12_geom_v10b` is the best learned model on analytic/spectral metrics, but
  not the best default because it gives back real-field/moment quality.
- Cleaned projection fixed the conservation residual floor for learned models.

## Moment/structure headline

| operator | Cartesian order-1 mean rel-L2 | Cartesian order-2 mean rel-L2 |
| --- | ---: | ---: |
| `np2` | `2.10e-4` | `3.42e-4` |
| `v12_geom_base` | `7.28e-4` | `1.09e-3` |
| `v12_geom_v10b` | `8.65e-4` | `1.27e-3` |
| `v10b` | `1.10e-3` | `1.61e-3` |

Interpretation: `v12_geom_base` is the structurally cleanest learned candidate.

## Efficiency headline

Source benchmarks, local/generated:

- `analysis_medium_improv/benchmarks/v12_clean_projection/`
- `analysis_medium_improv/benchmarks/tempest_generation_nonico/`
- `analysis_medium_improv/benchmarks/v12_clean_projection_r64/`
- `analysis_medium_improv/benchmarks/tempest_generation_r64/`

| resolution/pairs | learned build only | learned input+build | Tempest `np2` generation | build-only speedup | input+build speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| r32 non-ICO set | `0.244s` | `0.810s` | `1.738s` | `8.1x` | `2.3x` |
| r64 CS↔ICOD | `0.395s` | `2.179s` | `5.301s` | `13.4x` | `2.4x` |

Important caveat:

- Cached Tempest maps are still much faster to load/apply.
- The useful efficiency comparison is against generating a new conservative
  high-order map, especially the `np2` supermesh/offline-map path.

## Recommended wording

Use:

> The learned operator is a conservative, supermesh-free remapping prototype
> that approaches `np2` accuracy while generating operators faster than the
> TempestRemap `np2` overlap/offline-map path at the tested resolutions.

Avoid:

> The learned operator beats TempestRemap.

That is not supported by the current accuracy or cached-map timing results.
