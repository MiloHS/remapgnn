# Analytic refinement convergence study

This experiment checks whether the learned conservative remapper behaves like a convergent numerical method under mesh refinement.

## Setup

We use a four-level CS to ICOD refinement:

| Level | Pair |
|---:|---|
| 1 | `CS-r16_to_ICOD-r16` |
| 2 | `CS-r32_to_ICOD-r32` |
| 3 | `CS-r64_to_ICOD-r64` |
| 4 | `CS-r128_to_ICOD-r128` |

For each level, analytic fields are evaluated on the source mesh, remapped to the target mesh, and compared against analytic target values using area-weighted relative L2 error.

We use the following mesh size proxy:

`h = max(sqrt(mean source cell area), sqrt(mean target cell area))`

We estimate the convergence order by fitting a line to `log(error)` versus `log(h)` across all four levels.

## Analytic fields

The smooth analytic test suite is:

- `x`
- `y`
- `z`
- `smooth1`
- `smooth2`

## Fitted convergence orders

| Method / stage | Mean fitted order | Min | Max | Mean R² |
|---|---:|---:|---:|---:|
| Tempest | 1.014 | 1.012 | 1.016 | 0.99998 |
| v16 base | 1.481 | 1.324 | 1.652 | 0.9535 |
| v18 corrected `lmax=8` | 1.504 | 1.412 | 1.637 | 0.9530 |
| v18 corrected `lmax=16` | 1.472 | 1.375 | 1.591 | 0.9530 |
| v18 corrected `lmax=24` | 1.412 | 1.320 | 1.538 | 0.9574 |

## Figures

![CS to ICOD four-level analytic convergence](../analysis_medium_improv/github_results/convergence_CS_to_ICOD_4level_loglog.png)

![Fitted convergence orders](../analysis_medium_improv/github_results/convergence_CS_to_ICOD_4level_fitted_orders.png)

## Curated result files

Curated result files are in `analysis_medium_improv/github_results/`.

Key files:

- `convergence_CS_to_ICOD_4level_tempest_smooth.csv`
- `convergence_CS_to_ICOD_4level_tempest_smooth_slopes.csv`
- `convergence_CS_to_ICOD_4level_v18_trajectory_smooth.csv`
- `convergence_CS_to_ICOD_4level_v18_trajectory_smooth_slopes.csv`
- `convergence_CS_to_ICOD_4level_stage_summary.csv`
