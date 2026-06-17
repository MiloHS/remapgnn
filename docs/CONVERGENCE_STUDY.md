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

## Reverse direction: ICOD to CS

We also ran the same four-level diagnostic in the reverse direction:

| Level | Pair |
|---:|---|
| 1 | `ICOD-r16_to_CS-r16` |
| 2 | `ICOD-r32_to_CS-r32` |
| 3 | `ICOD-r64_to_CS-r64` |
| 4 | `ICOD-r128_to_CS-r128` |

The model struggled a little more.

| Method / stage | Mean fitted order | Min | Max | Mean R² |
|---|---:|---:|---:|---:|
| Tempest | 1.083 | 1.032 | 1.179 | 0.9989 |
| v16 base | 0.806 | 0.662 | 0.925 | 0.9180 |
| v18 corrected `lmax=8` | 0.866 | 0.748 | 0.975 | 0.9445 |
| v18 corrected `lmax=16` | 0.882 | 0.776 | 1.010 | 0.9634 |
| v18 corrected `lmax=24` | 0.844 | 0.709 | 1.009 | 0.9686 |

The learned conservative remapper shows first-order-or-better aggregate convergence for CS→ICOD, but sub-first-order aggregate convergence for ICOD→CS.

![ICOD to CS four-level analytic convergence](../analysis_medium_improv/github_results/convergence_ICOD_to_CS_4level_loglog.png)

![ICOD to CS fitted convergence orders](../analysis_medium_improv/github_results/convergence_ICOD_to_CS_4level_fitted_orders.png)

## Analytic test field profiles

The convergence test fields `smooth1` and `smooth2` are synthetic analytic functions, not MIRA physical fields. They are evaluated directly from the spherical mesh coordinates.

For unit-sphere Cartesian coordinates `(x, y, z)` with longitude `lon` and latitude `lat`:

- `smooth1 = 1 + 0.25x - 0.15y + 0.10z + 0.20 sin(2 lon) cos(lat)`
- `smooth2 = exp(0.5x - 0.25y) + 0.10 cos(3 lon) cos^2(lat)`

These fields were chosen to be smooth, nontrivial, and independent of the MIRA training/evaluation variables.

### smooth1

![smooth1 global field](../analysis_medium_improv/github_results/smooth1_global.png)

![smooth1 longitude profiles](../analysis_medium_improv/github_results/smooth1_longitude_profiles.png)

![smooth1 latitude profiles](../analysis_medium_improv/github_results/smooth1_latitude_profiles.png)

### smooth2

![smooth2 global field](../analysis_medium_improv/github_results/smooth2_global.png)

![smooth2 longitude profiles](../analysis_medium_improv/github_results/smooth2_longitude_profiles.png)

![smooth2 latitude profiles](../analysis_medium_improv/github_results/smooth2_latitude_profiles.png)

## Finest-grid solution and error visualizations

The following figures show the finest-grid analytic solution and remapping errors for the final `lmax=24` v18 stage. Each figure includes the source analytic field, target analytic truth, learned prediction, Tempest prediction, Tempest error, and learned error.

### CS to ICOD, finest grid

![CS to ICOD smooth1 finest-grid solution and errors](../analysis_medium_improv/github_results/finest_CS-r128_to_ICOD-r128_smooth1_corrected_lmax24_solution_errors.png)

![CS to ICOD smooth2 finest-grid solution and errors](../analysis_medium_improv/github_results/finest_CS-r128_to_ICOD-r128_smooth2_corrected_lmax24_solution_errors.png)

### ICOD to CS, finest grid

![ICOD to CS smooth1 finest-grid solution and errors](../analysis_medium_improv/github_results/finest_ICOD-r128_to_CS-r128_smooth1_corrected_lmax24_solution_errors.png)

![ICOD to CS smooth2 finest-grid solution and errors](../analysis_medium_improv/github_results/finest_ICOD-r128_to_CS-r128_smooth2_corrected_lmax24_solution_errors.png)

## RLL third-topology diagnostic

We also evaluated RLL as a third mesh topology using three-level refinement sequences:

- `CS-r32 -> RLL-r90-180`
- `CS-r64 -> RLL-r180-360`
- `CS-r128 -> RLL-r360-720`

and the reverse direction:

- `RLL-r90-180 -> CS-r32`
- `RLL-r180-360 -> CS-r64`
- `RLL-r360-720 -> CS-r128`

This is not a completely unseen topology for v18, because v18 was trained with some coarse RLL pairs. However, the higher-resolution refinement sequence provides an out-of-distribution RLL-resolution diagnostic.

### Summary

For `CS -> RLL`, v18 generalizes strongly. The learned operator gives lower finest-grid relative L2 errors than Tempest on all five analytic test fields, with fitted orders above first order over the three refinement levels.

For `RLL -> CS`, behavior is mixed. The learned operator remains competitive or better on the coordinate fields `x`, `y`, and `z`, but is weaker than Tempest on the nonlinear smooth fields `smooth1` and `smooth2`. This supports the concern that RLL as a source topology, especially near the poles, introduces a harder one-to-many geometric ambiguity.

These RLL results motivate a follow-up training study:

1. train a topology-holdout model with RLL excluded,
2. evaluate zero-shot performance on RLL pairs,
3. retrain with RLL included,
4. add pole-aware features such as latitude, `abs(z)`, and local area/aspect information.

The small result CSVs are included in `analysis_medium_improv/github_results/`. Large generated RLL maps and edge parquet files are excluded from Git tracking.
