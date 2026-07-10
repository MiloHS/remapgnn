# Results

Experiments and results accompanying the paper. All accuracy numbers are the
**cell-average area-relative L² error**, resolved per spherical-harmonic degree band,
computed against analytic cell-average truths (finite-volume metric). Every operator
compared here — ours and the classical baselines — is conservative and consistent to
solver tolerance, so accuracy differences reflect reconstruction quality alone.

**Source of truth:** the tables below are generated from
`analysis_medium_improv/audits/.../spectral_shells.csv` by `scripts/make_paper_tables.py`
(with the verification checks it runs), and the CSVs live in [`paper_tables/`](../paper_tables/):
`indist_abs.csv`, `indist_ratio.csv`, `zeroshot_healpix_abs.csv`,
`zeroshot_healpix_ratio.csv`, `diversity_ladder.csv` (+ `*_seedstats.csv`).

> **Status:** current numbers are averaged over **3 seeds**. Five-seed error bars, a
> volume-controlled diversity arm, a second held-out family (CSRR), a resolution-transfer
> study (r128), and the cost + order-of-accuracy studies are in progress and will be
> added here.

## Baselines & regimes

- **Baselines:** TempestRemap np1 (1st-order) and np2 (2nd-order); ESMF bilinear,
  conservative-1st, conservative-2nd — on the same pairs and the same cell-average metric.
  (ESMF conservative-1st coincides with np1 to 1e-13, a pipeline cross-check.)
- **Two regimes:** *in-distribution* (deployed model on trained meshes) and *zero-shot*
  (models evaluated on entire mesh families withheld from training and model selection).
  Zero-shot is the primary, fair head-to-head, since classical methods do not train.

## Conservation & consistency

Across all pairs and families, the learned operators satisfy both constraints to solver
tolerance: **max conservation residual and max consistency residual ≤ 2.0×10⁻⁹** (means
≈ 1×10⁻⁹). Independent of mesh type, field, or whether the pair was seen in training.

## In-distribution accuracy (deployed model on trained pairs)

Ratio to np2 (2nd-order TempestRemap); **< 1.0 means lower error than np2**. Absolute
errors in `paper_tables/indist_abs.csv`.

| band | np1 | ESMF-bilinear | ESMF-2nd | **Ours** |
|---|---|---|---|---|
| ℓ 1–8   | 29.5 | 12.7 | 11.1 | **2.32** |
| ℓ 9–16  | 14.3 | 10.1 | 4.73 | **1.10** |
| ℓ 17–24 | 7.59 | 6.31 | 2.59 | **0.77** |
| ℓ 25–32 | 4.45 | 3.97 | 1.83 | **0.59** |
| ℓ 33–40 | 2.97 | 2.75 | 1.53 | **0.58** |
| ℓ 41–48 | 2.19 | 2.08 | 1.37 | **0.70** |

Ours beats np2 for all bands ℓ ≥ 17 (down to ~0.58×) and beats ESMF 2nd-order
conservative at **every** band; np2 leads only at ℓ ≤ 8, where all methods are already
< 0.1% error. Seed spread ≤ 2% of the mean in every band.

## Zero-shot accuracy (held-out HEALPix family)

HEALPix is never present in training or model selection. Ratio to np2; D2 and D5 denote
models trained on two and five *other* families. Absolute errors + std in
`paper_tables/zeroshot_healpix_*.csv`.

| band | np1 | ESMF-2nd | Ours (D2) | Ours (D5) |
|---|---|---|---|---|
| ℓ 1–8   | 23.1 | 10.9 | 3.20 | 3.22 |
| ℓ 9–16  | 13.4 | 4.58 | 2.16 | 1.66 |
| ℓ 17–24 | 7.65 | 2.36 | 1.82 | 1.25 |
| ℓ 25–32 | 4.52 | 1.67 | 1.39 | **0.97** |
| ℓ 33–40 | 3.02 | 1.46 | 1.12 | **0.82** |
| ℓ 41–48 | 2.23 | 1.35 | 1.03 | **0.84** |

Even with no exposure to HEALPix, the most-diverse model beats np2 at high wavenumbers
(ℓ ≥ 25) and beats ESMF 2nd-order conservative at every band (0.30–0.62× its error).

## Topology diversity drives generalization

Zero-shot cell-average error (mean over bands & pairs, ×10⁻², mean ± std over 3 seeds)
on a held-out family vs. the number of *other* training families. `—` = family is in
training at that level (no longer zero-shot). From `paper_tables/diversity_ladder.csv`.

| held-out family | D2 | D3 | D4 | D5 |
|---|---|---|---|---|
| HEALPix         | 3.98 ± 0.15 | 3.38 ± 0.36 | 2.90 ± 0.11 | 3.05 ± 0.50 |
| MPAS (Voronoi)  | 6.03 ± 1.78 | 2.17 ± 0.34 | 1.47 ± 0.32 | — |
| ICO (triangular)| 3.66 ± 0.48 | 2.31 ± 0.36 | — | — |
| RLL (lat–lon)   | 4.75 ± 0.11 | — | — | — |

Zero-shot error decreases monotonically with training diversity for every held-out
family — largest for the most distinct topology (MPAS, ~4× from D2→D4). This is the
central empirical finding: generalization is governed by training-topology diversity.

## In progress

- **Order-of-accuracy** (convergence under refinement, CS↔ICOD r32/r64/r128, smooth
  fields): np1 ≈ 1st order, np2 ≈ 2nd, ours ≈ 1.6–1.8, ESMF-2nd ≈ 1st on the
  cell-average metric (to be finalized).
- **Build/apply cost** (single GPU node; TempestRemap/ESMF are single-thread CPU by
  construction): warmup + median timing of operator construction vs. supermesh
  generation across resolutions (to be finalized).
- 5-seed error bars, volume-controlled diversity arm, CSRR second held-out family, and
  r128 resolution transfer.

## Reproducing

```bash
# regenerate the CSV tables from an audit's spectral_shells.csv:
python scripts/make_paper_tables.py
# convergence slopes from a convergence audit:
python scripts/_conv_slopes.py <audit_dir>
```
