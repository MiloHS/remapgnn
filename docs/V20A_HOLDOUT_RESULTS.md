# v20a topology-holdout results

v20a tests whether remapgnn transfers to RLL without seeing RLL during training.

## Setup

v20a was trained only on CS↔ICOD pairs:

- `CS-r32_to_ICOD-r32`
- `ICOD-r32_to_CS-r32`

All RLL pairs were removed from training, validation, testing, and checkpoint scoring.

The v20a model has:

- a topology-holdout base GNN/Sinkhorn remapper
- an IRNO-style corrector
- correction stages at `lmax=8`, `lmax=16`, and `lmax=24`

## Main result

v20a shows partial zero-shot transfer but is not competitive overall.

The most important result is not just fitted order. We report both:

1. finest-grid actual relative L2 error,
2. finest-grid error ratio versus Tempest,
3. fitted convergence order,
4. behavior across correction stages.

## Actual-error summary

From `analysis_medium_improv/github_results/v20a_actual_error_stage_summary.csv`:

- CS→ICOD:
  - v20a base mean finest-grid error ratio: about 1.72× Tempest
  - v20a base mean fitted order: about 1.14
  - Interpretation: decent refinement trend, but worse actual errors than Tempest.

- CS→RLL:
  - v20a base mean finest-grid error ratio: about 1.30× Tempest
  - v20a base mean fitted order: about 0.76
  - Interpretation: partial zero-shot transfer. Coordinate-like fields are sometimes competitive, but smooth nonlinear fields are worse.

- ICOD→CS:
  - v20a base mean finest-grid error ratio: about 4.51× Tempest
  - v20a base mean fitted order: about 0.62
  - Interpretation: clear failure relative to Tempest, especially on smooth fields.

- RLL→CS:
  - v20a base mean finest-grid error ratio: about 2.01× Tempest
  - v20a base mean fitted order: about 0.52
  - Interpretation: weak reverse-direction transfer.

## Clean re-run (audit)

After the evaluation-leakage fixes (see `AUDIT_REPORT.md`) and retraining with a clean split
(held-out validation `{CS-r16↔ICOD-r16, ICOD-r16_to_CS-r16}`, train-only stats), v20a **reproduces**
the numbers above within ~10–20%: clean base finest-grid ratios CS→ICOD ≈ 1.75×, ICOD→CS ≈ 4.10×,
CS→RLL ≈ 1.36×, RLL→CS ≈ 1.84× (vs published 1.72 / 4.51 / 1.30 / 2.01). v20a's leakage was minor
(it trains on only 2 pairs), so little changed — this run validates the eval pipeline (Tempest
errors match the originals to machine precision). The v20b result, by contrast, was notably
leakage-inflated; see `V20B_DIVERSE_TOPOLOGY_RESULTS.md`.

## Corrector behavior

The IRNO corrector is not robust under topology holdout.

Across all tested directions, the base stage is usually best. The correction stages generally increase finest-grid error:

- base is best,
- `lmax=8` is usually worse,
- `lmax=16` is worse still,
- `lmax=24` is often the worst.

This suggests that the corrector learned useful refinements only when the target topology family was represented during training.

## Interpretation

v18's strong RLL behavior was not purely zero-shot topology generalization. RLL exposure during training appears to matter.

v20a does learn some geometry-aware behavior, especially for coordinate-like fields and CS→RLL, but it does not reliably transfer to RLL or reverse topology directions.

## Next experiments

The next controlled experiment should be v20b:

- same architecture as v20a,
- include CS, ICOD, and RLL in training,
- evaluate on the same convergence suite,
- compare v20a versus v20b versus v18.

If v20b recovers v18-like RLL behavior, then RLL exposure is the main missing ingredient. If v20b still struggles, v20c should add pole-aware and topology-aware features.
