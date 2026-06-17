# Next experiments: topology generalization

The next research question is whether remapgnn learns a transferable conservative remapping rule across mesh topologies, or whether performance depends strongly on seeing each topology during training.

## Main experiment

Compare three controlled training settings.

### v20a: topology holdout

Train on CS↔ICOD mesh pairs only.

Do not include RLL source or target meshes during training.

Evaluate on:

- CS→ICOD
- ICOD→CS
- CS→RLL
- RLL→CS
- RLL→RLL

Purpose:

- Measures zero-shot transfer to RLL.
- Separates true topology generalization from RLL exposure during training.
- Includes RLL→RLL to test same-topology-family remapping across RLL resolutions.

### v20b: topology included

Train with CS, ICOD, and RLL mesh pairs using the same architecture and loss as v20a.

Evaluate on the same targets.

Purpose:

- Measures how much RLL improves when included in training.
- Provides a fair comparison against v20a.

### v20c: topology included plus pole-aware features

Train with CS, ICOD, and RLL mesh pairs, but add geometry features intended to help RLL source meshes.

Candidate added features:

- source latitude
- target latitude
- absolute source z-coordinate
- absolute target z-coordinate
- source area
- target area
- target/source area ratio
- local edge rank or neighbor count

Purpose:

- Tests whether the RLL→CS weakness is caused by insufficient local geometry information, especially near the poles.

## Metrics

For each model and direction, report:

- field relative L2 error against Tempest or analytic truth
- spherical-harmonic spectral error when available
- convergence order on analytic fields
- global conservation error
- operator-level source-area residual
- inference time and peak memory

## Hypotheses

Expected outcomes:

1. If v20a performs well on RLL, then remapgnn has strong topology generalization.
2. If v20a fails but v20b improves, then RLL performance is data-dependent.
3. If v20b still struggles on RLL→CS but v20c improves, then the issue is likely feature-limited and related to pole/area geometry.
4. If v20c does not improve RLL→CS or RLL→RLL, then the issue may be architectural or tied to Sinkhorn/candidate-graph limitations.

## Recommended first run

Start with v20a because it gives the cleanest scientific result.

The key comparison is:

- current v18: trained with some RLL exposure
- v20a: trained with RLL fully held out

If v20a still performs well on RLL, that is a strong result.
If v20a degrades, then v18's RLL performance depends on training exposure.
