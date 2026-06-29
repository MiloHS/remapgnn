# Model lineage and lessons learned

This document is the compact public history of the RemapGNN experiments.  It
keeps the ideas that matter without carrying every old sweep, log, and negative
result into the GitHub-facing repo.

## Current default

The current default is:

- model: `v12_geom_base`
- checkpoint: `models_medium_improv/highorder_signed_v12_geom_mom1e4.pt`
- config: `configs/v20b_base_a3p0_mink8_geom_v12.json`
- inference projection: float64, `eps_rel=1e-12`, `n_cg=800`

This is the version documented in `docs/INFERENCE.md`.

## Main technical shift

The project started with nonnegative GNN/Sinkhorn remapping operators.  Those
were useful as first conservative neural baselines, but they are structurally
limited: a nonnegative conservative linear operator is monotone, and monotone
conservative remapping is effectively first-order limited.

The current direction uses:

- signed learned edge masses;
- a doubly constrained projection layer instead of Sinkhorn;
- geometric features derived from centers, areas, local tangent coordinates,
  and candidate-graph statistics;
- float64 projection at inference for clean conservation.

That combination is the core of `v12_geom_base`.

## What worked

### Signed weights + projection

Allowing signed edge masses was the important high-order step.  The projection
then enforces the two remapping constraints:

- source marginal / conservation;
- target marginal / constant consistency.

The cleaned float64 projection is what brings learned conservation residuals to
about `2e-9` in the current audit.

### Geometry features

The v12 geometry features helped the learned model become more structurally
reasonable.  They improved real-field behavior and moment diagnostics compared
with earlier learned baselines.

### Real-field and moment audits

The useful evaluation suite is not only field relative error.  The current
audit checks:

- real climate-like fields;
- analytic fields;
- spectral shells;
- conservation and consistency residuals;
- Cartesian order-1/order-2 moment diagnostics;
- deployment timing.

Those audits are what make the current result defensible.

### Supermesh-free operator construction

The learned operator does not beat cached TempestRemap maps.  Its useful
efficiency story is different: it can construct a new conservative operator
faster than the tested TempestRemap `np2` overlap/offline-map generation path.

## What did not work well

### Iterative correctors as the main answer

The earlier v18/v10b-style corrector idea improved some in-sample or spectral
metrics, but it did not become the clean default.  The improvements were not
robust enough on real fields and moment diagnostics, and wider/more aggressive
correctors often gave back structural quality.

The lesson was: keep the useful evaluation discipline from the corrector work,
but do not make the corrector the headline model.

### More bands / wider correction stencils

Training on more spectral bands and using larger candidate stencils was a good
hypothesis, but the tested wide-stencil 6-band direction was not an improvement
overall.  It helped some spectral/analytic numbers but hurt the default
real-field/moment tradeoff.

### Claiming superiority over TempestRemap

That is not supported.  TempestRemap `np2` is still more accurate in the current
audit, and cached Tempest maps are still faster to load/apply.  The honest claim
is supermesh-free learned construction with good conservation and reasonable
accuracy.

## Current result in one paragraph

`v12_geom_base` with cleaned float64 projection is the current public prototype.
It is not as accurate as TempestRemap `np2`, but it is much better than `np1`,
better than earlier learned baselines on real fields, conservative to about
`2e-9`, and faster than generating new TempestRemap `np2` maps/supermeshes at
the tested r32/r64 resolutions.

See `docs/CURRENT_RESULTS.md` for the exact numbers.

## Good next steps

The most useful next work is tool hardening, not another speculative model
sweep:

1. publish the v12 checkpoint as a GitHub release asset;
2. test `scripts/build_remap_operator.py` on a fresh mesh pair outside the
   original audit set;
3. add a tiny example dataset or synthetic mesh fixture for CI/smoke tests;
4. reduce the remaining helper-script coupling by moving shared utilities from
   `scripts/` into the `remapgnn/` package.
