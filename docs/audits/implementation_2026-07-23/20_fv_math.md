# FV construction and mathematical-core audit

## Scope and snapshot

This report audits commit
`f98a0e8ed06678099a98b5a6dfc91ded5d580c97` and the active, non-archived
inputs consumed by `_next`. The work was read-only except for this report and
diagnostic programs under `/tmp`.

Inspected implementation paths:

- `_next/remapgnn_next/{types,sparse,constraints,geometry,fields,fv}.py`
- the FV use paths in `progressive.py`, `training.py`, `evaluation.py`,
  `checkpoint.py`, `scripts/build_fv.py`, `scripts/train.py`, and
  `scripts/audit.py`
- active FV and progressive checkpoint metadata
- all 11 edge/map pairs named by the two active configurations
- the checked-in equivalence record and existing mathematical tests

The active FV checkpoint SHA-256 is
`18df156b418835bbb6ece1bb3eb246156e7d58e48481778c68088bfe4b60efdc`.
It matches both the active progressive checkpoint's FV reference and the
checked-in equivalence record. Its internal tensor-state hash also recomputed
exactly as
`32382baf4c7ca6a5590edc24899c0e386a9b1d62239bf3dd750c4df04e5e8bed`.

## Implemented mathematics

For an edge `e=(i,j)` from source cell `j` to target cell `i`, sparse
application is

```text
y_i = sum_{e:t(e)=i} S_e x_{s(e)}.
```

`SparseOperator.mass` represents `M_e = area_target_i S_e`. Constant
consistency and integral conservation are therefore respectively

```text
sum_{e:t(e)=i} S_e = 1,
sum_{e:s(e)=j} M_e = area_source_j.
```

The FV geometry network produces a signed score. The builder forms

```text
q_e = area_target_i / degree_i * (1 + scale * signed_score_e),
```

then alternates local ridge-regularized degree-1/degree-2 moment corrections
with a joint marginal solve. The marginal implementation solves

```text
(A A^T + epsilon I) lambda = b - A q,
M = q + A^T lambda.
```

This is a regularized least-squares correction. It approaches the exact
Euclidean projection as `epsilon -> 0`; with nonzero `epsilon`, its exact
architectural guarantee is the regularized linear system, not exact
`A M = b`. Moment matching is also intentionally approximate because the
local systems contain ridge terms and each moment step is followed by a
marginal projection.

For a learned normalized correction `D`, the code first applies the
orthogonal target-row projector `R`, then solves the source Laplacian

```text
L = B R B^T,
phi = L^+ B R d,
D = R d - R B^T phi.
```

Here `B` contains the target-area factor on each source marginal. At a
converged float64 solve this is the Euclidean orthogonal projector onto

```text
sum_j D_ij = 0,
sum_i area_target_i D_ij = 0.
```

The custom backward applies the same projector, which is the correct adjoint
when the finite solve has converged closely enough to that orthogonal
projector. This is a numerical guarantee, not a finite-iteration proof for
arbitrary ill-conditioned inputs.

## Findings

### FV-MATH-001 — High — production equivalence is not bound to the accepted checkpoint

- **Requirement or claim:** A checkpoint accepted as production must be the
  checkpoint whose scientific behavior passed equivalence. Runtime
  normalization is scientific state because it changes FV/progressive edge
  features.
- **Expected behavior:** `require_production=True` should authenticate the
  complete behavior-affecting checkpoint against the equivalence result.
- **Observed behavior:** The active `_next/checkpoints/progressive.pt` hashes
  to `4a64d9c43f6f39059d390c3d2bca35f08b7e36309e6c72dcdc520e767d0d7c15`,
  while both `_next/reports/equivalence_completed_v24f.json` and the embedded
  equivalence values name
  `82ea246a65624c2471654e1e11165ef0dfbe40c811d388638183958cf00815ed`.
  `_validated_progressive_pack` checks only the format/schema, the booleans
  `production` and `equivalence.passed`, and individual stage-state hashes.
  It does not bind `runtime_data`, stage configurations, sources, or other
  behavior-affecting checkpoint content to the equivalence evidence.
- **Code and data locations:** `_next/remapgnn_next/checkpoint.py:34-51`;
  `_next/checkpoints/progressive.pt`;
  `_next/reports/equivalence_completed_v24f.json`.
- **Reproduction:** `sha256sum _next/checkpoints/progressive.pt
  _next/checkpoints/fv_relax1.pt`, followed by inspecting
  `conversion_checks.equivalence.values.clean_checkpoint.sha256`. The
  coordinating agent independently copied the pack under `/tmp`, added
  `1000` to `runtime_data.normalization.edge_mean[0,0]`, and confirmed
  `_validated_progressive_pack(..., require_production=True)` still returned
  successfully.
- **Independent evidence:** The perturbation changes a normalization value
  used at `_next/remapgnn_next/fv.py:255-257` yet does not invalidate
  production status. The active FV file itself *is* correctly bound by its
  full-file hash; the gap concerns the progressive production/equivalence
  envelope.
- **Impact:** An accidental or malicious change to scientific runtime data
  can be labeled production and inherit a passing equivalence result it did
  not earn. This can materially change model/FV inputs without an acceptance
  failure.
- **Recommended correction:** Use a detached signed manifest: hash the exact
  immutable checkpoint payload excluding only the manifest reference, store
  that payload hash in the equivalence report, and require it on load. Include
  runtime normalization, all stage configs/states, FV reference, feature
  names, and source provenance in the authenticated payload. Do not embed a
  whole-file self-hash into the same serialized file.
- **Confidence:** High. The hash mismatch is direct, and the independent
  normalization perturbation was accepted.

### FV-MATH-002 — Medium — marginal “convergence” does not assert the marginals

- **Requirement or claim:** The final FV step is a joint consistency and
  conservation projection with usable CG convergence assertions.
- **Expected behavior:** `assert_converged=True` should reject a result that
  violates requested source/target marginal tolerances, and the authoritative
  FV builder should fail closed on a failed or infeasible projection.
- **Observed behavior:** `project_marginals` computes actual target/source
  residual maxima at `_next/remapgnn_next/constraints.py:194-196`, but defines
  `converged` only from the regularized linear-system residual at line 198.
  `assert_converged` checks only that boolean. On an independently constructed
  graph with source total `0.9` and target total `1.0`, it warned that exact
  marginals were impossible but returned `converged=True` with target and
  source maximum residuals both `0.0125`; `assert_converged=True` would not
  raise. `project_with_moment_relaxation` discards all `ProjectionInfo`, and
  `build_fv_operator` has no final finite/marginal assertion.
- **Code locations:** `_next/remapgnn_next/constraints.py:157-204`,
  especially lines 166-169 and 194-203; calls that discard diagnostics at
  lines 253, 262, and 268; `_next/remapgnn_next/fv.py:168-181`.
- **Reproduction command:**  
  `PYTHONPATH=_next /gpfs/fs1/home/mschlittgenli/.conda/envs/remap_gpu/bin/python -B /tmp/fv_audit_numeric.py`
  
  The same check independently formed dense `A`, solved
  `(A A^T + epsilon I)lambda=b-Aq` with NumPy, and verified the implementation
  to `5.55e-17`, so the issue is the convergence/acceptance definition rather
  than the regularized solver algebra.
- **Independent evidence:** With compatible totals and
  `epsilon_relative=1e-9`, the function reported `converged=True` and linear
  residual `6.46e-17` while actual marginal residual was `3.84e-10`, exactly
  matching the independent dense regularized solution's bias.
- **Impact:** The current authenticated inputs happen to be safe within audit
  tolerances, but a future map, damaged graph, non-finite network output, or
  unsuitable build setting can silently create a non-conservative
  “authoritative” FV operator.
- **Recommended correction:** Validate finiteness, positive areas, complete
  graph coverage, and compatible total areas; define convergence using both
  linear-solve and actual marginal residuals with explicit tolerances; request
  and assert diagnostics after every projection or at least after the final
  one; make infeasible totals an error unless an explicitly documented
  reconciliation policy is selected.
- **Confidence:** High.

### FV-MATH-003 — Low — exact map areas are narrowed to float32 before projection

- **Requirement or claim:** The joint FV solve uses the map cell areas and
  preserves both marginals to tight numerical tolerance.
- **Expected behavior:** Geometry-network features may be float32, but
  constraint areas should retain the available float64 map/parquet precision.
- **Observed behavior:** `normalized_feature_tensors.unique` creates every
  cell array as float32, including `src_area` and `tgt_area`
  (`geometry.py:146-160`). The original map and parquet areas are float64 and
  have matching `4*pi` totals. After runtime float32 conversion, source/target
  total mismatches across active pairs range from `3.958e-09` to
  `4.680e-08` absolute (sign depends on direction). Exact simultaneous
  marginals are therefore impossible, although the discrepancy is below the
  current warning threshold.
- **Code/data locations:** `_next/remapgnn_next/geometry.py:138-168`; all
  active `analysis_medium_improv/edge_dataset_*.parquet` and corresponding
  `maps_medium_improv/map_*_conserve.nc`.
- **Reproduction command:**  
  `/gpfs/fs1/home/mschlittgenli/.conda/envs/remap_gpu/bin/python -B /tmp/fv_audit_data_stats.py`
- **Independent evidence:** Direct NumPy sums of unique float64 areas matched
  exactly for nine pairs and within `1.776e-15` for both r128 directions.
  Casting those same arrays to float32 reproduced the runtime mismatches. An
  actual clean CPU build of `CS-r32_to_ICOD-r32` had constant-row error
  `1.0814e-09` and source/target mass residual maxima about `8.97e-13`.
- **Impact:** Present errors are far below the active constant threshold
  (`2e-6`) and FV marginal audit scales, so no current scientific failure was
  observed. The narrowing weakens an otherwise avoidable conservation
  guarantee and hides the infeasibility from the `1e-6` relative warning.
- **Recommended correction:** Load coordinates/features in float32 as needed,
  but carry cell areas separately in float64 from the map/parquet file and
  compare their totals before solving.
- **Confidence:** High.

### FV-MATH-004 — Medium — automated tests omit the authoritative FV solve

- **Requirement or claim:** Sparse operations, marginal projection, moment
  relaxation, FV construction, convergence, adjoints, and supported mesh
  families should be regression-tested.
- **Expected behavior:** Discoverable tests should independently cover
  `project_marginals`, `local_moment_correction`,
  `project_with_moment_relaxation`, `build_fv_operator`, bad/infeasible
  inputs, actual marginal residuals, and at least affordable active-data
  builds.
- **Observed behavior:** `test_sparse.py` checks one scalar indexed sum and
  one complete uniform operator. `test_constraints.py` covers only correction
  projection on the same complete uniform synthetic graph. Its backward
  expected value is computed by calling the same projector again. No
  discoverable test invokes any FV marginal/moment/build function. Mesh-family
  equivalence exists only as a checked-in result artifact, not as a currently
  runnable test.
- **Locations:** `_next/tests/test_sparse.py`,
  `_next/tests/test_constraints.py`,
  `_next/reports/equivalence_completed_v24f.json`.
- **Reproduction:** `rg -n "project_marginals|local_moment_correction|
  project_with_moment_relaxation|build_fv_operator" _next/tests` returns no
  tests. The independent irregular-graph checks in
  `/tmp/fv_audit_numeric.py` exercise the missing algebra.
- **Independent evidence:** The independent dense projector agreed with
  `project_correction` to `3.61e-15`; its independently computed backward
  agreed to `4.22e-15`. Those strong results are not protected by the existing
  suite. The same diagnostic exposed FV-MATH-002.
- **Impact:** A regression in the scientific base construction or its failure
  behavior can pass all current unit tests. The static golden JSON cannot
  detect a future code change by itself.
- **Recommended correction:** Add small dense-reference tests for sparse and
  both projectors, incompatible totals, tolerance boundaries, finite/NaN
  behavior, ridge moment reduction, final FV marginals, signed masses, and
  map/edge ordering. Add scheduled r32/r64/HeALPix/r128 rebuild equivalence,
  with r128 outside the fast suite.
- **Confidence:** High.

## Checks performed and material non-findings

### Independent sparse and correction checks

`/tmp/fv_audit_numeric.py` used Python loops and explicit NumPy constraint
matrices rather than implementation helpers for expected results.

- Batched three-dimensional edge reduction matched the loop exactly.
- Batched sparse application matched the loop exactly.
- On an irregular 9-edge graph with unequal target areas, the float64
  correction projector matched
  `P = I - C^+ C` to `3.61e-15`.
- Constraint residual was `2.22e-16`; idempotence error was `1.11e-16`.
- Custom backward matched the independent dense adjoint to `4.22e-15`.
- Float32 input is not sufficient for the default `1e-10` column tolerance
  on that adversarial graph (`1.49e-08`), but the active progressive path
  explicitly converts corrections to float64 before projection
  (`progressive.py:239-246`).

These checks establish the intended projector algebra for a converged
float64 solve. They do not prove 200 iterations for every graph.

### Active graph/data structure

Direct inspection of all 11 configured pairs found:

- every source and target cell has at least 8 incident candidate edges;
- no duplicate `(target,source)` edges;
- edge rows are target-ordered, and the clean loader preserves that order;
- repeated per-edge cell areas are internally identical;
- float64 parquet areas match map `area_a/area_b` arrays exactly;
- float64 source/target totals equal `4*pi` (r128 differs only by
  `1.776e-15`);
- the map/edge center-order check is enforced by
  `grid_moments(..., expected_centers=...)`.

The active edge counts range from 290,032 (r32 reciprocal CS/ICOD) to
4,640,212 (r128 reciprocal CS/ICOD). Target degrees range from 10 to 256
across the active non-external transfers; all exceed the moment-system
minimums. This materially increases confidence in index/order and graph
coverage assumptions for current data.

### Real-support correction convergence

Two-field random float64 corrections were projected on six active supports:

| Pair | iterations of 200 | row max | area-column max |
|---|---:|---:|---:|
| CS-r32 to ICOD-r32 | 88 | `8.16e-15` | `8.57e-13` |
| CS-r32 to RLL-r90-180 | 119 | `8.91e-15` | `4.04e-13` |
| ICOD-r32 to RLL-r90-180 | 150 | `9.38e-15` | `1.33e-12` |
| CS-r64 to ICOD-r64 | 176 | `1.09e-14` | `3.27e-13` |
| ICO-r32 to CS-r32 | 109 | `1.96e-14` | `1.37e-12` |
| CS-r32 to HP-n32 | 82 | `9.55e-15` | `3.12e-13` |

All pass configured `1e-8` row and `1e-10` area-column limits. The r64
random check uses 176/200 iterations, so the iteration count remains an
empirical margin. Training independently fails immediately if an active
correction exceeds those residual limits (`training.py:353-355`), and audit
does the same (`evaluation.py:78-81`).

### Actual clean FV build

An independent CPU build of `CS-r32_to_ICOD-r32` from the active clean FV
checkpoint and active files completed successfully. NumPy `bincount`,
independent from the Torch reductions under test, measured:

```text
target mass marginal max abs    8.9654e-13
source mass marginal max abs    8.9727e-13
normalized row max abs          1.0814e-09
mass = area_target * S max abs  1.0842e-19
constant application max abs    1.0814e-09
negative mass fraction          0.46735
linear cell-average error max   3.7855e-04
quadratic cell-average error max 3.6350e-04
```

Thus current r32 consistency/conservation is within the documented audit
tolerances. Linear/quadratic moments are relaxed, not exact, as expected from
the ridge/alternating formulation; no active acceptance threshold for these
moment residuals was found.

### Recorded mesh-family evidence

The checked-in equivalence record reports exact edge ordering and FV mass/
weight agreement on r32, r64, HeALPix, and r128, with global maxima
`1.6862e-15` (mass) and `2.1329e-12` (normalized weight). This is useful
external evidence, but it was not regenerated in this initial audit, and
FV-MATH-001 prevents treating its progressive-checkpoint hash as current
authentication.

## Deferred checks

Deferred under the audit instructions:

- GPU projection parity and GPU nondeterministic reduction behavior.
- Rebuilding the r128 FV operators (recorded prior runtime was 644 seconds for
  one r128 pair).
- Full active rebuilds for every mesh direction and mesh-family golden
  comparison.
- Candidate-checkpoint acceptance while training is active.
- Large finite-difference gradient tests on active supports.

## Remaining uncertainty

- The affordable checks establish convergence on representative active
  supports, not a proof that 200 correction iterations or 800 marginal
  iterations suffice for every future mesh.
- The frozen geometry-network weights were authenticated and structurally
  loaded, but their historical training quality was not re-audited because FV
  retraining and archived implementation are outside scope.
- Moment errors were measured on one actual r32 build; no configured
  scientific bound defines how much degree-1/degree-2 relaxation is required.
- The recorded r64/HeALPix/r128 equivalence should be rerun after the
  production-manifest issue is corrected, so the resulting evidence can be
  bound to the exact accepted payload.
