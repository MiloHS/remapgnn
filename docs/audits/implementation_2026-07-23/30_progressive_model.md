# Progressive model execution audit

## Scope and audit state

This report audits commit
`f98a0e8ed06678099a98b5a6dfc91ded5d580c97`, specifically:

- `_next/remapgnn_next/progressive.py`;
- the progressive model's direct mathematical dependencies in
  `geometry.py`, `sparse.py`, `constraints.py`, `types.py`, and
  `checkpoint.py`;
- the active stage definitions in `_next/configs/progressive.json` and
  `_next/configs/high_band_candidate_01.json`;
- the approved `_next/checkpoints/progressive.pt`;
- the active `CS-r32_to_ICOD-r32` edge graph, conservative map, and clean FV
  checkpoint through `build_pair_from_files`;
- progressive-model tests and the documented test-discovery command.

The checked-out implementation file matched the target commit exactly:

```text
git show f98a0e8:_next/remapgnn_next/progressive.py | sha256sum
sha256sum _next/remapgnn_next/progressive.py
5a9aabc01d892aab927d78cbfaf9475fb56bd05d30efb6ea7b64a48176002536
```

No implementation, configuration, test, data, checkpoint, job, or existing
documentation was changed. All checks were CPU-only and read-only. The
candidate checkpoint and running job were not read for acceptance.

## Requirements traced

The requirements used were the clean architecture description in
`_next/README.md:3-28,40-49`, the active workflow description in
`docs/PROJECT_HISTORY.md:42-51`, the typed stage configuration in
`config.py:13-67`, and the implementation itself.

### Ordered execution and data flow

`ProgressiveRemapper.forward` (`progressive.py:277-299`) computes the FV result
once, initializes the prefix to FV, and invokes stages in configured order.
Every stage receives:

```text
raw_source = the unchanged input field
fv_output = the unchanged frozen base result
prefix_output = FV for stage 1, otherwise the immediately preceding result
```

Each correction is evaluated on the raw source and added to the current prefix:

```text
y_k = y_(k-1) + D_k(x, y_FV, y_(k-1)) x.
```

An independent two-stage spy check observed the same raw source and FV at both
stages, stage 1 receiving FV, stage 2 receiving stage 1, the requested
`soft`/`hard` modes in order, and
`ProgressiveDiagnostics.output == returned_output`.

### Feature layouts and semantics

Actual network pre-hooks observed the fixed dimensions claimed at
`progressive.py:23-32`:

```text
correction dynamic features: 13
global router features:       8
local router features:       24
```

The 13 dynamic correction features at `progressive.py:207-219` are:

1. absolute normalized source value at the edge;
2. absolute value centered in the FV-reference target stencil;
3. absolute standardized centered value;
4. absolute stencil mean;
5. stencil standard deviation;
6. absolute normalized prefix;
7. prefix high-pass magnitude;
8. absolute prefix/reference disagreement;
9. absolute prefix/FV update;
10. absolute directional-gradient estimate;
11. gradient magnitude;
12. source high-pass magnitude;
13. prefix curvature magnitude.

The global router receives four graph-energy features from the source and four
from the prefix (`geometry.py:171-204`). The local router receives source graph
feature mean, RMS, and maximum (12), target graph features (4), and the eight
global features, totaling 24 (`progressive.py:112-148`).

The geometry encoder receives the configured eight active edge features plus
eight intrinsic descriptors (`geometry.py:207-242`). All eight edge features
selected by both active configurations are scalar area, candidate-count,
rank, or tangent-distance quantities. They are rotation invariant. The eight
derived descriptors use norms, dot products, traces, quadratic forms, area
ratios, and the FV weight, and are invariant under a common orthogonal
transformation of source and target coordinates.

The approved checkpoint has the expected input shapes:

```text
geometry encoder:  32 x 16
message MLP:       48 x 45 = 48 x (32 + 13)
score MLP:         48 x 93 = 48 x (32 + 13 + 48)
global router:     32 x 8
local router:      32 x 24
```

The converted mid-band state contains zero columns in the later feature slots,
while retaining the common runtime layout. The selected-identity high-band
state has an exactly zero final score layer.

### Routing semantics

For probability `p` and configured thresholds `l < h`, the implemented hard
value is

```text
H(p; l, h) = clamp((p - l) / (h - l), 0, 1).
```

The five modes (`progressive.py:94-110`) are:

```text
forced_open       forward 1
forced_closed     forward 0
soft              forward p
hard              forward H
straight_through  forward H, nominal backward derivative as p
```

Thus "hard" is a clipped linear ramp between the two thresholds, not a binary
step. This is consistent with having separate low/high thresholds.

Direct boundary checks for `l=0.4`, `h=0.6` gave hard values 0 at/below 0.4,
0.5 at 0.5, and 1 at/above 0.6. Forced-open, forced-closed, soft, hard, and
straight-through all followed their stated forward equations within float32
roundoff.

The correction before projection is, in implemented notation,

```text
r_e = alpha * G_field * G_local[target(e)] * q_e
      * (score_e - sum_{e' in row(target(e))} q_e' score_e'),
```

where `q` is the positive row-normalized reference obtained from the absolute
FV weights. The projected edge correction is `D = P r`. A global field gate
whose forward value is zero is additionally selected by a boolean identity
mask after projection (`progressive.py:247-254`).

### Conservation, consistency, and affine behavior

If the finite projection produces

```text
sum_{e: target(e)=t} D_e = 0
sum_{e: source(e)=s} A_target[target(e)] D_e = 0,
```

then the correction reproduces constants and changes no global area-weighted
integral:

```text
D 1 = 0
sum_t A_target[t] (D x)[t] = 0.
```

These are numerical properties of a converged finite solve, not unconditional
architectural equalities; see PM-02.

Conditional on a row-consistent FV operator, fixed operator/stencil topology,
invariant static edge features, and nonzero affine scale `a`, the code is
affine equivariant in exact arithmetic. Area centering maps `a x + b` to
`sign(a)` times the normalized source. All field-dependent correction and
router features are absolute values, squares, norms, or sign-even graph
ratios. The projected edge correction is therefore unchanged, the row-zero
property removes `b`, and applying the correction to the raw source supplies
the factor `a`.

Rotation invariance is architectural for the correction stage under a common
orthogonal coordinate transform when the active invariant edge features,
operator, and graph topology are held fixed. It does not by itself prove that
rebuilding the FV operator on a rotated mesh is invariant; that belongs to the
FV audit.

## Independent CPU evidence

### Nontrivial synthetic stage

A complete nonuniform 5-source/4-target support was constructed independently,
with row-consistent and conservative base weights. A randomized nonzero
corrector was forced open for three fields. Residuals were recomputed with
explicit Python edge loops rather than the production reduction helper.

Observed:

```text
row residual max                         1.36e-19
area-column residual max                 1.36e-19
global integral change max               2.98e-09  (float32 output)
constant-field max error                 0
forced-closed output bit-equal to FV     true
forced-closed projected nonzeros         0
hard-below-low output bit-equal to FV    true
rotation output max absolute difference  2.98e-08
rotation projected-weight difference     5.87e-10
```

Affine results:

| Scale, offset | Relative L-infinity error | Field-gate change | Local-gate change |
|---|---:|---:|---:|
| `1.7, -0.3` | `1.36e-7` | `0` | `0` |
| `-1.2, 0.25` | `7.01e-8` | `0` | `0` |
| `1e-8, 0` | `7.56e-8` | `0` | `0` |
| `1e-12, 0` | `9.23e-8` | `0` | `0` |

### Frozen-prefix behavior

For two stages, `set_training_stage(1, phase)` produced:

| Phase | Earlier-stage trainable tensors | Selected corrector | Selected router | Module training modes |
|---|---:|---:|---:|---|
| capability | 0 | 16 | 0 | only corrector training |
| router | 0 | 0 | 12 | only router training |
| frozen | 0 | 0 | 0 | all evaluation |

After an AdamW capability step, all 40 frozen earlier-stage/router parameter
tensors remained bit-identical to their snapshots. This verifies optimizer
freezing for the exercised path. FV weights are ordinary input tensors rather
than model parameters and were not mutated.

### Approved checkpoint on active data

The approved production checkpoint and clean FV checkpoint were loaded through
the active authentication/build path. The
`CS-r32_to_ICOD-r32` pair built from:

```text
analysis_medium_improv/edge_dataset_CS-r32_to_ICOD-r32_kdist_a3p0_mink8.parquet
maps_medium_improv/map_CS-r32_to_ICOD-r32_conserve.nc
_next/checkpoints/fv_relax1.pt
_next/checkpoints/progressive.pt
```

has `6144 -> 10242` cells and 290,032 active edges. An independently generated
smoke harmonic panel for the approved mid-band stage exercised a nonzero
correction:

```text
target normalized frequency              1.25
mid-band field gate                      0.8791233
mid-band max output change               0.2036123
mid-band row residual max                2.08e-17
mid-band area-column residual max        2.86e-15
safety field gates                       all exactly 0
high-band output bit-equal to mid prefix true
```

The high-band exact identity is also structurally explained by the approved
checkpoint's exactly zero final score layer. This is stronger evidence than
only observing a closed router.

An additional active-pair random-field check (whose mid stage happened to be
globally rejected) observed exact constant reproduction, exact high-band
fallback, affine relative errors from `5.8e-8` to `1.3e-7`, and zero rotation
error. Because the active correction was closed on that random field, the
nontrivial synthetic and harmonic results above are the meaningful correction
checks.

## Findings

### PM-01 — High — exact global rejection defeats straight-through task gradients

**Requirement or claim.** The selected router is trained using
straight-through routing (`config.py:254-257`, `_next/README.md:46-49`).
Straight-through routing normally means hard forward behavior while task-loss
gradients pass through the routing decision.

**Expected behavior.** A target field whose hard global gate is closed should
return the prefix exactly in the forward pass, while a straight-through
backward pass should permit the output/task loss to teach the field router.

**Observed behavior.** `_route` constructs an STE at
`progressive.py:108-109`, but `accepted = field_gate != 0` and the subsequent
two `torch.where` operations at `progressive.py:249-254` select a constant-zero
branch whenever the global hard gate is closed. This removes the STE gradient
from the projected correction and output.

Independent test with a nonzero randomized corrector:

| Field probability | ST forward gate | Output change | Output-MSE gradient to field-router final bias |
|---:|---:|---:|---:|
| `0.2` (below low `0.4`) | `0` | `0` exactly | `0` exactly |
| `0.5` (inside ramp) | `0.5` | `3.35e-5` | `2.28e-5` |
| `0.8` (above high `0.6`) | approximately `1` | `6.69e-5` | `1.46e-5` |

Reproduction pattern:

```python
stage.set_training_phase("router")
set_field_router_probability(stage, 0.2)
output, diag = model(pair, source, gate_modes=["straight_through"])
loss = (output - truth).square().sum()
loss.backward()
assert torch.equal(output, diag.fv_output)
assert stage.field_gate_mlp.net[-1].bias.grad.item() == 0.0
```

**Independent evidence.** The derivative loss follows directly from the
boolean `accepted` tensor and the false constant branch of `torch.where`; it
does not depend on the implementation projection helper.

**Impact.** Target/guard performance losses cannot teach a globally closed
field router to reopen. The separate binary teacher and safety-probability
terms in `training.py:91-103` still produce probability gradients, so the
router is not completely stuck. However, closed fields are taught only by
role labels, not by the correction's actual benefit or harm. This materially
narrows the implemented meaning of straight-through routing and can affect
router quality near or below the rejection threshold.

**Recommended correction.** Preserve exact forward identity with an explicit
straight-through identity construction whose backward path retains the
projected correction derivative, or document and intentionally test that the
global exact-rejection mask overrides task-loss STE behavior. Add full-stage
gradient tests below, exactly at, and above both field thresholds.

**Confidence:** high.

### PM-02 — Medium — ordinary inference does not enforce projection convergence

**Requirement or claim.** Every correction is described as constrained to
preserve constants and global totals (`_next/README.md:22-28`,
`docs/PROJECT_HISTORY.md:47-51`).

**Expected behavior.** Either the stage should guarantee the configured
constraint tolerances, or inference should fail clearly when its finite solve
does not meet them.

**Observed behavior.** `StageConfig` accepts any positive
`projection_iterations` (`config.py:43-44`). The stage calls
`project_correction` without `assert_converged=True` or returned convergence
information (`progressive.py:243-246`). Training and the full audit separately
inspect residuals, but ordinary `model(pair, source)` inference returns any
finite-iteration result.

An irregular connected 7-source/6-target support with a legal
`projection_iterations=1` and nonzero randomized corrector returned:

```text
row residual max      8.67e-19
area-column residual  6.68e-4
output finite         true
```

Increasing the same test to 2 and 5 iterations left column residuals
`3.21e-4` and `3.56e-6`; 20 iterations reached `5.42e-19`.

Reproduction:

```python
cfg = StageConfig(..., projection_iterations=1)
stage = ConservativeCorrectionStage(cfg)
randomize_nonzero_corrector(stage)
_, diag = ProgressiveRemapper(operator, [stage])(
    irregular_pair, source, gate_modes=["forced_open"]
)
assert diag.stages[0].column_residual.abs().max() > 1e-10
```

**Independent evidence.** Column residuals were evaluated directly from
`sum A_target[t(e)] D_e` and are exposed unchanged in stage diagnostics.

**Impact.** Active configurations use 200 iterations, and the exercised active
pair passed by a wide margin (`2.86e-15` versus `1e-10`). Therefore this is not
evidence that current scientific results violate conservation. It is a
configuration/runtime guarantee gap: a valid future configuration or difficult
mesh can silently produce a model that contradicts the documented conservation
claim outside trainer/auditor entry points.

**Recommended correction.** Make tolerance-aware convergence part of the
stage/runtime contract, record iterations and relative residual in stage
diagnostics, and reject or explicitly label unconverged inference. Add an
adversarial legal-configuration test.

**Confidence:** high.

### PM-03 — Low — a zero local gate is not a zero applied correction in that region

**Requirement or claim.** `docs/PROJECT_HISTORY.md:47-50` says local decisions
can reduce a correction in individual regions.

**Expected behavior.** The exact semantics of local routing should make clear
whether zero means local rejection or only zeroing the pre-projection proposal.

**Observed behavior.** Local gates multiply `raw_delta` before the global
projection (`progressive.py:238-246`). Projection may redistribute edge
weights back into a target whose local gate was zero in order to satisfy source
column constraints. In an independent synthetic check, forcing local gate zero
on target 0 and one elsewhere produced projected target-0 edge magnitudes up to
`1.36e-4`.

**Impact.** This is mathematically compatible with "reduce" and is necessary
for joint conservation, so it is not a scientific violation by itself.
However, `local_gate` diagnostics are proposal gates, not exact masks on the
applied correction. Interpreting them as local acceptance/rejection can be
misleading.

**Recommended correction.** Document the pre-projection meaning and report an
applied per-target correction magnitude or projection leakage diagnostic when
local routing is analyzed.

**Confidence:** high.

### PM-04 — Low — progressive inference has an undocumented float32 geometry/area assumption

**Requirement or claim.** `PairData` and `SparseOperator` are public typed
interfaces, but do not constrain floating dtypes (`types.py:16-171`).

**Expected behavior.** A valid pair should either run with a supported dtype or
fail validation with a clear dtype requirement.

**Observed behavior.** A pair with float32 edge features and float64 areas
passes both dataclass constructors, but derived intrinsic features become
float64 and are concatenated with float32 edge features. The float32 geometry
MLP then fails with:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

The active file-built pair uses float32 edge features and areas with float64 FV
weights and runs correctly.

**Impact.** No active-data failure was observed. This is an interface and
diagnostic weakness for future pair construction or CPU precision studies.

**Recommended correction.** Validate the supported dtype combination or cast
all geometry-derived inputs to the stage parameter dtype in one documented
place.

**Confidence:** high.

### PM-05 — Medium — the documented test command silently omits 13 function tests

**Requirement or claim.** `_next/README.md:55-59` identifies unittest discovery
as the verification command.

**Expected behavior.** The command should execute the progressive, projection,
sparse, panel, and freezing tests stored under `_next/tests`.

**Observed behavior.** The documented command passed 12 methods from
`test_workflow_unittest.py` but did not collect any of the 13 top-level
pytest-style functions in:

- `test_model_equivalence.py`;
- `test_training.py`;
- `test_constraints.py`;
- `test_sparse.py`;
- `test_panels.py`.

The project Python environment has no pytest:

```text
python -m pytest ...
No module named pytest
```

The fallback object in `test_model_equivalence.py` makes import succeed but
does not make unittest discover top-level functions. In particular, the
dedicated optimizer-step freezing test and parametrized negative/tiny-scale
tests are not run by the advertised command.

Reproduction:

```text
PYTHONPATH=_next .../python -B -m unittest discover -s _next/tests -p 'test_*.py' -v
Ran 12 tests ... OK

rg '^def test_' _next/tests/*.py | wc -l
13
```

**Impact.** Important progressive checks appear present but are not part of
the normal verification result. Several are duplicated partially in the
12-method workflow test, and this audit independently exercised the relevant
properties, so this does not establish a current model defect. It weakens
regression protection.

**Recommended correction.** Standardize on a test runner that is installed and
collects every file, or convert all tests to `unittest.TestCase`; add a
collection-count/expected-test-name check in CI.

**Confidence:** high.

## Material non-findings

- Stage execution order, raw-source reuse, fixed FV reuse, and prefix chaining
  matched the documented architecture.
- The active 13/8/24 feature layouts were observed at runtime and agree with
  checkpoint matrix dimensions.
- Active static edge and derived correction geometry features are
  rotation-invariant scalars.
- Forced-closed and hard-below-low global rejection returned the prefix
  bit-for-bit and zeroed all projected correction weights in exercised cases.
- Positive, negative, and tiny nonzero scaling and offsets behaved affine
  within approximately `1.4e-7` relative error on a nonzero synthetic
  correction.
- Common coordinate rotation changed the synthetic nonzero output by only
  `2.98e-8` absolute; the active pair check was bit-identical.
- Capability/router/frozen parameter masks and module train/eval states were
  correct. Frozen earlier-stage and router parameters were bit-identical after
  an optimizer step.
- Correction row/area-column residuals passed active audit tolerances on the
  exercised real target harmonic by more than four orders of magnitude.
- The approved selected-identity high-band stage returns the mid-band prefix
  exactly; its final score layer is exactly zero, so this does not depend on
  a lucky sample.
- The active progressive implementation imports no archived implementation.

## Deferred checks

- GPU projection, routing, invariance, and CPU/GPU parity are deferred by the
  audit contract.
- Acceptance and behavioral testing of the currently training high-band
  candidate are deferred until the job finishes.
- Protected and r128 full-mesh execution are deferred; mesh-family coverage is
  also owned by the FV and acceptance specialists.
- End-to-end rotation invariance after rebuilding an FV operator is outside
  this correction-stage result and belongs to the FV construction audit.
- Empirical router quality, high-band gain, and safety promotion are model
  quality/acceptance questions, not architectural proofs.

## Remaining uncertainty

Only one active mesh pair and a small harmonic panel were exercised here.
Although the active residual margins were large, convergence across every
active mesh is empirical until the deferred full matrix is run. Float32
roundoff can change gates very near configured thresholds; exact boundary
policy was verified algebraically/directly, but no claim is made about
statistical robustness of learned probabilities near those thresholds.
