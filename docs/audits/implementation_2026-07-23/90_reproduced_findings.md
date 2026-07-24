# Coordinator reproduction and deduplication

## Method

The coordinator reviewed all six specialist reports, inspected the cited
implementation paths, reran the official test command, and independently
reproduced the highest-impact behaviors with read-only or `/tmp` diagnostics.
Overlapping findings are consolidated below.

No finding was tested by changing an active checkpoint, dataset, map,
configuration, job, or implementation file.

## Critical

### CRIT-01 — Audit promotion fails open on non-finite values

Source: `AUD-ACC-001`.

Confirmed independently. A complete synthetic target/safety/prior detail set
with every metric set to `NaN`, together with `NaN` structural residuals and
errors, produced:

```text
passed=True
failures=[]
```

The cause is direct: promotion uses only `value > threshold`; every ordered
comparison with `NaN` is false. Structural checks have the same fail-open
comparison pattern. This can return audit exit status zero for non-finite
scientific results.

Current-run interpretation: this is a real acceptance defect. It is not
evidence that the currently training candidate has produced a NaN.

## High

| ID | Consolidated finding | Source reports | Coordinator verification |
|---|---|---|---|
| HIGH-01 | Target panels can include harmonics above the configured inclusive upper band because the last degree uses `round(upper*K)` instead of `floor(upper*K)`. | `DATA-01` | Direct calculation confirmed ICOD-r32 `1.5006345`, RLL `1.5011107`, ICO-r32 `1.5062370`; CS-r32 is exactly `1.5`. |
| HIGH-02 | Real-field files consumed by training and selection are absent from resume authentication; audit provenance also omits them. | `DATA-02`, `TRAIN-RESUME-02`, `CFG-03` | Static trace confirmed `build_panel -> real_field_batch` consumes the files while `_auth()` hashes only edge/map/source checkpoint. Active inventory confirms the files and five variables are actually used on applicable pairs. |
| HIGH-03 | Production equivalence is not bound to the exact accepted checkpoint/runtime payload. | `FV-MATH-001`, `CFG-01` | Active checkpoint hash is `4a64d9c...`, while checked-in and embedded equivalence evidence names `82ea246...`. A `/tmp` copy with runtime `edge_mean[0,0]` increased by 1000 was still returned by `_validated_progressive_pack(..., require_production=True)`. |
| HIGH-04 | Exact global rejection removes straight-through task-loss gradients for a closed field gate. | `PM-01` | Direct PyTorch reconstruction of the exact STE-plus-boolean-`where` path produced exact zero forward correction and exact zero router gradient below the low threshold. Explicit BCE router teaching remains active. |
| HIGH-05 | The advertised no-argument `./next audit-candidate` points to `progressive_next.pt`, not the current `high_band_candidate_01.pt`. | `AUD-ACC-002` | Root command hard-codes `progressive.json`; resolving both configs confirmed the two different checkpoint paths. |
| HIGH-06 | Training provenance is recomputed at checkpoint time rather than frozen against the objects loaded at run start. | `TRAIN-RESUME-01` | Code trace confirms `_pack()` calls a fresh `_auth()` every epoch. A file changed after pair/model loading can therefore be recorded with its new hash while the completed epoch used old in-memory contents. Specialist isolated perturbation reproduced the mixed-generation acceptance. |
| HIGH-07 | Candidate audit does not bind the candidate state and saved training configuration/provenance to the separately supplied audit configuration. | `CFG-02` | `load_training_checkpoint()` checks schema/completion and source checkpoint hash, then `audit.py` uses the independent CLI config for panels, pairs, and thresholds. Best/identity model states have no verified integrity hash. |
| HIGH-08 | Unknown top-level JSON keys are silently ignored. | `CFG-04` | Adding `losss: {guard_weight: 999}` to an in-memory copy loaded successfully, retained `guard_weight=6`, and omitted `losss` from `to_dict()` and authentication. |
| HIGH-09 | The documented PBS `EXTRA="..." qsub ...` form does not reliably export `EXTRA` into the job. | `CFG-05` | Documentation and PBS script trace confirm the mismatch. It also matches the observed accidental default-config run. PBS `-v EXTRA="..."` is required by this workflow. |
| HIGH-10 | The active 80+24 epoch candidate cannot finish within the documented 12-hour single job. | `CFG-06` | Completed capability epochs take about 639–666 seconds. Eighty capability epochs alone project to roughly 14.4 hours, before setup and router training. |
| HIGH-11 | `--require-production` is ignored when audit loads a clean training checkpoint. | `CFG-07` | The audit entry point applies the flag only in the progressive-checkpoint branch. Wrapper arguments can override the pinned checkpoint, including for protected audits. |

No High finding establishes that the frozen FV weights or accepted mid-band
state are presently numerically wrong. Independent active-mesh checks found
large margins on the tested conservation and consistency residuals.

## Medium

The following distinct Medium findings were retained after deduplication:

| ID | Finding | Source |
|---|---|---|
| MED-01 | `frequency_cells_per_k_squared` changes target selection but not stored harmonic frequency or safety-degree selection. Active configs use the hard-coded value `6`, so this is latent for current runs. | `DATA-03`, `CFG-08` |
| MED-02 | Complete panel source identities are not globally/split-disjoint; analytic and real anchors are shared and mixture keys lack full semantic identity. The intended exception policy is undocumented. | `DATA-04` |
| MED-03 | np2 files used for reported comparisons are not hashed; report output files are also not content-authenticated by the report. | `DATA-05`, `AUD-ACC-003` |
| MED-04 | FV marginal “convergence” reflects the regularized solve residual rather than explicit final marginals, and authoritative construction discards those diagnostics. | `FV-MATH-002` |
| MED-05 | Automated tests do not exercise the authoritative FV moment/marginal construction. | `FV-MATH-004` |
| MED-06 | Ordinary progressive inference accepts any positive projection-iteration count and does not enforce final correction residual tolerances. | `PM-02` |
| MED-07 | The official unittest command silently omits 13 pytest-style functions, and pytest is not installed in the selected environment. | `PM-05`, `CFG-12` |
| MED-08 | Saved corrector/model integrity hashes are present but not comprehensively verified when training checkpoints are loaded. | `TRAIN-RESUME-03` |
| MED-09 | An identity-selected final checkpoint can retain selection metrics from a rejected candidate rather than metrics for the restored identity model. | `TRAIN-RESUME-04` |
| MED-10 | Smoke/full mode is stored but not included in resume authentication, allowing incompatible continuation intent. | `TRAIN-RESUME-05` |
| MED-11 | Ratio flooring can make two exact-zero errors appear as a gain rather than identity in audit comparisons. | `AUD-ACC-004` |
| MED-12 | The generic supported one-stage experiment path cannot form the required prior-band audit group and therefore cannot pass promotion. | `AUD-ACC-005` |
| MED-13 | Structural acceptance does not enforce every reported routing invariant or gate diagnostic. | `AUD-ACC-006` |
| MED-14 | Structural row/column failures raise before atomic report writing, leaving no complete failure report. | `AUD-ACC-007` |
| MED-15 | Automated tests do not exercise end-to-end audit and promotion decisions. | `AUD-ACC-008` |
| MED-16 | `features.source`, `features.target`, and `features.sample_per_pair` are accepted and authenticated but ignored by execution. | `CFG-09` |
| MED-17 | Most numeric configuration fields lack meaningful domain/finiteness validation. | `CFG-10` |
| MED-18 | Active PBS entry points are ignored by Git and are outside implementation authentication, despite controlling the documented workflow. | `CFG-11` |

All are confirmed by direct code trace, independent specialist diagnostics, or
both. MED-02 contains a policy ambiguity: shared analytic/real anchors may be
intentional, but the current “source-keyed split” claim and helper cannot
represent that distinction.

## Low

Five distinct Low findings remain:

- float64 map areas are narrowed to float32 in FV feature construction;
- a zero local proposal gate is not an exact zero applied correction after
  global conservation projection;
- public pair dataclasses permit dtype combinations that later fail in the
  float32 progressive MLP;
- resuming an already-completed checkpoint does not repair a history CSV that
  lagged the last atomic checkpoint;
- output overrides can collide with configured history/report naming.

These are documented in the specialist reports and are not promotion blockers
by themselves.

## Positive results reproduced or cross-confirmed

- Official baseline: 12 discovered unittest cases pass.
- The active FV checkpoint hash and internal state hash are consistent with
  its active progressive reference.
- All eleven configured active graph files exist and have complete source and
  target coverage; specialist inspection found no duplicate edges.
- Independent dense projection and adjoint calculations agree at about
  `4e-15`.
- Tested active correction supports meet `1e-8` row and `1e-10` column
  tolerances with large margins.
- An actual r32 FV build met its row and marginal tolerances.
- Ordered stage execution, raw-source reuse, fixed FV reuse, and prefix
  chaining match the documented architecture.
- Forced global rejection is bit-exact in exercised paths.
- Rotation and positive/negative/tiny-scale affine checks passed on nonzero
  synthetic corrections within the documented numerical scale.
- Frozen prefix and nonselected parameters remained bit-identical across an
  optimizer step.
- Isolated interrupted/resumed CPU smoke training matched uninterrupted
  execution in both capability and router phases, apart from elapsed-time
  fields.
- Independent finite-value recomputation matched an existing two-pair audit
  decision and confirmed that exact finite threshold equality passes while
  values above fail.

## Deferred

Per the audit charter, this report does not yet confirm:

- the completed high-band candidate;
- GPU projection or CPU/GPU parity;
- full expensive r64/r128 execution;
- protected/external promotion;
- end-to-end resume across a real PBS wall-time termination;
- high-frequency quadrature accuracy against an independent higher-resolution
  oracle.
