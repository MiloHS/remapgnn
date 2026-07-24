# Audit metrics and acceptance audit

## Scope and work performed

This report audits commit
`f98a0e8ed06678099a98b5a6dfc91ded5d580c97` for the active, non-archived
evaluation and acceptance workflow:

- `_next/remapgnn_next/evaluation.py`;
- `_next/scripts/audit.py`;
- the audit-facing parts of `next`, `jobs_next_audit.pbs`, configuration,
  checkpoint loading, panel construction, and provenance;
- active development audit artifacts under `_next/reports/`;
- the checked-in equivalence record
  `_next/reports/equivalence_completed_v24f.json`;
- audit and acceptance test discovery.

No implementation, test, configuration, data, checkpoint, job, Git state, or
running job was modified. CPU diagnostics were executed read-only. The only
file written is this report.

Work actually performed:

1. Traced FV, every stage prefix, final-stage, and np2 prediction/error paths.
2. Derived every implemented promotion equation and classified every reported
   value as a hard gate, score input, or diagnostic.
3. Independently recomputed the two-pair recorded development decision from
   its detail CSV using only Python arithmetic and CSV parsing.
4. Tested values immediately below, exactly at, and immediately above all four
   configured promotion thresholds and all five fixed structural tolerances.
5. Tested incomplete panels, exact rejection, NaN metrics, and zero-error
   ratio semantics.
6. Independently checked `area_relative_l2` against a direct NumPy formula.
7. Exercised the protected-role guard without `--allow-protected`.
8. Inspected active selection real fields for non-finite values.
9. Ran the documented unittest command and inspected test collection.
10. Reviewed Wave 1 findings for their direct effect on promotion validity.

Candidate acceptance, GPU execution/parity, protected scientific consumption,
and expensive r128 execution were deferred as required while training remains
active.

## Implemented metrics and decisions

For field `i`, the reported error is

```text
e_i(y) = sqrt(
    sum_j area_target_j * (y_ij - truth_ij)^2
    /
    max(sum_j area_target_j * truth_ij^2, 1e-30)
).
```

`audit_pair()` evaluates FV, np2, and every stage output. For the final stage,
it records ratios to the immediately preceding prefix, FV, and np2. The
promotion calculation is performed independently for every requested pair.

For target set `T`, safety set `S`, and prior-band safety set `P`, the
implemented hard gates are:

```text
mean_{i in T}(e_model_i / max(e_prefix_i, 1e-20))
    <= 1 - minimum_target_gain

max_{i in S}(e_model_i / max(e_prefix_i, 1e-20))
    <= 1 + maximum_safety_regression

max_{i in S}(e_model_i / max(e_fv_i, 1e-20))
    <= 1 + maximum_fv_regression

max_{i in P}(e_model_i / max(e_prefix_i, 1e-20))
    <= 1 + maximum_prior_band_regression.
```

The active limits are respectively `0.97`, `1.02`, `1.02`, and `1.01`.
Every pair must have nonempty `T`, `S`, and `P`. Failure on any requested pair
fails the full audit.

The structural hard gates are:

| Property | Passing condition |
|---|---:|
| forced rejection | exact tensor equality with final stage's prefix |
| correction row residual | `<= 1e-8` |
| correction area-column residual | `<= 1e-10` |
| constant reproduction | `<= 2e-6` absolute |
| positive/negative/tiny affine output error | each `<= 1e-4` relative-L-infinity |
| rotation output error | `<= 1e-5` relative-L-infinity |

All comparisons are inclusive at equality. The independent next-representable
float tests returned `[pass, pass, fail]` below/at/above every threshold.
Turning exact rejection from true to false independently changed pass to fail.

The following are diagnostics only and do not affect promotion:

- np2 errors and model/np2 ratios;
- target worst ratio and target regression count;
- per-stage field/local gate means;
- `stage_gate_affine_max_abs`;
- timing;
- family summaries.

The training-time selection score is a different, penalty-based rule in
`training.py:145-153`; the final audit above uses hard per-pair constraints.
A candidate can therefore be selected by training and still correctly fail
the audit.

## Independent evidence

### Recorded development decision

The active detail CSV
`_next/reports/progressive_next_audit_self_audit_detail.csv` contains:

| Pair | Total | Target | Safety | Prior-band safety |
|---|---:|---:|---:|---:|
| CS-r64 to ICOD-r64 | 136 | 64 | 72 | 18 |
| ICO-r32 to CS-r32 | 130 | 64 | 66 | 18 |

Independent CSV parsing reproduced every value in the JSON
`promotion.pair_metrics` exactly. Both pairs have a final/prefix target mean
ratio of `1.0`; therefore both fail the required `<= 0.97` target gate. The
recorded `passed=false` decision and failure list are correct for those
recorded rows.

The target panels weight every field equally: 40 harmonic modes and 24
mixtures per pair. The prior-band guard contains 12 harmonic modes and 6
mixtures. Pair results are not averaged together: each pair must pass.

### Error formula

On an independent three-cell, two-field float64 example, direct NumPy
calculation produced errors `[0.09655068, 0.28713930]`; the implementation
matched with maximum absolute difference `0.0`.

### Protected role guard and exit behavior

A direct CPU invocation requesting `CS-r32_to_HP-n32` without
`--allow-protected` stopped before pair construction with:

```text
ValueError: protected pairs require --allow-protected:
['CS-r32_to_HP-n32']
```

This is fail-closed with process status 1. Ordinary promotion pass and fail
return status 0 and 2 respectively (`audit.py:62-64`). The wrapper's
`audit-protected` command supplies both a production checkpoint requirement
and `--allow-protected`.

### Current input finiteness

All five configured variables in both real-field files consumed by the
CS-r64-to-ICOD-r64 selection pair are finite. The ICO source real-field file
is absent, so real fields are intentionally omitted for that pair. This
materially reduces the likelihood of the NaN finding below affecting the
already recorded development audit, but it does not protect against
non-finite model output or future inputs.

## Findings

### AUD-ACC-001 — Critical — NaN values fail open and can be promoted

**Requirement or claim.** Invalid numerical output must never satisfy
scientific promotion gates.

**Expected behavior.** Predictions, truths, errors, ratios, residuals, gates,
and aggregate metrics must be finite. Any non-finite scientific value should
be an explicit hard failure.

**Observed behavior.**

- `evaluation.py:18-21` calculates errors without a finiteness check.
- `evaluation.py:147-150` calculates ratios without a finiteness check.
- `evaluation.py:193-204` rejects only when `value > threshold`.
- `structural_checks()` likewise uses only `> threshold` at lines 78-93.

Every ordered comparison with NaN is false. An independently constructed,
complete target/safety/prior detail set with all ratios NaN, together with
NaN row, column, constant, affine, and rotation values, returned:

```text
{"passed": true, "failures": [], "pair_metrics": {... NaN ...}}
```

Python's JSON writer also permits non-standard `NaN` tokens by default, so
such a passed report can be serialized.

**Reproduction.** Construct one target row and one safety/prior row with
`model_over_prefix=float("nan")` and `model_over_fv=float("nan")`; construct
a structurally complete record with every floating check equal to NaN; call
`promotion_report(..., pairs=["p"])`.

**Independent evidence.** IEEE/Python comparison semantics alone establish
the fail-open behavior; no production helper was used to form an expected
answer. Active selection real data was separately checked and is finite.

**Impact.** A numerically broken candidate can be marked passed and cause
exit status 0, invalidating scientific promotion.

**Recommended correction.** Fail immediately unless every scientific tensor
and aggregate is finite. Check inputs, predictions, truth, all operator
outputs, errors, ratios, residuals, gates, and final metrics. Permit NaN only
in the explicitly unknown `frequency` metadata channel. Serialize reports
with `allow_nan=False` and test NaN/positive-infinity/negative-infinity at
every acceptance boundary.

**Confidence.** High.

### AUD-ACC-002 — High — the advertised candidate command targets the wrong active experiment

**Requirement or claim.** `next:41-43`,
`docs/ACTIVE_WORKFLOW.md:30-31`, and `jobs_next_audit.pbs:10` say
`./next audit-candidate` audits the latest completed trained candidate.
The documented current experiment is
`_next/configs/high_band_candidate_01.json`.

**Expected behavior.** The no-argument candidate audit should resolve the
current experiment's output
`_next/checkpoints/high_band_candidate_01.pt`.

**Observed behavior.** `next:16` hard-codes
`_next/configs/progressive.json`; `next:76-79` invokes the audit with that
configuration. It therefore resolves
`_next/checkpoints/progressive_next.pt`. `jobs_next_audit.pbs` invokes the
same no-argument command. At inspection time, both checkpoint names existed
as distinct training artifacts.

Passing a later explicit
`--config _next/configs/high_band_candidate_01.json` works because argparse
uses the final occurrence, but the documented no-argument command does not.

**Reproduction.**

```text
inspect next:16,76-79
inspect paths.output_checkpoint in both active JSON configurations
```

**Independent evidence.** Path resolution follows directly from
`config.paths.checkpoint_path` at `audit.py:32-35`.

**Impact.** An operator can audit and make a promotion decision about a
different candidate than the one just trained.

**Recommended correction.** Make the active experiment explicit in one
authoritative place, require/configure it in PBS, print the resolved config
and checkpoint before expensive work, and refuse ambiguous multiple
candidate artifacts. Update `status` and documentation to use the same
resolution rule.

**Confidence.** High.

### AUD-ACC-003 — Medium — audit reports do not authenticate all inputs or outputs

**Requirement or claim.** A report should retain enough provenance to
reproduce its scientific comparisons and decision.

**Expected behavior.** Record the exact implementation/commit, device, FV,
edge/map/np2/real-field inputs, panel manifest, and hashes of detail and
summary artifacts.

**Observed behavior.** `evaluation.py:240-246` records checkpoint and config
hashes plus edge and ordinary map hashes. It omits:

- consumed real-field paths, hashes, variables, and omission reasons
  (Wave 1 DATA-02);
- np2 map hashes even though np2 values are reported (Wave 1 DATA-05);
- audit implementation/commit and device;
- hashes of detail and summary CSVs.

The three output files are replaced atomically one at a time, but not committed
as one authenticated set. A same-tag rerun or interruption can leave a mixed
set that the JSON cannot detect.

**Reproduction.** Compare the files read by `build_panel()` and
`load_map_operator()` at `evaluation.py:225-236` with the report mapping
constructed at lines 240-245.

**Independent evidence.** The two active selection np2 files have concrete,
distinct SHA-256 values, but neither appears in the active JSON report.

**Impact.** Promotion fields in the JSON can be recomputed from the JSON
itself, but the complete per-field and np2 evidence cannot be tied to the
exact scientific inputs and CSV bytes that produced it.

**Recommended correction.** Write outputs into a run-unique directory, hash
detail and summary after writing them, and write an authenticated manifest
last. Include all real/np2 inputs, inclusion manifest, implementation commit
or module hashes, device/dtype, exact command, and FV hash.

**Confidence.** High.

### AUD-ACC-004 — Medium — the ratio floor can turn identity into apparent gain

**Requirement or claim.** An output identical to its prefix has zero gain and
must not pass a positive target-gain requirement.

**Expected behavior.** Equal model and prefix errors should yield ratio 1,
including when both are zero or extremely small.

**Observed behavior.** `audit_pair()` uses
`model_error / max(prefix_error, 1e-20)` (`evaluation.py:147`). If both errors
are zero the ratio is 0, and if both are `1e-25` the ratio is `1e-5`, despite
the model being identical to the prefix. Enough such target fields can make
the mean satisfy the 3% gain gate.

**Reproduction.** The arithmetic is direct:
`0/max(0,1e-20)=0` and `1e-25/max(1e-25,1e-20)=1e-5`.

**Independent evidence.** `area_relative_l2` correctly returns zero when both
prediction and truth are equal; the semantic error is introduced only by the
later ratio floor.

**Impact.** Exact or near-exact target fields can create false improvement.
Active recorded high-band target errors are around `0.1-0.25`, so this did
not change the recorded identity rejection.

**Recommended correction.** Define explicit paired near-zero semantics:
equal/indistinguishable errors should have ratio 1; a nonzero model error
against a zero prefix should fail. Add exact-zero and values on both sides of
the numerical floor.

**Confidence.** High.

### AUD-ACC-005 — Medium — a supported one-stage experiment cannot pass the audit

**Requirement or claim.** Configuration supports one train stage with
`prefix_through=null`; its prefix is FV.

**Expected behavior.** A valid first-stage experiment should have an
intentional prior/FV safety policy and be auditable.

**Observed behavior.** For a one-stage model, `audit_pair()` chooses the
current stage itself as `previous` (`evaluation.py:132-136`) and marks prior
fields using that stage's target band (`lines 143-144`). Proper safety fields
are outside the target band, so the prior subset is empty. `promotion_report`
then always emits `incomplete target/safety/prior panel` (`lines 181-184`).

Training evaluation behaves differently: if its prior mask is empty it falls
back to all safety fields (`training.py:178-184`).

**Reproduction.** A complete target+safety detail set without a prior row was
independently passed to `promotion_report`; it failed solely as incomplete.

**Impact.** The generic clean architecture's first-stage path can train but
cannot be promoted by the active auditor.

**Recommended correction.** Define first-stage prior protection explicitly
(probably all safety versus FV), use the same rule in training and audit, and
add one-stage and three-stage decision tests.

**Confidence.** High.

### AUD-ACC-006 — Medium — structural acceptance does not cover every reported routing invariant

**Requirement or claim.** Exact rejection and affine/sign/offset-invariant
routing are important properties of every ordered stage.

**Expected behavior.** Acceptance should force-close each stage in turn and
compare it with that stage's prefix; any measured gate-affine violation should
have an explicit tolerance or be documented as diagnostic only.

**Observed behavior.**

- `structural_checks()` force-closes only the final stage
  (`evaluation.py:94-97`).
- It calculates `stage_gate_affine_max_abs` across stages
  (`lines 88-89,108`) but `promotion_report()` never checks it.
- Final output affine/rotation behavior can pass for an identity correction
  even if an internal gate invariant is broken.

Wave 1 independently established correct rejection and invariance on
representative current/synthetic cases, so this is an acceptance coverage gap,
not evidence of a current wrong output.

**Impact.** A regression in an earlier stage's rejection or in gate invariance
can escape the formal promotion decision.

**Recommended correction.** Exercise every stage's forced/hard rejection
against its immediate prefix, add explicit field/local gate invariance
tolerances, and test below/equal/above router thresholds.

**Confidence.** High.

### AUD-ACC-007 — Medium — row/column failures produce no audit report

**Requirement or claim.** Reports and exit status should agree and preserve
the evidence for a failed decision.

**Expected behavior.** A constraint failure should return nonzero and write a
failure report containing the observed residual and authenticated inputs.

**Observed behavior.** `structural_checks()` raises immediately when row or
column tolerance is exceeded (`evaluation.py:78-81`). Report files are not
written until after all pairs and structural checks finish
(`evaluation.py:237-246`). Thus these hard failures exit 1 with a traceback
and no current run report, while ordinary gate failures write a report and
exit 2.

**Impact.** The behavior is safely nonzero, but failed numerical evidence and
provenance are lost; stale same-tag reports can remain and be mistaken for the
failed run.

**Recommended correction.** Convert expected scientific failures into
structured failure records, always write a run-unique final manifest, and
reserve unhandled status 1 for unexpected software faults.

**Confidence.** High.

### AUD-ACC-008 — Medium — automated tests do not exercise audit decisions

**Requirement or claim.** Acceptance boundaries, protected roles, report
provenance, and exit status should be regression-tested.

**Expected behavior.** The documented test command should collect all tests
and directly cover evaluation/acceptance behavior.

**Observed behavior.** The documented unittest command ran 12 methods, all
from `test_workflow_unittest.py`. It collected none of the 13 top-level
pytest-style functions in the other five files (corroborating Wave 1 PM-05).
No discovered or undiscovered test references `promotion_report`,
`audit_pair`, `structural_checks`, the protected guard, report hashes, np2
provenance, or audit exit codes.

**Impact.** The Critical NaN path, wrong zero-error ratio, threshold boundary
semantics, and command-routing issue have no automated regression protection.

**Recommended correction.** Standardize on an installed collector or convert
all tests to unittest; assert collected test names/count; add independent
decision tables for every threshold and non-finite value; mock the CLI only at
expensive pair-building boundaries and verify resolved config/checkpoint,
protected authorization, report content, and process status.

**Confidence.** High.

## Wave 1 promotion challenges

- **DATA-01 (target upper-bound rounding):** This materially affects what is
  labelled target during training and can affect promotion when the extra
  degree is selected. The existing recorded selection panel happened to top
  out at normalized frequencies `1.46875` and `1.48912`, so its fixed audit
  sample did not contain the above-1.5 degree. Candidate training semantics
  remain affected.
- **DATA-02 (real inputs unauthenticated):** Directly weakens safety evidence,
  because real fields are safety rows and the audit uses the worst safety
  ratio. It is incorporated in AUD-ACC-003.
- **DATA-03 (partly honored frequency divisor):** Active value 6 is internally
  consistent. Any change can corrupt the prior-band mask and therefore the
  stricter 1% prior gate.
- **DATA-04 (shared/non-disjoint full-panel identities):** Harmonic target
  modes are split, while shared analytic/real safety anchors mainly affect
  interpretation of generalization rather than arithmetic. The policy must be
  explicit before treating audit performance as wholly unseen data.
- **DATA-05 (np2 hash omitted):** Does not change current promotion because
  np2 is diagnostic only, but invalidates reproducibility of np2 claims.
- **FV-MATH-001 (equivalence not bound to current production payload):**
  Directly weakens the foundation of `--require-production`. The active
  checkpoint hash differs from the hash named by its equivalence record, so
  recorded CPU/GPU/mesh evidence cannot authenticate all current
  behavior-affecting bytes.
- **FV-MATH-002 and PM-02 (finite projection acceptance):** Training and full
  audit check observed correction residuals on active calls, but FV build and
  ordinary inference can accept unconverged results. AUD-ACC-001 makes
  non-finite residual handling especially urgent.
- **PM-01 (closed straight-through task gradient):** This affects how the
  candidate router is learned. A later empirical audit can reject the
  resulting candidate, but cannot demonstrate that the implemented training
  rule matched its claimed STE behavior.
- **PM-05 (test discovery):** Confirmed independently and expanded in
  AUD-ACC-008.

## Material non-findings

- FV, np2, and every stage output are present in per-field details.
- The final/prefix selection is correct for the active two-stage model.
- Target, safety, prior, and structural gates are hard per-pair failures;
  they are not merely score penalties.
- The implemented threshold direction and inclusive equality behavior are
  consistent for all finite values tested.
- Missing target, safety, or prior sets fail closed.
- The protected/external exact pair names cannot be consumed by default or by
  `--pairs` without explicit `--allow-protected`.
- Promotion failure propagates as status 2; promotion success propagates as
  status 0 through the PBS script's `set -e`.
- The independent active detail-CSV recomputation matches the saved JSON
  decision exactly.
- Active selection real values inspected are finite.
- The checked-in equivalence record reports exact FV edge ordering and FV
  mass/weight differences within the intended `1e-10` bound for r32, r64,
  HeALPix, and r128. It also records small CPU/GPU model differences and
  constraint residuals within audit tolerances. This is useful recorded
  evidence, subject to FV-MATH-001's authentication limitation.

## Deferred checks

- Acceptance of the currently training high-band checkpoint.
- GPU audit execution and direct CPU/GPU same-input parity.
- Protected and external-resolution scientific audits.
- Full r128 panels and expensive mesh builds.
- Regeneration of the legacy-to-clean equivalence record; archived
  implementation is outside scope.

## Remaining uncertainty

The initial audit establishes the finite-value equations and decision
boundaries, but it does not establish current candidate quality. The largest
unresolved acceptance risks are the NaN fail-open, wrong default candidate
resolution, and lack of authenticated linkage between active production bytes
and recorded equivalence. Once those are corrected, the deferred final audit
must run the completed candidate on development first, then separately
authorized protected/external pairs, with a direct CPU/GPU comparison and a
run-unique authenticated output manifest.

## Representative commands

```bash
git rev-parse HEAD
rg -n "audit|promotion|protected|np2|parity|equivalence" _next next docs
PYTHONPATH=_next .../python -B -m unittest discover \
  -s _next/tests -p 'test_*.py' -v
PYTHONPATH=_next .../python -B _next/scripts/audit.py \
  --config _next/configs/progressive.json --device cpu \
  --checkpoint _next/checkpoints/progressive.pt \
  --pairs CS-r32_to_HP-n32
sha256sum _next/checkpoints/progressive.pt \
  _next/checkpoints/fv_relax1.pt \
  _next/reports/equivalence_completed_v24f.json
```
