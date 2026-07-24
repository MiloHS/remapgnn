# Interim implementation audit

## Bottom line

The clean `_next` implementation has a sound core structure, and the tested
FV/progressive numerical paths showed strong conservation, consistency,
invariance, exact-fallback, and freezing behavior.

It is not yet safe to say that the complete workflow is implemented correctly.
The initial audit found:

- 1 Critical acceptance defect;
- 11 distinct High findings;
- 18 distinct Medium findings;
- 5 Low findings.

The most urgent result is that the promotion audit fails open on `NaN`. A
candidate with non-finite metrics and structural values can currently receive
`passed=True`. Therefore no candidate should be promoted with the current
audit implementation, even if its ordinary printed metrics look reasonable.

The currently training high-band candidate should be treated as an
experimental checkpoint. Training may continue without damaging the approved
prefix, but promotion must wait for remediation and re-audit. Some fixes,
especially the target-band boundary and router-gradient behavior, affect
training semantics and may justify retraining rather than merely re-auditing
this candidate.

## What was audited

- Git commit `f98a0e8ed06678099a98b5a6dfc91ded5d580c97`
- `_next/remapgnn_next`
- `_next/scripts`
- both active JSON configurations
- root `next`
- active root PBS entry points
- all configured non-archived edge/map/np2 paths
- real-field inputs and their loading/use
- active clean FV and progressive checkpoints
- recorded equivalence evidence
- existing tests and their actual discovery behavior

Archived implementation was excluded. It was not treated as the source of
truth.

Candidate-checkpoint acceptance, GPU parity, protected pairs, and expensive
full r128 execution remain deferred until training finishes and the Critical
audit defect is fixed.

## Highest-priority findings

### 1. Promotion can pass non-finite results

Promotion and structural checks use comparisons such as
`value > threshold`. For `NaN`, those comparisons are false. An independent
complete synthetic report containing `NaN` values passed with no failures.

Required response: make every audit stage fail closed on non-finite
predictions, errors, ratios, residuals, gates, aggregate metrics, and report
values.

### 2. The default candidate audit targets the wrong experiment

`./next audit-candidate` uses `progressive.json`, resolving
`progressive_next.pt`. The active experiment is configured by
`high_band_candidate_01.json` and writes `high_band_candidate_01.pt`.

Required response: make the experiment/config explicit and unambiguous in the
root command and PBS job, and print the resolved config/checkpoint before
expensive work.

### 3. Target panels do not exactly implement the configured band

The upper degree uses rounding. Several active non-integer effective
resolutions therefore include a target just above `1.5`.

Required response: use the mathematical inclusive upper bound and add
mesh-specific boundary tests. Decide whether the current experimental
candidate needs retraining under the corrected panel.

### 4. Production and candidate authentication is incomplete

The active production checkpoint is accepted based on embedded flags and
stage hashes, not a manifest binding the exact runtime payload to external
equivalence evidence. A temporary checkpoint with deliberately changed
normalization still passed production validation.

Training resume omits real fields and recomputes its manifest at checkpoint
time, allowing mixed-generation state if inputs change during a run. Candidate
audit also does not bind its model state and saved training config/provenance
to the separate audit config.

Required response: use a detached, immutable manifest covering the complete
scientific payload and every consumed input; freeze it at run start and verify
it on every save/load/resume/audit.

### 5. Closed straight-through gates lose task gradients

Exact global fallback uses a boolean mask after the nominal
straight-through gate. A closed field therefore receives no task-loss
gradient through the router. Explicit role-label BCE still produces a
gradient, so training is not completely stuck, but the implemented behavior is
narrower than the documented straight-through behavior.

Required response: either preserve exact forward fallback with an intentional
straight-through backward path or explicitly define the current behavior and
test it as a deliberate training rule.

### 6. Operational commands are not reliable enough

- documented PBS `EXTRA` forwarding can silently run the default config;
- the active candidate needs more than 12 hours for capability alone;
- PBS scripts are ignored by Git even though they control the documented run;
- `--require-production` can be bypassed when a training checkpoint is
  supplied;
- unknown top-level JSON keys are silently discarded.

These are not cosmetic: the first issue already caused an unintended training
run.

## Important positive findings

- The active FV checkpoint itself matches its recorded FV hash and internal
  tensor-state hash.
- Active graph structures were present and internally coherent across all
  configured role pairs.
- Independent sparse projection and adjoint calculations agreed closely with
  the implementation.
- Tested active FV and correction paths met row and column tolerances with
  substantial margins.
- Ordered stages receive the intended raw source, FV result, and prefix.
- Exact forced rejection returns the previous prefix bit-for-bit.
- Tested rotation and affine behavior, including negative and tiny scales,
  remained within tight numerical errors on nonzero corrections.
- Frozen prefixes and nonselected parameters remained bit-identical across an
  optimizer step.
- Small CPU interrupted/resumed runs matched uninterrupted capability and
  router execution.
- Finite-value threshold decisions use the intended inclusive acceptance
  boundaries.
- The active clean runtime does not import archived implementation.

These results support keeping the clean architecture. The findings call for
targeted hardening and a few scientific corrections, not a return to the
archived codebase.

## Test-suite conclusion

`./next test` passes 12 unittest methods. However, it silently skips 13
pytest-style function tests in five other files, and pytest is not installed
in the selected environment. Important test names therefore exist without
being part of the advertised verification command.

The corrected workflow should use one installed test runner and verify the
collected test count/names. Additional required coverage includes:

- NaN/Inf promotion failure;
- target-band upper boundaries on every mesh family;
- production and candidate manifest tampering;
- real-field and np2 provenance;
- authoritative FV build/marginal checks;
- correction projection nonconvergence;
- capability/router resume and threshold boundaries;
- end-to-end audit/promotion decisions;
- exact config/checkpoint resolution for every root/PBS command.

## Recommended remediation order

Do not change the implementation while the active training checkpoint still
needs to resume: implementation hashing will correctly reject continuation
after code changes.

After the current job reaches a safe stopping point:

1. Freeze and hash the experimental candidate and history for analysis.
2. Fix audit finiteness checks and add fail-closed tests.
3. Fix command/config/checkpoint resolution and production/protected guards.
4. Fix target-band degree selection and frequency semantics.
5. Decide and document the intended closed-gate straight-through gradient.
6. Replace checkpoint/provenance flags with complete immutable manifests.
7. Make resume authentication cover real fields and freeze it at startup.
8. Correct PBS forwarding, tracking, wall time, and continuation behavior.
9. Unify test discovery and add the missing mathematical/audit tests.
10. Re-run CPU tests and independent synthetic checks.
11. Retrain any candidate affected by scientific/training changes.
12. Run development audit, then GPU parity/full mesh checks, and only then
    protected/external audits.

## Reports

- [Scope and requirements](00_scope_and_requirements.md)
- [Configuration ledger](01_configuration_ledger.md)
- [Data and panels](10_data_panels.md)
- [FV and mathematical core](20_fv_math.md)
- [Progressive model](30_progressive_model.md)
- [Training and resume](40_training_resume.md)
- [Configuration and interfaces](50_config_interfaces.md)
- [Audit and acceptance](60_audit_acceptance.md)
- [Reproduced and deduplicated findings](90_reproduced_findings.md)

## Status

This is an interim report because the active candidate, GPU parity, protected
pairs, and expensive full-resolution checks were intentionally deferred. The
static and CPU implementation audit is complete. No remediation has been
implemented.
