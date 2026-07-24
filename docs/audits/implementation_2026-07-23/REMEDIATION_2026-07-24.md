# Implementation-audit remediation

## Status

The code remediation was implemented on 2026-07-24 without editing the
original specialist reports. The current production checkpoint was not
overwritten and no PBS job was submitted.

The CPU test command now uses pytest, collects both historical test styles,
and passes:

```text
39 passed
```

## Implemented

- Schema 4 rejects unknown/ignored fields and validates scientific domains.
- Target bands use an inclusive floored upper degree and one frequency
  convention for target, safety, and metadata.
- Analytic/real fields are explicit shared safety anchors; generated mixture
  identities include their components.
- FV and correction projections fail closed on non-finite values and actual
  residual tolerances; FV areas remain float64.
- Closed straight-through routing has an exact hard forward value and a soft
  task-gradient path.
- Router teaching is based on measured forced-open field/local benefit.
- Run manifests are frozen before data loading and include implementation,
  configuration, source/FV checkpoints, edge/map files, and real-field
  availability and hashes.
- Training checkpoints hash all resumable/best/identity/optimizer states,
  authenticate smoke/full mode, restore lagging history, and reset metrics
  when identity wins.
- Audit ratios define exact-zero behavior and all decisions fail closed on
  NaN/Inf.
- One-stage audits treat prior-band protection as not applicable.
- Audit failures still produce atomic reports; np2, real inputs, and output
  artifacts are hashed.
- Candidate audit requires explicit config/checkpoint binding, and production
  mode rejects training checkpoints.
- PBS scripts are tracked, arguments use `qsub -v`, and the submission helper
  creates a visible train/resume/audit dependency chain with dry-run support.

## Deliberately pending

`_next/configs/production.json` records `approved=false`. Production remains
blocked until all of the following are performed against the exact hardened
checkpoint:

1. regenerate external equivalence evidence;
2. run CPU/GPU parity;
3. run r32, r64, HEALPix, and r128 equivalence;
4. create and validate the detached production manifest;
5. atomically activate the production pointer;
6. run development, protected, and external-resolution audits.

The rejected schema-3 high-band candidate is historical evidence and is not
resumable under corrected schema-4 training semantics. A new candidate must be
trained after production cutover.
