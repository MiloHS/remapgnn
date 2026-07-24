# Audit scope and requirements ledger

## Fixed target

- Git commit: `f98a0e8ed06678099a98b5a6dfc91ded5d580c97`
- Commit subject: `Replace old linear model with a clean progressive corrector workflow`
- Repository state before the audit: clean
- Audit charter added after fixing the target: root `AGENTS.md`
- Active implementation: `_next/`, root `next`, active root PBS entry points,
  and active documentation
- Active external inputs: non-archived edge datasets, maps, real fields, and
  clean checkpoints selected by the two active JSON configurations
- Excluded as implementation: `_archive/` and other archived historical code

The audit does not modify implementation code, tests, configurations,
checkpoints, datasets, maps, job scripts, Git state, or running jobs.

## Recorded input identities

These hashes were recorded at audit start:

| Input | SHA-256 |
|---|---|
| `_next/configs/progressive.json` | `10c06d797c03676b6d032ff08d9fd94399fa86a7dffd6311ffc37a543c930684` |
| `_next/configs/high_band_candidate_01.json` | `44fe582cf7302f04f09a4e33f7e587f70dc28bcf0324d339b1068e5bb3a0c034` |
| `_next/checkpoints/progressive.pt` | `4a64d9c43f6f39059d390c3d2bca35f08b7e36309e6c72dcdc520e767d0d7c15` |
| `_next/checkpoints/fv_relax1.pt` | `18df156b418835bbb6ece1bb3eb246156e7d58e48481778c68088bfe4b60efdc` |
| `_next/reports/equivalence_completed_v24f.json` | `37715f4fe684472e5d80f1241782423442ad190b585bd0cccc314b90bf77a5d5` |

The actively training candidate is deliberately not frozen here. Candidate
checkpoint acceptance and GPU checks are deferred until training ends.

The configured external data roots contain 158 files totaling approximately
2.23 GB. Specialist reports determine which of these files are reachable from
the active pair roles rather than treating every file in those directories as
an active input.

## Baseline verification

At audit start:

```text
./next test
Ran 12 tests in 1.302s
OK
```

The official command uses `unittest` discovery. Five additional files contain
pytest-style function tests. The configured Python environment does not have
pytest installed. Whether the 12 unittest cases fully duplicate those tests is
an open audit item, not yet a confirmed finding.

## Requirement sources

Requirements are drawn from:

- `README.md`
- `_next/README.md`
- `docs/ACTIVE_WORKFLOW.md`
- `docs/PROJECT_HISTORY.md`
- the validated active configurations
- public behavior of the root `next` entry point
- equations and explicit assertions in the clean implementation
- `_next/reports/equivalence_completed_v24f.json`, as recorded evidence rather
  than executable implementation

When documentation, configuration, and code disagree, the audit records the
disagreement rather than silently choosing one as authoritative.

## Scientific and architectural requirements

| ID | Requirement |
|---|---|
| SCI-01 | Applying the FV base preserves a constant field to the configured row tolerance. |
| SCI-02 | The FV base preserves the area-weighted global total to the configured column tolerance. |
| SCI-03 | Every correction stage has zero row sum and zero area-weighted column sum within configured tolerances. |
| SCI-04 | A correction is applied to the raw source field and added to the current ordered prefix. |
| SCI-05 | Global stage rejection returns the prefix exactly, not approximately. |
| SCI-06 | The model is invariant/equivariant as documented under rotation, offsets, signs, affine transforms, and negative or tiny scales. |
| SCI-07 | Correction features and routing features have fixed, validated layouts matching their networks and checkpoints. |
| SCI-08 | The frozen FV base and accepted earlier stages cannot change while training a later stage. |
| SCI-09 | Numerical projection either meets its stated residual guarantees or fails explicitly; finite iteration is not silently accepted as convergence. |
| SCI-10 | Supported r32, r64, HeALPix, and r128 families retain the declared conservation, consistency, and equivalence tolerances. |

## Data and panel requirements

| ID | Requirement |
|---|---|
| DATA-01 | Train, selection, protected, and external-resolution roles are disjoint and enforced. |
| DATA-02 | Source-keyed field splits prevent source realization leakage across splits and pairs. |
| DATA-03 | Target fields lie in the selected stage band and safety fields lie outside it after realizable-degree rounding. |
| DATA-04 | Labels, roles, frequencies, families, source keys, and target masks remain mutually consistent. |
| DATA-05 | Harmonics, balanced mixtures, smooth analytic fields, and available configured real fields are generated or loaded deterministically and normalized as declared. |
| DATA-06 | Both coarse-to-fine and fine-to-coarse regimes receive the intended aggregate training weight. |
| DATA-07 | Missing or incompatible configured data fails clearly or follows a documented optional-data rule. |
| DATA-08 | Every edge dataset, map, np2 operator, and real field used by a run is traceable through configuration and provenance. |

## Training and checkpoint requirements

| ID | Requirement |
|---|---|
| TRAIN-01 | Capability training opens only the selected stage gate and optimizes only its corrector. |
| TRAIN-02 | Capability admission compares the best capability score with the identity score using the configured strict minimum improvement. |
| TRAIN-03 | Router training starts from the admitted best corrector and freezes that corrector exactly. |
| TRAIN-04 | Router teaching uses the documented target/safety labels and configured loss terms. |
| TRAIN-05 | Final selection compares the best deployable routed model with the original identity floor using the configured strict gain. |
| TRAIN-06 | Rejection restores and stores the exact identity model state, not the last unsuccessful state. |
| TRAIN-07 | Checkpoints are written atomically after every completed epoch and contain sufficient state for an authenticated resume. |
| TRAIN-08 | Capability and router resumes reproduce uninterrupted execution, including best-state and final-decision behavior. |
| TRAIN-09 | A change to any authenticated implementation, configuration, checkpoint, edge-data, or map input is detected on resume. |
| TRAIN-10 | Smoke execution traverses capability, router, checkpoint, and resume control flow without being treated as scientific acceptance evidence. |

## Audit and promotion requirements

| ID | Requirement |
|---|---|
| AUDIT-01 | Audit metrics compare FV, every prefix, final output, and np2 where available using independently verifiable definitions. |
| AUDIT-02 | Development promotion requires the configured target gain on every requested development pair. |
| AUDIT-03 | Safety regression versus both prefix and FV is bounded on every requested pair. |
| AUDIT-04 | Prior-band regression is bounded independently from general safety regression. |
| AUDIT-05 | Row and column residual limits are hard promotion failures. |
| AUDIT-06 | Protected and external-resolution pairs require explicit authorization and cannot be consumed accidentally by the default development audit. |
| AUDIT-07 | Boundary behavior at every threshold is intentional and tested below, exactly at, and above the threshold. |
| AUDIT-08 | Reports and exit status agree on promotion outcome and retain enough provenance to reproduce it. |
| AUDIT-09 | CPU/GPU paths agree within declared tolerances; larger GPU tolerances are explicit rather than accidental. |
| AUDIT-10 | The active checkpoint being audited is cryptographically tied to the equivalence or acceptance evidence claimed for it. |

## Configuration and operational requirements

| ID | Requirement |
|---|---|
| CFG-01 | Every behavior-affecting JSON field is typed, validated, used consistently, and covered by a perturbation check where feasible. |
| CFG-02 | Metadata-only fields are identified; unexplained ignored fields are rejected or reported. |
| CFG-03 | Frozen prefix definitions match the production source checkpoint exactly. |
| CFG-04 | Fresh and checkpoint stage initialization enforce the documented structural compatibility rules. |
| CFG-05 | Runtime modules do not import archived/versioned modules or scripts. |
| CFG-06 | Root commands and direct Python entry points select the same configuration, checkpoint, and role behavior when given equivalent arguments. |
| CFG-07 | Relative paths behave from the documented repository-root working directory and fail clearly otherwise. |
| CFG-08 | PBS argument forwarding is explicit and documented; submitted settings can be recovered from job metadata and resulting checkpoints. |
| CFG-09 | Requested wall time and memory are sufficient, or continuation/resume is an explicitly supported and tested operational path. |
| CFG-10 | Missing, damaged, incomplete, or schema-incompatible checkpoints fail safely and clearly. |

## Deferred work

The following work waits for the active candidate training job to finish:

- candidate checkpoint acceptance and identity-floor verification;
- GPU projection and CPU/GPU parity;
- full expensive r64/r128 execution;
- end-to-end candidate development audit;
- protected/external audit, and only after development promotion succeeds;
- timing and peak-memory measurements of the completed path.
