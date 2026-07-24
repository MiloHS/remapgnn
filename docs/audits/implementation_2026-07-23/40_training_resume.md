# Training, selection, checkpoint, and resume audit

## Scope and work performed

This report audits commit
`f98a0e8ed06678099a98b5a6dfc91ded5d580c97` for:

- capability and router loss equations;
- target/safety batching and transfer-regime weighting;
- forced-open capability behavior and strict phase freezing;
- capability admission, best-state restoration, router teaching, and final
  identity-floor selection;
- per-epoch checkpoint/history writes;
- capability- and router-phase resume;
- resume authentication and state hashes.

The implementation paths traced were:

- `_next/remapgnn_next/training.py`;
- the training controls in `_next/remapgnn_next/progressive.py`;
- `_next/remapgnn_next/{panels,fields,checkpoint,provenance,config}.py`;
- `_next/scripts/train.py`, root `next`, and `jobs_next_train.pbs`;
- both active JSON configurations and active workflow documentation;
- existing training/workflow tests;
- Wave 1 reports `10_data_panels.md` and `30_progressive_model.md`.

No implementation, configuration, test, checkpoint, dataset, map, job, or Git
state was modified. All independent diagnostics used temporary directories and
small synthetic CPU pairs. The only repository file written is this report.

## Implemented training mathematics

For field `f`, with target area `a_i`, current output `y`, truth `t`, frozen
prefix `p`, and FV output `v`, the code defines

```text
E_y(f) = sum_i a_i (y_fi - t_fi)^2
         / max(sum_i a_i t_fi^2, 1e-20)
```

and analogously `E_p` and `E_v`. `E_p` and `E_v` are detached. For target set
`T` and safety set `S`, the phase loss is

```text
L_target = mean_{f in T} E_y(f)

r_p(f) = sqrt(E_y(f) / max(E_p(f), 1e-20))
r_v(f) = sqrt(E_y(f) / max(E_v(f), 1e-20))

L_guard = CVaR_q(
    {relu(r_p - (1 + tau_p))^2} union
    {relu(r_v - (1 + tau_v))^2}
)

L_local = CVaR_q(
    {relu(E_y - E_p)} union {relu(E_y - E_v)}
)

L_delta = mean_e D_e^2
```

Here `CVaR_q` is the mean of the largest
`max(1, ceil(q * number_of_values))` values. Despite its name, `L_local` is a
fieldwise normalized-MSE excess term; it is not a spatially local error.

During capability, router terms are zero. During router training,

```text
L_teacher =
    BCE(field_probability, is_target)
  + BCE(local_probability, is_target broadcast to every target cell)

L_safety_gate =
    CVaR_q(field_probability on S)
  + CVaR_q(mean_cells(local_probability) on S)

L = L_target
  + guard_weight * L_guard
  + local_weight * L_local
  + gate_teacher_weight * L_teacher
  + safety_gate_weight * L_safety_gate
  + correction_weight * L_delta.
```

Thus router teaching is role-based: every target field is taught open and
every safety field closed, regardless of whether the fixed corrector helped a
particular field. Task-loss gradients supplement those labels only when the
global straight-through forward gate is nonzero; see the cross-confirmation
of `PM-01` below.

For pair `j`, each optimizer step accumulates

```text
L_step = sum_j w_j L_j.
```

With more than one pair, each coarse-to-fine pair receives
`0.5 / number_of_coarse_to_fine_pairs` and each fine-to-coarse pair receives
`0.5 / number_of_fine_to_coarse_pairs`. Therefore each transfer regime has
aggregate weight `0.5` per step. Every pair is processed at every step. The
number of steps is the maximum over every pair of

```text
max(ceil(number_of_targets / target_batch),
    ceil(number_of_safety / safety_batch)).
```

Shorter pair panels are cyclically oversampled to that common step count.
Target and safety indices are independently shuffled from deterministic
epoch/pair/phase seeds.

The optimizer is AdamW over only the selected corrector during capability and
only the selected routers during router training. The selected gradient norm
is clipped to the configured value before each optimizer step.

## Selection and threshold rules

For every selection pair `j`, let:

```text
t_j = mean target error ratio versus prefix
s_j = worst safety error ratio versus prefix
f_j = worst safety error ratio versus FV
p_j = worst previous-stage-band error ratio versus prefix.
```

The selection score is

```text
Q = max_j t_j
  + 5 max(0, max_j s_j - (1 + selection.safety_tolerance))
  + 5 max(0, max_j f_j - (1 + audit.maximum_fv_regression))
  + 5 max(0, max_j p_j - (1 + selection.prior_band_tolerance)).
```

The safety terms are score penalties, not hard selection constraints.
Projection residual limits are hard runtime failures during training.

The identity-floor helper accepts a candidate exactly when

```text
candidate_score < identity_score - configured_minimum_gain.
```

Independent `nextafter` boundary checks for identity score `1` and gain
`0.02` observed:

| Candidate score | Result |
|---|---|
| next float below `0.98` | accepted |
| exactly `0.98` | rejected |
| next float above `0.98` | rejected |

The safety penalty is zero immediately below and exactly at
`1 + tolerance`, and positive immediately above. A synthetic example with
target ratio `0.90` and safety ratio `1.025` produced score `0.925` and passed
both the active capability and final identity-floor gains despite exceeding
the `1.02` safety tolerance. This is consistent with the implemented
safety-aware score; the later scientific audit applies separate hard safety
gates.

Capability keeps only forced-open evaluations that strictly improve the best
score. Admission requires

```text
best_forced_open_score
    < identity_score - capability_minimum_improvement.
```

On admission, the complete best capability model state is restored before
router training. Router selection evaluates the deployment gate mode, begins
with the restored capability state, and retains only strict score
improvements. Final selection requires

```text
best_deployable_router_score
    < identity_score - final_minimum_gain.
```

Failure of either admission restores the complete pre-training identity state.
The configured capability gate is validated as `forced_open`; the router gate
is validated as `straight_through`.

## Findings

### TRAIN-RESUME-01 — High — the authenticated manifest is not frozen at run start

- **Requirement or claim:** `_next/README.md:45-46` says training
  authenticates all inputs. `TRAIN-07` and `TRAIN-09` require an authenticated
  resume and rejection after a behavior-affecting implementation, source
  checkpoint, edge file, or map changes.
- **Expected behavior:** The trainer should capture the exact identities used
  to build its in-memory model and pairs once, before training, and every
  checkpoint should retain that frozen manifest. A live-file change should
  abort checkpointing or cause resume rejection.
- **Observed behavior:** `SequentialTrainer._pack()` calls `_auth()` for every
  epoch checkpoint (`training.py:250-265`). `_auth()` re-reads the current
  package modules, configured edge/map files, and source checkpoint from the
  filesystem (`training.py:236-248`). It does not preserve the identities from
  the time the already-loaded model and `PairData` objects were constructed.
- **Independent evidence:** In an isolated temporary setup, an edge file was
  hashed, its corresponding in-memory `PairData.edge_features` was snapshotted,
  and the edge file was then changed. `_auth()` returned the new file hash
  while the in-memory pair remained bit-identical. A synthetic saved manifest
  containing that new hash passed `_validate_resume()`. This is precisely the
  manifest a subsequent `_pack()` would write.
- **Reproduction:** Instantiate a trainer over temporary edge/map/source
  files, save `old = trainer._auth()`, modify only the temporary edge file,
  save `new = trainer._auth()`, and call `_validate_resume()` with `new`.
  Observe `old["data_sha256"][pair]["edge"] !=
  new["data_sha256"][pair]["edge"]`, unchanged in-memory tensors, and no
  validation error for `new`.
- **Impact:** If code or an input changes while a long job is active, the next
  checkpoint can claim the new bytes even though completed optimizer steps
  used the old loaded objects. Resume then rebuilds from the new bytes and
  accepts a scientifically mixed old/new run. This defeats the main purpose
  of authenticated continuation.
- **Recommended correction:** Resolve and hash all inputs before model/pair
  construction, store one immutable run manifest in the trainer, compare live
  inputs with it before every save, and always write the frozen manifest.
  Abort clearly on drift. Include the active entry script and environment
  identity in the manifest as appropriate.
- **Confidence:** High.

### TRAIN-RESUME-02 — High — consumed real-field inputs are absent from resume authentication

- **Requirement or claim:** Every input that changes panel contents and losses
  must be authenticated.
- **Expected behavior:** Both per-pair real-field files, their hashes, and the
  included/skipped variable manifest should participate in resume validation.
- **Observed behavior:** `panels.py:166-172` reads available real fields into
  every non-smoke training and selection panel. `_auth()` records only edge
  and conservative-map files for a pair (`training.py:242-245`). It contains
  no real-field path, availability, variable manifest, or hash.
- **Independent evidence:** Wave 1 `DATA-02` established from active files
  that all five configured real variables are actually consumed for every
  training pair and for `CS-r64_to_ICOD-r64`. An isolated temporary real-file
  content change left `_auth()` exactly unchanged.
- **Impact:** Real safety fields can change between epochs or resume without
  rejection, altering training losses and selection. Combined with
  `TRAIN-RESUME-01`, a checkpoint can also record no evidence of which
  real-field bytes were used before or after a continuation.
- **Recommended correction:** Add resolved endpoint hashes, availability, cell
  counts, and included/skipped variables to the immutable run manifest.
- **Confidence:** High.

### TRAIN-RESUME-03 — Medium — saved state-integrity hashes are not validated

- **Requirement or claim:** Resume should verify model/best/identity state and
  hashes, and a damaged checkpoint should fail safely.
- **Expected behavior:** Every state selected or resumed should be bound to a
  verified digest, especially the frozen corrector during router training.
- **Observed behavior:** `_pack()` writes `corrector_state_sha256`
  (`training.py:261`), but `_validate_resume()` checks only format, schema,
  stage index, and external provenance (`training.py:270-274`). It does not
  compare the recorded corrector digest with `model_state`,
  `capability_best_state`, or `best_model_state`. On router resume,
  `state["corrector_hash"]` starts empty and is then replaced by a digest
  computed from the just-loaded state (`training.py:312-313,411-413`), so the
  recorded digest is not an integrity check. `load_training_checkpoint()`
  likewise does not validate it (`checkpoint.py:159-192`).
- **Static reproduction:** Change a corrector tensor in an isolated copied
  training pack while leaving `corrector_state_sha256` unchanged, re-save it,
  and observe that the resume validator does not inspect the mismatch. The
  normal router path adopts the altered state as its new expected corrector.
- **Impact:** Validly parseable accidental corruption or state substitution
  can pass resume/load. Atomic rename protects against a partially written
  destination but not silent state alteration after a successful write.
- **Recommended correction:** Store and validate hashes for the resumable
  model, optimizer, identity, capability-best, and final-best states. At
  minimum, validate the existing corrector digest before loading/resuming and
  again before publishing a completed checkpoint.
- **Confidence:** High.

### TRAIN-RESUME-04 — Medium — identity selection can retain metrics from a rejected model

- **Requirement or claim:** Checkpoint selection metadata should describe the
  state actually selected by the identity floor.
- **Expected behavior:** If identity is selected, `selection_metrics` should be
  the forced-closed identity metrics. Scores, state, epoch, and metrics should
  refer to the same candidate.
- **Observed behavior:** Initial identity metrics are computed into
  `identity_metrics` (`training.py:299-301`) but never placed in `state`.
  `state["metrics"]` begins empty (`training.py:312`). Capability rejection
  therefore completes with empty metrics. If capability is admitted but the
  router candidate is finally rejected, lines 431-433 restore the identity
  state/score/epoch but do not restore identity metrics; `selection_metrics`
  can describe the last best rejected capability/router candidate instead.
- **Reproduction:** Follow the two branches statically: force capability
  admission false and inspect the packed `selection_metrics`; then force
  capability admission true but final selection false and compare packed
  identity state/score with retained candidate metrics.
- **Impact:** The loaded model is correctly restored to identity, so this does
  not by itself alter inference. It can mislead checkpoint inspection,
  provenance review, and any downstream consumer that treats
  `selection_metrics` as evidence for the selected model.
- **Recommended correction:** Store `identity_metrics` alongside
  `identity_score`, maintain metrics with each best state, and restore all
  three atomically whenever identity wins.
- **Confidence:** High.

### TRAIN-RESUME-05 — Medium — smoke/full mode is recorded but not authenticated on resume

- **Requirement or claim:** Smoke execution traverses control flow but must
  not be treated as scientific training; resume must restore phase, panel, and
  run mode consistently.
- **Expected behavior:** A checkpoint created with `smoke=True` should be
  resumable only as smoke, and a scientific checkpoint only with
  `smoke=False`.
- **Observed behavior:** `_pack()` records `smoke` (`training.py:253`), but
  `_validate_resume()` does not compare it with the current `run(smoke=...)`
  argument. Smoke changes pair selection and quadrature in `scripts/train.py`,
  panel composition/batch sizes, and phase epoch counts in
  `training.py:280-289,331-339,395,414`.
- **Impact:** Default CLI filename suffixing and the active multi-pair
  provenance make an accidental mismatch less likely, but an explicit
  `--output` or direct API use can cross the modes. That can mix a one-step
  smoke panel/state into a scientific continuation.
- **Recommended correction:** Include run mode and resolved pair/panel build
  settings in the immutable manifest and reject a mismatch before loading
  model or optimizer state.
- **Confidence:** High for the validation gap; medium for operational
  likelihood.

### TRAIN-RESUME-06 — Low — a completed checkpoint resume does not repair a lagging history CSV

- **Requirement or claim:** Checkpoint and history should provide a consistent,
  resumable record.
- **Expected behavior:** After a crash between the two atomic writes, resume
  should reconstruct the CSV from authoritative checkpoint history.
- **Observed behavior:** Each epoch writes the checkpoint first and CSV second
  (`training.py:389-391`), and final completion does the same
  (`training.py:443-445`). If interruption occurs after the completed
  checkpoint rename but before the final CSV rename, resume loads the
  checkpoint and immediately returns because `completed` is true
  (`training.py:291-294`); it does not rewrite the CSV.
- **Impact:** Model state remains safe, but the human-facing history can omit
  its final row indefinitely.
- **Recommended correction:** On completed resume, validate and regenerate
  history from the checkpoint before returning, or write a small transactional
  run directory manifest that binds both files.
- **Confidence:** High.

## Cross-confirmed Wave 1 finding

`PM-01` is independently confirmed. With a nonzero corrector, field probability
`0.2`, thresholds `0.4/0.6`, and straight-through mode:

```text
output bit-equal to FV/prefix                         true
task MSE gradient to field-router final bias          0.0
full router-loss gradient to field-router final bias -0.0220000017
```

The boolean exact-rejection mask at `progressive.py:249-254` removes the
task-loss straight-through gradient for a globally closed field. BCE teacher
and safety-gate terms still train router probabilities, as the nonzero full
loss gradient demonstrates. Therefore router training is not stuck, but a
closed target cannot be reopened based on the correction's measured task
benefit; it is reopened by its target role label.

## Resume equivalence evidence

An isolated CPU training harness exercised a complete 3-epoch capability and
3-epoch router run on a nontrivial synthetic conservative pair. Panel tensors,
target/safety order, and selection scores were deterministic functions of
phase/epoch/model state. A simulated process failure was raised immediately
after the atomic checkpoint rename:

1. after capability epoch 1; and
2. after router epoch 1.

Each checkpoint was then resumed through `SequentialTrainer.run(resume=True)`.
The interrupted checkpoint at each cut was also compared with the
corresponding uninterrupted per-epoch pack.

Observed results:

| Compared property | Capability cut | Router cut |
|---|---|---|
| Model at cut | bit-identical | bit-identical |
| Optimizer state at cut | bit-identical | bit-identical |
| Final model state | bit-identical | bit-identical |
| Best capability state | bit-identical | bit-identical |
| Best final state | bit-identical | bit-identical |
| History excluding elapsed seconds | identical | identical |
| Capability/final epochs and identity decision | identical | identical |

This materially supports the current CPU resume control flow. Model
parameters, optimizer moments/step counts, deterministic panel reconstruction,
best states, identity floor, and history were restored correctly in the
exercised paths. The code does not save global Python/NumPy/Torch RNG states,
but the current panel and batching path derives randomness explicitly from
config/phase/epoch/pair seeds, and the current model has no stochastic layer
after initialization. The equivalence statement does not extend to GPU
nondeterministic reductions.

## Other checks that increased confidence

- `ExperimentConfig` rejects any selected capability gate other than
  `forced_open` and router gate other than `straight_through`.
- `set_training_stage()` freezes every unselected stage. The production loop
  snapshots all earlier-stage parameters and all phase-frozen parameters, then
  checks bit equality after each epoch.
- Capability and router optimizers receive only the intended selected-stage
  parameter iterator.
- The best forced-open capability state is restored before router evaluation
  and training.
- Corrector parameters remained bit-identical across the independently
  exercised router phase; the final runtime check also compares their digest
  within one uninterrupted process.
- Exact identity restoration is architectural: a deep copy of the complete
  pre-training model is retained and loaded on either rejection branch.
- Checkpoint and CSV writes use a same-directory temporary file followed by
  `Path.replace()`, so each individual destination update is atomic under the
  expected filesystem semantics. A failed save before replace leaves the
  previous checkpoint available.
- Current source-keyed panel construction and stratified order use explicit
  deterministic seeds; the isolated CPU resume result confirms reconstruction
  for both phases.
- Current active source/FV validation detects a changed FV file before
  training when the production progressive checkpoint carries its expected FV
  hash. This does not cure the live-manifest drift or production-envelope
  issue reported elsewhere.

## Test-coverage assessment

Existing discovered tests cover:

- one frozen-parameter optimizer step;
- phase `requires_grad` transitions;
- simple identity-floor examples;
- target/safety stratification and two-regime weights;
- finite loss/backward in capability;
- exact forced rejection.

They do not execute `SequentialTrainer.run()` and therefore do not cover:

- per-epoch production checkpoint packs;
- either resume phase;
- optimizer restoration;
- capability admission and best-state restoration;
- final identity restoration plus metadata;
- authentication perturbations;
- atomic-write interruption windows;
- smoke/full resume mismatch;
- router task gradients below/exactly at/above routing thresholds.

The isolated checks in this report provide evidence but should become
maintained tests after remediation.

## Deferred checks and remaining uncertainty

Deferred under the audit charter:

- candidate-checkpoint admission and final identity-floor verification;
- GPU interrupted/uninterrupted parity;
- GPU optimizer-state device restoration;
- nondeterministic CUDA indexed-reduction effects;
- full production panels, r64/r128 selection, protected pairs, and external
  meshes;
- end-to-end wall-time continuation across real PBS jobs.

The active high-band candidate's capability epochs are longer than a complete
one-job 12-hour allocation in the observed workflow, making verified
continuation operationally important. This report did not query, alter, or
submit any job.

CPU resume correctness is supported for the current deterministic path.
Final audit acceptance should nevertheless remain blocked on the two High
authentication findings: a continuation is not scientifically authenticated
if the manifest can silently drift during the original job or if consumed real
fields are outside it.
