# Configuration, checkpoint, and command interfaces

## Scope and work performed

This report audits commit `f98a0e8ed06678099a98b5a6dfc91ded5d580c97`
and the active, non-archived files it invokes. I inspected:

- both active JSON configurations and every dataclass field in
  `_next/remapgnn_next/config.py`;
- the root `next` wrapper, all four `_next/scripts/*.py` entry points, and the
  two active PBS files;
- progressive, FV, and training checkpoint loading, model assembly,
  authentication, atomic writing, and resume validation;
- configured path construction for active edge data, conservative maps, np2
  maps, real fields, checkpoints, histories, and reports;
- the active checkpoint headers, embedded hashes, stage-state hashes, and the
  committed equivalence record;
- active import isolation and execution from outside the repository through
  the root wrapper.

No implementation, configuration, test, data, checkpoint, job, Git, or
non-audit documentation file was changed. Isolated mutations were written
under `/tmp` and discarded.

CPU/static checks actually performed included:

- loading both active configurations;
- perturbing unknown keys, invalid epoch counts, and the configurable
  frequency divisor;
- independently recomputing active FV and progressive stage-state hashes;
- mutating runtime normalization in a temporary production checkpoint;
- mutating model state and configuration metadata in a temporary completed
  training checkpoint;
- checking the active production and golden-equivalence whole-file hashes;
- checking active imports for legacy dependencies;
- invoking the root wrapper from `/tmp`;
- comparing observed epoch durations with the PBS wall-time request;
- attempting the full pytest suite (pytest is not installed in the project
  environment, which is itself relevant to the test-interface finding).

## Configuration-field ledger

“Required” below means there is no dataclass/parser default. “Auth-only” means
the value enters the canonical configuration hash but has no scientific
execution use. Line references are to the audited commit.

| Field | Default / active value(s) | Validation | Execution path and classification |
|---|---|---|---|
| `schema_version` | required / `3` | Must equal 3 (`config.py:225-227`) | Schema dispatch; used |
| `run_name` | `progressive_next` / `progressive_next`, `high_band_candidate_01` | Converted to string; emptiness not checked | Serialized and authenticated, but does not name checkpoints, histories, or reports; auth-only metadata |
| `seed` | `2407` / `2407` | Converted to integer | Global RNG, panel/mode/mixture seeds, audit panel epoch (`train.py:40`, `panels.py:137`, `evaluation.py:230`); used |
| `pair_roles.train` | required / five r32 pairs | Cross-role set disjointness only | Training-pair construction and training provenance; used |
| `pair_roles.selection` | required in practice / r64 and ICO pairs | Cross-role set disjointness only | Candidate selection and default development audit; used |
| `pair_roles.protected` | no explicit default / two HeALPix pairs | Cross-role set disjointness only | Protected-pair CLI guard; used |
| `pair_roles.external_resolution` | no explicit default / two r128 pairs | Cross-role set disjointness only | Protected-pair CLI guard; used |
| other `pair_roles.*` | accepted | Not validated or rejected | Never selected by active commands; silently inert |
| `paths.analysis` | required / `analysis_medium_improv` | None | Edge path prefix via `edge_path()`; used |
| `paths.maps` | required / `maps_medium_improv` | None | Conservative map path and np2 audit map path; used |
| `paths.models` | required / `_next/checkpoints` | None | Candidate and history directories; used |
| `paths.reports` | `_next/reports` / same | None | Audit report destination; used |
| `paths.real_fields` | `data/MIRA-Datasets` / same | None | Paired real-field NetCDF paths; used, but missing files/variables are silently omitted |
| `paths.graph_suffix` | `kdist_a3p0_mink8` / same | None | Edge filename suffix; used |
| `paths.output_checkpoint` | `progressive_next.pt` / `progressive_next.pt`, `high_band_candidate_01.pt` | None | Candidate checkpoint name; used |
| `paths.history` | `progressive_next_history.csv` / corresponding run history names | None | Training history name; used |
| `features.edge` | required / eight named features | Tuple conversion; no duplicate/name validation | Must exactly match source checkpoint `runtime_data.edge_features`; actual values and normalization still come from the checkpoint (`checkpoint.py:95-100`, `fv.py:254-258`); used as compatibility binding |
| `features.source` | `src_area,src_h,log_src_area` / same | Tuple conversion only | Never read after parsing; FV source features come from the FV checkpoint; **ignored** |
| `features.target` | `tgt_area,tgt_h,log_tgt_area` / same | Tuple conversion only | Never read after parsing; FV target features come from the FV checkpoint; **ignored** |
| `features.sample_per_pair` | `80000` / `80000` | None | Never read after parsing; **ignored** |
| `fv_checkpoint` | required / `_next/checkpoints/fv_relax1.pt` | Nonempty only | FV load and progressive-to-FV reference check; used |
| `model.source_checkpoint` | required / `_next/checkpoints/progressive.pt` | Nonempty | Source prefix and optional train-stage weights; used |
| `model.prefix_through` | required field, nullable / `mid_band` | Must immediately precede final train stage | Exact frozen-prefix construction; used |
| `model.train_stage` | required / `high_band` | Must exist and be final configured stage | Panel band, stage selection, phase freezing; used |
| `model.initialization` | `fresh` / `fresh` | `fresh` or `checkpoint` | Chooses random fresh stage or checkpoint state; used |
| `model.checkpoint_stage` | `null` / omitted | Forbidden for fresh initialization | Source stage for checkpoint initialization, falling back to train-stage name; used only for `checkpoint` |
| `stages[].name` | required / `mid_band`, `high_band` | Nonempty and unique | Model order, lookup, reports; used |
| `stages[].band_lower` | required through `band.lower` / `1.0`, `1.25` | Must be less than upper | Target degree selection and prior-band masks; used |
| `stages[].band_upper` | required through `band.upper` / `1.25`, `1.5` | Must exceed lower | Target degree selection and prior-band masks; used |
| `stages[].edge_dim` | `8` / `8` | Positive only | Network input shape and runtime shape check; used |
| `stages[].hidden` | `48` / `48` | Positive only | Corrector network width; used |
| `stages[].geometry_hidden` | `32` / `32` | Positive only | Geometry encoder width; used |
| `stages[].router_hidden` | `32` / `32` | Positive only | Router widths; used |
| `stages[].delta_scale` | `0.25` / mid `0.25`; high `0.20` or candidate `0.25` | None | Multiplies raw correction; used |
| `stages[].reference_floor` | `1e-3` / default | None | FV-reference floor; used |
| `stages[].edge_chunk` | `50000` / default | None; nonpositive deliberately means all edges | Corrector edge chunking; operationally used |
| `stages[].projection_iterations` | `200` / `200` | Positive | Correction projection and FV helper when supplied a stage-like config; used |
| `stages[].field_gate_low` | `0.4` / default | `0 <= low < high <= 1` | Field hard/straight-through routing; used |
| `stages[].field_gate_high` | `0.6` / default | Same | Field routing; used |
| `stages[].local_gate_low` | `0.1` / default | Same | Local routing; used |
| `stages[].local_gate_high` | `0.9` / default | Same | Local routing; used |
| `stages[].gate_feature_epsilon` | `1e-4` / default | None | Router graph features; used |
| `stages[].epsilon` | `1e-8` / default | None | Geometry/statistic denominators and clamps; used |
| `stages[].capability_gate_mode` | `forced_open` / default | Known mode and selected stage must be `forced_open` | Capability forward path; used |
| `stages[].router_gate_mode` | `straight_through` / default | Known mode and selected stage must be `straight_through` | Router training forward path; used |
| `stages[].deployment_gate_mode` | `hard` / default | Known gate mode | Selection and deployment; used |
| `panel.quadrature_resolution` | `8` / `8` | None | Panel cell-average quadrature; smoke overrides it to 4 |
| `panel.smoother_neighbors` | `9` / `9` | None | Source/target graph smoothers; used |
| `panel.frequency_cells_per_k_squared` | `6.0` / `6.0` | None | Used for target-degree selection only; **inconsistently ignored** by safety selection and reported harmonic frequency |
| `panel.max_degrees_per_epoch` | `4` / `4`, candidate `8` | None | Non-audit target degree count; used |
| `panel.modes_per_degree` | `6` / `6`, candidate `10` | None | Non-audit target modes; used |
| `panel.target_mixtures` | `16` / `16`, candidate `32` | None | Non-audit target mixtures; used |
| `panel.safety_levels` | listed tuple / seven configured levels | Tuple conversion only | Safety harmonic levels; used |
| `panel.safety_modes_per_level` | `3` / `3` | None | Non-audit safety modes; used |
| `panel.safety_mixtures` | `16` / `16` | None | Non-audit safety mixtures; used |
| `panel.audit_max_degrees` | `5` / `5` | None | Selection/audit target degree count; used |
| `panel.audit_modes_per_degree` | `8` / `8` | None | Selection/audit target modes; used |
| `panel.audit_target_mixtures` | `24` / `24` | None | Selection/audit target mixtures; used |
| `panel.audit_safety_modes_per_level` | `6` / `6` | None | Selection/audit safety modes; used |
| `panel.audit_safety_mixtures` | `24` / `24` | None | Selection/audit safety mixtures; used |
| `panel.real_fields` | five defaults / same five | Tuple conversion only | Requested NetCDF variables; used, with silent per-field omission |
| `phases.capability_epochs` | `60` / `60`, candidate `80` | None | Capability loop length; used |
| `phases.capability_learning_rate` | `2e-4` / same | None | Capability AdamW rate; used |
| `phases.router_epochs` | `24` / `24` | None | Router loop length; used |
| `phases.router_learning_rate` | `3e-4` / same | None | Router AdamW rate; used |
| `phases.weight_decay` | `1e-5` / same | None | Both AdamW phases; used |
| `phases.gradient_clip` | `1.0` / same | None | Both phase gradient clipping; used |
| `phases.target_batch` | `2` / `2`, candidate `4` | None | Training stratification and selection evaluation batch; used |
| `phases.safety_batch` | `4` / `4`, candidate `2` | None | Training stratification; used |
| `phases.evaluation_interval` | `4` / `4` | None | Best-checkpoint evaluation cadence; used |
| `loss.guard_tolerance` | `0.005` / same | None | Prefix safety hinge; used |
| `loss.fv_guard_tolerance` | `0.02` / same | None | FV safety hinge; used |
| `loss.cvar_fraction` | `0.25` / same | None | Guard/local/router CVaR tail size; used |
| `loss.guard_weight` | `6.0` / `6.0`, candidate `4.0` | None | Guard-loss multiplier; used |
| `loss.local_weight` | `0.5` / same | None | Local-loss multiplier; used |
| `loss.gate_teacher_weight` | `0.1` / same | None | Router teacher multiplier; used |
| `loss.safety_gate_weight` | `0.05` / same | None | Safety-gate multiplier; used |
| `loss.correction_weight` | `1e-5` / same | None | Correction regularization; used |
| `selection.capability_minimum_improvement` | `0.001` / same | None | Hard capability admission relative to identity; used |
| `selection.final_minimum_gain` | `0.02` / same | None | Hard final identity-floor selection; used |
| `selection.safety_tolerance` | `0.02` / same | None | Selection-score penalty onset; used |
| `selection.prior_band_tolerance` | `0.01` / same | None | Selection-score penalty onset; used |
| `audit.row_tolerance` | `1e-8` / same | None | Training hard failure and audit promotion; used |
| `audit.column_tolerance` | `1e-10` / same | None | Training hard failure and audit promotion; used |
| `audit.minimum_target_gain` | `0.03` / same | None | Audit promotion threshold; used |
| `audit.maximum_safety_regression` | `0.02` / same | None | Audit promotion threshold; used |
| `audit.maximum_prior_band_regression` | `0.01` / same | None | Audit promotion threshold; used |
| `audit.maximum_fv_regression` | `0.02` / same | None | Selection penalty and audit promotion; used |
| `audit.field_batch` | `2` / same | None | Audit batching only; operational |
| `audit.timing_repeats` | `5` / same | None | Structural timing diagnostic only; diagnostic |

Aliases `geom_hidden`, `gate_hidden`, `q_floor`, `gate_feature_eps`, `eps`,
`projection_iterations_train`, `src_node`, and `tgt_node` are accepted parser
compatibility names but do not occur in either active JSON file.

## Command and path behavior

The root `next` wrapper resolves its own directory, changes to the repository
root, prepends `_next` to `PYTHONPATH`, and therefore works when invoked through
an absolute path from another directory. No active clean module or script
imports archived `remapgnn` or historical `scripts` code.

Command mapping is otherwise direct:

- `status` is hard-coded to `progressive.json` and has no config option.
- `train`, `smoke`, and `resume` prepend the default config/device; later user
  arguments can override `--config`, `--device`, `--checkpoint`, and `--output`.
- `audit` and `audit-protected` pin `progressive.pt`, but a later user
  `--checkpoint` overrides that path.
- `audit-candidate` defaults to the candidate path from the selected config.
- `build-fv` requires a pair and output, with optional edge/map/checkpoint
  overrides.

Direct Python scripts use cwd-relative defaults and cwd-relative paths from
the JSON. The documented root wrapper avoids this ambiguity. `--output`
overrides only the training checkpoint: history still goes to the configured
history path, so two runs with different output overrides can overwrite the
same history.

Checkpoint writes and history/report writes use sibling `.tmp` files followed
by `Path.replace`, providing atomic replacement on the active filesystem.
Truncated or non-pickle checkpoint files fail in `torch.load`, and incomplete
training checkpoints are rejected by normal candidate loading. These are
material non-findings.

## Findings

### CFG-01 — High — active production checkpoint is not bound to its golden equivalence result

**Requirement.** `--require-production` and status must establish that the
exact runtime checkpoint being used is the checkpoint that passed conversion
equivalence.

**Expected.** The active whole-file checkpoint hash, including runtime feature
normalization, matches an externally authenticated equivalence record; a
changed payload is rejected.

**Observed.** The active `_next/checkpoints/progressive.pt` hash is
`4a64d9c43f6f39059d390c3d2bca35f08b7e36309e6c72dcdc520e767d0d7c15`,
while committed `_next/reports/equivalence_completed_v24f.json` records
`82ea246a65624c2471654e1e11165ef0dfbe40c811d388638183958cf00815ed`.
`./next status` nevertheless reports `Equivalence passed: True`.

`_validated_progressive_pack` (`checkpoint.py:34-51`) trusts mutable embedded
`production` and `equivalence.passed` flags and verifies only optional stage
state hashes. It does not authenticate `runtime_data`, normalization, FV
reference metadata, or the whole pack against the committed golden result.

**Independent reproduction.**

```text
active 4a64d9c4...
embedded_equivalence_hash 82ea246a...
mutated a0d9b034...
MUTATED_ACCEPTED 124.5881118774414
```

This came from changing only
`runtime_data.normalization.edge_mean[0,0]` in a `/tmp` copy and calling
`load_progressive_checkpoint(copy, require_production=True)`.

**Impact.** Changed runtime normalization can change scientific output while
the checkpoint remains “production” and “equivalent.”

**Recommendation.** Pin the final whole-file hash in an external trusted
manifest and have all production commands compare it, or define and verify a
non-self-referential authenticated payload digest covering all behavioral
fields. Make status fail visibly on mismatch.

**Confidence:** High.

### CFG-02 — High — candidate audit does not authenticate candidate state or bind it to the audit configuration

**Requirement.** Candidate promotion must evaluate the saved model with the
same authenticated scientific configuration, implementation, and data under
which it was selected.

**Expected.** Candidate load verifies model-state hashes and saved provenance;
audit rejects a different config, implementation, data set, FV checkpoint, or
modified best/identity state.

**Observed.** `load_training_checkpoint` (`checkpoint.py:159-192`) checks
format, schema, completion, and the source progressive checkpoint hash only.
It does not verify the saved `config_sha256`, implementation hashes, data
hashes, model state, identity state, or `corrector_state_sha256`. `audit.py`
then uses its separately loaded config for panels, pairs, and promotion
thresholds without comparing it to the checkpoint config.

A `/tmp` copy of the active candidate was marked completed, one tensor in
`best_model_state` was changed, and its saved config run name was changed.
`load_training_checkpoint` printed:

```text
MUTATED_CANDIDATE_ACCEPTED stages.0.geom_encoder.net.0.weight
```

**Impact.** A modified model or a checkpoint audited under different bands,
panels, and thresholds can receive a passing audit.

**Recommendation.** Authenticate full selected model/identity states and
compare the canonical supplied config plus all saved provenance categories
before evaluation. Candidate audit should fail closed on any mismatch.

**Confidence:** High.

### CFG-03 — High — resume and audit omit behavioral data inputs from authentication

**Requirement.** `_next/README.md:44-46` states that training “authenticates
all inputs,” and resume must reject changes to every authenticated input
category.

**Expected.** Every file that affects panels, reference predictions, or
decisions is hashed.

**Observed.** `_auth()` (`training.py:236-248`) covers semantic config, package
Python files, train/selection edge and conservative map files, and the source
checkpoint. It omits the paired real-field NetCDF inputs loaded by
`panels.py:166-172`. Audit provenance (`evaluation.py:240-245`) also omits
real-field files and the `*_conserve_np2.nc` map used at
`evaluation.py:233-235`.

**Impact.** Resume accepts changed real training/selection fields. An audit
report cannot establish which real fields or np2 reference produced its
decision.

**Recommendation.** Hash all resolved real-field and np2 inputs, record
explicit absence/omission, and validate them on resume/audit.

**Confidence:** High (static complete path trace).

### CFG-04 — High — unknown top-level configuration keys are silently ignored

**Requirement.** Typed configuration should reject typos rather than silently
change scientific behavior through defaults.

**Observed.** `ExperimentConfig.from_dict` consumes known fields but never
checks remaining top-level keys (`config.py:260-278`). A perturbation adding:

```json
"losss": {"guard_weight": 999}
```

loaded successfully, retained the real default/configured `guard_weight=6`,
and omitted `losss` from `to_dict()` and therefore from resume authentication.
Unknown nested loss keys did reject with `TypeError`.

**Impact.** A misspelled top-level scientific section can silently run with
defaults and leave no authenticated trace.

**Recommendation.** Reject `set(raw) - allowed_top_level_fields` with a clear
error; provide the same explicit unknown-key validation for every nested
section.

**Confidence:** High.

### CFG-05 — High — documented PBS `EXTRA` invocation does not reliably forward `EXTRA`

**Requirement.** The documented command for selecting a config/stage must
cause the batch job to receive it.

**Observed.** `docs/ACTIVE_WORKFLOW.md:48-52` recommends:

```bash
EXTRA="--stage high_band" /opt/pbs/bin/qsub jobs_next_train.pbs
```

The PBS script reads `${EXTRA:-}`, but arbitrary submit-process environment
variables require PBS `-v` (or `-V`) to enter the job environment. With no
forwarded value, the script silently executes default `./next train`. This is
consistent with the already observed accidental default-config job.

**Impact.** Expensive training can run the wrong scientific configuration
without an early error.

**Recommendation.** Document and use:

```bash
/opt/pbs/bin/qsub -v EXTRA="--config _next/configs/high_band_candidate_01.json" jobs_next_train.pbs
```

Also print the resolved config path/hash at process start.

**Confidence:** High.

### CFG-06 — High — active candidate cannot finish in one documented PBS allocation

**Requirement.** The production submission path should either finish or
document and automate authenticated continuation.

**Observed.** `high_band_candidate_01.json` requests 80 capability plus 24
router epochs. Capability epochs 1-9 in the active history took 639-666
seconds (about 648 seconds mean), projecting about 14.4 hours for capability
alone, before mesh startup, initial evaluation, and router training. The active
PBS job requests 12 hours. Documentation shows only one submission and does
not describe required resume/chaining.

**Impact.** The documented job necessarily ends before completion. Manual
resubmission is error-prone, especially after failures unrelated to wall time.

**Recommendation.** Request sufficient wall time or provide a scheduler-aware
continuation workflow that resumes only from a validated incomplete
checkpoint and distinguishes wall-time termination from other failures.

**Confidence:** High.

### CFG-07 — High — `--require-production` is bypassed for training checkpoints

**Requirement.** Root `audit` and `audit-protected` are documented as auditing
the approved production checkpoint.

**Observed.** Wrapper user arguments follow the pinned checkpoint and can
override it. In `audit.py:37-42`, `--require-production` is applied only to the
clean-progressive branch. A completed clean-training checkpoint takes the
other branch and is accepted without production status.

**Impact.** For example, `./next audit-protected --checkpoint CANDIDATE
--pairs ...` can evaluate an unpromoted candidate through the production /
protected command despite the guard.

**Recommendation.** If `require_production` is set, reject every training
checkpoint and disallow or explicitly validate checkpoint overrides.

**Confidence:** High (direct static control-flow proof).

### CFG-08 — Medium — configurable frequency divisor has inconsistent semantics

**Requirement.** A behavior-affecting frequency scale must be applied
consistently to target selection, safety selection, and recorded frequency.

**Observed.** `panel.frequency_cells_per_k_squared` affects target degree
selection at `panels.py:138`. `_harmonics` and `harmonic_batch` hard-code
division by 6 (`panels.py:98`, `fields.py:235`), and `_level_harmonics`
hard-codes the same effective K (`panels.py:112`).

With `n_src=10242` and a temporary value of 24, target selection chose degrees
26-31 for configured band `(1.25,1.5]`, but their recorded frequencies were
0.629-0.750, while lower-bound safety remained degree 51. Both active configs
use 6, so the current run is aligned; the perturbable field is not.

**Recommendation.** Pass one effective-K definition through all three paths,
or remove the field and validate that the fixed value is 6.

**Confidence:** High.

### CFG-09 — Medium — three feature configuration fields are ignored

`features.source`, `features.target`, and `features.sample_per_pair` are parsed,
serialized, and authenticated but never used. Source/target FV features come
from `fv_checkpoint["features"]`; runtime correction features and
normalization come from the progressive checkpoint. Changing the three JSON
fields therefore suggests a scientific change but only invalidates resume.

**Recommendation.** Remove/mark them metadata-only, or bind and use them
against the corresponding checkpoint definitions. Add perturbation tests.

**Confidence:** High.

### CFG-10 — Medium — most numeric configuration domains are unvalidated

Only selected stage dimensions, projection iterations, and routing thresholds
receive numerical range checks. Panel counts/divisors, epochs, rates, batches,
evaluation interval, loss weights/CVaR, selection thresholds, audit
tolerances, and epsilon-like stage values accept zero, negative, non-finite,
or otherwise nonsensical values. A temporary `capability_epochs=-4` and
`frequency_cells_per_k_squared=24` both parsed successfully.

This can yield empty phases, modulo/division errors, NaNs, inverted decisions,
or silently meaningless training.

**Recommendation.** Define finite/range/integer validation for every numeric
field and boundary-test it.

**Confidence:** High.

### CFG-11 — Medium — active operational entry points are outside commit and resume implementation authentication

The two active PBS files are ignored by root `*.pbs` and are not in commit
`f98a0e8`. Training provenance hashes `_next/remapgnn_next/*.py` only; it does
not hash `_next/scripts/train.py`, root `next`, PBS scripts, or environment /
dependency versions.

Core scientific helpers are covered, and pair-set changes usually alter the
data-hash mapping, but orchestration, source/output selection, smoke handling,
or launch environment can change across resume without rejection.

**Recommendation.** Version active job scripts and include all executed entry
points plus a minimal environment manifest in provenance.

**Confidence:** High.

### CFG-12 — Medium — the advertised test command skips most test functions

`./next test` uses `unittest discover`. Five test files primarily contain
pytest-style free functions, which unittest imports but does not execute.
The 12 reported tests come from `test_workflow_unittest.py`; constraints,
model-equivalence, panel, sparse, and training free-function tests are skipped.
The project environment has no `pytest` module, so the alternate suite cannot
currently be run directly.

**Impact.** The public verification command reports success without executing
much of the stated coverage.

**Recommendation.** Convert tests to discovered `unittest.TestCase` methods or
install/use pytest consistently, and assert the expected collected test count.

**Confidence:** High.

### CFG-13 — Low — output override and run naming can collide

`--output` changes only the checkpoint destination; history remains
`config.paths.history_path`. `run_name` does not derive checkpoint, history, or
report names, and audit report basenames are hard-coded to
`progressive_next_audit_<tag>`. Concurrent or manually overridden runs can
overwrite histories/reports.

**Recommendation.** Derive all artifacts from a validated run identifier or
offer explicit paired output/history/report paths with collision checks.

**Confidence:** High.

## Material non-findings

- The audited commit matches the active tracked implementation; only
  `AGENTS.md` and audit reports are untracked during this review.
- No clean runtime module or active script imports archived implementation.
- Root `next` normalizes cwd and prepends `_next` to `PYTHONPATH`.
- Both active configs pass their existing schema, unique-stage, stage-order,
  role-disjointness, and selected training-gate checks.
- Frozen prefix configuration is compared exactly to the source checkpoint.
- Fresh initialization creates a genuinely new final stage; checkpoint
  initialization rejects incompatible network dimensions and loads state
  strictly.
- Configured edge feature order is checked exactly against source runtime
  data.
- Current FV network state and both active progressive stage states match
  their embedded tensor hashes.
- The active progressive checkpoint contains and matches the exact active FV
  file hash.
- Epoch checkpoint and CSV/report replacement are atomic at the file level.
- Normal candidate loading rejects an incomplete training checkpoint.
- Resume compares canonical typed configuration, package Python hashes,
  train/selection edge/map hashes, source path/hash, and stage index.

These controls materially reduce accidental drift, but they do not close the
whole-checkpoint, candidate/config, real-field/np2, or entry-point gaps above.

## Deferred checks and remaining uncertainty

Per audit instructions, I did not:

- inspect or audit the candidate as an accepted scientific result;
- run GPU parity, expensive full-mesh builds, protected/external audits, or
  candidate promotion;
- interrupt, resume, submit, cancel, or alter the running job;
- mutate real authenticated inputs;
- perform a full interrupted-vs-uninterrupted training comparison (owned by
  the training/resume specialist);
- test actual PBS export semantics by submitting a job;
- test atomicity under real process/node/filesystem failure.

The production hash mismatch may have arisen from a post-equivalence metadata
rewrite, but the interface neither explains nor authenticates that rewrite;
the observed acceptance weakness is independent of its cause.
