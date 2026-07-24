# Configuration-field ledger

## Result

Every dataclass field and every field appearing in the two active JSON files
was traced through the audited implementation. The full field-by-field table,
including defaults, validation, exact execution paths, and active values, is
in [50_config_interfaces.md](50_config_interfaces.md#configuration-field-ledger).

This coordinator ledger records the resulting classifications and exceptions.

## Used scientific fields

These fields change model, data, training, selection, or audit behavior:

- `seed`
- `pair_roles.train`
- `pair_roles.selection`
- `pair_roles.protected`
- `pair_roles.external_resolution`
- `paths.analysis`
- `paths.maps`
- `paths.real_fields`
- `paths.graph_suffix`
- `features.edge`
- `fv_checkpoint`
- `model.source_checkpoint`
- `model.prefix_through`
- `model.train_stage`
- `model.initialization`
- `model.checkpoint_stage`
- `stages[].name`
- `stages[].band.lower`
- `stages[].band.upper`
- `stages[].edge_dim`
- `stages[].hidden`
- `stages[].geometry_hidden`
- `stages[].router_hidden`
- `stages[].delta_scale`
- `stages[].reference_floor`
- `stages[].projection_iterations`
- `stages[].field_gate_low`
- `stages[].field_gate_high`
- `stages[].local_gate_low`
- `stages[].local_gate_high`
- `stages[].gate_feature_epsilon`
- `stages[].epsilon`
- `stages[].capability_gate_mode`
- `stages[].router_gate_mode`
- `stages[].deployment_gate_mode`
- `panel.quadrature_resolution`
- `panel.smoother_neighbors`
- `panel.max_degrees_per_epoch`
- `panel.modes_per_degree`
- `panel.target_mixtures`
- `panel.safety_levels`
- `panel.safety_modes_per_level`
- `panel.safety_mixtures`
- `panel.audit_max_degrees`
- `panel.audit_modes_per_degree`
- `panel.audit_target_mixtures`
- `panel.audit_safety_modes_per_level`
- `panel.audit_safety_mixtures`
- `panel.real_fields`
- every `phases.*` field
- every `loss.*` field
- every `selection.*` field
- `audit.row_tolerance`
- `audit.column_tolerance`
- `audit.minimum_target_gain`
- `audit.maximum_safety_regression`
- `audit.maximum_prior_band_regression`
- `audit.maximum_fv_regression`

## Used operational fields

These fields affect paths, batching, performance, reporting, or serialization
without changing the intended mathematical result:

- `schema_version`
- `paths.models`
- `paths.reports`
- `paths.output_checkpoint`
- `paths.history`
- `stages[].edge_chunk`
- `audit.field_batch`
- `audit.timing_repeats` (diagnostic only)

## Conditional fields

- `model.checkpoint_stage` is used only when
  `model.initialization == "checkpoint"`.
- `model.prefix_through` may be null only when there is no configured frozen
  prefix before the train stage.
- `panel.quadrature_resolution` is overridden to `4` in smoke mode.
- `paths.output_checkpoint` can be overridden with `--output`, but
  `paths.history` is not changed with it.
- `features.edge` binds the config to the source checkpoint's ordered feature
  names; feature values and normalization come from checkpoint runtime data.

## Metadata or authentication-only

- `run_name` is serialized and included in the configuration hash but does not
  name the checkpoint, history, or reports.

## Ignored fields

The following accepted fields are never read after parsing:

- `features.source`
- `features.target`
- `features.sample_per_pair`

Changing them changes resume authentication but not execution. This is
reported as `CFG-09`.

Unrecognized names inside `pair_roles` are also accepted but inert.

## Inconsistently interpreted field

`panel.frequency_cells_per_k_squared` is used for target-degree selection, but
safety-degree selection and stored harmonic frequencies independently
hard-code the divisor `6`. Both active configurations use `6`, so the active
candidate is aligned, but supported perturbations are not. This is reported as
`DATA-03` / `CFG-08`.

## Fields with insufficient domain validation

Most numeric fields are converted but not range-checked. Important examples
include:

- nonpositive learning rates, epoch counts, batch sizes, and evaluation
  intervals;
- `cvar_fraction` outside `(0, 1]`;
- negative loss weights or tolerances;
- nonpositive quadrature/smoother sizes;
- negative or non-finite `delta_scale`, floors, and epsilons;
- empty role lists and path strings;
- duplicate feature names.

Known validation that does exist:

- schema version must be `3`;
- stage names are unique and bands increase;
- network dimensions and projection iterations are positive;
- router thresholds are ordered within `[0, 1]`;
- gate modes are from the supported set;
- role pair names cannot overlap across configured roles;
- the train stage exists and is final;
- frozen prefix order and definitions match the source checkpoint;
- initialization is `fresh` or `checkpoint`;
- nested dataclass keys are generally rejected when unknown.

Top-level unknown JSON keys are not rejected. A perturbation with a misspelled
`losss` section loaded successfully, silently used the real `loss` defaults,
and discarded the typo from `to_dict()` and authentication. This is reported
as High `CFG-04`.

## Active-config differences that were verified

Compared with `progressive.json`, `high_band_candidate_01.json` changes:

- `run_name`;
- checkpoint and history filenames;
- high-band `delta_scale`: `0.20 -> 0.25`;
- target degrees per epoch: `4 -> 8`;
- modes per target degree: `6 -> 10`;
- target mixtures: `16 -> 32`;
- capability epochs: `60 -> 80`;
- target/safety batch sizes: `2/4 -> 4/2`;
- guard weight: `6 -> 4`.

The live PBS metadata recorded
`EXTRA=--config _next/configs/high_band_candidate_01.json`, and the resulting
history filename and approximately 648-second epochs were consistent with
those larger candidate settings.

## Perturbation coverage

Performed or independently reproduced:

- top-level unknown key;
- stage structural dimensions;
- prefix definitions and order;
- initialization mode and checkpoint stage;
- edge-feature layout mismatch;
- band upper-bound rounding;
- frequency divisor;
- router threshold boundaries;
- selection and audit threshold boundaries;
- projection iteration count;
- production checkpoint runtime normalization;
- resume provenance inputs;
- smoke/full resume mode;
- output/config/checkpoint CLI overrides.

Deferred:

- exhaustive behavioral perturbation of every numeric field on production
  data;
- GPU-only settings and parity;
- completed active-candidate checkpoint fields;
- full PBS continuation execution.
