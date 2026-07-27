# Active clean workflow

`_next` supports the preserved FV model and the new conservative-bilinear
progressive model. The production pointer remains unchanged until a new model
passes development, protected, and external-resolution audits.

- the frozen converted `relax1` finite-volume base;
- the accepted mid-band correction;
- an ordered high-band stage with an exact identity fallback (if the correction is 
  judgeed insufficient).

The completed v24F high-band candidate was rejected, so
the approved clean checkpoint records `selected_identity=true`. 

## Commands

From the repository root:

```bash
./next help
```

The useful commands are:

| Command | Meaning |
|---|---|
| `./next status` | Show the approved checkpoint and any trained candidate. |
| `./next test` | Run the dependency-free clean unit tests. |
| `./next smoke` | Exercise one capability epoch and one router epoch. |
| `./next baseline-check` | Validate every fixed conservative bilinear baseline. |
| `./next train` | Train the configured correction stage from the approved clean checkpoint. |
| `./next resume` | Resume the authenticated candidate checkpoint. |
| `./next audit` | Audit the approved converted checkpoint. |
| `./next audit-candidate --config ... --checkpoint ...` | Audit one explicit completed candidate. |
| `./next audit-protected --pairs ...` | Explicitly consume protected or external pairs. |
| `./next compare list\|run ...` | Compare FV, a named progressive prefix, and np2 on chosen fields. |
| `./next build-fv PAIR OUTPUT` | Build and save a clean FV operator for one pair. |

`./next` uses `REMAPGNN_PYTHON` when supplied, then the active virtual/Conda
environment, then `python3`/`python` from `PATH`. PBS jobs use
`conda run -n remap_gpu`; override the portable environment name with
`REMAPGNN_CONDA_ENV` when needed.

## Comparing methods and visualizing errors

List the available pairs, fields, and stage names:

```bash
./next compare list --checkpoint _next/checkpoints/progressive.pt
```

Run FV, the accepted first correction, and np2 on one exact harmonic and a
sampled frequency band:

```bash
./next compare run \
  --checkpoint _next/checkpoints/progressive.pt \
  --pair CS-r32_to_ICOD-r32 \
  --methods fv stage:mid_band np2 \
  --field harmonic:36:0 \
  --band 0.9 1.3
```

Fields may also be selected as `analytic:smooth1`, `analytic:smooth2`, or
`real:Topography` (and the other real fields shown by `compare list`).
The command prints the error table and writes `metrics.csv`,
`frequency_profile.csv`, `values.nc`, and PNG figures beneath the ignored
`.generated/comparisons/` directory. The request record contains checkpoint
and input hashes, so these exploratory results remain traceable without
cluttering Git.

For the categorical bilinear line:

```bash
./next compare --config _next/configs/bilinear_progressive.json list \
  --checkpoint _next/checkpoints/bilinear_progressive_low.pt

./next compare --config _next/configs/bilinear_progressive.json run \
  --checkpoint _next/checkpoints/bilinear_progressive_low.pt \
  --pair CS-r32_to_ICOD-r32 \
  --methods bilinear_raw bilinear stage:low fv np2 \
  --field analytic:smooth2 \
  --band low
```

Exact fields and named band sweeps are independent. Omitting `--band` evaluates
only the explicitly requested field.

## Conservative bilinear model

The fixed ESMF bilinear result receives one field-wide constant adjustment
that restores the source area integral. Every learned stage then predicts a
correction on the k-distance graph and projects it to zero target-row sum and
zero area-weighted source-column sum. Consequently, the baseline and every
accepted prefix preserve constants and global conservation.

The learned correction uses the existing k-distance graph, which contains
every ESMF bilinear stencil edge on all active pairs. Its row-normalized
reference blends 75% of the actual bilinear weights with 25% uniform graph
support. This exposes the baseline stencil while retaining non-bilinear
correction edges. The only configured external edge feature is normalized
neighbor rank; area ratio, distance, and bilinear-reference information are
already supplied by the intrinsic geometry features. Raw candidate-count
features are excluded because they identify mesh topology and shifted sharply
on held-out ICO geometry.

The first experiment uses global degree bands `low=1–16`, `mid=17–32`,
`high=33–48`, and the untrained guard band `49–64`. Training includes pure
harmonics, within-band mixtures, target-plus-guard mixtures at 25/50/75%
target energy, cross-band guard mixtures, smooth analytic fields, and
available real fields. Mixtures with 25% or 50% target energy are guards;
75% mixtures are targets. Training also guarantees coverage of the four
degrees immediately above each stage boundary. No normalized frequency is
used by this model.

`ICO-r32_to_CS-r32` is a training pair and also contributes a disjoint
validation-order panel during selection. This tests field generalization on
the previously missing topology without reusing training harmonic modes.
`CS-r64_to_ICOD-r64` remains the held-out development mesh pair; protected
HeALPix and external r128 pairs remain untouched until their audit phases.

Capability selection uses a bounded forced-open safety allowance so a useful
corrector can reach router training. Final hard-routed selection and auditing
retain the strict safety thresholds and identity fallback. First-stage scoring
counts bilinear/prefix safety once and has no prior-band term. Evaluation rows
in the history CSV name every score component and the worst safety field.

Before training:

```bash
./next baseline-check --config _next/configs/bilinear_progressive.json
./next test
./next smoke --config _next/configs/bilinear_progressive.json \
  --all-stages
```

Train and audit low first:

```bash
./submit_next_workflow.sh \
  --config _next/configs/bilinear_progressive.json \
  --stage low
```

After its development audit passes, train mid using the accepted low
checkpoint and audit report:

```bash
./submit_next_workflow.sh \
  --config _next/configs/bilinear_progressive.json \
  --stage mid \
  --checkpoint _next/checkpoints/bilinear_progressive_low.pt \
  --source-audit _next/reports/bilinear_progressive_low_development.json
```

Repeat the same pattern for high using the accepted mid checkpoint and report.

## Cluster jobs

Submit these from the repository root:

```bash
./submit_next_workflow.sh --config CONFIG --checkpoint CHECKPOINT --dry-run
./submit_next_workflow.sh --config CONFIG --checkpoint CHECKPOINT
```

Optional command-line arguments can be supplied through `EXTRA`, for example:

```bash
/opt/pbs/bin/qsub -v 'EXTRA=--config _next/configs/progressive.json --stage high_band' jobs_next_train.pbs
```

The audit job returns a nonzero status when promotion gates fail.

## Safe development order

For a new high-band candidate:

1. Edit `_next/configs/progressive.json`.
2. Run `./next test`.
3. Run `./next smoke` when the two-phase integration path changed.
4. Inspect the smoke checkpoint and history.
5. Submit `jobs_next_train.pbs`.
6. Audit development pairs with `jobs_next_audit.pbs`.
7. Use protected/external pairs only after development promotion succeeds.

Do not train directly from a legacy checkpoint. Legacy checkpoints enter the
clean system only through the already-audited conversion boundary.

## Model initialization

The `model` configuration is explicit:

- `source_checkpoint` is the approved clean checkpoint supplying the prefix.
- `prefix_through` is the final frozen stage retained from that checkpoint.
- `train_stage` names the final configured stage to train.
- `initialization` is `fresh` or `checkpoint`.
- `checkpoint_stage` optionally names the source stage used for checkpoint
  initialization.

Every frozen prefix stage must match the checkpoint configuration exactly.
Checkpoint initialization permits behavioral changes such as routing
thresholds, but rejects structural changes to network dimensions. Fresh
initialization permits structural changes within the generic correction-stage
implementation.

## Data and outputs

- Edge graphs: `analysis_medium_improv/edge_dataset_*.parquet`
- Conservative maps and np2 references: `maps_medium_improv/`
- Available real fields: `data/MIRA-Datasets/`
- Clean checkpoints: `_next/checkpoints/`
- Clean reports: `_next/reports/`
