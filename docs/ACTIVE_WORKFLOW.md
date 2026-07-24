# Active clean workflow

`_next` is the default runtime, trainer, and auditor. The current checkpoint
remains available, but production training/audit is intentionally blocked
until `_next/configs/production.json` names an approved detached manifest.

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
| `./next train` | Train the configured correction stage from the approved clean checkpoint. |
| `./next resume` | Resume the authenticated candidate checkpoint. |
| `./next audit` | Audit the approved converted checkpoint. |
| `./next audit-candidate --config ... --checkpoint ...` | Audit one explicit completed candidate. |
| `./next audit-protected --pairs ...` | Explicitly consume protected or external pairs. |
| `./next build-fv PAIR OUTPUT` | Build and save a clean FV operator for one pair. |

`./next` uses `REMAPGNN_PYTHON` when supplied, then the active virtual/Conda
environment, then `python3`/`python` from `PATH`. PBS jobs use
`conda run -n remap_gpu`; override the portable environment name with
`REMAPGNN_CONDA_ENV` when needed.

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
