# Active clean workflow

`_next` is now the default runtime, trainer, and auditor. It uses:

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
| `./next audit-candidate` | Audit the latest completed trained candidate. |
| `./next audit-protected --pairs ...` | Explicitly consume protected or external pairs. |
| `./next build-fv PAIR OUTPUT` | Build and save a clean FV operator for one pair. |

Set `REMAPGNN_PYTHON=/path/to/python` only when a different Python environment
is needed. On the project system, `./next` automatically uses the established
GPU environment.

## Cluster jobs

Submit these from the repository root:

```bash
/opt/pbs/bin/qsub jobs_next_train.pbs
/opt/pbs/bin/qsub jobs_next_audit.pbs
```

Optional command-line arguments can be supplied through `EXTRA`, for example:

```bash
EXTRA="--stage high_band" /opt/pbs/bin/qsub jobs_next_train.pbs
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
