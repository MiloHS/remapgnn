# RemapGNN

Conservative remapping between spherical climate meshes.

## Start here

The active implementation is `_next`. It consists of:

```text
frozen finite-volume base
        ↓
ordered conservative correction stages
        ↓
development and protected audits
```

I am currently developing the next high-band candidate, the previous version did not
learn/correct enough.

Use the commands:

```bash
./next status
./next test
./next smoke
./next train
./next resume
./next audit --pairs CS-r64_to_ICOD-r64 ICO-r32_to_CS-r32
```

Run `./next help` for the complete command list.

## Workflow

1. Check the active checkpoint:

   ```bash
   ./next status
   ```

2. After changing training settings, run the short two-phase check:

   ```bash
   ./next smoke
   ```

3. Train or resume the candidate:

   ```bash
   ./next train
   ./next resume
   ```

4. Audit the completed candidate:

   ```bash
   ./next audit-candidate
   ```

The cluster equivalents are:

- `jobs_next_train.pbs`
- `jobs_next_audit.pbs`

These jobs are intentionally not submitted automatically.
Run `./next smoke` interactively when the capability/router/resume integration
path needs a short check.

## Important files

- Active configuration: `_next/configs/progressive.json`
- Development configuration: `_next/configs/high_band_candidate_01.json`
- Approved clean checkpoint (gitignored): `_next/checkpoints/progressive.pt`
- Frozen clean FV checkpoint (gitignored): `_next/checkpoints/fv_relax1.pt`
- Detailed guide: `docs/ACTIVE_WORKFLOW.md`
- Project history and current research direction: `docs/PROJECT_HISTORY.md`
- Package architecture: `_next/README.md`

Large edge datasets, maps, and real fields remain in
`analysis_medium_improv/`, `maps_medium_improv/`, and `data/` (gitignored).

The model section of the configuration names an approved source checkpoint,
the last frozen prefix stage, the train stage, and whether that stage starts
fresh or from compatible clean checkpoint weights. Frozen prefix definitions
must match exactly; structural stage changes automatically require fresh
initialization.

## Old implementation

The historical `remapgnn/`, `scripts/`, versioned configs, and checkpoints are
archived.

A migration summary is stored in:

```text
_archive/legacy_progressive_2026-07-23/
```

The archive description is local only and is deliberately not part of GitHub.

## Current research boundary

The next task is improving the high-band correction training recipe.
Later we will rebuild FV cleanly.
