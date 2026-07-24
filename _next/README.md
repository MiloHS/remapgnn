# Clean progressive remapper

`_next` is an implementation of the frozen FV base followed
by an ordered list of conservative correction stages.

The flow is

```text
geometry checkpoint + FV cell moments -> frozen FV operator -> mid band -> high band
                                      raw source field ------^----------^
```

FV construction predicts signed edge masses `q`. It applies the linear and
quadratic finite-volume moment relaxations and finishes with the joint mass
projection

```text
M = q + A^T lambda,       (A A^T + epsilon I) lambda = b - A q,
S_ij = M_ij / area_target_i.
```

Every learned stage predicts normalized edge corrections `D`. The projection
enforces

```text
sum_j D_ij = 0,           sum_i area_target_i D_ij = 0,
y_next = y_prefix + D(x) x_raw.
```


Typical commands:

```bash
PYTHONPATH=_next python _next/scripts/build_fv.py --config _next/configs/progressive.json --pair PAIR --output fv.pt
PYTHONPATH=_next python _next/scripts/train.py --config _next/configs/progressive.json --device cuda
PYTHONPATH=_next python _next/scripts/train.py --config _next/configs/progressive.json --device cuda --resume
PYTHONPATH=_next python _next/scripts/audit.py --config CONFIG --checkpoint CHECKPOINT --device cuda
```

Schema-4 configuration rejects unknown and ignored fields and is validated
into nested dataclasses. Its model section
selects an approved clean checkpoint, an exact frozen prefix, a named train
stage, and fresh or checkpoint initialization. Training panels are built for
that selected stage rather than implicitly using the last checkpoint stage.
Training builds source-keyed harmonic/mixture panels plus explicitly shared
analytic/real safety anchors, balances both transfer regimes, freezes an input
manifest before loading, and writes an atomic checkpoint after every epoch.
Capability selection restores the best
forced-open corrector; router training freezes that corrector and uses
benefit-taught straight-through routing; hard deployment retains the original prefix as an
identity floor.

Auditing loads np2 maps automatically and writes atomic detail CSV, summary
CSV, and JSON reports beneath the configured reports directory. Protected and
external-resolution pairs require the explicit `--allow-protected` flag.

The verification command, run in the project PyTorch environment, is:

```bash
./next test
```

## Hardened production equivalence

The production replacement is deliberately parallel: it never overwrites
`progressive.pt`.

```bash
./next harden
qsub _next/equivalence.pbs
```

`harden` refuses a dirty Git tree. The job first runs the CPU/r32 gate and
only then runs CUDA equivalence on r32, r64, HeALPix, and r128. Generated
checkpoints, reports, extracted legacy files, and PBS output remain untracked.
The small verifier and PBS recipe stay in Git so a production checkpoint can
be reauthenticated later.

After the full report says both `passed: true` and
`acceptance_ready: true`, create the detached manifest without activation:

```bash
PYTHONPATH=_next python _next/scripts/create_production_manifest.py \
  --checkpoint _next/checkpoints/progressive_hardened.pt \
  --fv-checkpoint _next/checkpoints/fv_relax1.pt \
  --equivalence-report _next/reports/equivalence_hardened_COMMIT.json \
  --output _next/checkpoints/progressive_hardened.manifest.json
```

Activation is a later explicit run with `--activate` and one `--config` for
each active training configuration. It updates those source references before
changing `production.json`, preventing production/training drift.

For use, refer to the repository-level `./next` command documented in
`docs/ACTIVE_WORKFLOW.md`.
