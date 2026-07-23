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
PYTHONPATH=_next python _next/scripts/audit.py --config _next/configs/progressive.json --device cuda
```

The configuration is validated into nested dataclasses. Its model section
selects an approved clean checkpoint, an exact frozen prefix, a named train
stage, and fresh or checkpoint initialization. Training panels are built for
that selected stage rather than implicitly using the last checkpoint stage.
Training builds source-keyed harmonic/mixture/analytic/real-field panels,
balances both transfer regimes, authenticates all inputs, and writes an atomic
checkpoint after every epoch. Capability selection restores the best
forced-open corrector; router training freezes that corrector and uses
straight-through routing; hard deployment retains the original prefix as an
identity floor.

Auditing loads np2 maps automatically and writes atomic detail CSV, summary
CSV, and JSON reports beneath the configured reports directory. Protected and
external-resolution pairs require the explicit `--allow-protected` flag.

The verification command, run in the project PyTorch environment, is:

```bash
PYTHONPATH=_next python -m unittest discover -s _next/tests -p 'test_*.py' -v
```

For use, refer to the repository-level `./next` command documented in
`docs/ACTIVE_WORKFLOW.md`.
