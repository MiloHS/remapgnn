# Clean progressive remapper

`_next` contains two isolated model lines:

- the preserved frozen finite-volume (FV) base with its existing correction
  stages; and
- the new fixed ESMF bilinear base with fresh low-, mid-, and high-degree
  correction stages.

The bilinear model does not learn its baseline. It first applies the ESMF
weights `B`, then adds one spatially constant adjustment so the target integral
equals the source integral:

```text
z = Bx
y_base = z + (area_src^T x - area_tgt^T z) / (area_tgt^T 1)
```

Each learned stage receives the raw source field, the fixed base result, and
the current prefix. It predicts edge corrections `D`, projected so that

```text
sum_j D_ij = 0
sum_i area_target_i D_ij = 0
y_next = y_prefix + D(x) x_raw
```

Thus every prefix preserves constants and conservation. Stage routing can
return the prefix exactly. The bilinear training bands are literal spherical
harmonic degrees—low 1–16, mid 17–32, high 33–48—with degrees 49–64 reserved
as a guard band. No mesh-normalized frequency is used for training decisions.
The correction reference blends actual ESMF bilinear weights with uniform
k-distance support. Compact intrinsic geometry replaces redundant raw area,
distance, and candidate-count features.

Typical FV-line commands:

```bash
PYTHONPATH=_next python _next/scripts/build_fv.py --config _next/configs/progressive.json --pair PAIR --output fv.pt
PYTHONPATH=_next python _next/scripts/train.py --config _next/configs/progressive.json --device cuda
PYTHONPATH=_next python _next/scripts/train.py --config _next/configs/progressive.json --device cuda --resume
PYTHONPATH=_next python _next/scripts/audit.py --config CONFIG --checkpoint CHECKPOINT --device cuda
```

Typical bilinear-line commands:

```bash
./next baseline-check
PYTHONPATH=_next python _next/scripts/train.py \
  --config _next/configs/bilinear_progressive.json --stage low --device cuda
PYTHONPATH=_next python _next/scripts/audit.py \
  --config _next/configs/bilinear_progressive.json \
  --checkpoint _next/checkpoints/bilinear_progressive_low.pt --device cuda
```

Later stages require the preceding checkpoint and its passing audit. Training
uses pure harmonics, within-band mixtures, controlled target/guard mixtures,
explicit guards immediately above each band boundary, and analytic/real
safety anchors. Capability training learns the correction with its gate open
under a bounded safety allowance. Router training then freezes the correction
and learns where it is beneficial. Final selection restores strict safety
thresholds, and the unchanged prefix remains the identity floor.

Both schemas reject unknown fields, authenticate their inputs, save atomic
resumable checkpoints, and keep protected/external pairs behind an explicit
audit flag. Detailed workflow and PBS examples are in
`docs/ACTIVE_WORKFLOW.md`.

The verification command, run in the project PyTorch environment, is:

```bash
./next test
```

For use, refer to the repository-level `./next` command documented in
`docs/ACTIVE_WORKFLOW.md`.
