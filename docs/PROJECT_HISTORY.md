# RemapGNN: project history and current direction

## Current status

The project has a reliable base remapper and is testing whether a small
neural add-on can safely improve it.

- The base is a frozen finite-volume (FV) map.  It uses only the two meshes,
  so one map can be built once and then applied to any scalar field (linear).
- The approved clean model keeps the base and the accepted **mid-band**
  correction. Its high-band correction is deliberately turned off: when that
  correction is not trusted, the model returns the earlier result exactly.
- The previous high-band attempt did not meet the required accuracy and safety
  checks. That is a useful result, not a broken deployment: the fallback makes
  the approved model safe.
- The current experiment is
  [`high_band_candidate_01.json`](../_next/configs/high_band_candidate_01.json).
  It retrains only a fresh high-band correction, with more examples in the
  band we want to improve. The base and accepted earlier stage stay frozen.

We already have a dependable remapper. We are trying to add detail for rougher 
fields without making ordinary or smooth fields worse.

## What the model is made of

```text
source field
    |
    v
frozen FV base map  ----> dependable result for every field
    |
    v
accepted mid-band correction (optional improvement)
    |
    v
high-band correction (being researched; can close completely)
    |
    v
remapped field
```

The FV base is a sparse table of source-to-target weights. It was built by a
frozen geometry network and then adjusted so that it preserves a constant
field and the global total of the field. Those are essential physical checks
for remapping climate data.

Each correction stage looks at the input field and decides both how much of a
correction to propose and whether to trust it. A global decision can reject a
stage altogether; local decisions can reduce it in individual regions. The
correction is constrained so it does not undo the base model's constant-field
and global-total properties.

The correction is more expensive than applying the base map, because it must
be evaluated for every field. That is why it must show a clear benefit before
it becomes part of the approved model.

## A short history

| Period | What was tried | What we learned |
|---|---|---|
| Early work | Nonnegative learned maps | They could be conservative, but could not represent enough detail for a strong higher-order method. |
| Signed FV maps | Signed weights plus a final constraint step | This became the useful fixed-map approach: it can retain more detail while keeping the important physical totals. |
| Broad FV training | Training the fixed map on several mesh families | Variety of mesh shapes was important for a map that works on unfamiliar meshes. |
| v22--v23 | Large field-dependent remappers | They could help on familiar cases but did not transfer safely to different meshes and often damaged low-frequency fields. |
| v24A--v24C | Small add-ons on top of the frozen FV base | A small add-on can improve some rough fields. The gains were narrower than needed, and repeated correction was too costly and did not transfer reliably. |
| v24D | A newly learned low-band base followed by frozen stages | Replacing the strong FV base was a mistake: the learned base itself did not transfer well. |
| v24E--v24F | Frozen FV base, then carefully gated correction stages | The mid-band result was accepted. The first high-band result was rejected because its improvement was not large or safe enough. |
| Current `_next` work | Clean, isolated implementation and a new high-band training recipe | The clean version matches the accepted previous model. We can now improve one stage at a time without changing the reliable parts. |

## What has been shown, and what has not

### Solid conclusions

- Keep the fixed FV base as the default and safety net.
- Train a later correction only after earlier parts have been accepted and
  frozen.
- Test improvement on at least two different development mesh situations, not
  just the one used for training.
- Check ordinary fields, smooth fields, and available real fields separately
  from the rough fields a correction is meant to improve.
- Treat runtime as part of the decision: a field-dependent correction must be
  worth its additional cost.

## How a new candidate is judged

A candidate is first tested on development pairs that were held out from the
training examples. It must:

1. improve the intended rough-field band by at least 3% on both development
   mesh situations;
2. avoid more than a 2% regression on the safety fields;
3. preserve constant fields and global totals to tight numerical tolerances;
4. pass a separate protected/external check only after it succeeds in
   development.

Failing any of these tests means the new stage is not promoted. The earlier
approved prefix remains available unchanged.

## What was verified during the clean transition

The active implementation was rebuilt under `_next/` without importing the
old runtime code. The one-time conversion of the accepted model was checked on
several mesh types and resolutions. The sparse FV map, converted model output,
field panels, and audit values matched the previous implementation within the
specified numerical tolerances. The recorded result is in
[`_next/reports/equivalence_completed_v24f.json`](../_next/reports/equivalence_completed_v24f.json).

## High level locations

- Day-to-day commands and the safe order of work:
  [`docs/ACTIVE_WORKFLOW.md`](ACTIVE_WORKFLOW.md)
- Clean package layout and design notes: [`_next/README.md`](../_next/README.md)
- Current training settings:
  [`_next/configs/progressive.json`](../_next/configs/progressive.json)
- Current high-band experiment:
  [`_next/configs/high_band_candidate_01.json`](../_next/configs/high_band_candidate_01.json)
- The old-code archive is local.
