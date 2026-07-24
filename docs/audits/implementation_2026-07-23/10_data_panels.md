# Data loading and panel audit

## Scope and audit target

This report audits commit
`f98a0e8ed06678099a98b5a6dfc91ded5d580c97` for active, non-archived:

- edge, map, np2, and real-field input discovery;
- map quadrature and field loading paths;
- harmonic, mixture, analytic, and real-field generation;
- roles, families, labels, masks, frequencies, and source keys;
- train, selection, audit, protected, and external-resolution separation;
- panel balancing, normalization, deterministic generation, and provenance.

No implementation, configuration, test, checkpoint, data, map, job, or Git
state was modified. The only file written is this report. One attempted
production-resolution CPU panel build was stopped without output after it
proved too expensive for the initial audit; the checks below use static
derivation, active-file metadata, and an isolated synthetic CPU panel.

## Implementation path traced

The complete active path is:

1. `_next/configs/high_band_candidate_01.json:5-44` defines pair roles, data
   roots, frequency convention, panel sizes, safety levels, and real fields.
2. `_next/scripts/train.py:51-61` builds train and selection pairs from the
   configured edge Parquet and conservative-map NetCDF paths.
3. `_next/scripts/audit.py:46-58` builds requested audit pairs and protects
   protected/external pair names behind `--allow-protected`.
4. `_next/remapgnn_next/fv.py:201-285` loads geometry, checks edge/map center
   ordering, constructs FV and panel quadratures, builds smoothers, and stores
   quadratures and the source-grid key in `PairData.metadata`.
5. `_next/remapgnn_next/panels.py:46-173` selects target and guard degrees,
   generates harmonic and mixture panels, adds analytic and available real
   safety fields, assigns roles/families/masks, and rejects duplicate keys
   within one pair panel.
6. `_next/remapgnn_next/fields.py:59-149` reads map cell geometry and constructs
   quadrature; lines 152-260 generate deterministic source-keyed harmonic
   modes; lines 263-296 make mixtures; lines 299-384 generate analytic fields
   and load available paired real fields.
7. `_next/remapgnn_next/training.py:276-445` fixes selection panels at
   validation split/epoch zero, rebuilds deterministic train panels by phase,
   epoch, and pair, stratifies target/safety batches, and authenticates a
   subset of inputs.
8. `_next/remapgnn_next/evaluation.py:225-246` builds audit-split panels, loads
   np2 operators, and writes audit details and input hashes.

## Active input inventory

Parquet metadata inspection found every configured pair's active edge file and
both conservative and np2 map files. Source-index maxima imply these active
source counts:

| Role | Pair | Source cells | Edge rows |
|---|---|---:|---:|
| train | CS-r32_to_ICOD-r32 | 6,144 | 290,032 |
| train | ICOD-r32_to_CS-r32 | 10,242 | 290,032 |
| train | CS-r32_to_RLL-r90-180 | 6,144 | 371,728 |
| train | RLL-r90-180_to_CS-r32 | 16,200 | 366,992 |
| train | ICOD-r32_to_RLL-r90-180 | 10,242 | 459,855 |
| selection | CS-r64_to_ICOD-r64 | 24,576 | 1,159,988 |
| selection | ICO-r32_to_CS-r32 | 20,480 | 441,111 |
| protected | CS-r32_to_HP-n32 | 6,144 | 322,384 |
| protected | ICOD-r32_to_HP-n32 | 10,242 | 404,695 |
| external | CS-r128_to_ICOD-r128 | 98,304 | 4,640,212 |
| external | ICOD-r128_to_CS-r128 | 163,842 | 4,640,212 |

All five configured variables are present with the expected cell count in
both endpoint files for all five training pairs, CS-r64_to_ICOD-r64, and both
external-resolution pairs. Thus these real files are actual inputs, not merely
configured possibilities. ICO and HP files are absent, so real fields are
silently omitted for ICO-r32_to_CS-r32 and the two HP protected pairs.

## Findings

### DATA-01 — High — Target degrees can exceed the configured upper band

**Requirement or claim.** A stage band is open at its lower boundary and
closed at its upper boundary: a target harmonic must satisfy

`band_lower < degree / K <= band_upper`.

The same interpretation appears in `_next/remapgnn_next/panels.py:60-63`,
`_next/remapgnn_next/training.py:183`, and
`_next/remapgnn_next/evaluation.py:143`.

**Expected behavior.** Integer target degrees should run from
`floor(band_lower*K)+1` through `floor(band_upper*K)`.

**Observed behavior.** `_next/remapgnn_next/panels.py:46-49` computes the last
degree using `round(band_upper*K)`. When the fractional part is at least one
half, this includes a degree whose normalized frequency is above the configured
upper limit. Independent calculations from active Parquet source-index
metadata give:

| Active source | K = sqrt(n/6) | Code's last degree | Last frequency | Correct last degree |
|---|---:|---:|---:|---:|
| ICOD-r32 | 41.315856520 | 62 | 1.500634507 | 61 |
| RLL-r90-180 | 51.961524227 | 78 | 1.501110700 | 77 |
| ICO-r32 | 58.423739467 | 88 | 1.506237033 | 87 |
| ICOD-r128 | 165.248298025 | 248 | 1.500771887 | 247 |

This affects three training mappings, one selection mapping, one protected
mapping, and one external-resolution mapping. Maximum-degree subsampling means
the extra degree is not necessarily selected every epoch, but it is in the
target pool and can be selected and labelled `target`.

**Reproduction.**

```bash
PYTHONPATH=_next python - <<'PY'
from math import sqrt, floor
from remapgnn_next.panels import band_degrees
for n in (10242, 16200, 20480, 163842):
    k = sqrt(n / 6)
    ds = band_degrees(k, 1.25, 1.5)
    print(n, ds[-1], ds[-1] / k, floor(1.5 * k))
PY
```

The reported values were also independently reproduced by the coordinating
agent.

**Impact.** Training and selection do not implement the configured scientific
band exactly. ICO's admitted field is about 0.416% beyond the upper frequency
boundary, and target metrics can include out-of-band fields.

**Recommended correction.** Use `floor(upper*K)` for the inclusive upper
boundary, validate that a configured band has at least one realizable degree,
and add below/exactly-at/above boundary tests for every active mesh family.

**Confidence.** High.

### DATA-02 — High — Consumed real-field data is not authenticated

**Requirement or claim.** `_next/README.md:44-46` says training authenticates
all inputs. The audit contract also requires rejection after changing each
authenticated input category.

**Expected behavior.** Every real-field file whose contents can affect a
panel, loss, selection score, or audit report should be represented by path,
content hash, variable manifest, and availability in training resume and audit
provenance.

**Observed behavior.**

- `_next/remapgnn_next/fields.py:347-384` reads configured variables from real
  NetCDF files.
- `_next/remapgnn_next/panels.py:166-172` adds them as safety fields whenever
  both endpoint files exist.
- Active-file inspection confirms five real fields are consumed for every
  training pair and for CS-r64_to_ICOD-r64.
- `_next/remapgnn_next/training.py:236-248` authenticates implementation
  modules, config, edge/map files, and the source checkpoint, but not real
  fields.
- `_next/remapgnn_next/evaluation.py:240-245` likewise records checkpoint,
  config, edge, and conservative-map hashes, but no real-field hashes or
  inclusion manifest.

Static comparison of `_auth()` before and after a hypothetical real-file
content change is sufficient: no term in the returned mapping depends on those
files, while subsequent `build_panel()` output does.

**Impact.** A real-field file can change between epochs or before resume and
alter scientific training data without causing resume rejection. Audit results
using real fields cannot be tied to their exact input bytes, and silent
availability differences change panel composition without appearing in the
report.

**Recommended correction.** Resolve the per-pair real paths before training,
record both endpoint hashes plus included/skipped variable names and reasons,
include this manifest in resume authentication, and include it in audit
provenance. Open or hash a stable snapshot rather than re-discovering mutable
files each epoch.

**Confidence.** High.

### DATA-03 — Medium — Configurable frequency divisor is only partly honored

**Requirement or claim.** `PanelConfig.frequency_cells_per_k_squared` is a
scientific configuration field and should define the normalized-frequency
convention consistently.

**Expected behavior.** Changing the divisor should change both degree
selection and stored `frequency=degree/K` metadata with the same K.

**Observed behavior.**

- `_next/remapgnn_next/panels.py:138-142` uses the configured divisor to select
  degrees.
- `_next/remapgnn_next/fields.py:235,254` independently hard-codes
  `K=sqrt(n_source/6)` when storing frequency.

For example, with 6,144 source cells and a perturbed divisor of 24, panel
selection uses K=16 while a generated degree's metadata uses K=32. A field
selected for `(1.25,1.5]` can therefore be reported near `(0.625,0.75]` and
misclassified by prior-band logic.

**Impact.** Both active configs use 6, so the current candidate is internally
aligned. A future otherwise-valid config change would make target masks,
frequency metadata, and frequency-derived audit masks disagree.

**Recommended correction.** Calculate K once and pass it explicitly into field
generation, or remove the field from the public schema and validate that the
only supported value is 6. Add a perturbation test.

**Confidence.** High.

### DATA-04 — Medium — Full-panel source identities are not split-disjoint

**Requirement or claim.** The clean design calls for source-keyed
train/validation/audit splits and provides `assert_split_disjoint()` at
`_next/remapgnn_next/panels.py:37-43`.

**Expected behavior.** Either complete panels should have disjoint semantic
source identities across splits, or intentionally shared safety anchors should
be explicitly exempted and identified separately from split-controlled data.

**Observed behavior.**

- Harmonic orders are correctly partitioned by source mesh and degree in
  `_next/remapgnn_next/fields.py:157-181`.
- Analytic keys are fixed (`analytic:smooth1`, `analytic:smooth2`) for every
  split (`fields.py:324-328`).
- Real keys are fixed as `real:{name}` (`fields.py:377-384`).
- Mixture keys encode only seed and local index, not source mesh, split,
  component mode keys, or pair (`fields.py:291`).
- `assert_split_disjoint()` is not called by training or audit. `make_panel()`
  checks duplicates only within a single pair panel.

An isolated deterministic CPU panel built for train, validation, and audit
splits found exactly the two analytic keys in every pairwise split
intersection. For an active pair with available real data, five fixed real
keys are also shared. Selection panels for different pairs can report identical
mixture and real `source_key` strings for different tensors.

**Impact.** The existing disjointness helper cannot validate complete
production panels, and report source keys are not globally meaningful
identifiers. Shared safety anchors may be intentional, but the implementation
does not encode or document that exception, so leakage checks and provenance
claims are ambiguous.

**Recommended correction.** Define two explicit concepts: semantic field
identity and split membership. Include source mesh/pair and mixture component
identities in semantic keys. Mark intentionally shared analytic/real safety
anchors as shared rather than presenting all keys as split-controlled, and
enforce disjointness for the classes that are required to be held out.

**Confidence.** High for the behavior; medium for scientific severity because
shared safety anchors may be intended.

### DATA-05 — Medium — np2 audit inputs are omitted from report hashes

**Requirement or claim.** Audit provenance should identify every scientific
input used to calculate its reported comparisons.

**Expected behavior.** The exact `map_*_conserve_np2.nc` file used for each
pair should be hashed in the audit report.

**Observed behavior.** `_next/remapgnn_next/evaluation.py:233-235` loads np2
and writes `model_over_np2` and `np2_rel_l2`, but lines 240-245 hash only the
edge and ordinary conservative-map files. All eleven configured active pairs
currently have np2 files.

**Impact.** np2 does not enter the present promotion gates, so this does not
change pass/fail. It does make the detailed and summary np2 comparisons
non-reproducible from the report's recorded hashes.

**Recommended correction.** Add the resolved np2 path and SHA-256 to each
pair's `audit_data_sha256` entry.

**Confidence.** High.

## Checks that materially increased confidence

### Harmonic split behavior

Static derivation and CPU checks confirm:

- split RNG seed depends on source mesh and degree, not target mesh;
- train, validation, and audit order sets are mutually disjoint for degrees
  with at least three orders and their union contains every order;
- `val` aliases `validation`, and `test` aliases `audit`;
- pair/epoch sampling shuffles only within the already separated order set;
- harmonic source keys identify source mesh, degree, and order.

This is an architectural guarantee for harmonics, subject to the explicitly
different treatment of analytic/real anchors described in DATA-04.

### Lower safety boundary

For ICOD-r32, K=41.315856520. The high-band first target degree is 52 and the
configured 1.25 safety guard is clamped to degree 51, whose frequency is below
1.25. The lower boundary therefore does not collide with the target pool.
`safety_degree()` also rejects safety levels strictly inside the target band.

### Roles, masks, normalization, and determinism

An isolated CPU panel with 600 source and 500 target cells exercised harmonic,
mixture, guard, and analytic generation:

- repeated construction with identical config/split/epoch was bitwise
  identical for source tensor, truth tensor, labels, and keys;
- every within-panel source key was unique;
- `target_mask` exactly matched `role == "target"`;
- source area-weighted RMS differed from one by at most
  `1.1920929e-7` in float32;
- analytic source means after centering were at most `5.59e-9`;
- requested role/family assignments were preserved.

The normalization result is numerical verification for the isolated panel,
while deterministic construction follows architecturally from explicit stable
seeds and deterministic field formulas.

### Active role separation and regime balancing

`ExperimentConfig.__post_init__()` rejects exact pair-name overlap among train,
selection, protected, and external-resolution roles. The active configuration
passes. Active training contains three coarse-to-fine and two fine-to-coarse
pairs. `pair_weights()` therefore gives each regime total weight 0.5 (each
coarse-to-fine pair 1/6 and each fine-to-coarse pair 1/4). This is an
architectural per-optimizer-step regime balance, not proof of equal independent
field information.

### Active data presence and dimensions

All configured edge, conservative map, and np2 paths exist. All real-field
variables actually included in the train and CS-r64 selection panels have
lengths matching the relevant endpoint cell counts. Pair construction also
checks map center ordering against edge-table centers at a `1e-6` maximum
coordinate tolerance (`fields.py:129-137`).

## Test coverage observations

- Existing tests cover source-keyed harmonic partitioning and the corrected
  lower safety boundary.
- No existing test checks the inclusive upper boundary using non-integral K;
  the lower-bound test obtains the target list but asserts only its first
  element.
- Existing tests do not authenticate real inputs, validate an availability
  manifest, perturb `frequency_cells_per_k_squared`, or require complete-panel
  split semantics.
- `_next/tests/test_panels.py` uses pytest-style free functions, whereas the
  documented command is `unittest discover`; those functions are not
  `unittest.TestCase` methods. Some panel coverage is duplicated in
  `test_workflow_unittest.py`, but this discovery mismatch should be handled by
  the test-coverage specialist.

## Deferred checks

- Candidate-checkpoint acceptance and all GPU checks were deferred by audit
  instruction while training is active.
- Full r128 panel construction and full-mesh execution were deferred as
  expensive.
- A production-resolution r32 CPU pair/panel build was stopped after the
  geometry/FV/quadrature build exceeded the affordable initial-audit window;
  it produced no result and made no writes.
- Quadrature convergence (resolution 8 against a higher independent
  resolution) for high-degree cell averages remains unverified.
- Scientific correspondence of same-named real variables across source and
  target NetCDFs was not independently established; only path, variable
  presence, and dimensions were checked.
- Actual epoch-by-epoch mixture allocation and degree-rotation coverage across
  the full 80-epoch candidate run were derived from code but not replayed over
  production quadratures.

## Remaining uncertainty

The most important unresolved data question is quadrature accuracy at the
highest normalized frequencies and external resolution. The current code
constructs deterministic cell averages, but this initial audit does not
establish an independent error bound for resolution-8 quadrature. It also does
not establish whether sharing analytic/real safety anchors across phases was an
intentional scientific policy; the implementation and documentation need to
make that policy explicit regardless.

## Commands used

Representative read-only commands:

```bash
git rev-parse HEAD
rg -n "build_panel|source_key|split|real_field|provenance" _next docs next
find analysis_medium_improv maps_medium_improv data/MIRA-Datasets -type f
```

Active Parquet metadata and band-boundary calculation:

```bash
/gpfs/fs1/home/mschlittgenli/.conda/envs/remap_gpu/bin/python - <<'PY'
import json, math
from pathlib import Path
import pyarrow.parquet as pq
cfg=json.load(open("_next/configs/high_band_candidate_01.json"))
for role, pairs in cfg["pair_roles"].items():
    for pair in pairs:
        path=Path(cfg["paths"]["analysis"]) / (
            f"edge_dataset_{pair}_{cfg['paths']['graph_suffix']}.parquet")
        table=pq.read_table(path, columns=["source_index"])
        n=int(table["source_index"].to_numpy().max())+1
        k=math.sqrt(n/cfg["panel"]["frequency_cells_per_k_squared"])
        print(role, pair, n, round(1.5*k), round(1.5*k)/k,
              math.floor(1.5*k))
PY
```

Real-field inventory used `xarray.open_dataset()` read-only to enumerate the
configured variable names and sizes at paths resolved by
`PathsConfig.real_field_paths()`. The synthetic panel command constructed
temporary in-memory quadratures and `PairData`; it wrote no files.
