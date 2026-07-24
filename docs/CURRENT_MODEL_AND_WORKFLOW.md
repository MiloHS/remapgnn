# Current model and scientific workflow

## Purpose and status

This document explains the active clean remapping system in `_next/`. It describes what the model does,
which data it uses, how it is trained, how candidates are accepted or rejected,
and which verification and provenance guarantees are complete or still open.

Document was last edited **2026-07-24**.

It contains a finite-volume base, an accepted mid-band correction, and a
serialized high-band stage. The high-band stage was rejected by its identity
floor (`selected_identity=true`). Therefore the effective production output is:

```text
raw source field
       |
       v
frozen FV remapper
       |
       v
accepted mid-band correction
       |
       v
production output
```

The recently completed experiment
`_next/checkpoints/high_band_candidate_01.pt` was also rejected. Its corrector
showed some high-band capability, but the safely routed result did not achieve
eenough development gain.

The file `_next/checkpoints/progressive_next.pt` is a different, incomplete
older candidate. It must not be confused with either the approved production
checkpoint or `high_band_candidate_01.pt`.

## What problem the model solves

The model remaps a cell-average scalar field from one spherical grid to
another.

The grid names used in the active experiment are:

- **CS**: cubed sphere;
- **ICOD**: dual icosahedral grid;
- **ICO**: primal icosahedral grid;
- **RLL**: regular latitude-longitude grid;
- **HP**: HEALPix grid.

The input is one value per source cell. The output is one value per target
cell. The remapper is designed to preserve two physical properties:

1. **Consistency:** a constant source field remains constant.
2. **Conservation:** the area-weighted global integral is unchanged.

Accuracy is measured against known target-grid cell averages. The main error
metric is area-weighted relative L2 error:

```text
sqrt(
    sum_i target_area_i * (prediction_i - truth_i)^2
    ------------------------------------------------
    sum_i target_area_i * truth_i^2
)
```

We also compare against the second-order remapping operator **np2**.

## Architecture at a glance

The clean runtime is an ordered progressive model:

```text
geometry + frozen FV checkpoint
              |
              v
       pair-specific FV operator --------------------+
              |                                      |
raw field ----+--> FV output --> correction 1 --> correction 2 --> ...
     |                               ^              ^
     +-------------------------------+--------------+
```

Every correction stage receives:

- the original raw source field;
- the frozen FV result;
- the result produced by all earlier stages.

It predicts a correction to the edge weights applied to the original source
field.

The implementation is generic: `ProgressiveRemapper` owns an ordered list of
identical `ConservativeCorrectionStage` objects. “Mid band” and “high band”
are configurations of the same stage class.

## The frozen finite-volume base

### Inputs

For each source-target grid pair, the base reads:

- a sparse candidate-edge graph from
  `analysis_medium_improv/edge_dataset_*.parquet`;
- grid areas and centers from the corresponding conservative NetCDF map in
  `maps_medium_improv/`;
- the frozen geometry-network checkpoint
  `_next/checkpoints/fv_relax1.pt`.

The edge graph says which source cells may contribute to each target cell.
Its suffix is `kdist_a3p0_mink8` (source cells  within distance alpha to target cells with min of 8), which identifies the active neighbor-graph
construction.

### Geometry network

The frozen base network is a residual-gated bipartite graph network with:

- source-cell encoders;
- target-cell encoders;
- edge encoders;
- source-to-target messages and attention;
- target-to-source context;
- an edge decoder producing signed edge scores.

It uses source and target coordinates, cell areas and length scales, plus 27
edge geometry features such as distance, tangent displacement, area ratio,
candidate counts, and normalized neighbor rank.

FV retraining is not part of `_next`. The clean checkpoint records the frozen
network, feature normalization, build settings, and source hash.

### Turning network scores into an operator

For an edge from source cell `j` to target cell `i`, the builder first creates
a signed candidate mass. It then:

1. applies local degree-1 cell-average moment corrections;
2. applies local degree-2 cell-average moment corrections;
3. performs a joint marginal projection.

The final sparse operator satisfies, to numerical tolerance:

```text
sum_j weight_ij = 1
sum_i target_area_i * weight_ij = source_area_j
```

The first equation preserves constants. The second preserves the global
integral.

The geometry network is frozen during progressive training. A pair-specific
FV operator is rebuilt from its authenticated checkpoint and geometry.

## A conservative correction stage

Each learned correction stage has four parts.

### 1. Invariant field preparation

The source field is centered by its source-area mean and divided by its
source-area RMS.

The stage constructs:

- 13 fixed correction features;
- 8 global router features;
- 24 local router features;
- intrinsic geometry features.

Examples include local source variation, high-pass values, prefix-versus-FV
change, source-to-target disagreement, estimated gradient magnitude,
curvature, and normalized edge geometry. 

### 2. Edge corrector

Geometry and field features pass through small multilayer perceptrons. Messages
are aggregated per target cell, and an edge score is produced. The score is
centered within each target row so that it starts with zero row sum.

The current stage dimensions are:

| Component | Size |
|---|---:|
| correction hidden width | 48 |
| geometry hidden width | 32 |
| router hidden width | 32 |
| progressive edge features | 8 |

### 3. Conservative projection

The proposed edge correction is projected into the joint nullspace:

```text
sum_j correction_ij = 0
sum_i target_area_i * correction_ij = 0
```

Adding such a correction to a consistent, conservative prefix preserves both
properties. The active stages use 200 projection iterations.

### 4. Router

The router decides how much of the correction is used:

- a global field gate decides whether the field appears to need the stage;
- a local target-cell gate controls where the correction is active.

Training supports forced-open, forced-closed, soft, hard, and
straight-through modes. Deployment uses hard routing. If the global stage is
rejected, the code returns the previous prefix.

## Current stage order

The production checkpoint serializes:

| Stage | Intended normalized band | Status |
|---|---:|---|
| FV base | all fields | frozen base |
| `mid_band` | `(1.0, 1.25]` | accepted and active |
| `high_band` | `(1.25, 1.5]` | rejected; identity fallback |

Normalized frequency is based on
`K = sqrt(number_of_source_cells / 6)` and is recorded approximately as
`degree / K`.

The high-band candidate uses the same architecture as the mid-band stage. It
is a new ordered residual stage trained on top of a frozen FV + mid-band
prefix.

## Data used by progressive training

### Pair roles

The active configurations define four disjoint operational roles.

#### Training pairs

These update model parameters:

| Pair | Direction |
|---|---|
| `CS-r32_to_ICOD-r32` | coarse-to-fine |
| `ICOD-r32_to_CS-r32` | fine-to-coarse |
| `CS-r32_to_RLL-r90-180` | coarse-to-fine |
| `RLL-r90-180_to_CS-r32` | fine-to-coarse |
| `ICOD-r32_to_RLL-r90-180` | coarse-to-fine |

The trainer gives coarse-to-fine and fine-to-coarse regimes equal total
weight.

#### Development-selection pairs

These do not update parameters. They select the best epoch and determine
whether a candidate is admitted:

```text
CS-r64_to_ICOD-r64
ICO-r32_to_CS-r32
```

These deliberately test a higher resolution and a source mesh family not used
as an exact training pair.

#### Protected pairs

These are withheld until a candidate passes development:

```text
CS-r32_to_HP-n32
ICOD-r32_to_HP-n32
```

#### External-resolution pairs

These test resolution transfer after development promotion:

```text
CS-r128_to_ICOD-r128
ICOD-r128_to_CS-r128
```

Protected and external pairs require an explicit command-line flag. The
ordinary candidate audit cannot consume them accidentally.

### Field families

Panels are generated independently for each grid pair.

#### Cell-average spherical harmonics

Spherical harmonics provide fields with known degree and order. The code uses
cell quadrature to create source and target **cell averages**, rather than
only sampling cell centers. Harmonic orders are divided deterministically into
train, validation, and audit splits using source-grid keys.

Target harmonics lie in the stage's configured frequency band.

#### Balanced mixtures

Multiple harmonic modes are combined into mixtures. Components are balanced
using their source-area RMS so that one component does not dominate merely
because of scale.

#### Safety-frequency fields

The active safety levels are:

```text
0.25, 0.5, 0.75, 1.0, 1.125, 1.25, 1.75
```

They test whether a stage damages low frequencies, the previous stage's band,
the target-band boundary, or frequencies above the new target band. Safety
mixtures provide less regular combinations of these modes.

#### Smooth analytic fields

Two deterministic smooth functions of spherical position provide additional
low-frequency safety anchors.

#### Available real fields

The workflow requests five variables from `data/MIRA-Datasets/`:

```text
AnalyticalFun1
AnalyticalFun2
TotalPrecipWater
CloudFraction
Topography
```

When both endpoint files exist with the expected cell counts, these fields are
included as safety fields. They are available for all five training pairs and
for `CS-r64_to_ICOD-r64`. They are currently absent, and therefore omitted,
for `ICO-r32_to_CS-r32` and the HEALPix protected pairs.

### Geometry, reference maps, and np2

The three principal active data roots have different jobs:

| Location | Purpose |
|---|---|
| `analysis_medium_improv/` | sparse candidate graphs and geometry features |
| `maps_medium_improv/*_conserve.nc` | cell areas, centers, quadrature support, and ordering |
| `maps_medium_improv/*_conserve_np2.nc` | np2 comparison operator used in audits |
| `data/MIRA-Datasets/` | available paired real and analytic fields |

## Training workflow

Only one new stage is trained at a time. FV and all earlier stages remain
frozen.

### Step 1: initialize from an approved prefix

The configuration identifies:

- a clean production source checkpoint;
- the exact final prefix stage;
- the new stage to train;
- fresh or checkpoint initialization.

Every configured frozen-prefix stage must exactly match the checkpoint.
Structural dimensions must match when a stage is initialized from an existing
checkpoint. This prevents silently attaching a candidate to the wrong prefix.

### Step 2: capability phase

The new router is forced fully open. Only corrector parameters are trainable.
This asks:

> If this correction were always applied, can it improve the intended band
> without too much damage elsewhere?

Every batch contains target and safety fields. The loss combines:

- mean normalized target error;
- a worst-tail safety penalty versus the frozen prefix;
- a worst-tail safety penalty versus FV;
- an additional error-excess penalty;
- a small correction-size penalty.

Training panels change deterministically with pair and epoch. Development
panels remain fixed. Every four epochs in the active configurations, the
forced-open corrector is evaluated on the fixed selection pairs.

The best capability state is retained. It is admitted only if its selection
score improves on identity by more than the configured capability minimum
(currently 0.1%). If it fails, the entire new stage is rejected and the
previous prefix is restored.

### Step 3: router phase

If capability is admitted:

- the best corrector state is restored;
- all corrector parameters are frozen;
- only global and local router parameters are trained;
- straight-through routing is used during training;
- hard routing is used for selection and deployment.

The router loss includes the same target and safety objectives plus teaching
terms that currently label every target field “open” and every safety field
“closed.”

### Step 4: final identity floor

For each selection pair, the trainer measures:

- mean target error ratio versus the prefix;
- worst safety ratio versus the prefix;
- worst safety ratio versus FV;
- worst previous-band ratio versus the prefix.

The selection score is the worst pair's mean target ratio plus weighted
penalties for exceeding the safety limits. Lower is better:

```text
identity score = 1.0
```

The deployed routed candidate is admitted only when:

```text
candidate score < 1.0 - final_minimum_gain
```

The current final minimum gain is 2%, so the score must be better than
`0.98`. If the requirement is missed, the complete pre-training identity 
state is restored.

### Step 5: checkpoint and resume

Training writes:

- an atomic PyTorch checkpoint after every epoch;
- a CSV history after every epoch;
- model, optimizer, phase, epoch, best capability state, best router state,
  identity state, and provenance information.

The trainer checks that earlier stages and phase-frozen parameters remain
bit-identical across optimizer steps. Resume verifies selected external hashes
and reconstructs deterministic panels from phase, epoch, pair, and seed.

The standard PBS flow may use a second job with a dependency to resume after a
wall-time limit. A separate dependent audit job should run only after the
training job exits successfully.

## What an audit measures

### Field-by-field accuracy

For every audit pair and field, reports include:

- FV error;
- every progressive prefix error;
- final model error;
- np2 error;
- model-to-prefix, model-to-FV, and model-to-np2 ratios;
- field role, family, degree, normalized frequency, and source key;
- global and local gate values.

The auditor writes:

```text
detail CSV
summary CSV
JSON report with promotion decision and provenance
```

### Structural and mathematical checks

The audit also checks:

- constant-field preservation;
- positive affine behavior;
- negative-scale equivariance;
- tiny-scale behavior;
- rotation invariance;
- exact forced rejection to the previous prefix;
- correction row-sum residuals;
- area-weighted correction column residuals;
- runtime per field.

The configured correction tolerances are:

```text
maximum row residual:    1e-8
maximum column residual: 1e-10
```

### Scientific promotion gates

The configured full audit requires:

| Requirement | Threshold |
|---|---:|
| mean target gain over previous prefix | at least 3% |
| worst safety regression versus prefix | no more than 2% |
| worst prior-band regression | no more than 1% |
| worst safety regression versus FV | no more than 2% |

The intended order is:

```text
unit/CPU checks
    -> development selection and audit
    -> GPU numerical checks
    -> protected mesh-family audit
    -> external-resolution audit
    -> production promotion
```

## Current high-band experiment result

`high_band_candidate_01` used:

- the accepted FV + mid-band prefix;
- five r32/RLL training pairs;
- two fixed development-selection pairs;
- 80 capability epochs;
- 24 router epochs;
- a fresh high-band stage with correction scale 0.25.

Its best forced-open capability state was:

```text
epoch: 20
selection score: 0.9874117978844421
```

The best observed deployable router score in the history was approximately:

```text
epoch: 12
score: 0.9906964136168825
```

The router increasingly distinguished target from safety fields and recovered
much of the safety behavior, but the retained target improvement remained too
small. The completed checkpoint therefore records:

```text
completed: true
selected_identity: true
```

The candidate is useful experimental evidence, not a promoted model.

## Where to find each part

| Concern | Active location |
|---|---|
| root command | `next` |
| active workflow guide | `docs/ACTIVE_WORKFLOW.md` |
| default experiment | `_next/configs/progressive.json` |
| completed high-band experiment | `_next/configs/high_band_candidate_01.json` |
| production checkpoint | `_next/checkpoints/progressive.pt` |
| FV checkpoint | `_next/checkpoints/fv_relax1.pt` |
| progressive model | `_next/remapgnn_next/progressive.py` |
| FV construction | `_next/remapgnn_next/fv.py` |
| projections and constraints | `_next/remapgnn_next/constraints.py` |
| fields and panels | `_next/remapgnn_next/fields.py`, `_next/remapgnn_next/panels.py` |
| trainer | `_next/remapgnn_next/training.py` |
| auditor | `_next/remapgnn_next/evaluation.py` |
| checkpoint loading | `_next/remapgnn_next/checkpoint.py` |
| hashes/provenance | `_next/remapgnn_next/provenance.py` |
| command-line scripts | `_next/scripts/` |
| PBS entry points | `jobs_next_train.pbs`, `jobs_next_audit.pbs` |
| implementation audit | `docs/audits/implementation_2026-07-23/` |


