# Adaptive spectral stopping (Phase 1: field/mesh bandwidth diagnostic)

Goal: decide how many spherical-harmonic orders ("modes") are needed to represent a field on a
mesh to a user tolerance, so the iterative spectral corrector can **stop early** when a given
(mesh, field) does not need the highest bands. Tolerance is workflow-dependent — viz ≈ 1e-2,
AI ≈ 1e-8, simulation ≈ 1e-12…1e-16.

Script: `scripts/evaluate_adaptive_spectral_stopping.py` (CPU only, no model).

## What it computes

Project the field onto **orthonormal real spherical harmonics** by area-weighted quadrature on
the mesh (any mesh with cell centroids + areas — CS, ICOD, RLL, TRI, quads, HEALPix):

    a_lm = Σ_i area_i · f(x_i) · Y_lm(x_i)     (areas in steradians, Σ = 4π)
    C_l  = Σ_m a_lm²                            (angular power spectrum)
    P    = Σ_i area_i · f(x_i)²                 (total power)

From `C_l` it derives two stop criteria (one computation serves **all** tolerances at once):

- **Tail (truncation) error** — monotone in L, the actual error of stopping at order L, and the
  recommended criterion:  `E(L) = sqrt( (P − Σ_{l≤L} C_l) / P )`, `L* = smallest L with E(L) < tol`.
- **Cauchy increment** between successive probe levels (the mentor's "error between truncation
  levels") — non-monotone, so the stop requires it to hold for `--consecutive` (default 2) levels,
  which guards against parity notches.

## Robustness (the parts naive versions get wrong)

1. **Nyquist cap.** A mesh with N cells can't represent order ≫ √N; probe levels above ~√N are
   dropped with a warning (projecting beyond Nyquist aliases high frequencies downward).
2. **Achievable-error floor.** Quadrature on a finite mesh has its own error, so `E(L)` plateaus
   instead of reaching 0. A requested tolerance **below that floor cannot be certified** on this
   mesh — reported as `below_floor` rather than silently "met".
3. **Captured-energy fraction** `ΣC_l / P`: if < ~0.99 the field has content above `lmax`
   (under-resolved / aliasing) and the estimate is flagged as untrustworthy.

## Implementation note

Spherical harmonics are evaluated with a fully-normalized associated-Legendre **recurrence**
(Holmes & Featherstone 2002), fully vectorized over mesh points. This replaced per-`(l,m)`
`scipy.special.sph_harm_y` calls, which were compute-bound (~22 min at `lmax≈126` on a 16k-cell
mesh); the recurrence does the same run in **~6 s** and is stable to high degree. Normalization is
the orthonormal real-SH convention (∫Y²dΩ=1), verified end-to-end by the self-test.

## Validation

`python scripts/evaluate_adaptive_spectral_stopping.py --validate` builds pure `Y_l^m` fields on a
Fibonacci sphere; each must recover `L* = l` with energy concentrated at degree `l`.

    Y_2_1   L*=2   C_l/P=1.0000   PASS
    Y_5_-3  L*=5   C_l/P=1.0000   PASS
    Y_8_0   L*=8   C_l/P=1.0000   PASS
    Y_16_7  L*=16  C_l/P=1.0000   PASS
    Y_24_-11 L*=24 C_l/P=1.0000   PASS
    Y_32_5  L*=32  C_l/P=1.0000   PASS
    6/6 cases passed.

## Example (real mesh: RLL-r90-180 source, 16,200 cells, Nyquist ≈ 126)

| field | floor | L* @1e-2 | L* @1e-4 | L* @1e-8 | note |
|---|---|---|---|---|---|
| z | 0 | 16 | 16 | 16 | pure degree-1, exactly captured |
| x, y | 3.2e-3 | 16 | below_floor | below_floor | RLL pole-clustering quadrature floor |
| smooth1 | 0 | 16 | 32 | 32 | low bandwidth |
| smooth2 | 0 | 16 | 64 | 64 | slightly higher bandwidth |
| highfreq (Y_40) | 9.3e-3 | 64 | below_floor | below_floor | needs order ≥ 40 |

The `x/y` row is the floor guardrail doing its job: on this RLL mesh the quadrature cannot certify
field accuracy below ~3e-3, so tight tolerances are reported as `below_floor` rather than faked.
(The floor reflects the sampling actually provided — here the source nodes from the edge dataset;
using the full mesh `.nc` would change it.)

## Usage

    # self-test
    python scripts/evaluate_adaptive_spectral_stopping.py --validate

    # a real mesh from an edge dataset
    python scripts/evaluate_adaptive_spectral_stopping.py \
      --edge-parquet analysis_medium_improv/edge_dataset_<pair>_kdist_a2p0_mink8.parquet \
      --side source --tolerances 1e-2 1e-4 1e-8 --out <out>.csv

    # a synthetic equal-area mesh
    python scripts/evaluate_adaptive_spectral_stopping.py --synthetic-mesh 40000

## Scope / what this is and isn't

This is the **predictive** prior: it measures the *field's* spectral content (a lower bound on how
many correction bands you'll need), needs no model, and runs on analytic fields where the answer is
known. It is **not** yet wired to the operator.

## Phase 2 (not yet built)

- **Reactive stop:** run the actual base+corrector at increasing levels and apply the same Cauchy
  test to the *remapped output*. Note the **remap-error spectrum** concentrates at mesh-scale (high)
  frequencies regardless of field smoothness, so it can need more orders than the field alone.
- The current corrector only has bands `lmax = 8, 16, 24` (`lmax_denominator=32`); covering the full
  16…256 range requires retraining the corrector with more/higher bands.
- A least-squares projection option (better than quadrature for highly irregular/clustered meshes).
- Fast exact transforms on regular grids (pyshtools / healpy) as an optional accelerator.
