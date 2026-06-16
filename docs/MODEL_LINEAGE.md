# Model lineage

This document summarizes the evolution of the learned conservative remapping models.

## Goal

Approximate TempestRemap conservative remapping operators using a sparse learned GNN + Sinkhorn operator.

The learned model predicts positive scores on a fixed source-target candidate edge graph. A sparse Sinkhorn balancing step converts these scores into a conservative remapping mass matrix, preserving the remapping structure while enforcing conservation constraints.

## Main versions

| Version | Main idea | Outcome |
|---|---|---|
| v8 | Mean GNN baseline | Matched the sklearn baseline but did not clearly improve it. |
| v10 | Hybrid attention GNN | Improved multi-pair average and became the first strong neural baseline. |
| v11 | Gated hybrid attention | Slightly improved average accuracy and made attention more stable. |
| v15 | v11 + spherical harmonic loss up to l=16 | Introduced spectral supervision; modest improvement. |
| v16 | v11 + spherical harmonic loss up to l=24 | Best single-pass model; improved field and spectral errors. |
| v17 | More harmonic m-modes per degree | Improved some worst-case spectral errors, but slightly hurt average field accuracy. |
| v18 | IRNO-style iterative corrector from frozen v16 | Best overall result. Monotonic field and spectral improvement across correction steps. |
| v19 | Gentler v18 ablation with smaller update and stronger keep-close | Stable but did not beat v18 seed123. Useful ablation. |

## Current best model

`v18_irno_corrector_from_v16_l24_a2p0_mink8`

This model freezes the v16 conservative remapper and trains a shared conditional GNN corrector. The corrector is applied progressively with spectral-band conditioning:

1. base v16 operator
2. correction conditioned on lmax=8
3. correction conditioned on lmax=16
4. correction conditioned on lmax=24

After every correction, sparse Sinkhorn balancing is applied again to preserve conservative structure.

## Meaning of lmax

`lmax` is the maximum spherical harmonic degree included in that correction step.

A spherical harmonic mode is denoted `Y_lm`, where:

- `l` is the degree, corresponding to spatial frequency / scale.
- `m` is the order, corresponding to orientation or azimuthal structure within that degree.

The bands are cumulative:

- `lmax=8` means modes with degree `l <= 8`
- `lmax=16` means modes with degree `l <= 16`
- `lmax=24` means modes with degree `l <= 24`

