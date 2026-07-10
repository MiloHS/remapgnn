# Model Card — RemapGNN (`fv_gen_ALL_e400_s0`)

A learned, **supermesh-free, conservative and consistent** remapping operator for
scalar fields on the sphere. Given two meshes, it predicts signed edge affinities on a
k-nearest-neighbor candidate graph and projects them onto the conservation/consistency
constraints, producing one sparse operator applicable to any field on that mesh pair.

## Summary

| | |
|---|---|
| **Artifact** | `fv_gen_ALL_e400_s0.pt` (≈1.75 MiB) + `configs/v20b_base_a3p0_mink8_geom_v12.json` |
| **Architecture** | Gated hybrid-attention bipartite GNN (`gated_hybrid_attention`), hidden=128, 1 message-passing round, ≈452k parameters, **signed** edge weights |
| **Inputs** | Source & target mesh cell centers + areas (2 NetCDF files) |
| **Output** | Sparse remap operator `S` (COO), applied as `y = S x` |
| **Candidate graph** | Distance-cutoff kNN, `alpha=3.0`, `min_k=8` (`kdist_a3p0_mink8`) |
| **Higher-order** | Finite-volume cell-average moment corrections (degree ≤ 2, local-soft) |
| **Precision** | float64 constraint projection at inference |
| **Framework** | PyTorch 2.6; runs on CPU or GPU |

## Guarantees (by construction)

- **Conservation** (source-marginal): area-weighted global integral is preserved.
- **Consistency** (target-marginal): a constant field maps to that constant exactly.
- Both hold to solver tolerance — measured **max residual ≈ 2×10⁻⁹** (mean ≈ 1×10⁻⁹)
  across all evaluated mesh pairs and families, independent of accuracy.

## Training data

Reference operators and fields from six mesh families spanning the topologies common in
Earth-system modeling:

- **CS** cubed-sphere, **ICOD** icosahedral hexagonal-dual, **RLL** regular
  latitude–longitude, **ICO** icosahedral-triangular, **MPAS** Voronoi, **HEALPix**.

Trained from scratch on **13 directed mesh pairs** across all six families (seed 0).
Teacher: second-order TempestRemap (np2). Field losses use **cell-average** truths
(analytic spherical harmonics up to degree 32, plus MIRA climate fields: total
precipitable water, cloud fraction, topography) — consistent with the finite-volume
evaluation metric. Model selected by a field-first validation criterion.

## Performance (headline; full tables in [`RESULTS.md`](RESULTS.md))

Cell-average area-relative L² error per spherical-harmonic degree band.

- **In-distribution** (meshes it was trained on): **beats np2 (2nd-order TempestRemap)
  for all bands ℓ ≥ 17** (down to ~0.58× np2 error) and **beats ESMF 2nd-order
  conservative at every band** (2–5×); np2 is only ahead at the largest scales
  (ℓ ≤ 8), where all methods are already < 0.1% error.
- **Zero-shot** (a mesh family never seen in training or model selection): still
  **beats ESMF 2nd-order conservative at every band** and matches/beats np2 at high
  wavenumbers.
- **Generalization scales with training diversity**: zero-shot error on a held-out
  family decreases monotonically as more *other* families are added to training.

> Note: current numbers are averaged over 3 seeds (5-seed error bars and additional
> held-out families/resolutions in progress). See [`RESULTS.md`](RESULTS.md).

## Intended use

- Amortized conservative remapping between spherical meshes, especially when many
  fields must be remapped for a given mesh pair, or when a GPU pipeline without
  supermesh construction is desirable.
- Research/experimental. This is a learned operator; validate conservation and accuracy
  for your meshes before relying on it in a coupled model.

## Limitations

- Accuracy is weakest (relative to np2) at the **largest scales (ℓ ≤ 8)**, though
  absolute error there is already very small.
- Higher-order moment corrections are **degree-2, local-soft** (satisfied
  approximately) → measured convergence order is between 1st and 2nd (≈1.6–1.8 on the
  cell-average metric), not a clean 2.0.
- Trained mostly at resolutions ≤ r64; behavior at much finer resolutions is
  extrapolation (evaluated at r128 as a resolution-transfer test).
- The candidate graph must contain the support the constraints need; the default
  `alpha=3.0, min_k=8` was validated to recover essentially all of the np2 support on
  the tested families.

## Reproducibility

- Config: `configs/v20b_base_a3p0_mink8_geom_v12.json`. The checkpoint embeds its
  feature schema, normalization stats, architecture, and training config, so it is
  self-contained for inference.
- See [`USAGE.md`](USAGE.md) for the end-to-end build+apply commands.

## License

See the repository `LICENSE` (if present). If no license file is included, the code is
under default copyright (all rights reserved) — contact the author for reuse terms.

## Citation

See `CITATION.cff` if present, or cite the accompanying paper:
*"A Graph Neural Network for Conservative and Consistent Remappings on the Sphere,"*
Milo Schlittgen-Li (Cornell University; Argonne National Laboratory).
