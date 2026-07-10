# Release Notes

## v1.0.0 — Deployable conservative remap model

First public model release: a supermesh-free, conservative and consistent learned
remapping operator for scalar fields on the sphere.

### Release assets
- `fv_gen_ALL_e400_s0.pt` — the deployable, max-coverage model (≈1.75 MiB), trained
  across all six mesh families (CS, ICOD, RLL, ICO, MPAS, HEALPix).
- `v20b_base_a3p0_mink8_geom_v12.json` — the paired config (also tracked in the repo at
  `configs/`).

### Highlights
- **Conservative & consistent to solver tolerance** (max residual ≈ 2×10⁻⁹) by
  construction, on any mesh pair.
- **Supermesh-free**: builds an operator from cell centers/areas + a kNN candidate graph
  — no TempestRemap/ESMF/overlap-mesh needed at inference. Runs on CPU or GPU.
- **Accuracy** (cell-average metric, per spectral band): beats 2nd-order TempestRemap
  (np2) at high wavenumbers in-distribution, and beats ESMF 2nd-order conservative at
  every band in both in-distribution and zero-shot regimes.
- **Generalizes across mesh topologies**: zero-shot error on held-out families improves
  with training-topology diversity.

### How to use
See [`docs/USAGE.md`](docs/USAGE.md) (quickstart) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).
Minimal path: build candidate graph → build operator → apply to field. Only the two
mesh files are required.

### Results
See [`docs/RESULTS.md`](docs/RESULTS.md) and the tables in [`paper_tables/`](paper_tables/).

### Notes
- Reported metrics are averaged over 3 seeds; 5-seed error bars, additional held-out
  families/resolutions, and cost + order-of-accuracy studies are in progress.
- The `build_remap_operator.py` default `--model` is an older development checkpoint;
  pass `--model models_medium_improv/fv_gen_ALL_e400_s0.pt` explicitly.
