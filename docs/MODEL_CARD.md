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
