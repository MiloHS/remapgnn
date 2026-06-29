# High-Order, Conservation-Preserving Learned Remap Operators on the Sphere

**Status:** design proposal (2026-06-20). Supersedes the IRNO-corrector direction, which is closed as a
negative result (see "Why the corrector failed").

## 1. Motivation

Our current operator is a bipartite GNN that emits positive edge scores, Sinkhorn-balanced into a
sparse, **strictly non-negative** mass matrix satisfying conservation (source marginal) and consistency
(target marginal / rows sum to 1). It **matches first-order TempestRemap (TR) but cannot exceed it.**

Measured high-order headroom (this repo, `scripts/_eval_tr_order.py`, CS-r64↔ICOD-r64, error vs analytic
truth, np2/np1 ratio — lower = 2nd order beats 1st order):

| function | CS→ICOD | ICOD→CS |
|---|---|---|
| x (ℓ1) | 0.02 | 0.08 |
| Y_8_0 | 0.76 | 0.88 |
| Y_16_0 | 0.55 | 0.71 |
| Y_24_0 | 0.43 | 0.60 |
| Y_16_8 | 0.25 | 0.96 |
| Y_24_12 | **0.15** | **0.40** |

So a 2nd-order operator is **2–7× more accurate at high ℓ**. The generated np2 TR map has **negative
weights** (min −0.15) and is ~3× denser — empirical confirmation that the prize requires signed weights.

## 2. Why the corrector failed, and why first order is the ceiling

The corrector (iterative edge-reweighting on a frozen base) was a no-op at high ℓ in every regime
(full base, low-band base, Tempest-target, and analytic-truth-target). The reason is structural, not a
training problem:

> **Godunov barrier (operator form).** A *conservative + monotone* linear remap operator is at most
> first-order accurate. Higher order requires a **nonmonotone (signed-weight)** operator.
> (Ullrich & Taylor 2015; Ullrich, Devendran & Johansen 2016 — the TempestRemap theory: "arbitrary order
> of accuracy is supported for each of the described *nonmonotone* maps," monotonicity "(optionally)".)

A non-negative Sinkhorn operator is monotone → first order. No reweighting of it can add high-order
content. **The fix is to change the operator class, not to add a corrector.**

## 3. The design: learn the sub-cell reconstruction (signed, conservative by construction)

High-order conservative FV remap = reconstruct a sub-cell polynomial in each source cell, then integrate
it over each source/target overlap. Order comes from the **reconstruction**, not the weights
(Ullrich/Lauritzen/Jablonowski 2012; CSLAM/Lauritzen; incremental remap, Dukowicz & Baumgardner 2000).
So we learn the reconstruction.

### 3.1 Geometry (precomputed, one-time, like the candidate graph / TR overlap mesh)
For each source cell `j` (area `A_j`, centroid `c_j`) and target cell `i` (area `B_i`), and each overlap
`(i,j)`:
- `a_ij` = overlap area
- `p_ij` = overlap centroid, and `d_ij = p_ij − c_j` (offset from source centroid)

Exact identities from the overlaps tiling each cell:
- `Σ_i a_ij = A_j` (overlaps tile the source cell)
- `Σ_j a_ij = B_i` (overlaps tile the target cell)
- `Σ_i a_ij d_ij = 0` (area-weighted overlap centroids = source centroid ⇒ first moment vanishes per source cell)

### 3.2 Operator from a linear (2nd-order) reconstruction
Reconstruct `f(x) ≈ f_j + g_j·(x − c_j)` in source cell `j`, with **per-cell gradient** `g_j`. Mass into
target `i` from source `j`:

```
∫_overlap_ij [f_j + g_j·(x−c_j)] dx = a_ij f_j + g_j·(a_ij d_ij)
```

The gradient is a **linear, field-agnostic stencil over neighbors**: `g_j = Σ_k w_jk f_k`
(`w_jk` a tangent vector per neighbor; classical least-squares / Green–Gauss gives a closed form). The
resulting **field-agnostic** mass matrix entry (coefficient of source value `f_l`):

```
M_il = a_il  +  Σ_j a_ij (d_ij · w_jl)         ;   S_il = M_il / B_i
```

This is **signed** (via `w`) and **wider-stencil** (gradient pulls in neighbors-of-overlaps) — exactly the
np2-style operator, and it stays a single matrix applied to any field.

### 3.3 Conservation and consistency are (nearly) free — and LOCAL
- **Conservation** (source marginal), for *any* `w`:
  `Σ_i M_il = Σ_i a_il + Σ_j (Σ_i a_ij d_ij)·w_jl = A_l + Σ_j 0·w_jl = A_l`.  **Exact, structural.**
- **Consistency** (target marginal / rows sum to 1): holds **iff** `Σ_k w_jk = 0` per source cell
  (the gradient annihilates constants):
  `Σ_l M_il = B_i + Σ_j a_ij d_ij·(Σ_l w_jl) = B_i`.  A **local** constraint on each cell's stencil.
- **2nd-order accuracy**: holds iff the stencil reproduces linear fields:
  `Σ_k w_jk = 0` and `Σ_k w_jk (c_k − c_j)^T = I_tangent`.  Also **local** affine constraints.

So unlike the Sinkhorn operator (global balancing), here **both marginals + the target order are local
affine conditions on each cell's gradient stencil** — no global iterative solve needed.

### 3.4 What the GNN learns (Bar-Sinai-style null-space parameterization)
The two local affine constraints (consistency `Σw=0`, linear-reproduction `Σ w (c_k−c_j)^T = I`) define an
affine set for each cell's stencil `{w_jk}`. Following **Bar-Sinai et al. 2019 (PNAS, "data-driven
discretization")**: write `w = w_particular + N·θ`, where `w_particular` is any solution (e.g. the LSQ
gradient), `N` spans the **null-space** of the constraint (Vandermonde) system, and the **GNN predicts the
free coefficients `θ`** from local features. Then:
- conservation: structural (always);
- consistency + 2nd-order accuracy: **guaranteed by construction** (we never leave the affine set);
- the network only learns the remaining degrees of freedom (which control higher-order behavior /
  conditioning / robustness).

This is strictly better than "init from LSQ, learn corrections": the order/constraints can't be violated
by training.

### 3.5 Stability / monotonicity (the accuracy↔stability knob)
The unlimited operator is **linear, signed, 2nd-order, nonmonotone** — it can overshoot (smaller stability
region; the mentor's point). This is the same class as TR's np2 map, and it keeps the
**one-matrix-for-all-fields** property (so the GPU/throughput story is intact). Two options if
monotonicity is required (e.g. positive tracers):
- **Learned troubled-cell limiter** (Physica Scripta 2024; learned ENO/WENO as *stencil selection /
  classification* + a hybrid limiter that switches between an accurate smooth-region model and a
  non-oscillatory one — recovers 3rd order while staying non-oscillatory). The GNN predicts a per-cell
  limiter `φ_j ∈ [0,1]` scaling `g_j`.
- **Caveat:** any limiter makes the reconstruction *field-dependent* → the operator becomes **nonlinear**
  (applied per field, not a precomputed matrix). This is inherent to monotone high-order schemes. So:
  - *Primary target:* unlimited linear signed 2nd-order matrix (matches np2, matrix property kept).
  - *Optional:* learned limiter for monotone applications, accepting field-dependence.

## 4. Training & evaluation
- **Target:** analytic ground truth on the target grid (harness already built:
  `build_harmonic_fields_with_truth`, `harmonic_target=truth`). The operator should reproduce the true
  remap of each test field.
- **Benchmark:** the generated **np2 TR maps** (`map_*_conserve_np2.nc`) — the measured 2–7× headroom is
  the target. Success = approach np2 accuracy at high ℓ while *keeping* the low-ℓ advantage we already
  have over np1.
- **Metrics:** per-degree `area_rel_l2` vs truth (zonal + tesseral), plus conservation residual
  (should be ~machine-zero by construction) and an overshoot/monotonicity diagnostic.
- **Sanity:** with `θ=0` (pure LSQ gradient) the operator should already be ~2nd order — a floor the GNN
  must not fall below.

## 5. Implementation plan (phased)
1. **Geometry extraction:** add overlap centroids `p_ij` (→ `d_ij`) to the edge-dataset builder (the TR
   overlap mesh `ov_*.nc` has the overlap polygons; compute area-weighted centroids). Cheap, one-time.
2. **Static 2nd-order baseline (no learning):** assemble `M_il = a_il + Σ_j a_ij(d_ij·w_jl)` with the
   classical **LSQ gradient** `w`. Verify it reproduces ~np2 accuracy + machine-zero conservation. This
   validates the whole operator-assembly path before any training.
3. **Learnable head:** GNN predicts `θ` (null-space coeffs of the per-cell gradient stencil); train vs
   analytic truth; compare to np2 + the static LSQ baseline.
4. **(Optional) limiter:** learned per-cell `φ_j` for monotone variants; measure accuracy↔monotonicity.

## 6. Alternatives (kept as fallbacks)
- **Design 2 — signed weights + global constraint projection.** If the moment formulation is too
  restrictive, let the GNN emit signed edge weights and enforce both marginals via a **differentiable
  projection** `x' = x − Aᵀ(AAᵀ)⁻¹(Ax−b)` (KKT-hPINN, Chen et al. 2024) or **completion** (DC3, Donti et
  al. 2021). More general, less structured; doesn't *guarantee* order. Design 1's local constraints are
  preferable when applicable.
- **Design 3 — learned dual potentials (throughput, only if Sinkhorn is retained).** Predict per-node
  potentials → near-zero-iteration balancing (amortized Sinkhorn; Thornton & Cuturi 2023). Moot under
  Design 1 (no Sinkhorn).
- **Design 4 — geometry-free remap (moat).** Where exact cell polygons are unavailable (scattered/
  observational data, ML-emulator latent grids), use a moving-least-squares backbone + GNN + the Design-2
  projection with density/Voronoi-estimated areas. Most speculative; conservation must be redefined
  without overlap areas (open question).

## 7. Novelty
All enabling ML machinery exists only in **1D / regular-grid** form (learned discretization, learned
ENO/WENO, DC3/KKT projection). None has been applied to **conservation- and consistency-preserving
high-order remap on the sphere with doubly-constrained marginals.** That gap is the contribution: *the
first high-order, conservative, consistent, learned spherical remap operator.*

## 8. Risks / open questions
- Overlap-centroid extraction accuracy (exact from `ov_*.nc`; a cell-center approximation `d_ij ≈
  weighted (c_i^tgt − c_j^src)` is cheaper but lower-order — measure the hit).
- Limiter ⇒ field-dependence ⇒ loss of the precomputed-matrix property; quantify the accuracy/stability/
  reusability trilemma.
- Does the GNN's learned `θ` actually beat the classical LSQ gradient, or is the static 2nd-order operator
  already near-optimal? (If the latter, the "learned" part may be unnecessary — itself a clean finding.)
- Higher than 2nd order (quadratic reconstruction, curved overlaps + quadrature) for further high-ℓ gains.

## 9. Key references
- **Classical / anchor:** Ullrich & Taylor 2015 (MWR, linear-maps theory); Ullrich, Devendran & Johansen
  2016 (MWR, Part II); Ullrich, Lauritzen & Jablonowski 2012 (IJNMF, order from reconstruction);
  Lauritzen et al. 2010/2012 (CSLAM); Dukowicz & Baumgardner 2000 (incremental remap).
- **Learned high-order discretization:** Bar-Sinai, Hoyer, Hickey & Brenner 2019 (PNAS); de Romemont et
  al. 2024 (arXiv:2412.07541, flux-limited FV); (U)NFV 2025 (arXiv:2505.23702, conservation-preserving
  extended stencils — cite for architecture only; its "10× beats Godunov" claim is unverified).
- **Stability / limiters:** learned ENO/WENO + hybrid limiter, Physica Scripta 2024 (10.1088/1402-4896/ad7f97).
- **Hard constraints:** DC3 (Donti, Rolnick & Kolter 2021, arXiv:2104.12225); KKT-hPINN (Chen et al. 2024,
  arXiv:2402.07251); OptNet (Amos & Kolter 2017); Beucler et al. 2021 (PRL, constraint-aware NNs).
- **Amortized OT / learned duals:** Thornton & Cuturi 2023 and related (arXiv:2206.05262, 2212.00133).
