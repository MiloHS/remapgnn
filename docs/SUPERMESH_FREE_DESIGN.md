# Supermesh-Free Learned Remap Operator (the deliverable)

**Goal (reframed):** *match* TempestRemap accuracy — including its **2nd-order** accuracy — while winning on
**efficiency / another advantage**, NOT beat it. The advantage is being **supermesh-free**: produce the
operator from cheap inputs (cell centroids/areas + a kNN candidate graph) with **no overlap-mesh
computation**, GPU-fast, generalizable across mesh pairs, and differentiable. Classical 2nd-order remap
(incl. our validated moment-form operator) *needs the supermesh*; this learns to do without it.

Grounded in deep-research (`wzw5mm1uz`): no published learned GNN does conservative + consistent +
higher-order together (genuine gap), and nobody has quantified learning's value vs classical for this task.

## 1. Why learning is required here
Without the supermesh we have no exact overlap areas `a_ij` / centroids `d_ij`, so the classical moment
form cannot be assembled. We must **learn** the operator from local geometry. The non-negative Sinkhorn
operator is capped at first order (Godunov), so higher order requires **signed** weights — which Sinkhorn
cannot balance. The fix: signed weights + a **doubly-constrained linear projection** (not Sinkhorn).

## 2. Inputs (all cheap, no supermesh)
- **Candidate bipartite graph** `E`: for each target cell `i`, its candidate source cells `j` (kNN /
  distance-cutoff by centroid). One-time, GPU-able. *Must be generous enough to contain the 2nd-order
  stencil* (wider than 1st order) — validate recall against the np2 support.
- **Node features:** source/target centroids (xyz), cell areas `A_j`, `B_i`, local cell size / resolution.
- **Edge features:** relative position `c_i − c_j`, distance, area ratio, local anisotropy.

## 3. GNN (encode–process–decode, bipartite)
Standard mesh-GNN backbone (MeshGraphNets-style): encode source & target nodes, message-pass on the
candidate graph (+ intra-mesh edges for context), decode a **single signed scalar `q_e` per candidate
edge** (unconstrained — may be negative). Optional inductive bias: `q_e = base_e + Δ_e`, where `base_e` is
a positive distance kernel (first-order-like) and `Δ_e` the learned signed correction.

## 4. Doubly-constrained projection layer (the crux — the survey's open gap)
Let `M_e` be edge mass (`S_e = M_e / B_{tgt(e)}` the operator weight; `y_i = (1/B_i) Σ_{e:tgt=i} M_e x_{src(e)}`).
Both physical properties are **linear marginal equalities** on `M`:
- **Consistency** (rows sum to 1 / constants reproduced): `Σ_{e:tgt(e)=i} M_e = B_i`  for all targets `i`.
- **Conservation** (mass preserved per source): `Σ_{e:src(e)=j} M_e = A_j`  for all sources `j`.

Stack as `A M = b`, where `A ∈ R^{(n_tgt+n_src)×|E|}` has, per edge column `e=(i,j)`, a 1 in target-row `i`
and a 1 in source-row `j`; `b = [B ; A_src]`. The **Euclidean projection** of the GNN's raw `q` onto
`{M : AM=b}` is closed-form — **one solve, no iteration** (unlike Sinkhorn):

```
M* = q + Aᵀ λ,   where  (A Aᵀ) λ = b − A q
```

- `A Aᵀ = [[D_tgt, B_adj],[B_adjᵀ, D_src]]` is the bipartite **signless Laplacian** (diagonal = node
  degree, off-diagonal = candidate adjacency) — sparse, SPD up to a **rank-1 deficiency** (the marginals
  are dependent: `Σ_i B_i = Σ_j A_j = 4π`). Handle by pinning one dual / `ε`-regularization / pseudo-inverse.
- Solve with a few **conjugate-gradient** iterations (sparse, GPU) — differentiable (implicit-function
  theorem, or unroll CG). For a fixed pair the system is fixed → can precompute a factorization.
- **Signed-compatible:** unlike Sinkhorn (which needs positivity and iterates), this projects *signed*
  weights onto *both* marginals exactly in one linear solve — also a potential *speed* win over Sinkhorn.

Output `S_e = M*_e / B_{tgt(e)}`. Conservation + consistency now hold to solver precision **by construction**.

## 5. Training (classical operator as TEACHER)
The literature's defensible pattern is "classical + learned correction"; we **invert** it — classical is the
*teacher*, the cheap supermesh-free GNN is the student:
- **Targets (offline, computed once WITH the supermesh):** the TR `np2` operator `S_np2`, and analytic
  ground-truth harmonic remaps (the `harmonic_target=truth` harness already built).
- **Loss:** operator-match `‖S_pred − S_np2‖` + harmonic/field error vs analytic truth (the accuracy
  signal), with the **projection layer in the loop** so the GNN learns weights that are accurate *after*
  projection. Conservation/consistency are free (projection) → the GNN spends all capacity on accuracy.
- Train across many mesh pairs / families → one model that generalizes (the amortization advantage).

## 6. Inference (the payoff)
`kNN candidate graph (cheap) → GNN forward → one sparse CG solve → S`. **No supermesh.** GPU-fast,
conservative + consistent by construction, ~2nd-order accuracy if training succeeds. Compare wall-clock to
TR (which pays the supermesh + is CPU-only) — the quantified efficiency advantage nobody has measured.

## 7. What this fills (vs literature)
- **Open Q1:** both-marginals (conservation **and** consistency) on a **signed** bipartite operator — via
  the doubly-constrained projection. Not done before.
- **Open Q4:** the concrete, quantified value of learning (supermesh-free + GPU + generalization) vs
  classical reconstruction — the measurement nobody has reported.

## 8. Risks / research questions (go in clear-eyed)
1. **Can the GNN match np2 from cheap features alone?** Message passing has the *capacity* (Brandstetter
   2022) but "training yields it" is unproven for remap. This is THE research question.
2. **Stencil recall:** the kNN candidate graph must contain the (wider) 2nd-order support — measure recall
   vs np2 nnz; widen `k` if needed (trades compute).
3. **Projection preserves accuracy:** projecting raw weights could damage learned high-order structure —
   mitigated by training with the projection in the loop (the GNN learns post-projection-accurate weights).
4. **Projection stability/differentiability:** rank-1 deficiency + CG through the layer; validate gradients.
5. **Fallback if (1) fails:** the field's proven recipe — classical reconstruction + a learned *correction*
   where overlaps are available; pure-learned only where they are not (truly geometry-free data). Either
   way the deliverable is the *quantified advantage*, not an accuracy win.

## 9. Validation / ablation plan
- **Static sanity:** feed the *classical* signed 2nd-order weights through the projection → confirm it
  preserves them (projection is near-identity on feasible inputs) and stays machine-exact.
- **Stencil recall** of the kNN graph vs np2 support, per family.
- **Core result:** GNN (supermesh-free) per-degree error vs `np2` and vs analytic truth, on held-out pairs
  AND held-out families (HEALPix), with conservation/consistency residuals (~solver precision).
- **Efficiency:** wall-clock GNN+projection (GPU) vs TR np2 (CPU, incl. supermesh) across resolutions.
- **Ablations:** projection vs Sinkhorn (signed vs non-negative → first-order cap); with/without the
  classical-teacher loss; `k` (stencil width) sweep; learned-correction-on-classical vs pure-learned.

## 10. References (from `wzw5mm1uz`)
Harder et al. constrained downscaling 2023–24 (arXiv:2208.05424) — softmax/projection constraint layer
(conservation, single marginal). DC3 (Donti et al. ICLR 2021, arXiv:2104.12225) — completion+correction for
hard constraints. Brandstetter, Worrall & Welling (ICLR 2022, arXiv:2202.03376) — message passing contains
FD/FV/WENO. Bar-Sinai PNAS 2019; Kochkov PNAS 2021 — learned discretization (effective high order, 1D/
structured). Barwey et al. 2024 (arXiv:2409.07769) — classical interpolation + learned correction.
MeshGraphNets (Pfaff et al. ICLR 2021, arXiv:2010.03409). FluxGNN (Horie & Mitsume ICML 2024,
arXiv:2405.16183) — conservation-by-construction GNN (PDE). Lipnikov & Shashkov (JCP 2023) — classical
high-order conservative bounds-preserving remap. GMD 17:415 (2024) — consistency `Σ_i w_ij=1` condition.
