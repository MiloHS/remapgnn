# Sinkhorn convergence: consistency vs conservation

## The finding

The remapping operator was **conservative but not consistent** at the iteration counts used
(train 30, eval 300–2000): target row sums deviated from 1 by ~10⁻³. Because the GNN's learned
weights are already ~Tempest-quality (its consistency-normalized error ≈ Tempest's), that small
consistency drift was the **dominant** remaining error — a smooth, large-scale bias that showed up
strongly in the reverse-to-CS directions (ICOD→CS, RLL→CS) and made them look 3–6× worse than
Tempest in the convergence study.

Error maps localized the error to a low-degree dipole on the CS target (not poles, not cube edges).
A constant-field test confirmed it: the error equals `(row_sum − 1)`, correlation 0.96–0.99 with
`(row_sum − 1)·field`, and dividing the output by the row sums removed ~80% of it.

## Root cause: under-convergence, not a tradeoff

An iteration sweep settled the mechanism. Both marginals converge **together** (the bipartite
problem is feasible — totals match at 4π); the operator just needed more iterations:

```
RLL-r90-180_to_CS-r16   src_resid(conserv)   tgt_resid(consist)
   300 iters                8.5e-08              1.01e-03   <- what eval/train used
  3000                      8.5e-08              6.40e-05
 10000                      3.9e-08              1.34e-07
 30000                      2.9e-08              2.09e-08   <- both converged
```

The "end-on-source-scaling" truncation kept conservation tight while leaving consistency
unconverged. RLL/ICOD→CS converge slowly because RLL's tiny anisotropic pole cells are
ill-conditioned — which is exactly why those directions looked broken and CS→RLL did not.

## Impact at convergence (30k iters)

Finest-grid mean error ratio vs Tempest (functions x/y/z/smooth1/smooth2), base operator:

| direction | @2000 iters | **@30k (converged)** | conservation |
|---|---:|---:|---|
| v20b CS→ICOD | 0.77× | **0.29×** | 2e-9 ✓ |
| v20b ICOD→CS | 3.74× | **0.57×** | 4e-9 ✓ |
| v20b CS→RLL  | 0.64× | **0.34×** | 9e-9 ✓ |
| v20b RLL→CS  | 2.58× | **0.47×** | 1e-8 ✓ |

So with a converged balancer the operator is simultaneously **conservative and consistent**, and
**beats Tempest on all four directions** (v20a becomes competitive everywhere too). The 2000-iter
convergence study was consistency-limited and *understated* the model across the board; "reverse-to-CS
unsolved" was a Sinkhorn-convergence artifact, not a model limitation.

## The fix (implemented)

`remapgnn/sinkhorn.py: sparse_sinkhorn_balance` gained a **convergence mode**: pass `tol` (and
optional `max_iter`) to iterate until *both* marginal relative residuals are `< tol`, instead of a
fixed count. Backward-compatible (`tol=None` → old fixed-`n_iter` behavior, used by the
gradient-tracked training path).

Eval/inference now converge by default: `evaluate_irno_corrector.compute_operator_from_logq_eval`
(the single operator-build used by the corrector eval, the refinement-convergence study, and
`infer_prepared_pair`) calls the balancer with `tol=1e-6` and `max_iter = max(--balance-iters, 50000)`.
Verified: at `--balance-iters 300` the operator now reports `row_sum_rel_l2 ≈ 1e-6` (consistent) with
`source_mass_rel_l2 ≈ 5e-8` (conserved).

## Open / optional follow-ups

- **Cheaper deployment convergence:** up-weight the consistency loss in training (currently
  `row_weight=0.05`) so the learned `q` is already near-balanced and needs far fewer iterations to
  converge at inference. Requires retraining; the accuracy win above is already obtained with the
  existing models via eval-time convergence.
- The training path still unrolls a small fixed iteration count (for gradient memory); that is fine
  because eval converges and the learned weights transfer. Documented here so train/eval iteration
  counts are understood rather than silently mismatched.
- The corrector, trained against the *under-converged* operator, slightly *hurts* once the base is
  converged — it likely needs retraining in the converged regime, or may be unnecessary.
- Other eval scripts that build operators directly (spectral) can adopt the same `tol` path.
