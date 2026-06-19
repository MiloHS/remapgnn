# v20b diverse-topology results

v20b tests whether adding topology diversity to training improves transfer relative to the v20a topology-holdout model.

## Setup

v20a trained only on CS↔ICOD.

v20b uses the same broad architecture and loss family, but restores a diverse training set containing RLL, CS, and ICOD directions.

## Update: converged Sinkhorn (supersedes the iteration-limited numbers below)

The numbers in the rest of this file were taken at 2000 Sinkhorn iterations, where the operator is
**conservative but not consistent** (rows sum to 1 only to ~10⁻³) — Sinkhorn under-convergence, not
a model limitation (see `SINKHORN_CONVERGENCE.md`). Running the balancer **to convergence** makes the
operator simultaneously conservative *and* consistent and changes the result qualitatively:
**v20b beats Tempest on all four directions**, and v20a is competitive everywhere.

Finest-grid mean error ratio vs Tempest (functions x/y/z/smooth1/smooth2, base operator, converged
balancing; conservation residual ~10⁻⁹ throughout):

| direction | v20a (no RLL) | v20b (diverse) |
|---|---:|---:|
| CS→ICOD | 0.88× | **0.29×** |
| ICOD→CS | 1.48× | **0.57×** |
| CS→RLL  | 0.72× | **0.34×** |
| RLL→CS  | 1.01× | **0.47×** |

Topology diversity still helps in every direction (v20b < v20a throughout), and the
previously-"unsolved" reverse-to-CS directions are now well below Tempest for v20b. Note the
corrector (lmax24) is **no longer beneficial** on the converged base — it was trained against the
under-converged operator and now slightly *increases* finest-grid error (e.g. v20b CS→ICOD
0.29→0.38), so the converged base alone is the best model; the corrector would need retraining in
the converged regime, or may be dropped. Source CSV:
`analysis_medium_improv/github_results/v20_converged_finest_error_summary.csv`.

The sections below are retained for context but are superseded by these converged numbers.

## Main result (clean re-run, audit-corrected; iteration-limited — superseded by the section above)

v20b supports the topology-diversity hypothesis, but **more modestly than the original numbers
suggested**. The original v20b numbers were optimistically biased by an evaluation-leakage bug (the
test pair and several eval pairs were in `checkpoint_pairs`, and normalization stats were fit on
eval pairs — see `AUDIT_REPORT.md`). After fixing the split (test fully held out, a **representative
held-out validation set** `{CS-r16↔ICOD-r16, RLL-r30-60_to_CS-r16}`, train-only stats) and
retraining, the finding holds on 3 of 4 directions with smaller margins.

Metric: finest-grid mean ratio of `area_rel_l2` vs Tempest, averaged over functions
`{x, y, z, smooth1, smooth2}` (base stage).

| direction | v20a (clean) | v20b (clean) | diversity effect | v20b vs Tempest |
|---|---:|---:|---|---|
| CS→ICOD | 1.75× | **0.77×** | 2.3× better | beats Tempest |
| CS→RLL  | 1.36× | **0.64×** | 2.1× better | beats Tempest |
| ICOD→CS | 4.10× | 3.74× | ~1.1× (marginal) | worse than Tempest |
| RLL→CS  | 1.84× | **2.58×** | 0.71× (diversity hurts) | worse than Tempest |

For reference, the original (leakage-inflated) v20b base ratios were 0.56 / 0.54 / 3.23 / 2.19. The
clean numbers are ~15–40% worse, and **RLL→CS reverses** (clean v20b is worse than v20a there).
Most of an even larger apparent gap in a first clean attempt turned out to be a too-narrow
(CS↔ICOD-only) validation set; adding an RLL→CS validation pair recovered v20b substantially, so the
true leakage inflation is the ~15–40% residual above.

## Corrector behavior (clean)

In the clean re-run the v20b corrector adds a small, consistent improvement at the finest grid
(~10%): CS→ICOD 0.77×→0.69×, CS→RLL 0.64×→0.59×, ICOD→CS 3.74×→3.27×, RLL→CS 2.58×→2.44×
(base→lmax24). This is unlike the v18 / topology-holdout case where the corrector did not help on
held-out resolutions — suggesting the corrector generalizes only when trained on diverse
topologies, and even then the effect is modest.

## Interpretation (clean)

Topology diversity genuinely improves transfer on **CS-source directions** (CS→ICOD, CS→RLL), where
the learned operator beats Tempest at the finest grid. On **reverse-to-CS directions** it is at best
marginal (ICOD→CS) or slightly harmful (RLL→CS), so reverse-to-CS transfer remains the weak spot.
The original "≈2–3× across the board, beats Tempest" overstated the effect; the corrected story is
"diversity clearly helps forward directions; reverse-to-CS is still unsolved." This motivates v20c
with pole-aware / topology-aware features.

Caveats: single seed, no error bars; the metric is finest-grid agreement-style error (the ratio is
vs Tempest, with accuracy measured against analytic truth at held-out resolutions). Clean per-pair
CSVs: `analysis_medium_improv/clean_v20_convergence/` (v20a + v20b narrow-val) and
`analysis_medium_improv/clean_v20b_repval_convergence/` (v20b representative-val); the original
pre-audit CSVs remain under `analysis_medium_improv/github_results/`.

## Next experiments

1. Add RLL→RLL evaluation to test same-topology resolution transfer.
2. Run v20c with additional geometry/topology features:
   - latitude
   - absolute z
   - source area
   - target area
   - area ratio
   - candidate rank
   - local source/target degree
3. Compare v20a, v20b, v20c, and v18 using actual finest-grid error ratios, not just fitted order.
4. Repeat the clean v20b result under multiple seeds to put error bars on the topology-diversity effect.
