# Results summary

## Main result

v18 is an IRNO-style iterative corrector trained on top of a frozen v16 conservative remapping
operator. It lowers in-sample agreement error with Tempest substantially, but under a clean
held-out split its generalization gain is marginal (≈1%) — see the Conclusion. The project's
generalization headline is topology diversity (v20), not the corrector.

The v18 corrector improves the operator progressively:

| Step | Meaning |
|---|---|
| 0 | frozen v16 base operator |
| 1 | corrected with lmax=8 conditioning |
| 2 | corrected with lmax=16 conditioning |
| 3 | corrected with lmax=24 conditioning |

## Field trajectory

The reported metric is **agreement error with TempestRemap**: the relative L2 distance between
the learned operator's remapped field and Tempest's remapped field, averaged over the field set.
It measures how closely the model reproduces Tempest (lower = closer), **not** accuracy against
a reference solution. It is zero iff the learned operator equals Tempest. For accuracy against
analytic truth, see the refinement-convergence study.

Numbers below are from a **clean retrain** (seed123): full 8-pair training, a 2-pair held-out
validation set (`CS-r16_to_ICOD-r16`, `ICOD-r16_to_CS-r16`) used for model selection, and the test
pair `RLL-r90-180_to_CS-r16` held out of both training and selection. Train-only normalization stats.

| Step | Mean agreement error (6-pair) | Held-out test pair |
|---|---:|---:|
| base (v16)        | 0.003872 | 0.003892 |
| corrected lmax=8  | 0.003681 | 0.003879 |
| corrected lmax=16 | 0.003574 | 0.003864 |
| corrected lmax=24 | 0.003540 | 0.003853 |

The six-pair mean improves **≈8.6%**, but that figure is **in-sample-dominated** (5 of the 6 pairs
are training pairs). Per pair, the corrector improves the five training pairs by ≈12–19% (e.g.
`ICOD-r32_to_CS-r32` 19.1%, `CS-r32_to_ICOD-r32` 12.8%) while the genuinely held-out pair
`RLL-r90-180_to_CS-r16` improves only **≈1.0%** (0.003892 → 0.003853). This large train-vs-held-out
gap is an **overfitting signature**: with ~8 same-family training pairs the corrector learns to mimic
Tempest on pairs it has seen and barely transfers to an unseen one. The corrector's honest
generalization is therefore marginal; the project's generalization result is topology diversity
(v20a/v20b), not the corrector alone. (Held-out n=1, single seed — no error bars.)

A pre-audit "second seed" (seed456) exists but it varied **only the corrector RNG** and reused the
identical frozen v16 base, so it was a corrector-only check, not a full-pipeline seed-robustness
test. Its numbers predate the leakage fixes and have not been re-evaluated on the clean retrain —
treat them as superseded.

## Spectral trajectory

> **Superseded / pending.** The tables in this section are from the **pre-audit (leaky) run** and
> have **not** been re-evaluated on the clean retrain. They also conflate in-objective, in-sample
> measurement (see below). Re-run `evaluate_irno_spectral_trajectory.py` on the clean v18 checkpoint
> before citing these.

The spectral evaluation uses spherical harmonic test fields and compares learned remap outputs against TempestRemap outputs (again *agreement with Tempest*, not accuracy). Note this is largely an in-objective, in-sample measurement: the evaluation degrees (≤24) coincide with the harmonic-loss training degrees and the pairs are mostly training pairs, so it primarily measures how well the model met its own training objective. Only degree 32 and the held-out pair are genuine extrapolation.

For v18 seed123 (pre-audit), mean spectral error improved as:

| Step | Mean spectral error |
|---|---:|
| base | 1.583641e-02 |
| corrected lmax=8 | 1.503359e-02 |
| corrected lmax=16 | 1.449893e-02 |
| corrected lmax=24 | 1.425947e-02 |

This is about a 10% improvement in average spectral error.

The second seed below again varies only the corrector RNG and reuses the same frozen v16 base:

| Step | Mean spectral error, seed456 (corrector only) |
|---|---:|
| base | 1.583641e-02 |
| corrected lmax=8 | 1.512504e-02 |
| corrected lmax=16 | 1.473847e-02 |
| corrected lmax=24 | 1.461824e-02 |

## Conclusion

A frozen conservative GNN/Sinkhorn remapper *can* be made to reproduce TempestRemap more closely by a
shared conditional iterative corrector, and Sinkhorn balancing keeps (source) conservation tight
throughout. However, under a clean train/validation/test split the corrector's benefit is **largely
in-sample**: ≈12–19% on training pairs but only **≈1%** on the genuinely held-out pair. With the
current ~8 same-family training pairs the corrector overfits and does not meaningfully generalize on
its own.

The project's actual generalization result is **topology diversity** (v20a → v20b): broadening the
training topologies, not adding correction stages, is what improved transfer (see
`V20A_HOLDOUT_RESULTS.md` / `V20B_DIVERSE_TOPOLOGY_RESULTS.md`). The natural next experiment is to
combine the corrector with v20-style diverse training and re-measure held-out generalization.

Caveats and open items: metrics are agreement-with-Tempest, not accuracy against truth (the
refinement-convergence study has the accuracy view); held-out evidence is a single pair, single seed,
with no error bars; the spectral trajectory above is pre-audit and needs a clean re-eval; and
v20a/v20b should be clean-retrained with the same leakage fixes. See `docs/AUDIT_REPORT.md`.

