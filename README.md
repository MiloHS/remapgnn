# remapgnn

Learned TempestRemap-style conservative remapping operators for spherical climate meshes.

## Core idea

For a source mesh and target mesh, we build candidate source-target edges. The model predicts edge scores on this graph. Sinkhorn balancing turns those scores into a sparse mass matrix. The balancing ends on the source marginal, so global (source) mass conservation is enforced tightly; the target row sums (consistency — constants mapping to constants) are tracked and softly penalized rather than satisfied exactly. The remapping weights are then applied to fields on the source mesh.

The goal is to learn a fast conservative approximation to TempestRemap operators across mesh pairs.

## Current best result

The current best model is `v18_irno_corrector_from_v16_l24_a2p0_mink8`.

It uses:

- a frozen v16 gated-hybrid-attention GNN/Sinkhorn base remapper
- a shared gated-hybrid-attention corrector
- iterative correction steps conditioned on increasing spherical harmonic bands
- Sinkhorn balancing after each correction step

The correction iteration is:

1. base v16 operator
2. correction with `lmax=8`
3. correction with `lmax=16`
4. correction with `lmax=24`

The reported metric is **agreement error with TempestRemap** — the relative L2 distance between the
learned remap and Tempest's remap of the same field, averaged over the field set. Lower means the
model reproduces Tempest more closely; this is *fidelity to Tempest*, **not** accuracy against an
analytic truth.

Numbers below are from a clean retrain (full 8-pair training, a 2-pair held-out validation set used
for model selection, and the test pair `RLL-r90-180_to_CS-r16` held out of both — see
`docs/AUDIT_REPORT.md`). Across the six evaluation pairs the corrector lowers the mean agreement
error from `0.00387` to `0.00354` (≈8.6%).

**This six-pair mean is in-sample-dominated and overstates generalization.** Five of the six pairs
are training pairs, where the corrector improves a lot (≈12–19%); on the genuinely held-out pair
`RLL-r90-180_to_CS-r16` it improves only `0.00389` → `0.00385` (**≈1%**). The large train-vs-held-out
gap is an overfitting signature: with the current ~8 same-family pairs the corrector learns to mimic
Tempest on pairs it has seen, and that barely transfers to a new pair. (Held-out n=1, single seed.)
The generalization story for this project is topology diversity (v20a/v20b), not the corrector alone.

## Repository structure

- `remapgnn/` — reusable package code
- `scripts/` — training and evaluation scripts
- `configs/` — experiment configurations
- `docs/` — experiment notes, model lineage, and result summaries
- `analysis_medium_improv/github_results/` — curated result CSVs and figures

## Important docs

- `docs/INFERENCE.md` — how to download the trained v18 weights and current inference workflow

- `docs/MODEL_LINEAGE.md`
- `docs/RESULTS_SUMMARY.md`
- `docs/CONVERGENCE_STUDY.md`

## Main training scripts

- `scripts/train_config.py`
- `scripts/train_config_balanced.py`
- `scripts/train_config_balanced_harmonic.py`
- `scripts/train_config_irno_corrector.py`

## Main evaluation scripts

- `scripts/evaluate_config.py`
- `scripts/evaluate_spectral_harmonics.py`
- `scripts/evaluate_irno_corrector.py`
- `scripts/evaluate_irno_spectral_trajectory.py`

## Model lineage summary

- v10: hybrid attention GNN
- v11: gated hybrid attention GNN
- v16: gated hybrid attention with spherical harmonic loss up to degree 24
- v18: frozen v16 plus iterative conditional corrector
- v19: gentler v18 ablation

## Status

Current best result is v18. v19 shows that gentler correction remains stable but does not outperform the main v18 run.
