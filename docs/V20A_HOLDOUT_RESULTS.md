# v20a topology-holdout results

v20a tests whether remapgnn transfers to RLL without seeing RLL during training.

## Setup

v20a was trained only on CS↔ICOD pairs:

- `CS-r32_to_ICOD-r32`
- `ICOD-r32_to_CS-r32`

All RLL pairs were removed from training, validation, testing, and checkpoint scoring.

The trained v20a pipeline has:

- v20a base GNN/Sinkhorn remapper
- v20a IRNO-style corrector
- correction stages at `lmax=8`, `lmax=16`, and `lmax=24`

## Main result

v20a does not transfer to RLL as well as v18.

This indicates that v18's strong RLL behavior depended at least partly on RLL exposure during training, rather than being purely zero-shot topology generalization.

## Stage-level summary

The base stage is usually the strongest v20a stage. The learned corrector often worsens extrapolation, especially at higher correction stages.

Summary from `analysis_medium_improv/github_results/v20a_convergence_stage_summary.csv`:

- CS→ICOD:
  - Tempest mean order: about 1.01
  - v20a base mean order: about 1.14
  - v20a lmax24 mean order: about 1.00, but with larger finest errors than Tempest

- ICOD→CS:
  - Tempest mean order: about 1.08
  - v20a base mean order: about 0.62
  - v20a lmax24 mean order: about 0.45

- CS→RLL:
  - Tempest mean order: about 1.02
  - v20a base mean order: about 0.76
  - v20a lmax24 mean order: about 0.65

- RLL→CS:
  - Tempest mean order: about 1.02
  - v20a base mean order: about 0.52
  - v20a lmax24 mean order: about 0.42

## Interpretation

The topology-holdout model shows weak zero-shot transfer to RLL.

The fact that v20a is worse than v18 on RLL suggests that RLL examples in training were important for v18. The corrector appears more brittle than the base model under topology holdout, which suggests that future work should either:

1. include RLL in training,
2. add geometry/pole-aware features,
3. regularize the corrector more strongly, or
4. use a topology-balanced training split.

## Next experiment

The next controlled run should be v20b:

- train with CS, ICOD, and RLL included,
- keep the same architecture and loss as v20a,
- evaluate on CS↔ICOD, CS↔RLL, RLL↔CS, and RLL→RLL.

