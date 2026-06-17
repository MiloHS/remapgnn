# v20b diverse-topology results

v20b tests whether adding topology diversity to training improves transfer relative to the v20a topology-holdout model.

## Setup

v20a trained only on CS↔ICOD.

v20b uses the same broad architecture and loss family, but restores a diverse training set containing RLL, CS, and ICOD directions.

## Main result

v20b strongly supports the topology-diversity hypothesis.

Compared with v20a, v20b substantially improves finest-grid error ratios and observed convergence behavior in the forward directions.

## Actual-error summary

From `analysis_medium_improv/github_results/v20a_vs_v20b_actual_error_summary.csv`:

- CS→ICOD:
  - v20a base mean ratio vs Tempest: about 1.72
  - v20b base mean ratio vs Tempest: about 0.56
  - v20b improves over v20a by about 3.1× and beats Tempest on mean finest-grid error.

- CS→RLL:
  - v20a base mean ratio vs Tempest: about 1.30
  - v20b base mean ratio vs Tempest: about 0.54
  - v20b improves over v20a by about 2.4× and beats Tempest on mean finest-grid error.

- ICOD→CS:
  - v20a base mean ratio vs Tempest: about 4.51
  - v20b base mean ratio vs Tempest: about 3.23
  - v20b improves, but this remains a hard reverse direction.

- RLL→CS:
  - v20a base mean ratio vs Tempest: about 2.01
  - v20b base mean ratio vs Tempest: about 2.19
  - base does not improve, but the v20b corrector helps. The lmax24 stage improves to about 1.86× Tempest.

## Corrector behavior

The v20a corrector generally worsened actual finest-grid error.

In v20b, the corrector still slightly worsens the already-strong forward directions, but it helps in the hard reverse directions:

- ICOD→CS improves from about 3.23× Tempest at base to about 2.75× at lmax24.
- RLL→CS improves from about 2.19× Tempest at base to about 1.86× at lmax24.

This suggests the corrector needs topology diversity to learn useful refinements.

## Interpretation

v20a showed partial zero-shot topology transfer, but was limited by its narrow training topology set.

v20b shows that adding diverse topology exposure can produce much stronger transfer. The learned model can outperform Tempest on some finest-grid mean errors in forward directions, especially CS→ICOD and CS→RLL.

The remaining weakness is reverse-to-CS transfer, especially ICOD→CS and RLL→CS. This motivates v20c with pole-aware and topology-aware features.

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
