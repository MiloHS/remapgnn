# remapgnn audit report

Audited from a local copy of `~/remapgnn` (swing). 47-agent review, 6 dimensions, every
finding adversarially re-verified. 41 raised → 39 confirmed, 2 rejected. Deduped below.

The single most important theme: **the headline generalization results are measured on data
the model was trained and/or selected on.** The code is largely correct; the *evaluation
protocol and the claims built on it* are the main problem.

---

## Tier 1 — Research-validity (these change what the results mean)

### 1. [CRITICAL] Test pair is used for model selection (test-set leakage into checkpointing)
Across **13 configs**, `training.test_pair` is also listed in `training.checkpoint_pairs`.
Best-epoch / `best_pack` is chosen by minimizing the mean score over `checkpoint_pairs`
(`train_config_balanced_harmonic.py:664-673`, `train_config_irno_corrector.py:700-707`,
`train_config_balanced.py:322-328`). So the held-out pair drives early stopping and is then
reported as the "test" result. Affects flagship v16, v18 corrector, and the seed456 reruns.
Fix: make `checkpoint_pairs` disjoint from `test_pair`; report test once, after selection.

### 2. [CRITICAL] Headline v18 number is ~5/6 in-sample
`README.md` / `RESULTS_SUMMARY.md`: "across six mesh pairs … 0.002956 → 0.002715". That mean
comes from `evaluate_irno_corrector.py --all-pairs` over `cfg.pairs` (6 pairs). **5 of those 6
are in `train_pairs`** (`v18_...json:19-37`); only `RLL-r90-180_to_CS-r16` is held out — and
that one is in `checkpoint_pairs` (issue #1). The headline is dominated by training-fit, and
the held-out pair is never broken out.
Fix: report the headline only on genuinely held-out pairs; label the 6-pair mean as in-sample.

### 3. [HIGH] Validation/checkpoint pairs overlap training pairs (no clean validation signal)
In v16/v18, `val_pair` is a training pair, and 3 of 4 `checkpoint_pairs` are training pairs.
In `train_config.py` it's starker: the single `val_pair` used for `is_best` (line 191) is a
training pair, so selection is on training loss. No split is disjoint from train and test.
Fix: define a validation set disjoint from both train and test; select only on it.

### 4. [HIGH] Headline field metric measures *agreement with Tempest*, not accuracy
`evaluate_irno_corrector.py:339-356`: `mean_rel_l2_vs_tempest = ||y_gnn − y_tempest|| / ||y_tempest||`.
Lower = closer to Tempest's output, not more accurate than it. Consistent with the project goal
("learn a fast approximation to Tempest"), but `RESULTS_SUMMARY.md:18` calls it "improved field
relative L2 **versus** TempestRemap" → reads as beating Tempest. No ground-truth field is
compared in this script.
Fix: rename to "agreement with Tempest", or add a true ground-truth comparison (the convergence
script does have analytic truth) and report GNN-vs-truth and Tempest-vs-truth side by side.

### 5. [HIGH] "Best model" is not reproducible — non-deterministic `hash()` seeds harmonic mode selection
`train_config_balanced_harmonic.py:129`: `rng = default_rng(seed + abs(hash(pair)) % 1000000)`.
Python str `hash()` is salted per process (`PYTHONHASHSEED` is never pinned anywhere). For
degrees with `2l+1 > modes_per_degree` (l=4,8,12,16,24), `choose_m_values` picks a *different*
random m-subset each run → different harmonic loss → different checkpoint score → different
"best" epoch. Re-running the same config with the same `seed` does not reproduce the same model.
Reused by the IRNO corrector via `build_harmonic_cache`.
Fix: deterministic hash (`hashlib.sha256`) or fixed per-pair index; pin `PYTHONHASHSEED`.

### 6. [HIGH] The "second seed reproduced it" claim reuses the identical frozen v16 base
`v18_...seed456` loads the *same* seed-0 v16 base checkpoint as seed123; only the corrector RNG
changes (123→456). The base operator produces all 6 base-row numbers and ~5/6 of the eval set,
so base rows are byte-identical across the two tables. A v16 seed456 base exists but is not wired
in. Seed robustness of the dominant component is never sampled.
Fix: retrain the full pipeline (v16 base + corrector) under independent seeds; report mean±std.

### 7. [MEDIUM] Normalization stats are fit on the test/eval pairs (normalization leakage)
`stat_pairs = train_pairs + checkpoint_pairs + [test_pair] + cfg.pairs`
(`train_config_balanced_harmonic.py:542`; same in `train_config_balanced.py:216`,
`train_config.py:80/86`). Global feature mean/std (geometry, areas) are fit partly on the
held-out pair and baked into the saved pack, reused at eval. Input-feature leakage (not labels),
so mild — but it does compromise the one genuine holdout (notably v20b). Bites v20b; not v20a.
Fix: compute stats from `train_pairs` only; freeze and apply to val/test.

### 8. [MEDIUM] No variance / per-pair / significance for an ~8% shift over 6 pairs
Δ ≈ 0.00024 reported as a single mean, no std, no per-pair values, no paired test
(`RESULTS_SUMMARY.md:18-27`). With 5/6 pairs in-sample and the dominant base operator unvaried
across seeds, the shift can't be distinguished from fit/optimization noise.
Fix: per-pair errors with the holdout separated; multiple end-to-end seeds with mean±std; paired test.

### 9. [MEDIUM] Spectral "test" reuses the training degrees on the training pairs
`evaluate_irno_spectral_trajectory.py` / `evaluate_spectral_harmonics.py`: default degrees
`0 1 2 4 8 12 16 24 32` — 8 of 9 are exactly the harmonic-loss training degrees — evaluated via
`--all-pairs` (5/6 training pairs), against Tempest output (not truth). The ~10% spectral
improvement is largely in-objective, in-sample. Only l=32 and the one held-out pair are genuine.
Fix: evaluate on held-out degrees and held-out pairs; label which cells are train vs extrapolation.

---

## Tier 2 — Correctness / numerical (affect reported numbers)

### 10. [MEDIUM] `build_model` misroutes `'pair_conditioned_gated_hybrid_attention'`
`models.py:674-681` returns plain `GatedHybridAttentionGNNSinkhorn`, not
`GateConditionedHybridAttentionGNNSinkhorn`. The pair-conditioning class (which splits off the 6
mesh-family columns into the gate only) is reachable solely via `'gate_conditioned_hybrid_attention'`.
Latent — no current config uses the string — but a "pair-conditioned" experiment would silently
run the wrong model.
Fix: route to `GateConditionedHybridAttentionGNNSinkhorn(..., pair_cond_dim=6)` or delete the alias.

### 11. [MEDIUM] Sinkhorn ends on source scaling → operator rows don't sum to 1 (not *consistent*)
`sinkhorn.py:37-44`: last op each iter is the source rescale, so source mass (conservation) is
tight but target row sums (`S_ij = M_ij/area_tgt_i`) ≠ 1 → constants aren't preserved. This is
by design and *is* tracked (`row_sum_rel_l2`, soft-weighted at 0.05), so it's monitored, not
hidden. The real issue is the README wording "satisfies conservative remapping constraints"
(plural) overstates it: conservation yes, consistency no.
Fix: reconcile the README claim; if consistency is needed, end on target scaling or post-normalize rows.

### 12. [MEDIUM] Train uses 30 Sinkhorn iters, eval/selection uses 300
`sinkhorn_iters_train=30` vs `sinkhorn_iters_eval=300` everywhere. Gradients flow through a
30-iter operator; selection and reported metrics use a 300-iter operator. Mismatch concentrates
in the (loose) row-sum dimension. Magnitude unverified in-repo.
Fix: same count for train and eval, or verify 30 is effectively converged for these meshes.

### 13. [MEDIUM] Eval scripts default to 2000 Sinkhorn iters; selection used 300
`evaluate_spectral_harmonics.py`, `evaluate_irno_spectral_trajectory.py`,
`evaluate_refinement_convergence.py`, `evaluate_global_conservation_for_pairs.py`,
`evaluate_config.py` → `--balance-iters 2000`; only `evaluate_irno_corrector.py` uses 300. Same
checkpoint yields different headline numbers across scripts.
Fix: pin one iteration count (load from checkpoint) everywhere.

### 14. [MEDIUM] bf16 autocast at train/selection vs fp32 at eval
`losses.py:41-43` runs the GNN forward under bf16 autocast on CUDA; eval scripts run fp32.
Checkpoints selected on bf16 outputs; headline metrics from fp32 outputs → device-dependent skew.
(Note: `train_config_balanced_harmonic.py` already runs fp32, so no skew there.)
Fix: match the eval precision to training, or document it.

### 15. [MEDIUM] `select_state` matches the wrong key → silent positional fallback
`evaluate_global_conservation_for_pairs.py` and `infer_prepared_pair.py` look up
`state['step_label']`, but `operator_sequence` writes `state['label']`. The name match never
succeeds; it always falls through to a hardcoded `{base:0, lmax8:1, lmax16:2, lmax24:3}` map.
Correct only because current bands are exactly `[8,16,24]`; any other band set silently selects
the wrong stage while still labeling it `lmax24`.
Fix: match on `state['label']`; raise on no match instead of positional fallback.

### 16. [MEDIUM] `infer_prepared_pair.py --edge-parquet` is not used by the model
`infer_prepared_pair.py:226-250`: `--edge-parquet` only sets output geometry; the operator `S`
comes from `cfg.edge_path(pair)` via `get_irno_states`→`load_pair_tensors`. If the two differ in
ordering (same edge count) the remapped field is silently wrong; usually it crashes instead. The
documented happy-path works only because it writes to exactly the cfg path. No length/order assert.
Fix: route `--edge-parquet` through the model path, or assert `len(S)==len(geom.src_index)` and matching order.

---

## Tier 3 — Latent bugs & robustness (low severity / dormant today)

- **[LOW] node counts `max(index)+1`** drop a trailing edgeless source cell; `n_a` is known at
  build time (`build_..._kdist.py:70`) but not persisted. Hard-fails external-field inference for
  degenerate meshes. (`data.py:127-131,353-354`)
- **[LOW] Sinkhorn source marginal omits absent source cells** — latent; forced true edges cover
  all positive-mass sources on valid conservative maps. (`data.py:368-371`)
- **[LOW] `area_ratio_tgt_over_src` has no zero-area guard** → a single zero/inf/NaN poisons the
  global normalization for the whole run. (`build_..._kdist.py:168`, `build_external_kdist_graph.py:182`)
- **[INFO] eps mismatch**: `losses.py` passes `eps=1e-12` into Sinkhorn vs the `1e-30` default
  used by the harmonic path → bit-non-identical operators between training paths (numerically inert).
- **[LOW] no `cudnn.deterministic` / `use_deterministic_algorithms` / `PYTHONHASHSEED`** despite
  seeding → GPU runs not bitwise reproducible. (`train_config.py:21-25`)
- **[LOW] saved pack `architecture='irno_corrector'`** is not constructible by `build_model`
  (evaluator uses `corrector_architecture` correctly; trap for other loaders). (`train_config_irno_corrector.py:466`)
- **[LOW] convergence uses centroid point-samples as cell averages** (`evaluate_refinement_convergence.py:410-411,280`)
  — an O(h²) consistency floor in absolute errors; fair to both methods, so relative ranking holds.
- **[LOW] spectral test fields share one RNG across pairs** in `evaluate_spectral_harmonics.py`
  (order-dependent); the trajectory script reseeds per pair correctly.

## Tier 4 — Reproducibility / infrastructure

- **[HIGH → doc fixed] `INFERENCE.md` omits the frozen v16 base checkpoint** that inference
  hard-requires (it carries the normalization `stats` and base weights, lives in gitignored
  `models_medium_improv/`). Following the docs verbatim fails. *INFERENCE.md now documents the
  requirement + extraction layout.* The stronger fix — embed `stats` + base state_dict into the
  v18 pack for true single-file inference (a `make_pack` change in `train_config_irno_corrector.py`)
  — is deferred until after the queued retrain (it touches a script the job runs).
- **[LOW] `evaluate_config.py` references a non-existent `eval_template`** script — hard
  `FileNotFoundError` even under `--dry-run`, for every config. (Use `evaluate_irno_corrector.py` instead.)
- **[RETRACTED — false alarm] "`analysis_medium_improv/github_results/` is absent"**: this was an
  artifact of the audit pull (`git archive` was run with `*.py/*.sh/*.json/*.md` only, excluding
  `*.csv`/`*.png`). In reality **72 files under `github_results/` are committed** and on GitHub, and
  all v20 docs are committed. The curated CSVs/figures and the numbers they back ARE present and
  reproducible. Disregard this finding.
- **[LOW → FIXED] `requirements.txt` is unpinned**: now pinned to the tested `remap_gpu` versions
  (Python 3.11.15, torch 2.6.0, numpy 2.4.4, pandas 3.0.3, pyarrow 24.0.0, xarray 2026.4.0,
  netCDF4 1.7.4, scipy 1.17.1, matplotlib 3.10.9).
- **[LOW — partially retracted] v20a/v20b RLL-transfer numbers**: the summary CSVs
  (`v20a_actual_error_stage_summary.csv`, `v20a_vs_v20b_actual_error_summary.csv`, etc.) ARE
  committed — they are not absent. The only residual gap is that the exact `--pairs` list / driver
  command passed to `evaluate_refinement_convergence.py` is not captured in a committed script, so
  the precise eval set is documented only in prose. Minor; commit a driver script to close it.
- **[LOW] convergence "order > Tempest" framing** doesn't disclose the convergence meshes are the
  training family and the order fit is 4 points, R²≈0.95. (`CONVERGENCE_STUDY.md:36-42`)

---

## Rejected (verified NOT real)

- "Harmonic field loss compares y_pred over all edges vs y_true over positive edges" — `S_true`
  is exactly 0 on negative edges (build sets weight=0), so it's a correct like-for-like comparison.
- "Sinkhorn source-scaling is an *undisclosed* validity issue" — the residual is explicitly
  tracked and soft-penalized, so it's a known monitored trade-off (captured as #11, framed correctly).

## What's solid (for balance)

- The frozen base in the corrector **is** correctly frozen (`eval()` + `requires_grad_(False)` +
  `no_grad`); corrector gradients are correct.
- Conservation (global mass / source marginal) **is** delivered tightly by design.
- Sinkhorn numerics (max-shift softmax, clamps, double precision at eval) are careful.
- The convergence study's Tempest-vs-GNN comparison is *fair* (identical inputs/truth/areas).
- Eval uses `model.eval()` + `no_grad` + fp64 operator construction.
