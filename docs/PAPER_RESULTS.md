# Tier-1 Paper Results — Fast, Topology-General Learned Conservative Remapping

Consolidated, freshly-measured results for the paper. All "ours" numbers use the bipartite-GNN base
operator balanced with **over-relaxed (SOR) Sinkhorn** to convergence (conservative AND consistent);
the iterative correction layer is **not** part of the proposed method (see §5). Accuracy is measured
**vs analytic ground truth** on the target grid; "ratio" is our error divided by first-order
TempestRemap's error on the same field.

## Hardware / setup
- **TempestRemap (TR):** serial CPU, single core (it cannot use a GPU — computational geometry).
  Conservative FV→FV first-order map (`GenerateOverlapMesh` + `GenerateOfflineMap --in_np 1
  --correct_areas`).
- **Ours:** NVIDIA A100-SXM4-40GB for the GNN forward pass + SOR-Sinkhorn; candidate-graph (kNN)
  build on CPU (scipy `cKDTree`, 8 threads).
- Mesh families: CS (quad/gnomonic), RLL (quad/isolatitude), ICOD (hexagon/icosa-dual),
  ICO (triangle/geodesic), MPAS (Voronoi), CSRR (variable-resolution CS), HEALPix (equal-area quad).
  All candidate graphs have **recall = 1.0** vs the TR reference support.

---

## Table 1 — Sinkhorn iterations (hardware-INDEPENDENT; the primary efficiency claim)
Over-relaxation (ω = 1.95, with a residual safeguard) converges to the **same fixed point** (operator
identical to ~1e-5) in far fewer iterations.

| Resolution | vanilla iters | SOR iters (ω=1.95) | reduction |
|---|---|---|---|
| r32  | 9,750 | 350 | 28× |
| r64  | 27,300 | 1,050 | 26× |
| r128 | 37,550 | 1,500 | 25× |
| r256 | >60,000 (capped, not converged) | 3,750 | >16× |

→ **~25–28× fewer iterations, on any hardware.** (At r256 vanilla did not converge within the 60k cap,
so the true factor is larger.)

## Table 2 — End-to-end weight generation (deployment comparison)
Ours = kNN build (CPU) + GNN forward (GPU) + SOR-Sinkhorn (GPU). Median of 5, warmed up, cuda-synced.

| Res | cells src→tgt | edges | TR (CPU) | ours: kNN + fwd + SOR | ours total | speedup |
|---|---|---|---|---|---|---|
| r32  | 6.1k→10.2k | 0.13M | 1.52 s | 0.019 + 0.014 + 0.063 | **0.10 s** | **16×** |
| r64  | 24.6k→41k  | 0.52M | 4.96 s | 0.079 + 0.054 + 0.174 | **0.31 s** | **16×** |
| r128 | 98.3k→164k | 2.06M | 16.9 s | 0.320 + 0.213 + 0.328 | **0.86 s** | **20×** |
| r256 | 393k→655k  | 8.25M | 82.1 s | 1.318 + 0.851 + 2.943 | **5.11 s** | **16×** |

→ **~16–20× faster end-to-end** (including the kNN prep, the fair analog of TR's overlap-mesh build).
Forward+Sinkhorn only (excluding kNN prep): **20–31×**. Multi-pair GPU throughput (amortization):
~3.6 pairs/s across an r32/r64/r128 mix.

**Honest framing:** the absolute ratio is hardware-dependent; report it *with* the hardware. The
structural reason it holds on any hardware: TR is serial-CPU and cannot parallelize to GPU, while ours
is data-parallel. On equal CPU footing ours is ~parity with TR; the speedup comes from GPU portability.

## Table 3 — Accuracy vs first-order TempestRemap (vs analytic truth, broad spectrum)
v20b diverse base, SOR-converged operator. Ratio aggregated over {x, smooth1, Y_4, Y_8, Y_16, Y_24
incl. tesseral}; const sentinel excluded.

| Direction | our err | TR err | ratio |
|---|---|---|---|
| CS→ICOD (r64)  | 1.74e-2 | 2.50e-2 | 0.64 |
| ICOD→CS (r64)  | 1.73e-2 | 1.43e-2 | 1.00 |
| CS→RLL         | 2.63e-1 | 2.65e-1 | 0.94 |
| RLL→CS         | 1.78e-1 | 1.79e-1 | 0.95 |
| **mean** |  |  | **0.88** |

→ **Matches or beats first-order TR on every direction, ~12% lower error on average.**
Conservation residual ~1e-9, consistency (row-sum) residual ~1e-6, by construction.

> CAVEAT: an earlier internal number ("CS→ICOD 0.29", 3× better) was a **favorable subset**
> (smooth functions only, finest grid). Do **not** use 0.29 in the abstract. The honest broad-spectrum
> figure is **0.88**. A scoped secondary claim ("up to ~3× better on smooth fields at high resolution")
> is possible but must be re-verified before use.

## Table 4 — Zero-shot topology generalization (held-out families, vs analytic truth)
Base trained at increasing diversity (D2={CS,ICOD}, D3={+RLL}, D5={+ICO,+MPAS}), 3 seeds, then evaluated
**without any training** on held-out HEALPix (unseen family) and CSRR (unseen variable-resolution).

| Diversity | HEALPix err (epochs-ctrl) | CSRR err | HEALPix vs TR |
|---|---|---|---|
| D2 {CS,ICOD}     | 0.0998 | 0.0627 | ~1.25× |
| D3 {+RLL}        | 0.0959 | 0.0632 | — |
| D5 {+ICO,+MPAS}  | **0.0949** | **0.0549** | — |

→ **Even a 2-family model zero-shots to an unseen family at ~1.25× first-order TR.** Training diversity
improves it (HEALPix −5%, CSRR −12%, monotonic; D5 seed std 3e-4 vs D2 1.1e-3 ⇒ >3σ separation) **when
each family is adequately trained** (100 epochs). Under a *fixed total-step budget* (volume-controlled),
the benefit is masked (D5 → 0.1020) because spreading the budget across 25 pairs under-trains each —
an honest nuance: diversity pays off given sufficient training budget, not for free.

## Table 5 — Iterative correction layer: a delimiting negative result
An IRNO-style corrector (residual edge-reweighting on top of a frozen base) does **not** improve accuracy,
tested in three regimes (full base; low-band base + Tempest target; low-band base + ground-truth target,
dominant spectral loss). Corrected/base error ≈ **0.97–1.00 at every degree** — flat.

→ **Cause is structural:** the operator is non-negative (positive affinities + Sinkhorn), and a
non-negative conservative operator is at most first-order accurate. Genuine higher-order remapping needs
**negative weights** (sub-cell reconstruction), which edge-reweighting cannot introduce. This bounds what
learned correction can do and motivates a signed-operator design as future work.

---

## Provenance (reproduction)
- **SOR / converged Sinkhorn:** `remapgnn/sinkhorn.py` (`DEFAULT_OMEGA=1.95`, `omega`/safeguard in
  `sparse_sinkhorn_balance`, `converged_balance`).
- **Compute (Table 1,2):** `scripts/_bench_compute_full.py` (GPU, PBS 43384); TR-CPU
  `scripts/_bench_tempest.sh`; SOR-vs-vanilla iter sweeps `scripts/_bench_anderson2.py`,
  `scripts/_bench_sor_stiff.py`.
- **Accuracy (Table 3):** `scripts/eval_base_spectral.py` →
  `analysis_medium_improv/phase2/accuracy_v20b.csv`.
- **Topology generalization (Table 4):** data tooling `scripts/gen_maps_topo.sh`,
  `scripts/normalize_mesh_unit.py` (MPAS km→unit), `scripts/healpix_to_scrip.py`; configs
  `scripts/_gen_phase2_configs.py`; jobs `jobs_phase2_topo.pbs` (volume-ctrl, PBS 43285) /
  `jobs_phase2e_topo.pbs` (epochs-ctrl, PBS 43335); results in `analysis_medium_improv/phase2{,e}/`.
- **Correction boundary (Table 5):** truth-target option `harmonic_target=truth` in
  `scripts/train_config_irno_corrector.py` + `scripts/train_config_balanced_harmonic.py`;
  eval `scripts/evaluate_refinement_convergence.py`.
