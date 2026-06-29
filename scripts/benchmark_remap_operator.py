#!/usr/bin/env python
"""Benchmark RemapGNN deployment costs.

This is deliberately separate from the accuracy audit.  It measures:

  * precomputed Tempest map load time,
  * learned-operator input load time,
  * learned operator construction time,
  * sparse apply time for batches of source fields,
  * amortized "one-off" time = load/build + apply.

It does NOT run TempestRemap map generation unless those timings are supplied
from another script; precomputed Tempest map load/apply is a different
deployment regime from supermesh/offline-map generation.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from train_config_irno_corrector import as_int
from sweep_projection_conservation import (
    LearnedOp,
    load_learned_op,
    load_tempest_operator,
    parse_projection_dtype,
    parse_spec,
    learned_operator_arrays,
)


DEFAULT_PAIRS = [
    "CS-r32_to_ICOD-r32",
    "ICOD-r32_to_CS-r32",
    "CS-r32_to_RLL-r90-180",
    "RLL-r90-180_to_CS-r32",
    "ICOD-r32_to_RLL-r90-180",
]

DEFAULT_PACKS = [
    "v10b=models_medium_improv/highorder_corrector_v10b_safe.pt@configs/v20b_base_a3p0_mink8.json",
    "v12_geom_base=models_medium_improv/highorder_signed_v12_geom_mom1e4.pt@configs/v20b_base_a3p0_mink8_geom_v12.json",
    "v12_geom_v10b=models_medium_improv/highorder_corrector_v12_geom_v10b.pt@configs/v20b_base_a3p0_mink8_geom_v12.json",
    "v12_geom_guarded=models_medium_improv/highorder_corrector_v12_geom_guarded.pt@configs/v20b_base_a3p0_mink8_geom_v12.json",
]


@dataclass
class OperatorArrays:
    pair: str
    label: str
    family: str
    S: np.ndarray
    si: np.ndarray
    ti: np.ndarray
    n_src: int
    n_tgt: int
    n_edges: int


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def median_time(fn, repeats: int, device: torch.device | None = None):
    times = []
    out = None
    for _ in range(max(1, repeats)):
        if device is not None:
            sync_if_needed(device)
        t0 = time.perf_counter()
        out = fn()
        if device is not None:
            sync_if_needed(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out, times


def build_csr(op: OperatorArrays) -> sparse.csr_matrix:
    return sparse.csr_matrix((op.S, (op.ti, op.si)), shape=(op.n_tgt, op.n_src))


def benchmark_apply(op: OperatorArrays, n_fields_values: list[int], repeats: int, rng: np.random.Generator):
    A = build_csr(op)
    rows = []
    for n_fields in n_fields_values:
        X = rng.standard_normal((op.n_src, n_fields), dtype=np.float64)
        # Warm once to avoid first-call allocation noise.
        _ = A @ X
        times = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            Y = A @ X
            times.append(time.perf_counter() - t0)
            # Keep the result live until timing stops.
            if Y.shape[0] != op.n_tgt:
                raise RuntimeError("bad sparse apply output shape")
        t = float(np.median(times))
        rows.append(
            dict(
                pair=op.pair,
                operator=op.label,
                family=op.family,
                n_src=op.n_src,
                n_tgt=op.n_tgt,
                n_edges=op.n_edges,
                n_fields=int(n_fields),
                apply_median_s=t,
                apply_per_field_ms=1000.0 * t / max(int(n_fields), 1),
                apply_repeats=repeats,
            )
        )
    return rows


def load_tempest_for_bench(cfg, pair: str, order: str) -> OperatorArrays:
    S, _M, si, ti, area_src, area_tgt = load_tempest_operator(cfg, pair, order)
    return OperatorArrays(
        pair=pair,
        label=order,
        family="tempest_precomputed",
        S=S.astype(np.float64, copy=False),
        si=si.astype(np.int64, copy=False),
        ti=ti.astype(np.int64, copy=False),
        n_src=len(area_src),
        n_tgt=len(area_tgt),
        n_edges=len(S),
    )


def load_learned_for_bench(op: LearnedOp, b: dict, n_cg: int, device: torch.device, projection_dtype, projection_eps_rel: float):
    S, _M, si, ti, area_src, area_tgt, elapsed = learned_operator_arrays(
        op,
        b,
        n_cg,
        device,
        projection_dtype,
        projection_eps_rel,
    )
    return (
        OperatorArrays(
            pair=b["pair"],
            label=op.label,
            family="learned_corrector" if op.is_corrector else "learned_base",
            S=S.astype(np.float64, copy=False),
            si=si.astype(np.int64, copy=False),
            ti=ti.astype(np.int64, copy=False),
            n_src=len(area_src),
            n_tgt=len(area_tgt),
            n_edges=len(S),
        ),
        elapsed,
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            av = abs(float(v))
            if av != 0.0 and (av < 1e-3 or av >= 1e4):
                return f"{float(v):.4e}"
            return f"{float(v):.4f}"
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    header = "| " + " | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def summarize(out_dir: Path, build_df: pd.DataFrame, apply_df: pd.DataFrame, amort_df: pd.DataFrame, args) -> None:
    lines = ["# Remap operator deployment benchmark\n"]
    lines.append(f"- config: `{args.config}`")
    lines.append(f"- pairs: `{', '.join(args.pairs)}`")
    lines.append(f"- learned projection: dtype=`{args.projection_dtype}`, eps_rel=`{args.projection_eps_rel:.3e}`, n_cg=`{args.n_cg}`")
    lines.append(f"- repeats: build/load=`{args.build_repeats}`, apply=`{args.apply_repeats}`")
    lines.append("")
    lines.append("This benchmark times precomputed Tempest map load/apply, not TempestRemap map generation.")
    lines.append("")

    if not build_df.empty:
        lines.append("## Operator load/build summary\n")
        cols = [
            "operator",
            "family",
            "mean_input_load_s",
            "mean_operator_build_s",
            "mean_total_once_s",
            "mean_edges",
            "rows",
        ]
        summary = (
            build_df.groupby(["operator", "family"], as_index=False)
            .agg(
                mean_input_load_s=("input_load_median_s", "mean"),
                mean_operator_build_s=("operator_build_median_s", "mean"),
                mean_total_once_s=("total_once_median_s", "mean"),
                mean_edges=("n_edges", "mean"),
                rows=("pair", "size"),
            )
            .sort_values("mean_total_once_s")
        )
        lines.append(markdown_table(summary[cols]))
        lines.append("")

    if not apply_df.empty:
        lines.append("## Sparse apply summary\n")
        summary = (
            apply_df.groupby(["operator", "n_fields"], as_index=False)
            .agg(
                mean_apply_s=("apply_median_s", "mean"),
                mean_apply_per_field_ms=("apply_per_field_ms", "mean"),
                mean_edges=("n_edges", "mean"),
            )
            .sort_values(["n_fields", "mean_apply_s"])
        )
        lines.append(markdown_table(summary))
        lines.append("")

    if not amort_df.empty:
        lines.append("## Amortized one-off time summary\n")
        lines.append("`total_once_plus_apply_s = input/load + operator build + CPU CSR apply for N fields`.")
        lines.append("")
        summary = (
            amort_df.groupby(["operator", "n_fields"], as_index=False)
            .agg(mean_total_once_plus_apply_s=("total_once_plus_apply_s", "mean"))
            .sort_values(["n_fields", "mean_total_once_plus_apply_s"])
        )
        lines.append(markdown_table(summary))
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v20b_base_a3p0_mink8.json")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--packs", nargs="+", default=DEFAULT_PACKS, help="learned specs: label=pack.pt[@config.json]")
    ap.add_argument("--include-tempest", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--orders", nargs="+", default=["np1", "np2"])
    ap.add_argument("--n-cg", type=int, default=800)
    ap.add_argument("--projection-dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--projection-eps-rel", type=float, default=1e-12)
    ap.add_argument("--n-fields", nargs="+", type=int, default=[1, 5, 25, 100])
    ap.add_argument("--build-repeats", type=int, default=3)
    ap.add_argument("--apply-repeats", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    projection_dtype = parse_projection_dtype(args.projection_dtype)
    rng = np.random.default_rng(12345)

    print("device:", device)
    if device.type == "cuda":
        print("cuda_device:", torch.cuda.get_device_name(0))
    print("out_dir:", out_dir)
    print("projection:", args.projection_dtype, args.projection_eps_rel, "n_cg", args.n_cg)

    model_rows = []
    learned_ops = []
    for spec_s in args.packs:
        spec = parse_spec(spec_s)
        t0 = time.perf_counter()
        op = load_learned_op(spec, cfg_path, device)
        sync_if_needed(device)
        model_rows.append(dict(operator=op.label, pack=str(spec.pack_path), model_load_s=time.perf_counter() - t0))
        learned_ops.append(op)

    build_rows = []
    apply_rows = []
    amort_rows = []

    for pair in args.pairs:
        print(f"\n=== {pair} ===")
        if args.include_tempest:
            for order in args.orders:
                suffix = "" if order == "np1" else "_np2"
                map_path = cfg.maps_dir / f"map_{pair}_conserve{suffix}.nc"
                if not map_path.exists():
                    print(f"  skip {order}: {map_path} missing")
                    continue
                load_s, tempest_op, load_times = median_time(lambda order=order: load_tempest_for_bench(cfg, pair, order), args.build_repeats)
                file_mb = os.path.getsize(map_path) / 1e6
                print(f"  {order:16s} load={load_s:.4f}s edges={tempest_op.n_edges}")
                build_rows.append(
                    dict(
                        pair=pair,
                        operator=order,
                        family=tempest_op.family,
                        n_src=tempest_op.n_src,
                        n_tgt=tempest_op.n_tgt,
                        n_edges=tempest_op.n_edges,
                        input_load_median_s=load_s,
                        operator_build_median_s=0.0,
                        total_once_median_s=load_s,
                        file_mb=file_mb,
                        repeats=args.build_repeats,
                    )
                )
                rows = benchmark_apply(tempest_op, args.n_fields, args.apply_repeats, rng)
                apply_rows.extend(rows)
                for r in rows:
                    amort_rows.append(
                        dict(
                            pair=pair,
                            operator=order,
                            family=tempest_op.family,
                            n_fields=r["n_fields"],
                            total_once_plus_apply_s=load_s + r["apply_median_s"],
                        )
                    )
                del tempest_op
                gc.collect()

        for op in learned_ops:
            def load_batch():
                return load_pair_tensors(op.cfg, pair, op.pack["stats"], device=device)

            input_s, b, input_times = median_time(load_batch, args.build_repeats, device)
            # Warm once.
            with torch.no_grad():
                warm_op, _ = load_learned_for_bench(
                    op, b, int(args.n_cg), device, projection_dtype, float(args.projection_eps_rel)
                )
                del warm_op
                sync_if_needed(device)

            def build_once():
                with torch.no_grad():
                    built, elapsed_internal = load_learned_for_bench(
                        op, b, int(args.n_cg), device, projection_dtype, float(args.projection_eps_rel)
                    )
                return built, elapsed_internal

            build_times = []
            built_op = None
            internal_times = []
            for _ in range(max(1, args.build_repeats)):
                sync_if_needed(device)
                t0 = time.perf_counter()
                built_op, internal = build_once()
                sync_if_needed(device)
                build_times.append(time.perf_counter() - t0)
                internal_times.append(float(internal))
            build_s = float(np.median(build_times))
            total_s = input_s + build_s
            print(f"  {op.label:16s} input={input_s:.4f}s build={build_s:.4f}s edges={built_op.n_edges}")
            build_rows.append(
                dict(
                    pair=pair,
                    operator=op.label,
                    family="learned_corrector" if op.is_corrector else "learned_base",
                    n_src=built_op.n_src,
                    n_tgt=built_op.n_tgt,
                    n_edges=built_op.n_edges,
                    input_load_median_s=input_s,
                    operator_build_median_s=build_s,
                    total_once_median_s=total_s,
                    file_mb=np.nan,
                    repeats=args.build_repeats,
                    internal_operator_build_median_s=float(np.median(internal_times)),
                )
            )
            rows = benchmark_apply(built_op, args.n_fields, args.apply_repeats, rng)
            apply_rows.extend(rows)
            for r in rows:
                amort_rows.append(
                    dict(
                        pair=pair,
                        operator=op.label,
                        family="learned_corrector" if op.is_corrector else "learned_base",
                        n_fields=r["n_fields"],
                        total_once_plus_apply_s=total_s + r["apply_median_s"],
                    )
                )
            del built_op, b
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    model_df = pd.DataFrame(model_rows)
    build_df = pd.DataFrame(build_rows)
    apply_df = pd.DataFrame(apply_rows)
    amort_df = pd.DataFrame(amort_rows)
    model_df.to_csv(out_dir / "model_load.csv", index=False)
    build_df.to_csv(out_dir / "operator_build.csv", index=False)
    apply_df.to_csv(out_dir / "apply.csv", index=False)
    amort_df.to_csv(out_dir / "amortized.csv", index=False)
    summarize(out_dir, build_df, apply_df, amort_df, args)
    print(f"wrote {out_dir / 'model_load.csv'}")
    print(f"wrote {out_dir / 'operator_build.csv'}")
    print(f"wrote {out_dir / 'apply.csv'}")
    print(f"wrote {out_dir / 'amortized.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print("BENCHMARK_REMAP_OPERATOR_DONE")


if __name__ == "__main__":
    main()
