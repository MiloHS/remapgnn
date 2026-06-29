#!/usr/bin/env python
"""Benchmark TempestRemap overlap/supermesh and offline map generation.

This complements ``benchmark_remap_operator.py``:

* ``benchmark_remap_operator.py`` times cached-map deployment.
* this script times the expensive TempestRemap generation path:
  GenerateOverlapMesh + GenerateOfflineMap.

For each pair it builds the overlap mesh once, then generates requested map
orders from that overlap.  It writes a CSV suitable for comparing against the
learned operator build times.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config


DEFAULT_PAIRS = [
    "CS-r32_to_ICOD-r32",
    "ICOD-r32_to_CS-r32",
    "CS-r32_to_RLL-r90-180",
    "RLL-r90-180_to_CS-r32",
    "ICOD-r32_to_RLL-r90-180",
]


def mesh_path(cfg, mesh_name: str) -> Path:
    family = mesh_name.split("-")[0]
    if family in {"CS", "ICOD", "RLL"}:
        return cfg.mira_dir / f"Meshes/UniformlyRefined/{family}/sample_NM16_O10_{mesh_name}_TPW_CFR_TPO_A1_A2.nc"
    if family == "ICO":
        return Path("data/gen_meshes/ICO") / f"{mesh_name}.g"
    if family == "MPAS":
        return Path("data/gen_meshes/MPAS") / f"{mesh_name}_unit.nc"
    if family == "CSRR":
        return Path("data/gen_meshes/CSRR") / f"{mesh_name}_unit.nc"
    if mesh_name.startswith("HP-n"):
        return Path("data/gen_meshes/HEALPIX") / f"{mesh_name}.nc"
    raise ValueError(f"do not know how to resolve mesh {mesh_name!r}")


def run_cmd(cmd: list[str], log_path: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    with log_path.open("ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode), time.perf_counter() - t0


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except FileNotFoundError:
        return 0


def log_tail(path: Path, n_lines: int = 8) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(errors="replace").splitlines()
    return "\n".join(txt[-n_lines:])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    def fmt(v):
        if isinstance(v, float):
            av = abs(v)
            if av != 0.0 and (av < 1e-3 or av >= 1e5):
                return f"{v:.4e}"
            return f"{v:.4f}"
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    header = "| " + " | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def summarize(out_dir: Path, df: pd.DataFrame, args) -> None:
    lines = ["# TempestRemap generation benchmark\n"]
    lines.append(f"- config: `{args.config}`")
    lines.append(f"- pairs: `{', '.join(args.pairs)}`")
    lines.append(f"- orders: `{', '.join(args.orders)}`")
    lines.append("- timing path: `GenerateOverlapMesh` once per pair, then `GenerateOfflineMap` for each order")
    lines.append("")

    if not df.empty:
        lines.append("## Per-pair generation times\n")
        cols = [
            "pair",
            "order",
            "overlap_s",
            "offline_map_s",
            "total_if_single_order_s",
            "overlap_mb",
            "map_mb",
            "ok",
        ]
        lines.append(markdown_table(df[cols]))
        lines.append("")

        lines.append("## Summary by order\n")
        summ = (
            df.groupby("order", as_index=False)
            .agg(
                mean_overlap_s=("overlap_s", "mean"),
                mean_offline_map_s=("offline_map_s", "mean"),
                mean_total_if_single_order_s=("total_if_single_order_s", "mean"),
                max_total_if_single_order_s=("total_if_single_order_s", "max"),
                rows=("pair", "size"),
            )
            .sort_values("order")
        )
        lines.append(markdown_table(summ))
        lines.append("")

        both = (
            df.groupby("pair", as_index=False)
            .agg(
                overlap_s=("overlap_s", "first"),
                offline_total_s=("offline_map_s", "sum"),
                all_requested_orders_s=("offline_map_s", "sum"),
            )
        )
        both["all_requested_orders_s"] = both["overlap_s"] + both["offline_total_s"]
        lines.append("## If generating all requested orders from one overlap\n")
        lines.append(markdown_table(both[["pair", "overlap_s", "offline_total_s", "all_requested_orders_s"]]))
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v20b_base_a3p0_mink8.json")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--orders", nargs="+", default=["np1", "np2"], choices=["np1", "np2"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tempest-bin", default=None, help="directory containing GenerateOverlapMesh/GenerateOfflineMap")
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = out_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    tempest_bin = Path(args.tempest_bin) if args.tempest_bin else None

    def exe(name: str) -> str:
        if tempest_bin is not None:
            return str(tempest_bin / name)
        found = shutil.which(name)
        if found is None:
            raise FileNotFoundError(f"{name} not found on PATH; pass --tempest-bin")
        return found

    rows = []
    for pair in args.pairs:
        src_name, tgt_name = pair.split("_to_")
        src = mesh_path(cfg, src_name)
        tgt = mesh_path(cfg, tgt_name)
        pair_dir = Path(tempfile.mkdtemp(prefix=f"{pair}_", dir=artifact_root))
        log_path = pair_dir / "tempest.log"
        ov = pair_dir / f"ov_{pair}.nc"
        print(f"\n=== {pair} ===")
        print(f"src={src}")
        print(f"tgt={tgt}")
        if not src.exists() or not tgt.exists():
            reason = f"missing src={src.exists()} tgt={tgt.exists()}"
            print("SKIP", reason)
            for order in args.orders:
                rows.append(
                    dict(
                        pair=pair,
                        order=order,
                        src=str(src),
                        tgt=str(tgt),
                        overlap_s=float("nan"),
                        offline_map_s=float("nan"),
                        total_if_single_order_s=float("nan"),
                        overlap_exit=-1,
                        offline_exit=-1,
                        overlap_bytes=0,
                        map_bytes=0,
                        overlap_mb=0.0,
                        map_mb=0.0,
                        ok=False,
                        reason=reason,
                        log_tail="",
                    )
                )
            continue

        ov_cmd = [
            exe("GenerateOverlapMesh"),
            "--a",
            str(src),
            "--b",
            str(tgt),
            "--out",
            str(ov),
            "--out_format",
            "netcdf4",
            "--method",
            "fuzzy",
        ]
        ov_exit, ov_s = run_cmd(ov_cmd, log_path)
        ov_bytes = file_size(ov)
        print(f"overlap_s={ov_s:.3f} exit={ov_exit} size={ov_bytes/1e6:.2f} MB")

        for order in args.orders:
            in_np = "1" if order == "np1" else "2"
            mp = pair_dir / f"map_{pair}_{order}.nc"
            if ov_exit == 0:
                mp_cmd = [
                    exe("GenerateOfflineMap"),
                    "--in_mesh",
                    str(src),
                    "--out_mesh",
                    str(tgt),
                    "--ov_mesh",
                    str(ov),
                    "--in_type",
                    "fv",
                    "--out_type",
                    "fv",
                    "--in_np",
                    in_np,
                    "--out_np",
                    "1",
                    "--correct_areas",
                    "--out_map",
                    str(mp),
                ]
                mp_exit, mp_s = run_cmd(mp_cmd, log_path)
            else:
                mp_exit, mp_s = -1, float("nan")
            mp_bytes = file_size(mp)
            ok = ov_exit == 0 and mp_exit == 0 and mp_bytes > 0
            total = ov_s + mp_s if ok else float("nan")
            print(f"  {order}: offline_s={mp_s:.3f} exit={mp_exit} total={total:.3f} map={mp_bytes/1e6:.2f} MB")
            rows.append(
                dict(
                    pair=pair,
                    order=order,
                    src=str(src),
                    tgt=str(tgt),
                    overlap_s=ov_s,
                    offline_map_s=mp_s,
                    total_if_single_order_s=total,
                    overlap_exit=ov_exit,
                    offline_exit=mp_exit,
                    overlap_bytes=ov_bytes,
                    map_bytes=mp_bytes,
                    overlap_mb=ov_bytes / 1e6,
                    map_mb=mp_bytes / 1e6,
                    ok=ok,
                    reason="" if ok else "command failed or output missing",
                    log_tail=log_tail(log_path),
                )
            )
        if not args.keep_artifacts:
            # Preserve the per-pair log for debugging, but remove large generated NetCDFs.
            keep_log = out_dir / f"{pair}_tempest.log"
            shutil.copy2(log_path, keep_log)
            shutil.rmtree(pair_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tempest_generation.csv", index=False)
    summarize(out_dir, df, args)
    print(f"wrote {out_dir / 'tempest_generation.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print("TEMPEST_GENERATION_BENCHMARK_DONE")


if __name__ == "__main__":
    main()
