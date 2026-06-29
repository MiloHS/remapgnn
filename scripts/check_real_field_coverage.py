#!/usr/bin/env python
"""Check real-field file/variable coverage for remap audit pairs.

The audit script can only evaluate physical fields when both source and target
mesh files exist and both contain the requested variable.  This helper makes
the skipped cases explicit and also marks which pairs have candidate graphs and
np2 maps available.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remapgnn.config import load_config


DEFAULT_FIELDS = [
    "AnalyticalFun1",
    "AnalyticalFun2",
    "TotalPrecipWater",
    "CloudFraction",
    "Topography",
]


def has_variable(path: Path, var: str) -> bool | None:
    if not path.exists():
        return None
    with Dataset(path) as ds:
        return var in ds.variables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v20b_base_a3p0_mink8.json")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--fields", nargs="+", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pairs = list(args.pairs or cfg.pairs)
    fields = list(args.fields or cfg.raw.get("fields", DEFAULT_FIELDS))

    rows: list[dict[str, object]] = []
    for pair in pairs:
        src, tgt = pair.split("_to_")
        src_file, tgt_file = cfg.source_target_files(pair)
        edge_exists = cfg.edge_path(pair).exists()
        np1_exists = cfg.map_path(pair).exists()
        np2_exists = (cfg.maps_dir / f"map_{pair}_conserve_np2.nc").exists()
        src_file_exists = src_file.exists()
        tgt_file_exists = tgt_file.exists()
        for field in fields:
            src_has = has_variable(src_file, field)
            tgt_has = has_variable(tgt_file, field)
            usable = bool(src_file_exists and tgt_file_exists and src_has and tgt_has)
            rows.append(
                {
                    "pair": pair,
                    "src": src,
                    "tgt": tgt,
                    "field": field,
                    "usable": usable,
                    "src_file_exists": src_file_exists,
                    "tgt_file_exists": tgt_file_exists,
                    "src_has_field": src_has,
                    "tgt_has_field": tgt_has,
                    "edge_exists": edge_exists,
                    "np1_map_exists": np1_exists,
                    "np2_map_exists": np2_exists,
                    "src_file": str(src_file),
                    "tgt_file": str(tgt_file),
                }
            )

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")

    total = len(rows)
    usable = sum(1 for r in rows if r["usable"])
    print(f"config: {args.config}")
    print(f"pairs: {len(pairs)}  fields: {len(fields)}  usable field cases: {usable}/{total}")
    print()
    print("Per-pair real-field coverage:")
    for pair in pairs:
        rs = [r for r in rows if r["pair"] == pair]
        n_usable = sum(1 for r in rs if r["usable"])
        first = rs[0]
        status = "OK" if n_usable == len(fields) else "MISSING"
        print(
            f"  {status:7s} {pair:28s} "
            f"usable={n_usable}/{len(fields)} "
            f"src_file={bool(first['src_file_exists'])} "
            f"tgt_file={bool(first['tgt_file_exists'])} "
            f"edge={bool(first['edge_exists'])} "
            f"np2={bool(first['np2_map_exists'])}"
        )

    missing = [r for r in rows if not r["usable"]]
    if missing:
        print()
        print("Missing cases:")
        for r in missing:
            reason = []
            if not r["src_file_exists"]:
                reason.append("no src file")
            elif not r["src_has_field"]:
                reason.append("src missing variable")
            if not r["tgt_file_exists"]:
                reason.append("no tgt file")
            elif not r["tgt_has_field"]:
                reason.append("tgt missing variable")
            print(f"  {r['pair']:28s} {r['field']:16s} {', '.join(reason)}")


if __name__ == "__main__":
    main()
