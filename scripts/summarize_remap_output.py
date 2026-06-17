#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def first_existing(ds, names):
    for name in names:
        if name in ds:
            return name
    return None


def read_flat(ds, field):
    if field not in ds:
        raise KeyError(f"{field} not found. Available: {list(ds.data_vars)}")
    return np.asarray(ds[field].values, dtype=np.float64).reshape(-1)


def read_area(ds, mesh_path=None, n=None):
    area_name = first_existing(ds, ["cell_area", "area", "areaCell", "cellArea", "area_cell"])
    if area_name is not None:
        area = np.asarray(ds[area_name].values, dtype=np.float64).reshape(-1)
    elif mesh_path is not None:
        m = xr.open_dataset(mesh_path)
        area = read_area(m, None, n)
        m.close()
        return area
    else:
        return None

    if n is not None and area.size != n:
        raise ValueError(f"area size {area.size} does not match expected {n}")
    return area


def area_rel_l2(pred, truth, area):
    return np.sqrt(np.sum(area * (pred - truth) ** 2) / max(np.sum(area * truth ** 2), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-nc", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--target-mesh-nc", default=None)

    ap.add_argument("--truth-nc", default=None)
    ap.add_argument("--truth-field", default=None)

    ap.add_argument("--source-nc", default=None)
    ap.add_argument("--source-field", default=None)
    ap.add_argument("--source-mesh-nc", default=None)

    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    pred_ds = xr.open_dataset(args.pred_nc)
    pred = read_flat(pred_ds, args.field)
    area_tgt = read_area(pred_ds, args.target_mesh_nc, len(pred))

    row = {
        "pred_file": args.pred_nc,
        "field": args.field,
        "n_target": pred.size,
        "pred_min": float(np.nanmin(pred)),
        "pred_max": float(np.nanmax(pred)),
        "pred_mean": float(np.nanmean(pred)),
        "pred_std": float(np.nanstd(pred)),
    }

    if area_tgt is not None:
        row["target_area_sum"] = float(np.sum(area_tgt))
        row["target_integral"] = float(np.sum(area_tgt * pred))

    if args.truth_nc:
        truth_ds = xr.open_dataset(args.truth_nc)
        truth_field = args.truth_field or args.field
        truth = read_flat(truth_ds, truth_field)
        truth_ds.close()

        if truth.size != pred.size:
            raise ValueError(f"truth size {truth.size} != pred size {pred.size}")

        row["truth_min"] = float(np.nanmin(truth))
        row["truth_max"] = float(np.nanmax(truth))
        row["truth_mean"] = float(np.nanmean(truth))
        row["plain_rel_l2_vs_truth"] = float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-300))

        if area_tgt is not None:
            row["area_rel_l2_vs_truth"] = float(area_rel_l2(pred, truth, area_tgt))
            row["truth_integral"] = float(np.sum(area_tgt * truth))
            row["target_integral_error_vs_truth"] = float(np.sum(area_tgt * pred) - np.sum(area_tgt * truth))

    if args.source_nc:
        source_field = args.source_field or args.field
        src_ds = xr.open_dataset(args.source_nc)
        src = read_flat(src_ds, source_field)
        area_src = read_area(src_ds, args.source_mesh_nc, len(src))
        src_ds.close()

        row["n_source"] = src.size
        if area_src is not None and area_tgt is not None:
            source_integral = float(np.sum(area_src * src))
            target_integral = float(np.sum(area_tgt * pred))
            row["source_area_sum"] = float(np.sum(area_src))
            row["source_integral"] = source_integral
            row["target_integral"] = target_integral
            row["signed_global_conservation_error"] = target_integral - source_integral
            row["relative_global_conservation_error"] = abs(target_integral - source_integral) / max(abs(source_integral), 1e-300)

    pred_ds.close()

    df = pd.DataFrame([row])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
