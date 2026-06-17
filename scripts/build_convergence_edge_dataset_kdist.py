#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from scipy.spatial import cKDTree


def arr(ds, name):
    return np.asarray(ds[name].values).reshape(-1)


def lonlat_to_xyz(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if np.nanmax(np.abs(lon)) > 2.0 * np.pi + 1e-6 or np.nanmax(np.abs(lat)) > np.pi / 2 + 1e-6:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1), lon, lat


def parse_pair(path):
    m = re.match(r"map_(.+)_to_(.+)_conserve\.nc$", path.name)
    if not m:
        raise ValueError(f"Could not parse pair name from {path.name}")
    return m.group(1), m.group(2)


def write_rows(rows, writer, out_path):
    if not rows:
        return writer, 0
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer, len(df)


def build_one(map_path, out_dir, alpha, min_k, flush_rows, overwrite):
    src_name, tgt_name = parse_pair(map_path)
    pair = f"{src_name}_to_{tgt_name}"
    alpha_tag = str(alpha).replace(".", "p")
    out_path = out_dir / f"edge_dataset_{pair}_kdist_a{alpha_tag}_mink{min_k}.parquet"

    if out_path.exists() and not overwrite:
        print(f"SKIP existing: {out_path}")
        return {"pair": pair, "status": "skipped", "out_file": str(out_path)}

    print(f"\nBuilding {pair}")
    print(f"  map: {map_path}")
    print(f"  out: {out_path}")

    ds = xr.open_dataset(map_path)

    row = arr(ds, "row").astype(np.int64) - 1
    col = arr(ds, "col").astype(np.int64) - 1
    S = arr(ds, "S").astype(np.float64)

    n_src = int(ds.sizes["n_a"])
    n_tgt = int(ds.sizes["n_b"])

    src_xyz, src_lon, src_lat = lonlat_to_xyz(arr(ds, "xc_a"), arr(ds, "yc_a"))
    tgt_xyz, tgt_lon, tgt_lat = lonlat_to_xyz(arr(ds, "xc_b"), arr(ds, "yc_b"))

    src_area = arr(ds, "area_a").astype(np.float64)
    tgt_area = arr(ds, "area_b").astype(np.float64)

    src_r = np.sqrt(src_area / np.pi)
    tgt_r = np.sqrt(tgt_area / np.pi)
    max_src_r = float(np.max(src_r))

    true_weight = {}
    true_by_tgt = [[] for _ in range(n_tgt)]
    for r, c, w in zip(row, col, S):
        ti = int(r)
        sj = int(c)
        ww = float(w)
        true_weight[(ti, sj)] = ww
        if ww != 0.0:
            true_by_tgt[ti].append(sj)

    tree = cKDTree(src_xyz)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    writer = None
    total_rows = 0
    total_pos = 0
    total_forced = 0

    for ti in range(n_tgt):
        if ti % 10000 == 0:
            print(f"  target {ti:,}/{n_tgt:,}; written rows {total_rows:,}", flush=True)

        kk = min(min_k, n_src)
        _, knn = tree.query(tgt_xyz[ti], k=kk)
        cand = set(int(j) for j in np.atleast_1d(knn))

        tx, ty, tz = tgt_xyz[ti]
        radius = alpha * (max_src_r + tgt_r[ti])
        nearby = tree.query_ball_point(tgt_xyz[ti], r=radius)

        for sj in nearby:
            sx, sy, sz = src_xyz[sj]
            dx = tx - sx
            dy = ty - sy
            dz = tz - sz
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if dist <= alpha * (src_r[sj] + tgt_r[ti]):
                cand.add(int(sj))

        before = len(cand)
        for sj in true_by_tgt[ti]:
            cand.add(int(sj))
        total_forced += len(cand) - before

        cand_list = np.array(sorted(cand), dtype=np.int64)
        dists = np.linalg.norm(src_xyz[cand_list] - tgt_xyz[ti], axis=1)
        order = np.argsort(dists)
        cand_list = cand_list[order]
        dists = dists[order]

        for rank, sj in enumerate(cand_list):
            sx, sy, sz = src_xyz[sj]
            dx = tx - sx
            dy = ty - sy
            dz = tz - sz
            w = true_weight.get((ti, int(sj)), 0.0)
            exists = 1 if w != 0.0 else 0
            total_pos += exists

            rows.append({
                "src_mesh": src_name,
                "tgt_mesh": tgt_name,
                "pair": pair,
                "target_index": int(ti),
                "source_index": int(sj),
                "knn_rank": int(rank),
                "edge_exists": int(exists),
                "weight": float(w),
                "src_x": float(sx),
                "src_y": float(sy),
                "src_z": float(sz),
                "tgt_x": float(tx),
                "tgt_y": float(ty),
                "tgt_z": float(tz),
                "dx": float(dx),
                "dy": float(dy),
                "dz": float(dz),
                "chord_dist": float(dists[rank]),
                "src_lon": float(src_lon[sj]),
                "src_lat": float(src_lat[sj]),
                "tgt_lon": float(tgt_lon[ti]),
                "tgt_lat": float(tgt_lat[ti]),
                "src_area": float(src_area[sj]),
                "tgt_area": float(tgt_area[ti]),
                "area_ratio_tgt_over_src": float(tgt_area[ti] / src_area[sj]),
            })

        if len(rows) >= flush_rows:
            writer, n = write_rows(rows, writer, out_path)
            total_rows += n
            rows = []

    writer, n = write_rows(rows, writer, out_path)
    total_rows += n
    if writer is not None:
        writer.close()

    ds.close()

    recall = total_pos / len(S) if len(S) else np.nan

    print(f"  n_src={n_src:,} n_tgt={n_tgt:,}")
    print(f"  map nnz={len(S):,}")
    print(f"  candidate edges={total_rows:,}")
    print(f"  positives in candidates={total_pos:,}")
    print(f"  recall={recall:.8f}")
    print(f"  true edges forced in={total_forced:,}")
    print(f"  wrote {out_path}")

    return {
        "pair": pair,
        "status": "built",
        "n_src": n_src,
        "n_tgt": n_tgt,
        "map_nnz": len(S),
        "candidate_edges": total_rows,
        "positive_edges_in_candidates": total_pos,
        "recall": recall,
        "true_edges_forced_in": total_forced,
        "out_file": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", default="maps_medium_improv")
    ap.add_argument("--out-dir", default="analysis_medium_improv")
    ap.add_argument("--levels", nargs="+", type=int, required=True)
    ap.add_argument("--direction", choices=["CS_to_ICOD", "ICOD_to_CS"], default="CS_to_ICOD")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--min-k", type=int, default=8)
    ap.add_argument("--flush-rows", type=int, default=200000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    map_dir = Path(args.map_dir)
    out_dir = Path(args.out_dir)

    summary = []
    for r in args.levels:
        if args.direction == "CS_to_ICOD":
            pair = f"CS-r{r}_to_ICOD-r{r}"
        else:
            pair = f"ICOD-r{r}_to_CS-r{r}"

        map_path = map_dir / f"map_{pair}_conserve.nc"
        if not map_path.exists():
            print(f"MISSING map: {map_path}")
            summary.append({"pair": pair, "status": "missing_map", "out_file": ""})
            continue

        summary.append(build_one(map_path, out_dir, args.alpha, args.min_k, args.flush_rows, args.overwrite))

    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = str(args.alpha).replace(".", "p")
    summary_path = out_dir / f"convergence_edge_dataset_summary_{args.direction}_a{alpha_tag}_mink{args.min_k}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
