#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


FOUR_PI = 4.0 * np.pi


def first_existing(ds: xr.Dataset, names: list[str]):
    for name in names:
        if name in ds:
            return name
    return None


def to_radians_if_needed(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(np.abs(finite)) > 2.0 * np.pi + 1.0e-6:
        return np.deg2rad(arr)
    return arr


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def read_lon_lat_area(ds: xr.Dataset):
    lon_name = first_existing(ds, ["lon", "longitude", "lonCell", "xlon"])
    lat_name = first_existing(ds, ["lat", "latitude", "latCell", "ylat"])
    area_name = first_existing(ds, ["cell_area", "area", "areaCell", "cellArea", "area_cell"])

    if lon_name is None or lat_name is None:
        raise KeyError(f"Could not find lon/lat variables. Available: {list(ds.variables)}")

    lon_raw = np.asarray(ds[lon_name].values)
    lat_raw = np.asarray(ds[lat_name].values)

    if area_name is not None:
        area_raw = np.asarray(ds[area_name].values, dtype=np.float64)
    else:
        area_raw = None

    # Case: RLL-style coordinate axes plus 2D area or field.
    if lon_raw.ndim == 1 and lat_raw.ndim == 1 and area_raw is not None and area_raw.size == lon_raw.size * lat_raw.size:
        LON, LAT = np.meshgrid(lon_raw, lat_raw)
        lon = LON.reshape(-1)
        lat = LAT.reshape(-1)
        area = area_raw.reshape(-1)
    else:
        lon = lon_raw.reshape(-1)
        lat = lat_raw.reshape(-1)
        if area_raw is not None:
            area = area_raw.reshape(-1)
        else:
            area = np.full(lon.size, FOUR_PI / lon.size, dtype=np.float64)
            print("WARNING: no area variable found; using uniform unit-sphere areas.")

    lon = to_radians_if_needed(lon)
    lat = to_radians_if_needed(lat)

    if area.size != lon.size:
        raise ValueError(f"Area size {area.size} does not match lon/lat size {lon.size}")

    return lon, lat, area


def read_xyz(ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray):
    x_name = first_existing(ds, ["x", "xCell", "cell_x"])
    y_name = first_existing(ds, ["y", "yCell", "cell_y"])
    z_name = first_existing(ds, ["z", "zCell", "cell_z"])

    if x_name and y_name and z_name:
        x = np.asarray(ds[x_name].values, dtype=np.float64).reshape(-1)
        y = np.asarray(ds[y_name].values, dtype=np.float64).reshape(-1)
        z = np.asarray(ds[z_name].values, dtype=np.float64).reshape(-1)
        xyz = np.stack([x, y, z], axis=1)
        norm = np.linalg.norm(xyz, axis=1)
        if np.nanmedian(norm) > 0:
            xyz = xyz / norm[:, None]
        return xyz.astype(np.float64)

    return lonlat_to_xyz(lon, lat)


def load_mesh(path: Path, normalize_area_sum: bool):
    ds = xr.open_dataset(path)
    lon, lat, area = read_lon_lat_area(ds)
    xyz = read_xyz(ds, lon, lat)
    ds.close()

    if normalize_area_sum:
        area = area * (FOUR_PI / np.sum(area))

    if xyz.shape[0] != area.size:
        raise ValueError(f"xyz count {xyz.shape[0]} does not match area count {area.size}")

    return {
        "lon": lon.astype(np.float64),
        "lat": lat.astype(np.float64),
        "xyz": xyz.astype(np.float64),
        "area": area.astype(np.float64),
    }


def build_edges(src, tgt, alpha: float, min_k: int, max_k: int):
    tree = cKDTree(src["xyz"])
    n_tgt = tgt["xyz"].shape[0]

    rows = []
    for tgt_i in range(n_tgt):
        point = tgt["xyz"][tgt_i]

        # First get min_k nearest neighbors.
        k_query = min(max(min_k, 1), src["xyz"].shape[0])
        d_knn, idx_knn = tree.query(point, k=k_query)

        idx_knn = np.atleast_1d(idx_knn).astype(np.int64)
        d_knn = np.atleast_1d(d_knn).astype(np.float64)

        radius_base = float(d_knn[0])
        if radius_base <= 0 and d_knn.size > 1:
            radius_base = float(d_knn[-1])
        radius = max(alpha * radius_base, float(d_knn[-1]))

        cand = tree.query_ball_point(point, r=radius)

        # Always include min_k nearest.
        cand = set(int(i) for i in cand)
        cand.update(int(i) for i in idx_knn)

        cand = np.array(sorted(cand), dtype=np.int64)
        d = np.linalg.norm(src["xyz"][cand] - point[None, :], axis=1)
        order = np.argsort(d)

        if max_k > 0 and order.size > max_k:
            order = order[:max_k]

        cand = cand[order]
        d = d[order]

        for rank, (src_j, dist) in enumerate(zip(cand, d)):
            sx, sy, sz = src["xyz"][src_j]
            tx, ty, tz = tgt["xyz"][tgt_i]
            dx = tx - sx
            dy = ty - sy
            dz = tz - sz

            rows.append(
                {
                    "target_index": int(tgt_i),
                    "source_index": int(src_j),
                    "knn_rank": int(rank),
                    "edge_exists": 0,
                    "weight": 0.0,
                    "src_x": float(sx),
                    "src_y": float(sy),
                    "src_z": float(sz),
                    "tgt_x": float(tx),
                    "tgt_y": float(ty),
                    "tgt_z": float(tz),
                    "dx": float(dx),
                    "dy": float(dy),
                    "dz": float(dz),
                    "chord_dist": float(dist),
                    "src_lon": float(src["lon"][src_j]),
                    "src_lat": float(src["lat"][src_j]),
                    "tgt_lon": float(tgt["lon"][tgt_i]),
                    "tgt_lat": float(tgt["lat"][tgt_i]),
                    "src_area": float(src["area"][src_j]),
                    "tgt_area": float(tgt["area"][tgt_i]),
                    "area_ratio_tgt_over_src": float(tgt["area"][tgt_i] / src["area"][src_j]),
                }
            )

        if (tgt_i + 1) % 10000 == 0:
            print(f"  built target {tgt_i + 1:,}/{n_tgt:,}; edges so far={len(rows):,}")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-mesh", required=True)
    ap.add_argument("--tgt-mesh", required=True)
    ap.add_argument("--src-name", required=True)
    ap.add_argument("--tgt-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--min-k", type=int, default=8)
    ap.add_argument("--max-k", type=int, default=256)
    ap.add_argument("--normalize-area-sums", action="store_true")
    args = ap.parse_args()

    src_mesh = Path(args.src_mesh)
    tgt_mesh = Path(args.tgt_mesh)
    out_path = Path(args.out)

    print(f"Loading source mesh: {src_mesh}")
    src = load_mesh(src_mesh, normalize_area_sum=args.normalize_area_sums)

    print(f"Loading target mesh: {tgt_mesh}")
    tgt = load_mesh(tgt_mesh, normalize_area_sum=args.normalize_area_sums)

    print(f"n_src={src['xyz'].shape[0]:,} n_tgt={tgt['xyz'].shape[0]:,}")
    print(f"source area sum={src['area'].sum():.12e}")
    print(f"target area sum={tgt['area'].sum():.12e}")

    if abs(src["area"].sum() - tgt["area"].sum()) / max(abs(src["area"].sum()), 1e-300) > 1e-6:
        print("WARNING: source and target area sums differ. Consider --normalize-area-sums.")

    print("Building k-distance candidate graph.")
    df = build_edges(src, tgt, alpha=args.alpha, min_k=args.min_k, max_k=args.max_k)

    pair = f"{args.src_name}_to_{args.tgt_name}"
    df.insert(0, "pair", pair)
    df.insert(0, "tgt_mesh", args.tgt_name)
    df.insert(0, "src_mesh", args.src_name)

    ordered_cols = [
        "src_mesh", "tgt_mesh", "pair",
        "target_index", "source_index", "knn_rank",
        "edge_exists", "weight",
        "src_x", "src_y", "src_z",
        "tgt_x", "tgt_y", "tgt_z",
        "dx", "dy", "dz", "chord_dist",
        "src_lon", "src_lat", "tgt_lon", "tgt_lat",
        "src_area", "tgt_area", "area_ratio_tgt_over_src",
    ]
    df = df[ordered_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"edges={len(df):,}")
    print(f"avg edges per target={len(df) / tgt['xyz'].shape[0]:.2f}")
    print(f"max rank={df['knn_rank'].max()}")


if __name__ == "__main__":
    main()
