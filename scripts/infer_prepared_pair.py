#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_refinement_convergence import (
    load_config,
    load_pair_geometry_and_tempest,
    try_load_irno,
    get_irno_states,
    scatter_numpy,
)


STAGE_ALIASES = {
    "base": "base",
    "lmax8": "corrected_lmax8",
    "lmax16": "corrected_lmax16",
    "lmax24": "corrected_lmax24",
    "corrected_lmax8": "corrected_lmax8",
    "corrected_lmax16": "corrected_lmax16",
    "corrected_lmax24": "corrected_lmax24",
}


def select_state(states, wanted_label: str):
    label_to_index = {
        "base": 0,
        "corrected_lmax8": 1,
        "corrected_lmax16": 2,
        "corrected_lmax24": 3,
    }

    for state in states:
        if state.get("step_label") == wanted_label:
            return state

    if wanted_label in label_to_index:
        idx = label_to_index[wanted_label]
        if idx < len(states):
            return states[idx]

    labels = [s.get("step_label") for s in states]
    raise ValueError(
        f"Could not find stage {wanted_label}. "
        f"Available labels={labels}; number of states={len(states)}"
    )


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_state_weights(state):
    for key in ["S", "weights", "S_pred", "remap_weights", "edge_weights"]:
        if key in state:
            return to_numpy(state[key]).astype(np.float64)

    raise KeyError(f"Could not find remap weights in state. Available keys: {list(state.keys())}")


def load_source_field(path: Path, field_name: str, n_src: int) -> np.ndarray:
    ds = xr.open_dataset(path)
    if field_name not in ds:
        raise KeyError(f"Field {field_name!r} not found in {path}. Available variables: {list(ds.data_vars)}")

    values = np.asarray(ds[field_name].values, dtype=np.float64).reshape(-1)
    ds.close()

    if values.size != n_src:
        raise ValueError(
            f"Source field size mismatch: field has {values.size} values, "
            f"but graph expects n_src={n_src}."
        )

    return values


def read_flat_var(ds: xr.Dataset, name: str, expected: int):
    if name not in ds:
        return None
    arr = np.asarray(ds[name].values).reshape(-1)
    if arr.size != expected:
        return None
    return arr


def write_output(
    out_path: Path,
    field_name: str,
    y_tgt: np.ndarray,
    geom: dict,
    target_mesh_path: Path | None,
    attrs: dict,
):
    n_tgt = int(geom["n_tgt"])

    data_vars = {
        field_name: (("cell",), y_tgt.astype(np.float64)),
    }

    coords = {
        "cell": np.arange(n_tgt, dtype=np.int64),
    }

    if target_mesh_path is not None:
        tds = xr.open_dataset(target_mesh_path)

        lon = read_flat_var(tds, "lon", n_tgt)
        lat = read_flat_var(tds, "lat", n_tgt)
        area = read_flat_var(tds, "cell_area", n_tgt)

        if lon is not None:
            data_vars["lon"] = (("cell",), lon)
        if lat is not None:
            data_vars["lat"] = (("cell",), lat)
        if area is not None:
            data_vars["cell_area"] = (("cell",), area)

        tds.close()
    else:
        if "tgt_area" in geom:
            data_vars["cell_area"] = (("cell",), np.asarray(geom["tgt_area"], dtype=np.float64))

    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)



def load_geometry_from_edge_parquet(edge_parquet: Path) -> dict:
    df = pd.read_parquet(edge_parquet)

    required = [
        "source_index", "target_index",
        "src_area", "tgt_area",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {edge_parquet}: {missing}")

    src_index = df["source_index"].to_numpy(np.int64)
    tgt_index = df["target_index"].to_numpy(np.int64)

    n_src = int(src_index.max()) + 1
    n_tgt = int(tgt_index.max()) + 1

    src_area = np.zeros(n_src, dtype=np.float64)
    tgt_area = np.zeros(n_tgt, dtype=np.float64)

    src_area[src_index] = df["src_area"].to_numpy(np.float64)
    tgt_area[tgt_index] = df["tgt_area"].to_numpy(np.float64)

    geom = {
        "src_index": src_index,
        "tgt_index": tgt_index,
        "src_area": src_area,
        "tgt_area": tgt_area,
        "n_src": n_src,
        "n_tgt": n_tgt,
    }

    # Optional coordinate arrays for diagnostics/visualization.
    if {"src_x", "src_y", "src_z"}.issubset(df.columns):
        src_xyz = np.zeros((n_src, 3), dtype=np.float64)
        src_xyz[src_index, 0] = df["src_x"].to_numpy(np.float64)
        src_xyz[src_index, 1] = df["src_y"].to_numpy(np.float64)
        src_xyz[src_index, 2] = df["src_z"].to_numpy(np.float64)
        geom["src_xyz"] = src_xyz

    if {"tgt_x", "tgt_y", "tgt_z"}.issubset(df.columns):
        tgt_xyz = np.zeros((n_tgt, 3), dtype=np.float64)
        tgt_xyz[tgt_index, 0] = df["tgt_x"].to_numpy(np.float64)
        tgt_xyz[tgt_index, 1] = df["tgt_y"].to_numpy(np.float64)
        tgt_xyz[tgt_index, 2] = df["tgt_z"].to_numpy(np.float64)
        geom["tgt_xyz"] = tgt_xyz

    return geom


def main():
    ap = argparse.ArgumentParser(
        description="Run v18 remapgnn inference on a prepared source-target edge dataset."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--src-field-nc", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-mesh-nc", default=None)
    ap.add_argument("--edge-parquet", default=None, help="Optional external prepared edge parquet for geometry/output sizing.")
    ap.add_argument("--out-map", default=None)
    ap.add_argument("--stage", default="lmax24")
    ap.add_argument("--balance-iters", type=int, default=2000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stage_label = STAGE_ALIASES.get(args.stage, args.stage)

    print(f"Config: {args.config}")
    print(f"Pair:   {args.pair}")
    print(f"Field:  {args.field}")
    print(f"Stage:  {stage_label}")
    print(f"Device: {device}")

    print("\nLoading geometry and prepared edge graph.")
    if args.edge_parquet:
        print(f"Using external edge parquet: {args.edge_parquet}")
        geom = load_geometry_from_edge_parquet(Path(args.edge_parquet))
    else:
        geom = load_pair_geometry_and_tempest(cfg, args.pair)

    n_src = int(geom["n_src"])
    n_tgt = int(geom["n_tgt"])
    print(f"n_src={n_src:,} n_tgt={n_tgt:,} n_edges={len(geom['src_index']):,}")

    print("\nLoading source field.")
    x_src = load_source_field(Path(args.src_field_nc), args.field, n_src)

    print("\nLoading model and computing learned operator.")
    irno = try_load_irno(cfg, device)
    _, states = get_irno_states(cfg, irno, args.pair, args.balance_iters, device)
    state = select_state(states, stage_label)
    S = get_state_weights(state)

    print("\nApplying sparse learned operator.")
    y_tgt = scatter_numpy(
        n_tgt,
        geom["tgt_index"],
        S * x_src[geom["src_index"]],
    )

    attrs = {
        "remapgnn_pair": args.pair,
        "remapgnn_config": args.config,
        "remapgnn_stage": stage_label,
        "remapgnn_balance_iters": args.balance_iters,
        "source_field_file": args.src_field_nc,
        "source_field_name": args.field,
        "note": "Research inference output from prepared source-target graph.",
    }

    print(f"\nWriting remapped field: {args.out}")
    write_output(
        out_path=Path(args.out),
        field_name=args.field,
        y_tgt=y_tgt,
        geom=geom,
        target_mesh_path=Path(args.target_mesh_nc) if args.target_mesh_nc else None,
        attrs=attrs,
    )

    if args.out_map:
        print(f"Writing learned sparse operator: {args.out_map}")
        out_map = Path(args.out_map)
        out_map.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_map,
            source_index=np.asarray(geom["src_index"], dtype=np.int64),
            target_index=np.asarray(geom["tgt_index"], dtype=np.int64),
            S=np.asarray(S, dtype=np.float64),
            n_src=np.asarray([n_src], dtype=np.int64),
            n_tgt=np.asarray([n_tgt], dtype=np.int64),
            stage=np.asarray([stage_label]),
        )

    print("\nDone.")
    print(f"Output field min={np.nanmin(y_tgt):.6e} max={np.nanmax(y_tgt):.6e} mean={np.nanmean(y_tgt):.6e}")


if __name__ == "__main__":
    main()
