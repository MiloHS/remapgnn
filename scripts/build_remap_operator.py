#!/usr/bin/env python
"""Build a learned conservative remap operator from a prepared edge graph.

This is the current user-facing inference path for the v12 learned base model:

  1. prepare a source-target candidate graph, usually with
     ``scripts/build_external_kdist_graph.py``;
  2. run this script to build a sparse conservative operator;
  3. optionally apply the operator to one source field.

The script intentionally starts from a prepared edge parquet rather than raw
mesh files.  Raw-mesh support is the graph-building step; this script is the
model/projection step.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "models_medium_improv/highorder_signed_v12_geom_mom1e4.pt"
DEFAULT_CONFIG = "configs/v20b_base_a3p0_mink8_geom_v12.json"


def torch_load(path: Path, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_projection_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unknown projection dtype {name!r}; expected float32 or float64")


def model_outputs_to_q(out):
    import torch

    if isinstance(out, dict):
        edge_logit = out.get("edge_logit", out.get("logit"))
        raw_weight = out.get("raw_weight", out.get("positive_weight"))
        q = out.get("q")
        if q is not None:
            return edge_logit, raw_weight, q
    elif isinstance(out, (tuple, list)):
        if len(out) == 3:
            return out
        if len(out) == 2:
            edge_logit, raw_weight = out
            q = torch.sqrt(torch.sigmoid(edge_logit).clamp_min(1.0e-12)) * torch.nn.functional.softplus(raw_weight)
            return edge_logit, raw_weight, q
    raise TypeError(f"Unsupported model output type/shape: {type(out)!r}")


def load_base_model(pack_path: Path, cfg, device):
    from remapgnn.models import build_model

    pack = torch_load(pack_path, map_location=device)
    if pack.get("kind") == "highorder_corrector":
        raise NotImplementedError(
            "build_remap_operator.py currently supports base packs such as v12_geom_base. "
            "Corrector-pack inference is not part of the cleaned public v12 path."
        )

    model = build_model(
        architecture=pack.get("architecture", cfg.architecture),
        src_dim=len(pack["src_node_features"]),
        tgt_dim=len(pack["tgt_node_features"]),
        edge_dim=len(pack["edge_features"]),
        hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
    ).to(device)
    model.load_state_dict(pack["model_state_dict"])
    model.eval()
    model.num_rounds = int(pack.get("rounds", 1))
    return model, pack


def build_operator(
    model,
    pack: dict,
    batch: dict,
    *,
    n_cg: int,
    projection_dtype,
    projection_eps_rel: float,
):
    import torch
    from remapgnn.models import scatter_sum_torch
    from remapgnn.projection import doubly_constrained_project

    si = batch["src_index"]
    ti = batch["tgt_index"]
    n_src = int(batch["n_src"])
    n_tgt = int(batch["n_tgt"])
    area_src = batch["area_src"].float()
    area_tgt = batch["area_tgt"].float()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            batch["src_node_attr"],
            batch["tgt_node_attr"],
            batch["edge_attr"],
            si,
            ti,
            n_src,
            n_tgt,
        )
        edge_logit, raw_weight, _q_unused = model_outputs_to_q(out)
        deg_t = scatter_sum_torch(torch.ones_like(raw_weight.float()), ti, n_tgt)
        M_base = area_tgt[ti] / torch.clamp(deg_t[ti], min=1.0)
        signed = bool(pack.get("signed", False))
        scale = float(pack.get("scale", 1.0))
        w = edge_logit.float() if signed else raw_weight.float()
        q = M_base * (1.0 + scale * w)
        M = doubly_constrained_project(
            q,
            si,
            ti,
            area_src,
            area_tgt,
            n_src,
            n_tgt,
            eps_rel=projection_eps_rel,
            n_cg=n_cg,
            solve_dtype=projection_dtype,
        )
        S = M / torch.clamp(area_tgt.to(dtype=M.dtype)[ti], min=1.0e-30)
    elapsed_s = time.perf_counter() - t0
    return S, M, elapsed_s


def scatter_numpy(n: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, index, values)
    return out


def residuals(M: np.ndarray, si: np.ndarray, ti: np.ndarray, area_src: np.ndarray, area_tgt: np.ndarray):
    src_sum = scatter_numpy(len(area_src), si, M)
    tgt_sum = scatter_numpy(len(area_tgt), ti, M)
    cons = float(np.linalg.norm(src_sum - area_src) / max(np.linalg.norm(area_src), 1.0e-30))
    row = float(np.linalg.norm(tgt_sum - area_tgt) / max(np.linalg.norm(area_tgt), 1.0e-30))
    return cons, row


def zero_degree_counts(si: np.ndarray, ti: np.ndarray, n_src: int, n_tgt: int):
    src_deg = np.bincount(si, minlength=n_src)
    tgt_deg = np.bincount(ti, minlength=n_tgt)
    return int(np.sum(src_deg == 0)), int(np.sum(tgt_deg == 0))


def write_npz(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata, indent=2)))


def write_netcdf_map(path: Path, arrays: dict, metadata: dict) -> None:
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ds.createDimension("n_s", arrays["S"].shape[0])
        ds.createDimension("n_a", arrays["area_src"].shape[0])
        ds.createDimension("n_b", arrays["area_tgt"].shape[0])

        ds.createVariable("S", "f8", ("n_s",))[:] = arrays["S"]
        # Tempest/SCRIP-style maps conventionally store 1-based row/col.
        ds.createVariable("row", "i8", ("n_s",))[:] = arrays["tgt_index"] + 1
        ds.createVariable("col", "i8", ("n_s",))[:] = arrays["src_index"] + 1
        ds.createVariable("area_a", "f8", ("n_a",))[:] = arrays["area_src"]
        ds.createVariable("area_b", "f8", ("n_b",))[:] = arrays["area_tgt"]

        for key, value in metadata.items():
            setattr(ds, key, str(value))


def read_flat_var(ds, names: list[str], expected: int):
    for name in names:
        if name in ds:
            arr = np.asarray(ds[name].values).reshape(-1)
            if arr.size == expected:
                return arr
    return None


def load_source_field(path: Path, field_name: str, n_src: int) -> np.ndarray:
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        if field_name not in ds:
            raise KeyError(f"Field {field_name!r} not found in {path}. Available variables: {list(ds.data_vars)}")
        values = np.asarray(ds[field_name].values, dtype=np.float64).reshape(-1)
    finally:
        ds.close()
    if values.size != n_src:
        raise ValueError(f"Source field has {values.size} values, but operator expects n_src={n_src}.")
    return values


def write_field_output(
    path: Path,
    field_name: str,
    y_tgt: np.ndarray,
    area_tgt: np.ndarray,
    target_mesh: Path | None,
    attrs: dict,
) -> None:
    import xarray as xr

    n_tgt = y_tgt.size
    data_vars = {
        field_name: (("cell",), y_tgt.astype(np.float64)),
        "cell_area": (("cell",), area_tgt.astype(np.float64)),
    }
    coords = {"cell": np.arange(n_tgt, dtype=np.int64)}

    if target_mesh is not None:
        ds = xr.open_dataset(target_mesh)
        try:
            lon = read_flat_var(ds, ["lon", "longitude", "lonCell", "xlon"], n_tgt)
            lat = read_flat_var(ds, ["lat", "latitude", "latCell", "ylat"], n_tgt)
        finally:
            ds.close()
        if lon is not None:
            data_vars["lon"] = (("cell",), lon)
        if lat is not None:
            data_vars["lat"] = (("cell",), lat)

    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)


def infer_pair_from_edge_path(edge_path: Path, fallback_pair: str | None) -> str:
    if fallback_pair:
        return fallback_pair
    stem = edge_path.stem
    if stem.startswith("edge_dataset_"):
        stem = stem[len("edge_dataset_") :]
    for suffix in [
        "_kdist_a3p0_mink8",
        "_kdist_a2p0_mink8",
        "_kdist_a4p0_mink16",
        "_kdist",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if "_to_" not in stem:
        raise ValueError("Could not infer --pair from edge parquet name; pass --pair explicitly.")
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to v12-style base model pack.")
    ap.add_argument("--edge-parquet", required=True, help="Prepared source-target candidate graph parquet.")
    ap.add_argument("--pair", default=None, help="Pair label, e.g. SRC_to_TGT. Inferred from edge filename if omitted.")
    ap.add_argument("--out-map", required=True, help="Output sparse map path. Use .npz or .nc.")
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-cg", type=int, default=800)
    ap.add_argument("--projection-dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--projection-eps-rel", type=float, default=1.0e-12)

    ap.add_argument("--src-field-nc", default=None, help="Optional source field NetCDF to remap.")
    ap.add_argument("--field", default=None, help="Variable name in --src-field-nc.")
    ap.add_argument("--out-field", default=None, help="Optional remapped target field NetCDF output.")
    ap.add_argument("--target-mesh-nc", default=None, help="Optional target mesh for lon/lat metadata in --out-field.")
    args = ap.parse_args()

    import torch
    from remapgnn.config import load_config
    from remapgnn.data import load_pair_tensors_from_path

    edge_path = Path(args.edge_parquet)
    pair = infer_pair_from_edge_path(edge_path, args.pair)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    projection_dtype = parse_projection_dtype(args.projection_dtype)

    print(f"Config: {args.config}")
    print(f"Model:  {args.model}")
    print(f"Pair:   {pair}")
    print(f"Graph:  {edge_path}")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    model, pack = load_base_model(Path(args.model), cfg, device)
    feature_lists = (
        list(pack["edge_features"]),
        list(pack["src_node_features"]),
        list(pack["tgt_node_features"]),
    )

    batch = load_pair_tensors_from_path(
        edge_path,
        cfg,
        pack["stats"],
        device=device,
        pair=pair,
        feature_lists=feature_lists,
    )
    n_src = int(batch["n_src"])
    n_tgt = int(batch["n_tgt"])
    n_edges = int(batch["n_edges"])
    print(f"n_src={n_src:,} n_tgt={n_tgt:,} n_edges={n_edges:,}")

    S_t, M_t, build_s = build_operator(
        model,
        pack,
        batch,
        n_cg=args.n_cg,
        projection_dtype=projection_dtype,
        projection_eps_rel=args.projection_eps_rel,
    )

    S = S_t.detach().cpu().numpy().astype(np.float64)
    M = M_t.detach().cpu().numpy().astype(np.float64)
    si = batch["src_index"].detach().cpu().numpy().astype(np.int64)
    ti = batch["tgt_index"].detach().cpu().numpy().astype(np.int64)
    area_src = batch["area_src"].detach().cpu().numpy().astype(np.float64)
    area_tgt = batch["area_tgt"].detach().cpu().numpy().astype(np.float64)

    cons_resid, row_resid = residuals(M, si, ti, area_src, area_tgt)
    zero_src, zero_tgt = zero_degree_counts(si, ti, n_src, n_tgt)

    metadata = {
        "remapgnn_model": str(args.model),
        "remapgnn_config": str(args.config),
        "remapgnn_pair": pair,
        "remapgnn_edge_parquet": str(edge_path),
        "projection_dtype": args.projection_dtype,
        "projection_eps_rel": args.projection_eps_rel,
        "projection_n_cg": args.n_cg,
        "operator_build_s": build_s,
        "n_src": n_src,
        "n_tgt": n_tgt,
        "n_edges": n_edges,
        "zero_degree_source_cells": zero_src,
        "zero_degree_target_cells": zero_tgt,
        "conservation_residual": cons_resid,
        "consistency_residual": row_resid,
        "note": "Research prototype learned conservative remap operator.",
    }
    arrays = {
        "S": S,
        "src_index": si,
        "tgt_index": ti,
        "area_src": area_src,
        "area_tgt": area_tgt,
    }

    out_map = Path(args.out_map)
    if out_map.suffix == ".npz":
        write_npz(out_map, arrays, metadata)
    elif out_map.suffix == ".nc":
        write_netcdf_map(out_map, arrays, metadata)
    else:
        raise ValueError("--out-map must end in .npz or .nc")
    print(f"Wrote map: {out_map}")

    if args.src_field_nc or args.field or args.out_field:
        if not (args.src_field_nc and args.field and args.out_field):
            raise ValueError("--src-field-nc, --field, and --out-field must be supplied together.")
        x_src = load_source_field(Path(args.src_field_nc), args.field, n_src)
        y_tgt = scatter_numpy(n_tgt, ti, S * x_src[si])
        write_field_output(
            Path(args.out_field),
            args.field,
            y_tgt,
            area_tgt,
            Path(args.target_mesh_nc) if args.target_mesh_nc else None,
            attrs=metadata,
        )
        print(f"Wrote remapped field: {args.out_field}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote summary: {args.summary_json}")

    print(
        "Audit: conservation_residual=%.3e consistency_residual=%.3e "
        "zero_degree_source=%d zero_degree_target=%d build_s=%.3f"
        % (cons_resid, row_resid, zero_src, zero_tgt, build_s)
    )


if __name__ == "__main__":
    main()
