#!/usr/bin/env python
"""Sweep projection CG iterations for learned remap operators.

This is an evaluation-only conservation cleanup tool: it rebuilds each learned
operator at several projection CG counts, then records marginal residuals and
real-field errors.  No model weights are changed.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from remapgnn.projection import (
    doubly_constrained_project_implicit,
    doubly_constrained_project_local_moment,
)
from train_config_balanced_harmonic import model_outputs_to_q, read_source_xyz_from_edges, read_target_xyz_from_edges
from train_config_highorder import quadratic_moment_coef
from train_config_highorder_corrector import base_w_and_geom, run_corrector_steps
from train_config_irno_corrector import as_int, torch_load_pack


DEFAULT_FIELDS = [
    "AnalyticalFun1",
    "AnalyticalFun2",
    "TotalPrecipWater",
    "CloudFraction",
    "Topography",
]


@dataclass
class LearnedSpec:
    label: str
    pack_path: Path
    cfg_path: Path | None


@dataclass
class LearnedOp:
    label: str
    cfg: object
    pack: dict
    model: torch.nn.Module | None
    base_model: torch.nn.Module | None
    corrector: torch.nn.Module | None
    is_corrector: bool
    signed: bool


def parse_spec(spec: str) -> LearnedSpec:
    if "=" in spec:
        label, rest = spec.split("=", 1)
    else:
        label, rest = Path(spec.split("@", 1)[0]).stem, spec
    if "@" in rest:
        pack_s, cfg_s = rest.rsplit("@", 1)
        cfg_path = Path(cfg_s)
    else:
        pack_s, cfg_path = rest, None
    return LearnedSpec(label=label, pack_path=Path(pack_s), cfg_path=cfg_path)


def resolve_cfg_path(spec_cfg: Path | None, pack: dict, default_cfg: Path) -> Path:
    if spec_cfg is not None:
        return spec_cfg
    meta = pack.get("config_path")
    if meta and Path(meta).exists():
        return Path(meta)
    return default_cfg


def load_model_from_pack(pack: dict, cfg, device: torch.device, edge_dim_extra: int = 0) -> torch.nn.Module:
    model = build_model(
        architecture=pack.get("architecture", cfg.architecture),
        src_dim=len(pack["src_node_features"]),
        tgt_dim=len(pack["tgt_node_features"]),
        edge_dim=len(pack["edge_features"]) + edge_dim_extra,
        hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
    ).to(device)
    model.load_state_dict(pack["model_state_dict"])
    model.eval()
    model.num_rounds = int(pack.get("rounds", 1))
    return model


def load_learned_op(spec: LearnedSpec, default_cfg: Path, device: torch.device) -> LearnedOp:
    pack = torch_load_pack(spec.pack_path, map_location=device)
    cfg_path = resolve_cfg_path(spec.cfg_path, pack, default_cfg)
    cfg = load_config(cfg_path)
    is_corrector = pack.get("kind") == "highorder_corrector"
    if is_corrector:
        base_pack = torch_load_pack(pack["base_pack"], map_location=device)
        base_model = load_model_from_pack(base_pack, cfg, device)
        corrector = load_model_from_pack(pack, cfg, device, edge_dim_extra=4)
        model = None
        signed = True
    else:
        model = load_model_from_pack(pack, cfg, device)
        base_model = None
        corrector = None
        signed = bool(pack.get("signed", False))
    print(f"loaded {spec.label}: {spec.pack_path} cfg={cfg_path} corrector={is_corrector}")
    return LearnedOp(
        label=spec.label,
        cfg=cfg,
        pack=pack,
        model=model,
        base_model=base_model,
        corrector=corrector,
        is_corrector=is_corrector,
        signed=signed,
    )


def parse_projection_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unknown projection dtype {name!r}; expected float32 or float64")


def base_q_and_geom(op: LearnedOp, b: dict, n_src: int, n_tgt: int, scale: float):
    assert op.model is not None
    out = op.model(
        b["src_node_attr"],
        b["tgt_node_attr"],
        b["edge_attr"],
        b["src_index"],
        b["tgt_index"],
        n_src,
        n_tgt,
    )
    logit, raw_weight, _q = model_outputs_to_q(out)
    ti = b["tgt_index"]
    deg_t = scatter_sum_torch(torch.ones_like(raw_weight.float()), ti, n_tgt)
    M_base = b["area_tgt"].float()[ti] / torch.clamp(deg_t[ti], min=1.0)
    w = logit.float() if op.signed else raw_weight.float()
    q = M_base * (1.0 + scale * w)
    return q


def conservation_resid_np(M: np.ndarray, si: np.ndarray, ti: np.ndarray, area_src: np.ndarray, area_tgt: np.ndarray):
    sm = np.zeros(len(area_src), dtype=np.float64)
    tm = np.zeros(len(area_tgt), dtype=np.float64)
    np.add.at(sm, si, M)
    np.add.at(tm, ti, M)
    cons = float(np.linalg.norm(sm - area_src) / max(np.linalg.norm(area_src), 1.0e-30))
    row = float(np.linalg.norm(tm - area_tgt) / max(np.linalg.norm(area_tgt), 1.0e-30))
    return cons, row


def area_rel_l2(pred: np.ndarray, truth: np.ndarray, area: np.ndarray) -> float:
    num = float(np.sum(area * (pred - truth) ** 2))
    den = max(float(np.sum(area * truth * truth)), 1.0e-30)
    return math.sqrt(num / den)


def load_field(path: Path, field: str) -> np.ndarray:
    with Dataset(path) as ds:
        if field not in ds.variables:
            raise KeyError(f"{field} not found in {path}")
        return np.asarray(ds.variables[field][:], dtype=np.float64).reshape(-1)


def apply_sparse(S: np.ndarray, si: np.ndarray, ti: np.ndarray, n_tgt: int, src_field: np.ndarray) -> np.ndarray:
    out = np.zeros(n_tgt, dtype=np.float64)
    np.add.at(out, ti, S * src_field[si])
    return out


def load_tempest_operator(cfg, pair: str, order: str):
    suffix = "" if order == "np1" else "_np2"
    path = cfg.maps_dir / f"map_{pair}_conserve{suffix}.nc"
    with Dataset(path) as ds:
        S = np.asarray(ds.variables["S"][:], dtype=np.float64).reshape(-1)
        ti = np.asarray(ds.variables["row"][:], dtype=np.int64).reshape(-1) - 1
        si = np.asarray(ds.variables["col"][:], dtype=np.int64).reshape(-1) - 1
        area_src = np.asarray(ds.variables["area_a"][:], dtype=np.float64).reshape(-1)
        area_tgt = np.asarray(ds.variables["area_b"][:], dtype=np.float64).reshape(-1)
    M = S * area_tgt[ti]
    return S, M, si, ti, area_src, area_tgt


def learned_operator_arrays(
    op: LearnedOp,
    b: dict,
    n_cg: int,
    device: torch.device,
    projection_dtype,
    projection_eps_rel: float,
):
    si_t = b["src_index"]
    ti_t = b["tgt_index"]
    n_src = as_int(b["n_src"])
    n_tgt = as_int(b["n_tgt"])
    asrc_t = b["area_src"].float()
    atgt_t = b["area_tgt"].float()
    t0 = time.perf_counter()
    with torch.no_grad():
        if op.is_corrector:
            p = op.pack
            bands = [int(x) for x in p["bands"]]
            alpha = [float(x) for x in p["step_alphas"]] if p.get("step_alphas") else float(p["alpha"])
            lmax_denom = float(p.get("lmax_denom", 32.0))
            moment_coef = None
            if bool(p.get("moment_l1_hard", False)):
                sx = read_source_xyz_from_edges(op.cfg.edge_path(b["pair"]), n_src)
                tx = read_target_xyz_from_edges(op.cfg.edge_path(b["pair"]), n_tgt)
                moment_coef = (
                    torch.tensor(sx, dtype=torch.float32, device=device)[si_t]
                    - torch.tensor(tx, dtype=torch.float32, device=device)[ti_t]
                )
            w0, M_base = base_w_and_geom(op.base_model, b, n_src, n_tgt)
            _S0, steps, _w = run_corrector_steps(
                op.base_model,
                op.corrector,
                b,
                bands,
                alpha,
                float(p.get("scale", 1.0)),
                int(n_cg),
                lmax_denom,
                w0=w0,
                M_base=M_base,
                use_ckpt=False,
                moment_coef=moment_coef,
                solve_dtype=projection_dtype,
                eps_rel=projection_eps_rel,
            )
            S_t, M_t = steps[-1][0], steps[-1][1]
        else:
            q = base_q_and_geom(op, b, n_src, n_tgt, float(op.pack.get("scale", 1.0)))
            moment_mode = op.pack.get("moment_mode") or (
                "local_soft_l2" if bool(op.pack.get("moment_l2_local_soft", False))
                else "local_soft" if bool(op.pack.get("moment_l1_local_soft", False))
                else ("hard" if bool(op.pack.get("moment_l1_hard", False)) else "none")
            )
            moment_mode = str(moment_mode)
            moment_coef = None
            moment_coef2 = None
            if moment_mode != "none":
                sx = read_source_xyz_from_edges(op.cfg.edge_path(b["pair"]), n_src)
                tx = read_target_xyz_from_edges(op.cfg.edge_path(b["pair"]), n_tgt)
                sxyz_t = torch.tensor(sx, dtype=torch.float32, device=device)
                txyz_t = torch.tensor(tx, dtype=torch.float32, device=device)
                moment_coef = (
                    sxyz_t[si_t]
                    - txyz_t[ti_t]
                )
                if moment_mode == "local_soft_l2":
                    moment_coef2 = quadratic_moment_coef(sxyz_t, txyz_t, si_t, ti_t)
            if moment_mode in ("local_soft", "local_soft_l2"):
                M_t = doubly_constrained_project_local_moment(
                    q, si_t, ti_t, asrc_t, atgt_t, n_src, n_tgt,
                    eps_rel=projection_eps_rel,
                    n_cg=int(n_cg),
                    solve_dtype=projection_dtype,
                    moment_coef=moment_coef,
                    moment_ridge=float(op.pack.get("moment_ridge", 1.0e-4)),
                    moment_relax=float(op.pack.get("moment_relax", 1.0)),
                    moment_iters=int(op.pack.get("moment_iters", 1)),
                    moment_coef2=moment_coef2,
                    moment2_ridge=float(op.pack.get("moment2_ridge", 1.0e-3)),
                    moment2_relax=float(op.pack.get("moment2_relax", 0.5)),
                    moment2_iters=int(op.pack.get("moment2_iters", 1)),
                    use_implicit=True,
                )
            else:
                M_t = doubly_constrained_project_implicit(
                    q, si_t, ti_t, asrc_t, atgt_t, n_src, n_tgt,
                    eps_rel=projection_eps_rel,
                    n_cg=int(n_cg),
                    solve_dtype=projection_dtype,
                    moment_coef=(moment_coef if moment_mode == "hard" else None),
                )
            S_t = M_t / torch.clamp(atgt_t[ti_t], min=1.0e-30)
    elapsed = time.perf_counter() - t0
    return (
        S_t.detach().cpu().numpy().astype(np.float64),
        M_t.detach().cpu().numpy().astype(np.float64),
        si_t.detach().cpu().numpy().astype(np.int64),
        ti_t.detach().cpu().numpy().astype(np.int64),
        asrc_t.detach().cpu().numpy().astype(np.float64),
        atgt_t.detach().cpu().numpy().astype(np.float64),
        elapsed,
    )


def _markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer.

    pandas.to_markdown() requires the optional ``tabulate`` package, which is
    not installed in the cluster env. Keep this script self-contained so audit
    jobs do not fail after doing the expensive operator work.
    """
    if df.empty:
        return ""

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4e}"
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[j]), *(len(row[j]) for row in rows))
        for j in range(len(headers))
    ]
    header = "| " + " | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def summarize(
    out_dir: Path,
    field_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    *,
    projection_dtype: str,
    projection_eps_rel: float,
) -> None:
    lines = ["# Projection/conservation sweep\n"]
    lines.append(f"- projection dtype: `{projection_dtype}`")
    lines.append(f"- projection eps_rel: `{projection_eps_rel:.3e}`")
    lines.append("")
    if not field_df.empty:
        lines.append("## Real-field error summary\n")
        summary = (
            field_df.groupby(["operator", "n_cg"], as_index=False)
            .agg(mean_area_rel_l2=("area_rel_l2", "mean"), worst_area_rel_l2=("area_rel_l2", "max"), rows=("area_rel_l2", "size"))
            .sort_values(["mean_area_rel_l2", "operator", "n_cg"])
        )
        lines.append(_markdown_table(summary))
        lines.append("")
    if not cons_df.empty:
        lines.append("## Conservation/consistency summary\n")
        summary = (
            cons_df.groupby(["operator", "n_cg"], as_index=False)
            .agg(
                max_conservation_resid=("conservation_resid", "max"),
                max_consistency_resid=("consistency_resid", "max"),
                mean_conservation_resid=("conservation_resid", "mean"),
                rows=("conservation_resid", "size"),
            )
            .sort_values(["operator", "n_cg"])
        )
        lines.append(_markdown_table(summary))
        lines.append("")
    if not runtime_df.empty:
        lines.append("## Runtime summary\n")
        summary = (
            runtime_df.groupby(["operator", "n_cg"], as_index=False)
            .agg(mean_build_s=("operator_build_s", "mean"), max_build_s=("operator_build_s", "max"))
            .sort_values(["operator", "n_cg"])
        )
        lines.append(_markdown_table(summary))
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--packs", nargs="+", required=True, help="label=pack.pt[@config.json]")
    ap.add_argument("--n-cg-values", nargs="+", type=int, default=[200, 400, 800, 1200, 1600])
    ap.add_argument("--real-fields", nargs="+", default=DEFAULT_FIELDS)
    ap.add_argument("--include-tempest", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--projection-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--projection-eps-rel", type=float, default=1e-9)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    default_cfg_path = Path(args.config)
    default_cfg = load_config(default_cfg_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    projection_dtype = parse_projection_dtype(args.projection_dtype)
    print("device:", device)
    print("projection_dtype:", args.projection_dtype)
    print("projection_eps_rel:", args.projection_eps_rel)
    print("out_dir:", out_dir)
    print("n_cg_values:", args.n_cg_values)

    learned_ops = [load_learned_op(parse_spec(s), default_cfg_path, device) for s in args.packs]

    field_rows = []
    cons_rows = []
    runtime_rows = []

    for pair in args.pairs:
        print(f"\n=== pair {pair} ===")
        src_file, tgt_file = default_cfg.source_target_files(pair)
        src_fields = {f: load_field(src_file, f) for f in args.real_fields}
        tgt_fields = {f: load_field(tgt_file, f) for f in args.real_fields}

        if args.include_tempest:
            for order in ["np1", "np2"]:
                t0 = time.perf_counter()
                S, M, si, ti, area_src, area_tgt = load_tempest_operator(default_cfg, pair, order)
                elapsed = time.perf_counter() - t0
                cr, rr = conservation_resid_np(M, si, ti, area_src, area_tgt)
                cons_rows.append(
                    dict(pair=pair, operator=order, n_cg=0, conservation_resid=cr, consistency_resid=rr, n_edges=len(S))
                )
                runtime_rows.append(dict(pair=pair, operator=order, n_cg=0, operator_build_s=elapsed, n_edges=len(S)))
                for field in args.real_fields:
                    pred = apply_sparse(S, si, ti, len(area_tgt), src_fields[field])
                    field_rows.append(
                        dict(pair=pair, operator=order, n_cg=0, field=field, area_rel_l2=area_rel_l2(pred, tgt_fields[field], area_tgt))
                    )

        for op in learned_ops:
            print(f"  learned {op.label}")
            b = load_pair_tensors(op.cfg, pair, op.pack["stats"], device=device)
            for n_cg in args.n_cg_values:
                S, M, si, ti, area_src, area_tgt, elapsed = learned_operator_arrays(
                    op, b, int(n_cg), device, projection_dtype, float(args.projection_eps_rel)
                )
                cr, rr = conservation_resid_np(M, si, ti, area_src, area_tgt)
                cons_rows.append(
                    dict(pair=pair, operator=op.label, n_cg=int(n_cg), conservation_resid=cr, consistency_resid=rr, n_edges=len(S))
                )
                runtime_rows.append(dict(pair=pair, operator=op.label, n_cg=int(n_cg), operator_build_s=elapsed, n_edges=len(S)))
                print(f"    n_cg={n_cg:5d} cons={cr:.3e} row={rr:.3e} build_s={elapsed:.3f}")
                for field in args.real_fields:
                    pred = apply_sparse(S, si, ti, len(area_tgt), src_fields[field])
                    field_rows.append(
                        dict(
                            pair=pair,
                            operator=op.label,
                            n_cg=int(n_cg),
                            field=field,
                            area_rel_l2=area_rel_l2(pred, tgt_fields[field], area_tgt),
                        )
                    )
            del b
            if device.type == "cuda":
                torch.cuda.empty_cache()

    field_df = pd.DataFrame(field_rows)
    cons_df = pd.DataFrame(cons_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    field_df.to_csv(out_dir / "field_metrics.csv", index=False)
    cons_df.to_csv(out_dir / "conservation.csv", index=False)
    runtime_df.to_csv(out_dir / "runtime.csv", index=False)
    summarize(
        out_dir,
        field_df,
        cons_df,
        runtime_df,
        projection_dtype=args.projection_dtype,
        projection_eps_rel=float(args.projection_eps_rel),
    )
    print(f"wrote {out_dir / 'field_metrics.csv'}")
    print(f"wrote {out_dir / 'conservation.csv'}")
    print(f"wrote {out_dir / 'runtime.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print("PROJECTION_SWEEP_DONE")


if __name__ == "__main__":
    main()
