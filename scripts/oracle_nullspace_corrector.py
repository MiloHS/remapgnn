#!/usr/bin/env python
"""Oracle nullspace-corrector headroom test.

This is not a trainable corrector.  For a frozen learned operator it solves the
best high-band harmonic residual correction that is constrained to stay in the
nullspace of selected remap constraints:

  n0   : source/target marginals only
  n01  : marginals + target-local degree-1 moments
  n012 : marginals + target-local degree-1 and degree-2 Cartesian moments

The point is to test whether the current candidate graph has useful remaining
degrees of freedom for a future constrained corrector.  If the oracle cannot
improve high bands, a trainable nullspace corrector is unlikely to help.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from remapgnn.projection import _local_target_moment_correction, doubly_constrained_project_implicit
from train_config_balanced_harmonic import (
    _real_sph_unnorm,
    _stable_pair_seed,
    choose_m_values,
    model_outputs_to_q,
    read_source_xyz_from_edges,
    read_target_xyz_from_edges,
)
from train_config_highorder import operator_from_model, quadratic_moment_coef
from train_config_irno_corrector import as_int, torch_load_pack


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unknown dtype {name!r}")


def field_label(l: int, m: int) -> str:
    return f"Y_{l}_{m}" if m >= 0 else f"Y_{l}_m{abs(m)}"


def build_harmonic_fields(
    pair: str,
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    degrees: list[int],
    modes_per_degree: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    rng = np.random.default_rng(seed + _stable_pair_seed(pair) % 1000000)
    src_fields = []
    tgt_fields = []
    labels = []
    degrees_out = []
    modes_out = []
    for l in degrees:
        for m in choose_m_values(l, modes_per_degree, rng):
            ys = _real_sph_unnorm(l, m, src_xyz)
            yt = _real_sph_unnorm(l, m, tgt_xyz)
            norm = float(np.sqrt(np.mean(ys * ys)))
            if norm > 0.0:
                ys = ys / norm
                yt = yt / norm
            src_fields.append(ys.astype("float64"))
            tgt_fields.append(yt.astype("float64"))
            labels.append(field_label(l, m))
            degrees_out.append(int(l))
            modes_out.append(int(m))
    src = torch.tensor(np.stack(src_fields, axis=0), device=device, dtype=dtype)
    tgt = torch.tensor(np.stack(tgt_fields, axis=0), device=device, dtype=dtype)
    return src, tgt, labels, degrees_out, modes_out


def load_model(pack: dict, cfg, device: torch.device):
    model = build_model(
        architecture=pack.get("architecture", cfg.architecture),
        src_dim=len(pack["src_node_features"]),
        tgt_dim=len(pack["tgt_node_features"]),
        edge_dim=len(pack["edge_features"]),
        hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
    ).to(device)
    model.load_state_dict(pack["model_state_dict"])
    model.num_rounds = int(pack.get("rounds", 1))
    model.eval()
    return model


def checkpoint_moment_mode(pack: dict) -> str:
    mode = pack.get("moment_mode")
    if mode:
        return str(mode)
    if bool(pack.get("moment_l2_local_soft", False)):
        return "local_soft_l2"
    if bool(pack.get("moment_l1_local_soft", False)):
        return "local_soft"
    if bool(pack.get("moment_l1_hard", False)):
        return "hard"
    return "none"


def build_current_operator(
    model,
    pack: dict,
    batch: dict,
    sxyz: torch.Tensor,
    txyz: torch.Tensor,
    n_cg: int,
    eps_rel: float,
    solve_dtype: torch.dtype,
):
    si = batch["src_index"]
    ti = batch["tgt_index"]
    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])
    mode = checkpoint_moment_mode(pack)
    moment_coef = None
    moment_coef2 = None
    if mode != "none":
        moment_coef = sxyz[si] - txyz[ti]
        if mode == "local_soft_l2":
            moment_coef2 = quadratic_moment_coef(sxyz, txyz, si, ti)
    t0 = time.perf_counter()
    S, M = operator_from_model(
        model,
        batch,
        batch["area_src"].float(),
        batch["area_tgt"].float(),
        n_src,
        n_tgt,
        float(pack.get("scale", 1.0)),
        signed=bool(pack.get("signed", False)),
        n_cg=n_cg,
        solve_dtype=solve_dtype,
        eps_rel=eps_rel,
        moment_coef=moment_coef,
        moment_mode=mode,
        moment_ridge=float(pack.get("moment_ridge", 1.0e-4)),
        moment_relax=float(pack.get("moment_relax", 1.0)),
        moment_iters=int(pack.get("moment_iters", 1)),
        moment_coef2=moment_coef2,
        moment2_ridge=float(pack.get("moment2_ridge", 1.0e-3)),
        moment2_relax=float(pack.get("moment2_relax", 0.5)),
        moment2_iters=int(pack.get("moment2_iters", 1)),
        implicit_projection=bool(pack.get("implicit_projection", False)),
    )
    return S, M, time.perf_counter() - t0


def make_constraint_coef(kind: str, sxyz: torch.Tensor, txyz: torch.Tensor, si: torch.Tensor, ti: torch.Tensor):
    if kind == "n0":
        return None
    lin = sxyz[si] - txyz[ti]
    if kind == "n01":
        return lin
    if kind == "n012":
        quad = quadratic_moment_coef(sxyz, txyz, si, ti)
        return torch.cat([lin, quad], dim=1)
    raise ValueError(f"unknown constraint kind {kind!r}")


def scatter_fields_from_mass(
    mass: torch.Tensor,
    src_fields: torch.Tensor,
    si: torch.Tensor,
    ti: torch.Tensor,
    area_tgt: torch.Tensor,
    n_tgt: int,
    field_chunk: int,
) -> torch.Tensor:
    out = torch.zeros((src_fields.shape[0], n_tgt), device=mass.device, dtype=mass.dtype)
    inv_area = 1.0 / torch.clamp(area_tgt[ti].to(dtype=mass.dtype), min=1.0e-30)
    for start in range(0, src_fields.shape[0], field_chunk):
        end = min(start + field_chunk, src_fields.shape[0])
        vals = src_fields[start:end, si].to(dtype=mass.dtype) * (mass * inv_area)[None, :]
        out[start:end].index_add_(1, ti, vals)
    return out


def rel_l2_by_field(pred: torch.Tensor, truth: torch.Tensor, area_tgt: torch.Tensor) -> torch.Tensor:
    area = area_tgt.to(dtype=pred.dtype)[None, :]
    num = (area * (pred - truth) ** 2).sum(dim=1)
    den = (area * truth ** 2).sum(dim=1).clamp_min(1.0e-30)
    return torch.sqrt(torch.clamp(num / den, min=0.0))


def grad_from_residual(
    residual: torch.Tensor,
    src_fields: torch.Tensor,
    si: torch.Tensor,
    ti: torch.Tensor,
    den: torch.Tensor,
    field_chunk: int,
) -> torch.Tensor:
    grad = torch.zeros(si.numel(), device=residual.device, dtype=residual.dtype)
    scale = (1.0 / den.to(dtype=residual.dtype).clamp_min(1.0e-30)) / max(int(residual.shape[0]), 1)
    for start in range(0, residual.shape[0], field_chunk):
        end = min(start + field_chunk, residual.shape[0])
        r_edge = residual[start:end, ti] * scale[start:end, None]
        grad = grad + (r_edge * src_fields[start:end, si].to(dtype=residual.dtype)).sum(dim=0)
    return grad


def residual_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x).detach().cpu())


def cg_solve(matvec, rhs: torch.Tensor, max_iter: int, tol: float):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    rhs_norm = torch.sqrt(rsold).clamp_min(1.0e-30)
    relres = float((torch.sqrt(rsold) / rhs_norm).detach().cpu())
    it_done = 0
    for it in range(1, max_iter + 1):
        ap = matvec(p)
        denom = torch.dot(p, ap).clamp_min(1.0e-30)
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = torch.dot(r, r)
        relres = float((torch.sqrt(rsnew) / rhs_norm).detach().cpu())
        it_done = it
        if relres < tol:
            break
        beta = rsnew / rsold.clamp_min(1.0e-30)
        p = r + beta * p
        rsold = rsnew
    return x, it_done, relres


def projected_gradient_solve(
    project_null,
    base_res: torch.Tensor,
    obj_src: torch.Tensor,
    obj_truth: torch.Tensor,
    si: torch.Tensor,
    ti: torch.Tensor,
    area_tgt: torch.Tensor,
    n_tgt: int,
    den: torch.Tensor,
    ridge_abs: float,
    mass_norm: torch.Tensor,
    args,
):
    delta = torch.zeros(si.numel(), device=base_res.device, dtype=base_res.dtype)

    def residual_for(d: torch.Tensor) -> torch.Tensor:
        return base_res + scatter_fields_from_mass(d, obj_src, si, ti, area_tgt, n_tgt, args.field_chunk)

    def objective_for(d: torch.Tensor) -> torch.Tensor:
        r = residual_for(d)
        area = area_tgt.to(dtype=r.dtype)[None, :]
        rel2 = (area * r ** 2).sum(dim=1) / den.to(dtype=r.dtype).clamp_min(1.0e-30)
        return 0.5 * rel2.mean() + 0.5 * float(ridge_abs) * torch.dot(d, d)

    loss = objective_for(delta)
    relgrad = float("inf")
    it_done = 0
    for it in range(1, args.oracle_iters + 1):
        res = residual_for(delta)
        grad = grad_from_residual(res, obj_src, si, ti, den, args.field_chunk)
        if ridge_abs > 0.0:
            grad = grad + float(ridge_abs) * delta
        direction = project_null(-grad)
        dir_norm = torch.linalg.norm(direction)
        grad_norm = torch.linalg.norm(grad).clamp_min(1.0e-30)
        relgrad = float((dir_norm / grad_norm).detach().cpu())
        if not math.isfinite(relgrad) or relgrad < args.oracle_tol:
            it_done = it
            break
        gd = torch.dot(grad, direction)
        if not torch.isfinite(gd) or float(gd.detach().cpu()) >= 0.0:
            it_done = it
            break
        hd = grad_from_residual(
            scatter_fields_from_mass(direction, obj_src, si, ti, area_tgt, n_tgt, args.field_chunk),
            obj_src,
            si,
            ti,
            den,
            args.field_chunk,
        )
        if ridge_abs > 0.0:
            hd = hd + float(ridge_abs) * direction
        denom = torch.dot(direction, hd)
        if torch.isfinite(denom) and float(denom.detach().cpu()) > 0.0:
            alpha = float((-gd / denom).detach().cpu())
        else:
            alpha = 1.0
        if args.max_step_rel > 0.0:
            max_alpha = float(args.max_step_rel) * float(mass_norm.detach().cpu()) / max(float(dir_norm.detach().cpu()), 1.0e-30)
            alpha = min(alpha, max_alpha)
        alpha = max(alpha, 1.0e-30)

        accepted = False
        for _ in range(args.line_search_steps):
            cand = delta + alpha * direction
            cand_loss = objective_for(cand)
            if torch.isfinite(cand_loss) and float(cand_loss.detach().cpu()) <= float(loss.detach().cpu()) + 1.0e-4 * alpha * float(gd.detach().cpu()):
                delta = cand
                loss = cand_loss
                accepted = True
                break
            alpha *= 0.5
        it_done = it
        if not accepted:
            break
    return delta, it_done, relgrad, float(loss.detach().cpu())


def conservation_residuals(mass: torch.Tensor, si: torch.Tensor, ti: torch.Tensor, area_src: torch.Tensor, area_tgt: torch.Tensor):
    src_sum = scatter_sum_torch(mass, si, int(area_src.numel()))
    tgt_sum = scatter_sum_torch(mass, ti, int(area_tgt.numel()))
    cons = torch.linalg.norm(src_sum - area_src.to(dtype=mass.dtype)) / torch.linalg.norm(area_src.to(dtype=mass.dtype))
    row = torch.linalg.norm(tgt_sum - area_tgt.to(dtype=mass.dtype)) / torch.linalg.norm(area_tgt.to(dtype=mass.dtype))
    return float(cons.detach().cpu()), float(row.detach().cpu())


def zero_residuals(mass: torch.Tensor, si: torch.Tensor, ti: torch.Tensor, n_src: int, n_tgt: int, area_src: torch.Tensor, area_tgt: torch.Tensor):
    src_sum = scatter_sum_torch(mass, si, n_src)
    tgt_sum = scatter_sum_torch(mass, ti, n_tgt)
    cons = torch.linalg.norm(src_sum) / torch.linalg.norm(area_src.to(dtype=mass.dtype))
    row = torch.linalg.norm(tgt_sum) / torch.linalg.norm(area_tgt.to(dtype=mass.dtype))
    return float(cons.detach().cpu()), float(row.detach().cpu())


def solve_oracle(
    kind: str,
    base_mass: torch.Tensor,
    obj_src: torch.Tensor,
    obj_truth: torch.Tensor,
    sxyz: torch.Tensor,
    txyz: torch.Tensor,
    si: torch.Tensor,
    ti: torch.Tensor,
    area_src: torch.Tensor,
    area_tgt: torch.Tensor,
    n_src: int,
    n_tgt: int,
    args,
):
    dtype = base_mass.dtype
    lin_coef = sxyz.to(dtype=dtype)[si] - txyz.to(dtype=dtype)[ti]
    quad_coef = quadratic_moment_coef(sxyz.to(dtype=dtype), txyz.to(dtype=dtype), si, ti)
    hard_coef = make_constraint_coef(kind, sxyz.to(dtype=dtype), txyz.to(dtype=dtype), si, ti)
    zeros_src = torch.zeros_like(area_src, dtype=dtype)
    zeros_tgt = torch.zeros_like(area_tgt, dtype=dtype)

    def project_marginal(v: torch.Tensor) -> torch.Tensor:
        return doubly_constrained_project_implicit(
            v,
            si,
            ti,
            zeros_src,
            zeros_tgt,
            n_src,
            n_tgt,
            eps_rel=args.nullspace_eps_rel,
            n_cg=args.nullspace_n_cg,
            tol=args.nullspace_tol,
            moment_coef=None,
            solve_dtype=dtype,
        )

    def project_null(v: torch.Tensor) -> torch.Tensor:
        if args.moment_projector == "hard":
            return doubly_constrained_project_implicit(
                v,
                si,
                ti,
                zeros_src,
                zeros_tgt,
                n_src,
                n_tgt,
                eps_rel=args.nullspace_eps_rel,
                n_cg=args.nullspace_n_cg,
                tol=args.nullspace_tol,
                moment_coef=hard_coef,
                solve_dtype=dtype,
            )
        out = project_marginal(v)
        if kind in ("n01", "n012"):
            corr = _local_target_moment_correction(
                out,
                ti,
                n_tgt,
                lin_coef,
                moment_ridge=args.moment_ridge,
            )
            out = out + float(args.moment_relax) * (corr - out)
            out = project_marginal(out)
        if kind == "n012":
            corr = _local_target_moment_correction(
                out,
                ti,
                n_tgt,
                quad_coef,
                moment_ridge=args.moment2_ridge,
            )
            out = out + float(args.moment2_relax) * (corr - out)
            out = project_marginal(out)
        return out

    with torch.no_grad():
        base_pred = scatter_fields_from_mass(base_mass, obj_src, si, ti, area_tgt, n_tgt, args.field_chunk)
        base_res = base_pred - obj_truth.to(dtype=dtype)
        den = (area_tgt.to(dtype=dtype)[None, :] * obj_truth.to(dtype=dtype) ** 2).sum(dim=1).clamp_min(1.0e-30)
        g0 = grad_from_residual(base_res, obj_src, si, ti, den, args.field_chunk)
        rhs = project_null(-g0)

        diag = torch.zeros_like(base_mass)
        inv_area = 1.0 / torch.clamp(area_tgt[ti].to(dtype=dtype), min=1.0e-30)
        for start in range(0, obj_src.shape[0], args.field_chunk):
            end = min(start + args.field_chunk, obj_src.shape[0])
            diag = diag + (
                obj_src[start:end, si].to(dtype=dtype) ** 2
                * ((1.0 / den[start:end]).to(dtype=dtype)[:, None])
            ).sum(dim=0) * inv_area / max(int(obj_src.shape[0]), 1)
        ridge_abs = float(args.ridge_rel) * float(diag.mean().detach().cpu())

        t0 = time.perf_counter()
        if args.solver == "cg":
            def matvec(v: torch.Tensor) -> torch.Tensor:
                y = scatter_fields_from_mass(v, obj_src, si, ti, area_tgt, n_tgt, args.field_chunk)
                hv = grad_from_residual(y, obj_src, si, ti, den, args.field_chunk)
                if ridge_abs > 0.0:
                    hv = hv + ridge_abs * v
                return project_null(hv)

            delta, cg_iters, cg_relres = cg_solve(matvec, rhs, args.oracle_iters, args.oracle_tol)
            final_loss = float("nan")
        else:
            delta, cg_iters, cg_relres, final_loss = projected_gradient_solve(
                project_null,
                base_res,
                obj_src,
                obj_truth,
                si,
                ti,
                area_tgt,
                n_tgt,
                den,
                ridge_abs,
                torch.linalg.norm(base_mass),
                args,
            )
        delta = project_null(delta)
        elapsed = time.perf_counter() - t0
        return delta, {
            "oracle_solver": str(args.solver),
            "moment_projector": str(args.moment_projector),
            "oracle_cg_iters": int(cg_iters),
            "oracle_cg_relres": float(cg_relres),
            "oracle_final_loss": float(final_loss),
            "oracle_elapsed_s": float(elapsed),
            "ridge_abs": float(ridge_abs),
            "rhs_norm": residual_norm(rhs),
        }


def mean_by_degree(labels, degrees, rel):
    rows = []
    rel_cpu = rel.detach().cpu().numpy()
    for deg in sorted(set(degrees)):
        idx = [i for i, d in enumerate(degrees) if d == deg]
        vals = rel_cpu[idx]
        rows.append({
            "degree": int(deg),
            "n_modes": int(len(idx)),
            "mean_rel_l2": float(np.mean(vals)),
            "worst_rel_l2": float(np.max(vals)),
            "worst_mode": labels[int(idx[int(np.argmax(vals))])],
        })
    return rows


def pct_improvement(before: float, after: float) -> float:
    if not math.isfinite(before) or before == 0.0:
        return float("nan")
    return 100.0 * (before - after) / before


def write_outputs(out_dir: Path, summary_rows: list[dict], degree_rows: list[dict], field_rows: list[dict], meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"meta": meta, "summary": summary_rows, "by_degree": degree_rows, "by_field": field_rows}, f, indent=2)

    for name, rows in [("summary.csv", summary_rows), ("by_degree.csv", degree_rows), ("by_field.csv", field_rows)]:
        if not rows:
            continue
        with (out_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# Nullspace oracle corrector", ""]
    lines.append("## Summary")
    lines.append("")
    if summary_rows:
        headers = [
            "pair", "constraint", "objective_base_mean_rel_l2", "objective_oracle_mean_rel_l2",
            "objective_improve_pct", "eval_base_mean_rel_l2", "eval_oracle_mean_rel_l2",
            "eval_improve_pct", "delta_rel_l2_vs_mass", "delta_cons_resid", "delta_row_resid",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in summary_rows:
            vals = []
            for h in headers:
                v = r[h]
                if isinstance(v, float):
                    vals.append(f"{v:.4e}" if abs(v) < 1.0e3 else f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## By Degree")
    lines.append("")
    if degree_rows:
        headers = ["pair", "constraint", "degree", "n_modes", "base_mean_rel_l2", "oracle_mean_rel_l2", "improve_pct"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in degree_rows:
            vals = []
            for h in headers:
                v = r[h]
                if isinstance(v, float):
                    vals.append(f"{v:.4e}" if abs(v) < 1.0e3 else f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v20b_base_a3p0_mink8_geom_v12.json")
    ap.add_argument("--pack", default="models_medium_improv/highorder_signed_v12_geom_localmom_l2_fieldfirst.pt")
    ap.add_argument("--pairs", nargs="+", default=["CS-r32_to_RLL-r90-180"])
    ap.add_argument("--constraints", nargs="+", default=["n0", "n01", "n012"], choices=["n0", "n01", "n012"])
    ap.add_argument("--objective-degrees", nargs="+", type=int, default=[40, 48])
    ap.add_argument("--eval-degrees", nargs="+", type=int, default=[8, 16, 24, 32, 40, 48])
    ap.add_argument("--modes-per-degree", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--projection-n-cg", type=int, default=800)
    ap.add_argument("--projection-eps-rel", type=float, default=1.0e-12)
    ap.add_argument("--projection-dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--nullspace-n-cg", type=int, default=400)
    ap.add_argument("--nullspace-eps-rel", type=float, default=1.0e-12)
    ap.add_argument("--nullspace-tol", type=float, default=1.0e-12)
    ap.add_argument("--oracle-iters", type=int, default=30)
    ap.add_argument("--oracle-tol", type=float, default=1.0e-6)
    ap.add_argument("--solver", choices=["pg", "cg"], default="pg")
    ap.add_argument("--moment-projector", choices=["local_soft", "hard"], default="local_soft")
    ap.add_argument("--moment-ridge", type=float, default=1.0e-4)
    ap.add_argument("--moment-relax", type=float, default=1.0)
    ap.add_argument("--moment2-ridge", type=float, default=1.0e-3)
    ap.add_argument("--moment2-relax", type=float, default=0.5)
    ap.add_argument("--max-step-rel", type=float, default=0.25)
    ap.add_argument("--line-search-steps", type=int, default=12)
    ap.add_argument("--ridge-rel", type=float, default=1.0e-6)
    ap.add_argument("--field-chunk", type=int, default=8)
    ap.add_argument("--out-dir", default="analysis_medium_improv/audits/nullspace_oracle_v12_localmom_l2")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    solve_dtype = parse_dtype(args.projection_dtype)
    cfg = load_config(args.config)
    pack = torch_load_pack(args.pack, map_location=device)
    model = load_model(pack, cfg, device)
    out_dir = Path(args.out_dir)

    summary_rows = []
    degree_rows = []
    field_rows = []
    meta = {
        "config": args.config,
        "pack": args.pack,
        "pairs": args.pairs,
        "constraints": args.constraints,
        "objective_degrees": args.objective_degrees,
        "eval_degrees": args.eval_degrees,
        "modes_per_degree": args.modes_per_degree,
        "projection_n_cg": args.projection_n_cg,
        "projection_eps_rel": args.projection_eps_rel,
        "projection_dtype": args.projection_dtype,
        "nullspace_n_cg": args.nullspace_n_cg,
        "ridge_rel": args.ridge_rel,
        "oracle_iters": args.oracle_iters,
        "solver": args.solver,
        "moment_projector": args.moment_projector,
        "max_step_rel": args.max_step_rel,
        "device": str(device),
    }

    print("Nullspace oracle")
    print("  device:", device)
    print("  pack:", args.pack)
    print("  constraints:", args.constraints)
    print("  objective degrees:", args.objective_degrees, "eval degrees:", args.eval_degrees)

    with torch.no_grad():
        for pair in args.pairs:
            print(f"\n=== {pair} ===")
            batch = load_pair_tensors(cfg, pair, pack["stats"], device=device)
            n_src, n_tgt = as_int(batch["n_src"]), as_int(batch["n_tgt"])
            si, ti = batch["src_index"], batch["tgt_index"]
            area_src = batch["area_src"].to(device=device, dtype=solve_dtype)
            area_tgt = batch["area_tgt"].to(device=device, dtype=solve_dtype)

            src_xyz_np = read_source_xyz_from_edges(cfg.edge_path(pair), n_src)
            tgt_xyz_np = read_target_xyz_from_edges(cfg.edge_path(pair), n_tgt)
            sxyz = torch.tensor(src_xyz_np, dtype=solve_dtype, device=device)
            txyz = torch.tensor(tgt_xyz_np, dtype=solve_dtype, device=device)

            print("  building current operator...")
            S, M, build_s = build_current_operator(
                model, pack, batch, sxyz.float(), txyz.float(),
                n_cg=args.projection_n_cg,
                eps_rel=args.projection_eps_rel,
                solve_dtype=solve_dtype,
            )
            M = M.to(dtype=solve_dtype)
            base_cons, base_row = conservation_residuals(M, si, ti, area_src, area_tgt)
            print(f"  base build_s={build_s:.3f} cons={base_cons:.3e} row={base_row:.3e}")

            obj_src, obj_tgt, obj_labels, obj_degrees, obj_modes = build_harmonic_fields(
                pair, src_xyz_np, tgt_xyz_np, args.objective_degrees,
                args.modes_per_degree, args.seed, device, solve_dtype)
            eval_src, eval_tgt, eval_labels, eval_degrees, eval_modes = build_harmonic_fields(
                pair, src_xyz_np, tgt_xyz_np, args.eval_degrees,
                args.modes_per_degree, args.seed, device, solve_dtype)

            base_obj_pred = scatter_fields_from_mass(M, obj_src, si, ti, area_tgt, n_tgt, args.field_chunk)
            base_obj_rel = rel_l2_by_field(base_obj_pred, obj_tgt, area_tgt)
            base_eval_pred = scatter_fields_from_mass(M, eval_src, si, ti, area_tgt, n_tgt, args.field_chunk)
            base_eval_rel = rel_l2_by_field(base_eval_pred, eval_tgt, area_tgt)
            print(f"  base objective mean={float(base_obj_rel.mean()):.4e} eval mean={float(base_eval_rel.mean()):.4e}")

            for kind in args.constraints:
                print(f"  solving {kind} oracle...")
                delta, info = solve_oracle(
                    kind,
                    M,
                    obj_src,
                    obj_tgt,
                    sxyz,
                    txyz,
                    si,
                    ti,
                    area_src,
                    area_tgt,
                    n_src,
                    n_tgt,
                    args,
                )
                M_new = M + delta
                delta_cons, delta_row = zero_residuals(delta, si, ti, n_src, n_tgt, area_src, area_tgt)
                new_cons, new_row = conservation_residuals(M_new, si, ti, area_src, area_tgt)
                obj_pred = scatter_fields_from_mass(M_new, obj_src, si, ti, area_tgt, n_tgt, args.field_chunk)
                obj_rel = rel_l2_by_field(obj_pred, obj_tgt, area_tgt)
                eval_pred = scatter_fields_from_mass(M_new, eval_src, si, ti, area_tgt, n_tgt, args.field_chunk)
                eval_rel = rel_l2_by_field(eval_pred, eval_tgt, area_tgt)

                base_obj_mean = float(base_obj_rel.mean().detach().cpu())
                obj_mean = float(obj_rel.mean().detach().cpu())
                base_eval_mean = float(base_eval_rel.mean().detach().cpu())
                eval_mean = float(eval_rel.mean().detach().cpu())
                delta_rel = float((torch.linalg.norm(delta) / torch.linalg.norm(M)).detach().cpu())
                max_abs_ratio = float((delta.abs().max() / M.abs().mean().clamp_min(1.0e-30)).detach().cpu())
                row = {
                    "pair": pair,
                    "constraint": kind,
                    "objective_base_mean_rel_l2": base_obj_mean,
                    "objective_oracle_mean_rel_l2": obj_mean,
                    "objective_improve_pct": pct_improvement(base_obj_mean, obj_mean),
                    "eval_base_mean_rel_l2": base_eval_mean,
                    "eval_oracle_mean_rel_l2": eval_mean,
                    "eval_improve_pct": pct_improvement(base_eval_mean, eval_mean),
                    "delta_rel_l2_vs_mass": delta_rel,
                    "delta_max_abs_over_mean_abs_mass": max_abs_ratio,
                    "base_cons_resid": base_cons,
                    "base_row_resid": base_row,
                    "new_cons_resid": new_cons,
                    "new_row_resid": new_row,
                    "delta_cons_resid": delta_cons,
                    "delta_row_resid": delta_row,
                    **info,
                }
                summary_rows.append(row)
                print(
                    "    obj %.4e -> %.4e (%+.2f%%), eval %.4e -> %.4e (%+.2f%%), "
                    "delta_rel=%.3e cons_delta=%.3e row_delta=%.3e iters=%d rel=%.2e"
                    % (
                        base_obj_mean,
                        obj_mean,
                        row["objective_improve_pct"],
                        base_eval_mean,
                        eval_mean,
                        row["eval_improve_pct"],
                        delta_rel,
                        delta_cons,
                        delta_row,
                        info["oracle_cg_iters"],
                        info["oracle_cg_relres"],
                    )
                )

                base_by_deg = {r["degree"]: r for r in mean_by_degree(eval_labels, eval_degrees, base_eval_rel)}
                new_by_deg = {r["degree"]: r for r in mean_by_degree(eval_labels, eval_degrees, eval_rel)}
                for deg in sorted(base_by_deg):
                    b = base_by_deg[deg]
                    n = new_by_deg[deg]
                    degree_rows.append({
                        "pair": pair,
                        "constraint": kind,
                        "degree": int(deg),
                        "n_modes": int(b["n_modes"]),
                        "base_mean_rel_l2": float(b["mean_rel_l2"]),
                        "oracle_mean_rel_l2": float(n["mean_rel_l2"]),
                        "improve_pct": pct_improvement(float(b["mean_rel_l2"]), float(n["mean_rel_l2"])),
                        "base_worst_rel_l2": float(b["worst_rel_l2"]),
                        "oracle_worst_rel_l2": float(n["worst_rel_l2"]),
                        "base_worst_mode": b["worst_mode"],
                        "oracle_worst_mode": n["worst_mode"],
                    })
                for i, label in enumerate(eval_labels):
                    b = float(base_eval_rel[i].detach().cpu())
                    n = float(eval_rel[i].detach().cpu())
                    field_rows.append({
                        "pair": pair,
                        "constraint": kind,
                        "degree": int(eval_degrees[i]),
                        "mode": int(eval_modes[i]),
                        "field": label,
                        "base_rel_l2": b,
                        "oracle_rel_l2": n,
                        "improve_pct": pct_improvement(b, n),
                    })

    write_outputs(out_dir, summary_rows, degree_rows, field_rows, meta)
    print(f"\nwrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
