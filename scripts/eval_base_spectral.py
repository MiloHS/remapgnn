#!/usr/bin/env python
"""Base-only spectral scorer: apply the CONVERGED base operator to analytic spherical-harmonic
fields and report per-degree relative error vs analytic truth and vs Tempest, at the finest pair.

Used to score low-band-base sweep configs: a good low-band base should be accurate at low degree
(<= Lb) and have clear headroom at high degree (worse than the full base / Tempest), without
collapsing edges. No corrector. Operator is balanced to convergence (conservative AND consistent).
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model
from remapgnn.sinkhorn import converged_balance, sparse_operator_weights
from train_config_irno_corrector import torch_load_pack, base_q_from_model, as_int
from train_config_balanced_harmonic import read_source_xyz_from_edges
from evaluate_refinement_convergence import analytic_function


def read_target_xyz(edge_path, n_tgt):
    df = pd.read_parquet(edge_path, columns=["target_index", "tgt_x", "tgt_y", "tgt_z"])
    g = df.groupby("target_index", sort=False)[["tgt_x", "tgt_y", "tgt_z"]].first()
    xyz = np.full((n_tgt, 3), np.nan, dtype=np.float64)
    xyz[g.index.to_numpy(dtype=np.int64)] = g.to_numpy(dtype=np.float64)
    return xyz


def scatter(n, idx, vals):
    y = np.zeros(n, dtype=np.float64)
    np.add.at(y, idx, vals)
    return y


def area_rel_l2(a, b, area):
    num = np.sqrt(np.sum(area * (a - b) ** 2))
    den = np.sqrt(np.sum(area * b * b))
    return float(num / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--functions", nargs="+",
                    default=["const", "x", "smooth1", "Y_4_0", "Y_8_0", "Y_16_0", "Y_24_0", "Y_8_4", "Y_16_8", "Y_24_12"])
    ap.add_argument("--tol", type=float, default=1.0e-6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pack = torch_load_pack(cfg.model_path, map_location=device)
    stats = pack["stats"]
    ef = list(pack["edge_features"]); sf = list(pack["src_node_features"]); tf = list(pack["tgt_node_features"])
    base = build_model(architecture=pack.get("architecture", cfg.architecture),
                       src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef),
                       hidden=int(pack.get("hidden", 128)),
                       decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(device)
    base.load_state_dict(pack["model_state_dict"]); base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    rows = []
    for pair in args.pairs:
        batch = load_pair_tensors(cfg, pair, stats, device=device)
        si = batch["src_index"]; ti = batch["tgt_index"]
        n_src = as_int(batch["n_src"]); n_tgt = as_int(batch["n_tgt"])
        with torch.no_grad():
            q = base_q_from_model(base, batch).double()
            M = converged_balance(q, si, ti, batch["area_src"].double(), batch["area_tgt"].double(),
                                  n_src, n_tgt, tol=args.tol, max_iter=50000)
            S = sparse_operator_weights(M, ti, batch["area_tgt"].double())
        S = S.detach().cpu().numpy(); siN = si.cpu().numpy(); tiN = ti.cpu().numpy()
        area_tgt = batch["area_tgt"].cpu().numpy().astype(np.float64)
        S_true = batch["S_true"].cpu().numpy().astype(np.float64)
        pos = batch["edge_exists"].cpu().numpy() > 0.5
        ep = cfg.edge_path(pair)
        src_xyz = read_source_xyz_from_edges(ep, n_src); tgt_xyz = read_target_xyz(ep, n_tgt)
        for fn in args.functions:
            x_src = analytic_function(fn, src_xyz); truth = analytic_function(fn, tgt_xyz)
            y_pred = scatter(n_tgt, tiN, S * x_src[siN])
            y_temp = scatter(n_tgt, tiN[pos], S_true[pos] * x_src[siN][pos])
            deg = 0
            if fn.startswith("Y_"):
                deg = int(fn.split("_")[1])
            elif fn in ("x", "y", "z"):
                deg = 1
            elif fn in ("smooth1", "smooth2"):
                deg = 2
            be = area_rel_l2(y_pred, truth, area_tgt)
            te = area_rel_l2(y_temp, truth, area_tgt)
            rows.append({"config": cfg.run_name, "pair": pair, "function": fn, "degree": deg,
                         "base_err": be, "tempest_err": te,
                         "ratio_base_vs_tempest": (be / te) if te > 0 else np.nan})
            print("  %-26s %-9s deg%2d base=%.3e tempest=%.3e ratio=%.2f" % (pair, fn, deg, be, te, be / te if te > 0 else float("nan")))
        del batch
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
