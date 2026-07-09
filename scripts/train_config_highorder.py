#!/usr/bin/env python
"""Supermesh-free higher-order remap operator.

GNN -> SIGNED edge weights -> doubly-constrained projection (exact conservation + consistency) -> operator S.
Inputs are supermesh-free (cell features + kNN candidate graph). Trained to MATCH 2nd-order TempestRemap
(np2 = teacher) + reproduce analytic-truth harmonic remaps. Tests whether a GNN can match 2nd-order TR
accuracy WITHOUT the overlap supermesh, staying conservative + consistent by construction.

Parameterization: q = M_base + scale * raw_weight (signed), M_base = uniform-within-target mass (cheap,
supermesh-free, correctly scaled); projection enforces both marginals.
"""
import argparse, sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from netCDF4 import Dataset

from remapgnn.config import load_config
from remapgnn.data import compute_feature_stats, get_feature_lists, load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from remapgnn.projection import (
    doubly_constrained_project,
    doubly_constrained_project_implicit,
    doubly_constrained_project_local_moment,
)
from train_config_balanced_harmonic import (
    set_seed, build_harmonic_fields_with_truth, build_harmonic_fields_with_truth_cellavg,
    harmonic_loss_from_operator,
    model_outputs_to_q, warn_split_leakage,
    read_source_xyz_from_edges, read_target_xyz_from_edges,
)
from train_config_irno_corrector import torch_load_pack, as_int
from remapgnn import fv_moments as fv

DEGREES = [0, 1, 2, 4, 8, 16, 24]


def read_flat_nc_field(path, field):
    with Dataset(path) as ds:
        if field not in ds.variables:
            raise KeyError(f"{field} not found in {path}")
        return np.asarray(ds.variables[field][:], dtype=np.float64).reshape(-1)


def load_real_field_tensors(cfg, pair, fields, n_src, n_tgt, device):
    if not fields:
        return None, None, []
    src_path, tgt_path = cfg.source_target_files(pair)
    if not src_path.exists() or not tgt_path.exists():
        print(f"  real-field cache {pair}: missing source/target files")
        return None, None, []

    src_fields = []
    tgt_fields = []
    kept = []
    for field in fields:
        try:
            src = read_flat_nc_field(src_path, field)
            tgt = read_flat_nc_field(tgt_path, field)
        except Exception as e:
            print(f"  real-field cache {pair}: skip {field}: {e}")
            continue
        if src.size != n_src or tgt.size != n_tgt:
            print(f"  real-field cache {pair}: skip {field}: size {src.size}->{tgt.size}, expected {n_src}->{n_tgt}")
            continue
        scale = float(np.sqrt(np.mean(src * src)))
        if scale > 0.0:
            src = src / scale
            tgt = tgt / scale
        src_fields.append(src.astype("float32"))
        tgt_fields.append(tgt.astype("float32"))
        kept.append(field)

    if not src_fields:
        return None, None, []
    print(f"  real-field cache {pair}: {len(kept)} fields {kept}")
    return (
        torch.tensor(np.stack(src_fields, axis=0), dtype=torch.float32, device=device),
        torch.tensor(np.stack(tgt_fields, axis=0), dtype=torch.float32, device=device),
        kept,
    )


def applied_field_loss_from_operator(
    S,
    src_index,
    tgt_index,
    src_fields,
    tgt_fields,
    area_tgt,
    n_tgt,
    max_fields_per_step=0,
    eps=1.0e-20,
):
    if src_fields is None or tgt_fields is None or src_fields.shape[0] == 0:
        z = S.new_zeros(())
        return z, z.detach()

    n_fields = src_fields.shape[0]
    if max_fields_per_step > 0 and n_fields > max_fields_per_step:
        perm = torch.randperm(n_fields, device=src_fields.device)[:max_fields_per_step]
        sf = src_fields[perm]
        tf = tgt_fields[perm]
    else:
        sf = src_fields
        tf = tgt_fields

    sf = sf.to(device=S.device, dtype=S.dtype)
    tf = tf.to(device=S.device, dtype=S.dtype)
    area = area_tgt.to(device=S.device, dtype=S.dtype)
    x_edge = sf[:, src_index]
    pred = torch.zeros((sf.shape[0], n_tgt), dtype=S.dtype, device=S.device)
    pred.index_add_(1, tgt_index, S[None, :] * x_edge)
    rel2 = (area[None, :] * (pred - tf) ** 2).sum(dim=1) / torch.clamp(
        (area[None, :] * tf * tf).sum(dim=1),
        min=eps,
    )
    return rel2.mean(), torch.sqrt(torch.clamp(rel2, min=0.0)).mean().detach()


def load_np2_target(pair, src_index, tgt_index, device):
    v = Dataset(f"maps_medium_improv/map_{pair}_conserve_np2.nc").variables
    r = np.asarray(v["row"][:]).ravel().astype(np.int64) - 1
    c = np.asarray(v["col"][:]).ravel().astype(np.int64) - 1
    S = np.asarray(v["S"][:]).ravel().astype(np.float64)
    dd = {(int(t), int(s)): sv for t, s, sv in zip(r, c, S)}
    ti = tgt_index.cpu().numpy(); si = src_index.cpu().numpy()
    out = np.array([dd.get((int(t), int(s)), 0.0) for t, s in zip(ti, si)], dtype=np.float32)
    insupp = np.array([(int(t), int(s)) in dd for t, s in zip(ti, si)], dtype=bool)
    return (torch.tensor(out, device=device), torch.tensor(insupp, device=device))


def quadratic_moment_coef(sxyz, txyz, si, ti):
    src = sxyz[si]
    tgt = txyz[ti]
    return torch.stack(
        [
            src[:, 0] * src[:, 0] - tgt[:, 0] * tgt[:, 0],
            src[:, 0] * src[:, 1] - tgt[:, 0] * tgt[:, 1],
            src[:, 0] * src[:, 2] - tgt[:, 0] * tgt[:, 2],
            src[:, 1] * src[:, 1] - tgt[:, 1] * tgt[:, 1],
            src[:, 1] * src[:, 2] - tgt[:, 1] * tgt[:, 2],
            src[:, 2] * src[:, 2] - tgt[:, 2] * tgt[:, 2],
        ],
        dim=1,
    )


def operator_from_model(
    model,
    batch,
    area_src,
    area_tgt,
    n_src,
    n_tgt,
    scale,
    signed=False,
    n_cg=400,
    solve_dtype=None,
    eps_rel=1e-9,
    moment_coef=None,
    moment_mode="hard",
    moment_ridge=1.0e-4,
    moment_relax=1.0,
    moment_iters=1,
    moment_coef2=None,
    moment2_ridge=1.0e-3,
    moment2_relax=0.5,
    moment2_iters=1,
    moment_coef3=None,
    moment3_ridge=1.0e-2,
    moment3_relax=0.5,
    moment3_iters=0,
    implicit_projection=False,
):
    out = model(batch["src_node_attr"], batch["tgt_node_attr"], batch["edge_attr"],
                batch["src_index"], batch["tgt_index"], n_src, n_tgt)
    logit, raw_weight, _ = model_outputs_to_q(out)
    ti = batch["tgt_index"]; si = batch["src_index"]
    deg_t = scatter_sum_torch(torch.ones_like(raw_weight.float()), ti, n_tgt)
    M_base = area_tgt[ti] / torch.clamp(deg_t[ti], min=1.0)          # supermesh-free, correctly scaled
    # signed=True: use the RAW (pre-sigmoid) edge head -> per-edge multiplier can pass through 0 and go
    # NEGATIVE. This is the high-order fix: a non-negative conservative linear operator is at most 1st order
    # (Godunov); 2nd-order (and even exact LINEAR reproduction) needs signed weights. The softplus path
    # (signed=False, legacy) forced q >= M_base > 0, capping the operator at ~1st order.
    w = logit.float() if signed else raw_weight.float()             # raw signed head vs legacy softplus(>=0)
    q = M_base * (1.0 + scale * w)                                  # signed -> SIGNED per-edge mass (can be <0)
    if moment_coef is not None and moment_mode in ("local_soft", "local_soft_l2", "local_soft_l3"):
        M = doubly_constrained_project_local_moment(
            q, si, ti, area_src, area_tgt, n_src, n_tgt,
            eps_rel=eps_rel,
            n_cg=n_cg,
            solve_dtype=solve_dtype,
            moment_coef=moment_coef,
            moment_ridge=moment_ridge,
            moment_relax=moment_relax,
            moment_iters=moment_iters,
            moment_coef2=(moment_coef2 if moment_mode in ("local_soft_l2", "local_soft_l3") else None),
            moment2_ridge=moment2_ridge,
            moment2_relax=moment2_relax,
            moment2_iters=moment2_iters,
            moment_coef3=(moment_coef3 if moment_mode == "local_soft_l3" else None),
            moment3_ridge=moment3_ridge,
            moment3_relax=moment3_relax,
            moment3_iters=moment3_iters,
            use_implicit=implicit_projection,
        )
    else:
        project = doubly_constrained_project_implicit if implicit_projection else doubly_constrained_project
        M = project(
            q, si, ti, area_src, area_tgt, n_src, n_tgt,
            eps_rel=eps_rel,
            n_cg=n_cg,
            solve_dtype=solve_dtype,
            moment_coef=(moment_coef if moment_mode == "hard" else None),
        )
    S = M / torch.clamp(area_tgt.to(dtype=M.dtype)[ti], min=1e-30)
    return S, M


def conservation_resid(M, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt):
    sm = scatter_sum_torch(M, src_index, n_src); tm = scatter_sum_torch(M, tgt_index, n_tgt)
    cr = torch.linalg.norm(sm - area_src) / torch.linalg.norm(area_src)
    rr = torch.linalg.norm(tm - area_tgt) / torch.linalg.norm(area_tgt)
    return float(cr), float(rr)


def moment_loss_from_xyz(S, si, ti, sxyz, txyz, atgt, n_tgt):
    num = S.new_zeros(())
    den = S.new_zeros(())
    for d in range(3):
        pred_d = scatter_sum_torch(S * sxyz[si, d].to(dtype=S.dtype), ti, n_tgt)
        truth_d = txyz[:, d].to(dtype=S.dtype)
        area = atgt.to(dtype=S.dtype)
        num = num + (area * (pred_d - truth_d) ** 2).sum()
        den = den + (area * truth_d ** 2).sum()
    return num / den.clamp_min(1e-12)


def make_cache_entry(cfg, pair, b, n_src, n_tgt, S_np2, insupp, args, device):
    """Build one per-pair training/val cache entry: np2 teacher, harmonic (point or FV
    cell-average) source+truth fields, cell centers, real fields, and -- when
    --moment-geometry fv -- the precomputed FV cell-average coordinate [N,3] and quadratic
    [N,6] moment arrays for source and target grids (static per pair, computed once)."""
    if args.harmonic_truth == "cellavg":
        sfld, tfld = build_harmonic_fields_with_truth_cellavg(
            cfg, pair, n_src, n_tgt, args.degrees, args.modes_per_degree, args.seed, quad_m=args.quad_m)
    else:
        sfld, tfld = build_harmonic_fields_with_truth(
            cfg, pair, n_src, n_tgt, args.degrees, args.modes_per_degree, args.seed)
    sxyz_np = read_source_xyz_from_edges(cfg.edge_path(pair), n_src)
    txyz_np = read_target_xyz_from_edges(cfg.edge_path(pair), n_tgt)
    sxyz = torch.tensor(sxyz_np, dtype=torch.float32, device=device)
    txyz = torch.tensor(txyz_np, dtype=torch.float32, device=device)
    real_src, real_tgt, real_names = load_real_field_tensors(
        cfg, pair, args.real_fields, n_src, n_tgt, device)
    entry = dict(b=b, n_src=n_src, n_tgt=n_tgt, S_np2=S_np2, insupp=insupp,
                 sfld=sfld.to(device), tfld=tfld.to(device), sxyz=sxyz, txyz=txyz,
                 real_src=real_src, real_tgt=real_tgt, real_names=real_names)
    if args.moment_geometry == "fv":
        mp = str(cfg.maps_dir / f"map_{pair}_conserve.nc")
        Vs, nvs, _, cas = fv.load_corners_from_map(mp, "a")
        Vt, nvt, _, cat = fv.load_corners_from_map(mp, "b")
        if float(np.abs(cas - sxyz_np).max()) > 1e-6 or float(np.abs(cat - txyz_np).max()) > 1e-6:
            raise ValueError(f"FV cell-order mismatch for {pair} (map vs edge dataset)")
        cubic = args.moment_mode == "local_soft_l3"
        ms = fv.compute_grid_moments(Vs, nvs, m=args.quad_m, cubic=cubic)
        mt = fv.compute_grid_moments(Vt, nvt, m=args.quad_m, cubic=cubic)
        tt = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        entry.update(fv_coord_s=tt(ms["coord"]), fv_coord_t=tt(mt["coord"]),
                     fv_quad_s=tt(ms["quad"]), fv_quad_t=tt(mt["quad"]))
        if cubic:
            entry.update(fv_cubic_s=tt(ms["cubic"]), fv_cubic_t=tt(mt["cubic"]))
    return entry


def _moment_coefs_for(c, si, ti, args):
    """degree-1/2/3 edge coefficients for a cache entry, using FV cell-average moments
    or legacy cell-center point values.  Returns (mc, mc2, mc3); mc3 only for local_soft_l3
    (degree-3 requires FV cell-average geometry)."""
    if args.moment_mode == "none":
        return None, None, None
    l2 = args.moment_mode in ("local_soft_l2", "local_soft_l3")
    l3 = args.moment_mode == "local_soft_l3"
    if args.moment_geometry == "fv":
        mc = c["fv_coord_s"][si] - c["fv_coord_t"][ti]
        mc2 = (c["fv_quad_s"][si] - c["fv_quad_t"][ti]) if l2 else None
        mc3 = (c["fv_cubic_s"][si] - c["fv_cubic_t"][ti]) if l3 else None
    else:
        mc = c["sxyz"][si] - c["txyz"][ti]
        mc2 = quadratic_moment_coef(c["sxyz"], c["txyz"], si, ti) if l2 else None
        mc3 = None  # degree-3 requires FV geometry
    return mc, mc2, mc3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)            # base config (pairs, graph suffix, features)
    ap.add_argument("--base-pack", required=True)         # an existing base .pt for stats + feature lists
    ap.add_argument("--init-from-base", action="store_true",
                    help="initialize model weights from --base-pack instead of training from scratch")
    ap.add_argument("--use-config-features", action="store_true",
                    help="use feature lists from --config and recompute normalization stats on train pairs; "
                         "needed for synthetic/new feature experiments")
    ap.add_argument("--stat-sample-per-pair", type=int, default=80000)
    ap.add_argument("--pairs", nargs="+", default=None)   # override train pairs (must have np2 maps)
    ap.add_argument("--val-pairs", nargs="+", default=None)  # held-out val for best-checkpoint selection
    ap.add_argument("--out", default="models_medium_improv/highorder_signed.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-op", type=float, default=1.0)
    ap.add_argument("--lam-field", type=float, default=1.0)
    ap.add_argument("--lam-real-field", type=float, default=0.0,
                    help="applied real-field loss weight; fields come from the config source/target files")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n-cg", type=int, default=400, help="CG iterations in the projection solve")
    ap.add_argument("--degrees", type=int, nargs="+", default=DEGREES,
                    help="harmonic degrees used for applied-field training")
    ap.add_argument("--modes-per-degree", type=int, default=4)
    ap.add_argument("--max-fields-per-step", type=int, default=0,
                    help="subsample harmonic/real fields per pair step when >0")
    ap.add_argument("--real-fields", nargs="*", default=[],
                    help="optional real field variables to include as applied-field training targets")
    ap.add_argument("--moment-l1-hard", action="store_true",
                    help="enforce degree-1 Cartesian moment reproduction in the projection")
    ap.add_argument("--moment-l1-local-soft", action="store_true",
                    help="apply local moment correction between hard marginal projections")
    ap.add_argument("--moment-l2-local-soft", action="store_true",
                    help="apply damped local quadratic moment correction after local degree-1 correction")
    ap.add_argument("--moment-mode", choices=["none", "hard", "local_soft", "local_soft_l2", "local_soft_l3"], default=None)
    ap.add_argument("--moment-ridge", type=float, default=1.0e-4,
                    help="damping for the local soft moment correction")
    ap.add_argument("--moment-relax", type=float, default=1.0,
                    help="blend factor for the local soft moment correction")
    ap.add_argument("--moment-iters", type=int, default=1,
                    help="number of local moment correction / hard marginal projection cycles")
    ap.add_argument("--moment2-ridge", type=float, default=1.0e-3,
                    help="damping for the local soft quadratic moment correction")
    ap.add_argument("--moment2-relax", type=float, default=0.5,
                    help="blend factor for the local soft quadratic moment correction")
    ap.add_argument("--moment2-iters", type=int, default=1,
                    help="number of local quadratic moment correction / hard marginal projection cycles")
    ap.add_argument("--moment3-ridge", type=float, default=1.0e-2,
                    help="damping for the local soft CUBIC (degree-3) moment correction (larger: 10 monomials)")
    ap.add_argument("--moment3-relax", type=float, default=0.5,
                    help="blend factor for the local soft cubic moment correction")
    ap.add_argument("--moment3-iters", type=int, default=0,
                    help="number of local cubic moment correction cycles (0 = off; requires --moment-mode local_soft_l3)")
    ap.add_argument("--implicit-projection", action="store_true",
                    help="use implicit projection gradients; recommended with --moment-l1-hard")
    ap.add_argument("--field-first", action="store_true",
                    help="checkpoint on applied-field validation score instead of operator MSE")
    ap.add_argument("--checkpoint-score", choices=["op", "field_first"], default="op")
    ap.add_argument("--val-score-op-weight", type=float, default=0.05)
    ap.add_argument("--val-score-harmonic-weight", type=float, default=1.0)
    ap.add_argument("--val-score-real-weight", type=float, default=1.0)
    ap.add_argument("--val-score-moment-weight", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=1)   # message-passing rounds (receptive field)
    ap.add_argument("--signed", action="store_true",   # THE high-order fix: q can go negative
                    help="use the raw signed edge head so per-edge mass can be negative (needed for >1st order)")
    ap.add_argument("--rel-op", action="store_true",   # scale-free op-loss so lam-field is meaningful
                    help="normalize op-loss by the np2 entry scale (relative MSE), comparable to the field loss")
    ap.add_argument("--lam-moment", type=float, default=0.0,   # explicit degree-1 (linear) reproduction penalty
                    help="soft penalty on the linear-reproduction (deg-1 moment) residual ||S@x_src - x_tgt|| for "
                         "the 3 coordinate fields (area-weighted relative MSE). Tests whether the loss ALONE can "
                         "enforce the exact moment cancellation that defines 2nd-order accuracy.")
    ap.add_argument("--lam-moment-final", type=float, default=None,   # anneal moment weight: lam-moment -> this
                    help="if set, LINEARLY ramp the moment weight from --lam-moment (epoch 1) to this value "
                         "(final epoch). Tests whether annealing the moment UP -- learn field structure first, "
                         "then tighten linear reproduction -- beats the fixed 1e5 that over-distorted in v6a.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (was hardcoded 0); vary for multi-seed sweeps")
    ap.add_argument("--moment-geometry", choices=["center", "fv"], default="center",
                    help="moment-coef monomials: cell-CENTER point values (legacy) or true finite-volume "
                         "CELL-AVERAGE moments via per-cell quadrature (remapgnn.fv_moments)")
    ap.add_argument("--harmonic-truth", choices=["point", "cellavg"], default="point",
                    help="spectral (h_loss) truth: point value at cell centers (legacy) or cell-average "
                         "harmonics.  Use cellavg WITH --moment-geometry fv so the loss matches the FV operator")
    ap.add_argument("--quad-m", type=int, default=8, help="quadrature subdivisions for FV cell averages")
    ap.add_argument("--test-pair", default=None, help="held-out test pair; asserted NOT in --val-pairs")
    args = ap.parse_args()
    if args.field_first:
        args.checkpoint_score = "field_first"
    if args.moment_mode is None:
        args.moment_mode = (
            "local_soft_l2" if args.moment_l2_local_soft
            else ("local_soft" if args.moment_l1_local_soft else ("hard" if args.moment_l1_hard else "none"))
        )
    args.moment_l1_hard = args.moment_mode == "hard"
    args.moment_l1_local_soft = args.moment_mode in ("local_soft", "local_soft_l2", "local_soft_l3")
    args.moment_l2_local_soft = args.moment_mode in ("local_soft_l2", "local_soft_l3")
    if args.moment_mode == "local_soft_l3" and args.moment_geometry != "fv":
        raise SystemExit("--moment-mode local_soft_l3 requires --moment-geometry fv (cubic cell-average moments)")

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    pack = torch_load_pack(args.base_pack, map_location=device)
    print("message-passing rounds:", args.rounds, " signed:", args.signed, " rel_op:", args.rel_op,
          " lam_moment:", args.lam_moment, " lam_real_field:", args.lam_real_field,
          " moment_mode:", args.moment_mode,
          " moment_ridge:", args.moment_ridge, " moment_relax:", args.moment_relax,
          " moment_iters:", args.moment_iters,
          " moment2_ridge:", args.moment2_ridge, " moment2_relax:", args.moment2_relax,
          " moment2_iters:", args.moment2_iters,
          " implicit_projection:", args.implicit_projection,
          " checkpoint_score:", args.checkpoint_score,
          " use_config_features:", args.use_config_features)
    print("harmonic degrees:", args.degrees, " modes_per_degree:", args.modes_per_degree,
          " real_fields:", args.real_fields)

    tr = cfg.training if hasattr(cfg, "training") else {}
    train_pairs = list(getattr(cfg, "pairs", []))
    train_pairs = list(tr.get("train_pairs", train_pairs)) if isinstance(tr, dict) else train_pairs
    if args.pairs:
        train_pairs = args.pairs
    if args.smoke:
        train_pairs = train_pairs[:1]; args.epochs = 2
    print("train pairs:", train_pairs)

    if args.use_config_features:
        ef, sf, tf = get_feature_lists(cfg)
        stats = compute_feature_stats(
            cfg, train_pairs, sample_per_pair=args.stat_sample_per_pair, seed=123)
    else:
        stats = pack["stats"]
        sf = list(pack["src_node_features"])
        tf = list(pack["tgt_node_features"])
        ef = list(pack["edge_features"])
    print("features:")
    print("  edge:", ef)
    print("  src: ", sf)
    print("  tgt: ", tf)

    model = build_model(architecture=pack.get("architecture", cfg.architecture),
                        src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef),
                        hidden=int(pack.get("hidden", 128)),
                        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(device)
    if args.init_from_base:
        model.load_state_dict(pack["model_state_dict"])
        print("initialized model weights from", args.base_pack)
    model.num_rounds = args.rounds
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # per-pair caches (supermesh-free graph + np2 teacher + truth harmonics)
    cache = {}
    for pair in train_pairs:
        b = load_pair_tensors(cfg, pair, stats, device=device)
        n_src, n_tgt = as_int(b["n_src"]), as_int(b["n_tgt"])
        S_np2, insupp = load_np2_target(pair, b["src_index"], b["tgt_index"], device)
        cache[pair] = make_cache_entry(cfg, pair, b, n_src, n_tgt, S_np2, insupp, args, device)
        print(f"  cached {pair}: edges={b['src_index'].numel()} np2_support_in_cand={int(insupp.sum())}/{insupp.numel()}")

    val_pairs = args.val_pairs or []
    valcache = {}
    for pair in val_pairs:
        bv = load_pair_tensors(cfg, pair, stats, device=device)
        n_src, n_tgt = as_int(bv["n_src"]), as_int(bv["n_tgt"])
        S2v, insv = load_np2_target(pair, bv["src_index"], bv["tgt_index"], device)
        valcache[pair] = make_cache_entry(cfg, pair, bv, n_src, n_tgt, S2v, insv, args, device)
    print("val pairs:", val_pairs)
    if args.test_pair is not None and args.test_pair in set(val_pairs):
        raise ValueError(f"LEAKAGE: test_pair {args.test_pair} is in val_pairs {val_pairs}")
    warn_split_leakage(train_pairs, val_pairs, args.test_pair)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best = float("inf"); best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_op = []; ep_hrel = []; ep_real = []; ep_mom = []
        # moment weight for THIS epoch: linear ramp from lam_moment -> lam_moment_final if annealing requested
        if args.lam_moment_final is not None and args.epochs > 1:
            frac = (epoch - 1) / (args.epochs - 1)
            lam_moment_ep = args.lam_moment + frac * (args.lam_moment_final - args.lam_moment)
        else:
            lam_moment_ep = args.lam_moment
        for pair in train_pairs:
            c = cache[pair]; b = c["b"]
            si, ti = b["src_index"], b["tgt_index"]
            asrc = b["area_src"].float(); atgt = b["area_tgt"].float()
            opt.zero_grad(set_to_none=True)
            moment_coef, moment_coef2, moment_coef3 = _moment_coefs_for(c, si, ti, args)
            S, M = operator_from_model(
                model, b, asrc, atgt, c["n_src"], c["n_tgt"], args.scale,
                signed=args.signed,
                n_cg=args.n_cg,
                moment_coef=moment_coef,
                moment_mode=args.moment_mode,
                moment_ridge=args.moment_ridge,
                moment_relax=args.moment_relax,
                moment_iters=args.moment_iters,
                moment_coef2=moment_coef2,
                moment2_ridge=args.moment2_ridge,
                moment2_relax=args.moment2_relax,
                moment2_iters=args.moment2_iters,
                moment_coef3=moment_coef3,
                moment3_ridge=args.moment3_ridge,
                moment3_relax=args.moment3_relax,
                moment3_iters=args.moment3_iters,
                implicit_projection=args.implicit_projection,
            )
            diff2 = (S - c["S_np2"]) ** 2
            insup = c["insupp"]; outsup = ~insup
            # guard empty masks: .mean() on an empty selection is NaN (0/0) and would poison backward
            op_in = diff2[insup].mean() if bool(insup.any()) else S.new_zeros(())
            op_out = diff2[outsup].mean() if bool(outsup.any()) else S.new_zeros(())
            op_loss = op_in + 0.05 * op_out
            if args.rel_op:
                denom = (c["S_np2"][insup] ** 2).mean() if bool(insup.any()) else S.new_ones(())
                op_loss = op_loss / denom.clamp_min(1e-12)
            h_loss, h_rel = harmonic_loss_from_operator(
                S_pred=S, S_true=c["S_np2"], src_index=si, tgt_index=ti,
                edge_exists=torch.ones_like(S), harmonic_fields=c["sfld"], n_tgt=c["n_tgt"],
                max_fields_per_step=args.max_fields_per_step, target_fields=c["tfld"])
            real_loss, real_rel = applied_field_loss_from_operator(
                S, si, ti, c["real_src"], c["real_tgt"], atgt, c["n_tgt"],
                max_fields_per_step=args.max_fields_per_step)
            loss = args.lam_op * op_loss + args.lam_field * h_loss + args.lam_real_field * real_loss
            _mc_s = c["fv_coord_s"] if args.moment_geometry == "fv" else c["sxyz"]
            _mc_t = c["fv_coord_t"] if args.moment_geometry == "fv" else c["txyz"]
            mom_loss = moment_loss_from_xyz(S, si, ti, _mc_s, _mc_t, atgt, c["n_tgt"])
            if lam_moment_ep > 0.0:
                # degree-1 moment / linear-reproduction residual: applying S to each source coordinate
                # field must return the target coordinate field. area-weighted relative MSE over x,y,z.
                loss = loss + lam_moment_ep * mom_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_op.append(float(op_loss))
            ep_hrel.append(float(h_rel))
            ep_real.append(float(real_rel))
            ep_mom.append(float(mom_loss.detach()))
        sched.step()

        model.eval()
        with torch.no_grad():
            vops = []; vharm = []; vreal = []; vmom = []
            for pair in val_pairs:
                cv = valcache[pair]; bv = cv["b"]
                vsi, vti = bv["src_index"], bv["tgt_index"]
                vasrc = bv["area_src"].float()
                vatgt = bv["area_tgt"].float()
                vmoment_coef, vmoment_coef2, vmoment_coef3 = _moment_coefs_for(cv, vsi, vti, args)
                Sv, _ = operator_from_model(
                    model, bv, vasrc, vatgt, cv["n_src"], cv["n_tgt"], args.scale,
                    signed=args.signed,
                    n_cg=args.n_cg,
                    moment_coef=vmoment_coef,
                    moment_mode=args.moment_mode,
                    moment_ridge=args.moment_ridge,
                    moment_relax=args.moment_relax,
                    moment_iters=args.moment_iters,
                    moment_coef2=vmoment_coef2,
                    moment2_ridge=args.moment2_ridge,
                    moment2_relax=args.moment2_relax,
                    moment2_iters=args.moment2_iters,
                    moment_coef3=vmoment_coef3,
                    moment3_ridge=args.moment3_ridge,
                    moment3_relax=args.moment3_relax,
                    moment3_iters=args.moment3_iters,
                    implicit_projection=args.implicit_projection,
                )
                d = (Sv - cv["S_np2"]) ** 2
                vm = cv["insupp"]
                vops.append(float(d[vm].mean()) if bool(vm.any()) else 0.0)
                hv, _ = harmonic_loss_from_operator(
                    S_pred=Sv, S_true=cv["S_np2"], src_index=vsi, tgt_index=vti,
                    edge_exists=torch.ones_like(Sv), harmonic_fields=cv["sfld"], n_tgt=cv["n_tgt"],
                    max_fields_per_step=0, target_fields=cv["tfld"])
                rv, _ = applied_field_loss_from_operator(
                    Sv, vsi, vti, cv["real_src"], cv["real_tgt"], vatgt, cv["n_tgt"],
                    max_fields_per_step=0)
                mv = moment_loss_from_xyz(Sv, vsi, vti, cv["sxyz"], cv["txyz"], vatgt, cv["n_tgt"])
                vharm.append(float(hv))
                vreal.append(float(rv))
                vmom.append(float(mv))
            val_op = float(np.mean(vops)) if vops else float(np.mean(ep_op))
            val_harm = float(np.mean(vharm)) if vharm else float(np.mean(ep_hrel) ** 2)
            val_real = float(np.mean(vreal)) if vreal else float(np.mean(ep_real) ** 2)
            val_mom = float(np.mean(vmom)) if vmom else float(np.mean(ep_mom))
            if args.checkpoint_score == "field_first":
                val_score = (
                    args.val_score_harmonic_weight * val_harm
                    + args.val_score_real_weight * val_real
                    + args.val_score_op_weight * val_op
                    + args.val_score_moment_weight * val_mom
                )
            else:
                val_score = val_op
        if val_score < best:
            best = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cr, rr = conservation_resid(M.detach(), b["src_index"], b["tgt_index"], asrc, atgt, c["n_src"], c["n_tgt"])
        print("epoch %04d  train_op=%.4e  train_h_rel=%.4e  train_real_rel=%.4e  val_score=%.4e (best %.4e)  val_op=%.4e val_h=%.4e val_real=%.4e val_mom=%.4e  mom=%.4e (lam=%.2e)  cons=%.2e row=%.2e"
              % (epoch, float(np.mean(ep_op)),
                 (float(np.mean(ep_hrel)) if ep_hrel else 0.0),
                 (float(np.mean(ep_real)) if ep_real else 0.0),
                 val_score, best, val_op, val_harm, val_real, val_mom,
                 (float(np.mean(ep_mom)) if ep_mom else 0.0), lam_moment_ep, cr, rr))

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(dict(
        model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        architecture=pack.get("architecture", cfg.architecture), hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
        src_node_features=sf, tgt_node_features=tf, edge_features=ef, stats=stats,
        scale=args.scale, rounds=args.rounds, signed=bool(args.signed),
        n_cg=int(args.n_cg),
        lam_moment=float(args.lam_moment),
        lam_moment_final=(float(args.lam_moment_final) if args.lam_moment_final is not None else None),
        lam_field=float(args.lam_field),
        lam_real_field=float(args.lam_real_field),
        moment_l1_hard=bool(args.moment_l1_hard),
        moment_l1_local_soft=bool(args.moment_l1_local_soft),
        moment_l2_local_soft=bool(args.moment_l2_local_soft),
        moment_mode=str(args.moment_mode),
        moment_ridge=float(args.moment_ridge),
        moment_relax=float(args.moment_relax),
        moment_iters=int(args.moment_iters),
        moment2_ridge=float(args.moment2_ridge),
        moment2_relax=float(args.moment2_relax),
        moment2_iters=int(args.moment2_iters),
        moment3_ridge=float(args.moment3_ridge),
        moment3_relax=float(args.moment3_relax),
        moment3_iters=int(args.moment3_iters),
        moment_geometry=str(args.moment_geometry),
        quad_m=int(args.quad_m),
        harmonic_truth=str(args.harmonic_truth),
        seed=int(args.seed),
        implicit_projection=bool(args.implicit_projection),
        field_first=bool(args.field_first),
        checkpoint_score=str(args.checkpoint_score),
        val_score_op_weight=float(args.val_score_op_weight),
        val_score_harmonic_weight=float(args.val_score_harmonic_weight),
        val_score_real_weight=float(args.val_score_real_weight),
        val_score_moment_weight=float(args.val_score_moment_weight),
        degrees=[int(x) for x in args.degrees],
        modes_per_degree=int(args.modes_per_degree),
        real_fields=list(args.real_fields),
        best_val_score=float(best),
        config_path=str(args.config), graph_suffix=cfg.graph_suffix,
        graph=dict(cfg.raw.get("graph", {})),
        use_config_features=bool(args.use_config_features),
        init_from_base=bool(args.init_from_base),
        train_pairs=train_pairs), args.out)
    print("wrote", args.out, "best_val_score=%.4e checkpoint_score=%s" % (best, args.checkpoint_score))
    print("HIGHORDER_TRAIN_DONE")


if __name__ == "__main__":
    main()
