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
from remapgnn.projection import doubly_constrained_project
from train_config_balanced_harmonic import (
    set_seed, build_harmonic_fields_with_truth, harmonic_loss_from_operator,
    model_outputs_to_q, warn_split_leakage,
    read_source_xyz_from_edges, read_target_xyz_from_edges,
)
from train_config_irno_corrector import torch_load_pack, as_int

DEGREES = [0, 1, 2, 4, 8, 16, 24]


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
    M = doubly_constrained_project(
        q, si, ti, area_src, area_tgt, n_src, n_tgt,
        eps_rel=eps_rel,
        n_cg=n_cg,
        solve_dtype=solve_dtype,
    )
    S = M / torch.clamp(area_tgt[ti], min=1e-30)
    return S, M


def conservation_resid(M, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt):
    sm = scatter_sum_torch(M, src_index, n_src); tm = scatter_sum_torch(M, tgt_index, n_tgt)
    cr = torch.linalg.norm(sm - area_src) / torch.linalg.norm(area_src)
    rr = torch.linalg.norm(tm - area_tgt) / torch.linalg.norm(area_tgt)
    return float(cr), float(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)            # base config (pairs, graph suffix, features)
    ap.add_argument("--base-pack", required=True)         # an existing base .pt for stats + feature lists
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
    ap.add_argument("--scale", type=float, default=1.0)
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)
    pack = torch_load_pack(args.base_pack, map_location=device)
    print("message-passing rounds:", args.rounds, " signed:", args.signed, " rel_op:", args.rel_op,
          " lam_moment:", args.lam_moment, " use_config_features:", args.use_config_features)

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
    model.num_rounds = args.rounds
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # per-pair caches (supermesh-free graph + np2 teacher + truth harmonics)
    cache = {}
    for pair in train_pairs:
        b = load_pair_tensors(cfg, pair, stats, device=device)
        n_src, n_tgt = as_int(b["n_src"]), as_int(b["n_tgt"])
        S_np2, insupp = load_np2_target(pair, b["src_index"], b["tgt_index"], device)
        sfld, tfld = build_harmonic_fields_with_truth(cfg, pair, n_src, n_tgt, DEGREES, 4, 0)
        sxyz = torch.tensor(read_source_xyz_from_edges(cfg.edge_path(pair), n_src), dtype=torch.float32, device=device)
        txyz = torch.tensor(read_target_xyz_from_edges(cfg.edge_path(pair), n_tgt), dtype=torch.float32, device=device)
        cache[pair] = dict(b=b, n_src=n_src, n_tgt=n_tgt, S_np2=S_np2, insupp=insupp,
                           sfld=sfld.to(device), tfld=tfld.to(device), sxyz=sxyz, txyz=txyz)
        print(f"  cached {pair}: edges={b['src_index'].numel()} np2_support_in_cand={int(insupp.sum())}/{insupp.numel()}")

    val_pairs = args.val_pairs or []
    valcache = {}
    for pair in val_pairs:
        bv = load_pair_tensors(cfg, pair, stats, device=device)
        S2v, insv = load_np2_target(pair, bv["src_index"], bv["tgt_index"], device)
        valcache[pair] = dict(b=bv, n_src=as_int(bv["n_src"]), n_tgt=as_int(bv["n_tgt"]), S_np2=S2v, insupp=insv)
    print("val pairs:", val_pairs)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best = float("inf"); best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_op = []; ep_mom = []
        # moment weight for THIS epoch: linear ramp from lam_moment -> lam_moment_final if annealing requested
        if args.lam_moment_final is not None and args.epochs > 1:
            frac = (epoch - 1) / (args.epochs - 1)
            lam_moment_ep = args.lam_moment + frac * (args.lam_moment_final - args.lam_moment)
        else:
            lam_moment_ep = args.lam_moment
        for pair in train_pairs:
            c = cache[pair]; b = c["b"]
            asrc = b["area_src"].float(); atgt = b["area_tgt"].float()
            opt.zero_grad(set_to_none=True)
            S, M = operator_from_model(model, b, asrc, atgt, c["n_src"], c["n_tgt"], args.scale, signed=args.signed)
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
                S_pred=S, S_true=c["S_np2"], src_index=b["src_index"], tgt_index=b["tgt_index"],
                edge_exists=torch.ones_like(S), harmonic_fields=c["sfld"], n_tgt=c["n_tgt"],
                max_fields_per_step=0, target_fields=c["tfld"])
            loss = args.lam_op * op_loss + args.lam_field * h_loss
            if lam_moment_ep > 0.0:
                # degree-1 moment / linear-reproduction residual: applying S to each source coordinate
                # field must return the target coordinate field. area-weighted relative MSE over x,y,z.
                si, ti = b["src_index"], b["tgt_index"]
                mom_num = S.new_zeros(()); mom_den = S.new_zeros(())
                for d in range(3):
                    pred_d = scatter_sum_torch(S * c["sxyz"][si, d], ti, c["n_tgt"])
                    mom_num = mom_num + (atgt * (pred_d - c["txyz"][:, d]) ** 2).sum()
                    mom_den = mom_den + (atgt * c["txyz"][:, d] ** 2).sum()
                mom_loss = mom_num / mom_den.clamp_min(1e-12)
                loss = loss + lam_moment_ep * mom_loss
                ep_mom.append(float(mom_loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_op.append(float(op_loss))
        sched.step()

        model.eval()
        with torch.no_grad():
            vops = []
            for pair in val_pairs:
                cv = valcache[pair]; bv = cv["b"]
                Sv, _ = operator_from_model(model, bv, bv["area_src"].float(), bv["area_tgt"].float(),
                                            cv["n_src"], cv["n_tgt"], args.scale, signed=args.signed)
                d = (Sv - cv["S_np2"]) ** 2
                vm = cv["insupp"]
                vops.append(float(d[vm].mean()) if bool(vm.any()) else 0.0)
            val_op = float(np.mean(vops)) if vops else float(np.mean(ep_op))
        if val_op < best:
            best = val_op
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cr, rr = conservation_resid(M.detach(), b["src_index"], b["tgt_index"], asrc, atgt, c["n_src"], c["n_tgt"])
        print("epoch %04d  train_op=%.4e  val_op=%.4e (best %.4e)  h_rel=%.4e  mom=%.4e (lam=%.2e)  cons=%.2e row=%.2e"
              % (epoch, float(np.mean(ep_op)), val_op, best, float(h_rel),
                 (float(np.mean(ep_mom)) if ep_mom else 0.0), lam_moment_ep, cr, rr))

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(dict(
        model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        architecture=pack.get("architecture", cfg.architecture), hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
        src_node_features=sf, tgt_node_features=tf, edge_features=ef, stats=stats,
        scale=args.scale, rounds=args.rounds, signed=bool(args.signed),
        lam_moment=float(args.lam_moment),
        lam_moment_final=(float(args.lam_moment_final) if args.lam_moment_final is not None else None),
        config_path=str(args.config), graph_suffix=cfg.graph_suffix,
        graph=dict(cfg.raw.get("graph", {})),
        use_config_features=bool(args.use_config_features),
        train_pairs=train_pairs), args.out)
    print("wrote", args.out, "best_val_op=%.4e" % best)
    print("HIGHORDER_TRAIN_DONE")


if __name__ == "__main__":
    main()
