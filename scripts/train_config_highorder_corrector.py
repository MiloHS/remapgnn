#!/usr/bin/env python
"""v7: signed-projection IRNO-style corrector for the supermesh-free higher-order remap operator.

Takes a FROZEN signed base (e.g. v6a mom1e4) and unrolls a small SHARED corrector that nudges the
SIGNED edge multiplier w over a spectral curriculum of bands (default lmax 8 -> 16 -> 24). After every
nudge the operator is RE-PROJECTED with doubly_constrained_project (signed, exact conservation +
consistency) -- NOT Sinkhorn-balanced. This is the one change that makes IRNO refinement compatible
with >1st order: the original irno_corrector re-balances with Sinkhorn (non-negative) and so re-caps
the operator at 1st order every step; the signed projection does not.

Parameterization (identical to train_config_highorder.py):
    q_k = M_base * (1 + scale * w_k),   w_k = w_{k-1} + alpha * tanh(corrector_delta_k)
    M_k = doubly_constrained_project(q_k),   S_k = M_k / area_tgt
with w_0 = the frozen base's signed head. M_base = uniform-within-target mass (supermesh-free).

Per-step supervision (spectral escalation): operator-edge MSE vs np2 (light on intermediate steps,
full on the final step), progressive harmonic loss (band lmax escalates), the deg-1 moment penalty
(linear reproduction -- the v6a lever), plus keep (drift) and delta (small-step) regularizers.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from remapgnn.projection import doubly_constrained_project_implicit
from train_config_balanced_harmonic import (
    set_seed, build_harmonic_fields_with_truth, harmonic_loss_from_operator,
    model_outputs_to_q, warn_split_leakage,
    read_source_xyz_from_edges, read_target_xyz_from_edges,
)
from train_config_irno_corrector import torch_load_pack, as_int
from train_config_highorder import load_np2_target

DEGREES_ALL = [0, 1, 2, 4, 8, 16, 24]


# --------------------------------------------------------------------------------------------------
# operator construction (signed projection path, shared by trainer + eval)
# --------------------------------------------------------------------------------------------------
def base_w_and_geom(base_model, batch, n_src, n_tgt):
    """Frozen base -> its SIGNED multiplier w0 (logit head) and the supermesh-free M_base. Constant
    per pair across epochs since the base is frozen, so callers cache the result."""
    out = base_model(batch["src_node_attr"], batch["tgt_node_attr"], batch["edge_attr"],
                     batch["src_index"], batch["tgt_index"], n_src, n_tgt)
    logit, raw, _ = model_outputs_to_q(out)
    ti = batch["tgt_index"]
    deg_t = scatter_sum_torch(torch.ones_like(raw.float()), ti, n_tgt)
    atgt = batch["area_tgt"].float()
    M_base = atgt[ti] / torch.clamp(deg_t[ti], min=1.0)
    w0 = (logit if logit is not None else raw).float()
    return w0, M_base


def operator_from_w(
    w,
    M_base,
    si,
    ti,
    asrc,
    atgt,
    n_src,
    n_tgt,
    scale,
    n_cg,
    moment_coef=None,
    solve_dtype=None,
    eps_rel=1e-9,
):
    q = M_base * (1.0 + scale * w)
    # implicit-diff projection: exact gradients with O(nodes) backward memory (no unrolled-CG graph),
    # which is what lets the 3-band rollout fit on the GPU for the large pairs. moment_coef (when given)
    # additionally enforces exact ℓ=1 (linear) reproduction -- the low-frequency band of the spectrum.
    M = doubly_constrained_project_implicit(
        q, si, ti, asrc, atgt, n_src, n_tgt,
        eps_rel=eps_rel,
        n_cg=n_cg,
        moment_coef=moment_coef,
        solve_dtype=solve_dtype,
    )
    S = M / torch.clamp(atgt[ti], min=1e-30)
    return S, M


def zscore_edge(x, eps=1e-6):
    return (x - x.mean()) / torch.clamp(x.std(), min=eps)


def make_aug(edge_attr, w, S, step_frac, lmax_frac):
    """Augment base edge features with current state (z-scored signed w and signed S, both safe for
    negative values -- no log) plus the step/band conditioning that makes ONE corrector behave as an
    iterated fixed-point solver. +4 columns, matching corrector edge_dim = len(edge_features)+4."""
    w_z = zscore_edge(w.float())
    S_z = zscore_edge(S.float())
    n = edge_attr.shape[0]
    step_col = torch.full((n, 1), float(step_frac), device=edge_attr.device, dtype=edge_attr.dtype)
    lmax_col = torch.full((n, 1), float(lmax_frac), device=edge_attr.device, dtype=edge_attr.dtype)
    return torch.cat([edge_attr, w_z[:, None].to(edge_attr.dtype),
                      S_z[:, None].to(edge_attr.dtype), step_col, lmax_col], dim=1)


def corrector_delta_from_output(out):
    logit, raw, _ = model_outputs_to_q(out)
    return (logit if logit is not None else raw).float()


def move_batch(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def run_corrector_steps(base_model, corrector, batch, bands, alpha, scale, n_cg, lmax_denom,
                        w0=None, M_base=None, use_ckpt=True, moment_coef=None, solve_dtype=None,
                        eps_rel=1e-9):
    """Unroll the shared corrector over the bands. Returns (S0, steps, w_final) where steps is a list
    of (S_k, M_k, bounded_delta_k). Grad flows for training; call under no_grad for eval.

    Each corrector forward is gradient-CHECKPOINTED: holding all K full GNN activation graphs at once
    OOMs on the >1M-edge pairs, so we recompute each forward in backward (peak ~= one step's worth, and
    a single forward is known to fit). Disabled automatically when grad is off (eval)."""
    si, ti = batch["src_index"], batch["tgt_index"]
    n_src, n_tgt = as_int(batch["n_src"]), as_int(batch["n_tgt"])
    asrc, atgt = batch["area_src"].float(), batch["area_tgt"].float()
    if w0 is None or M_base is None:
        w0, M_base = base_w_and_geom(base_model, batch, n_src, n_tgt)
    sa, ta = batch["src_node_attr"], batch["tgt_node_attr"]

    def corrector_delta(aug):
        if use_ckpt and torch.is_grad_enabled():
            def _fwd(aug_, sa_, ta_):
                return corrector_delta_from_output(corrector(sa_, ta_, aug_, si, ti, n_src, n_tgt))
            return checkpoint(_fwd, aug, sa, ta, use_reentrant=False)
        return corrector_delta_from_output(corrector(sa, ta, aug, si, ti, n_src, n_tgt))

    w = w0
    S_cur, _ = operator_from_w(
        w, M_base, si, ti, asrc, atgt, n_src, n_tgt, scale, n_cg,
        moment_coef=moment_coef,
        solve_dtype=solve_dtype,
        eps_rel=eps_rel,
    )
    S0 = S_cur
    steps = []
    K = len(bands)
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != K:
            raise ValueError(f"alpha sequence length {len(alpha)} must match bands length {K}")
        alpha_seq = [float(a) for a in alpha]
    else:
        alpha_seq = [float(alpha)] * K
    for k, lmax in enumerate(bands, start=1):
        aug = make_aug(batch["edge_attr"], w, S_cur, k / max(K, 1), float(lmax) / lmax_denom)
        bd = torch.tanh(corrector_delta(aug))
        w = w + alpha_seq[k - 1] * bd
        S_cur, M_cur = operator_from_w(
            w, M_base, si, ti, asrc, atgt, n_src, n_tgt, scale, n_cg,
            moment_coef=moment_coef,
            solve_dtype=solve_dtype,
            eps_rel=eps_rel,
        )
        steps.append((S_cur, M_cur, bd))
    return S0, steps, w


# --------------------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------------------
def op_loss_fn(S, S_np2, insupp, rel):
    diff2 = (S - S_np2) ** 2
    insup = insupp; outsup = ~insup
    op_in = diff2[insup].mean() if bool(insup.any()) else S.new_zeros(())
    op_out = diff2[outsup].mean() if bool(outsup.any()) else S.new_zeros(())
    l = op_in + 0.05 * op_out
    if rel:
        denom = (S_np2[insup] ** 2).mean() if bool(insup.any()) else S.new_ones(())
        l = l / denom.clamp_min(1e-12)
    return l, op_in.detach()


def moment_loss_fn(S, si, ti, sxyz, txyz, atgt, n_tgt):
    num = S.new_zeros(()); den = S.new_zeros(())
    for d in range(3):
        pred = scatter_sum_torch(S * sxyz[si, d], ti, n_tgt)
        num = num + (atgt * (pred - txyz[:, d]) ** 2).sum()
        den = den + (atgt * txyz[:, d] ** 2).sum()
    return num / den.clamp_min(1e-12)


def conservation_resid(M, si, ti, asrc, atgt, n_src, n_tgt):
    sm = scatter_sum_torch(M, si, n_src); tm = scatter_sum_torch(M, ti, n_tgt)
    cr = torch.linalg.norm(sm - asrc) / torch.linalg.norm(asrc)
    rr = torch.linalg.norm(tm - atgt) / torch.linalg.norm(atgt)
    return float(cr), float(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-pack", required=True, help="frozen base pack (a signed highorder .pt, e.g. v6a mom1e4)")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--val-pairs", nargs="+", default=None)
    ap.add_argument("--out", default="models_medium_improv/highorder_corrector_v7.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bands", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--alpha", type=float, default=0.2, help="per-step log-multiplier step size (bounded by tanh)")
    ap.add_argument("--lmax-denom", type=float, default=32.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=1, help="message-passing rounds for the corrector")
    ap.add_argument("--n-cg", type=int, default=400, help="CG iters in the doubly-constrained projection")
    ap.add_argument("--lam-op", type=float, default=1.0, help="operator-edge weight on the FINAL step")
    ap.add_argument("--lam-step-op", type=float, default=0.15, help="operator-edge weight on intermediate steps")
    ap.add_argument("--lam-field", type=float, default=3.0, help="harmonic (spectral) loss weight per step")
    ap.add_argument("--lam-moment", type=float, default=1e4, help="deg-1 linear-reproduction penalty per step")
    ap.add_argument("--lam-keep", type=float, default=0.01, help="drift-from-previous-step penalty (contraction)")
    ap.add_argument("--lam-delta", type=float, default=1e-4, help="small-step penalty on the bounded delta")
    ap.add_argument("--rel-op", action="store_true", help="relative (scale-free) operator-edge loss")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)

    pack = torch_load_pack(args.base_pack, map_location=device)
    stats = pack["stats"]
    sf = list(pack["src_node_features"]); tf = list(pack["tgt_node_features"]); ef = list(pack["edge_features"])
    base_scale = float(pack.get("scale", args.scale))

    # frozen base
    base_model = build_model(architecture=pack.get("architecture", cfg.architecture),
                             src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef),
                             hidden=int(pack.get("hidden", 128)),
                             decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(device)
    base_model.load_state_dict(pack["model_state_dict"])
    base_model.num_rounds = int(pack.get("rounds", 1))
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    # fresh corrector: same architecture, edge_dim + 4 (w_z, S_z, step_frac, lmax_frac)
    corrector = build_model(architecture=pack.get("architecture", cfg.architecture),
                            src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef) + 4,
                            hidden=int(pack.get("hidden", 128)),
                            decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(device)
    corrector.num_rounds = args.rounds
    opt = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=1e-5)

    bands = list(args.bands)
    print("bands:", bands, " alpha:", args.alpha, " scale:", base_scale, " rounds:", args.rounds,
          " n_cg:", args.n_cg, " base:", args.base_pack)

    tr = cfg.training if hasattr(cfg, "training") else {}
    train_pairs = list(getattr(cfg, "pairs", []))
    train_pairs = list(tr.get("train_pairs", train_pairs)) if isinstance(tr, dict) else train_pairs
    if args.pairs:
        train_pairs = args.pairs
    if args.smoke:
        train_pairs = train_pairs[:1]; args.epochs = 2; args.n_cg = 60
    print("train pairs:", train_pairs)

    unique_bands = sorted(set(int(b) for b in bands))

    cpu = torch.device("cpu")

    def build_cache(pairs):
        # CPU-resident cache: holding all 7 pairs (incl. >1M-edge ones) on the GPU at once OOMs, so we
        # stream -- store everything on CPU and move one pair to the GPU per step in the loop.
        c = {}
        for pair in pairs:
            b = load_pair_tensors(cfg, pair, stats, device=cpu)
            n_src, n_tgt = as_int(b["n_src"]), as_int(b["n_tgt"])
            S_np2, insupp = load_np2_target(pair, b["src_index"], b["tgt_index"], cpu)
            sxyz = torch.tensor(read_source_xyz_from_edges(cfg.edge_path(pair), n_src), dtype=torch.float32)
            txyz = torch.tensor(read_target_xyz_from_edges(cfg.edge_path(pair), n_tgt), dtype=torch.float32)
            bandfields = {}
            for lmax in unique_bands:
                degs = [d for d in DEGREES_ALL if d <= lmax]
                sfld, tfld = build_harmonic_fields_with_truth(cfg, pair, n_src, n_tgt, degs, 4, 0)
                bandfields[lmax] = (sfld.cpu(), tfld.cpu())
            # frozen-base multiplier w0 + M_base: compute once on GPU, store on CPU (constant per pair)
            with torch.no_grad():
                bd = move_batch(b, device)
                w0, M_base = base_w_and_geom(base_model, bd, n_src, n_tgt)
                w0, M_base = w0.cpu(), M_base.cpu()
                del bd
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            c[pair] = dict(b=b, n_src=n_src, n_tgt=n_tgt, S_np2=S_np2, insupp=insupp,
                           sxyz=sxyz, txyz=txyz, bandfields=bandfields, w0=w0, M_base=M_base)
            print(f"  cached {pair}: edges={b['src_index'].numel()} np2_in_cand={int(insupp.sum())}/{insupp.numel()}")
        return c

    def to_device(c):
        # move one pair's cache entry onto the GPU for a step; freed after use
        return dict(
            b=move_batch(c["b"], device), n_src=c["n_src"], n_tgt=c["n_tgt"],
            S_np2=c["S_np2"].to(device), insupp=c["insupp"].to(device),
            sxyz=c["sxyz"].to(device), txyz=c["txyz"].to(device),
            w0=c["w0"].to(device), M_base=c["M_base"].to(device),
            bandfields={lm: (sf.to(device), tf.to(device)) for lm, (sf, tf) in c["bandfields"].items()})

    cache = build_cache(train_pairs)
    val_pairs = args.val_pairs or []
    valcache = build_cache(val_pairs) if val_pairs else {}
    print("val pairs:", val_pairs)
    warn_split_leakage(train_pairs, val_pairs, val_pairs[0] if val_pairs else train_pairs[0])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best = float("inf"); best_state = None
    K = len(bands)

    for epoch in range(1, args.epochs + 1):
        corrector.train()
        ep_op = []; ep_mom = []; ep_drms = []
        cr = rr = 0.0
        for pair in train_pairs:
            c = to_device(cache[pair]); b = c["b"]
            si, ti = b["src_index"], b["tgt_index"]
            asrc, atgt = b["area_src"].float(), b["area_tgt"].float()
            scale_den = (c["S_np2"][c["insupp"]] ** 2).mean().clamp_min(1e-20) if bool(c["insupp"].any()) else atgt.new_ones(())
            opt.zero_grad(set_to_none=True)
            S0, steps, _ = run_corrector_steps(base_model, corrector, b, bands, args.alpha, base_scale,
                                               args.n_cg, args.lmax_denom, w0=c["w0"], M_base=c["M_base"])
            total = atgt.new_zeros(())
            final_op_in = None; last_M = None
            for k, (S_k, M_k, bd_k) in enumerate(steps, start=1):
                is_final = (k == K)
                op, op_in = op_loss_fn(S_k, c["S_np2"], c["insupp"], rel=args.rel_op)
                sfld, tfld = c["bandfields"][bands[k - 1]]
                h_loss, _ = harmonic_loss_from_operator(
                    S_pred=S_k, S_true=c["S_np2"], src_index=si, tgt_index=ti,
                    edge_exists=torch.ones_like(S_k), harmonic_fields=sfld, n_tgt=c["n_tgt"],
                    max_fields_per_step=0, target_fields=tfld)
                mom = moment_loss_fn(S_k, si, ti, c["sxyz"], c["txyz"], atgt, c["n_tgt"])
                S_prev = S0 if k == 1 else steps[k - 2][0]
                keep = ((S_k - S_prev.detach()) ** 2).mean() / scale_den
                dloss = (bd_k ** 2).mean()
                op_w = args.lam_op if is_final else args.lam_step_op
                total = total + op_w * op + args.lam_field * h_loss + args.lam_moment * mom \
                    + args.lam_keep * keep + args.lam_delta * dloss
                if is_final:
                    final_op_in = float(op_in); ep_mom.append(float(mom)); ep_drms.append(float(dloss.sqrt()))
                    last_M = M_k
            total.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            opt.step()
            ep_op.append(final_op_in)
            cr, rr = conservation_resid(last_M.detach(), si, ti, asrc, atgt, c["n_src"], c["n_tgt"])
            del c, b, steps, total, last_M, S0
            if device.type == "cuda":
                torch.cuda.empty_cache()
        sched.step()

        corrector.eval()
        with torch.no_grad():
            vops = []
            for pair in val_pairs:
                cv = to_device(valcache[pair]); bv = cv["b"]
                _, vsteps, _ = run_corrector_steps(base_model, corrector, bv, bands, args.alpha, base_scale,
                                                   args.n_cg, args.lmax_denom, w0=cv["w0"], M_base=cv["M_base"])
                Sv = vsteps[-1][0]
                d = (Sv - cv["S_np2"]) ** 2
                vm = cv["insupp"]
                vops.append(float(d[vm].mean()) if bool(vm.any()) else 0.0)
                del cv, bv, vsteps, Sv
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            val_op = float(np.mean(vops)) if vops else float(np.mean(ep_op))
        if val_op < best:
            best = val_op
            best_state = {k: v.detach().cpu().clone() for k, v in corrector.state_dict().items()}
        print("epoch %04d  train_op=%.4e  val_op=%.4e (best %.4e)  mom=%.4e  drms=%.3f  cons=%.2e row=%.2e"
              % (epoch, float(np.mean(ep_op)), val_op, best,
                 (float(np.mean(ep_mom)) if ep_mom else 0.0),
                 (float(np.mean(ep_drms)) if ep_drms else 0.0), cr, rr))

    if best_state is not None:
        corrector.load_state_dict(best_state)
    torch.save(dict(
        kind="highorder_corrector",
        model_state_dict={k: v.detach().cpu() for k, v in corrector.state_dict().items()},
        architecture=pack.get("architecture", cfg.architecture), hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
        src_node_features=sf, tgt_node_features=tf, edge_features=ef, stats=stats,
        scale=base_scale, rounds=args.rounds, bands=bands, alpha=args.alpha, lmax_denom=args.lmax_denom,
        n_cg=args.n_cg, base_pack=str(args.base_pack), train_pairs=train_pairs), args.out)
    print("wrote", args.out, "best_val_op=%.4e" % best)
    print("HIGHORDER_CORRECTOR_TRAIN_DONE")


if __name__ == "__main__":
    main()
