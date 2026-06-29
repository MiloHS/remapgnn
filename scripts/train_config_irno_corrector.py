from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors, get_feature_lists
from remapgnn.models import build_model
from remapgnn.sinkhorn import sparse_sinkhorn_balance, sparse_operator_weights, converged_balance

# Reuse harmonic utilities from the previous trainer.
from train_config_balanced_harmonic import (
    set_seed,
    build_harmonic_fields,
    build_harmonic_fields_with_truth,
    harmonic_loss_from_operator,
    model_outputs_to_q,
    warn_split_leakage,
)


def torch_load_pack(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unique_keep_order(xs):
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def safe_pair_name(pair: str) -> str:
    return pair.replace("-", "_").replace(".", "p").replace("/", "_").replace(":", "_")


def as_int(x):
    return int(x.item() if hasattr(x, "item") else x)


def get_batch_true_weight(batch):
    candidates = [
        "weight",
        "S_true",
        "s_true",
        "target_weight",
        "true_weight",
        "remap_weight",
        "edge_weight",
        "S",
        "s",
    ]
    for k in candidates:
        if k in batch:
            return batch[k].float()
    raise KeyError(f"No true Tempest edge weight key found. Available keys: {sorted(batch.keys())}")


def zscore_edge(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return (x - x.mean()) / torch.clamp(x.std(), min=eps)


def make_augmented_edge_attr(
    edge_attr: torch.Tensor,
    logq: torch.Tensor,
    S: torch.Tensor,
    step_frac: float,
    lmax_frac: float,
) -> torch.Tensor:
    logq_z = zscore_edge(logq.float())
    logS_z = zscore_edge(torch.log(torch.clamp(S.float(), min=1.0e-30)))

    n = edge_attr.shape[0]
    step_col = torch.full((n, 1), float(step_frac), device=edge_attr.device, dtype=edge_attr.dtype)
    lmax_col = torch.full((n, 1), float(lmax_frac), device=edge_attr.device, dtype=edge_attr.dtype)

    return torch.cat(
        [
            edge_attr,
            logq_z[:, None].to(edge_attr.dtype),
            logS_z[:, None].to(edge_attr.dtype),
            step_col,
            lmax_col,
        ],
        dim=1,
    )


def compute_operator_from_logq(
    logq: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    area_src: torch.Tensor,
    area_tgt: torch.Tensor,
    n_src: int,
    n_tgt: int,
    n_iter: int,
    warm_scale: torch.Tensor | None = None,
    return_scale: bool = False,
):
    q = torch.exp(torch.clamp(logq.float(), min=-60.0, max=40.0))
    # Train the corrector against a CONVERGED operator (conservative + consistent) using a
    # frozen-dual gradient, instead of a fixed under-converged unroll. `n_iter` is the max-iter cap.
    # warm_scale reuses the previous band/step's converged duals to converge in far fewer iters
    # (identical fixed point). return_scale yields that scale for the next warm start.
    out = converged_balance(
        q=q,
        src_index=src_index,
        tgt_index=tgt_index,
        area_src=area_src.float(),
        area_tgt=area_tgt.float(),
        n_src=n_src,
        n_tgt=n_tgt,
        tol=1.0e-6,
        max_iter=max(int(n_iter), 20000),
        warm_scale=warm_scale,
        return_scale=return_scale,
    )
    if return_scale:
        M, s = out
        S = sparse_operator_weights(M=M, tgt_index=tgt_index, area_tgt=area_tgt.float())
        return M, S, s
    M = out
    S = sparse_operator_weights(M=M, tgt_index=tgt_index, area_tgt=area_tgt.float())
    return M, S


def operator_edge_loss(S_pred, S_true, edge_exists, lambda_neg_s: float = 0.05):
    pos = edge_exists > 0.5
    neg = ~pos
    scale = torch.clamp(torch.mean(S_true[pos] ** 2), min=1.0e-20)

    loss_pos = torch.mean((S_pred[pos] - S_true[pos]) ** 2) / scale
    if neg.any():
        loss_neg = torch.mean(S_pred[neg] ** 2) / scale
    else:
        loss_neg = torch.zeros_like(loss_pos)

    rel_l2 = torch.sqrt(
        torch.sum((S_pred[pos] - S_true[pos]) ** 2)
        / torch.clamp(torch.sum(S_true[pos] ** 2), min=1.0e-20)
    )

    return loss_pos + lambda_neg_s * loss_neg, loss_pos, loss_neg, rel_l2


def conservation_metrics(M, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt):
    src_mass = torch.zeros(n_src, device=M.device, dtype=M.dtype)
    src_mass.index_add_(0, src_index, M)
    source_mass_rel_l2 = torch.linalg.norm(src_mass - area_src) / torch.clamp(
        torch.linalg.norm(area_src), min=1.0e-20
    )

    tgt_mass = torch.zeros(n_tgt, device=M.device, dtype=M.dtype)
    tgt_mass.index_add_(0, tgt_index, M)
    row_sum_rel_l2 = torch.linalg.norm(tgt_mass - area_tgt) / torch.clamp(
        torch.linalg.norm(area_tgt), min=1.0e-20
    )

    return source_mass_rel_l2, row_sum_rel_l2


def corrector_delta_from_output(out):
    edge_logit, raw_weight, _ = model_outputs_to_q(out)
    if edge_logit is not None:
        return edge_logit.float()
    return raw_weight.float()


def base_q_from_model(base_model, batch):
    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])
    out = base_model(
        batch["src_node_attr"],
        batch["tgt_node_attr"],
        batch["edge_attr"],
        batch["src_index"],
        batch["tgt_index"],
        n_src,
        n_tgt,
    )
    _, _, q = model_outputs_to_q(out)
    return q.float()


def rollout_irno_corrector(
    base_model,
    corrector,
    batch,
    harmonic_cache_for_pair,
    bands,
    alpha: float,
    lmax_denominator: float,
    n_sinkhorn_iter: int,
    lambda_operator: float,
    lambda_step_operator: float,
    lambda_field: float,
    lambda_keep: float,
    lambda_delta: float,
    lambda_neg_s: float,
    max_fields_per_step: int,
    pair_key: str | None = None,
    base_op_cache: dict | None = None,
):
    src_index = batch["src_index"]
    tgt_index = batch["tgt_index"]
    edge_exists = batch["edge_exists"].float()
    S_true = get_batch_true_weight(batch)

    area_src = batch["area_src"].float()
    area_tgt = batch["area_tgt"].float()
    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])

    # The base model is FROZEN, so its q0 and converged operator (M0, S0) are constant for a given
    # pair across all epochs -- cache them to skip one converged balance per step. s_base is the
    # base per-edge scale, reused to warm-start band 1.
    if base_op_cache is not None and pair_key in base_op_cache:
        logq, S0, s_base = base_op_cache[pair_key]
        last_M = None
    else:
        with torch.no_grad():
            q0 = base_q_from_model(base_model, batch)
            logq0 = torch.log(torch.clamp(q0, min=1.0e-30))
            M0, S0, s_base = compute_operator_from_logq(
                logq0,
                src_index,
                tgt_index,
                area_src,
                area_tgt,
                n_src,
                n_tgt,
                n_sinkhorn_iter,
                return_scale=True,
            )
        logq = logq0.detach()
        S0 = S0.detach()
        s_base = s_base.detach()
        last_M = M0
        if base_op_cache is not None:
            base_op_cache[pair_key] = (logq, S0, s_base)

    S_current = S0.detach()
    s_prev = s_base

    total = torch.zeros((), device=src_index.device)
    step_metrics = {}

    K = len(bands)
    last_S = S0

    for k, lmax in enumerate(bands, start=1):
        step_frac = k / max(K, 1)
        lmax_frac = float(lmax) / float(lmax_denominator)

        aug_edge_attr = make_augmented_edge_attr(
            batch["edge_attr"],
            logq,
            S_current,
            step_frac=step_frac,
            lmax_frac=lmax_frac,
        )

        out = corrector(
            batch["src_node_attr"],
            batch["tgt_node_attr"],
            aug_edge_attr,
            src_index,
            tgt_index,
            n_src,
            n_tgt,
        )
        delta = corrector_delta_from_output(out)

        bounded_delta = torch.tanh(delta)
        logq_new = logq + alpha * bounded_delta

        M_new, S_new, s_new = compute_operator_from_logq(
            logq_new,
            src_index,
            tgt_index,
            area_src,
            area_tgt,
            n_src,
            n_tgt,
            n_sinkhorn_iter,
            warm_scale=s_prev,
            return_scale=True,
        )
        s_prev = s_new

        op_loss_k, op_pos_k, op_neg_k, rel_l2_k = operator_edge_loss(
            S_new, S_true, edge_exists, lambda_neg_s=lambda_neg_s
        )

        src_f, tgt_f = harmonic_cache_for_pair[int(lmax)]
        src_f = src_f.to(S_new.device, dtype=S_new.dtype)
        tgt_f = tgt_f.to(S_new.device, dtype=S_new.dtype) if tgt_f is not None else None
        h_loss_k, h_rel_k = harmonic_loss_from_operator(
            S_pred=S_new,
            S_true=S_true,
            src_index=src_index,
            tgt_index=tgt_index,
            edge_exists=edge_exists,
            harmonic_fields=src_f,
            n_tgt=n_tgt,
            max_fields_per_step=max_fields_per_step,
            target_fields=tgt_f,
        )

        scale = torch.clamp(torch.mean(S_true[edge_exists > 0.5] ** 2), min=1.0e-20)
        keep_loss_k = torch.mean((S_new - S_current.detach()) ** 2) / scale
        delta_loss_k = torch.mean(bounded_delta ** 2)

        # Intermediate steps get harmonic supervision and light operator supervision.
        # Final step gets full operator supervision.
        is_final = k == K
        op_weight = lambda_operator if is_final else lambda_step_operator

        total = (
            total
            + op_weight * op_loss_k
            + lambda_field * h_loss_k
            + lambda_keep * keep_loss_k
            + lambda_delta * delta_loss_k
        )

        step_metrics[f"step{k}_lmax"] = float(lmax)
        step_metrics[f"step{k}_rel_l2_positive_edges"] = float(rel_l2_k.detach().cpu())
        step_metrics[f"step{k}_harmonic_rel_l2"] = float(h_rel_k.detach().cpu())
        step_metrics[f"step{k}_keep_loss"] = float(keep_loss_k.detach().cpu())
        step_metrics[f"step{k}_delta_rms"] = float(torch.sqrt(delta_loss_k).detach().cpu())

        logq = logq_new
        S_current = S_new
        last_M, last_S = M_new, S_new

    final_op_loss, final_pos_loss, final_neg_loss, final_rel_l2 = operator_edge_loss(
        last_S, S_true, edge_exists, lambda_neg_s=lambda_neg_s
    )
    source_rel, row_rel = conservation_metrics(
        last_M, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt
    )

    final_lmax = int(bands[-1])
    final_src, final_tgt = harmonic_cache_for_pair[final_lmax]
    final_src = final_src.to(last_S.device, dtype=last_S.dtype)
    final_tgt = final_tgt.to(last_S.device, dtype=last_S.dtype) if final_tgt is not None else None
    final_h_loss, final_h_rel = harmonic_loss_from_operator(
        S_pred=last_S,
        S_true=S_true,
        src_index=src_index,
        tgt_index=tgt_index,
        edge_exists=edge_exists,
        harmonic_fields=final_src,
        n_tgt=n_tgt,
        max_fields_per_step=0,  # all final-band fields for diagnostics
        target_fields=final_tgt,
    )

    metrics = {
        "loss": float(total.detach().cpu()),
        "final_operator_loss": float(final_op_loss.detach().cpu()),
        "final_pos_loss": float(final_pos_loss.detach().cpu()),
        "final_neg_loss": float(final_neg_loss.detach().cpu()),
        "final_rel_l2_positive_edges": float(final_rel_l2.detach().cpu()),
        "final_harmonic_rel_l2": float(final_h_rel.detach().cpu()),
        "source_mass_rel_l2": float(source_rel.detach().cpu()),
        "row_sum_rel_l2": float(row_rel.detach().cpu()),
        **step_metrics,
    }

    return total, metrics


def eval_one_pair(
    base_model,
    corrector,
    cfg,
    pair,
    stats,
    harmonic_cache,
    device,
    bands,
    params,
    base_op_cache=None,
):
    base_model.eval()
    corrector.eval()
    batch = load_pair_tensors(cfg, pair, stats, device=device)

    with torch.no_grad():
        _, metrics = rollout_irno_corrector(
            base_model=base_model,
            corrector=corrector,
            batch=batch,
            harmonic_cache_for_pair=harmonic_cache[pair],
            bands=bands,
            alpha=params["alpha"],
            lmax_denominator=params["lmax_denominator"],
            n_sinkhorn_iter=params["n_eval_iter"],
            lambda_operator=params["lambda_operator"],
            lambda_step_operator=params["lambda_step_operator"],
            lambda_field=params["lambda_field"],
            lambda_keep=params["lambda_keep"],
            lambda_delta=params["lambda_delta"],
            lambda_neg_s=params["lambda_neg_s"],
            max_fields_per_step=0,
            pair_key=pair,
            base_op_cache=base_op_cache,
        )

    del batch
    return metrics


def eval_pair_set(base_model, corrector, cfg, pairs, stats, harmonic_cache, device, bands, params, prefix, base_op_cache=None):
    rows = {}
    rels, rowsums, fields = [], [], []

    for pair in pairs:
        metrics = eval_one_pair(
            base_model=base_model,
            corrector=corrector,
            cfg=cfg,
            pair=pair,
            stats=stats,
            harmonic_cache=harmonic_cache,
            device=device,
            bands=bands,
            params=params,
            base_op_cache=base_op_cache,
        )

        tag = safe_pair_name(pair)
        for k, v in metrics.items():
            rows[f"{prefix}_{tag}_{k}"] = v

        rels.append(metrics["final_rel_l2_positive_edges"])
        rowsums.append(metrics["row_sum_rel_l2"])
        fields.append(metrics["final_harmonic_rel_l2"])

    rows[f"{prefix}_mean_final_rel_l2_positive_edges"] = float(np.mean(rels))
    rows[f"{prefix}_max_final_rel_l2_positive_edges"] = float(np.max(rels))
    rows[f"{prefix}_mean_row_sum_rel_l2"] = float(np.mean(rowsums))
    rows[f"{prefix}_max_row_sum_rel_l2"] = float(np.max(rowsums))
    rows[f"{prefix}_mean_final_harmonic_rel_l2"] = float(np.mean(fields))
    rows[f"{prefix}_max_final_harmonic_rel_l2"] = float(np.max(fields))
    return rows


def build_harmonic_cache(cfg, pairs, stats, bands, modes_per_degree, seed, truth_target=False):
    # Each cache entry is a (src_fields, tgt_truth_fields) tuple. tgt_truth_fields is None unless
    # truth_target=True, in which case the spectral loss targets analytic truth on the target grid
    # (ground truth) instead of TempestRemap's action.
    cache = {}
    unique_bands = sorted(set(int(b) for b in bands))

    for pair in pairs:
        tmp = load_pair_tensors(cfg, pair, stats, device=torch.device("cpu"))
        n_src = as_int(tmp["n_src"])
        n_tgt = as_int(tmp["n_tgt"])
        del tmp

        cache[pair] = {}
        for lmax in unique_bands:
            degrees = [0, 1, 2, 4, 8, 12, 16, 24, 32]
            degrees = [d for d in degrees if d <= lmax]
            if truth_target:
                src_f, tgt_f = build_harmonic_fields_with_truth(
                    cfg=cfg, pair=pair, n_src=n_src, n_tgt=n_tgt,
                    degrees=degrees, modes_per_degree=modes_per_degree, seed=seed,
                )
            else:
                src_f = build_harmonic_fields(
                    cfg=cfg, pair=pair, n_src=n_src,
                    degrees=degrees, modes_per_degree=modes_per_degree, seed=seed,
                )
                tgt_f = None
            cache[pair][int(lmax)] = (src_f, tgt_f)

    return cache


def make_pack(
    corrector,
    cfg,
    base_cfg_path,
    base_model_path,
    params,
    edge_features,
    src_features,
    tgt_features,
    train_pairs,
    checkpoint_pairs,
    test_pair,
):
    return {
        "kind": "irno_corrector",
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in corrector.state_dict().items()},
        "run_name": cfg.run_name,
        "model_tag": cfg.model_tag,
        "architecture": cfg.architecture,
        "corrector_architecture": params["corrector_architecture"],
        "base_config": str(base_cfg_path),
        "base_model_path": str(base_model_path),
        "edge_features": edge_features,
        "src_node_features": src_features,
        "tgt_node_features": tgt_features,
        "corrector_edge_dim": len(edge_features) + 4,
        "hidden": params["hidden"],
        "decoder_chunk_size": params["decoder_chunk_size"],
        "bands": params["bands"],
        "alpha": params["alpha"],
        "lmax_denominator": params["lmax_denominator"],
        "modes_per_degree": params["modes_per_degree"],
        "train_pairs": list(train_pairs),
        "checkpoint_pairs": list(checkpoint_pairs),
        "test_pair": test_pair,
        "loss_params": {
            "lambda_operator": params["lambda_operator"],
            "lambda_step_operator": params["lambda_step_operator"],
            "lambda_field": params["lambda_field"],
            "lambda_keep": params["lambda_keep"],
            "lambda_delta": params["lambda_delta"],
            "lambda_neg_s": params["lambda_neg_s"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg.raw.get("training", {})
    irno = cfg.raw.get("irno_corrector", {})

    seed = int(tr.get("seed", 123))
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    base_cfg_path = Path(irno.get("base_config", "configs/v20b_base_a3p0_mink8.json"))
    base_cfg = load_config(str(base_cfg_path))
    base_model_path = Path(irno.get("base_model_path", str(base_cfg.model_path)))

    base_pack = torch_load_pack(base_model_path, map_location=device)
    stats = base_pack["stats"]

    edge_features = list(base_pack.get("edge_features", get_feature_lists(base_cfg)[0]))
    src_features = list(base_pack.get("src_node_features", get_feature_lists(base_cfg)[1]))
    tgt_features = list(base_pack.get("tgt_node_features", get_feature_lists(base_cfg)[2]))

    params = {
        "corrector_architecture": irno.get("corrector_architecture", "gated_hybrid_attention"),
        "hidden": int(irno.get("hidden", tr.get("hidden", 128))),
        "decoder_chunk_size": int(irno.get("decoder_chunk_size", tr.get("decoder_chunk_size", 10000))),
        "bands": [int(x) for x in irno.get("bands", [8, 16, 24])],
        "alpha": float(irno.get("alpha", 0.2)),
        "lmax_denominator": float(irno.get("lmax_denominator", 32.0)),
        "modes_per_degree": int(irno.get("modes_per_degree", 5)),
        "lambda_operator": float(irno.get("lambda_operator", 1.0)),
        "lambda_step_operator": float(irno.get("lambda_step_operator", 0.15)),
        "lambda_field": float(irno.get("lambda_field", 0.05)),
        "lambda_keep": float(irno.get("lambda_keep", 0.01)),
        "lambda_delta": float(irno.get("lambda_delta", 1.0e-4)),
        "lambda_neg_s": float(irno.get("lambda_neg_s", 0.05)),
        "checkpoint_field_weight": float(irno.get("checkpoint_field_weight", 0.15)),
        "row_weight": float(irno.get("row_weight", tr.get("checkpoint_score", {}).get("row_weight", 0.05))),
        "n_train_iter": int(irno.get("sinkhorn_iters_train", tr.get("sinkhorn_iters_train", 30))),
        "n_eval_iter": int(irno.get("sinkhorn_iters_eval", tr.get("sinkhorn_iters_eval", 300))),
        "max_fields_per_step": int(irno.get("max_fields_per_step", 8)),
        # "tempest" (default, legacy) = spectral loss matches Tempest's action; "truth" = matches analytic
        # ground truth on the target grid (won't regress low-l where the base already beats Tempest).
        "harmonic_target": str(irno.get("harmonic_target", "tempest")),
    }

    epochs = int(args.epochs or tr.get("epochs", 80))
    lr = float(irno.get("lr", tr.get("lr", 2.0e-4)))
    weight_decay = float(irno.get("weight_decay", tr.get("weight_decay", 1.0e-5)))

    train_pairs = list(tr.get("train_pairs", cfg.pairs))
    checkpoint_pairs = list(tr.get("checkpoint_pairs", tr.get("validation_pairs", [train_pairs[0]])))
    test_pair = tr.get("test_pair", cfg.pairs[0])

    warn_split_leakage(train_pairs, checkpoint_pairs, test_pair)

    if args.smoke:
        print()
        print("SMOKE MODE: one train pair, one epoch, low Sinkhorn iterations.")
        train_pairs = train_pairs[:1]
        checkpoint_pairs = checkpoint_pairs[:1]
        epochs = 1
        params["n_train_iter"] = 3
        params["n_eval_iter"] = 3
        params["max_fields_per_step"] = min(params["max_fields_per_step"], 4)

    print(f"Config:       {args.config}")
    print(f"Run name:     {cfg.run_name}")
    print(f"Model tag:    {cfg.model_tag}")
    print(f"Device:       {device}")
    print()
    print(f"Frozen base config: {base_cfg_path}")
    print(f"Frozen base model:  {base_model_path}")
    print()
    print("IRNO corrector:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    base_model = build_model(
        architecture=base_pack.get("architecture", base_cfg.architecture),
        src_dim=len(src_features),
        tgt_dim=len(tgt_features),
        edge_dim=len(edge_features),
        hidden=int(base_pack.get("hidden", 128)),
        decoder_chunk_size=int(base_pack.get("decoder_chunk_size", params["decoder_chunk_size"])),
    ).to(device)
    base_model.load_state_dict(base_pack["model_state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    corrector_edge_dim = len(edge_features) + 4
    corrector = build_model(
        architecture=params["corrector_architecture"],
        src_dim=len(src_features),
        tgt_dim=len(tgt_features),
        edge_dim=corrector_edge_dim,
        hidden=params["hidden"],
        decoder_chunk_size=params["decoder_chunk_size"],
    ).to(device)

    optimizer = torch.optim.AdamW(corrector.parameters(), lr=lr, weight_decay=weight_decay)

    cache_pairs = unique_keep_order(train_pairs + checkpoint_pairs + [test_pair])
    print()
    print("Building harmonic cache...")
    harmonic_cache = build_harmonic_cache(
        cfg=cfg,
        pairs=cache_pairs,
        stats=stats,
        bands=params["bands"],
        modes_per_degree=params["modes_per_degree"],
        seed=seed,
        truth_target=(params["harmonic_target"] == "truth"),
    )

    model_out = cfg.model_path
    history_out = cfg.history_path
    if args.smoke:
        model_out = cfg.model_path.parent / f"smoke_{cfg.model_tag}.pt"
        history_out = cfg.history_path.parent / f"smoke_{cfg.model_tag}_history.csv"

    print()
    print("Training IRNO-style corrector...")
    print(f"Train pairs:      {train_pairs}")
    print(f"Checkpoint pairs: {checkpoint_pairs}")
    print(f"Test pair:        {test_pair}")
    print(f"Epochs:           {epochs}")
    print()

    best_score = float("inf")
    best_epoch = None
    best_pack = None
    history = []
    t0 = time.time()

    # Frozen base -> its converged operator is constant per pair; cache across epochs (train+eval).
    base_op_cache = {}

    for epoch in range(1, epochs + 1):
        epoch_pairs = list(train_pairs)
        np.random.shuffle(epoch_pairs)

        last_train_metrics = {}

        for pair in epoch_pairs:
            corrector.train()
            batch = load_pair_tensors(cfg, pair, stats, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = rollout_irno_corrector(
                base_model=base_model,
                corrector=corrector,
                batch=batch,
                harmonic_cache_for_pair=harmonic_cache[pair],
                bands=params["bands"],
                alpha=params["alpha"],
                lmax_denominator=params["lmax_denominator"],
                n_sinkhorn_iter=params["n_train_iter"],
                lambda_operator=params["lambda_operator"],
                lambda_step_operator=params["lambda_step_operator"],
                lambda_field=params["lambda_field"],
                lambda_keep=params["lambda_keep"],
                lambda_delta=params["lambda_delta"],
                lambda_neg_s=params["lambda_neg_s"],
                max_fields_per_step=params["max_fields_per_step"],
                pair_key=pair,
                base_op_cache=base_op_cache,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=1.0)
            optimizer.step()

            last_train_metrics = train_metrics
            del batch

        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            ckpt_rows = eval_pair_set(
                base_model=base_model,
                corrector=corrector,
                cfg=cfg,
                pairs=checkpoint_pairs,
                stats=stats,
                harmonic_cache=harmonic_cache,
                device=device,
                bands=params["bands"],
                params=params,
                prefix="ckpt",
                base_op_cache=base_op_cache,
            )

            test_metrics = eval_one_pair(
                base_model=base_model,
                corrector=corrector,
                cfg=cfg,
                pair=test_pair,
                stats=stats,
                harmonic_cache=harmonic_cache,
                device=device,
                bands=params["bands"],
                params=params,
                base_op_cache=base_op_cache,
            )

            row = {
                "epoch": epoch,
                "elapsed_sec": time.time() - t0,
                **{f"train_last_{k}": v for k, v in last_train_metrics.items()},
                **ckpt_rows,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            score = (
                row["ckpt_mean_final_rel_l2_positive_edges"]
                + params["row_weight"] * row["ckpt_mean_row_sum_rel_l2"]
                + params["checkpoint_field_weight"] * row["ckpt_mean_final_harmonic_rel_l2"]
            )
            row["checkpoint_score"] = float(score)

            is_best = score < best_score
            if is_best:
                best_score = score
                best_epoch = epoch
                best_pack = make_pack(
                    corrector=corrector,
                    cfg=cfg,
                    base_cfg_path=base_cfg_path,
                    base_model_path=base_model_path,
                    params=params,
                    edge_features=edge_features,
                    src_features=src_features,
                    tgt_features=tgt_features,
                    train_pairs=train_pairs,
                    checkpoint_pairs=checkpoint_pairs,
                    test_pair=test_pair,
                )

            history.append(row)
            pd.DataFrame(history).to_csv(history_out, index=False)

            print(
                f"epoch {epoch:04d} "
                f"score={score:.6e} "
                f"ckpt_edge={row['ckpt_mean_final_rel_l2_positive_edges']:.4e} "
                f"ckpt_harm={row['ckpt_mean_final_harmonic_rel_l2']:.4e} "
                f"ckpt_row={row['ckpt_mean_row_sum_rel_l2']:.2e} "
                f"test_edge={row['test_final_rel_l2_positive_edges']:.4e} "
                f"test_harm={row['test_final_harmonic_rel_l2']:.4e} "
                f"test_row={row['test_row_sum_rel_l2']:.2e}"
                + (" *BEST*" if is_best else "")
            )

    if best_pack is None:
        raise RuntimeError("No best checkpoint selected.")

    torch.save(best_pack, model_out)

    print()
    print(f"Wrote best corrector: {model_out}")
    print(f"Best epoch:           {best_epoch}")
    print(f"Best score:           {best_score:.6e}")
    print(f"Wrote history:        {history_out}")


if __name__ == "__main__":
    main()
