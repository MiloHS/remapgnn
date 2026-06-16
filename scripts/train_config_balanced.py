from __future__ import annotations

from pathlib import Path
import argparse
import time
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config
from remapgnn.data import compute_feature_stats, load_pair_tensors, get_feature_lists
from remapgnn.models import build_model
from remapgnn.losses import pair_loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_keep_order(xs):
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def safe_pair_name(pair: str) -> str:
    return (
        pair.replace("-", "_")
            .replace(".", "p")
            .replace("/", "_")
            .replace(":", "_")
    )


def eval_one_pair(model, cfg, pair, stats, device, n_sinkhorn_iter, lambdas):
    model.eval()
    batch = load_pair_tensors(cfg, pair, stats, device=device)
    with torch.no_grad():
        _, metrics = pair_loss(
            model=model,
            batch=batch,
            n_sinkhorn_iter=n_sinkhorn_iter,
            lambda_pos_s=lambdas["lambda_pos_s"],
            lambda_neg_s=lambdas["lambda_neg_s"],
            lambda_bce=lambdas["lambda_bce"],
        )
    del batch

    out = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = float(v)
    return out


def eval_pair_set(model, cfg, pairs, stats, device, n_sinkhorn_iter, lambdas, prefix):
    rows = {}
    rels = []
    rowsums = []
    src_cons = []

    for pair in pairs:
        metrics = eval_one_pair(
            model=model,
            cfg=cfg,
            pair=pair,
            stats=stats,
            device=device,
            n_sinkhorn_iter=n_sinkhorn_iter,
            lambdas=lambdas,
        )

        tag = safe_pair_name(pair)
        for k, v in metrics.items():
            rows[f"{prefix}_{tag}_{k}"] = v

        rels.append(metrics["rel_l2_positive_edges"])
        rowsums.append(metrics.get("row_sum_rel_l2", 0.0))
        src_cons.append(metrics.get("source_mass_rel_l2", 0.0))

    rows[f"{prefix}_mean_rel_l2_positive_edges"] = float(np.mean(rels))
    rows[f"{prefix}_max_rel_l2_positive_edges"] = float(np.max(rels))
    rows[f"{prefix}_mean_row_sum_rel_l2"] = float(np.mean(rowsums))
    rows[f"{prefix}_max_row_sum_rel_l2"] = float(np.max(rowsums))
    rows[f"{prefix}_mean_source_mass_rel_l2"] = float(np.mean(src_cons))

    return rows


def make_pack(
    model,
    cfg,
    stats,
    edge_features,
    src_features,
    tgt_features,
    hidden,
    decoder_chunk_size,
    train_pairs,
    checkpoint_pairs,
    test_pair,
    n_train_iter,
    n_eval_iter,
    checkpoint_score_formula,
):
    return {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "edge_features": edge_features,
        "src_node_features": src_features,
        "tgt_node_features": tgt_features,
        "architecture": cfg.architecture,
        "run_name": cfg.run_name,
        "model_tag": cfg.model_tag,
        "graph_suffix": cfg.graph_suffix,
        "k": cfg.K,
        "hidden": hidden,
        "stats": stats,
        "train_pairs": list(train_pairs),
        "checkpoint_pairs": list(checkpoint_pairs),
        "val_pairs": list(checkpoint_pairs),
        "val_pair": checkpoint_pairs[0] if checkpoint_pairs else None,
        "test_pair": test_pair,
        "score_formula": "sqrt(sigmoid(edge_logit)) * softplus(raw_weight)",
        "checkpoint_score_formula": checkpoint_score_formula,
        "decoder_chunk_size": decoder_chunk_size,
        "sinkhorn_iters_train": n_train_iter,
        "sinkhorn_iters_eval": n_eval_iter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg.raw.get("training", {})

    seed = int(tr.get("seed", 123))
    set_seed(seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = int(tr.get("hidden", 128))
    epochs = int(args.epochs or tr.get("epochs", 80))
    lr = float(tr.get("lr", 2.0e-4))
    weight_decay = float(tr.get("weight_decay", 1.0e-5))
    n_train_iter = int(tr.get("sinkhorn_iters_train", 30))
    n_eval_iter = int(tr.get("sinkhorn_iters_eval", 300))
    decoder_chunk_size = int(tr.get("decoder_chunk_size", 10000))
    stat_sample_per_pair = int(tr.get("stat_sample_per_pair", 80000))

    lambdas = {
        "lambda_pos_s": float(tr.get("lambda_pos_s", 1.0)),
        "lambda_neg_s": float(tr.get("lambda_neg_s", 0.05)),
        "lambda_bce": float(tr.get("lambda_bce", 0.05)),
    }

    train_pairs = list(tr.get("train_pairs", cfg.pairs))
    val_pair = tr.get("val_pair", train_pairs[0])
    test_pair = tr.get("test_pair", cfg.pairs[0])

    checkpoint_pairs = list(
        tr.get(
            "checkpoint_pairs",
            tr.get("validation_pairs", [val_pair]),
        )
    )

    score_cfg = tr.get("checkpoint_score", {})
    row_weight = float(score_cfg.get("row_weight", 0.05))

    if args.smoke:
        print()
        print("SMOKE MODE: one train pair, one epoch, low Sinkhorn iterations.")
        train_pairs = train_pairs[:1]
        checkpoint_pairs = checkpoint_pairs[:1]
        epochs = 1
        n_train_iter = 3
        n_eval_iter = 3
        stat_sample_per_pair = min(stat_sample_per_pair, 10000)

    edge_features, src_features, tgt_features = get_feature_lists(cfg)

    print(f"Config:       {args.config}")
    print(f"Run name:     {cfg.run_name}")
    print(f"Model tag:    {cfg.model_tag}")
    print(f"Architecture: {cfg.architecture}")
    print(f"Graph suffix: {cfg.graph_suffix}")
    print(f"Device:       {device}")
    print()
    print("Features:")
    print(f"  edge: {edge_features}")
    print(f"  src:  {src_features}")
    print(f"  tgt:  {tgt_features}")

    stat_pairs = unique_keep_order(train_pairs + checkpoint_pairs + [test_pair] + list(cfg.pairs))
    stats = compute_feature_stats(
        cfg,
        stat_pairs,
        sample_per_pair=stat_sample_per_pair,
        seed=seed,
    )

    model = build_model(
        architecture=cfg.architecture,
        src_dim=len(src_features),
        tgt_dim=len(tgt_features),
        edge_dim=len(edge_features),
        hidden=hidden,
        decoder_chunk_size=decoder_chunk_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_out = cfg.model_path
    history_out = cfg.history_path
    if args.smoke:
        model_out = cfg.model_path.parent / f"smoke_{cfg.model_tag}_balanced.pt"
        history_out = cfg.history_path.parent / f"smoke_{cfg.model_tag}_balanced_history.csv"

    checkpoint_score_formula = (
        "mean_checkpoint_rel_l2_positive_edges"
        f" + {row_weight:g} * mean_checkpoint_row_sum_rel_l2"
    )

    print()
    print("Training config-driven bipartite GNN + sparse Sinkhorn with balanced checkpointing...")
    print(f"Train pairs:      {train_pairs}")
    print(f"Checkpoint pairs: {checkpoint_pairs}")
    print(f"Test pair:        {test_pair}")
    print(f"Epochs:           {epochs}")
    print(f"Train iters:      {n_train_iter}")
    print(f"Eval iters:       {n_eval_iter}")
    print(f"Checkpoint score: {checkpoint_score_formula}")
    print()

    history = []
    t0 = time.time()

    best_score = float("inf")
    best_epoch = None
    best_pack = None

    for epoch in range(1, epochs + 1):
        epoch_train_pairs = list(train_pairs)
        np.random.shuffle(epoch_train_pairs)

        last_train_metrics = {}

        for pair in epoch_train_pairs:
            model.train()
            batch = load_pair_tensors(cfg, pair, stats, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = pair_loss(
                model=model,
                batch=batch,
                n_sinkhorn_iter=n_train_iter,
                **lambdas,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            last_train_metrics = {
                k: float(v.item() if hasattr(v, "item") else v)
                for k, v in train_metrics.items()
            }

            del batch

        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            ckpt_rows = eval_pair_set(
                model=model,
                cfg=cfg,
                pairs=checkpoint_pairs,
                stats=stats,
                device=device,
                n_sinkhorn_iter=n_eval_iter,
                lambdas=lambdas,
                prefix="ckpt",
            )

            test_metrics = eval_one_pair(
                model=model,
                cfg=cfg,
                pair=test_pair,
                stats=stats,
                device=device,
                n_sinkhorn_iter=n_eval_iter,
                lambdas=lambdas,
            )

            row = {
                "epoch": epoch,
                "elapsed_sec": time.time() - t0,
                **{f"train_last_{k}": v for k, v in last_train_metrics.items()},
                **ckpt_rows,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            checkpoint_score = (
                row["ckpt_mean_rel_l2_positive_edges"]
                + row_weight * row["ckpt_mean_row_sum_rel_l2"]
            )
            row["checkpoint_score"] = float(checkpoint_score)

            is_best = checkpoint_score < best_score
            if is_best:
                best_score = checkpoint_score
                best_epoch = epoch
                best_pack = make_pack(
                    model=model,
                    cfg=cfg,
                    stats=stats,
                    edge_features=edge_features,
                    src_features=src_features,
                    tgt_features=tgt_features,
                    hidden=hidden,
                    decoder_chunk_size=decoder_chunk_size,
                    train_pairs=train_pairs,
                    checkpoint_pairs=checkpoint_pairs,
                    test_pair=test_pair,
                    n_train_iter=n_train_iter,
                    n_eval_iter=n_eval_iter,
                    checkpoint_score_formula=checkpoint_score_formula,
                )

            history.append(row)
            pd.DataFrame(history).to_csv(history_out, index=False)

            print(
                f"epoch {epoch:04d} "
                f"ckpt_score={row['checkpoint_score']:.6e} "
                f"ckpt_meanRelL2pos={row['ckpt_mean_rel_l2_positive_edges']:.4e} "
                f"ckpt_maxRelL2pos={row['ckpt_max_rel_l2_positive_edges']:.4e} "
                f"ckpt_row={row['ckpt_mean_row_sum_rel_l2']:.2e} "
                f"test_relL2pos={row['test_rel_l2_positive_edges']:.4e} "
                f"test_srcCons={row['test_source_mass_rel_l2']:.2e} "
                f"test_row={row['test_row_sum_rel_l2']:.2e}"
                + (" *BEST*" if is_best else "")
            )

    if best_pack is None:
        raise RuntimeError("No checkpoint was selected.")

    torch.save(best_pack, model_out)

    print()
    print(f"Wrote best model: {model_out}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best score:       {best_score:.6e}")
    print(f"Wrote history:    {history_out}")


if __name__ == "__main__":
    main()
