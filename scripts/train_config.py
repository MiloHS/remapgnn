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


def train_eval_pair(model, cfg, pair, stats, device, n_sinkhorn_iter, lambdas):
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
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="One quick low-iteration training check.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg.raw.get("training", {})

    seed = int(tr.get("seed", 123))
    set_seed(seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Config:       {cfg.path}")
    print(f"Run name:     {cfg.run_name}")
    print(f"Model tag:    {cfg.model_tag}")
    print(f"Architecture: {cfg.architecture}")
    print(f"Graph suffix: {cfg.graph_suffix}")
    print(f"Device:       {device}")

    train_pairs = list(tr["train_pairs"])
    val_pair = tr["val_pair"]
    test_pair = tr["test_pair"]

    if args.smoke:
        print("\nSMOKE MODE: one train pair, one epoch, low Sinkhorn iterations.")
        train_pairs = train_pairs[:1]
        epochs = 1
        sinkhorn_iters_train = 3
        sinkhorn_iters_eval = 3
        stat_pairs = train_pairs + [val_pair, test_pair]
    else:
        epochs = int(args.epochs if args.epochs is not None else tr.get("epochs", 80))
        sinkhorn_iters_train = int(tr.get("sinkhorn_iters_train", 30))
        sinkhorn_iters_eval = int(tr.get("sinkhorn_iters_eval", 300))
        # Match old scripts: TRAIN_PAIRS + [VAL_PAIR, TEST_PAIR], including duplicates.
        stat_pairs = train_pairs + [val_pair, test_pair]

    hidden = int(tr.get("hidden", 128))
    lr = float(tr.get("lr", 2.0e-4))
    weight_decay = float(tr.get("weight_decay", 1.0e-5))
    decoder_chunk_size = int(tr.get("decoder_chunk_size", 10000))
    stat_sample_per_pair = int(tr.get("stat_sample_per_pair", 80000))

    lambdas = {
        "lambda_pos_s": float(tr.get("lambda_pos_s", 1.0)),
        "lambda_neg_s": float(tr.get("lambda_neg_s", 0.05)),
        "lambda_bce": float(tr.get("lambda_bce", 0.05)),
    }

    edge_features, src_features, tgt_features = get_feature_lists(cfg)

    print()
    print("Features:")
    print("  edge:", edge_features)
    print("  src: ", src_features)
    print("  tgt: ", tgt_features)

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

    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        output_model_path = cfg.models_dir / f"smoke_{cfg.model_tag}.pt"
        output_history_path = cfg.models_dir / f"smoke_{cfg.model_tag}_history.csv"
    else:
        output_model_path = cfg.model_path
        output_history_path = cfg.history_path

    history = []
    best_val_rel_l2 = float("inf")
    best_epoch = None
    best_pack = None

    t0 = time.time()

    print()
    print("Training config-driven bipartite GNN + sparse Sinkhorn...")
    print(f"Train pairs: {train_pairs}")
    print(f"Val pair:    {val_pair}")
    print(f"Test pair:   {test_pair}")
    print(f"Epochs:      {epochs}")
    print(f"Train iters: {sinkhorn_iters_train}")
    print(f"Eval iters:  {sinkhorn_iters_eval}")
    print()

    for epoch in range(1, epochs + 1):
        np.random.shuffle(train_pairs)

        for pair in train_pairs:
            model.train()
            batch = load_pair_tensors(cfg, pair, stats, device=device)

            optimizer.zero_grad(set_to_none=True)

            loss, train_metrics = pair_loss(
                model=model,
                batch=batch,
                n_sinkhorn_iter=sinkhorn_iters_train,
                lambda_pos_s=lambdas["lambda_pos_s"],
                lambda_neg_s=lambdas["lambda_neg_s"],
                lambda_bce=lambdas["lambda_bce"],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            del batch

        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            val_metrics = train_eval_pair(
                model, cfg, val_pair, stats, device, sinkhorn_iters_eval, lambdas
            )
            test_metrics = train_eval_pair(
                model, cfg, test_pair, stats, device, sinkhorn_iters_eval, lambdas
            )

            row = {
                "epoch": epoch,
                "elapsed_sec": time.time() - t0,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            is_best = row["val_rel_l2_positive_edges"] < best_val_rel_l2
            if is_best:
                best_val_rel_l2 = row["val_rel_l2_positive_edges"]
                best_epoch = epoch
                best_pack = {
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "architecture": cfg.architecture,
                    "run_name": cfg.run_name,
                    "model_tag": cfg.model_tag,
                    "graph_suffix": cfg.graph_suffix,
                    "edge_features": edge_features,
                    "src_node_features": src_features,
                    "tgt_node_features": tgt_features,
                    "k": cfg.K,
                    "hidden": hidden,
                    "stats": stats,
                    "train_pairs": train_pairs,
                    "val_pair": val_pair,
                    "test_pair": test_pair,
                    "score_formula": "sqrt(sigmoid(edge_logit)) * softplus(raw_weight)",
                    "decoder_chunk_size": decoder_chunk_size,
                    "sinkhorn_iters_train": sinkhorn_iters_train,
                    "sinkhorn_iters_eval": sinkhorn_iters_eval,
                    "config": cfg.raw,
                    "best_epoch": best_epoch,
                    "best_val_rel_l2": best_val_rel_l2,
                }

            history.append(row)

            star = " *BEST*" if is_best else ""
            print(
                f"epoch {epoch:04d} "
                f"val_relL2pos={row['val_rel_l2_positive_edges']:.4e} "
                f"test_relL2pos={row['test_rel_l2_positive_edges']:.4e} "
                f"val_prec={row['val_precision_at_0p5']:.3f} "
                f"val_rec={row['val_recall_at_0p5']:.3f} "
                f"test_prec={row['test_precision_at_0p5']:.3f} "
                f"test_rec={row['test_recall_at_0p5']:.3f} "
                f"test_srcCons={row['test_source_mass_rel_l2']:.2e} "
                f"test_row={row['test_row_sum_rel_l2']:.2e}"
                f"{star}"
            )

    if best_pack is None:
        raise RuntimeError("No best checkpoint was recorded.")

    torch.save(best_pack, output_model_path)
    pd.DataFrame(history).to_csv(output_history_path, index=False)

    print()
    print(f"Wrote best model: {output_model_path}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best val relL2:   {best_val_rel_l2:.6e}")
    print(f"Wrote history:    {output_history_path}")


if __name__ == "__main__":
    main()
