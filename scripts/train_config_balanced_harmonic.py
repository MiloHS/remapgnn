from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config
from remapgnn.data import compute_feature_stats, load_pair_tensors, get_feature_lists
from remapgnn.models import build_model
from remapgnn.sinkhorn import sparse_sinkhorn_balance, sparse_operator_weights


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _stable_pair_seed(pair: str) -> int:
    """Deterministic per-pair seed (independent of PYTHONHASHSEED salting)."""
    return int.from_bytes(hashlib.sha256(pair.encode("utf-8")).digest()[:4], "big")


def warn_split_leakage(train_pairs, checkpoint_pairs, test_pair) -> None:
    """Loudly flag train/val/test contamination in the configured split."""
    tp = set(train_pairs)
    cp = set(checkpoint_pairs or [])
    if test_pair in cp:
        print(
            f"WARNING: test_pair {test_pair!r} is in checkpoint_pairs — model selection "
            f"peeks at the test pair (test-set leakage). Remove it from checkpoint_pairs."
        )
    if cp and cp.issubset(tp):
        print(
            f"WARNING: every checkpoint/validation pair is also a training pair — there is "
            f"no clean held-out validation signal for model selection: {sorted(cp)}"
        )
    elif cp & tp:
        print(f"WARNING: checkpoint_pairs overlap train_pairs: {sorted(cp & tp)}")


def unique_keep_order(xs):
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def safe_pair_name(pair: str) -> str:
    return pair.replace("-", "_").replace(".", "p").replace("/", "_").replace(":", "_")


def get_sph_harm_func():
    try:
        from scipy.special import sph_harm_y

        def sph(l, m, theta, phi):
            return sph_harm_y(l, m, theta, phi)

        return sph
    except Exception:
        pass

    try:
        from scipy.special import sph_harm

        def sph(l, m, theta, phi):
            return sph_harm(m, l, phi, theta)

        return sph
    except Exception as e:
        raise RuntimeError("scipy.special.sph_harm or sph_harm_y is required.") from e


SPH = get_sph_harm_func()


def parse_degrees(x):
    if isinstance(x, str):
        return [int(v) for v in x.replace(",", " ").split()]
    return [int(v) for v in x]


def choose_m_values(l: int, modes_per_degree: int, rng: np.random.Generator) -> list[int]:
    all_m = list(range(-l, l + 1))
    if len(all_m) <= modes_per_degree:
        return all_m

    keep = {-l, 0, l}
    remaining = [m for m in all_m if m not in keep]
    n_extra = max(0, modes_per_degree - len(keep))
    if n_extra > 0:
        keep.update(rng.choice(remaining, size=n_extra, replace=False).tolist())
    return sorted(keep)


def xyz_to_angles(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    z = np.clip(xyz[:, 2] / np.maximum(r, 1.0e-30), -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    return theta, phi


def real_spherical_harmonic(l: int, m: int, xyz: np.ndarray) -> np.ndarray:
    theta, phi = xyz_to_angles(xyz)

    if m == 0:
        y = SPH(l, 0, theta, phi).real
    elif m > 0:
        y = np.sqrt(2.0) * ((-1.0) ** m) * SPH(l, m, theta, phi).real
    else:
        mp = abs(m)
        y = np.sqrt(2.0) * ((-1.0) ** mp) * SPH(l, mp, theta, phi).imag

    y = np.asarray(y, dtype=np.float64)
    norm = np.sqrt(np.mean(y * y))
    if norm > 0:
        y = y / norm
    return y.astype("float32")


def read_source_xyz_from_edges(edge_path: Path, n_src: int) -> np.ndarray:
    df = pd.read_parquet(edge_path, columns=["source_index", "src_x", "src_y", "src_z"])
    g = df.groupby("source_index", sort=False)[["src_x", "src_y", "src_z"]].first()

    xyz = np.full((n_src, 3), np.nan, dtype=np.float64)
    idx = g.index.to_numpy(dtype=np.int64)
    xyz[idx] = g.to_numpy(dtype=np.float64)

    if np.isnan(xyz).any():
        missing = np.where(np.isnan(xyz[:, 0]))[0][:10]
        raise RuntimeError(f"Missing source coordinates, first few: {missing}")

    return xyz


def build_harmonic_fields(cfg, pair: str, n_src: int, degrees, modes_per_degree: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed + _stable_pair_seed(pair) % 1000000)
    xyz = read_source_xyz_from_edges(cfg.edge_path(pair), n_src=n_src)

    fields = []
    labels = []

    for l in degrees:
        for m in choose_m_values(l, modes_per_degree, rng):
            fields.append(real_spherical_harmonic(l, m, xyz))
            labels.append((l, m))

    arr = np.stack(fields, axis=0).astype("float32")
    print(f"  harmonic cache {pair}: {arr.shape[0]} fields, degrees={degrees}")
    return torch.from_numpy(arr)


def model_outputs_to_q(out):
    if isinstance(out, dict):
        edge_logit = out.get("edge_logit", out.get("logit"))
        raw_weight = out.get("raw_weight", out.get("positive_weight"))
        q = out.get("q")
        if q is not None:
            return edge_logit, raw_weight, q

    elif isinstance(out, (tuple, list)):
        if len(out) == 3:
            edge_logit, raw_weight, q = out
            return edge_logit, raw_weight, q
        if len(out) == 2:
            edge_logit, raw_weight = out
            q = torch.sqrt(torch.sigmoid(edge_logit).clamp_min(1.0e-12)) * Fnn.softplus(raw_weight)
            return edge_logit, raw_weight, q

    raise RuntimeError(f"Unsupported model output: {type(out)}")


def scatter_fields_to_target(edge_values: torch.Tensor, tgt_index: torch.Tensor, n_tgt: int) -> torch.Tensor:
    # edge_values shape: [n_fields, n_edges]
    y = torch.zeros(
        (edge_values.shape[0], n_tgt),
        dtype=edge_values.dtype,
        device=edge_values.device,
    )
    y.index_add_(1, tgt_index, edge_values)
    return y


def harmonic_loss_from_operator(
    S_pred: torch.Tensor,
    S_true: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    edge_exists: torch.Tensor,
    harmonic_fields: torch.Tensor,
    n_tgt: int,
    max_fields_per_step: int,
    rng_state: torch.Generator | None = None,
    eps: float = 1.0e-20,
):
    n_fields = harmonic_fields.shape[0]
    if max_fields_per_step > 0 and n_fields > max_fields_per_step:
        perm = torch.randperm(n_fields, device=harmonic_fields.device, generator=rng_state)
        fields = harmonic_fields[perm[:max_fields_per_step]]
    else:
        fields = harmonic_fields

    x_edge = fields[:, src_index]  # [F, E]

    y_pred = scatter_fields_to_target(
        S_pred[None, :] * x_edge,
        tgt_index,
        n_tgt,
    )

    pos = edge_exists > 0.5
    y_true = scatter_fields_to_target(
        S_true[pos][None, :] * x_edge[:, pos],
        tgt_index[pos],
        n_tgt,
    )

    rel2 = ((y_pred - y_true) ** 2).sum(dim=1) / torch.clamp((y_true ** 2).sum(dim=1), min=eps)
    rel = torch.sqrt(torch.clamp(rel2, min=0.0))

    return rel2.mean(), rel.mean().detach()



def get_batch_true_weight(batch):
    """
    Return Tempest/SCRIP true edge weights from a load_pair_tensors batch.

    Different versions of remapgnn/data.py have used different names.
    """
    candidates = [
        "weight",
        "S_true",
        "s_true",
        "target_weight",
        "true_weight",
        "remap_weight",
        "edge_weight",
    ]

    for k in candidates:
        if k in batch:
            return batch[k].float()

    raise KeyError(
        "Could not find true Tempest edge weights in batch. "
        f"Available keys: {sorted(batch.keys())}"
    )

def pair_loss_with_harmonics(
    model,
    batch,
    harmonic_fields,
    n_sinkhorn_iter: int,
    lambda_pos_s: float,
    lambda_neg_s: float,
    lambda_bce: float,
    lambda_field: float,
    max_fields_per_step: int,
):
    src_index = batch["src_index"]
    tgt_index = batch["tgt_index"]
    edge_exists = batch["edge_exists"].float()
    S_true = get_batch_true_weight(batch)

    area_src = batch["area_src"].float()
    area_tgt = batch["area_tgt"].float()
    n_src = int(batch["n_src"].item() if hasattr(batch["n_src"], "item") else batch["n_src"])
    n_tgt = int(batch["n_tgt"].item() if hasattr(batch["n_tgt"], "item") else batch["n_tgt"])

    out = model(
        batch["src_node_attr"],
        batch["tgt_node_attr"],
        batch["edge_attr"],
        src_index,
        tgt_index,
        n_src,
        n_tgt,
    )
    edge_logit, raw_weight, q = model_outputs_to_q(out)
    q = q.float()

    M = sparse_sinkhorn_balance(
        q=q,
        src_index=src_index,
        tgt_index=tgt_index,
        area_src=area_src,
        area_tgt=area_tgt,
        n_src=n_src,
        n_tgt=n_tgt,
        n_iter=n_sinkhorn_iter,
    )

    S_pred = sparse_operator_weights(
        M=M,
        tgt_index=tgt_index,
        area_tgt=area_tgt,
    )

    pos = edge_exists > 0.5
    neg = ~pos
    scale = torch.clamp(torch.mean(S_true[pos] ** 2), min=1.0e-20)

    loss_pos_s = torch.mean((S_pred[pos] - S_true[pos]) ** 2) / scale
    loss_neg_s = torch.mean(S_pred[neg] ** 2) / scale if neg.any() else torch.zeros_like(loss_pos_s)

    n_pos = torch.clamp(pos.float().sum(), min=1.0)
    n_neg = torch.clamp(neg.float().sum(), min=1.0)
    pos_weight = n_neg / n_pos

    if edge_logit is not None:
        loss_bce = Fnn.binary_cross_entropy_with_logits(
            edge_logit.float(),
            edge_exists,
            pos_weight=pos_weight,
        )
    else:
        loss_bce = torch.zeros_like(loss_pos_s)

    harmonic_fields = harmonic_fields.to(S_pred.device, dtype=S_pred.dtype)
    loss_field, field_rel_l2 = harmonic_loss_from_operator(
        S_pred=S_pred,
        S_true=S_true,
        src_index=src_index,
        tgt_index=tgt_index,
        edge_exists=edge_exists,
        harmonic_fields=harmonic_fields,
        n_tgt=n_tgt,
        max_fields_per_step=max_fields_per_step,
    )

    total = (
        lambda_pos_s * loss_pos_s
        + lambda_neg_s * loss_neg_s
        + lambda_bce * loss_bce
        + lambda_field * loss_field
    )

    with torch.no_grad():
        rel_l2_pos = torch.sqrt(
            torch.sum((S_pred[pos] - S_true[pos]) ** 2)
            / torch.clamp(torch.sum(S_true[pos] ** 2), min=1.0e-20)
        )

        src_mass = torch.zeros(n_src, device=S_pred.device, dtype=S_pred.dtype)
        src_mass.index_add_(0, src_index, M)
        source_mass_rel_l2 = torch.linalg.norm(src_mass - area_src) / torch.clamp(
            torch.linalg.norm(area_src),
            min=1.0e-20,
        )

        tgt_mass = torch.zeros(n_tgt, device=S_pred.device, dtype=S_pred.dtype)
        tgt_mass.index_add_(0, tgt_index, M)
        row_sum_rel_l2 = torch.linalg.norm(tgt_mass - area_tgt) / torch.clamp(
            torch.linalg.norm(area_tgt),
            min=1.0e-20,
        )

        pred_edge = edge_logit.float() > 0 if edge_logit is not None else S_pred > 0
        tp = (pred_edge & pos).float().sum()
        fp = (pred_edge & neg).float().sum()
        fn = ((~pred_edge) & pos).float().sum()
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)

    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_pos_s": float(loss_pos_s.detach().cpu()),
        "loss_neg_s": float(loss_neg_s.detach().cpu()),
        "loss_bce": float(loss_bce.detach().cpu()),
        "loss_field": float(loss_field.detach().cpu()),
        "harmonic_rel_l2": float(field_rel_l2.detach().cpu()),
        "rel_l2_positive_edges": float(rel_l2_pos.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "source_mass_rel_l2": float(source_mass_rel_l2.detach().cpu()),
        "row_sum_rel_l2": float(row_sum_rel_l2.detach().cpu()),
    }

    return total, metrics


def eval_one_pair(model, cfg, pair, stats, harmonic_cache, device, n_sinkhorn_iter, lambdas, harmonic_cfg):
    model.eval()
    batch = load_pair_tensors(cfg, pair, stats, device=device)

    with torch.no_grad():
        _, metrics = pair_loss_with_harmonics(
            model=model,
            batch=batch,
            harmonic_fields=harmonic_cache[pair],
            n_sinkhorn_iter=n_sinkhorn_iter,
            lambda_pos_s=lambdas["lambda_pos_s"],
            lambda_neg_s=lambdas["lambda_neg_s"],
            lambda_bce=lambdas["lambda_bce"],
            lambda_field=harmonic_cfg["lambda_field"],
            max_fields_per_step=0,  # all harmonic fields for checkpoint/eval
        )

    del batch
    return metrics


def eval_pair_set(model, cfg, pairs, stats, harmonic_cache, device, n_sinkhorn_iter, lambdas, harmonic_cfg, prefix):
    rows = {}
    rels, rowsums, fields, src_cons = [], [], [], []

    for pair in pairs:
        metrics = eval_one_pair(
            model=model,
            cfg=cfg,
            pair=pair,
            stats=stats,
            harmonic_cache=harmonic_cache,
            device=device,
            n_sinkhorn_iter=n_sinkhorn_iter,
            lambdas=lambdas,
            harmonic_cfg=harmonic_cfg,
        )

        tag = safe_pair_name(pair)
        for k, v in metrics.items():
            rows[f"{prefix}_{tag}_{k}"] = v

        rels.append(metrics["rel_l2_positive_edges"])
        rowsums.append(metrics["row_sum_rel_l2"])
        fields.append(metrics["harmonic_rel_l2"])
        src_cons.append(metrics["source_mass_rel_l2"])

    rows[f"{prefix}_mean_rel_l2_positive_edges"] = float(np.mean(rels))
    rows[f"{prefix}_max_rel_l2_positive_edges"] = float(np.max(rels))
    rows[f"{prefix}_mean_row_sum_rel_l2"] = float(np.mean(rowsums))
    rows[f"{prefix}_max_row_sum_rel_l2"] = float(np.max(rowsums))
    rows[f"{prefix}_mean_harmonic_rel_l2"] = float(np.mean(fields))
    rows[f"{prefix}_max_harmonic_rel_l2"] = float(np.max(fields))
    rows[f"{prefix}_mean_source_mass_rel_l2"] = float(np.mean(src_cons))
    return rows


def make_pack(model, cfg, stats, edge_features, src_features, tgt_features, hidden, decoder_chunk_size,
              train_pairs, checkpoint_pairs, test_pair, n_train_iter, n_eval_iter, checkpoint_score_formula, harmonic_cfg):
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
        "harmonic_loss": harmonic_cfg,
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

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    hidden = int(tr.get("hidden", 128))
    epochs = int(args.epochs or tr.get("epochs", 200))
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

    harmonic_raw = tr.get("harmonic_loss", {})
    harmonic_cfg = {
        "enabled": bool(harmonic_raw.get("enabled", True)),
        "degrees": parse_degrees(harmonic_raw.get("degrees", [0, 1, 2, 4, 8, 12, 16])),
        "modes_per_degree": int(harmonic_raw.get("modes_per_degree", 5)),
        "lambda_field": float(harmonic_raw.get("lambda_field", 0.02)),
        "max_fields_per_step": int(harmonic_raw.get("max_fields_per_step", 8)),
        "checkpoint_field_weight": float(harmonic_raw.get("checkpoint_field_weight", 0.10)),
    }

    train_pairs = list(tr.get("train_pairs", cfg.pairs))
    val_pair = tr.get("val_pair", train_pairs[0])
    test_pair = tr.get("test_pair", cfg.pairs[0])
    checkpoint_pairs = list(tr.get("checkpoint_pairs", tr.get("validation_pairs", [val_pair])))

    warn_split_leakage(train_pairs, checkpoint_pairs, test_pair)

    score_cfg = tr.get("checkpoint_score", {})
    row_weight = float(score_cfg.get("row_weight", 0.05))
    field_score_weight = harmonic_cfg["checkpoint_field_weight"]

    if args.smoke:
        print()
        print("SMOKE MODE: one train pair, one epoch, low Sinkhorn iterations.")
        train_pairs = train_pairs[:1]
        checkpoint_pairs = checkpoint_pairs[:1]
        epochs = 1
        n_train_iter = 3
        n_eval_iter = 3
        stat_sample_per_pair = min(stat_sample_per_pair, 10000)
        harmonic_cfg["max_fields_per_step"] = min(harmonic_cfg["max_fields_per_step"], 4)

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
    print()
    print("Harmonic loss:")
    print(f"  degrees:             {harmonic_cfg['degrees']}")
    print(f"  modes_per_degree:    {harmonic_cfg['modes_per_degree']}")
    print(f"  lambda_field:        {harmonic_cfg['lambda_field']}")
    print(f"  max_fields_per_step: {harmonic_cfg['max_fields_per_step']}")
    print(f"  ckpt_field_weight:   {harmonic_cfg['checkpoint_field_weight']}")

    # Normalization stats are fit on TRAINING pairs only (no checkpoint/test/eval leakage).
    stat_pairs = unique_keep_order(train_pairs)
    stats = compute_feature_stats(cfg, stat_pairs, sample_per_pair=stat_sample_per_pair, seed=seed)

    # The harmonic cache must still cover every pair we ever evaluate (train + checkpoint + test).
    cache_pairs = unique_keep_order(train_pairs + checkpoint_pairs + [test_pair] + list(cfg.pairs))

    print()
    print("Building harmonic field cache...")
    harmonic_cache = {}
    for pair in cache_pairs:
        tmp_batch = load_pair_tensors(cfg, pair, stats, device=torch.device("cpu"))
        n_src = int(tmp_batch["n_src"].item() if hasattr(tmp_batch["n_src"], "item") else tmp_batch["n_src"])
        del tmp_batch
        harmonic_cache[pair] = build_harmonic_fields(
            cfg=cfg,
            pair=pair,
            n_src=n_src,
            degrees=harmonic_cfg["degrees"],
            modes_per_degree=harmonic_cfg["modes_per_degree"],
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
        model_out = cfg.model_path.parent / f"smoke_{cfg.model_tag}_harmonic.pt"
        history_out = cfg.history_path.parent / f"smoke_{cfg.model_tag}_harmonic_history.csv"

    checkpoint_score_formula = (
        "mean_checkpoint_rel_l2_positive_edges"
        f" + {row_weight:g} * mean_checkpoint_row_sum_rel_l2"
        f" + {field_score_weight:g} * mean_checkpoint_harmonic_rel_l2"
    )

    print()
    print("Training with balanced checkpointing + harmonic field loss...")
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
            loss, train_metrics = pair_loss_with_harmonics(
                model=model,
                batch=batch,
                harmonic_fields=harmonic_cache[pair],
                n_sinkhorn_iter=n_train_iter,
                lambda_pos_s=lambdas["lambda_pos_s"],
                lambda_neg_s=lambdas["lambda_neg_s"],
                lambda_bce=lambdas["lambda_bce"],
                lambda_field=harmonic_cfg["lambda_field"],
                max_fields_per_step=harmonic_cfg["max_fields_per_step"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            last_train_metrics = train_metrics
            del batch

        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            ckpt_rows = eval_pair_set(
                model=model,
                cfg=cfg,
                pairs=checkpoint_pairs,
                stats=stats,
                harmonic_cache=harmonic_cache,
                device=device,
                n_sinkhorn_iter=n_eval_iter,
                lambdas=lambdas,
                harmonic_cfg=harmonic_cfg,
                prefix="ckpt",
            )

            test_metrics = eval_one_pair(
                model=model,
                cfg=cfg,
                pair=test_pair,
                stats=stats,
                harmonic_cache=harmonic_cache,
                device=device,
                n_sinkhorn_iter=n_eval_iter,
                lambdas=lambdas,
                harmonic_cfg=harmonic_cfg,
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
                + field_score_weight * row["ckpt_mean_harmonic_rel_l2"]
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
                    harmonic_cfg=harmonic_cfg,
                )

            history.append(row)
            pd.DataFrame(history).to_csv(history_out, index=False)

            print(
                f"epoch {epoch:04d} "
                f"score={row['checkpoint_score']:.6e} "
                f"ckpt_edge={row['ckpt_mean_rel_l2_positive_edges']:.4e} "
                f"ckpt_harm={row['ckpt_mean_harmonic_rel_l2']:.4e} "
                f"ckpt_row={row['ckpt_mean_row_sum_rel_l2']:.2e} "
                f"test_edge={row['test_rel_l2_positive_edges']:.4e} "
                f"test_harm={row['test_harmonic_rel_l2']:.4e} "
                f"test_row={row['test_row_sum_rel_l2']:.2e}"
                + (" *BEST*" if is_best else "")
            )

    if best_pack is None:
        raise RuntimeError("No checkpoint selected.")

    torch.save(best_pack, model_out)

    print()
    print(f"Wrote best model: {model_out}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best score:       {best_score:.6e}")
    print(f"Wrote history:    {history_out}")


if __name__ == "__main__":
    main()
