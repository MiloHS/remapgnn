from __future__ import annotations

import torch
import torch.nn.functional as F

from remapgnn.models import scatter_sum_torch
from remapgnn.sinkhorn import sparse_sinkhorn_balance, sparse_operator_weights


def pair_loss(
    model,
    batch: dict,
    n_sinkhorn_iter: int,
    lambda_pos_s: float = 1.0,
    lambda_neg_s: float = 0.05,
    lambda_bce: float = 0.05,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, dict]:
    """
    v8/v10 training loss, moved into the framework.

    Loss terms:
      1. positive-edge regression against Tempest weights
      2. negative-edge penalty
      3. binary edge-existence classification

    Sinkhorn turns model scores q_ij into conservative mass M_ij.
    """
    src_index = batch["src_index"]
    tgt_index = batch["tgt_index"]
    edge_exists = batch["edge_exists"]
    S_true = batch["S_true"]
    area_src = batch["area_src"]
    area_tgt = batch["area_tgt"]
    n_src = batch["n_src"]
    n_tgt = batch["n_tgt"]

    pos_mask = edge_exists > 0.5
    neg_mask = ~pos_mask

    use_amp = batch["edge_attr"].is_cuda

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        logit, positive_weight, q = model(
            batch["src_node_attr"],
            batch["tgt_node_attr"],
            batch["edge_attr"],
            batch["src_index"],
            batch["tgt_index"],
            batch["n_src"],
            batch["n_tgt"],
        )

    logit = logit.float()
    positive_weight = positive_weight.float()
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
        eps=eps,
    )

    S_pred = sparse_operator_weights(M, tgt_index, area_tgt, eps=eps)

    s_pos_scale = torch.mean(S_true[pos_mask] ** 2).detach() + eps

    loss_pos_s = torch.mean((S_pred[pos_mask] - S_true[pos_mask]) ** 2) / s_pos_scale

    if neg_mask.any():
        loss_neg_s = torch.mean(S_pred[neg_mask] ** 2) / s_pos_scale
    else:
        loss_neg_s = torch.zeros((), dtype=S_pred.dtype, device=S_pred.device)

    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        dtype=torch.float32,
        device=edge_exists.device,
    )

    loss_bce = F.binary_cross_entropy_with_logits(
        logit,
        edge_exists,
        pos_weight=pos_weight,
    )

    loss = (
        lambda_pos_s * loss_pos_s
        + lambda_neg_s * loss_neg_s
        + lambda_bce * loss_bce
    )

    with torch.no_grad():
        rel_l2_pos = (
            torch.linalg.norm(S_pred[pos_mask] - S_true[pos_mask])
            / torch.clamp(torch.linalg.norm(S_true[pos_mask]), min=eps)
        ).item()

        prob = torch.sigmoid(logit)
        pred_edge = prob >= 0.5

        tp = torch.sum(pred_edge & pos_mask).item()
        fp = torch.sum(pred_edge & neg_mask).item()
        fn = torch.sum((~pred_edge) & pos_mask).item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        row_sum = scatter_sum_torch(S_pred, tgt_index, n_tgt)
        row_err = row_sum - 1.0

        src_mass = scatter_sum_torch(M, src_index, n_src)
        src_err = src_mass - area_src

        source_mass_rel_l2 = (
            torch.linalg.norm(src_err)
            / torch.clamp(torch.linalg.norm(area_src), min=eps)
        ).item()

        row_sum_rel_l2 = (
            torch.linalg.norm(row_err)
            / torch.clamp(torch.linalg.norm(torch.ones_like(row_sum)), min=eps)
        ).item()

    metrics = {
        "loss": loss.item(),
        "loss_pos_s": loss_pos_s.item(),
        "loss_neg_s": loss_neg_s.item(),
        "loss_bce": loss_bce.item(),
        "rel_l2_positive_edges": rel_l2_pos,
        "precision_at_0p5": precision,
        "recall_at_0p5": recall,
        "source_mass_rel_l2": source_mass_rel_l2,
        "row_sum_rel_l2": row_sum_rel_l2,
    }

    return loss, metrics
