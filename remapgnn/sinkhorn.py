from __future__ import annotations

import torch

from remapgnn.models import scatter_sum_torch


def sparse_sinkhorn_balance(
    q: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    area_src: torch.Tensor,
    area_tgt: torch.Tensor,
    n_src: int,
    n_tgt: int,
    n_iter: int = 2000,
    eps: float = 1.0e-30,
    tol: float | None = None,
    max_iter: int | None = None,
    check_every: int = 100,
) -> torch.Tensor:
    """
    Balance positive sparse edge scores into conservative source-target masses.

    Edge convention:
      edge e connects source src_index[e] to target tgt_index[e]

    Constraints (BOTH satisfied at convergence):
      sum over targets i of M_ij ~= area_src[j]   (conservation: source marginal)
      sum over sources j of M_ij ~= area_tgt[i]   (consistency:  target marginal -> rows sum to 1)

    Modes:
      * tol is None  (default): run exactly `n_iter` alternating iterations. Use this for the
        gradient-tracked training path (fixed, small unroll). The loop ends on source scaling,
        so at low iteration counts conservation is tight but consistency may not be converged.
      * tol is set: iterate until BOTH marginal relative residuals are < tol (checked every
        `check_every` iterations), up to `max_iter`. Use this at eval/inference (under no_grad)
        to obtain an operator that is simultaneously conservative AND consistent. Iterating to
        convergence is what removes the smooth reverse-direction bias; both marginals converge
        together because the bipartite problem is feasible (totals match: sum area = 4*pi).
    """
    M = torch.clamp(q.float(), min=eps)

    area_src_f = area_src.float()
    area_tgt_f = area_tgt.float()

    def _step(M):
        # target normalization first, source normalization second.
        tgt_mass = scatter_sum_torch(M, tgt_index, n_tgt)
        tgt_scale = area_tgt_f / torch.clamp(tgt_mass, min=eps)
        M = M * tgt_scale[tgt_index]

        src_mass = scatter_sum_torch(M, src_index, n_src)
        src_scale = area_src_f / torch.clamp(src_mass, min=eps)
        M = M * src_scale[src_index]
        return M

    if tol is None:
        for _ in range(n_iter):
            M = _step(M)
        return M

    # Convergence mode: iterate until both marginals are satisfied to `tol` (relative L2).
    cap = max_iter if max_iter is not None else max(n_iter, 50000)
    asn = torch.clamp(torch.linalg.norm(area_src_f), min=eps)
    atn = torch.clamp(torch.linalg.norm(area_tgt_f), min=eps)
    it = 0
    while it < cap:
        for _ in range(min(check_every, cap - it)):
            M = _step(M)
            it += 1
        src_resid = torch.linalg.norm(scatter_sum_torch(M, src_index, n_src) - area_src_f) / asn
        tgt_resid = torch.linalg.norm(scatter_sum_torch(M, tgt_index, n_tgt) - area_tgt_f) / atn
        if float(torch.maximum(src_resid, tgt_resid)) < tol:
            break

    return M


def sparse_operator_weights(
    M: torch.Tensor,
    tgt_index: torch.Tensor,
    area_tgt: torch.Tensor,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    """
    Convert balanced mass matrix entries M_ij into remapping weights S_ij.

      S_ij = M_ij / area_tgt_i
    """
    return M / torch.clamp(area_tgt.float()[tgt_index], min=eps)


def operator_diagnostics(
    M: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    area_src: torch.Tensor,
    area_tgt: torch.Tensor,
    n_src: int,
    n_tgt: int,
) -> dict:
    """
    Compute source and target conservation diagnostics for sparse mass matrix.
    """
    M_f = M.float()
    area_src_f = area_src.float()
    area_tgt_f = area_tgt.float()

    src_mass = scatter_sum_torch(M_f, src_index, n_src)
    tgt_mass = scatter_sum_torch(M_f, tgt_index, n_tgt)

    src_err = src_mass - area_src_f
    tgt_err = tgt_mass - area_tgt_f

    row_sum = tgt_mass / torch.clamp(area_tgt_f, min=1.0e-30)
    row_sum_err = row_sum - 1.0

    def rel_l2(err: torch.Tensor, ref: torch.Tensor) -> float:
        return float(torch.linalg.norm(err) / torch.clamp(torch.linalg.norm(ref), min=1.0e-30))

    return {
        "row_sum_abs_max": float(row_sum_err.abs().max()),
        "row_sum_abs_mean": float(row_sum_err.abs().mean()),
        "row_sum_rel_l2": rel_l2(row_sum_err, torch.ones_like(row_sum_err)),
        "target_mass_abs_max": float(tgt_err.abs().max()),
        "target_mass_abs_mean": float(tgt_err.abs().mean()),
        "target_mass_rel_l2": rel_l2(tgt_err, area_tgt_f),
        "source_mass_abs_max": float(src_err.abs().max()),
        "source_mass_abs_mean": float(src_err.abs().mean()),
        "source_mass_rel_l2": rel_l2(src_err, area_src_f),
    }
