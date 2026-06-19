from __future__ import annotations

import math

import torch

from remapgnn.models import scatter_sum_torch

# Over-relaxation factor for the converge-to-tol path (eval/inference + frozen-dual training).
# omega=1.0 is plain Sinkhorn; ~1.95 accelerates ~20-27x to the SAME fixed point, with a residual
# safeguard that backs off toward 1.0 if it ever diverges. Benchmarked across mesh families incl.
# the stiff reverse-to-CS directions.
DEFAULT_OMEGA = 1.95


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
    init: torch.Tensor | None = None,
    omega: float = 1.0,
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

    Warm start:
      * init (optional): starting mass matrix. Sinkhorn converges to the unique diagonal scaling
        of the initial matrix that satisfies both marginals; `q * s_prev` (a previous converged
        per-edge scale applied to the current q) is itself a diagonal scaling of q, so the fixed
        point is IDENTICAL to a cold start from q -- it only converges faster. Pass init to reuse
        duals across bands / steps. Exact, not an approximation.

    Over-relaxation (acceleration):
      * omega (default 1.0 = plain Sinkhorn): raise each marginal scaling to the power `omega`
        (successive over-relaxation). For omega in (1, 2) this hugely accelerates convergence
        (~20-27x at omega~1.95 across mesh families) to the SAME fixed point. Only active in
        convergence mode (tol set). A safeguard monitors the residual every `check_every` iters:
        if it rises (over-relaxation diverging), it rolls back to the last good state and halves
        (omega - 1), so an over-eager omega degrades gracefully to plain Sinkhorn instead of
        diverging. Use omega > 1 at eval/inference; leave omega = 1.0 for the gradient path.
    """
    M = torch.clamp((init if init is not None else q).float(), min=eps)

    area_src_f = area_src.float()
    area_tgt_f = area_tgt.float()

    def _step(M, w):
        # target normalization first, source normalization second.
        tgt_mass = scatter_sum_torch(M, tgt_index, n_tgt)
        tgt_scale = area_tgt_f / torch.clamp(tgt_mass, min=eps)
        M = M * (tgt_scale[tgt_index] if w == 1.0 else tgt_scale[tgt_index] ** w)

        src_mass = scatter_sum_torch(M, src_index, n_src)
        src_scale = area_src_f / torch.clamp(src_mass, min=eps)
        M = M * (src_scale[src_index] if w == 1.0 else src_scale[src_index] ** w)
        return M

    if tol is None:
        for _ in range(n_iter):
            M = _step(M, omega)
        return M

    # Convergence mode: iterate until both marginals are satisfied to `tol` (relative L2).
    cap = max_iter if max_iter is not None else max(n_iter, 50000)
    asn = torch.clamp(torch.linalg.norm(area_src_f), min=eps)
    atn = torch.clamp(torch.linalg.norm(area_tgt_f), min=eps)
    it = 0
    w = float(omega)
    M_good = M
    last_resid = float("inf")
    while it < cap:
        for _ in range(min(check_every, cap - it)):
            M = _step(M, w)
            it += 1
        src_resid = torch.linalg.norm(scatter_sum_torch(M, src_index, n_src) - area_src_f) / asn
        tgt_resid = torch.linalg.norm(scatter_sum_torch(M, tgt_index, n_tgt) - area_tgt_f) / atn
        r = float(torch.maximum(src_resid, tgt_resid))
        if r < tol:
            return M
        if w > 1.0 and (not math.isfinite(r) or r > last_resid):
            # over-relaxation diverging: roll back and back off toward plain Sinkhorn.
            M = M_good
            w = 1.0 + (w - 1.0) * 0.5
        else:
            M_good = M
            last_resid = r

    return M


def converged_balance(
    q: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    area_src: torch.Tensor,
    area_tgt: torch.Tensor,
    n_src: int,
    n_tgt: int,
    tol: float = 1.0e-6,
    max_iter: int = 20000,
    eps: float = 1.0e-30,
    warm_scale: torch.Tensor | None = None,
    return_scale: bool = False,
    omega: float = DEFAULT_OMEGA,
):
    """
    Converged Sinkhorn balance with a frozen-dual (implicit) gradient, for the TRAINING path.

    The fixed-point scaling is solved under no_grad (cheap, no autograd graph, iterate to `tol`),
    giving per-edge scale s such that M = q * s satisfies both marginals. We then return q * s with
    s held constant, so M equals the converged mass matrix (conservative AND consistent) and is
    differentiable in q with the Sinkhorn duals frozen. This lets the corrector train against a
    converged operator without unrolling thousands of grad-tracked iterations (O(n_edges) memory).

    Warm start / dual reuse:
      * warm_scale (optional): a previous converged per-edge scale s_prev. We seed Sinkhorn from
        q * s_prev, which has the same fixed point as a cold start (see sparse_sinkhorn_balance) but
        converges in far fewer iterations when q is close to the q that produced s_prev (adjacent
        bands within a rollout, or the same pair across optimizer steps).
      * return_scale: also return the final per-edge scale s (detached) so the caller can cache it
        as the next warm_scale.
    """
    qf = q.float()
    with torch.no_grad():
        init = None if warm_scale is None else qf * warm_scale
        M_conv = sparse_sinkhorn_balance(
            q=qf, src_index=src_index, tgt_index=tgt_index,
            area_src=area_src, area_tgt=area_tgt, n_src=n_src, n_tgt=n_tgt,
            tol=tol, max_iter=max_iter, eps=eps, init=init, omega=omega,
        )
        s = M_conv / torch.clamp(qf, min=eps)
    M = qf * s
    if return_scale:
        return M, s.detach()
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
