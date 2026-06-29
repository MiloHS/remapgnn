"""Differentiable doubly-constrained projection for SIGNED remap weights.

Projects raw edge masses q onto {M : target-marginal sums = area_tgt (consistency),
source-marginal sums = area_src (conservation)} in the Euclidean sense:

    M* = q + Aᵀλ,   (A Aᵀ + εI) λ = b − A q

where A stacks the target- and source-incidence of the bipartite edge set. Unlike Sinkhorn this admits
NEGATIVE weights (needed for >1st order) and is a single linear solve. Matrix-free (scatter/gather only),
GPU-able, and differentiable (the solve is linear in q; we unroll CG so autograd flows).

OPTIONAL ℓ=1 (linear / moment) reproduction: pass moment_coef [E, D] with
    moment_coef[e, d] = src_centroid[src(e), d] − tgt_centroid[tgt(e), d]      (Cartesian d ∈ {x,y,z})
to additionally enforce, per target cell t,   Σ_{e: tgt(e)=t} M_e · moment_coef[e, d] = 0   for each d.
Together with consistency this is exact linear reproduction  Σ_j S_tj c_src[j] = c_tgt[t]  (S = M/area_tgt),
i.e. the operator reproduces degree-1 fields (the ℓ=1 spherical-harmonic band) exactly, by construction.
The block is appended to A (target-indexed, weighted incidence); A Aᵀ stays SYMMETRIC so the implicit
backward is unchanged in form. moment_coef is internally normalized to unit RMS for CG conditioning -- the
constraint RHS is 0, so this rescaling leaves the constraint set (and hence the projected M) unchanged.
"""
from __future__ import annotations
import torch
from remapgnn.models import scatter_sum_torch


def _prep_moment(moment_coef, tgt_index, n_tgt, dtype=torch.float32):
    """Prepare the ℓ=1 moment coefficient for the projection:
    (1) ZERO rows on degenerate target cells (degree < D+1, i.e. fewer incident edges than the
        1 consistency + D moment constraints they'd carry) -- there the ℓ=1 system is locally
        over-determined, so we drop it and fall back to no moment correction on those cells rather
        than letting a rank-deficient block blow up the (globally-coupled) solve;
    (2) normalize to unit RMS so the A Aᵀ moment-block diagonal (~ Σ coef²) matches the marginal
        blocks (~ node degree), keeping CG well-conditioned. The moment RHS is 0, so this rescaling
        leaves the constraint set -- and hence the projected M -- unchanged."""
    cf = moment_coef.to(dtype=dtype)
    D = cf.shape[1]
    deg_t = scatter_sum_torch(torch.ones(cf.shape[0], dtype=cf.dtype, device=cf.device), tgt_index, n_tgt)
    cf = cf * (deg_t[tgt_index] >= (D + 1)).to(cf.dtype).unsqueeze(1)
    rms = torch.sqrt((cf ** 2).mean()).clamp_min(1e-30)
    return cf / rms


def _AAt_matvec(vt, vs, src_index, tgt_index, n_src, n_tgt, eps, moment_coef=None, vm=None, eps_m=None):
    # (A Aᵀ + εI) [vt; vs (; vm)], matrix-free.  The moment block gets its own (larger) ridge eps_m so
    # near-degenerate geometry damps toward no-correction instead of exploding.
    #   (Aᵀv)_e = vt[tgt_e] + vs[src_e] + Σ_d vm[tgt_e, d] · coef[e, d]
    av = vt[tgt_index] + vs[src_index]
    if moment_coef is not None:
        for d in range(moment_coef.shape[1]):
            av = av + vm[:, d][tgt_index] * moment_coef[:, d]
    out_t = scatter_sum_torch(av, tgt_index, n_tgt) + eps * vt
    out_s = scatter_sum_torch(av, src_index, n_src) + eps * vs
    if moment_coef is not None:
        out_m = torch.stack([scatter_sum_torch(av * moment_coef[:, d], tgt_index, n_tgt)
                             for d in range(moment_coef.shape[1])], dim=1) + eps_m * vm
        return out_t, out_s, out_m
    return out_t, out_s


def doubly_constrained_project(q, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt,
                               eps_rel=1e-9, n_cg=200, tol=1e-12, moment_coef=None,
                               solve_dtype=None):
    """q: edge masses [E]. Returns projected masses M* [E] with both marginals enforced (and, when
    moment_coef [E, D] is given, exact ℓ=1 reproduction). Unrolled CG so autograd flows through the solve."""
    dtype = solve_dtype or torch.float32
    q = q.to(dtype=dtype)
    asrc = area_src.to(dtype=dtype); atgt = area_tgt.to(dtype=dtype)
    # Both marginals are simultaneously satisfiable ONLY if total areas match: AAᵀ has a rank-1 null
    # space [1_tgt; -1_src], and any mismatch (e.g. orphan source cells left at area 0, or grids whose
    # cell areas don't both sum to 4π) lands UNDAMPED on BOTH marginals -- silently breaking the
    # conservation AND consistency invariants. Warn (never abort) so it can't be mistaken for model error.
    _ts = float(atgt.sum()); _ss = float(asrc.sum())
    if abs(_ts - _ss) > 1e-6 * (0.5 * (abs(_ts) + abs(_ss)) + 1e-30):
        import warnings
        n_orphan_s = int((scatter_sum_torch(torch.ones_like(q), src_index, n_src) == 0).sum())
        n_orphan_t = int((scatter_sum_torch(torch.ones_like(q), tgt_index, n_tgt) == 0).sum())
        warnings.warn(
            "doubly_constrained_project: total-area mismatch sum(area_tgt)=%.6e vs sum(area_src)=%.6e "
            "(rel %.2e); BOTH marginals will be violated. zero-degree cells: src=%d tgt=%d. "
            "Check for orphan cells / area normalization." % (
                _ts, _ss, abs(_ts - _ss) / (0.5 * (abs(_ts) + abs(_ss)) + 1e-30), n_orphan_s, n_orphan_t),
            RuntimeWarning)
    has_m = moment_coef is not None
    if has_m:
        cf = _prep_moment(moment_coef, tgt_index, n_tgt, dtype=dtype); Dm = cf.shape[1]
    # rhs = b - A q   (moment block: b_m = 0)
    rt = atgt - scatter_sum_torch(q, tgt_index, n_tgt)
    rs = asrc - scatter_sum_torch(q, src_index, n_src)
    if has_m:
        rm = -torch.stack([scatter_sum_torch(q * cf[:, d], tgt_index, n_tgt) for d in range(Dm)], dim=1)
    # ridge ~ mean node degree (diagonal of AAᵀ); moment block gets a larger ridge for robustness
    deg_t = scatter_sum_torch(torch.ones_like(q), tgt_index, n_tgt)
    deg_s = scatter_sum_torch(torch.ones_like(q), src_index, n_src)
    eps = eps_rel * 0.5 * (deg_t.mean() + deg_s.mean())
    eps_m = 100.0 * eps

    # CG on the stacked system (matrix-free, unrolled for autograd)
    xt = torch.zeros_like(rt); xs = torch.zeros_like(rs)
    pt, ps = rt.clone(), rs.clone()
    rsold = (rt * rt).sum() + (rs * rs).sum()
    bnorm = torch.sqrt((atgt * atgt).sum() + (asrc * asrc).sum())
    if has_m:
        xm = torch.zeros_like(rm); pm = rm.clone(); rsold = rsold + (rm * rm).sum()
        bnorm = torch.sqrt(bnorm * bnorm + (rm * rm).sum())
    for _ in range(n_cg):
        if has_m:
            Apt, Aps, Apm = _AAt_matvec(pt, ps, src_index, tgt_index, n_src, n_tgt, eps, cf, pm, eps_m)
            denom = (pt * Apt).sum() + (ps * Aps).sum() + (pm * Apm).sum()
        else:
            Apt, Aps = _AAt_matvec(pt, ps, src_index, tgt_index, n_src, n_tgt, eps)
            denom = (pt * Apt).sum() + (ps * Aps).sum()
        alpha = rsold / torch.clamp(denom, min=1e-30)
        xt = xt + alpha * pt; xs = xs + alpha * ps
        rt = rt - alpha * Apt; rs = rs - alpha * Aps
        if has_m:
            xm = xm + alpha * pm; rm = rm - alpha * Apm
        rsnew = (rt * rt).sum() + (rs * rs).sum()
        if has_m:
            rsnew = rsnew + (rm * rm).sum()
        if torch.sqrt(rsnew) / torch.clamp(bnorm, min=1e-30) < tol:
            break
        beta = rsnew / torch.clamp(rsold, min=1e-30)
        pt = rt + beta * pt; ps = rs + beta * ps
        if has_m:
            pm = rm + beta * pm
        rsold = rsnew

    M = q + xt[tgt_index] + xs[src_index]
    if has_m:
        for d in range(Dm):
            M = M + xm[:, d][tgt_index] * cf[:, d]
    return M


def _cg_solve(rt, rs, src_index, tgt_index, n_src, n_tgt, eps, n_cg, tol, bnorm,
              moment_coef=None, rm=None, eps_m=None):
    """Solve (AAᵀ + εI) [xt; xs (; xm)] = [rt; rs (; rm)] matrix-free with CG. No autograd graph is needed
    (used inside the custom Function below), so this runs cheaply with O(nodes) memory regardless of n_cg."""
    has_m = moment_coef is not None
    xt = torch.zeros_like(rt); xs = torch.zeros_like(rs)
    pt, ps = rt.clone(), rs.clone()
    rsold = (rt * rt).sum() + (rs * rs).sum()
    if has_m:
        xm = torch.zeros_like(rm); pm = rm.clone(); rsold = rsold + (rm * rm).sum()
    for _ in range(n_cg):
        if has_m:
            Apt, Aps, Apm = _AAt_matvec(pt, ps, src_index, tgt_index, n_src, n_tgt, eps, moment_coef, pm, eps_m)
            denom = (pt * Apt).sum() + (ps * Aps).sum() + (pm * Apm).sum()
        else:
            Apt, Aps = _AAt_matvec(pt, ps, src_index, tgt_index, n_src, n_tgt, eps)
            denom = (pt * Apt).sum() + (ps * Aps).sum()
        alpha = rsold / torch.clamp(denom, min=1e-30)
        xt = xt + alpha * pt; xs = xs + alpha * ps
        rt = rt - alpha * Apt; rs = rs - alpha * Aps
        if has_m:
            xm = xm + alpha * pm; rm = rm - alpha * Apm
        rsnew = (rt * rt).sum() + (rs * rs).sum()
        if has_m:
            rsnew = rsnew + (rm * rm).sum()
        if torch.sqrt(rsnew) / torch.clamp(bnorm, min=1e-30) < tol:
            break
        beta = rsnew / torch.clamp(rsold, min=1e-30)
        pt = rt + beta * pt; ps = rs + beta * ps
        if has_m:
            pm = rm + beta * pm
        rsold = rsnew
    if has_m:
        return xt, xs, xm
    return xt, xs


class _DCProjectImplicit(torch.autograd.Function):
    """Memory-O(1) differentiable doubly-constrained projection.

    Forward map is M = q + Aᵀλ with (AAᵀ+εI)λ = b − Aq, i.e. M = q − Aᵀ(AAᵀ+εI)⁻¹A q + const = Pq + c
    with P = I − Aᵀ(AAᵀ+εI)⁻¹A. P is SYMMETRIC, so the adjoint is the same operator: for incoming
    grad_M, grad_q = P·grad_M = grad_M − Aᵀ(AAᵀ+εI)⁻¹(A·grad_M). Both forward and backward are one CG
    solve and store NO per-iteration graph -- this is the exact gradient of the regularized map, not an
    unroll approximation, and it makes the 400-iter solve free of autograd memory. Appending the ℓ=1
    moment block to A leaves P symmetric, so the backward is the identical operator on the larger A."""

    @staticmethod
    def forward(ctx, q, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt, eps_rel, n_cg, tol,
                moment_coef=None, solve_dtype=None):
        dtype = solve_dtype or torch.float32
        ctx.q_dtype = q.dtype
        q = q.to(dtype=dtype); asrc = area_src.to(dtype=dtype); atgt = area_tgt.to(dtype=dtype)
        deg_t = scatter_sum_torch(torch.ones_like(q), tgt_index, n_tgt)
        deg_s = scatter_sum_torch(torch.ones_like(q), src_index, n_src)
        eps = float(eps_rel) * 0.5 * (deg_t.mean() + deg_s.mean())
        eps_m = 100.0 * eps
        rt = atgt - scatter_sum_torch(q, tgt_index, n_tgt)
        rs = asrc - scatter_sum_torch(q, src_index, n_src)
        bnorm = torch.sqrt((atgt * atgt).sum() + (asrc * asrc).sum())
        has_m = moment_coef is not None
        if has_m:
            cf = _prep_moment(moment_coef, tgt_index, n_tgt, dtype=dtype); Dm = cf.shape[1]
            rm = -torch.stack([scatter_sum_torch(q * cf[:, d], tgt_index, n_tgt) for d in range(Dm)], dim=1)
            bnorm = torch.sqrt(bnorm * bnorm + (rm * rm).sum())
            xt, xs, xm = _cg_solve(rt, rs, src_index, tgt_index, n_src, n_tgt, eps, n_cg, tol, bnorm, cf, rm, eps_m)
            M = q + xt[tgt_index] + xs[src_index]
            for d in range(Dm):
                M = M + xm[:, d][tgt_index] * cf[:, d]
            ctx.save_for_backward(src_index, tgt_index, eps.detach(), moment_coef)
        else:
            xt, xs = _cg_solve(rt, rs, src_index, tgt_index, n_src, n_tgt, eps, n_cg, tol, bnorm)
            M = q + xt[tgt_index] + xs[src_index]
            ctx.save_for_backward(src_index, tgt_index, eps.detach())
        ctx.has_moment = has_m
        ctx.n_src = n_src; ctx.n_tgt = n_tgt; ctx.n_cg = n_cg; ctx.tol = tol
        return M

    @staticmethod
    def backward(ctx, grad_M):
        if ctx.has_moment:
            src_index, tgt_index, eps, moment_coef = ctx.saved_tensors
            cf = _prep_moment(moment_coef, tgt_index, ctx.n_tgt, dtype=grad_M.dtype); Dm = cf.shape[1]
        else:
            src_index, tgt_index, eps = ctx.saved_tensors
        n_src, n_tgt, n_cg, tol = ctx.n_src, ctx.n_tgt, ctx.n_cg, ctx.tol
        eps_m = 100.0 * eps
        g = grad_M.to(dtype=eps.dtype)
        # A·grad_M  ->  node residuals
        rt = scatter_sum_torch(g, tgt_index, n_tgt)
        rs = scatter_sum_torch(g, src_index, n_src)
        if ctx.has_moment:
            rm = torch.stack([scatter_sum_torch(g * cf[:, d], tgt_index, n_tgt) for d in range(Dm)], dim=1)
            bnorm = torch.sqrt((rt * rt).sum() + (rs * rs).sum() + (rm * rm).sum()).clamp_min(1e-30)
            mt, ms, mm = _cg_solve(rt, rs, src_index, tgt_index, n_src, n_tgt, eps, n_cg, tol, bnorm, cf, rm, eps_m)
            grad_q = g - (mt[tgt_index] + ms[src_index])
            for d in range(Dm):
                grad_q = grad_q - mm[:, d][tgt_index] * cf[:, d]
        else:
            bnorm = torch.sqrt((rt * rt).sum() + (rs * rs).sum()).clamp_min(1e-30)
            mt, ms = _cg_solve(rt, rs, src_index, tgt_index, n_src, n_tgt, eps, n_cg, tol, bnorm)
            grad_q = g - (mt[tgt_index] + ms[src_index])
        # only q (arg 0) needs a gradient
        return grad_q.to(dtype=ctx.q_dtype), None, None, None, None, None, None, None, None, None, None, None


def doubly_constrained_project_implicit(q, src_index, tgt_index, area_src, area_tgt, n_src, n_tgt,
                                        eps_rel=1e-9, n_cg=400, tol=1e-12, moment_coef=None,
                                        solve_dtype=None):
    """Drop-in for doubly_constrained_project with implicit-function-theorem gradients (no unrolled-CG
    autograd graph). Same projected masses; O(nodes) backward memory instead of O(n_cg·edges). Pass
    moment_coef [E, D] to additionally enforce exact ℓ=1 (linear) reproduction (see module docstring)."""
    return _DCProjectImplicit.apply(q, src_index, tgt_index, area_src, area_tgt,
                                    n_src, n_tgt, eps_rel, n_cg, tol, moment_coef, solve_dtype)
