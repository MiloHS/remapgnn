from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch

from .sparse import edge_sum_fields, index_sum

"""The conservation/consistency projection."""


@dataclass(frozen=True)
class ProjectionInfo:
    iterations: int
    converged: bool
    relative_residual: torch.Tensor
    row_max: torch.Tensor
    column_max: torch.Tensor


def correction_residuals(delta, src_index, tgt_index, area_tgt, n_src, n_tgt):
    squeeze = delta.ndim == 1
    if squeeze:
        delta = delta.unsqueeze(0)
    row = edge_sum_fields(delta, tgt_index, n_tgt)
    column = edge_sum_fields(
        delta * area_tgt.to(delta.dtype)[tgt_index].view(1, -1), src_index, n_src
    )
    return (row.squeeze(0), column.squeeze(0)) if squeeze else (row, column)


def _project_correction_impl(delta, src_index, tgt_index, area_tgt, n_src, n_tgt, iterations):
    """Orthogonal projection onto row-zero and area-column-zero edge weights.

    This uses the source-graph Laplacian formulation from the audited runtime.
    All arithmetic is float64 because squared spherical cell areas otherwise
    make the solve unnecessarily ill-conditioned.
    """
    if int(iterations) <= 0:
        return delta, 0, torch.zeros((), dtype=delta.dtype, device=delta.device)
    squeeze = delta.ndim == 1
    if squeeze:
        delta = delta.unsqueeze(0)
    original_dtype = delta.dtype
    d = delta.to(torch.float64)
    one = torch.ones_like(src_index, dtype=d.dtype)
    degree = index_sum(one, tgt_index, n_tgt).clamp_min(1.0)
    area_scale = area_tgt.to(d.dtype).mean().clamp_min(torch.finfo(d.dtype).tiny)
    area_edge = area_tgt.to(d.dtype)[tgt_index] / area_scale

    row = edge_sum_fields(d, tgt_index, n_tgt)
    d = d - row[:, tgt_index] / degree[tgt_index].view(1, -1)

    def laplacian(phi):
        edge_phi = phi[:, src_index]
        row_mean = edge_sum_fields(edge_phi, tgt_index, n_tgt) / degree.view(1, -1)
        centered = edge_phi - row_mean[:, tgt_index]
        return edge_sum_fields(
            area_edge.square().view(1, -1) * centered, src_index, n_src
        )

    rhs = edge_sum_fields(d * area_edge.view(1, -1), src_index, n_src)
    rhs = rhs - rhs.mean(dim=1, keepdim=True)
    phi = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = residual.square().sum(dim=1)
    rhs_norm = residual_sq.sqrt()
    target = torch.maximum(1.0e-11 * rhs_norm, torch.full_like(rhs_norm, 1.0e-13))
    used = 0
    for used in range(1, int(iterations) + 1):
        if bool(torch.all(residual_sq.sqrt() <= target)):
            used -= 1
            break
        lap_direction = laplacian(direction)
        denominator = (direction * lap_direction).sum(dim=1)
        active = residual_sq.sqrt() > target
        safe = torch.where(denominator.abs() > 1.0e-30, denominator, torch.ones_like(denominator))
        alpha = torch.where(active, residual_sq / safe, torch.zeros_like(safe))
        phi = phi + alpha.view(-1, 1) * direction
        new_residual = residual - alpha.view(-1, 1) * lap_direction
        new_residual = new_residual - new_residual.mean(dim=1, keepdim=True)
        new_sq = new_residual.square().sum(dim=1)
        beta = torch.where(
            active, new_sq / residual_sq.clamp_min(1.0e-300), torch.zeros_like(new_sq)
        )
        direction = new_residual + beta.view(-1, 1) * direction
        residual, residual_sq = new_residual, new_sq

    edge_phi = phi[:, src_index]
    row_mean = edge_sum_fields(edge_phi, tgt_index, n_tgt) / degree.view(1, -1)
    correction = area_edge.view(1, -1) * (edge_phi - row_mean[:, tgt_index])
    result = (d - correction).to(original_dtype)
    relative = (residual_sq.sqrt() / rhs_norm.clamp_min(1.0e-300)).max()
    return (result.squeeze(0) if squeeze else result), used, relative


class _CorrectionProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, src_index, tgt_index, area_tgt, n_src, n_tgt, iterations):
        ctx.save_for_backward(src_index, tgt_index, area_tgt)
        ctx.n_src, ctx.n_tgt, ctx.iterations = int(n_src), int(n_tgt), int(iterations)
        with torch.no_grad():
            projected, _, _ = _project_correction_impl(
                values, src_index, tgt_index, area_tgt, ctx.n_src, ctx.n_tgt, ctx.iterations
            )
        return projected

    @staticmethod
    def backward(ctx, gradient):
        src_index, tgt_index, area_tgt = ctx.saved_tensors
        with torch.no_grad():
            projected, _, _ = _project_correction_impl(
                gradient, src_index, tgt_index, area_tgt,
                ctx.n_src, ctx.n_tgt, ctx.iterations,
            )
        return projected, None, None, None, None, None, None


def project_correction(
    values, src_index, tgt_index, area_tgt, n_src, n_tgt, *, iterations=200,
    assert_converged=False, row_tolerance=1.0e-8, column_tolerance=1.0e-10,
    return_info=False,
):
    projected = _CorrectionProjection.apply(
        values, src_index, tgt_index, area_tgt, int(n_src), int(n_tgt), int(iterations)
    )
    if not (assert_converged or return_info):
        return projected
    with torch.no_grad():
        _, used, relative = _project_correction_impl(
            values.detach(), src_index, tgt_index, area_tgt, n_src, n_tgt, iterations
        )
        row, column = correction_residuals(
            projected.detach(), src_index, tgt_index, area_tgt, n_src, n_tgt
        )
        row_max, column_max = row.abs().max(), column.abs().max()
        converged = bool(row_max <= row_tolerance and column_max <= column_tolerance)
        info = ProjectionInfo(used, converged, relative, row_max, column_max)
    if assert_converged and not converged:
        raise RuntimeError(
            f"correction projection did not converge: row={float(row_max):.3e}, "
            f"column={float(column_max):.3e}, iterations={used}"
        )
    return (projected, info) if return_info else projected


def _marginal_matvec(vt, vs, src_index, tgt_index, n_src, n_tgt, epsilon):
    edge = vt[tgt_index] + vs[src_index]
    return (
        index_sum(edge, tgt_index, n_tgt) + epsilon * vt,
        index_sum(edge, src_index, n_src) + epsilon * vs,
    )


def project_marginals(
    raw_mass, src_index, tgt_index, area_src, area_tgt, *, iterations=400,
    epsilon_relative=1.0e-9, tolerance=1.0e-12, solve_dtype=torch.float64,
    assert_converged=False, return_info=False,
):
    """Project signed masses onto target and source area marginals."""
    n_src, n_tgt = int(area_src.numel()), int(area_tgt.numel())
    q = raw_mass.to(solve_dtype)
    source_area, target_area = area_src.to(solve_dtype), area_tgt.to(solve_dtype)
    mismatch = (target_area.sum() - source_area.sum()).abs()
    scale = 0.5 * (target_area.sum().abs() + source_area.sum().abs()).clamp_min(1.0e-30)
    if float(mismatch / scale) > 1.0e-6:
        warnings.warn("source and target total areas differ; both marginals cannot be exact", RuntimeWarning)
    degree_t = index_sum(torch.ones_like(q), tgt_index, n_tgt)
    degree_s = index_sum(torch.ones_like(q), src_index, n_src)
    epsilon = float(epsilon_relative) * 0.5 * (degree_t.mean() + degree_s.mean())
    rt = target_area - index_sum(q, tgt_index, n_tgt)
    rs = source_area - index_sum(q, src_index, n_src)
    xt, xs = torch.zeros_like(rt), torch.zeros_like(rs)
    pt, ps = rt.clone(), rs.clone()
    old = rt.square().sum() + rs.square().sum()
    initial = old.sqrt().clamp_min(1.0e-30)
    used = 0
    for used in range(1, int(iterations) + 1):
        apt, aps = _marginal_matvec(pt, ps, src_index, tgt_index, n_src, n_tgt, epsilon)
        denominator = (pt * apt).sum() + (ps * aps).sum()
        alpha = old / denominator.clamp_min(1.0e-30)
        xt, xs = xt + alpha * pt, xs + alpha * ps
        rt, rs = rt - alpha * apt, rs - alpha * aps
        new = rt.square().sum() + rs.square().sum()
        if bool(new.sqrt() / initial < tolerance):
            old = new
            break
        beta = new / old.clamp_min(1.0e-30)
        pt, ps = rt + beta * pt, rs + beta * ps
        old = new
    mass = q + xt[tgt_index] + xs[src_index]
    source_residual = index_sum(mass, src_index, n_src) - source_area
    target_residual = index_sum(mass, tgt_index, n_tgt) - target_area
    row_max, column_max = target_residual.abs().max(), source_residual.abs().max()
    relative = old.sqrt() / initial
    converged = bool(relative < tolerance)
    info = ProjectionInfo(used, converged, relative, row_max, column_max)
    if assert_converged and not converged:
        raise RuntimeError(
            f"marginal CG did not converge in {used} iterations (relative={float(relative):.3e})"
        )
    return (mass, info) if return_info else mass


def _prepare_moment(coefficient, tgt_index, n_tgt, dtype):
    result = coefficient.to(dtype)
    dimension = result.shape[1]
    degree = index_sum(torch.ones(result.shape[0], dtype=dtype, device=result.device), tgt_index, n_tgt)
    result = result * (degree[tgt_index] >= dimension + 1).to(dtype).unsqueeze(1)
    return result / result.square().mean().sqrt().clamp_min(1.0e-30)


def local_moment_correction(mass, tgt_index, n_tgt, coefficient, *, ridge=1.0e-4):
    coefficient = _prepare_moment(coefficient, tgt_index, n_tgt, mass.dtype)
    dimension = int(coefficient.shape[1])
    degree = index_sum(torch.ones_like(mass), tgt_index, n_tgt)
    sums = [index_sum(coefficient[:, d], tgt_index, n_tgt) for d in range(dimension)]
    moments = [index_sum(mass * coefficient[:, d], tgt_index, n_tgt) for d in range(dimension)]
    gram = mass.new_zeros((n_tgt, dimension + 1, dimension + 1))
    rhs = mass.new_zeros((n_tgt, dimension + 1))
    gram[:, 0, 0] = degree + 1.0e-12
    for d in range(dimension):
        gram[:, 0, d + 1] = sums[d]
        gram[:, d + 1, 0] = sums[d]
        rhs[:, d + 1] = -moments[d]
        for k in range(dimension):
            gram[:, d + 1, k + 1] = index_sum(
                coefficient[:, d] * coefficient[:, k], tgt_index, n_tgt
            )
        gram[:, d + 1, d + 1] += float(ridge)
    multiplier = torch.linalg.solve(gram, rhs)
    correction = multiplier[:, 0][tgt_index]
    for d in range(dimension):
        correction = correction + multiplier[:, d + 1][tgt_index] * coefficient[:, d]
    return mass + correction


def project_with_moment_relaxation(
    raw_mass, src_index, tgt_index, area_src, area_tgt, *,
    linear_coefficient=None, quadratic_coefficient=None,
    linear_relax=1.0, quadratic_relax=1.0,
    linear_iterations=1, quadratic_iterations=3,
    linear_ridge=1.0e-4, quadratic_ridge=1.0e-3,
    projection_iterations=400, epsilon_relative=1.0e-12,
):
    kwargs = dict(
        src_index=src_index, tgt_index=tgt_index, area_src=area_src, area_tgt=area_tgt,
        iterations=projection_iterations, epsilon_relative=epsilon_relative,
        solve_dtype=torch.float64,
    )
    mass = project_marginals(raw_mass, **kwargs)
    nl = int(linear_iterations) if linear_coefficient is not None else 0
    nq = int(quadratic_iterations) if quadratic_coefficient is not None else 0
    for iteration in range(max(nl, nq)):
        if iteration < nl:
            corrected = local_moment_correction(
                mass, tgt_index, int(area_tgt.numel()), linear_coefficient, ridge=linear_ridge
            )
            mass = mass + float(linear_relax) * (corrected - mass)
            mass = project_marginals(mass, **kwargs)
        if iteration < nq:
            corrected = local_moment_correction(
                mass, tgt_index, int(area_tgt.numel()), quadratic_coefficient, ridge=quadratic_ridge
            )
            mass = mass + float(quadratic_relax) * (corrected - mass)
            mass = project_marginals(mass, **kwargs)
    return mass
