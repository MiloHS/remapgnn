from __future__ import annotations

import torch

from .types import SparseOperator

"""Sparse-edge operations: applying an operator, gathering/scattering edge values,
    weighted reductions."""


def index_sum(values: torch.Tensor, index: torch.Tensor, size: int, *, dim: int = 0):
    """Torch-only indexed sum with a new output dimension of ``size``."""
    if dim < 0:
        dim += values.ndim
    shape = list(values.shape)
    shape[dim] = int(size)
    out = values.new_zeros(shape)
    out.index_add_(dim, index, values)
    return out


def edge_sum_fields(values: torch.Tensor, index: torch.Tensor, size: int):
    """Reduce edge dimension 1 of ``[field, edge, ...]`` tensors."""
    if values.ndim < 2:
        raise ValueError("edge_sum_fields expects at least two dimensions")
    return index_sum(values, index, size, dim=1)


def apply_edge_weights(weight, source, src_index, tgt_index, n_tgt):
    squeeze = source.ndim == 1
    if squeeze:
        source = source.unsqueeze(0)
    if source.ndim != 2:
        raise ValueError("source must have shape [source] or [field,source]")
    work = source.to(weight.dtype)
    result = edge_sum_fields(
        work[:, src_index] * weight.view(1, -1), tgt_index, int(n_tgt)
    ).to(source.dtype)
    return result.squeeze(0) if squeeze else result


def apply_operator(operator: SparseOperator, source: torch.Tensor):
    return apply_edge_weights(
        operator.weight, source, operator.src_index, operator.tgt_index, operator.n_tgt
    )


def row_normalized_reference(weight, tgt_index, n_tgt, floor=1.0e-3):
    degree = index_sum(torch.ones_like(weight), tgt_index, n_tgt).clamp_min(1.0)
    raw = weight.abs() + float(floor) / degree[tgt_index]
    denominator = index_sum(raw, tgt_index, n_tgt).clamp_min(torch.finfo(raw.dtype).tiny)
    return raw / denominator[tgt_index]
