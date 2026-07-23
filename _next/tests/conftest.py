from __future__ import annotations

import math

import pytest
import torch

from remapgnn_next.types import PairData, SparseOperator


@pytest.fixture
def synthetic_pair():
    torch.manual_seed(7)
    n_src, n_tgt = 4, 3
    tgt_index = torch.arange(n_tgt).repeat_interleave(n_src)
    src_index = torch.arange(n_src).repeat(n_tgt)
    area_src = torch.full((n_src,), 1.0 / n_src)
    area_tgt = torch.full((n_tgt,), 1.0 / n_tgt)
    weight = torch.full((n_src * n_tgt,), 1.0 / n_src, dtype=torch.float64)
    operator = SparseOperator.from_weight(src_index, tgt_index, weight, area_src, area_tgt)
    source_xyz = torch.tensor(
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]]
    )
    target_xyz = torch.tensor(
        [[0., -1., 0.], [0., 0., -1.], [1., 1., 1.]]
    )
    target_xyz[-1] /= target_xyz[-1].norm()
    source_neighbor = torch.tensor([
        [0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 1, 2]
    ])
    target_neighbor = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
    source_neighbor_weight = torch.full((n_src, 3), 1.0 / 3.0)
    target_neighbor_weight = torch.full((n_tgt, 3), 1.0 / 3.0)
    return PairData(
        pair="synthetic", edge_features=torch.randn(n_src * n_tgt, 8),
        src_xyz=source_xyz, tgt_xyz=target_xyz,
        src_neighbor_index=source_neighbor, src_neighbor_weight=source_neighbor_weight,
        tgt_neighbor_index=target_neighbor, tgt_neighbor_weight=target_neighbor_weight,
        fv_operator=operator,
    )
