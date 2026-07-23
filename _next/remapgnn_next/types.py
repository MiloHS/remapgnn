from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

"""Defines shared data types: sparse operators, mesh-pair data, batches of source/target
    fields, stage settings, and diagnostics."""

import torch


def _moved(value: Any, device: torch.device | str):
    return value.to(device) if torch.is_tensor(value) else value


@dataclass(frozen=True)
class SparseOperator:
    """One edge-ordered conservative operator.

    ``weight`` is the normalized field operator S and ``mass`` is
    ``area_tgt[tgt_index] * S``. Both are retained because FV construction is
    naturally expressed in mass space while field application uses S.
    """

    src_index: torch.Tensor
    tgt_index: torch.Tensor
    weight: torch.Tensor
    mass: torch.Tensor
    area_src: torch.Tensor
    area_tgt: torch.Tensor
    n_src: int
    n_tgt: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        e = int(self.src_index.numel())
        if self.src_index.dtype != torch.long or self.tgt_index.dtype != torch.long:
            raise TypeError("operator indices must be torch.long")
        if any(int(x.numel()) != e for x in (self.tgt_index, self.weight, self.mass)):
            raise ValueError("indices, weights, and masses must have equal length")
        if tuple(self.area_src.shape) != (int(self.n_src),):
            raise ValueError("area_src shape does not match n_src")
        if tuple(self.area_tgt.shape) != (int(self.n_tgt),):
            raise ValueError("area_tgt shape does not match n_tgt")
        if e and (int(self.src_index.min()) < 0 or int(self.src_index.max()) >= self.n_src):
            raise ValueError("source index outside operator dimensions")
        if e and (int(self.tgt_index.min()) < 0 or int(self.tgt_index.max()) >= self.n_tgt):
            raise ValueError("target index outside operator dimensions")

    @property
    def n_edges(self) -> int:
        return int(self.src_index.numel())

    @property
    def indices(self) -> torch.Tensor:
        return torch.stack((self.tgt_index, self.src_index), dim=0)

    @property
    def normalized_weights(self) -> torch.Tensor:
        return self.weight

    @property
    def mass_weights(self) -> torch.Tensor:
        return self.mass

    @property
    def areas(self):
        return self.area_src, self.area_tgt

    @property
    def dimensions(self):
        return self.n_src, self.n_tgt

    @classmethod
    def from_weight(
        cls, src_index, tgt_index, weight, area_src, area_tgt, *, provenance=None
    ) -> "SparseOperator":
        mass = weight * area_tgt.to(weight.dtype)[tgt_index]
        return cls(
            src_index=src_index,
            tgt_index=tgt_index,
            weight=weight,
            mass=mass,
            area_src=area_src,
            area_tgt=area_tgt,
            n_src=int(area_src.numel()),
            n_tgt=int(area_tgt.numel()),
            provenance={} if provenance is None else dict(provenance),
        )

    @classmethod
    def from_mass(
        cls, src_index, tgt_index, mass, area_src, area_tgt, *, provenance=None
    ) -> "SparseOperator":
        weight = mass / area_tgt.to(mass.dtype)[tgt_index].clamp_min(1.0e-30)
        return cls(
            src_index=src_index,
            tgt_index=tgt_index,
            weight=weight,
            mass=mass,
            area_src=area_src,
            area_tgt=area_tgt,
            n_src=int(area_src.numel()),
            n_tgt=int(area_tgt.numel()),
            provenance={} if provenance is None else dict(provenance),
        )

    def to(self, device: torch.device | str) -> "SparseOperator":
        return SparseOperator(
            self.src_index.to(device), self.tgt_index.to(device),
            self.weight.to(device), self.mass.to(device),
            self.area_src.to(device), self.area_tgt.to(device),
            self.n_src, self.n_tgt, dict(self.provenance),
        )


@dataclass(frozen=True)
class PairData:
    pair: str
    edge_features: torch.Tensor
    src_xyz: torch.Tensor
    tgt_xyz: torch.Tensor
    src_neighbor_index: torch.Tensor
    src_neighbor_weight: torch.Tensor
    tgt_neighbor_index: torch.Tensor
    tgt_neighbor_weight: torch.Tensor
    fv_operator: SparseOperator
    src_node_features: torch.Tensor | None = None
    tgt_node_features: torch.Tensor | None = None
    fv_coord_src: torch.Tensor | None = None
    fv_coord_tgt: torch.Tensor | None = None
    fv_quad_src: torch.Tensor | None = None
    fv_quad_tgt: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def src_index(self):
        return self.fv_operator.src_index

    @property
    def tgt_index(self):
        return self.fv_operator.tgt_index

    @property
    def area_src(self):
        return self.fv_operator.area_src

    @property
    def area_tgt(self):
        return self.fv_operator.area_tgt

    @property
    def n_src(self):
        return self.fv_operator.n_src

    @property
    def n_tgt(self):
        return self.fv_operator.n_tgt

    def to(self, device: torch.device | str) -> "PairData":
        names = (
            "edge_features", "src_xyz", "tgt_xyz", "src_neighbor_index",
            "src_neighbor_weight", "tgt_neighbor_index", "tgt_neighbor_weight",
            "src_node_features", "tgt_node_features", "fv_coord_src",
            "fv_coord_tgt", "fv_quad_src", "fv_quad_tgt",
        )
        values = {name: _moved(getattr(self, name), device) for name in names}
        return PairData(
            pair=self.pair, fv_operator=self.fv_operator.to(device),
            metadata=dict(self.metadata), **values,
        )


@dataclass(frozen=True)
class FieldBatch:
    source: torch.Tensor
    truth: torch.Tensor
    frequency: torch.Tensor
    labels: Sequence[Any]
    roles: Sequence[str]
    source_keys: Sequence[str] = ()
    families: Sequence[str] = ()
    target_mask: torch.Tensor | None = None

    def __post_init__(self):
        n = int(self.source.shape[0])
        if self.source.ndim != 2 or self.truth.ndim != 2:
            raise ValueError("source and truth must have shape [fields,cells]")
        if any(len(x) != n for x in (self.labels, self.roles)):
            raise ValueError("labels and roles must match the field count")
        if int(self.frequency.numel()) != n:
            raise ValueError("frequency must match the field count")
        if self.source_keys and len(self.source_keys) != n:
            raise ValueError("source_keys must be empty or match the field count")
        if self.families and len(self.families) != n:
            raise ValueError("families must be empty or match the field count")
        if self.target_mask is not None:
            if self.target_mask.dtype != torch.bool or tuple(self.target_mask.shape) != (n,):
                raise ValueError("target_mask must be boolean with one entry per field")

    @property
    def is_target(self):
        if self.target_mask is not None:
            return self.target_mask
        return torch.tensor([role == "target" for role in self.roles], dtype=torch.bool)

    def to(self, device: torch.device | str) -> "FieldBatch":
        return FieldBatch(
            self.source.to(device), self.truth.to(device), self.frequency.to(device),
            list(self.labels), list(self.roles), list(self.source_keys), list(self.families),
            None if self.target_mask is None else self.target_mask.to(device),
        )

    def subset(self, indices) -> "FieldBatch":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.source.device)
        host = index.detach().cpu().tolist()
        return FieldBatch(
            self.source[index], self.truth[index], self.frequency[index],
            [self.labels[i] for i in host], [self.roles[i] for i in host],
            [self.source_keys[i] for i in host] if self.source_keys else [],
            [self.families[i] for i in host] if self.families else [],
            None if self.target_mask is None else self.target_mask[index],
        )


@dataclass(frozen=True)
class StageDiagnostics:
    name: str
    output: torch.Tensor
    delta_weight: torch.Tensor
    row_residual: torch.Tensor
    column_residual: torch.Tensor
    field_gate: torch.Tensor
    local_gate: torch.Tensor
    field_probability: torch.Tensor
    local_probability: torch.Tensor


@dataclass(frozen=True)
class ProgressiveDiagnostics:
    fv_output: torch.Tensor
    stage_outputs: tuple[torch.Tensor, ...]
    stages: tuple[StageDiagnostics, ...]

    @property
    def output(self) -> torch.Tensor:
        return self.stage_outputs[-1] if self.stage_outputs else self.fv_output

    @property
    def gates(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((x.field_gate, x.local_gate) for x in self.stages)

    @property
    def projected_weights(self) -> tuple[torch.Tensor, ...]:
        return tuple(x.delta_weight for x in self.stages)

    @property
    def constraint_residuals(self):
        return tuple((x.row_residual, x.column_residual) for x in self.stages)
