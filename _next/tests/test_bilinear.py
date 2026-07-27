from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from remapgnn_next.band_panels import (
    _cross_mixtures, _cross_target_role, _within_band_mixtures,
)
from remapgnn_next.bilinear import apply_conservative_bilinear
from remapgnn_next.config import load_config
from remapgnn_next.sparse import index_sum
from remapgnn_next.types import FieldBatch, SparseOperator


def test_schema5_has_explicit_contiguous_bands():
    config = load_config("_next/configs/bilinear_progressive.json")
    assert [(band.name, band.degree_min, band.degree_max) for band in config.bands] == [
        ("low", 1, 16),
        ("mid", 17, 32),
        ("high", 33, 48),
        ("very_high_guard", 49, 64),
    ]
    assert [stage.target_band for stage in config.stages] == ["low", "mid", "high"]
    assert "frequency_cells_per_k_squared" not in config.to_dict()["panel"]
    assert config.features.edge == ("knn_rank_over_target_count",)
    assert "ICO-r32_to_CS-r32" in config.pair_roles["train"]
    assert config.panel.validation_train_pairs == ("ICO-r32_to_CS-r32",)
    assert config.baseline.correction_reference == "blended_bilinear"
    assert config.baseline.bilinear_reference_fraction == pytest.approx(0.75)


def test_rank_one_bilinear_preserves_constants_and_integrals(synthetic_pair):
    area_src = synthetic_pair.area_src.double()
    area_tgt = synthetic_pair.area_tgt.double()
    # Deliberately ignore one source in the nonconservative bilinear stencil.
    src = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.long)
    tgt = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    weight = torch.tensor(
        [0.75, 0.25, 0.4, 0.6, 0.3, 0.7], dtype=torch.float64
    )
    baseline = SparseOperator.from_weight(src, tgt, weight, area_src, area_tgt)
    pair = replace(
        synthetic_pair,
        base_operator=baseline,
        metadata={"baseline_kind": "conservative_esmf_bilinear"},
    )
    fields = torch.stack((
        torch.ones(pair.n_src),
        torch.arange(pair.n_src, dtype=torch.float32) - 2.0,
        1.0e-9 * torch.arange(pair.n_src, dtype=torch.float32),
    ))
    output, raw, shift = apply_conservative_bilinear(pair, fields)
    assert torch.equal(output[0], torch.ones_like(output[0]))
    source_integral = (fields.double() * area_src).sum(1)
    target_integral = (output.double() * area_tgt).sum(1)
    assert torch.allclose(target_integral, source_integral, atol=2e-7, rtol=2e-7)
    assert not torch.equal(raw[1], output[1])
    assert torch.isfinite(shift).all()


def _band_batch(name, role, offset=0):
    source = torch.stack((
        torch.tensor([1.0, -1.0, 0.5, -0.5]),
        torch.tensor([0.5, 0.25, -1.0, 0.25]),
        torch.tensor([-0.25, 1.0, -0.5, -0.25]),
    ))
    truth = source[:, :2].clone()
    return FieldBatch(
        source, truth, torch.full((3,), float("nan")),
        [(offset + index + 1, 0) for index in range(3)],
        [role] * 3,
        [f"{name}:{index}" for index in range(3)],
        [f"{name}_mode"] * 3,
        torch.full((3,), role == "target", dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
        torch.arange(offset + 1, offset + 4, dtype=torch.long),
        [name] * 3,
    )


def _sized_band_batch(name, role, count, offset):
    generator = torch.Generator().manual_seed(1000 + offset)
    source = torch.randn(count, 8, generator=generator)
    truth = source[:, :4].clone()
    return FieldBatch(
        source, truth, torch.full((count,), float("nan")),
        [(offset + index + 1, 0) for index in range(count)],
        [role] * count,
        [f"{name}:{index}" for index in range(count)],
        [f"{name}_mode"] * count,
        torch.full((count,), role == "target", dtype=torch.bool),
        torch.zeros(count, dtype=torch.bool),
        torch.arange(offset + 1, offset + count + 1, dtype=torch.long),
        [name] * count,
    )


def test_within_and_cross_band_mixtures_are_labeled_and_normalized():
    area = torch.full((4,), 0.25)
    target = _band_batch("low", "target")
    guard_a = _band_batch("mid", "safety", 16)
    guard_b = _band_batch("high", "safety", 32)
    within = _within_band_mixtures(
        target, area, 4, 7, role="target", family="target_mixture"
    )
    cross_target = _cross_mixtures(
        target, [guard_a, guard_b], area, 6, 8,
        role="target", family="cross_target_mixture",
        fractions=(0.25, 0.5, 0.75),
    )
    cross_guard = _cross_mixtures(
        guard_a, [guard_b], area, 3, 9,
        role="safety", family="cross_guard_mixture",
    )
    for batch in (within, cross_target, cross_guard):
        rms = (area.view(1, -1) * batch.source.square()).sum(1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6)
        assert len(batch.source_keys) == len(set(batch.source_keys))
        assert torch.isnan(batch.frequency).all()
        assert torch.equal(batch.degrees, torch.full_like(batch.degrees, -1))
    assert set(cross_target.bands) == {"low"}
    assert not bool(cross_guard.is_target.any())
    assert all("+" in band for band in cross_guard.bands)


def test_cross_mixtures_resample_exact_training_collision():
    # These component counts and this seed reproduce the collision that
    # stopped ICOD-r32_to_CS-r32, low-stage epoch 1 (panel epoch 1001).
    area = torch.full((8,), 1.0 / 8.0)
    target = _sized_band_batch("low", "target", 21, 0)
    guards = [
        _sized_band_batch("mid", "safety", 24, 100),
        _sized_band_batch("high", "safety", 24, 200),
        _sized_band_batch("very_high_guard", "safety", 24, 300),
    ]
    result = _cross_mixtures(
        target, guards, area, 12, 607176503,
        role="target", family="cross_target_mixture",
        fractions=(0.25, 0.5, 0.75),
    )
    assert result.source.shape[0] == 12
    assert len(result.source_keys) == len(set(result.source_keys))


def test_cross_target_roles_keep_guard_dominated_mixtures_as_safety():
    assert _cross_target_role(0.25) == "safety"
    assert _cross_target_role(0.50) == "safety"
    assert _cross_target_role(0.75) == "target"


def test_uniform_correction_reference_has_unit_rows(synthetic_pair):
    reference = torch.ones_like(
        synthetic_pair.fv_operator.weight, dtype=torch.float64
    )
    degree = index_sum(
        reference, synthetic_pair.tgt_index, synthetic_pair.n_tgt
    )
    normalized = reference / degree[synthetic_pair.tgt_index]
    rows = index_sum(normalized, synthetic_pair.tgt_index, synthetic_pair.n_tgt)
    assert torch.allclose(rows, torch.ones_like(rows))
