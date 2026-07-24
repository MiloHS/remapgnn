from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from remapgnn_next.checkpoint import (
    CLEAN_PROGRESSIVE_FORMAT, PROGRESSIVE_SCHEMA_VERSION,
    validate_production_manifest,
)
from remapgnn_next.config import ExperimentConfig, StageConfig, load_config
from remapgnn_next.constraints import project_marginals
from remapgnn_next.evaluation import promotion_report, safe_ratio
from remapgnn_next.panels import band_degrees
from remapgnn_next.progressive import ConservativeCorrectionStage, ProgressiveRemapper
from remapgnn_next.provenance import file_sha256
from remapgnn_next.equivalence import (
    compare_clean_checkpoints, harden_checkpoint,
)
from remapgnn_next.training import benefit_teacher_labels
from remapgnn_next.types import PairData, SparseOperator


def test_schema4_rejects_unknown_top_level_key():
    value = load_config("_next/configs/progressive.json").raw
    changed = dict(value)
    changed["losss"] = {"guard_weight": 999}
    with pytest.raises(ValueError, match="unknown experiment keys"):
        ExperimentConfig.from_dict(changed)


@pytest.mark.parametrize("cells", [6144, 10242, 16200, 20480, 163842])
def test_band_upper_bound_is_never_exceeded(cells):
    k = np.sqrt(cells / 6.0)
    degrees = band_degrees(k, 1.25, 1.5)
    assert degrees[-1] <= np.floor(1.5 * k)
    assert degrees[-1] / k <= 1.5


def test_safe_ratio_zero_semantics():
    assert safe_ratio(0.0, 0.0, 1e-14) == 1.0
    assert safe_ratio(1e-16, 1e-16, 1e-14) == 1.0
    assert np.isinf(safe_ratio(1.0, 0.0, 1e-14))
    assert safe_ratio(2.0, 4.0, 1e-14) == 0.5


def test_promotion_fails_closed_on_nan():
    config = load_config("_next/configs/progressive.json")
    detail = [
        {"pair": "p", "is_target_band": True, "is_prefix_band": False,
         "model_over_prefix": np.nan, "model_over_fv": np.nan},
        {"pair": "p", "is_target_band": False, "is_prefix_band": True,
         "model_over_prefix": np.nan, "model_over_fv": np.nan},
    ]
    result = promotion_report(detail, [{"pair": "p", "error": "synthetic"}], config, ["p"])
    assert not result["passed"]
    assert any("non-finite" in value for value in result["failures"])


def test_closed_straight_through_gate_has_task_gradient(synthetic_pair):
    stage = ConservativeCorrectionStage(
        StageConfig(name="high", band_lower=1.25, band_upper=1.5)
    )
    with torch.no_grad():
        stage.score_mlp.net[-1].weight.normal_(std=0.1)
        stage.field_gate_mlp.net[-1].bias.fill_(-10)
        stage.local_gate_mlp.net[-1].bias.fill_(10)
    stage.set_training_phase("router")
    model = ProgressiveRemapper(synthetic_pair.fv_operator, [stage])
    source = torch.randn(2, synthetic_pair.n_src)
    output, diagnostic = model(
        synthetic_pair, source, gate_modes=["straight_through"]
    )
    assert torch.equal(output, diagnostic.fv_output)
    loss = (output - torch.randn_like(output)).square().mean()
    loss.backward()
    gradient = stage.field_gate_mlp.net[-1].bias.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient)


def test_progressive_features_accept_float64_fv_areas(synthetic_pair):
    operator = SparseOperator.from_weight(
        synthetic_pair.src_index,
        synthetic_pair.tgt_index,
        synthetic_pair.fv_operator.weight.double(),
        synthetic_pair.area_src.double(),
        synthetic_pair.area_tgt.double(),
    )
    pair = PairData(
        pair=synthetic_pair.pair,
        edge_features=synthetic_pair.edge_features,
        src_xyz=synthetic_pair.src_xyz,
        tgt_xyz=synthetic_pair.tgt_xyz,
        src_neighbor_index=synthetic_pair.src_neighbor_index,
        src_neighbor_weight=synthetic_pair.src_neighbor_weight,
        tgt_neighbor_index=synthetic_pair.tgt_neighbor_index,
        tgt_neighbor_weight=synthetic_pair.tgt_neighbor_weight,
        fv_operator=operator,
    )
    stage = ConservativeCorrectionStage(
        StageConfig(name="mid", band_lower=1.0, band_upper=1.25)
    )
    output = ProgressiveRemapper(operator, [stage])(
        pair, torch.randn(2, pair.n_src), return_diagnostics=False
    )
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_detached_production_manifest_binds_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "equivalence.json"
    report.write_text(json.dumps({
        "passed": True, "checkpoint_sha256": file_sha256(checkpoint)
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "remapgnn.production_manifest", "schema_version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "fv_checkpoint": {
            "path": str(checkpoint), "sha256": file_sha256(checkpoint)
        },
        "equivalence_report": {
            "path": str(report), "sha256": file_sha256(report)
        },
    }))
    validate_production_manifest(checkpoint, manifest)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        validate_production_manifest(checkpoint, manifest)


def test_hardened_checkpoint_preserves_clean_payload(tmp_path):
    stage = ConservativeCorrectionStage(
        StageConfig(name="mid", band_lower=1.0, band_upper=1.25)
    )
    state = {
        name: value.detach().clone()
        for name, value in stage.state_dict().items()
    }
    source = tmp_path / "source.pt"
    hardened = tmp_path / "hardened.pt"
    torch.save({
        "format": CLEAN_PROGRESSIVE_FORMAT,
        "schema_version": PROGRESSIVE_SCHEMA_VERSION,
        "runtime_data": {"edge_features": ["a"]},
        "fv_checkpoint": {"path": "fv.pt", "sha256": "abc"},
        "selected_identity": True,
        "stages": [{
            "config": stage.config.to_dict(),
            "state": state,
        }],
    }, source)
    result = harden_checkpoint(
        source, hardened, repository=".", allow_dirty=True
    )
    comparison = compare_clean_checkpoints(source, hardened)
    assert comparison["passed"]
    assert result["source_checkpoint_sha256"] == file_sha256(source)
    assert result["checkpoint_sha256"] == file_sha256(hardened)


def test_hardened_manifest_rejects_cpu_only_evidence(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "equivalence.json"
    report.write_text(json.dumps({
        "format": "remapgnn.hardened_equivalence",
        "passed": True,
        "acceptance_ready": False,
        "checkpoint_sha256": file_sha256(checkpoint),
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "remapgnn.production_manifest", "schema_version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "fv_checkpoint": {
            "path": str(checkpoint), "sha256": file_sha256(checkpoint)
        },
        "equivalence_report": {
            "path": str(report), "sha256": file_sha256(report)
        },
    }))
    with pytest.raises(ValueError, match="not acceptance-ready"):
        validate_production_manifest(checkpoint, manifest)


def test_fv_marginal_projection_checks_actual_constraints():
    source = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    target = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    area_source = torch.tensor([0.4, 0.6], dtype=torch.float64)
    area_target = torch.tensor([0.5, 0.5], dtype=torch.float64)
    mass, info = project_marginals(
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
        source, target, area_source, area_target,
        iterations=200, epsilon_relative=0.0, assert_converged=True,
        return_info=True, row_tolerance=1e-10, column_tolerance=1e-10,
    )
    assert info.converged
    assert info.row_max <= 1e-10
    assert info.column_max <= 1e-10
    assert torch.isfinite(mass).all()


def test_fv_marginal_projection_rejects_incompatible_totals():
    source = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    target = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="incompatible"):
        project_marginals(
            torch.ones(4, dtype=torch.float64), source, target,
            torch.tensor([0.4, 0.5], dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )


def test_fv_total_compatibility_uses_aggregate_cell_tolerances():
    source = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    target = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mass, info = project_marginals(
        torch.full((4,), 0.25, dtype=torch.float64),
        source, target,
        torch.tensor([0.5, 0.5000000002], dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        epsilon_relative=0.0, iterations=100,
        row_tolerance=1e-10, column_tolerance=1e-10,
        assert_converged=True, return_info=True,
    )
    assert info.converged
    assert torch.isfinite(mass).all()


def test_benefit_teacher_labels_reward_helpful_corrections():
    truth = torch.zeros(3, 2)
    prefix = torch.tensor([[2., 2.], [1., 1.], [0., 0.]])
    opened = torch.tensor([[1., 1.], [2., 2.], [0., 0.]])
    field, local = benefit_teacher_labels(
        prefix, opened, truth, torch.ones(2), temperature=0.1
    )
    assert field[0] > 0.5 and torch.all(local[0] > 0.5)
    assert field[1] < 0.5 and torch.all(local[1] < 0.5)
    assert field[2] == pytest.approx(0.5)
