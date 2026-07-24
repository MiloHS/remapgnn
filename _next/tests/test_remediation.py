from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from remapgnn_next.checkpoint import validate_production_manifest
from remapgnn_next.config import ExperimentConfig, StageConfig, load_config
from remapgnn_next.constraints import project_marginals
from remapgnn_next.evaluation import promotion_report, safe_ratio
from remapgnn_next.panels import band_degrees
from remapgnn_next.progressive import ConservativeCorrectionStage, ProgressiveRemapper
from remapgnn_next.provenance import file_sha256
from remapgnn_next.training import benefit_teacher_labels


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
