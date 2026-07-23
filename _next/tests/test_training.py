import torch

from remapgnn_next.config import StageConfig
from remapgnn_next.progressive import ConservativeCorrectionStage, ProgressiveRemapper
from remapgnn_next.training import assert_unchanged, identity_floor_selection, parameter_snapshot


def test_phase_freezing_across_optimizer_step(synthetic_pair):
    stages = [
        ConservativeCorrectionStage(StageConfig(name="first", band_lower=1.0, band_upper=1.2)),
        ConservativeCorrectionStage(StageConfig(name="second", band_lower=1.2, band_upper=1.5)),
    ]
    model = ProgressiveRemapper(synthetic_pair.fv_operator, stages)
    model.set_training_stage(1, "capability")
    frozen = list(stages[0].parameters()) + list(stages[1].router_parameters())
    snapshot = parameter_snapshot(frozen)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    output, _ = model(synthetic_pair, torch.randn(2, synthetic_pair.n_src),
                      gate_modes=["forced_closed", "forced_open"])
    output.square().mean().backward()
    optimizer.step()
    assert_unchanged(frozen, snapshot)


def test_identity_floor_and_phase_transition():
    assert identity_floor_selection(0.89, 1.0, 0.10)
    assert not identity_floor_selection(0.91, 1.0, 0.10)
    stage = ConservativeCorrectionStage(StageConfig(name="x", band_lower=1.0, band_upper=2.0))
    stage.set_training_phase("capability")
    assert all(p.requires_grad for p in stage.corrector_parameters())
    assert not any(p.requires_grad for p in stage.router_parameters())
    stage.set_training_phase("router")
    assert not any(p.requires_grad for p in stage.corrector_parameters())
    assert all(p.requires_grad for p in stage.router_parameters())


def test_runtime_import_boundary():
    from pathlib import Path
    import ast
    root = Path("_next/remapgnn_next")
    forbidden = ("remapgnn", "scripts")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(forbidden) for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden)
