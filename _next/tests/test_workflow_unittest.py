from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from remapgnn_next.checkpoint import (
    CLEAN_PROGRESSIVE_FORMAT, CLEAN_TRAINING_FORMAT, PROGRESSIVE_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION, build_training_model, load_training_checkpoint,
)
from remapgnn_next.config import load_config, ModelConfig, StageConfig
from remapgnn_next.constraints import correction_residuals, project_correction
from remapgnn_next.fields import source_keyed_mode_split
from remapgnn_next.progressive import ConservativeCorrectionStage, ProgressiveRemapper
from remapgnn_next.provenance import file_sha256, tensor_state_sha256
from remapgnn_next.sparse import apply_operator, index_sum
from remapgnn_next.training import (
    cvar, identity_floor_selection, pair_weights, progressive_loss,
    stratified_index, stratified_orders,
)
from remapgnn_next.types import PairData, SparseOperator
from remapgnn_next.types import FieldBatch


def synthetic_pair(name="coarse", n_src=4, n_tgt=3):
    torch.manual_seed(7)
    target = torch.arange(n_tgt).repeat_interleave(n_src)
    source = torch.arange(n_src).repeat(n_tgt)
    area_source = torch.full((n_src,), 1 / n_src)
    area_target = torch.full((n_tgt,), 1 / n_tgt)
    operator = SparseOperator.from_weight(
        source, target, torch.full((n_src * n_tgt,), 1 / n_src, dtype=torch.float64),
        area_source, area_target,
    )
    source_xyz = torch.nn.functional.normalize(torch.randn(n_src, 3), dim=1)
    target_xyz = torch.nn.functional.normalize(torch.randn(n_tgt, 3), dim=1)
    def neighbors(n):
        k = min(3, n); index = torch.stack([torch.roll(torch.arange(n), -i) for i in range(k)], 1)
        return index, torch.full((n, k), 1 / k)
    si, sw = neighbors(n_src); ti, tw = neighbors(n_tgt)
    return PairData(name, torch.randn(n_src * n_tgt, 8), source_xyz, target_xyz,
                    si, sw, ti, tw, operator)


class WorkflowTests(unittest.TestCase):
    def test_typed_config_and_role_disjointness(self):
        config = load_config("_next/configs/progressive.json")
        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.model.train_stage, "high_band")
        self.assertEqual(config.model.prefix_through, "mid_band")
        self.assertEqual(config.model.initialization, "fresh")
        self.assertEqual(config.stages[-1].router_gate_mode, "straight_through")
        roles = [set(config.pair_roles[name]) for name in ("train", "selection", "protected", "external_resolution")]
        self.assertTrue(all(not (left & right) for i, left in enumerate(roles) for right in roles[i + 1:]))

    def test_configured_model_assembly_and_candidate_reconstruction(self):
        config = load_config("_next/configs/progressive.json")
        prefix_config, old_stage_config = config.stages
        prefix = ConservativeCorrectionStage(prefix_config)
        old_stage = ConservativeCorrectionStage(old_stage_config)
        with torch.no_grad():
            prefix.score_mlp.net[-1].weight.normal_(std=0.02)
            old_stage.score_mlp.net[-1].weight.normal_(std=0.02)

        def item(stage):
            state = {name: value.detach().clone() for name, value in stage.state_dict().items()}
            return {
                "config": stage.config.to_dict(),
                "state": state,
                "state_sha256": tensor_state_sha256(state),
            }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            torch.save({
                "format": CLEAN_PROGRESSIVE_FORMAT,
                "schema_version": PROGRESSIVE_SCHEMA_VERSION,
                "production": True,
                "conversion_checks": {"equivalence": {"passed": True}},
                "runtime_data": {"edge_features": list(config.features.edge)},
                "stages": [item(prefix), item(old_stage)],
            }, source)
            local = replace(
                config,
                model=ModelConfig(
                    source_checkpoint=str(source), prefix_through="mid_band",
                    train_stage="high_band", initialization="fresh",
                ),
            )
            model, _, initialization = build_training_model(local)
            self.assertEqual([stage.name for stage in model.stages], ["mid_band", "high_band"])
            self.assertEqual(initialization["prefix_stages"], ["mid_band"])
            self.assertIsNone(initialization["initialization_source"])
            self.assertEqual(
                tensor_state_sha256(model.stages[0].state_dict()),
                tensor_state_sha256(prefix.state_dict()),
            )
            self.assertEqual(
                int(torch.count_nonzero(model.stages[-1].score_mlp.net[-1].weight)), 0
            )

            candidate = {
                "format": CLEAN_TRAINING_FORMAT,
                "schema_version": TRAINING_SCHEMA_VERSION,
                "completed": True,
                "selected_identity": False,
                "model_stage_configs": [
                    stage.config.to_dict() for stage in model.stages
                ],
                "best_model_state": model.state_dict(),
                "identity_model_state": model.state_dict(),
                "provenance": {
                    "source_checkpoint": {
                        "path": str(source), "sha256": file_sha256(source)
                    }
                },
            }
            restored, _, restored_source = load_training_checkpoint(candidate)
            self.assertEqual(restored_source, source)
            self.assertEqual(
                tensor_state_sha256(restored.state_dict()),
                tensor_state_sha256(model.state_dict()),
            )

            mismatched_prefix = replace(prefix_config, delta_scale=0.3)
            invalid = replace(local, stages=(mismatched_prefix, old_stage_config))
            with self.assertRaisesRegex(ValueError, "frozen prefix"):
                build_training_model(invalid)

            incompatible_stage = replace(old_stage_config, hidden=64)
            checkpoint_initialized = replace(
                local,
                stages=(prefix_config, incompatible_stage),
                model=ModelConfig(
                    source_checkpoint=str(source), prefix_through="mid_band",
                    train_stage="high_band", initialization="checkpoint",
                ),
            )
            with self.assertRaisesRegex(ValueError, "structural fields"):
                build_training_model(checkpoint_initialized)

    def test_runtime_import_boundary(self):
        for path in Path("_next/remapgnn_next").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(not x.name.startswith(("remapgnn", "scripts")) for x in node.names), path)
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(("remapgnn", "scripts")), path)

    def test_source_keyed_splits(self):
        parts = [set(source_keyed_mode_split("CS-r32", 40, 2407, name))
                 for name in ("train", "val", "audit")]
        self.assertTrue(all(not (left & right) for i, left in enumerate(parts) for right in parts[i + 1:]))
        self.assertEqual(set.union(*parts), set(range(-40, 41)))

    def test_safety_panel_preserves_realizable_frequency(self):
        from remapgnn_next import panels
        frequency = torch.tensor([0.493])
        batch = FieldBatch(
            torch.zeros(1, 2), torch.zeros(1, 3), frequency,
            [(20, 1)], ["safety"], ["source:Y:20:1"],
        )
        pair = mock.Mock()
        pair.n_src = 10242
        with mock.patch.object(
            panels, "_harmonics", return_value=(batch, (10242 / 6) ** 0.5)
        ):
            result = panels._level_harmonics(
                pair, [0.5], 1, "audit", 2407, 99,
                band_lower=1.25, band_upper=1.5,
            )
        self.assertTrue(torch.equal(result.frequency, frequency))

    def test_icod_lower_boundary_safety_stays_outside_target_band(self):
        from remapgnn_next.panels import band_degrees, safety_degree

        source_k = (10242 / 6.0) ** 0.5
        targets = band_degrees(source_k, 1.25, 1.5)
        guard = safety_degree(source_k, 1.25, 1.25, 1.5)

        self.assertEqual(targets[0], 52)
        self.assertEqual(guard, 51)
        self.assertNotIn(guard, targets)
        self.assertLessEqual(guard / source_k, 1.25)

    def test_sparse_and_projection_adjoint(self):
        pair = synthetic_pair()
        values = torch.tensor([1., 2., 3., 4.]); indices = torch.tensor([0, 1, 0, 1])
        self.assertTrue(torch.equal(index_sum(values, indices, 2), torch.tensor([4., 6.])))
        source = torch.randn(pair.n_src)
        self.assertEqual(tuple(apply_operator(pair.fv_operator, source).shape), (pair.n_tgt,))
        x = torch.randn(pair.fv_operator.n_edges, dtype=torch.float64, requires_grad=True)
        y = torch.randn_like(x)
        px, info = project_correction(
            x, pair.src_index, pair.tgt_index, pair.area_tgt, pair.n_src, pair.n_tgt,
            iterations=200, assert_converged=True, return_info=True,
        )
        py = project_correction(y, pair.src_index, pair.tgt_index, pair.area_tgt,
                                pair.n_src, pair.n_tgt, iterations=200)
        row, column = correction_residuals(px, pair.src_index, pair.tgt_index,
                                           pair.area_tgt, pair.n_src, pair.n_tgt)
        self.assertLess(float(row.abs().max()), 1e-8); self.assertLess(float(column.abs().max()), 1e-10)
        self.assertTrue(torch.allclose(torch.dot(px, y), torch.dot(x, py), atol=1e-10, rtol=1e-10))
        (px * y).sum().backward()
        self.assertTrue(torch.allclose(x.grad, py, atol=1e-10, rtol=1e-10))

    def test_stratification_and_regime_weights(self):
        mask = torch.tensor([True, True, False, False, False])
        target, safety, steps = stratified_orders(mask, 1, 2, 17, "cpu")
        self.assertEqual(steps, 2)
        for step in range(steps):
            index = stratified_index(target, safety, step, 1, 2, "cpu")
            self.assertEqual(int(mask[index].sum()), 1)
        weights = pair_weights({"a": synthetic_pair(n_src=3, n_tgt=4),
                                "b": synthetic_pair(n_src=4, n_tgt=3)})
        self.assertEqual(weights, {"a": 0.5, "b": 0.5})

    def test_loss_and_phase_freezing(self):
        pair = synthetic_pair(); stage = ConservativeCorrectionStage(
            StageConfig(name="high", band_lower=1.25, band_upper=1.5))
        with torch.no_grad(): stage.score_mlp.net[-1].weight.normal_(std=.03)
        model = ProgressiveRemapper(pair.fv_operator, [stage]); model.set_training_stage(0, "capability")
        source = torch.randn(4, pair.n_src); truth = torch.randn(4, pair.n_tgt)
        _, diagnostic = model(pair, source, gate_modes=["forced_open"])
        class Weights:
            guard_tolerance=.005; fv_guard_tolerance=.02; cvar_fraction=.25
            guard_weight=6.; local_weight=.5; gate_teacher_weight=.1
            safety_gate_weight=.05; correction_weight=1e-5
        loss, log = progressive_loss(diagnostic.stages[0], diagnostic.fv_output,
                                     diagnostic.fv_output, truth,
                                     torch.tensor([True, True, False, False]), pair.area_tgt,
                                     Weights, train_router=False)
        self.assertTrue(torch.isfinite(loss)); self.assertIn("guard_cvar", log)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in stage.corrector_parameters()))
        self.assertTrue(all(p.grad is None for p in stage.router_parameters()))

    def test_forced_rejection_and_identity_floor(self):
        pair = synthetic_pair(); stage = ConservativeCorrectionStage(
            StageConfig(name="high", band_lower=1.25, band_upper=1.5))
        model = ProgressiveRemapper(pair.fv_operator, [stage]); source = torch.randn(2, pair.n_src)
        output, diagnostic = model(pair, source, gate_modes=["forced_closed"])
        self.assertTrue(torch.equal(output, diagnostic.fv_output))
        self.assertTrue(identity_floor_selection(.89, 1., .1))
        self.assertFalse(identity_floor_selection(.91, 1., .1))

    def test_affine_and_rotation_invariance(self):
        pair = synthetic_pair(); stage = ConservativeCorrectionStage(
            StageConfig(name="high", band_lower=1.25, band_upper=1.5))
        with torch.no_grad():
            stage.score_mlp.net[-1].weight.normal_(std=.03)
            stage.score_mlp.net[-1].bias.normal_(std=.01)
        model = ProgressiveRemapper(pair.fv_operator, [stage]).eval(); source = torch.randn(2, pair.n_src)
        base, _ = model(pair, source, gate_modes=["forced_open"])
        for scale, offset in ((-2.5, .3), (1e-9, 0.)):
            transformed, _ = model(pair, scale * source + offset, gate_modes=["forced_open"])
            self.assertTrue(torch.allclose(transformed, scale * base + offset, atol=2e-6, rtol=2e-5))
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        rotated = replace(pair, src_xyz=pair.src_xyz @ rotation, tgt_xyz=pair.tgt_xyz @ rotation)
        actual, _ = model(rotated, source, gate_modes=["forced_open"])
        self.assertTrue(torch.allclose(actual, base, atol=2e-6, rtol=2e-6))

    def test_cvar(self):
        self.assertEqual(float(cvar(torch.tensor([1., 2., 3., 4.]), .5)), 3.5)


if __name__ == "__main__":
    unittest.main()
