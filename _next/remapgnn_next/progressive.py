from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn

from .config import StageConfig
from .constraints import correction_residuals, project_correction
from .geometry import (
    ENHANCED_GEOMETRY_FEATURES, TRANSFER_GEOMETRY_FEATURES, MLP,
    area_centered_normalize, graph_features, intrinsic_geometry_features,
    smooth,
)
from .sparse import apply_edge_weights, apply_operator, edge_sum_fields, index_sum, row_normalized_reference
from .types import PairData, ProgressiveDiagnostics, SparseOperator, StageDiagnostics

"""Model:
    ProgressiveRemapper chains a baseline FV operator and any number of correction stages.
    ConservativeCorrectionStage predicts an edge correction, projects it to obey the
        constraints, and uses a router/gate to decide whether to apply it."""


class ConservativeCorrectionStage(nn.Module):
    """One generic invariant correction stage in an ordered remapper.

    The original magnitude-only layout has 13 correction features. The
    directional layout adds one sign-even alignment feature. Both layouts use
    8 global-router and 24 local-router values.
    """

    correction_feature_dim = 13
    global_router_feature_dim = 8
    local_router_feature_dim = 24
    graph_feature_dim = 4
    geometry_derived_dim = 8

    def __init__(self, config: StageConfig):
        super().__init__()
        self.config = config
        self.correction_feature_dim = (
            14 if getattr(config, "feature_layout", "magnitude_v1") == "directional_v2"
            else 13
        )
        enhanced_geometry = (
            getattr(config, "geometry_layout", "intrinsic_v1") == "bilinear_v2"
        )
        selected_geometry = (
            TRANSFER_GEOMETRY_FEATURES
            if getattr(config, "geometry_feature_set", "all") == "transfer"
            else ENHANCED_GEOMETRY_FEATURES
        )
        self.geometry_feature_indices = tuple(
            ENHANCED_GEOMETRY_FEATURES.index(name)
            for name in selected_geometry
        ) if enhanced_geometry else tuple(range(8))
        self.geometry_derived_dim = (
            len(self.geometry_feature_indices)
        )
        if enhanced_geometry:
            self.register_buffer(
                "geometry_mean", torch.zeros(self.geometry_derived_dim)
            )
            self.register_buffer(
                "geometry_std", torch.ones(self.geometry_derived_dim)
            )
        geometry_input = config.edge_dim + self.geometry_derived_dim
        self.geom_encoder = MLP(geometry_input, config.geometry_hidden, config.geometry_hidden, depth=3)
        message_input = config.geometry_hidden + self.correction_feature_dim
        self.message_mlp = MLP(message_input, config.hidden, config.hidden, depth=2)
        alternating = bool(
            getattr(config, "alternating_context_passes", 0)
        )
        self.context_refine_mlp = (
            MLP(
                message_input + 2 * config.hidden,
                config.hidden, config.hidden, depth=2,
            )
            if alternating else None
        )
        score_input = message_input + config.hidden * (
            2 if (
                getattr(config, "source_context_passes", 0) or alternating
            ) else 1
        )
        self.score_mlp = MLP(score_input, config.hidden, 1, depth=3, final_zero=True)
        self.field_scale_mlp = (
            MLP(
                self.global_router_feature_dim, config.hidden, 1,
                depth=2, final_zero=True,
            )
            if getattr(config, "amplitude_mode", "fixed") == "learned_field"
            else None
        )
        self.field_gate_mlp = MLP(
            self.global_router_feature_dim, config.router_hidden, 1, depth=3, final_zero=True
        )
        self.local_gate_mlp = MLP(
            self.local_router_feature_dim, config.router_hidden, 1, depth=3, final_zero=True
        )
        self._training_phase = "capability"
        self.set_training_phase("capability")

    @property
    def name(self):
        return self.config.name

    @property
    def training_phase(self):
        return self._training_phase

    def corrector_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.geom_encoder, self.message_mlp, self.context_refine_mlp,
            self.score_mlp,
        ):
            if module is None:
                continue
            yield from module.parameters()
        if self.field_scale_mlp is not None:
            yield from self.field_scale_mlp.parameters()

    def router_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.field_gate_mlp.parameters()
        if getattr(self.config, "router_scope", "global_local") == "global_local":
            yield from self.local_gate_mlp.parameters()

    def set_training_phase(self, phase: str):
        if phase not in {"capability", "router", "frozen"}:
            raise ValueError("phase must be capability, router, or frozen")
        for parameter in self.corrector_parameters():
            parameter.requires_grad_(phase == "capability")
        router_ids = {id(parameter) for parameter in self.router_parameters()}
        for module in (self.field_gate_mlp, self.local_gate_mlp):
            for parameter in module.parameters():
                parameter.requires_grad_(phase == "router" and id(parameter) in router_ids)
        self._training_phase = phase
        self.train(self.training)

    def set_geometry_normalization(self, normalization):
        if getattr(self.config, "geometry_layout", "intrinsic_v1") != "bilinear_v2":
            if normalization is not None:
                raise ValueError("intrinsic geometry does not accept normalization")
            return
        mean = torch.as_tensor(normalization["mean"], dtype=torch.float32)
        std = torch.as_tensor(normalization["std"], dtype=torch.float32)
        expected = (self.geometry_derived_dim,)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError("enhanced geometry normalization has wrong dimension")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(std).all()
        ) or bool((std <= 0).any()):
            raise ValueError("enhanced geometry normalization is invalid")
        self.geometry_mean.copy_(mean)
        self.geometry_std.copy_(std)

    def train(self, mode=True):
        super().train(mode)
        if self._training_phase != "capability":
            for module in (
                self.geom_encoder, self.message_mlp, self.context_refine_mlp,
                self.score_mlp,
            ):
                if module is None:
                    continue
                module.eval()
            if self.field_scale_mlp is not None:
                self.field_scale_mlp.eval()
        if self._training_phase != "router":
            self.field_gate_mlp.eval()
            self.local_gate_mlp.eval()
        elif getattr(self.config, "router_scope", "global_local") == "global":
            self.local_gate_mlp.eval()
        return self

    def _chunks(self, edges):
        size = edges if self.config.edge_chunk <= 0 else self.config.edge_chunk
        for start in range(0, edges, size):
            yield slice(start, min(start + size, edges))

    @staticmethod
    def _hard_gate(probability, low, high):
        return ((probability - float(low)) / float(high - low)).clamp(0.0, 1.0)

    def _route(self, probability, mode, low, high):
        if mode == "forced_open":
            return torch.ones_like(probability)
        if mode == "forced_closed":
            return torch.zeros_like(probability)
        if mode == "soft":
            return probability
        hard = self._hard_gate(probability, low, high)
        if mode == "hard":
            return hard
        if mode == "straight_through":
            return hard.detach() + probability - probability.detach()
        raise ValueError(f"unknown gate mode {mode!r}")

    def _router_values(self, pair, source_state, prefix_state, reference, mode):
        source_graph, source_global, _, _ = graph_features(
            source_state, pair.src_neighbor_index, pair.src_neighbor_weight,
            pair.area_src, self.config.gate_feature_epsilon,
        )
        target_graph, target_global, _, _ = graph_features(
            prefix_state, pair.tgt_neighbor_index, pair.tgt_neighbor_weight,
            pair.area_tgt, self.config.gate_feature_epsilon,
        )
        global_features = torch.cat((source_global, target_global), dim=1)
        field_probability = torch.sigmoid(
            self.field_gate_mlp(global_features).squeeze(-1)
        )
        field_gate = self._route(
            field_probability, mode,
            self.config.field_gate_low, self.config.field_gate_high,
        )
        if getattr(self.config, "router_scope", "global_local") == "global":
            local_probability = torch.ones(
                source_state.shape[0], pair.n_tgt,
                dtype=source_state.dtype, device=source_state.device,
            )
            return (
                field_gate, local_probability, field_probability,
                local_probability, global_features,
            )
        edge_graph = source_graph[:, pair.src_index, :]
        weighted = reference.view(1, -1, 1)
        local_mean = edge_sum_fields(edge_graph * weighted, pair.tgt_index, pair.n_tgt)
        local_rms = edge_sum_fields(
            edge_graph.square() * weighted, pair.tgt_index, pair.n_tgt
        ).clamp_min(0.0).sqrt()
        local_max = edge_graph.new_full(
            (source_state.shape[0], pair.n_tgt, self.graph_feature_dim), -1.0e30
        )
        local_max.index_reduce_(1, pair.tgt_index, edge_graph, "amax", include_self=False)
        local_max = torch.where(local_max <= -1.0e29, torch.zeros_like(local_max), local_max)
        local_features = torch.cat(
            (
                local_mean, local_rms, local_max, target_graph,
                global_features[:, None, :].expand(-1, pair.n_tgt, -1),
            ),
            dim=2,
        )
        local_probability = torch.sigmoid(
            self.local_gate_mlp(local_features).squeeze(-1)
        )
        local_gate = self._route(
            local_probability, mode,
            self.config.local_gate_low, self.config.local_gate_high,
        )
        return (
            field_gate, local_gate, field_probability, local_probability,
            global_features,
        )

    def forward(self, pair: PairData, raw_source, fv_output, prefix_output, *, gate_mode=None):
        if raw_source.ndim == 1:
            raw_source = raw_source.unsqueeze(0)
        fields, n_src = raw_source.shape
        if n_src != pair.n_src:
            raise ValueError("source field does not match PairData")
        if raw_source.dtype != torch.float32:
            raise TypeError("progressive source fields must be float32")
        if not bool(torch.isfinite(raw_source).all()):
            raise ValueError("progressive source field contains non-finite values")
        if pair.edge_features.shape[1] != self.config.edge_dim:
            raise ValueError("edge feature dimension does not match stage configuration")
        gate_mode = self.config.deployment_gate_mode if gate_mode is None else gate_mode
        source_state, mean, scale = area_centered_normalize(raw_source, pair.area_src)
        prefix_state = (prefix_output - mean) / scale
        fv_state = (fv_output - mean) / scale
        reference = (
            pair.correction_reference.to(raw_source.dtype)
            if pair.correction_reference is not None
            else row_normalized_reference(
                pair.fv_operator.weight.to(raw_source.dtype), pair.tgt_index,
                pair.n_tgt, self.config.reference_floor,
            )
        )
        (
            field_gate, local_gate, field_probability, local_probability,
            global_router_features,
        ) = self._router_values(
            pair, source_state, prefix_state, reference, gate_mode
        )
        reference_view = reference.view(1, -1)
        source_edge = source_state[:, pair.src_index]
        target_mean = edge_sum_fields(
            source_edge * reference_view, pair.tgt_index, pair.n_tgt
        )
        centered = source_edge - target_mean[:, pair.tgt_index]
        target_std = edge_sum_fields(
            centered.square() * reference_view, pair.tgt_index, pair.n_tgt
        ).clamp_min(self.config.epsilon).sqrt()
        source_smoothed = smooth(source_state, pair.src_neighbor_index, pair.src_neighbor_weight)
        source_highpass = (source_state - source_smoothed).abs()
        target_smoothed = smooth(prefix_state, pair.tgt_neighbor_index, pair.tgt_neighbor_weight)
        target_second = smooth(target_smoothed, pair.tgt_neighbor_index, pair.tgt_neighbor_weight)
        target_highpass = (prefix_state - target_smoothed).abs()
        target_curvature = (prefix_state - 2.0 * target_smoothed + target_second).abs()
        prefix_disagreement = (prefix_state - target_mean).abs()
        prefix_update = (prefix_state - fv_state).abs()

        source_xyz = pair.src_xyz[pair.src_index]
        target_xyz = pair.tgt_xyz[pair.tgt_index]
        tangent = source_xyz - (source_xyz * target_xyz).sum(dim=1, keepdim=True) * target_xyz
        target_feature_area = pair.area_tgt.to(raw_source.dtype)
        tangent_scaled = tangent / target_feature_area[
            pair.tgt_index
        ].clamp_min(1.0e-20).sqrt().view(-1, 1)
        if getattr(self.config, "gradient_layout", "scalar_v1") == "covariance_v2":
            if pair.gradient_coefficient is None:
                raise ValueError(
                    "covariance gradient requires enhanced pair coefficients"
                )
            gradient = edge_sum_fields(
                centered.unsqueeze(-1)
                * pair.gradient_coefficient.to(raw_source.dtype).view(1, -1, 3),
                pair.tgt_index, pair.n_tgt,
            )
        else:
            gradient_denominator = index_sum(
                reference * tangent_scaled.square().sum(dim=1),
                pair.tgt_index, pair.n_tgt,
            ).clamp_min(self.config.epsilon)
            gradient_numerator = edge_sum_fields(
                centered.unsqueeze(-1) * reference.view(1, -1, 1)
                * tangent_scaled.view(1, -1, 3), pair.tgt_index, pair.n_tgt,
            )
            gradient = gradient_numerator / gradient_denominator.view(
                1, pair.n_tgt, 1
            )
        gradient_magnitude = gradient.norm(dim=2)
        directional = (
            gradient[:, pair.tgt_index, :] * tangent_scaled.view(1, -1, 3)
        ).sum(dim=2)
        if getattr(self.config, "geometry_layout", "intrinsic_v1") == "bilinear_v2":
            if pair.correction_geometry is None:
                raise ValueError(
                    "bilinear_v2 geometry requires enhanced pair features"
                )
            derived = (
                pair.correction_geometry[:, self.geometry_feature_indices]
                - self.geometry_mean
            ) / self.geometry_std
        else:
            derived = intrinsic_geometry_features(
                pair, reference, self.config.epsilon
            )
        geometry = self.geom_encoder(torch.cat((pair.edge_features, derived), dim=1))
        context = raw_source.new_zeros((fields, pair.n_tgt, self.config.hidden))

        def dynamic(edge_slice):
            target = pair.tgt_index[edge_slice]
            values = [
                    source_edge[:, edge_slice].abs(), centered[:, edge_slice].abs(),
                    (centered[:, edge_slice] / target_std[:, target]).abs(),
                    target_mean[:, target].abs(), target_std[:, target],
                    prefix_state[:, target].abs(), target_highpass[:, target],
                    prefix_disagreement[:, target], prefix_update[:, target],
                    directional[:, edge_slice].abs(), gradient_magnitude[:, target],
                    source_highpass[:, pair.src_index[edge_slice]], target_curvature[:, target],
            ]
            if getattr(self.config, "feature_layout", "magnitude_v1") == "directional_v2":
                values.append(
                    (centered[:, edge_slice] / target_std[:, target])
                    * directional[:, edge_slice]
                )
            return torch.stack(values, dim=2)

        for edge_slice in self._chunks(pair.fv_operator.n_edges):
            target = pair.tgt_index[edge_slice]
            code = geometry[edge_slice].unsqueeze(0).expand(fields, -1, -1)
            message = self.message_mlp(torch.cat((code, dynamic(edge_slice)), dim=2))
            context.index_add_(
                1, target, message * reference[edge_slice].view(1, -1, 1)
            )
        source_context = None
        alternating = bool(
            getattr(self.config, "alternating_context_passes", 0)
        )
        if getattr(self.config, "source_context_passes", 0) or alternating:
            column_weight = (
                reference
                * pair.area_tgt.to(reference.dtype)[pair.tgt_index]
            )
            column_denominator = index_sum(
                column_weight, pair.src_index, pair.n_src
            ).clamp_min(self.config.epsilon)
            column_weight = column_weight / column_denominator[pair.src_index]
            source_context = edge_sum_fields(
                context[:, pair.tgt_index, :]
                * column_weight.view(1, -1, 1),
                pair.src_index,
                pair.n_src,
            )
        target_context = context
        if alternating:
            target_context = raw_source.new_zeros(
                (fields, pair.n_tgt, self.config.hidden)
            )
            for edge_slice in self._chunks(pair.fv_operator.n_edges):
                target = pair.tgt_index[edge_slice]
                source = pair.src_index[edge_slice]
                code = geometry[edge_slice].unsqueeze(0).expand(fields, -1, -1)
                refined = self.context_refine_mlp(torch.cat(
                    (
                        code, dynamic(edge_slice), context[:, target, :],
                        source_context[:, source, :],
                    ),
                    dim=2,
                ))
                target_context.index_add_(
                    1, target,
                    refined * reference[edge_slice].view(1, -1, 1),
                )
        scores = []
        for edge_slice in self._chunks(pair.fv_operator.n_edges):
            target = pair.tgt_index[edge_slice]
            code = geometry[edge_slice].unsqueeze(0).expand(fields, -1, -1)
            score_values = [
                code, dynamic(edge_slice), target_context[:, target, :]
            ]
            if source_context is not None:
                score_values.append(source_context[:, pair.src_index[edge_slice], :])
            scores.append(torch.tanh(
                self.score_mlp(torch.cat(score_values, dim=2)).squeeze(-1)
            ))
        score = torch.cat(scores, dim=1)
        if getattr(self.config, "amplitude_reference", "aggregation") == "uniform":
            amplitude_reference = torch.ones_like(reference)
            amplitude_degree = index_sum(
                amplitude_reference, pair.tgt_index, pair.n_tgt
            ).clamp_min(1.0)
            amplitude_reference = (
                amplitude_reference / amplitude_degree[pair.tgt_index]
            )
        else:
            amplitude_reference = reference
        amplitude_view = amplitude_reference.view(1, -1)
        score_mean = edge_sum_fields(
            score * amplitude_view, pair.tgt_index, pair.n_tgt
        )
        if self.field_scale_mlp is None:
            correction_scale = score.new_full(
                (fields,), float(self.config.delta_scale)
            )
        else:
            lower = float(getattr(self.config, "minimum_delta_scale", 0.0))
            upper = float(self.config.delta_scale)
            probability = torch.sigmoid(
                self.field_scale_mlp(global_router_features).squeeze(-1)
            )
            correction_scale = lower + (upper - lower) * probability
        raw_delta = (
            correction_scale.view(-1, 1) * field_gate.view(-1, 1)
            * local_gate[:, pair.tgt_index] * amplitude_view
            * (score - score_mean[:, pair.tgt_index])
        ).to(torch.float64)
        projected_delta = project_correction(
            raw_delta, pair.src_index, pair.tgt_index, pair.area_tgt,
            pair.n_src, pair.n_tgt, iterations=self.config.projection_iterations,
        )
        projection_norm_ratio = (
            projected_delta.norm(dim=1)
            / raw_delta.norm(dim=1).clamp_min(1.0e-30)
        )
        score_saturation = (score.abs() >= 0.95).to(score.dtype).mean(dim=1)
        # A global solve can redistribute an all-zero input at roundoff level.
        # Restore the exact identity floor after projection.
        accepted = field_gate != 0.0
        hard_delta = torch.where(
            accepted.view(-1, 1), projected_delta, torch.zeros_like(projected_delta)
        )
        delta = (
            hard_delta.detach() + projected_delta - projected_delta.detach()
            if gate_mode == "straight_through" else hard_delta
        )
        proposed_correction = edge_sum_fields(
            delta * raw_source.to(delta.dtype)[:, pair.src_index], pair.tgt_index, pair.n_tgt
        ).to(raw_source.dtype)
        hard_correction = torch.where(
            accepted.view(-1, 1), proposed_correction, torch.zeros_like(proposed_correction)
        )
        correction = (
            hard_correction.detach() + proposed_correction - proposed_correction.detach()
            if gate_mode == "straight_through" else hard_correction
        )
        output = prefix_output + correction
        row, column = correction_residuals(
            delta, pair.src_index, pair.tgt_index, pair.area_tgt, pair.n_src, pair.n_tgt
        )
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(delta).all()):
            raise RuntimeError(f"{self.name}: non-finite correction")
        if float(row.abs().max()) > self.config.projection_row_tolerance:
            raise RuntimeError(f"{self.name}: correction row constraint failed")
        if float(column.abs().max()) > self.config.projection_column_tolerance:
            raise RuntimeError(f"{self.name}: correction column constraint failed")
        return output, StageDiagnostics(
            self.name, output, delta, row, column, field_gate, local_gate,
            field_probability, local_probability,
            proposal_output=output if gate_mode == "forced_open" else None,
            correction_scale=correction_scale,
            score_saturation=score_saturation,
            projection_norm_ratio=projection_norm_ratio,
        )


class ProgressiveRemapper(nn.Module):
    def __init__(self, base_operator: SparseOperator | None, stages: Sequence[ConservativeCorrectionStage]):
        super().__init__()
        self.base_operator = base_operator
        self.stages = nn.ModuleList(stages)

    def set_training_stage(self, index: int, phase: str):
        if index < 0 or index >= len(self.stages):
            raise IndexError("training stage outside model")
        for position, stage in enumerate(self.stages):
            stage.set_training_phase(phase if position == index else "frozen")

    def forward(self, pair: PairData, source_field, *, gate_modes=None, return_diagnostics=True):
        operator = self.base_operator or pair.baseline_operator
        if operator.n_src != pair.n_src or operator.n_tgt != pair.n_tgt:
            raise ValueError("base operator dimensions do not match pair")
        source = source_field.unsqueeze(0) if source_field.ndim == 1 else source_field
        if operator.weight.device != source.device:
            operator = operator.to(source.device)
        if pair.metadata.get("baseline_kind") == "conservative_esmf_bilinear":
            from .bilinear import apply_conservative_bilinear
            fv_output, _, _ = apply_conservative_bilinear(pair, source)
        else:
            fv_output = apply_operator(operator, source)
        current = fv_output
        diagnostics = []
        if gate_modes is None:
            gate_modes = [None] * len(self.stages)
        if len(gate_modes) != len(self.stages):
            raise ValueError("gate_modes must contain one entry per stage")
        for stage, mode in zip(self.stages, gate_modes):
            current, diagnostic = stage(pair, source, fv_output, current, gate_mode=mode)
            diagnostics.append(diagnostic)
        complete = ProgressiveDiagnostics(
            fv_output=fv_output,
            stage_outputs=tuple(x.output for x in diagnostics),
            stages=tuple(diagnostics),
        )
        return (current, complete) if return_diagnostics else current
