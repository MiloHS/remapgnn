from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import (
    BILINEAR_TRAINING_FORMAT, BILINEAR_TRAINING_SCHEMA_VERSION,
    CLEAN_TRAINING_FORMAT, TRAINING_SCHEMA_VERSION,
)
from .evaluation import area_relative_l2, safe_ratio
from .panels import build_panel
from .provenance import object_sha256, tensor_state_sha256, verify_run_manifest
from .sparse import apply_operator


CHECKPOINT_FORMAT = CLEAN_TRAINING_FORMAT
CHECKPOINT_SCHEMA = TRAINING_SCHEMA_VERSION


def _panel(config, pair, stage, **kwargs):
    if getattr(config, "schema_version", None) == 5:
        from .band_panels import build_band_panel
        return build_band_panel(
            config, pair, stage_config=stage, **kwargs
        )
    return build_panel(config, pair, stage_config=stage, **kwargs)


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    best_epoch: int
    best_score: float
    selected_identity: bool


def set_seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def identity_floor_selection(candidate_score, prefix_score, minimum_improvement=0.0):
    if not all(np.isfinite(x) for x in (candidate_score, prefix_score, minimum_improvement)):
        return False
    return float(candidate_score) < float(prefix_score) - float(minimum_improvement)


def early_stopping_state(scores, minimum_delta):
    """Return significant best score and consecutive stale evaluations."""
    best, stale = float("inf"), 0
    for value in scores:
        score = float(value)
        if np.isfinite(score) and score < best - float(minimum_delta):
            best, stale = score, 0
        else:
            stale += 1
    return best, stale


def np2_gap_closed(prefix_error, model_error, np2_error, tolerance):
    gap = float(prefix_error) - float(np2_error)
    if not np.isfinite(gap) or gap <= float(tolerance):
        return None
    return float((float(prefix_error) - float(model_error)) / gap)


def parameter_snapshot(parameters: Iterable[torch.nn.Parameter]):
    return [value.detach().cpu().clone() for value in parameters]


def assert_unchanged(parameters, snapshot, *, context="frozen parameters"):
    current = list(parameters)
    if len(current) != len(snapshot) or any(
        not torch.equal(value.detach().cpu(), saved) for value, saved in zip(current, snapshot)
    ):
        raise RuntimeError(f"{context} changed across an optimizer step")


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def cvar(values, fraction):
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.new_zeros(())
    count = max(1, int(np.ceil(float(fraction) * flat.numel())))
    return torch.topk(flat, count).values.mean()


def normalized_mse(prediction, truth, area):
    area = area.to(prediction.dtype).view(1, -1)
    return (area * (prediction - truth).square()).sum(1) / \
        (area * truth.square()).sum(1).clamp_min(1.0e-20)


def component_gradient_diagnostics(components, parameters):
    """Gradient norms/cosines without changing optimizer gradients."""
    parameters = list(parameters)
    vectors = {}
    for name, value in components.items():
        if not value.requires_grad:
            vectors[name] = torch.zeros(
                sum(parameter.numel() for parameter in parameters),
                device=parameters[0].device,
            )
            continue
        gradients = torch.autograd.grad(
            value, parameters, retain_graph=True, allow_unused=True,
        )
        vectors[name] = torch.cat([
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None else gradient.reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients)
        ])
    target = vectors["target"]
    target_norm = target.norm()
    result = {}
    for name, vector in vectors.items():
        norm = vector.norm()
        result[f"grad_norm_{name}"] = float(norm.detach())
        result[f"grad_cos_{name}_vs_target"] = (
            1.0 if name == "target" and float(target_norm) > 0
            else float(
                (vector @ target)
                / (norm * target_norm).clamp_min(1.0e-30)
            )
        )
    return result


def benefit_teacher_labels(prefix_output, open_output, truth, area, temperature):
    prefix_error = normalized_mse(prefix_output, truth, area).detach()
    open_error = normalized_mse(open_output, truth, area).detach()
    field_benefit = (prefix_error - open_error) / \
        (prefix_error + open_error).clamp_min(1e-20)
    prefix_point = (prefix_output.detach() - truth).square()
    open_point = (open_output.detach() - truth).square()
    local_benefit = (prefix_point - open_point) / \
        (prefix_point + open_point).clamp_min(1e-20)
    return (
        torch.sigmoid(field_benefit / float(temperature)),
        torch.sigmoid(local_benefit / float(temperature)),
    )


def progressive_loss(
    stage_diagnostic, fv_output, prefix_output, truth, target_mask, area, weights,
    *, train_router, benefit_output=None, low_order_mask=None,
    router_scope="global_local", return_components=False,
):
    current = normalized_mse(stage_diagnostic.output, truth, area)
    prefix = normalized_mse(prefix_output, truth, area).detach()
    fv = normalized_mse(fv_output, truth, area).detach()
    target, safety = target_mask, ~target_mask
    if not bool(target.any()) or not bool(safety.any()):
        raise ValueError("every training batch must include target and safety fields")
    target_loss = current[target].mean()
    ratio_prefix = torch.sqrt(current[safety].clamp_min(0) / prefix[safety].clamp_min(1e-20))
    ratio_fv = torch.sqrt(current[safety].clamp_min(0) / fv[safety].clamp_min(1e-20))
    excess = torch.cat((
        torch.relu(ratio_prefix - (1 + weights.guard_tolerance)).square(),
        torch.relu(ratio_fv - (1 + weights.fv_guard_tolerance)).square(),
    ))
    guard = cvar(excess, weights.cvar_fraction)
    local = cvar(torch.cat((torch.relu(current[safety] - prefix[safety]),
                            torch.relu(current[safety] - fv[safety]))), weights.cvar_fraction)
    correction = stage_diagnostic.delta_weight.square().mean()
    teacher = current.new_zeros(()); safety_gate = current.new_zeros(())
    if train_router:
        if benefit_output is None:
            raise ValueError("router training requires a forced-open benefit output")
        field_label, local_label = benefit_teacher_labels(
            prefix_output, benefit_output, truth, area,
            weights.router_teacher_temperature,
        )
        field_label = field_label.to(stage_diagnostic.field_probability.dtype)
        local_label = local_label.to(stage_diagnostic.local_probability.dtype)
        teacher = F.binary_cross_entropy(stage_diagnostic.field_probability, field_label)
        safety_gate = cvar(
            stage_diagnostic.field_probability[safety], weights.cvar_fraction
        )
        if router_scope == "global_local":
            teacher = teacher + F.binary_cross_entropy(
                stage_diagnostic.local_probability, local_label
            )
            safety_gate = safety_gate + cvar(
                stage_diagnostic.local_probability[safety].mean(1),
                weights.cvar_fraction,
            )
    low_order = current.new_zeros(())
    if low_order_mask is not None and bool(low_order_mask.any()):
        low_order = current[low_order_mask].mean()
    components = {
        "target": target_loss,
        "guard": weights.guard_weight * guard,
        "local": weights.local_weight * local,
        "router_teacher": weights.gate_teacher_weight * teacher,
        "safety_gate": weights.safety_gate_weight * safety_gate,
        "correction": weights.correction_weight * correction,
        "low_order": getattr(weights, "low_order_weight", 0.0) * low_order,
    }
    loss = sum(components.values())
    log = {
        "target_rel": float(current[target].sqrt().mean().detach()),
        "prefix_target_rel": float(prefix[target].sqrt().mean().detach()),
        "safety_worst_prefix_ratio": float(ratio_prefix.max().detach()),
        "safety_worst_fv_ratio": float(ratio_fv.max().detach()),
        "guard_cvar": float(guard.detach()), "local_cvar": float(local.detach()),
        "gate_teacher": float(teacher.detach()), "safety_gate": float(safety_gate.detach()),
        "delta": float(correction.detach()),
        "low_order": float(low_order.detach()),
        **{
            f"loss_{name}": float(value.detach())
            for name, value in components.items()
        },
        "correction_scale": (
            float(stage_diagnostic.correction_scale.mean().detach())
            if stage_diagnostic.correction_scale is not None
            else float("nan")
        ),
        "score_saturation": (
            float(stage_diagnostic.score_saturation.mean().detach())
            if stage_diagnostic.score_saturation is not None else float("nan")
        ),
        "projection_norm_ratio": (
            float(stage_diagnostic.projection_norm_ratio.mean().detach())
            if stage_diagnostic.projection_norm_ratio is not None else float("nan")
        ),
        "target_field_probability": float(stage_diagnostic.field_probability[target].mean().detach()),
        "safety_field_probability": float(stage_diagnostic.field_probability[safety].mean().detach()),
    }
    return (loss, log, components) if return_components else (loss, log)


def stratified_orders(mask, target_batch, safety_batch, seed, device):
    target = torch.where(mask)[0].cpu().numpy(); safety = torch.where(~mask)[0].cpu().numpy()
    if not len(target) or not len(safety):
        raise ValueError("panel needs both target and safety fields")
    rng = np.random.default_rng(int(seed)); rng.shuffle(target); rng.shuffle(safety)
    steps = max(int(np.ceil(len(target) / target_batch)), int(np.ceil(len(safety) / safety_batch)))
    return torch.tensor(target, device=device), torch.tensor(safety, device=device), steps


def stratified_index(target, safety, step, target_batch, safety_batch, device):
    def cyclic(values, start, count):
        return values[(torch.arange(count, device=device) + start) % values.numel()]
    return torch.cat((cyclic(target, step * target_batch, target_batch),
                      cyclic(safety, step * safety_batch, safety_batch)))


def pair_weights(pairs: Mapping[str, object]):
    if len(pairs) == 1:
        return {next(iter(pairs)): 1.0}
    regimes = {"coarse_to_fine": [], "fine_to_coarse": []}
    for name, pair in pairs.items():
        regimes["coarse_to_fine" if pair.n_src < pair.n_tgt else "fine_to_coarse"].append(name)
    if not all(regimes.values()):
        raise ValueError("training pairs must contain both transfer regimes")
    return {name: 0.5 / len(values) for values in regimes.values() for name in values}


def selection_score(
    metrics, selection, audit, *, stage_index=0, capability=False,
):
    numeric = [
        item for value in metrics.values() for item in value.values()
        if isinstance(item, (float, np.floating))
    ]
    if not numeric or any(not np.isfinite(item) for item in numeric):
        return (float("inf"),) * 5
    target = max(value["target_mean_ratio_vs_prefix"] for value in metrics.values())
    safety = max(value["safety_worst_ratio_vs_prefix"] for value in metrics.values())
    base = max(value["safety_worst_ratio_vs_fv"] for value in metrics.values())
    prior = max(value["prefix_band_worst_ratio_vs_prefix"] for value in metrics.values())
    safety_tolerance = (
        selection.capability_safety_tolerance
        if capability else selection.safety_tolerance
    )
    score = target + 5 * max(0, safety - (1 + safety_tolerance))
    # At stage zero prefix and base are the same result, and no accepted prior
    # band exists. Penalizing those aliases again triple-counts one field.
    if int(stage_index) > 0:
        score += 5 * max(0, base - (1 + (
            selection.capability_safety_tolerance
            if capability else audit.maximum_fv_regression
        )))
        score += 5 * max(0, prior - (1 + (
            selection.capability_safety_tolerance
            if capability else selection.prior_band_tolerance
        )))
    return float(score), float(target), float(safety), float(base), float(prior)


@torch.no_grad()
def evaluate_selection(
    model, pairs, panels, stage_index, config, gate_mode, device,
    *, capability=False, np2_operators=None,
):
    model.eval(); metrics = {}
    for name, host_pair in pairs.items():
        pair, panel = host_pair.to(device), panels[name].to(device)
        errors, prefix_errors, fv_errors = [], [], []
        np2_errors = [] if np2_operators and name in np2_operators else None
        np2 = (
            np2_operators[name].to(device)
            if np2_errors is not None else None
        )
        gates, locals_, rows, columns = [], [], [], []
        batch_size = config.phases.target_batch
        for start in range(0, panel.source.shape[0], batch_size):
            part = panel.subset(range(start, min(start + batch_size, panel.source.shape[0])))
            modes = [None] * len(model.stages); modes[stage_index] = gate_mode
            _, diagnostic = model(pair, part.source, gate_modes=modes, return_diagnostics=True)
            stage = diagnostic.stages[stage_index]
            prefix = diagnostic.fv_output if stage_index == 0 else diagnostic.stage_outputs[stage_index - 1]
            errors.extend(area_relative_l2(stage.output, part.truth, pair.area_tgt).cpu().tolist())
            prefix_errors.extend(area_relative_l2(prefix, part.truth, pair.area_tgt).cpu().tolist())
            fv_errors.extend(area_relative_l2(diagnostic.fv_output, part.truth, pair.area_tgt).cpu().tolist())
            if np2 is not None:
                np2_errors.extend(
                    area_relative_l2(
                        apply_operator(np2, part.source), part.truth,
                        pair.area_tgt,
                    ).cpu().tolist()
                )
            gates.extend(stage.field_gate.cpu().tolist()); locals_.extend(stage.local_gate.mean(1).cpu().tolist())
            rows.append(float(stage.row_residual.abs().max())); columns.append(float(stage.column_residual.abs().max()))
        current, prefix, fv = map(np.asarray, (errors, prefix_errors, fv_errors))
        target = panel.is_target.cpu().numpy(); safety = ~target
        ratio_prefix = safe_ratio(current, prefix, config.audit.zero_error_tolerance)
        ratio_fv = safe_ratio(current, fv, config.audit.zero_error_tolerance)
        if getattr(config, "schema_version", None) == 5 and panel.bands:
            previous_names = {
                stage.config.target_band for stage in model.stages[:stage_index]
            }
            prior_mask = safety & np.asarray([
                band in previous_names for band in panel.bands
            ])
        else:
            frequency = panel.frequency.cpu().numpy()
            previous = (
                model.stages[stage_index - 1].config
                if stage_index
                else model.stages[stage_index].config
            )
            prior_mask = safety & np.isfinite(frequency) & (
                frequency > previous.band_lower
            ) & (frequency <= previous.band_upper)
        if not prior_mask.any():
            prior_mask = np.zeros_like(safety) if stage_index == 0 else safety
        gates, locals_ = np.asarray(gates), np.asarray(locals_)
        safety_indices = np.where(safety)[0]
        target_indices = np.where(target)[0]
        worst_safety_index = int(
            safety_indices[np.argmax(ratio_prefix[safety])]
        )
        worst_target_index = int(
            target_indices[np.argmax(ratio_prefix[target])]
        )
        metrics[name] = {
            "target_mean_ratio_vs_prefix": float(ratio_prefix[target].mean()),
            "target_worst_ratio_vs_prefix": float(ratio_prefix[target].max()),
            "target_mean_ratio_vs_fv": float(ratio_fv[target].mean()),
            "safety_worst_ratio_vs_prefix": float(ratio_prefix[safety].max()),
            "safety_worst_ratio_vs_fv": float(ratio_fv[safety].max()),
            "prefix_band_worst_ratio_vs_prefix": (
                1.0 if not prior_mask.any()
                else float(ratio_prefix[prior_mask].max())
            ),
            "worst_safety_source_key": panel.source_keys[worst_safety_index],
            "worst_safety_family": panel.families[worst_safety_index],
            "worst_target_source_key": panel.source_keys[worst_target_index],
            "worst_target_family": panel.families[worst_target_index],
            "target_model_rel": float(current[target].mean()), "target_prefix_rel": float(prefix[target].mean()),
            "target_fv_rel": float(fv[target].mean()), "target_field_gate": float(gates[target].mean()),
            "safety_field_gate": float(gates[safety].mean()), "target_local_gate": float(locals_[target].mean()),
            "safety_local_gate": float(locals_[safety].mean()), "row_residual_max": max(rows),
            "column_residual_max": max(columns),
        }
        if np2_errors is not None:
            np2_values = np.asarray(np2_errors)
            prefix_mean = float(prefix[target].mean())
            model_mean = float(current[target].mean())
            np2_mean = float(np2_values[target].mean())
            gap = prefix_mean - np2_mean
            metrics[name].update(
                target_np2_rel=np2_mean,
                target_model_over_np2=float(safe_ratio(
                    model_mean, np2_mean,
                    config.audit.zero_error_tolerance,
                )),
                bilinear_to_np2_gap=gap,
                bilinear_to_np2_gap_closed=np2_gap_closed(
                    prefix_mean, model_mean, np2_mean,
                    config.audit.zero_error_tolerance,
                ),
            )
    return (*selection_score(
        metrics, config.selection, config.audit,
        stage_index=stage_index, capability=capability,
    ), metrics)


def _atomic_torch_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(value, temporary); temporary.replace(path)


def _write_history(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


class SequentialTrainer:
    """Production capability-then-router trainer for one ordered stage."""

    def __init__(self, model, stage_index, *, config=None, train_pairs=None, selection_pairs=None,
                 source_checkpoint=None, model_initialization=None, output=None,
                 history_path=None, device="cpu", run_manifest=None,
                 capability_source=None, selection_operators=None):
        self.model = model; self.stage_index = int(stage_index); self.config = config
        self.train_pairs = train_pairs or {}; self.selection_pairs = selection_pairs or {}
        self.source_checkpoint = None if source_checkpoint is None else Path(source_checkpoint)
        self.model_initialization = (
            {} if model_initialization is None else copy.deepcopy(model_initialization)
        )
        self.output = Path(output) if output else (config.paths.checkpoint_path if config else None)
        self.history_path = Path(history_path) if history_path else (config.paths.history_path if config else None)
        self.device = torch.device(device)
        self.run_manifest = copy.deepcopy(run_manifest)
        self.capability_source = (
            None if capability_source is None else copy.deepcopy(capability_source)
        )
        self.selection_operators = selection_operators or {}
        if not 0 <= self.stage_index < len(model.stages): raise IndexError("stage_index outside model")

    @property
    def stage(self): return self.model.stages[self.stage_index]

    def _pack(self, state, optimizer, *, phase, epoch, completed, history, smoke):
        bilinear = getattr(self.config, "schema_version", None) == 5
        corrector_prefixes = tuple(
            f"stages.{self.stage_index}.{name}"
            for name in (
                "geom_encoder.", "message_mlp.", "context_refine_mlp.",
                "score_mlp.",
                "field_scale_mlp.",
            )
        )
        corrector_hash = lambda model_state: tensor_state_sha256({
            name: value for name, value in model_state.items()
            if name.startswith(corrector_prefixes)
        })
        pack = {
            "format": (
                BILINEAR_TRAINING_FORMAT if bilinear else CHECKPOINT_FORMAT
            ),
            "schema_version": (
                BILINEAR_TRAINING_SCHEMA_VERSION if bilinear else CHECKPOINT_SCHEMA
            ),
            "completed": bool(completed), "smoke": bool(smoke), "stage_index": self.stage_index,
            "phase": phase, "epoch": int(epoch), "model_state": cpu_state(self.model),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "identity_model_state": state["identity_state"], "identity_score": state["identity_score"],
            "identity_selection_metrics": copy.deepcopy(state["identity_metrics"]),
            "capability_best_state": state["capability_state"], "capability_best_score": state["capability_score"],
            "capability_best_epoch": state["capability_epoch"], "capability_selected": state["capability_selected"],
            "best_model_state": state["final_state"], "final_best_score": state["final_score"],
            "final_best_epoch": state["final_epoch"], "selected_identity": state["selected_identity"],
            "router_candidate_state": state["router_state"],
            "router_candidate_score": state["router_score"],
            "router_candidate_epoch": state["router_epoch"],
            "capability_corrector_state_sha256": corrector_hash(
                state["capability_state"]
            ),
            "router_candidate_corrector_state_sha256": corrector_hash(
                state["router_state"]
            ),
            "analysis_only": bool(state.get("analysis_only", False)),
            "selection_metrics": copy.deepcopy(state["metrics"]), "corrector_state_sha256": state["corrector_hash"],
            "history": copy.deepcopy(history), "pair_roles": copy.deepcopy(self.config.pair_roles),
            "model_stage_configs": [stage.config.to_dict() for stage in self.model.stages],
            "model_initialization": copy.deepcopy(self.model_initialization),
            "config": self.config.to_dict(), "provenance": copy.deepcopy(self.run_manifest),
            "behavior": {"known_frequency_required": False, "adaptive_stopping": False,
                         "sequential_residual_training": True, "strict_prefix_freezing": True},
        }
        pack["state_sha256"] = {
            name: object_sha256(pack[name]) for name in (
                "model_state", "optimizer_state", "identity_model_state",
                "capability_best_state", "best_model_state",
                "router_candidate_state",
            )
        }
        return pack

    def _validate_resume(self, saved):
        expected_format = (
            BILINEAR_TRAINING_FORMAT
            if getattr(self.config, "schema_version", None) == 5
            else CHECKPOINT_FORMAT
        )
        expected_schema = (
            BILINEAR_TRAINING_SCHEMA_VERSION
            if getattr(self.config, "schema_version", None) == 5
            else CHECKPOINT_SCHEMA
        )
        if saved.get("format") != expected_format or saved.get("schema_version") != expected_schema:
            raise ValueError("resume checkpoint has the wrong clean schema")
        if saved["stage_index"] != self.stage_index: raise ValueError("resume stage differs")
        if saved.get("provenance") != self.run_manifest:
            raise ValueError("run manifest differs from checkpoint")
        verify_run_manifest(self.run_manifest)
        required_hashes = {
            "model_state", "optimizer_state", "identity_model_state",
            "capability_best_state", "best_model_state",
            "router_candidate_state",
        }
        if set(saved.get("state_sha256", {})) != required_hashes:
            raise ValueError("checkpoint has incomplete state hashes")
        for name, expected in saved["state_sha256"].items():
            if object_sha256(saved.get(name)) != expected:
                raise ValueError(f"checkpoint state hash mismatch: {name}")

    def run(self, *, resume=False, smoke=False, capability_only=False):
        if self.config is None or not self.train_pairs or not self.selection_pairs or self.output is None:
            raise ValueError("config, train/selection pairs, and output are required")
        if self.run_manifest is None:
            raise ValueError("an immutable run manifest is required")
        if bool(self.run_manifest.get("smoke")) != bool(smoke):
            raise ValueError("smoke/full mode differs from run manifest")
        verify_run_manifest(self.run_manifest)
        if resume and self.capability_source is not None:
            raise ValueError("--resume cannot be combined with a router capability source")
        if capability_only and self.capability_source is not None:
            raise ValueError("capability-only run cannot start from capability")
        set_seed(self.config.seed); self.model.to(self.device)
        train_names = list(self.train_pairs)[:1] if smoke else list(self.train_pairs)
        selection_names = train_names if smoke else list(self.selection_pairs)
        train_pairs = {name: self.train_pairs[name] for name in train_names}
        selection_pairs = {name: (self.train_pairs.get(name) or self.selection_pairs[name]) for name in selection_names}
        weights = pair_weights(train_pairs)
        selection_panels = {name: _panel(
                                self.config, pair, self.stage.config,
                                split="train" if smoke else "val",
                                epoch=0, smoke=smoke, audit=True)
                            for name, pair in selection_pairs.items()}
        saved = None
        if resume:
            saved = torch.load(self.output, map_location="cpu", weights_only=False); self._validate_resume(saved)
            self.model.load_state_dict(saved["model_state"])
            if saved["completed"]:
                _write_history(self.history_path, saved.get("history", []))
                return saved
        capability_seed = self.capability_source
        if capability_seed is not None:
            self.model.load_state_dict(
                capability_seed["capability_best_state"], strict=True
            )
        identity_state = (
            copy.deepcopy(capability_seed["identity_model_state"])
            if capability_seed is not None
            else copy.deepcopy(saved["identity_model_state"])
            if saved else cpu_state(self.model)
        )
        if capability_seed is not None:
            identity_score = float(capability_seed["identity_score"])
            identity_metrics = copy.deepcopy(
                capability_seed["identity_selection_metrics"]
            )
        elif saved:
            identity_score = float(saved["identity_score"])
            identity_metrics = copy.deepcopy(saved["identity_selection_metrics"])
        else:
            identity_score, *_, identity_metrics = evaluate_selection(
                self.model, selection_pairs, selection_panels, self.stage_index,
                self.config, "forced_closed", self.device, capability=True,
                np2_operators=self.selection_operators,
            )
        history = (
            list(capability_seed.get("history", []))
            + [{
                "phase": "router_branch", "epoch": 0,
                "selection_score": capability_seed["capability_best_score"],
            }]
            if capability_seed is not None
            else list(saved.get("history", [])) if saved
            else [{"phase": "identity", "epoch": 0, "selection_score": identity_score}]
        )
        state = {
            "identity_state": identity_state, "identity_score": identity_score,
            "identity_metrics": identity_metrics,
            "capability_state": copy.deepcopy(saved["capability_best_state"]) if saved else copy.deepcopy(identity_state),
            # Track the best attempted capability independently of admission
            # against the identity floor. This preserves rejected candidates
            # for analysis without allowing them to be promoted.
            "capability_score": float(saved["capability_best_score"]) if saved else float("inf"),
            "capability_epoch": int(saved["capability_best_epoch"]) if saved else 0,
            "capability_selected": bool(saved.get("capability_selected", False)) if saved else False,
            "final_state": copy.deepcopy(saved["best_model_state"]) if saved else copy.deepcopy(identity_state),
            "final_score": float(saved["final_best_score"]) if saved else identity_score,
            "final_epoch": int(saved["final_best_epoch"]) if saved else 0,
            "router_state": copy.deepcopy(saved.get(
                "router_candidate_state", saved["best_model_state"]
            )) if saved else copy.deepcopy(identity_state),
            "router_score": float(saved.get(
                "router_candidate_score", saved["final_best_score"]
            )) if saved else identity_score,
            "router_epoch": int(saved.get(
                "router_candidate_epoch", saved["final_best_epoch"]
            )) if saved else 0,
            "selected_identity": True, "metrics": copy.deepcopy(saved.get("selection_metrics", {})) if saved else {},
            "corrector_hash": "",
            "analysis_only": bool(capability_only),
        }
        if capability_seed is not None:
            state.update(
                capability_state=copy.deepcopy(
                    capability_seed["capability_best_state"]
                ),
                capability_score=float(capability_seed["capability_best_score"]),
                capability_epoch=int(capability_seed["capability_best_epoch"]),
                capability_selected=True,
                final_state=copy.deepcopy(
                    capability_seed["capability_best_state"]
                ),
                final_score=identity_score,
                final_epoch=0,
            )
        frozen = [p for index, stage in enumerate(self.model.stages) for p in stage.parameters() if index != self.stage_index]
        frozen_snapshot = parameter_snapshot(frozen)

        def phase_run(phase, epochs, learning_rate, start_epoch=1, optimizer_state=None):
            self.model.set_training_stage(self.stage_index, phase)
            parameters = list(self.stage.corrector_parameters() if phase == "capability" else self.stage.router_parameters())
            phase_frozen = [parameter for parameter in self.model.parameters() if not parameter.requires_grad]
            phase_frozen_snapshot = parameter_snapshot(phase_frozen)
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=self.config.phases.weight_decay)
            if optimizer_state is not None: optimizer.load_state_dict(optimizer_state)
            gate_mode = self.stage.config.capability_gate_mode if phase == "capability" else self.stage.config.router_gate_mode
            patience = (
                int(self.config.phases.capability_early_stopping_patience)
                if phase == "capability" else 0
            )
            minimum_delta = float(
                self.config.phases.capability_early_stopping_min_delta
            )
            previous_scores = [
                row["selection_score"] for row in history
                if row.get("phase") == phase
                and row.get("selection_score") not in ("", None)
            ]
            early_best, stale_evaluations = early_stopping_state(
                previous_scores, minimum_delta
            )
            last_epoch = start_epoch - 1
            for epoch in range(start_epoch, epochs + 1):
                last_epoch = epoch
                started = time.time(); self.model.train(); panels = {}; orders = {}; steps = 0
                diagnostic_epoch = bool(
                    getattr(self.config.phases, "gradient_diagnostics", False)
                ) and (
                    epoch % (1 if smoke else self.config.phases.evaluation_interval) == 0
                    or epoch == epochs
                )
                for pair_index, (name, pair) in enumerate(train_pairs.items()):
                    panel_epoch = epoch + 1000 * pair_index
                    if phase == "router": panel_epoch += 10000
                    panel = _panel(
                        self.config, pair, self.stage.config,
                        split="train", epoch=panel_epoch, smoke=smoke,
                    )
                    panel = panel.to(self.device); panels[name] = panel
                    order = stratified_orders(panel.is_target, 1 if smoke else self.config.phases.target_batch,
                                              1 if smoke else self.config.phases.safety_batch,
                                              self.config.seed + (1000003 if phase == "capability" else 2000003) * epoch
                                              + (7919 if phase == "capability" else 15401) * pair_index,
                                              self.device)
                    orders[name] = order[:2]; steps = max(steps, order[2])
                logs = []
                gradient_log = {}
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    for pair_position, (name, host_pair) in enumerate(train_pairs.items()):
                        panel = panels[name]; index = stratified_index(*orders[name], step,
                            1 if smoke else self.config.phases.target_batch,
                            1 if smoke else self.config.phases.safety_batch, self.device)
                        part = panel.subset(index); pair = host_pair.to(self.device)
                        modes = [None] * len(self.model.stages); modes[self.stage_index] = gate_mode
                        benefit_output = None
                        if phase == "router":
                            benefit_modes = [None] * len(self.model.stages)
                            benefit_modes[self.stage_index] = "forced_open"
                            with torch.no_grad():
                                benefit_output = self.model(
                                    pair, part.source, gate_modes=benefit_modes,
                                    return_diagnostics=False,
                                )
                        _, diagnostic = self.model(pair, part.source, gate_modes=modes, return_diagnostics=True)
                        stage_diag = diagnostic.stages[self.stage_index]
                        if float(stage_diag.row_residual.abs().max()) > self.config.audit.row_tolerance or \
                           float(stage_diag.column_residual.abs().max()) > self.config.audit.column_tolerance:
                            raise RuntimeError(f"{name}: correction projection failed")
                        prefix = diagnostic.fv_output if self.stage_index == 0 else diagnostic.stage_outputs[self.stage_index - 1]
                        low_order_mask = None
                        curriculum_degrees = getattr(
                            self.config.panel, "curriculum_degrees", ()
                        )
                        if curriculum_degrees and part.degrees is not None:
                            allowed = torch.tensor(
                                curriculum_degrees,
                                device=part.degrees.device,
                            )
                            low_order_mask = (
                                part.degrees.view(-1, 1) == allowed.view(1, -1)
                            ).any(dim=1)
                        loss, log, components = progressive_loss(
                            stage_diag, diagnostic.fv_output, prefix, part.truth,
                            part.is_target, pair.area_tgt, self.config.loss,
                            train_router=phase == "router",
                            benefit_output=benefit_output,
                            low_order_mask=low_order_mask,
                            router_scope=getattr(
                                self.stage.config, "router_scope", "global_local"
                            ),
                            return_components=True,
                        )
                        if diagnostic_epoch and step == 0 and pair_position == 0:
                            gradient_log = component_gradient_diagnostics(
                                {
                                    key: value * weights[name]
                                    for key, value in components.items()
                                },
                                parameters,
                            )
                        (loss * weights[name]).backward(); logs.append(log)
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.phases.gradient_clip); optimizer.step()
                assert_unchanged(frozen, frozen_snapshot, context=f"earlier stages during {phase}")
                assert_unchanged(phase_frozen, phase_frozen_snapshot, context=f"frozen parameters during {phase}")
                evaluate = epoch % (1 if smoke else self.config.phases.evaluation_interval) == 0 or epoch == epochs
                selection_result = None
                stop_early = False
                if evaluate:
                    selection_result = evaluate_selection(self.model, selection_pairs, selection_panels,
                                                          self.stage_index, self.config,
                                                          gate_mode if phase == "capability" else self.stage.config.deployment_gate_mode,
                                                          self.device,
                                                          capability=phase == "capability",
                                                          np2_operators=self.selection_operators)
                    score, *_, metrics = selection_result
                    key = "capability" if phase == "capability" else "final"
                    if score < state[f"{key}_score"]:
                        state[f"{key}_score"] = score; state[f"{key}_epoch"] = epoch
                        state[f"{key}_state"] = cpu_state(self.model); state["metrics"] = metrics
                    if patience:
                        if score < early_best - minimum_delta:
                            early_best, stale_evaluations = score, 0
                        else:
                            stale_evaluations += 1
                        stop_early = stale_evaluations >= patience
                bilinear_run = getattr(self.config, "schema_version", None) == 5
                row = {"phase": phase,
                       "stage": self.stage.name if bilinear_run else phase,
                       "epoch": epoch,
                       "target_rel": float(np.mean([value["target_rel"] for value in logs])),
                       "prefix_target_rel": float(np.mean([value["prefix_target_rel"] for value in logs])),
                       "safety_worst_prefix_ratio": float(np.max([value["safety_worst_prefix_ratio"] for value in logs])),
                       "selection_score": "" if selection_result is None else selection_result[0],
                       "early_stop": bool(stop_early),
                       "seconds": time.time() - started}
                row[
                    "safety_worst_base_ratio"
                    if bilinear_run else "safety_worst_fv_ratio"
                ] = float(np.max([
                    value["safety_worst_fv_ratio"] for value in logs
                ]))
                for key in (
                    "guard_cvar", "local_cvar", "gate_teacher", "safety_gate",
                    "delta", "low_order", "correction_scale",
                    "score_saturation", "projection_norm_ratio",
                    "loss_target", "loss_guard", "loss_local",
                    "loss_router_teacher", "loss_safety_gate",
                    "loss_correction", "loss_low_order",
                ):
                    row[key] = float(np.mean([value[key] for value in logs]))
                row.update(gradient_log)
                if selection_result is not None:
                    _, target_ratio, safety_ratio, base_ratio, prior_ratio, metrics = (
                        selection_result
                    )
                    worst_pair = max(
                        metrics,
                        key=lambda name: metrics[name][
                            "safety_worst_ratio_vs_prefix"
                        ],
                    )
                    worst_target_pair = max(
                        metrics,
                        key=lambda name: metrics[name][
                            "target_mean_ratio_vs_prefix"
                        ],
                    )
                    row.update(
                        selection_target_ratio=target_ratio,
                        selection_safety_prefix_ratio=safety_ratio,
                        selection_safety_base_ratio=base_ratio,
                        selection_prior_band_ratio=prior_ratio,
                        selection_worst_pair=worst_pair,
                        selection_worst_source_key=metrics[worst_pair][
                            "worst_safety_source_key"
                        ],
                        selection_worst_family=metrics[worst_pair][
                            "worst_safety_family"
                        ],
                        selection_worst_target_pair=worst_target_pair,
                        selection_worst_target_source_key=metrics[
                            worst_target_pair
                        ]["worst_target_source_key"],
                        selection_worst_target_family=metrics[
                            worst_target_pair
                        ]["worst_target_family"],
                    )
                    gaps = [
                        value.get("bilinear_to_np2_gap_closed")
                        for value in metrics.values()
                    ]
                    gaps = [value for value in gaps if value is not None]
                    row["selection_min_np2_gap_closed"] = (
                        min(gaps) if gaps else ""
                    )
                if phase == "router":
                    row.update(target_field_probability=float(np.mean([v["target_field_probability"] for v in logs])),
                               safety_field_probability=float(np.mean([v["safety_field_probability"] for v in logs])))
                history.append(row); state["corrector_hash"] = tensor_state_sha256({
                    name: value for name, value in self.stage.state_dict().items()
                    if name.startswith((
                        "geom_encoder.", "message_mlp.", "context_refine_mlp.",
                        "score_mlp.",
                        "field_scale_mlp.",
                    ))})
                _atomic_torch_save(self._pack(state, optimizer, phase=phase, epoch=epoch,
                                             completed=False, history=history, smoke=smoke), self.output)
                _write_history(self.history_path, history)
                if stop_early:
                    break
            return optimizer, last_epoch

        saved_phase = (
            "capability_complete" if capability_seed is not None
            else saved.get("phase") if saved else None
        )
        capability_epochs = 1 if smoke else self.config.phases.capability_epochs
        capability_completed_epoch = (
            int(capability_seed["epoch"])
            if capability_seed is not None
            else int(saved["epoch"]) if saved_phase == "capability_complete"
            else 0
        )
        if saved_phase not in {"router", "capability_complete"}:
            _, capability_completed_epoch = phase_run(
                "capability", capability_epochs,
                self.config.phases.capability_learning_rate,
                int(saved["epoch"]) + 1 if saved_phase == "capability" else 1,
                saved["optimizer_state"] if saved_phase == "capability" else None,
            )
            # The smoke contract deliberately traverses both phases so router
            # checkpoint/resume code is exercised even when one capability
            # step cannot beat the production identity floor.
            state["capability_selected"] = bool(smoke) or identity_floor_selection(
                state["capability_score"], identity_score, self.config.selection.capability_minimum_improvement)
            self.model.load_state_dict(state["capability_state"] if state["capability_selected"] else identity_state)
        if capability_only:
            selected = bool(state["capability_selected"])
            state.update(
                final_state=copy.deepcopy(
                    state["capability_state"] if selected else identity_state
                ),
                final_score=(
                    state["capability_score"] if selected else identity_score
                ),
                final_epoch=state["capability_epoch"] if selected else 0,
                selected_identity=not selected,
                router_state=copy.deepcopy(identity_state),
                router_score=identity_score,
                router_epoch=0,
            )
            self.model.load_state_dict(state["final_state"])
            self.model.set_training_stage(self.stage_index, "frozen")
            state["corrector_hash"] = tensor_state_sha256({
                name: value for name, value in self.stage.state_dict().items()
                if name.startswith((
                    "geom_encoder.", "message_mlp.", "context_refine_mlp.",
                    "score_mlp.",
                    "field_scale_mlp.",
                ))
            })
            pack = self._pack(
                state, None, phase="capability_complete",
                epoch=capability_completed_epoch, completed=True,
                history=history, smoke=smoke,
            )
            _atomic_torch_save(pack, self.output)
            _write_history(self.history_path, history)
            return pack
        if not state["capability_selected"]:
            state.update(final_state=copy.deepcopy(identity_state), final_score=identity_score,
                         final_epoch=0, selected_identity=True,
                         metrics=copy.deepcopy(state["identity_metrics"]))
        else:
            if saved_phase != "router": self.model.load_state_dict(state["capability_state"])
            expected_corrector = tensor_state_sha256({name: value for name, value in self.stage.state_dict().items()
                if name.startswith((
                    "geom_encoder.", "message_mlp.", "context_refine_mlp.",
                    "score_mlp.",
                    "field_scale_mlp.",
                ))})
            state["corrector_hash"] = expected_corrector
            router_epochs = 1 if smoke else self.config.phases.router_epochs
            if saved_phase != "router":
                initial = evaluate_selection(self.model, selection_pairs, selection_panels,
                                             self.stage_index, self.config,
                                             self.stage.config.deployment_gate_mode,
                                             self.device,
                                             np2_operators=self.selection_operators)
                if initial[0] < state["final_score"]:
                    state["final_score"] = initial[0]; state["final_epoch"] = 0
                    state["final_state"] = cpu_state(self.model); state["metrics"] = initial[-1]
                history.append({"phase": "router",
                                "stage": (
                                    self.stage.name
                                    if getattr(self.config, "schema_version", None) == 5
                                    else "router"
                                ),
                                "epoch": 0,
                                "selection_score": initial[0], "candidate": "initial_hard_router"})
            phase_run(
                "router", router_epochs, self.config.phases.router_learning_rate,
                int(saved["epoch"]) + 1 if saved_phase == "router" else 1,
                saved["optimizer_state"] if saved_phase == "router" else None,
            )
            if state["corrector_hash"] != expected_corrector: raise RuntimeError("corrector changed during router phase")
            state.update(
                router_state=copy.deepcopy(state["final_state"]),
                router_score=state["final_score"],
                router_epoch=state["final_epoch"],
            )
            selected = identity_floor_selection(state["final_score"], identity_score,
                                                self.config.selection.final_minimum_gain)
            state["selected_identity"] = not selected
            if not selected:
                state.update(final_state=copy.deepcopy(identity_state), final_score=identity_score,
                             final_epoch=0, metrics=copy.deepcopy(state["identity_metrics"]))
            self.model.load_state_dict(state["final_state"])
            state["corrector_hash"] = tensor_state_sha256({
                name: value for name, value in self.stage.state_dict().items()
                if name.startswith((
                    "geom_encoder.", "message_mlp.", "context_refine_mlp.",
                    "score_mlp.",
                    "field_scale_mlp.",
                ))})
        self.model.set_training_stage(self.stage_index, "frozen")
        state["corrector_hash"] = tensor_state_sha256({
            name: value for name, value in self.stage.state_dict().items()
            if name.startswith((
                "geom_encoder.", "message_mlp.", "context_refine_mlp.",
                "score_mlp.",
                "field_scale_mlp.",
            ))})
        final_epoch = (
            (1 if smoke else self.config.phases.router_epochs)
            if state["capability_selected"] else capability_completed_epoch
        )
        pack = self._pack(state, None, phase="complete", epoch=final_epoch, completed=True,
                          history=history, smoke=smoke)
        _atomic_torch_save(pack, self.output); _write_history(self.history_path, history)
        return pack

    # Small callback interface retained for unit-sized experiments.
    def train(self, batches, loss_function, score_function, *, capability_epochs, capability_lr,
              router_epochs, router_lr, prefix_score=float("inf"), minimum_improvement=0.0,
              weight_decay=1e-5, grad_clip=1.0):
        original = copy.deepcopy(self.stage.state_dict()); results = []
        for phase, epochs, lr in (("capability", capability_epochs, capability_lr), ("router", router_epochs, router_lr)):
            self.model.set_training_stage(self.stage_index, phase)
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            best, best_epoch, best_state = float("inf"), 0, None
            for epoch in range(1, epochs + 1):
                for pair, batch in batches:
                    optimizer.zero_grad(set_to_none=True); modes = [None] * len(self.model.stages)
                    modes[self.stage_index] = self.stage.config.capability_gate_mode if phase == "capability" else self.stage.config.router_gate_mode
                    output, diag = self.model(pair, batch.source, gate_modes=modes, return_diagnostics=True)
                    loss_function(output, batch, diag).backward(); torch.nn.utils.clip_grad_norm_(params, grad_clip); optimizer.step()
                score = float(score_function(self.model, epoch))
                if score < best: best, best_epoch, best_state = score, epoch, copy.deepcopy(self.stage.state_dict())
            self.stage.load_state_dict(best_state); results.append(PhaseResult(phase, best_epoch, best, False))
        selected = identity_floor_selection(results[-1].best_score, prefix_score, minimum_improvement)
        if not selected: self.stage.load_state_dict(original)
        self.stage.set_training_phase("frozen")
        return results[0], PhaseResult("router", results[1].best_epoch, results[1].best_score, not selected)


def save_training_checkpoint(path, model, *, stage_index, phase, epoch, optimizer=None, metadata=None):
    pack = {"format": CHECKPOINT_FORMAT, "schema_version": CHECKPOINT_SCHEMA,
            "stage_index": int(stage_index), "phase": str(phase), "epoch": int(epoch),
            "model_state": model.state_dict(), "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "model_stage_configs": [stage.config.to_dict() for stage in model.stages],
            "metadata": {} if metadata is None else dict(metadata)}
    _atomic_torch_save(pack, path); return pack
