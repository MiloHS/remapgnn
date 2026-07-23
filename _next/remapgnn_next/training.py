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

from .checkpoint import CLEAN_TRAINING_FORMAT, TRAINING_SCHEMA_VERSION
from .evaluation import area_relative_l2
from .panels import build_panel
from .provenance import canonical_json_sha256, file_sha256, tensor_state_sha256


CHECKPOINT_FORMAT = CLEAN_TRAINING_FORMAT
CHECKPOINT_SCHEMA = TRAINING_SCHEMA_VERSION


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
    return float(candidate_score) < float(prefix_score) - float(minimum_improvement)


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


def progressive_loss(stage_diagnostic, fv_output, prefix_output, truth, target_mask, area, weights, *, train_router):
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
        label = target.to(stage_diagnostic.field_probability.dtype)
        teacher = F.binary_cross_entropy(stage_diagnostic.field_probability, label)
        teacher = teacher + F.binary_cross_entropy(
            stage_diagnostic.local_probability, label[:, None].expand_as(stage_diagnostic.local_probability)
        )
        safety_gate = cvar(stage_diagnostic.field_probability[safety], weights.cvar_fraction) + cvar(
            stage_diagnostic.local_probability[safety].mean(1), weights.cvar_fraction
        )
    loss = target_loss + weights.guard_weight * guard + weights.local_weight * local + \
        weights.gate_teacher_weight * teacher + weights.safety_gate_weight * safety_gate + \
        weights.correction_weight * correction
    log = {
        "target_rel": float(current[target].sqrt().mean().detach()),
        "prefix_target_rel": float(prefix[target].sqrt().mean().detach()),
        "safety_worst_prefix_ratio": float(ratio_prefix.max().detach()),
        "safety_worst_fv_ratio": float(ratio_fv.max().detach()),
        "guard_cvar": float(guard.detach()), "local_cvar": float(local.detach()),
        "gate_teacher": float(teacher.detach()), "safety_gate": float(safety_gate.detach()),
        "delta": float(correction.detach()),
        "target_field_probability": float(stage_diagnostic.field_probability[target].mean().detach()),
        "safety_field_probability": float(stage_diagnostic.field_probability[safety].mean().detach()),
    }
    return loss, log


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


def selection_score(metrics, selection, audit):
    target = max(value["target_mean_ratio_vs_prefix"] for value in metrics.values())
    safety = max(value["safety_worst_ratio_vs_prefix"] for value in metrics.values())
    fv = max(value["safety_worst_ratio_vs_fv"] for value in metrics.values())
    prior = max(value["prefix_band_worst_ratio_vs_prefix"] for value in metrics.values())
    score = target + 5 * max(0, safety - (1 + selection.safety_tolerance)) + \
        5 * max(0, fv - (1 + audit.maximum_fv_regression)) + \
        5 * max(0, prior - (1 + selection.prior_band_tolerance))
    return float(score), float(target), float(safety), float(fv), float(prior)


@torch.no_grad()
def evaluate_selection(model, pairs, panels, stage_index, config, gate_mode, device):
    model.eval(); metrics = {}
    for name, host_pair in pairs.items():
        pair, panel = host_pair.to(device), panels[name].to(device)
        errors, prefix_errors, fv_errors, gates, locals_, rows, columns = [], [], [], [], [], [], []
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
            gates.extend(stage.field_gate.cpu().tolist()); locals_.extend(stage.local_gate.mean(1).cpu().tolist())
            rows.append(float(stage.row_residual.abs().max())); columns.append(float(stage.column_residual.abs().max()))
        current, prefix, fv = map(np.asarray, (errors, prefix_errors, fv_errors))
        target = panel.is_target.cpu().numpy(); safety = ~target
        ratio_prefix = current / np.maximum(prefix, 1e-20); ratio_fv = current / np.maximum(fv, 1e-20)
        frequency = panel.frequency.cpu().numpy()
        previous = (
            model.stages[stage_index - 1].config
            if stage_index
            else model.stages[stage_index].config
        )
        prior_mask = safety & np.isfinite(frequency) & (frequency > previous.band_lower) & (frequency <= previous.band_upper)
        if not prior_mask.any(): prior_mask = safety
        gates, locals_ = np.asarray(gates), np.asarray(locals_)
        metrics[name] = {
            "target_mean_ratio_vs_prefix": float(ratio_prefix[target].mean()),
            "target_worst_ratio_vs_prefix": float(ratio_prefix[target].max()),
            "target_mean_ratio_vs_fv": float(ratio_fv[target].mean()),
            "safety_worst_ratio_vs_prefix": float(ratio_prefix[safety].max()),
            "safety_worst_ratio_vs_fv": float(ratio_fv[safety].max()),
            "prefix_band_worst_ratio_vs_prefix": float(ratio_prefix[prior_mask].max()),
            "target_model_rel": float(current[target].mean()), "target_prefix_rel": float(prefix[target].mean()),
            "target_fv_rel": float(fv[target].mean()), "target_field_gate": float(gates[target].mean()),
            "safety_field_gate": float(gates[safety].mean()), "target_local_gate": float(locals_[target].mean()),
            "safety_local_gate": float(locals_[safety].mean()), "row_residual_max": max(rows),
            "column_residual_max": max(columns),
        }
    return (*selection_score(metrics, config.selection, config.audit), metrics)


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
                 history_path=None, device="cpu"):
        self.model = model; self.stage_index = int(stage_index); self.config = config
        self.train_pairs = train_pairs or {}; self.selection_pairs = selection_pairs or {}
        self.source_checkpoint = None if source_checkpoint is None else Path(source_checkpoint)
        self.model_initialization = (
            {} if model_initialization is None else copy.deepcopy(model_initialization)
        )
        self.output = Path(output) if output else (config.paths.checkpoint_path if config else None)
        self.history_path = Path(history_path) if history_path else (config.paths.history_path if config else None)
        self.device = torch.device(device)
        if not 0 <= self.stage_index < len(model.stages): raise IndexError("stage_index outside model")

    @property
    def stage(self): return self.model.stages[self.stage_index]

    def _auth(self):
        config_value = self.config.to_dict()
        files = sorted(Path(__file__).parent.glob("*.py"))
        return {
            "config_sha256": canonical_json_sha256(config_value),
            "implementation_sha256": {str(path.name): file_sha256(path) for path in files},
            "data_sha256": {name: {
                "edge": file_sha256(self.config.paths.edge_path(name)),
                "map": file_sha256(self.config.paths.map_path(name)),
            } for name in dict.fromkeys((*self.train_pairs, *self.selection_pairs))},
            "source_checkpoint": None if self.source_checkpoint is None else {
                "path": str(self.source_checkpoint), "sha256": file_sha256(self.source_checkpoint)},
        }

    def _pack(self, state, optimizer, *, phase, epoch, completed, history, smoke):
        return {
            "format": CHECKPOINT_FORMAT, "schema_version": CHECKPOINT_SCHEMA,
            "completed": bool(completed), "smoke": bool(smoke), "stage_index": self.stage_index,
            "phase": phase, "epoch": int(epoch), "model_state": cpu_state(self.model),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "identity_model_state": state["identity_state"], "identity_score": state["identity_score"],
            "capability_best_state": state["capability_state"], "capability_best_score": state["capability_score"],
            "capability_best_epoch": state["capability_epoch"], "capability_selected": state["capability_selected"],
            "best_model_state": state["final_state"], "final_best_score": state["final_score"],
            "final_best_epoch": state["final_epoch"], "selected_identity": state["selected_identity"],
            "selection_metrics": copy.deepcopy(state["metrics"]), "corrector_state_sha256": state["corrector_hash"],
            "history": copy.deepcopy(history), "pair_roles": copy.deepcopy(self.config.pair_roles),
            "model_stage_configs": [stage.config.to_dict() for stage in self.model.stages],
            "model_initialization": copy.deepcopy(self.model_initialization),
            "config": self.config.to_dict(), "provenance": self._auth(),
            "behavior": {"known_frequency_required": False, "adaptive_stopping": False,
                         "sequential_residual_training": True, "strict_prefix_freezing": True},
        }

    def _validate_resume(self, saved):
        if saved.get("format") != CHECKPOINT_FORMAT or saved.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("resume checkpoint has the wrong clean schema")
        if saved["stage_index"] != self.stage_index: raise ValueError("resume stage differs")
        if saved["provenance"] != self._auth(): raise ValueError("authenticated inputs changed since checkpoint")

    def run(self, *, resume=False, smoke=False):
        if self.config is None or not self.train_pairs or not self.selection_pairs or self.output is None:
            raise ValueError("config, train/selection pairs, and output are required")
        set_seed(self.config.seed); self.model.to(self.device)
        train_names = list(self.train_pairs)[:1] if smoke else list(self.train_pairs)
        selection_names = train_names if smoke else list(self.selection_pairs)
        train_pairs = {name: self.train_pairs[name] for name in train_names}
        selection_pairs = {name: (self.train_pairs.get(name) or self.selection_pairs[name]) for name in selection_names}
        weights = pair_weights(train_pairs)
        selection_panels = {name: build_panel(
                                self.config, pair, stage_config=self.stage.config,
                                split="train" if smoke else "val",
                                epoch=0, smoke=smoke, audit=True)
                            for name, pair in selection_pairs.items()}
        saved = None
        if resume:
            saved = torch.load(self.output, map_location="cpu", weights_only=False); self._validate_resume(saved)
            self.model.load_state_dict(saved["model_state"])
            if saved["completed"]: return saved
        identity_state = copy.deepcopy(saved["identity_model_state"]) if saved else cpu_state(self.model)
        if saved:
            identity_score = float(saved["identity_score"])
        else:
            identity_score, *_, identity_metrics = evaluate_selection(
                self.model, selection_pairs, selection_panels, self.stage_index, self.config, "forced_closed", self.device
            )
        history = list(saved.get("history", [])) if saved else [{"phase": "identity", "epoch": 0, "selection_score": identity_score}]
        state = {
            "identity_state": identity_state, "identity_score": identity_score,
            "capability_state": copy.deepcopy(saved["capability_best_state"]) if saved else copy.deepcopy(identity_state),
            "capability_score": float(saved["capability_best_score"]) if saved else identity_score,
            "capability_epoch": int(saved["capability_best_epoch"]) if saved else 0,
            "capability_selected": bool(saved.get("capability_selected", False)) if saved else False,
            "final_state": copy.deepcopy(saved["best_model_state"]) if saved else copy.deepcopy(identity_state),
            "final_score": float(saved["final_best_score"]) if saved else identity_score,
            "final_epoch": int(saved["final_best_epoch"]) if saved else 0,
            "selected_identity": True, "metrics": copy.deepcopy(saved.get("selection_metrics", {})) if saved else {},
            "corrector_hash": "",
        }
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
            for epoch in range(start_epoch, epochs + 1):
                started = time.time(); self.model.train(); panels = {}; orders = {}; steps = 0
                for pair_index, (name, pair) in enumerate(train_pairs.items()):
                    panel_epoch = epoch + 1000 * pair_index
                    if phase == "router": panel_epoch += 10000
                    panel = build_panel(
                        self.config, pair, stage_config=self.stage.config,
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
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    for name, host_pair in train_pairs.items():
                        panel = panels[name]; index = stratified_index(*orders[name], step,
                            1 if smoke else self.config.phases.target_batch,
                            1 if smoke else self.config.phases.safety_batch, self.device)
                        part = panel.subset(index); pair = host_pair.to(self.device)
                        modes = [None] * len(self.model.stages); modes[self.stage_index] = gate_mode
                        _, diagnostic = self.model(pair, part.source, gate_modes=modes, return_diagnostics=True)
                        stage_diag = diagnostic.stages[self.stage_index]
                        if float(stage_diag.row_residual.abs().max()) > self.config.audit.row_tolerance or \
                           float(stage_diag.column_residual.abs().max()) > self.config.audit.column_tolerance:
                            raise RuntimeError(f"{name}: correction projection failed")
                        prefix = diagnostic.fv_output if self.stage_index == 0 else diagnostic.stage_outputs[self.stage_index - 1]
                        loss, log = progressive_loss(stage_diag, diagnostic.fv_output, prefix, part.truth,
                                                     part.is_target, pair.area_tgt, self.config.loss,
                                                     train_router=phase == "router")
                        (loss * weights[name]).backward(); logs.append(log)
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.phases.gradient_clip); optimizer.step()
                assert_unchanged(frozen, frozen_snapshot, context=f"earlier stages during {phase}")
                assert_unchanged(phase_frozen, phase_frozen_snapshot, context=f"frozen parameters during {phase}")
                evaluate = epoch % (1 if smoke else self.config.phases.evaluation_interval) == 0 or epoch == epochs
                selection_result = None
                if evaluate:
                    selection_result = evaluate_selection(self.model, selection_pairs, selection_panels,
                                                          self.stage_index, self.config,
                                                          gate_mode if phase == "capability" else self.stage.config.deployment_gate_mode,
                                                          self.device)
                    score, *_, metrics = selection_result
                    key = "capability" if phase == "capability" else "final"
                    if score < state[f"{key}_score"]:
                        state[f"{key}_score"] = score; state[f"{key}_epoch"] = epoch
                        state[f"{key}_state"] = cpu_state(self.model); state["metrics"] = metrics
                row = {"phase": phase, "stage": phase, "epoch": epoch,
                       "target_rel": float(np.mean([value["target_rel"] for value in logs])),
                       "prefix_target_rel": float(np.mean([value["prefix_target_rel"] for value in logs])),
                       "safety_worst_prefix_ratio": float(np.max([value["safety_worst_prefix_ratio"] for value in logs])),
                       "safety_worst_fv_ratio": float(np.max([value["safety_worst_fv_ratio"] for value in logs])),
                       "selection_score": "" if selection_result is None else selection_result[0],
                       "seconds": time.time() - started}
                if phase == "router":
                    row.update(target_field_probability=float(np.mean([v["target_field_probability"] for v in logs])),
                               safety_field_probability=float(np.mean([v["safety_field_probability"] for v in logs])))
                history.append(row); state["corrector_hash"] = tensor_state_sha256({
                    name: value for name, value in self.stage.state_dict().items()
                    if name.startswith(("geom_encoder.", "message_mlp.", "score_mlp."))})
                _atomic_torch_save(self._pack(state, optimizer, phase=phase, epoch=epoch,
                                             completed=False, history=history, smoke=smoke), self.output)
                _write_history(self.history_path, history)
            return optimizer

        saved_phase = saved.get("phase") if saved else None
        capability_epochs = 1 if smoke else self.config.phases.capability_epochs
        if saved_phase != "router":
            phase_run("capability", capability_epochs, self.config.phases.capability_learning_rate,
                      int(saved["epoch"]) + 1 if saved_phase == "capability" else 1,
                      saved["optimizer_state"] if saved_phase == "capability" else None)
            # The smoke contract deliberately traverses both phases so router
            # checkpoint/resume code is exercised even when one capability
            # step cannot beat the production identity floor.
            state["capability_selected"] = bool(smoke) or identity_floor_selection(
                state["capability_score"], identity_score, self.config.selection.capability_minimum_improvement)
            self.model.load_state_dict(state["capability_state"] if state["capability_selected"] else identity_state)
        if not state["capability_selected"]:
            state.update(final_state=copy.deepcopy(identity_state), final_score=identity_score,
                         final_epoch=0, selected_identity=True)
        else:
            if saved_phase != "router": self.model.load_state_dict(state["capability_state"])
            expected_corrector = tensor_state_sha256({name: value for name, value in self.stage.state_dict().items()
                if name.startswith(("geom_encoder.", "message_mlp.", "score_mlp."))})
            state["corrector_hash"] = expected_corrector
            router_epochs = 1 if smoke else self.config.phases.router_epochs
            if saved_phase != "router":
                initial = evaluate_selection(self.model, selection_pairs, selection_panels,
                                             self.stage_index, self.config,
                                             self.stage.config.deployment_gate_mode, self.device)
                if initial[0] < state["final_score"]:
                    state["final_score"] = initial[0]; state["final_epoch"] = 0
                    state["final_state"] = cpu_state(self.model); state["metrics"] = initial[-1]
                history.append({"phase": "router", "stage": "router", "epoch": 0,
                                "selection_score": initial[0], "candidate": "initial_hard_router"})
            phase_run("router", router_epochs, self.config.phases.router_learning_rate,
                      int(saved["epoch"]) + 1 if saved_phase == "router" else 1,
                      saved["optimizer_state"] if saved_phase == "router" else None)
            if state["corrector_hash"] != expected_corrector: raise RuntimeError("corrector changed during router phase")
            selected = identity_floor_selection(state["final_score"], identity_score,
                                                self.config.selection.final_minimum_gain)
            state["selected_identity"] = not selected
            if not selected:
                state.update(final_state=copy.deepcopy(identity_state), final_score=identity_score,
                             final_epoch=0)
            self.model.load_state_dict(state["final_state"])
            state["corrector_hash"] = tensor_state_sha256({
                name: value for name, value in self.stage.state_dict().items()
                if name.startswith(("geom_encoder.", "message_mlp.", "score_mlp."))})
        self.model.set_training_stage(self.stage_index, "frozen")
        state["corrector_hash"] = tensor_state_sha256({
            name: value for name, value in self.stage.state_dict().items()
            if name.startswith(("geom_encoder.", "message_mlp.", "score_mlp."))})
        final_epoch = (1 if smoke else self.config.phases.router_epochs) if state["capability_selected"] else capability_epochs
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
