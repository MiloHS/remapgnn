from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .panels import build_panel
from .provenance import file_sha256
from .sparse import apply_operator
from .types import SparseOperator


def area_relative_l2(prediction, truth, area, epsilon=1.0e-30):
    numerator = (area.view(1, -1) * (prediction - truth).square()).sum(dim=1)
    denominator = (area.view(1, -1) * truth.square()).sum(dim=1).clamp_min(epsilon)
    return (numerator / denominator).clamp_min(0.0).sqrt()


@dataclass(frozen=True)
class AuditResult:
    names: tuple[str, ...]
    errors: torch.Tensor
    roles: tuple[str, ...]
    passed: bool
    regressions: dict[str, float]


@dataclass(frozen=True)
class AuditReport:
    detail: tuple[dict, ...]
    summary: tuple[dict, ...]
    structures: tuple[dict, ...]
    promotion: dict
    provenance: dict


def audit_progressive(model, pair, fields, *, np2_operator=None, safety_tolerance=0.02):
    output, diagnostic = model(pair, fields.source, return_diagnostics=True)
    predictions = [diagnostic.fv_output, *diagnostic.stage_outputs]
    names = ["fv", *[stage.name for stage in diagnostic.stages]]
    if np2_operator is not None:
        predictions.append(apply_operator(np2_operator, fields.source)); names.append("np2")
    errors = torch.stack([area_relative_l2(value, fields.truth.to(value.device), pair.area_tgt.to(value.device))
                          for value in predictions])
    safety = ~fields.is_target.to(errors.device); regressions = {}; passed = True
    if len(predictions) > 1 and bool(safety.any()):
        baseline, final = errors[0, safety], errors[len(diagnostic.stage_outputs), safety]
        relative = ((final - baseline) / baseline.clamp_min(1e-30)).max()
        regressions["safety_vs_fv"] = float(relative); passed = bool(relative <= safety_tolerance)
    return AuditResult(tuple(names), errors, tuple(fields.roles), passed, regressions), diagnostic


def load_map_operator(path):
    import xarray as xr
    with xr.open_dataset(path) as data:
        weight = torch.tensor(np.asarray(data["S"].values).reshape(-1), dtype=torch.float64)
        target = torch.tensor(np.asarray(data["row"].values).reshape(-1) - 1, dtype=torch.long)
        source = torch.tensor(np.asarray(data["col"].values).reshape(-1) - 1, dtype=torch.long)
        area_source = torch.tensor(np.asarray(data["area_a"].values).reshape(-1), dtype=torch.float64)
        area_target = torch.tensor(np.asarray(data["area_b"].values).reshape(-1), dtype=torch.float64)
    return SparseOperator.from_weight(source, target, weight, area_source, area_target,
                                      provenance={"path": str(path)})


def _synchronize(device):
    if torch.device(device).type == "cuda": torch.cuda.synchronize(device)


@torch.no_grad()
def structural_checks(model, pair, sample, config, *, device):
    model.eval(); pair = pair.to(device); sample = sample.to(device)
    output, diagnostic = model(pair, sample, return_diagnostics=True)
    rows = {stage.name: float(stage.row_residual.abs().max()) for stage in diagnostic.stages}
    columns = {stage.name: float(stage.column_residual.abs().max()) for stage in diagnostic.stages}
    if max(rows.values(), default=0) > config.audit.row_tolerance or max(columns.values(), default=0) > config.audit.column_tolerance:
        raise RuntimeError(f"{pair.pair}: projection constraints failed")
    constant = model(pair, torch.ones((1, pair.n_src), dtype=sample.dtype, device=device), return_diagnostics=False)
    affine_errors, gate_errors = [], []
    for scale, offset in ((1.7, -0.3), (-1.2, 0.25), (1e-8, 0.0)):
        transformed, transformed_diag = model(pair, scale * sample + offset, return_diagnostics=True)
        expected = scale * output + offset
        affine_errors.append(float((transformed - expected).abs().max() / expected.abs().max().clamp_min(1e-12)))
        gate_errors.append(max(float((new.field_gate - old.field_gate).abs().max())
                               for new, old in zip(transformed_diag.stages, diagnostic.stages)))
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), device=device))
    rotated_pair = replace(pair, src_xyz=pair.src_xyz @ rotation, tgt_xyz=pair.tgt_xyz @ rotation)
    rotated = model(rotated_pair, sample, return_diagnostics=False)
    rotation_error = float((rotated - output).abs().max() / output.abs().max().clamp_min(1e-12))
    modes = [None] * len(model.stages); modes[-1] = "forced_closed"
    rejected, rejected_diag = model(pair, sample, gate_modes=modes, return_diagnostics=True)
    prefix = rejected_diag.fv_output if len(model.stages) == 1 else rejected_diag.stage_outputs[-2]
    exact_rejection = torch.equal(rejected, prefix)
    for _ in range(2): model(pair, sample, return_diagnostics=False)
    _synchronize(device); started = time.perf_counter()
    for _ in range(config.audit.timing_repeats): model(pair, sample, return_diagnostics=False)
    _synchronize(device)
    return {
        "pair": pair.pair, "stage_row_residuals": rows, "stage_column_residuals": columns,
        "stage_delta_row_sum_max_abs": max(rows.values(), default=0),
        "stage_delta_area_column_sum_max_abs": max(columns.values(), default=0),
        "constant_max_abs": float((constant - 1).abs().max()),
        "positive_affine_rel_linf": affine_errors[0], "negative_affine_rel_linf": affine_errors[1],
        "tiny_scale_affine_rel_linf": affine_errors[2], "stage_gate_affine_max_abs": max(gate_errors),
        "rotation_rel_linf": rotation_error, "forced_rejection_exact_prefix": bool(exact_rejection),
        "forced_rejection_vs_prefix_max_abs": float((rejected - prefix).abs().max()),
        "field_gate_means": {stage.name: float(stage.field_gate.mean()) for stage in diagnostic.stages},
        "local_gate_means": {stage.name: float(stage.local_gate.mean()) for stage in diagnostic.stages},
        "model_apply_ms_per_field": 1000 * (time.perf_counter() - started) /
                                    (config.audit.timing_repeats * sample.shape[0]),
    }


@torch.no_grad()
def audit_pair(model, pair, fields, np2, config, *, device):
    pair_device, fields_device, np2 = pair.to(device), fields.to(device), np2.to(device)
    detail = []; batch_size = config.audit.field_batch
    for start in range(0, fields.source.shape[0], batch_size):
        stop = min(start + batch_size, fields.source.shape[0]); part = fields_device.subset(range(start, stop))
        _, diagnostics = model(pair_device, part.source, return_diagnostics=True)
        predictions = {"fv": diagnostics.fv_output, "np2": apply_operator(np2, part.source)}
        predictions.update({stage.name: stage.output for stage in diagnostics.stages})
        errors = {name: area_relative_l2(value, part.truth, pair_device.area_tgt) for name, value in predictions.items()}
        final_name = diagnostics.stages[-1].name
        prefix_name = diagnostics.stages[-2].name if len(diagnostics.stages) > 1 else "fv"
        for local, index in enumerate(range(start, stop)):
            frequency = float(fields.frequency[index])
            previous = (
                model.stages[-2].config
                if len(model.stages) > 1
                else model.stages[0].config
            )
            model_error, prefix_error, fv_error = (float(errors[name][local]) for name in (final_name, prefix_name, "fv"))
            row = {
                "pair": pair.pair, "field_index": index,
                "family": (fields.families or fields.roles)[index], "role": fields.roles[index],
                "source_key": fields.source_keys[index] if fields.source_keys else "",
                "is_target_band": bool(fields.is_target[index]),
                "is_prefix_band": bool(np.isfinite(frequency) and frequency > previous.band_lower and frequency <= previous.band_upper),
                "degree": fields.labels[index][0], "order": fields.labels[index][1], "nu": frequency,
                "model_rel_l2": model_error, "prefix_rel_l2": prefix_error, "fv_rel_l2": fv_error,
                "np2_rel_l2": float(errors["np2"][local]),
                "model_over_prefix": model_error / max(prefix_error, 1e-20),
                "model_over_fv": model_error / max(fv_error, 1e-20),
                "model_over_np2": model_error / max(float(errors["np2"][local]), 1e-20),
                "prefix_over_fv": prefix_error / max(fv_error, 1e-20),
            }
            for stage_index, stage in enumerate(diagnostics.stages):
                row[f"{stage.name}_rel_l2"] = float(errors[stage.name][local])
                row[f"{stage.name}_field_gate"] = float(stage.field_gate[local])
                row[f"{stage.name}_local_gate_mean"] = float(stage.local_gate[local].mean())
            detail.append(row)
    return detail


def summarize(detail):
    result = []
    for pair, family in sorted({(row["pair"], row["family"]) for row in detail}):
        group = [row for row in detail if row["pair"] == pair and row["family"] == family]
        result.append({
            "pair": pair, "family": family, "n_fields": len(group),
            "is_target_band": bool(group[0]["is_target_band"]),
            "model_rel_l2_mean": float(np.mean([x["model_rel_l2"] for x in group])),
            "prefix_rel_l2_mean": float(np.mean([x["prefix_rel_l2"] for x in group])),
            "fv_rel_l2_mean": float(np.mean([x["fv_rel_l2"] for x in group])),
            "np2_rel_l2_mean": float(np.mean([x["np2_rel_l2"] for x in group])),
            "model_over_prefix_mean": float(np.mean([x["model_over_prefix"] for x in group])),
            "model_over_prefix_worst": float(np.max([x["model_over_prefix"] for x in group])),
            "model_over_fv_mean": float(np.mean([x["model_over_fv"] for x in group])),
        })
    return result


def promotion_report(detail, structures, config, pairs):
    failures, pair_metrics = [], {}
    for pair in pairs:
        rows = [x for x in detail if x["pair"] == pair]; target = [x for x in rows if x["is_target_band"]]
        safety = [x for x in rows if not x["is_target_band"]]; prior = [x for x in safety if x["is_prefix_band"]]
        if not target or not safety or not prior:
            failures.append(f"{pair}: incomplete target/safety/prior panel"); continue
        metric = {
            "target_model_over_prefix_mean": float(np.mean([x["model_over_prefix"] for x in target])),
            "target_model_over_prefix_worst": float(np.max([x["model_over_prefix"] for x in target])),
            "target_regression_count": sum(x["model_over_prefix"] > 1 for x in target),
            "safety_model_over_prefix_worst": float(np.max([x["model_over_prefix"] for x in safety])),
            "safety_model_over_fv_worst": float(np.max([x["model_over_fv"] for x in safety])),
            "prefix_band_model_over_prefix_worst": float(np.max([x["model_over_prefix"] for x in prior])),
        }; pair_metrics[pair] = metric
        if metric["target_model_over_prefix_mean"] > 1 - config.audit.minimum_target_gain: failures.append(f"{pair}: insufficient target gain")
        if metric["safety_model_over_prefix_worst"] > 1 + config.audit.maximum_safety_regression: failures.append(f"{pair}: safety regression vs prefix")
        if metric["safety_model_over_fv_worst"] > 1 + config.audit.maximum_fv_regression: failures.append(f"{pair}: safety regression vs FV")
        if metric["prefix_band_model_over_prefix_worst"] > 1 + config.audit.maximum_prior_band_regression: failures.append(f"{pair}: prior-band regression")
    for value in structures:
        pair = value["pair"]
        if not value["forced_rejection_exact_prefix"]: failures.append(f"{pair}: forced rejection is not exact")
        if value["stage_delta_row_sum_max_abs"] > config.audit.row_tolerance: failures.append(f"{pair}: row constraint")
        if value["stage_delta_area_column_sum_max_abs"] > config.audit.column_tolerance: failures.append(f"{pair}: column constraint")
        if value["constant_max_abs"] > 2e-6: failures.append(f"{pair}: constant reproduction")
        if max(value["positive_affine_rel_linf"], value["negative_affine_rel_linf"], value["tiny_scale_affine_rel_linf"]) > 1e-4: failures.append(f"{pair}: affine equivariance")
        if value["rotation_rel_linf"] > 1e-5: failures.append(f"{pair}: rotation invariance")
    return {"passed": not failures, "failures": failures, "pair_metrics": pair_metrics,
            "thresholds": {"minimum_target_gain": config.audit.minimum_target_gain,
                           "maximum_safety_regression": config.audit.maximum_safety_regression,
                           "maximum_prior_band_regression": config.audit.maximum_prior_band_regression,
                           "maximum_fv_regression": config.audit.maximum_fv_regression}}


def _atomic_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); temporary.replace(path)


def audit_experiment(model, config, pairs, checkpoint, *, device="cpu", smoke=False, tag="development"):
    detail, structures = [], []
    for name, pair in pairs.items():
        fields = build_panel(
            config, pair, stage_config=model.stages[-1].config,
            split="train" if smoke else "audit", epoch=config.seed,
            smoke=smoke, audit=True,
        )
        np2_path = Path(config.paths.maps) / f"map_{name}_conserve_np2.nc"
        if not np2_path.is_file(): raise FileNotFoundError(f"missing np2 map: {np2_path}")
        detail.extend(audit_pair(model, pair, fields, load_map_operator(np2_path), config, device=device))
        structures.append(structural_checks(model, pair, fields.source[:min(2, len(fields.source))], config, device=device))
    summary = summarize(detail); promotion = promotion_report(detail, structures, config, list(pairs))
    base = Path(config.paths.reports) / f"progressive_next_audit_{tag}{'_smoke' if smoke else ''}"
    detail_path, summary_path, report_path = (base.with_name(base.name + suffix) for suffix in ("_detail.csv", "_summary.csv", "_report.json"))
    provenance = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                  "config_sha256": file_sha256(config.path) if config.path else None,
                  "pairs": list(pairs), "outputs": {"detail": str(detail_path), "summary": str(summary_path), "report": str(report_path)}}
    report_value = {**provenance, "structures": structures, "promotion": promotion,
                    "audit_data_sha256": {name: {"edge": file_sha256(config.paths.edge_path(name)),
                                                   "map": file_sha256(config.paths.map_path(name))} for name in pairs}}
    _atomic_csv(detail_path, detail); _atomic_csv(summary_path, summary); _atomic_json(report_path, report_value)
    return AuditReport(tuple(detail), tuple(summary), tuple(structures), promotion, provenance)
