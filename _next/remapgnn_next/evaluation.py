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


def safe_ratio(numerator, denominator, tolerance):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.empty(numerator.shape, dtype=np.float64)
    both = (np.abs(numerator) <= tolerance) & (np.abs(denominator) <= tolerance)
    zero_denominator = (np.abs(denominator) <= tolerance) & ~both
    ordinary = ~(both | zero_denominator)
    result[both] = 1.0
    result[zero_denominator] = np.inf
    result[ordinary] = numerator[ordinary] / denominator[ordinary]
    return float(result) if result.ndim == 0 else result


def research_stopping_evidence(pair_metrics, stopping, *, allow_transfer):
    """Apply the explicit stop/go rule for learned-corrector experiments."""
    minimum_gain = float(stopping["minimum_mapping_target_gain"])
    maximum_safety = float(stopping["maximum_mapping_safety_ratio"])
    mapping_checks = {
        name: {
            "target_pass": (
                value["target_mean_ratio_vs_prefix"] <= 1.0 - minimum_gain
            ),
            "safety_pass": (
                value["safety_worst_ratio_vs_prefix"] <= maximum_safety
            ),
            "np2_gap_pass": (
                value.get("bilinear_to_np2_gap_closed") is not None
                and value["bilinear_to_np2_gap_closed"]
                >= float(stopping["minimum_np2_gap_closed"])
            ),
        }
        for name, value in pair_metrics.items()
    }
    consistent = bool(mapping_checks) and all(
        value["target_pass"] and value["safety_pass"]
        for value in mapping_checks.values()
    )
    closes_gap = bool(mapping_checks) and all(
        value["np2_gap_pass"] for value in mapping_checks.values()
    )
    ico_name = stopping["ico_transfer_pair"]
    ico = pair_metrics.get(ico_name)
    ico_transfer = bool(
        allow_transfer and ico
        and ico["target_mean_ratio_vs_prefix"]
        <= 1.0 - float(stopping["minimum_ico_transfer_gain"])
        and ico["safety_worst_ratio_vs_prefix"] <= maximum_safety
    )
    proceed = bool(consistent and (closes_gap or ico_transfer))
    return {
        "mapping_checks": mapping_checks,
        "consistent_heldout_admission": consistent,
        "meaningful_np2_gap_closure": closes_gap,
        "convincing_ico_transfer": ico_transfer,
        "proceed_to_full_training": proceed,
        "conclusion": (
            "architecture_has_transfer_evidence" if proceed
            else "useful_corrections_are_mostly_pair_specific"
        ),
    }


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
        ratio = safe_ratio(final.cpu().numpy(), baseline.cpu().numpy(), 1e-14)
        relative = float(np.max(ratio - 1.0))
        regressions["safety_vs_fv"] = relative
        passed = bool(np.isfinite(relative) and relative <= safety_tolerance)
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
    tensors = [output, diagnostic.fv_output]
    for stage in diagnostic.stages:
        tensors.extend((stage.output, stage.delta_weight, stage.row_residual,
                        stage.column_residual, stage.field_gate, stage.local_gate,
                        stage.field_probability, stage.local_probability))
        for value in (stage.field_gate, stage.local_gate, stage.field_probability,
                      stage.local_probability):
            if bool((value < 0).any()) or bool((value > 1).any()):
                raise RuntimeError(f"{pair.pair}: routing value outside [0,1]")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError(f"{pair.pair}: non-finite structural result")
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
                "shared_anchor": bool(fields.shared_anchor[index]) if fields.shared_anchor is not None else False,
                "is_target_band": bool(fields.is_target[index]),
                "is_prefix_band": bool(np.isfinite(frequency) and frequency > previous.band_lower and frequency <= previous.band_upper),
                "degree": fields.labels[index][0], "order": fields.labels[index][1], "nu": frequency,
                "model_rel_l2": model_error, "prefix_rel_l2": prefix_error, "fv_rel_l2": fv_error,
                "np2_rel_l2": float(errors["np2"][local]),
                "model_over_prefix": safe_ratio(model_error, prefix_error, config.audit.zero_error_tolerance),
                "model_over_fv": safe_ratio(model_error, fv_error, config.audit.zero_error_tolerance),
                "model_over_np2": safe_ratio(model_error, float(errors["np2"][local]), config.audit.zero_error_tolerance),
                "prefix_over_fv": safe_ratio(prefix_error, fv_error, config.audit.zero_error_tolerance),
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
    if any(
        not np.isfinite(value)
        for row in detail for value in row.values()
        if isinstance(value, (float, np.floating))
    ):
        failures.append("non-finite audit detail")
    for pair in pairs:
        rows = [x for x in detail if x["pair"] == pair]; target = [x for x in rows if x["is_target_band"]]
        safety = [x for x in rows if not x["is_target_band"]]; prior = [x for x in safety if x["is_prefix_band"]]
        if not target or not safety:
            failures.append(f"{pair}: incomplete target/safety panel"); continue
        metric = {
            "target_model_over_prefix_mean": float(np.mean([x["model_over_prefix"] for x in target])),
            "target_model_over_prefix_worst": float(np.max([x["model_over_prefix"] for x in target])),
            "target_regression_count": sum(x["model_over_prefix"] > 1 for x in target),
            "safety_model_over_prefix_worst": float(np.max([x["model_over_prefix"] for x in safety])),
            "safety_model_over_fv_worst": float(np.max([x["model_over_fv"] for x in safety])),
            "prefix_band_model_over_prefix_worst": float(np.max(
                [x["model_over_prefix"] for x in prior]
            )) if prior else 1.0,
            "prior_band_applicable": bool(prior),
        }; pair_metrics[pair] = metric
        if metric["target_model_over_prefix_mean"] > 1 - config.audit.minimum_target_gain: failures.append(f"{pair}: insufficient target gain")
        if metric["safety_model_over_prefix_worst"] > 1 + config.audit.maximum_safety_regression: failures.append(f"{pair}: safety regression vs prefix")
        if metric["safety_model_over_fv_worst"] > 1 + config.audit.maximum_fv_regression: failures.append(f"{pair}: safety regression vs FV")
        if prior and metric["prefix_band_model_over_prefix_worst"] > 1 + config.audit.maximum_prior_band_regression: failures.append(f"{pair}: prior-band regression")
    for value in structures:
        if value.get("error"):
            failures.append(f"{value.get('pair', 'unknown')}: structural execution failed")
            continue
        if any(
            isinstance(item, (float, np.floating)) and not np.isfinite(item)
            for item in value.values()
        ):
            failures.append(f"{value.get('pair', 'unknown')}: non-finite structure")
            continue
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
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(child) for child in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return "NaN" if np.isnan(item) else ("Infinity" if item > 0 else "-Infinity")
        return item
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ); temporary.replace(path)


def audit_experiment(
    model, config, pairs, checkpoint, *, device="cpu", smoke=False,
    tag="development", overwrite=False,
):
    detail, structures, data_hashes = [], [], {}
    for name, pair in pairs.items():
        np2_path = Path(config.paths.maps) / f"map_{name}_conserve_np2.nc"
        real = []
        for path in config.paths.real_field_paths(name):
            record = {"path": str(path), "available": path.is_file()}
            if path.is_file():
                record["sha256"] = file_sha256(path)
            real.append(record)
        data_hashes[name] = {
            "edge": file_sha256(config.paths.edge_path(name)),
            "map": file_sha256(config.paths.map_path(name)),
            "np2": file_sha256(np2_path) if np2_path.is_file() else None,
            "real": real,
        }
        try:
            fields = build_panel(
                config, pair, stage_config=model.stages[-1].config,
                split="train" if smoke else "audit", epoch=config.seed,
                smoke=smoke, audit=True,
            )
            if not np2_path.is_file():
                raise FileNotFoundError(f"missing np2 map: {np2_path}")
            detail.extend(audit_pair(
                model, pair, fields, load_map_operator(np2_path), config, device=device
            ))
            structures.append(structural_checks(
                model, pair, fields.source[:min(2, len(fields.source))],
                config, device=device,
            ))
        except Exception as error:
            structures.append({"pair": name, "error": f"{type(error).__name__}: {error}"})
    summary = summarize(detail); promotion = promotion_report(detail, structures, config, list(pairs))
    base = Path(config.paths.reports) / f"{config.run_name}_audit_{tag}{'_smoke' if smoke else ''}"
    detail_path, summary_path, report_path = (base.with_name(base.name + suffix) for suffix in ("_detail.csv", "_summary.csv", "_report.json"))
    manifest_path = base.with_name(base.name + "_manifest.json")
    if not overwrite and any(path.exists() for path in (detail_path, summary_path, report_path, manifest_path)):
        raise FileExistsError(f"audit output exists for tag {tag!r}; use --overwrite")
    provenance = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                  "config_sha256": file_sha256(config.path) if config.path else None,
                  "pairs": list(pairs), "outputs": {"detail": str(detail_path), "summary": str(summary_path), "report": str(report_path)}}
    report_value = {**provenance, "structures": structures, "promotion": promotion,
                    "audit_data_sha256": data_hashes}
    _atomic_csv(detail_path, detail); _atomic_csv(summary_path, summary); _atomic_json(report_path, report_value)
    _atomic_json(manifest_path, {
        "format": "remapgnn.audit_outputs", "schema_version": 1,
        "inputs": {"checkpoint": provenance["checkpoint_sha256"],
                   "config": provenance["config_sha256"], "data": data_hashes},
        "outputs": {
            str(path): file_sha256(path)
            for path in (detail_path, summary_path, report_path)
        },
    })
    return AuditReport(tuple(detail), tuple(summary), tuple(structures), promotion, provenance)
