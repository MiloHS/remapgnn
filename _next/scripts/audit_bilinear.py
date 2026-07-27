#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from remapgnn_next.band_panels import build_band_panel
from remapgnn_next.bilinear import build_bilinear_pair, apply_conservative_bilinear
from remapgnn_next.checkpoint import load_bilinear_training_checkpoint
from remapgnn_next.config import load_config
from remapgnn_next.evaluation import area_relative_l2, load_map_operator, safe_ratio
from remapgnn_next.fv import build_pair_from_files
from remapgnn_next.provenance import (
    authenticated_load, canonical_json_sha256, file_sha256,
)
from remapgnn_next.sparse import apply_operator


def _write_json(path, value, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"audit report exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _ratio_summary(rows):
    if not rows:
        return {"count": 0, "mean_model_over_prefix": None,
                "worst_model_over_prefix": None}
    ratios = [row["model_over_prefix"] for row in rows]
    return {
        "count": len(rows),
        "mean_model_over_prefix": float(np.mean(ratios)),
        "worst_model_over_prefix": float(np.max(ratios)),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Audit a bilinear progressive candidate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--allow-protected", action="store_true")
    parser.add_argument("--tag", default="development")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.schema_version != 5:
        raise ValueError("audit_bilinear requires a schema-5 config")
    checkpoint = Path(args.checkpoint)
    model, pack = load_bilinear_training_checkpoint(checkpoint)
    if canonical_json_sha256(pack["config"]) != canonical_json_sha256(config.to_dict()):
        raise ValueError("candidate scientific config differs from audit config")
    normalization = pack["model_initialization"]["runtime_data"]["edge_normalization"]
    names = args.pairs or list(config.pair_roles["selection"])
    protected = set(config.pair_roles.get("protected", ())) | set(
        config.pair_roles.get("external_resolution", ())
    )
    if protected & set(names) and not args.allow_protected:
        raise ValueError("protected/external pairs require --allow-protected")
    device = torch.device(args.device)
    model = model.to(device).eval()
    fv_pack, _ = authenticated_load(config.benchmarks.fv_checkpoint)
    fv_progressive, _ = authenticated_load(
        config.benchmarks.fv_progressive_checkpoint
    )
    detail = []
    structures = {}
    final_stage = model.stages[-1]
    for name in names:
        pair = build_bilinear_pair(
            name, config.paths.edge_path(name),
            config.paths.bilinear_map_path(name),
            feature_names=config.features.edge,
            normalization=normalization,
            correction_reference_kind=config.baseline.correction_reference,
            bilinear_reference_fraction=config.baseline.bilinear_reference_fraction,
            quadrature_resolution=(
                2 if args.smoke else config.panel.quadrature_resolution
            ),
            smoother_neighbors=config.panel.smoother_neighbors,
        )
        panel = build_band_panel(
            config, pair, stage_config=final_stage.config,
            split="audit", epoch=config.seed, smoke=args.smoke, audit=True,
        )
        fv_pair = build_pair_from_files(
            name, config.paths.edge_path(name), config.paths.map_path(name),
            fv_pack, fv_progressive, device="cpu",
            quadrature_resolution=2,
            smoother_neighbors=config.panel.smoother_neighbors,
        )
        np2_path = (
            Path(config.paths.maps)
            / f"map_{name}_{config.benchmarks.np2_suffix}.nc"
        )
        np2 = load_map_operator(np2_path)
        pair_device = pair.to(device)
        source = panel.source.to(device)
        truth = panel.truth.to(device)
        output, diagnostics = model(
            pair_device, source, return_diagnostics=True
        )
        adjusted, raw, shifts = apply_conservative_bilinear(pair_device, source)
        predictions = {
            "bilinear_raw": raw,
            "bilinear": adjusted,
            "fv": apply_operator(fv_pair.fv_operator.to(device), source),
            "np2": apply_operator(np2.to(device), source),
        }
        predictions.update({
            stage.name: stage.output for stage in diagnostics.stages
        })
        errors = {
            method: area_relative_l2(value, truth, pair_device.area_tgt)
            for method, value in predictions.items()
        }
        prefix_name = (
            model.stages[-2].name if len(model.stages) > 1 else "bilinear"
        )
        for index in range(panel.source.shape[0]):
            current = float(errors[final_stage.name][index])
            prefix = float(errors[prefix_name][index])
            base = float(errors["bilinear"][index])
            detail.append({
                "pair": name,
                "field_index": index,
                "source_key": panel.source_keys[index],
                "family": panel.families[index],
                "role": panel.roles[index],
                "band": panel.bands[index],
                "degree": int(panel.degrees[index]),
                "model_rel_l2": current,
                "prefix_rel_l2": prefix,
                "bilinear_rel_l2": base,
                "raw_bilinear_rel_l2": float(errors["bilinear_raw"][index]),
                "fv_rel_l2": float(errors["fv"][index]),
                "np2_rel_l2": float(errors["np2"][index]),
                "model_over_prefix": safe_ratio(
                    current, prefix, config.audit.zero_error_tolerance
                ),
                "model_over_bilinear": safe_ratio(
                    current, base, config.audit.zero_error_tolerance
                ),
                "conservation_shift": float(shifts[index]),
            })
        structures[name] = {
            "row_residual_max": max(
                float(stage.row_residual.abs().max())
                for stage in diagnostics.stages
            ),
            "column_residual_max": max(
                float(stage.column_residual.abs().max())
                for stage in diagnostics.stages
            ),
        }
    target_band = final_stage.config.target_band
    target = [row for row in detail if row["band"] == target_band and row["role"] == "target"]
    safety = [row for row in detail if row["role"] != "target"]
    earlier = {
        stage.config.target_band for stage in model.stages[:-1]
    }
    prior = [row for row in safety if row["band"] in earlier]
    cross_guard = [
        row for row in safety if row["family"] == "cross_guard_mixture"
    ]
    cross_target_strata = {
        family: _ratio_summary([
            row for row in target if row["family"] == family
        ])
        for family in sorted({
            row["family"] for row in target
            if row["family"].startswith("cross_target_mixture_")
        })
    }
    failures = []
    if not target or not safety:
        failures.append("audit lacks target or safety fields")
    if target and np.mean([row["model_over_prefix"] for row in target]) > (
        1.0 - config.audit.minimum_target_gain
    ):
        failures.append("target-band mean gain is below the configured minimum")
    if safety and max(row["model_over_prefix"] for row in safety) > (
        1.0 + config.audit.maximum_safety_regression
    ):
        failures.append("general safety regression exceeds tolerance")
    if prior and max(row["model_over_prefix"] for row in prior) > (
        1.0 + config.audit.maximum_prior_band_regression
    ):
        failures.append("accepted earlier-band regression exceeds tolerance")
    if cross_guard and max(row["model_over_prefix"] for row in cross_guard) > 1.01:
        failures.append("cross-band guard regression exceeds one percent")
    if any(
        value["row_residual_max"] > config.audit.row_tolerance
        or value["column_residual_max"] > config.audit.column_tolerance
        for value in structures.values()
    ):
        failures.append("correction projection constraints failed")
    report = {
        "format": "remapgnn.bilinear_audit",
        "schema_version": 1,
        "passed": not failures,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "stage": final_stage.name,
        "pairs": names,
        "promotion": {
            "passed": not failures,
            "failures": failures,
            "target": _ratio_summary(target),
            "safety": _ratio_summary(safety),
            "prior_bands": _ratio_summary(prior),
            "cross_guard": _ratio_summary(cross_guard),
            "cross_target_strata": cross_target_strata,
        },
        "structures": structures,
        "detail": detail,
    }
    output = (
        Path(config.paths.reports)
        / f"bilinear_progressive_{final_stage.name}_{args.tag}.json"
    )
    _write_json(output, report, args.overwrite)
    print(f"AUDIT_DONE passed={not failures} report={output}")
    for failure in failures:
        print(f"  FAIL: {failure}")
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
