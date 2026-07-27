#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from remapgnn_next.bilinear import (
    apply_conservative_bilinear, build_bilinear_pair,
    compute_edge_normalization,
)
from remapgnn_next.config import load_config
from remapgnn_next.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(
        description="Validate all fixed conservative bilinear baselines"
    )
    parser.add_argument(
        "--config", default="_next/configs/bilinear_progressive.json"
    )
    parser.add_argument("--output", default="_next/reports/bilinear_baseline_check.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.schema_version != 5:
        raise ValueError("baseline-check requires a schema-5 config")
    normalization = compute_edge_normalization(
        [config.paths.edge_path(pair) for pair in config.pair_roles["train"]],
        config.features.edge,
    )
    results = {}
    for pair_name in config.pairs(
        "train", "selection", "protected", "external_resolution"
    ):
        pair = build_bilinear_pair(
            pair_name, config.paths.edge_path(pair_name),
            config.paths.bilinear_map_path(pair_name),
            feature_names=config.features.edge,
            normalization=normalization,
            correction_reference_kind=config.baseline.correction_reference,
            bilinear_reference_fraction=config.baseline.bilinear_reference_fraction,
            quadrature_resolution=1,
            smoother_neighbors=2,
        )
        generator = torch.Generator().manual_seed(config.seed)
        source = torch.stack((
            torch.ones(pair.n_src),
            -2.5 * torch.ones(pair.n_src) + 0.125,
            torch.randn(pair.n_src, generator=generator),
        ))
        adjusted, raw, shift = apply_conservative_bilinear(pair, source)
        source_integral = (
            pair.area_src.double().view(1, -1) * source.double()
        ).sum(1)
        target_integral = (
            pair.area_tgt.double().view(1, -1) * adjusted.double()
        ).sum(1)
        relative = (
            (target_integral - source_integral).abs()
            / source_integral.abs().clamp_min(1.0)
        )
        constant_error = float((adjusted[:2] - source[:2, :1]).abs().max())
        results[pair_name] = {
            "bilinear_map": str(config.paths.bilinear_map_path(pair_name)),
            "bilinear_map_sha256": file_sha256(
                config.paths.bilinear_map_path(pair_name)
            ),
            "n_source": pair.n_src,
            "n_target": pair.n_tgt,
            "bilinear_edges": pair.base_operator.n_edges,
            "correction_edges": pair.fv_operator.n_edges,
            "constant_max_abs": constant_error,
            "conservation_relative_max": float(relative.max()),
            "shift_max_abs": float(shift.abs().max()),
            "passed": constant_error <= 1.0e-6 and float(relative.max()) <= 1.0e-7,
        }
        print(
            f"[{pair_name}] passed={results[pair_name]['passed']} "
            f"constant={constant_error:.3e} "
            f"conservation={float(relative.max()):.3e}",
            flush=True,
        )
    report = {
        "format": "remapgnn.bilinear_baseline_check",
        "schema_version": 1,
        "passed": all(value["passed"] for value in results.values()),
        "edge_normalization": normalization,
        "pairs": results,
    }
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"baseline report exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"BASELINE_CHECK passed={report['passed']} report={output}")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
