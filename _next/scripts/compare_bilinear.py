#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from remapgnn_next.band_panels import _harmonics
from remapgnn_next.bilinear import apply_conservative_bilinear, build_bilinear_pair
from remapgnn_next.checkpoint import load_bilinear_training_checkpoint
from remapgnn_next.comparison import (
    _analytic, _combine, _from_batch, _harmonic, _metrics, _real,
    plot_spatial, request_hash, safe_name, write_values_netcdf,
)
from remapgnn_next.config import load_config
from remapgnn_next.evaluation import load_map_operator
from remapgnn_next.fv import build_pair_from_files
from remapgnn_next.provenance import authenticated_load, file_sha256
from remapgnn_next.sparse import apply_operator


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / ".generated/comparisons"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".generated/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".generated/cache"))


def _methods(values, stages):
    allowed = {
        "bilinear_raw", "bilinear", "fv", "np2",
        *(f"stage:{name}" for name in stages),
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}; available: {sorted(allowed)}")
    return tuple(dict.fromkeys(values))


def _fields(config, pair, model, selectors, band_name):
    parts = []
    for selector in selectors:
        kind, _, payload = selector.partition(":")
        if kind == "harmonic":
            degree, order = map(int, payload.split(":"))
            part = _harmonic(pair, degree, order, 1.0, explicit=True)
            matching = [
                band.name for band in config.bands
                if band.degree_min <= degree <= band.degree_max
            ]
            part = replace(
                part,
                batch=replace(
                    part.batch,
                    frequency=torch.full((1,), float("nan"), dtype=torch.float32),
                    degrees=torch.tensor([degree], dtype=torch.long),
                    bands=matching or ["out_of_range"],
                ),
            )
            parts.append(part)
        elif kind == "analytic":
            parts.append(_analytic(pair, payload))
        elif kind == "real":
            parts.append(_real(config, pair, payload))
        else:
            raise ValueError(f"unknown field selector {selector!r}")
    if band_name:
        band = config.band(band_name)
        stage = next(
            (value for value in model.stages if value.config.target_band == band_name),
            model.stages[-1],
        )
        batch = _harmonics(
            config, pair, band, "audit", config.seed,
            config.panel.audit_modes_per_degree,
            config.panel.audit_max_degrees,
            "analysis", "band_mode",
        )
        parts.append(_from_batch(batch, explicit=False))
    return _combine(parts)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_band(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for row in rows:
        if row["degree"] >= 0:
            groups.setdefault((row["method"], row["degree"]), []).append(
                row["area_relative_l2"]
            )
    if not groups:
        return None
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in dict.fromkeys(row["method"] for row in rows):
        values = sorted(
            (degree, np.mean(errors))
            for (name, degree), errors in groups.items() if name == method
        )
        if values:
            axis.plot(
                [item[0] for item in values],
                [item[1] for item in values],
                marker="o", label=method,
            )
    axis.set(
        xlabel="spherical-harmonic degree",
        ylabel="area-weighted relative L2 error",
        yscale="log",
        title="Error by configured degree band",
    )
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


@torch.no_grad()
def run(args):
    config = load_config(args.config)
    model, pack = load_bilinear_training_checkpoint(args.checkpoint)
    normalization = pack["model_initialization"]["runtime_data"]["edge_normalization"]
    pair = build_bilinear_pair(
        args.pair, config.paths.edge_path(args.pair),
        config.paths.bilinear_map_path(args.pair),
        feature_names=config.features.edge, normalization=normalization,
        correction_reference_kind=config.baseline.correction_reference,
        bilinear_reference_fraction=config.baseline.bilinear_reference_fraction,
        quadrature_resolution=config.panel.quadrature_resolution,
        smoother_neighbors=config.panel.smoother_neighbors,
    )
    fields = _fields(config, pair, model, args.field, args.band)
    methods = _methods(args.methods, [stage.name for stage in model.stages])
    device = torch.device(args.device)
    pair_device = pair.to(device)
    source = fields.batch.source.to(device)
    adjusted, raw, _ = apply_conservative_bilinear(pair_device, source)
    values = {"bilinear_raw": raw, "bilinear": adjusted}
    if any(method.startswith("stage:") for method in methods):
        _, diagnostics = model.to(device).eval()(
            pair_device, source, return_diagnostics=True
        )
        values.update({
            f"stage:{stage.name}": stage.output
            for stage in diagnostics.stages
        })
    if "fv" in methods:
        fv_pack, _ = authenticated_load(config.benchmarks.fv_checkpoint)
        fv_progressive, _ = authenticated_load(
            config.benchmarks.fv_progressive_checkpoint
        )
        fv_pair = build_pair_from_files(
            args.pair, config.paths.edge_path(args.pair),
            config.paths.map_path(args.pair), fv_pack, fv_progressive,
            device="cpu", quadrature_resolution=2,
            smoother_neighbors=config.panel.smoother_neighbors,
        )
        values["fv"] = apply_operator(
            fv_pair.fv_operator.to(device), source
        )
    if "np2" in methods:
        np2 = load_map_operator(
            Path(config.paths.maps)
            / f"map_{args.pair}_{config.benchmarks.np2_suffix}.nc"
        )
        values["np2"] = apply_operator(np2.to(device), source)
    rows = []
    predictions = {method: {} for method in methods}
    for method in methods:
        host = values[method].cpu()
        for index in range(source.shape[0]):
            if bool(fields.explicit[index]):
                predictions[method][index] = host[index]
            rows.append({
                "pair": args.pair,
                "field": fields.display_names[index],
                "source_key": fields.batch.source_keys[index],
                "family": fields.batch.families[index],
                "band": (
                    fields.batch.bands[index]
                    if fields.batch.bands else "unbanded"
                ),
                "degree": (
                    int(fields.batch.degrees[index])
                    if fields.batch.degrees is not None
                    else int(fields.batch.labels[index][0])
                ),
                "method": method,
                **_metrics(
                    host[index], fields.batch.truth[index],
                    fields.batch.source[index], pair.area_tgt, pair.area_src,
                    fields.scales[index], fields.offsets[index],
                ),
            })
    request = {
        "format": "remapgnn.bilinear_comparison",
        "schema_version": 1,
        "pair": args.pair,
        "methods": methods,
        "fields": args.field,
        "band": args.band,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "inputs": {
            "edge": file_sha256(config.paths.edge_path(args.pair)),
            "bilinear": file_sha256(config.paths.bilinear_map_path(args.pair)),
        },
    }
    digest = request_hash(request)
    output = (
        Path(args.output).resolve() if args.output
        else (GENERATED / f"{safe_name(args.pair)}_bilinear_{digest}").resolve()
    )
    if GENERATED.resolve() not in output.parents:
        raise ValueError("comparison output must remain under .generated/comparisons")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    _write_csv(output / "metrics.csv", rows)
    write_values_netcdf(output / "values.nc", pair, fields, methods, predictions)
    for index in torch.where(fields.explicit)[0].tolist():
        plot_spatial(
            output / f"spatial_{safe_name(fields.display_names[index])}.png",
            pair, fields, index, methods, predictions,
        )
    _plot_band(output / "band_profile.png", rows)
    for row in rows:
        print(
            f"{row['field']:20s} {row['method']:18s} "
            f"band={row['band']:16s} degree={row['degree']:3d} "
            f"rel_l2={row['area_relative_l2']:.8g}"
        )
    print(f"COMPARISON_DONE output={output}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare conservative bilinear progressive methods"
    )
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--checkpoint", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--pair", required=True)
    run_parser.add_argument("--methods", nargs="+", required=True)
    run_parser.add_argument("--field", action="append", default=[])
    run_parser.add_argument("--band")
    run_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run_parser.add_argument("--output")
    run_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    model, _ = load_bilinear_training_checkpoint(args.checkpoint)
    if args.command == "list":
        print("methods: bilinear_raw, bilinear, " + ", ".join(
            f"stage:{stage.name}" for stage in model.stages
        ) + ", fv, np2")
        print("bands: " + ", ".join(
            f"{band.name}={band.degree_min}-{band.degree_max}"
            for band in config.bands
        ))
        print("fields: harmonic:DEGREE:ORDER, analytic:smooth1, "
              "analytic:smooth2, real:NAME")
        return
    if not args.field and not args.band:
        raise ValueError("select at least one --field or --band")
    run(args)


if __name__ == "__main__":
    main()
