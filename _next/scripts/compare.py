#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch

from remapgnn_next.checkpoint import (
    validate_fv_reference, validate_production_manifest,
)
from remapgnn_next.comparison import (
    REAL_FIELDS, evaluate_methods, frequency_summary, git_state,
    load_comparison_model, parse_methods, plot_frequency, plot_spatial,
    request_hash, safe_name, select_fields, write_values_netcdf,
)
from remapgnn_next.config import load_config
from remapgnn_next.evaluation import load_map_operator
from remapgnn_next.fv import build_pair_from_files
from remapgnn_next.provenance import authenticated_load, file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "_next/configs/progressive.json"
PRODUCTION_POINTER = ROOT / "_next/configs/production.json"
GENERATED_ROOT = ROOT / ".generated/comparisons"
CACHE_ROOT = ROOT / ".generated/cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def atomic_csv(path, rows):
    frame = pd.DataFrame(rows)
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def production_pointer():
    return json.loads(PRODUCTION_POINTER.read_text())


def resolve_checkpoint(value, *, listing=False):
    pointer = production_pointer()
    if value:
        return Path(value), pointer
    if pointer.get("approved") or listing:
        return Path(pointer["checkpoint"]), pointer
    raise ValueError(
        "production is not approved; pass --checkpoint explicitly for analysis"
    )


def available_pairs(config):
    suffix = f"_{config.paths.graph_suffix}.parquet"
    prefix = "edge_dataset_"
    pairs = []
    for path in Path(config.paths.analysis).glob(f"{prefix}*{suffix}"):
        name = path.name
        pair = name[len(prefix):-len(suffix)]
        if config.paths.map_path(pair).is_file():
            pairs.append(pair)
    return sorted(set(pairs))


def available_real_fields(config, pair):
    source, target = config.paths.real_field_paths(pair)
    if not source.is_file() or not target.is_file():
        return []
    import xarray as xr
    with xr.open_dataset(source) as left, xr.open_dataset(target) as right:
        return [name for name in REAL_FIELDS if name in left and name in right]


def checkpoint_status(checkpoint, pointer, manifest=None):
    approved = bool(
        pointer.get("approved")
        and Path(pointer.get("checkpoint", "")).resolve() == checkpoint.resolve()
    )
    manifest_path = Path(manifest) if manifest else (
        Path(pointer["manifest"]) if approved else None
    )
    valid = False
    if manifest_path is not None:
        validate_production_manifest(checkpoint, manifest_path)
        valid = True
    return approved, valid, manifest_path


def list_command(args):
    config = load_config(args.config)
    checkpoint, pointer = resolve_checkpoint(args.checkpoint, listing=True)
    model, _, _, checkpoint_sha = load_comparison_model(checkpoint, config)
    print(f"checkpoint: {checkpoint} ({checkpoint_sha})")
    print(f"stages: {', '.join(stage.name for stage in model.stages)}")
    print("methods: fv, " + ", ".join(
        f"stage:{stage.name}" for stage in model.stages
    ) + ", np2")
    pairs = [args.pair] if args.pair else available_pairs(config)
    print("pairs:")
    for pair in pairs:
        if pair not in available_pairs(config):
            print(f"  {pair}: unavailable")
            continue
        np2 = Path(config.paths.maps) / f"map_{pair}_conserve_np2.nc"
        real = available_real_fields(config, pair)
        print(
            f"  {pair}: np2={'yes' if np2.is_file() else 'no'}; "
            f"real={','.join(real) if real else 'none'}"
        )
    print("analytic fields: smooth1, smooth2")
    print("harmonics: harmonic:DEGREE:ORDER")


def output_directory(args, request):
    digest = request_hash(request)
    default = GENERATED_ROOT / f"{safe_name(args.pair)}_{digest}"
    output = Path(args.output).resolve() if args.output else default.resolve()
    generated = GENERATED_ROOT.resolve()
    if output != generated and generated not in output.parents:
        raise ValueError(f"comparison output must remain under {generated}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"comparison output exists: {output}; pass --overwrite to replace files"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output, digest


def run_command(args):
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    checkpoint, pointer = resolve_checkpoint(args.checkpoint)
    model, progressive_pack, source_path, checkpoint_sha = load_comparison_model(
        checkpoint, config
    )
    approved, manifest_valid, manifest_path = checkpoint_status(
        checkpoint, pointer, args.manifest
    )
    fv_pack, fv_sha = authenticated_load(config.fv_checkpoint)
    validate_fv_reference(progressive_pack, config.fv_checkpoint, fv_sha)
    if args.pair not in available_pairs(config):
        raise ValueError(
            f"pair {args.pair!r} lacks an edge dataset or base map; "
            "use `./next compare list`"
        )
    methods = parse_methods(
        args.methods, [stage.name for stage in model.stages]
    )
    np2_path = Path(config.paths.maps) / f"map_{args.pair}_conserve_np2.nc"
    if "np2" in methods and not np2_path.is_file():
        raise FileNotFoundError(f"np2 map is unavailable: {np2_path}")
    edge_path, map_path = (
        config.paths.edge_path(args.pair), config.paths.map_path(args.pair)
    )
    pair = build_pair_from_files(
        args.pair, edge_path, map_path, fv_pack, progressive_pack,
        device="cpu", quadrature_resolution=config.panel.quadrature_resolution,
        smoother_neighbors=config.panel.smoother_neighbors,
    )
    fields = select_fields(
        config, pair, model, args.field, band=args.band,
        profile_degrees=args.profile_degrees,
        profile_modes=args.profile_modes, field_set=args.field_set,
    )
    request = {
        "format": "remapgnn.comparison_request",
        "schema_version": 1,
        "pair": args.pair,
        "methods": list(methods),
        "fields": list(args.field),
        "band": None if args.band is None else list(args.band),
        "field_set": args.field_set,
        "profile_degrees": args.profile_degrees,
        "profile_modes": args.profile_modes,
        "device": args.device,
        "batch_size": args.batch_size,
        "git": git_state(ROOT),
        "checkpoint": {
            "path": str(checkpoint), "sha256": checkpoint_sha,
            "approved_production": approved,
            "manifest_valid": manifest_valid,
            "manifest": None if manifest_path is None else str(manifest_path),
        },
        "progressive_source": {
            "path": str(source_path), "sha256": file_sha256(source_path),
        },
        "inputs": {
            "config": {"path": str(config.path), "sha256": file_sha256(config.path)},
            "fv": {"path": config.fv_checkpoint, "sha256": fv_sha},
            "edge": {"path": str(edge_path), "sha256": file_sha256(edge_path)},
            "map": {"path": str(map_path), "sha256": file_sha256(map_path)},
            "np2": (
                {"path": str(np2_path), "sha256": file_sha256(np2_path)}
                if "np2" in methods else None
            ),
            "real": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in config.paths.real_field_paths(args.pair)
                if path.is_file() and (
                    args.field_set is not None
                    or any(x.startswith("real:") for x in args.field)
                )
            ],
        },
    }
    output, digest = output_directory(args, request)
    np2 = load_map_operator(np2_path) if "np2" in methods else None
    rows, predictions = evaluate_methods(
        model, pair, fields, methods, np2, device=args.device,
        batch_size=args.batch_size,
    )
    profile = frequency_summary(rows)
    atomic_json(output / "request.json", {**request, "request_hash": digest})
    atomic_csv(output / "metrics.csv", rows)
    atomic_csv(output / "frequency_profile.csv", profile)
    values_path = write_values_netcdf(
        output / "values.nc", pair, fields, methods, predictions
    )
    figures = []
    for index in torch.where(fields.explicit)[0].tolist():
        figures.append(str(plot_spatial(
            output / f"spatial_{safe_name(fields.display_names[index])}.png",
            pair, fields, index, methods, predictions,
        )))
    frequency_path = plot_frequency(
        output / "frequency_profile.png", profile, model.stages
    )
    if frequency_path is not None:
        figures.append(str(frequency_path))
    frame = pd.DataFrame(rows)
    columns = [
        "field", "method", "nu", "area_relative_l2", "area_rmse",
        "max_absolute_error", "conservation_error", "error_over_fv",
        "error_over_np2",
    ]
    print(frame[columns].to_string(index=False))
    print(f"COMPARISON_DONE output={output}")
    print(f"values={values_path if values_path else 'none (no explicit fields)'}")
    for figure in figures:
        print(f"figure={figure}")


def parser():
    result = argparse.ArgumentParser(
        description="Compare FV, progressive prefixes, and np2 on chosen fields"
    )
    result.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = result.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list available methods and inputs")
    listing.add_argument("--checkpoint")
    listing.add_argument("--pair")
    listing.set_defaults(function=list_command)
    run = commands.add_parser("run", help="run a comparison")
    run.add_argument("--checkpoint")
    run.add_argument("--manifest")
    run.add_argument("--pair", required=True)
    run.add_argument("--methods", nargs="+", required=True)
    run.add_argument("--field", action="append", default=[])
    run.add_argument("--band", nargs=2, type=float, metavar=("LOWER", "UPPER"))
    run.add_argument("--field-set")
    run.add_argument("--profile-degrees", type=int, default=16)
    run.add_argument("--profile-modes", type=int, default=3)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    run.add_argument("--output")
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(function=run_command)
    return result


def main():
    config_path = DEFAULT_CONFIG
    if "--config" in sys.argv:
        position = sys.argv.index("--config")
        if position + 1 >= len(sys.argv):
            raise ValueError("--config requires a path")
        config_path = Path(sys.argv[position + 1])
    if json.loads(Path(config_path).read_text()).get("schema_version") == 5:
        from compare_bilinear import main as bilinear_main
        return bilinear_main()
    args = parser().parse_args()
    if hasattr(args, "profile_degrees") and (
        args.profile_degrees <= 0 or args.profile_modes <= 0 or args.batch_size <= 0
    ):
        raise ValueError("profile degree/mode counts and batch size must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
