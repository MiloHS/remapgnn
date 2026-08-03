#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from remapgnn_next.bilinear import (
    build_bilinear_pair, compute_edge_normalization,
    compute_geometry_normalization, pair_feature_options,
)
from remapgnn_next.checkpoint import load_bilinear_training_checkpoint
from remapgnn_next.config import load_config
from remapgnn_next.evaluation import load_map_operator
from remapgnn_next.progressive import (
    ConservativeCorrectionStage, ProgressiveRemapper,
)
from remapgnn_next.provenance import (
    build_bilinear_run_manifest, canonical_json_sha256, file_sha256,
)
from remapgnn_next.training import SequentialTrainer, set_seed


def _audit_source(path, checkpoint):
    report = json.loads(Path(path).read_text())
    passed = report.get("passed", report.get("promotion", {}).get("passed", False))
    recorded = (
        report.get("checkpoint_sha256")
        or report.get("provenance", {}).get("checkpoint_sha256")
    )
    if not passed:
        raise ValueError("source audit did not pass")
    if recorded != file_sha256(checkpoint):
        raise ValueError("source audit names a different checkpoint")


def _assemble(config, stage_name, source, source_audit, smoke):
    names = [stage.name for stage in config.stages]
    if stage_name not in names:
        raise ValueError(f"unknown stage {stage_name!r}; available: {names}")
    index = names.index(stage_name)
    prior_geometry_normalization = {}
    if index == 0:
        if source is not None or source_audit is not None:
            raise ValueError("the first bilinear stage must start without a checkpoint")
        prefix = []
        normalization = compute_edge_normalization(
            [config.paths.edge_path(pair) for pair in config.pair_roles["train"]],
            config.features.edge,
        )
    else:
        if source is None:
            raise ValueError("later bilinear stages require --checkpoint")
        if source_audit is None and not smoke:
            raise ValueError("later bilinear stages require --source-audit")
        if source_audit is not None:
            _audit_source(source_audit, source)
        source_model, pack = load_bilinear_training_checkpoint(source)
        if pack.get("selected_identity") and not smoke:
            raise ValueError("cannot extend an identity-selected stage")
        if canonical_json_sha256(pack.get("config")) != canonical_json_sha256(
            config.to_dict()
        ):
            raise ValueError("source checkpoint uses a different scientific config")
        if len(source_model.stages) != index:
            raise ValueError("source checkpoint does not contain the exact prior prefix")
        expected = names[:index]
        if [stage.name for stage in source_model.stages] != expected:
            raise ValueError("source checkpoint stage order differs")
        prefix = list(source_model.stages)
        normalization = pack["model_initialization"]["runtime_data"][
            "edge_normalization"
        ]
        prior_geometry_normalization = copy.deepcopy(
            pack["model_initialization"]["runtime_data"].get(
                "geometry_normalization", {}
            )
        )
    train_stage = ConservativeCorrectionStage(config.stages[index])
    train_stage.set_training_phase("frozen")
    model = ProgressiveRemapper(None, [*prefix, train_stage])
    initialization = {
        "baseline": config.baseline.__dict__,
        "runtime_data": {
            "edge_features": list(config.features.edge),
            "edge_normalization": normalization,
            **(
                {"geometry_normalization": prior_geometry_normalization}
                if prior_geometry_normalization else {}
            ),
        },
        "source_checkpoint": (
            None if source is None
            else {"path": str(source), "sha256": file_sha256(source)}
        ),
        "source_audit": (
            None if source_audit is None
            else {"path": str(source_audit), "sha256": file_sha256(source_audit)}
        ),
    }
    return model, index, normalization, initialization


def main():
    parser = argparse.ArgumentParser(
        description="Train one categorical-band bilinear correction stage"
    )
    parser.add_argument(
        "--config", default="_next/configs/bilinear_progressive.json"
    )
    parser.add_argument("--stage")
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-audit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capability-only", action="store_true")
    parser.add_argument("--router-from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument("--history")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.schema_version != 5:
        raise ValueError("train_bilinear requires a schema-5 config")
    if args.all_stages:
        if not args.smoke:
            raise ValueError("--all-stages is restricted to smoke integration runs")
        previous = None
        for stage in (value.name for value in config.stages):
            output = (
                Path(config.paths.models)
                / f"bilinear_progressive_{stage}_smoke.pt"
            )
            command = [
                sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                "--config", args.config, "--stage", stage, "--smoke",
                "--device", args.device, "--output", str(output),
            ]
            if previous is not None:
                command.extend(("--checkpoint", str(previous)))
            subprocess.run(command, check=True)
            previous = output
        print(f"ALL_STAGE_SMOKE_DONE checkpoint={previous}")
        return
    if not args.stage:
        raise ValueError("--stage is required unless --all-stages is used")
    if args.capability_only and args.output is None:
        raise ValueError("--capability-only requires an explicit --output")
    if args.capability_only and args.resume:
        raise ValueError("--capability-only cannot be combined with --resume")
    if args.router_from and (args.capability_only or args.resume):
        raise ValueError("--router-from cannot be combined with capability-only or resume")
    if args.router_from and args.output is None:
        raise ValueError("--router-from requires an explicit --output")
    capability_pack = None
    capability_path = None
    if args.router_from:
        capability_path = Path(args.router_from)
        _, capability_pack = load_bilinear_training_checkpoint(
            capability_path, allow_analysis=True
        )
        if (
            not capability_pack.get("analysis_only", False)
            or not capability_pack.get("capability_selected", False)
        ):
            raise ValueError("--router-from requires an admitted capability-only checkpoint")
        expected = copy.deepcopy(config.to_dict())
        observed = copy.deepcopy(capability_pack["config"])
        for value in (expected, observed):
            for stage_value in value["stages"]:
                stage_value.pop("router_scope", None)
        if canonical_json_sha256(expected) != canonical_json_sha256(observed):
            raise ValueError(
                "router branch config may differ from capability only in router_scope"
            )
    source = None if args.checkpoint is None else Path(args.checkpoint)
    source_audit = None if args.source_audit is None else Path(args.source_audit)
    model, stage_index, normalization, initialization = _assemble(
        config, args.stage, source, source_audit, args.smoke
    )
    suffix = "_smoke" if args.smoke else ""
    output = Path(args.output) if args.output else (
        Path(config.paths.models)
        / f"bilinear_progressive_{args.stage}{suffix}.pt"
    )
    if output.suffix != ".pt" or (output.exists() and output.is_dir()):
        raise ValueError("--output must name a .pt checkpoint file, not a directory")
    history = Path(args.history) if args.history else output.with_name(
        output.stem + "_history.csv"
    )
    train_names = list(
        config.pair_roles["train"][:1]
        if args.smoke else config.pair_roles["train"]
    )
    selection_names = (
        train_names if args.smoke else list(dict.fromkeys(
            [
                *config.pair_roles["selection"],
                *config.panel.validation_train_pairs,
            ]
        ))
    )
    manifest = build_bilinear_run_manifest(
        config,
        pair_names=list(dict.fromkeys(train_names + selection_names)),
        stage=args.stage,
        normalization=normalization,
        source_checkpoint=source,
        source_audit=source_audit,
        smoke=args.smoke,
        analysis_mode=(
            "capability_only" if args.capability_only
            else "router_branch" if args.router_from else None
        ),
        capability_checkpoint=capability_path,
    )
    set_seed(config.seed)
    pairs = {}
    feature_options = pair_feature_options(model.stages)
    for name in dict.fromkeys(train_names + selection_names):
        started = time.time()
        pairs[name] = build_bilinear_pair(
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
            **feature_options,
        )
        print(
            f"[{name}] {pairs[name].n_src}->{pairs[name].n_tgt} "
            f"built={time.time()-started:.1f}s", flush=True,
        )
    train_stage = model.stages[stage_index]
    if train_stage.config.geometry_layout == "bilinear_v2":
        geometry_normalization = compute_geometry_normalization(
            (pairs[name] for name in train_names),
            train_stage.geometry_feature_indices,
        )
        train_stage.set_geometry_normalization(geometry_normalization)
        initialization["runtime_data"].setdefault(
            "geometry_normalization", {}
        )[train_stage.name] = geometry_normalization
        manifest["geometry_normalization"] = copy.deepcopy(
            initialization["runtime_data"]["geometry_normalization"]
        )
    selection_operators = {}
    for name in selection_names:
        path = (
            Path(config.paths.maps)
            / f"map_{name}_{config.benchmarks.np2_suffix}.nc"
        )
        if path.is_file():
            selection_operators[name] = load_map_operator(path)
    trainer = SequentialTrainer(
        model, stage_index, config=config,
        train_pairs={name: pairs[name] for name in train_names},
        selection_pairs={name: pairs[name] for name in selection_names},
        source_checkpoint=source,
        model_initialization=initialization,
        output=output,
        history_path=history,
        device=args.device,
        run_manifest=manifest,
        capability_source=capability_pack,
        selection_operators=selection_operators,
    )
    result = trainer.run(
        resume=args.resume, smoke=args.smoke,
        capability_only=args.capability_only,
    )
    print(
        f"TRAIN_DONE checkpoint={output} "
        f"selected_identity={result['selected_identity']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
