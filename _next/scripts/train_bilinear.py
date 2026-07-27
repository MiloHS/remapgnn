#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from remapgnn_next.bilinear import (
    build_bilinear_pair, compute_edge_normalization,
)
from remapgnn_next.checkpoint import load_bilinear_training_checkpoint
from remapgnn_next.config import load_config
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
    train_stage = ConservativeCorrectionStage(config.stages[index])
    train_stage.set_training_phase("frozen")
    model = ProgressiveRemapper(None, [*prefix, train_stage])
    initialization = {
        "baseline": config.baseline.__dict__,
        "runtime_data": {
            "edge_features": list(config.features.edge),
            "edge_normalization": normalization,
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
    )
    set_seed(config.seed)
    pairs = {}
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
        )
        print(
            f"[{name}] {pairs[name].n_src}->{pairs[name].n_tgt} "
            f"built={time.time()-started:.1f}s", flush=True,
        )
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
    )
    result = trainer.run(resume=args.resume, smoke=args.smoke)
    print(
        f"TRAIN_DONE checkpoint={output} "
        f"selected_identity={result['selected_identity']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
