#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from remapgnn_next.config import load_config
from remapgnn_next.checkpoint import (
    CLEAN_FV_FORMAT, build_training_model, validate_fv_reference,
)
from remapgnn_next.fv import build_pair_from_files
from remapgnn_next.provenance import authenticated_load
from remapgnn_next.training import SequentialTrainer, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train one clean ordered correction stage")
    parser.add_argument("--config", default="_next/configs/progressive.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", help="stage name; must match model.train_stage")
    parser.add_argument("--checkpoint", help="initial clean progressive conversion")
    parser.add_argument("--output", help="training checkpoint destination")
    args = parser.parse_args()

    config = load_config(args.config)
    source = Path(args.checkpoint or config.model.source_checkpoint)
    output = Path(args.output or config.paths.checkpoint_path)
    history = config.paths.history_path
    if args.smoke:
        output = output.with_name(output.stem + "_smoke.pt")
        history = history.with_name(history.stem + "_smoke.csv")
    fv_pack, fv_sha256 = authenticated_load(config.fv_checkpoint)
    if fv_pack.get("format") != CLEAN_FV_FORMAT:
        raise ValueError("training requires a clean FV checkpoint")
    set_seed(config.seed)
    model, progressive_pack, initialization = build_training_model(config, source)
    validate_fv_reference(progressive_pack, config.fv_checkpoint, fv_sha256)
    stage_name = args.stage or config.model.train_stage
    if stage_name != config.model.train_stage:
        raise ValueError(
            f"--stage must match configured model.train_stage {config.model.train_stage!r}"
        )
    stage_index = next(
        index for index, stage in enumerate(model.stages) if stage.name == stage_name
    )
    train_names = list(config.pair_roles["train"][:1] if args.smoke else config.pair_roles["train"])
    selection_names = train_names if args.smoke else list(config.pair_roles["selection"])
    pairs = {}
    for name in dict.fromkeys(train_names + selection_names):
        started = time.time()
        pairs[name] = build_pair_from_files(
            name, config.paths.edge_path(name), config.paths.map_path(name), fv_pack,
            progressive_pack, device="cpu",
            quadrature_resolution=4 if args.smoke else config.panel.quadrature_resolution,
            smoother_neighbors=config.panel.smoother_neighbors,
        )
        print(f"[{name}] {pairs[name].n_src}->{pairs[name].n_tgt} edges={pairs[name].fv_operator.n_edges} built={time.time()-started:.1f}s", flush=True)
    trainer = SequentialTrainer(
        model, stage_index, config=config,
        train_pairs={name: pairs[name] for name in train_names},
        selection_pairs={name: pairs[name] for name in selection_names},
        source_checkpoint=source, model_initialization=initialization,
        output=output, history_path=history, device=args.device,
    )
    result = trainer.run(resume=args.resume, smoke=args.smoke)
    print(f"TRAIN_DONE checkpoint={output} selected_identity={result['selected_identity']}", flush=True)


if __name__ == "__main__":
    main()
