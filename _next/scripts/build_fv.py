#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from remapgnn_next.config import load_config
from remapgnn_next.checkpoint import (
    CLEAN_FV_FORMAT, CLEAN_PROGRESSIVE_FORMAT, validate_fv_reference,
)
from remapgnn_next.fv import build_pair_from_files
from remapgnn_next.provenance import authenticated_load


def main():
    parser = argparse.ArgumentParser(description="Build one clean frozen FV operator")
    parser.add_argument("--config", default="_next/configs/progressive.json")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--edge")
    parser.add_argument("--map")
    parser.add_argument("--fv-checkpoint")
    parser.add_argument("--progressive-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    fv_path = args.fv_checkpoint or config.fv_checkpoint
    progressive_path = args.progressive_checkpoint or config.progressive_checkpoint
    edge_path = args.edge or config.paths.edge_path(args.pair)
    map_path = args.map or config.paths.map_path(args.pair)
    fv, fv_sha256 = authenticated_load(fv_path)
    progressive, _ = authenticated_load(progressive_path)
    if fv.get("format") != CLEAN_FV_FORMAT or progressive.get("format") != CLEAN_PROGRESSIVE_FORMAT:
        raise ValueError("clean checkpoints are required")
    validate_fv_reference(progressive, fv_path, fv_sha256)
    pair = build_pair_from_files(
        args.pair, edge_path, map_path, fv, progressive, device=args.device,
        quadrature_resolution=config.panel.quadrature_resolution,
        smoother_neighbors=config.panel.smoother_neighbors,
    )
    torch.save({"format": "remapgnn.clean_operator", "operator": pair.fv_operator}, args.output)
    print(f"wrote {args.output}: {pair.n_src}->{pair.n_tgt}, {pair.fv_operator.n_edges} edges")


if __name__ == "__main__":
    main()
