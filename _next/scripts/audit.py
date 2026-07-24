#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from remapgnn_next.config import load_config
from remapgnn_next.checkpoint import (
    CLEAN_PROGRESSIVE_FORMAT, CLEAN_TRAINING_FORMAT,
    load_progressive_checkpoint, load_training_checkpoint, validate_fv_reference,
)
from remapgnn_next.evaluation import audit_experiment
from remapgnn_next.fv import build_pair_from_files
from remapgnn_next.provenance import authenticated_load
from remapgnn_next.provenance import canonical_json_sha256


def main():
    parser = argparse.ArgumentParser(description="Audit all clean progressive prefixes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--allow-protected", action="store_true")
    parser.add_argument("--tag", default="development")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-production", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    print(
        f"AUDIT_INPUT config={Path(args.config).resolve()} "
        f"checkpoint={checkpoint.resolve()}",
        flush=True,
    )
    pack, _ = authenticated_load(checkpoint)
    if pack.get("format") == CLEAN_TRAINING_FORMAT:
        if args.require_production:
            raise ValueError("--require-production rejects training checkpoints")
        if canonical_json_sha256(pack.get("config")) != canonical_json_sha256(config.to_dict()):
            raise ValueError("audit config differs from candidate's saved scientific config")
        model, progressive_pack, source = load_training_checkpoint(pack)
    elif pack.get("format") == CLEAN_PROGRESSIVE_FORMAT:
        source = checkpoint; progressive_pack = pack
        model, _ = load_progressive_checkpoint(
            source, require_production=args.require_production,
            manifest_path=args.manifest if args.require_production else None,
        )
    else:
        raise ValueError("unsupported clean checkpoint")
    fv_pack, fv_sha256 = authenticated_load(config.fv_checkpoint)
    validate_fv_reference(progressive_pack, config.fv_checkpoint, fv_sha256)
    names = args.pairs or list(config.pair_roles["selection"])
    protected = set(config.pair_roles.get("protected", ())) | set(config.pair_roles.get("external_resolution", ()))
    requested_protected = protected & set(names)
    if requested_protected and not args.allow_protected:
        raise ValueError(f"protected pairs require --allow-protected: {sorted(requested_protected)}")
    pairs = {}
    for name in names:
        started = time.time()
        pairs[name] = build_pair_from_files(
            name, config.paths.edge_path(name), config.paths.map_path(name), fv_pack,
            progressive_pack, device="cpu", quadrature_resolution=4 if args.smoke else config.panel.quadrature_resolution,
            smoother_neighbors=config.panel.smoother_neighbors,
        )
        print(f"[{name}] built={time.time()-started:.1f}s", flush=True)
    report = audit_experiment(model.to(args.device), config, pairs, checkpoint,
                              device=args.device, smoke=args.smoke, tag=args.tag,
                              overwrite=args.overwrite)
    print(f"AUDIT_DONE passed={report.promotion['passed']} failures={len(report.promotion['failures'])}")
    for failure in report.promotion["failures"]: print(f"  FAIL: {failure}")
    raise SystemExit(0 if report.promotion["passed"] else 2)


if __name__ == "__main__":
    main()
