#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from remapgnn_next.checkpoint import validate_production_manifest
from remapgnn_next.provenance import file_sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Bind a verified clean checkpoint to equivalence evidence")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fv-checkpoint", required=True)
    parser.add_argument("--equivalence-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--production-pointer", default="_next/configs/production.json")
    args = parser.parse_args()

    checkpoint, fv, report = map(Path, (
        args.checkpoint, args.fv_checkpoint, args.equivalence_report
    ))
    manifest = {
        "format": "remapgnn.production_manifest",
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "fv_checkpoint": {"path": str(fv), "sha256": file_sha256(fv)},
        "equivalence_report": {"path": str(report), "sha256": file_sha256(report)},
    }
    atomic_json(args.output, manifest)
    validate_production_manifest(checkpoint, args.output)
    if args.activate:
        atomic_json(args.production_pointer, {
            "schema_version": 1, "approved": True,
            "checkpoint": str(checkpoint), "manifest": str(args.output),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "manifest_sha256": file_sha256(args.output),
        })
    print(f"PRODUCTION_MANIFEST checkpoint={checkpoint} manifest={args.output} activated={args.activate}")


if __name__ == "__main__":
    main()
