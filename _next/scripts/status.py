#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

from remapgnn_next.checkpoint import validate_production_manifest
from remapgnn_next.config import load_config
from remapgnn_next.provenance import file_sha256


def _load(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    config = load_config("_next/configs/progressive.json")
    pointer = json.loads(Path("_next/configs/production.json").read_text())
    checkpoint = Path(pointer["checkpoint"])
    pack = _load(checkpoint)
    approved = bool(pointer.get("approved", False))
    manifest_ok = False
    if approved:
        try:
            validate_production_manifest(checkpoint, pointer["manifest"])
            manifest_ok = True
        except Exception:
            manifest_ok = False
    print("Active workflow: _next")
    print(f"Production checkpoint: {checkpoint}")
    print(f"Checkpoint SHA256: {file_sha256(checkpoint)}")
    print(f"Production approved by detached manifest: {approved and manifest_ok}")
    print(f"Production manifest: {pointer['manifest']} (valid={manifest_ok})")
    print(f"Stages: {len(pack.get('stages', ())) }")
    print(f"Final converted stage selected identity: {bool(pack.get('selected_identity', False))}")
    candidate = Path(config.paths.checkpoint_path)
    if candidate.is_file():
        saved = _load(candidate)
        print(
            "Candidate checkpoint: "
            f"{candidate} (completed={bool(saved.get('completed', False))}, "
            f"phase={saved.get('phase')}, "
            f"selected_identity={bool(saved.get('selected_identity', False))})"
        )
    else:
        print(f"Candidate checkpoint: none ({candidate} does not exist)")


if __name__ == "__main__":
    main()
