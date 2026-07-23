#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

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
    checkpoint = Path(config.model.source_checkpoint)
    pack = _load(checkpoint)
    equivalence = pack.get("conversion_checks", {}).get("equivalence", {})
    print("Active workflow: _next")
    print(f"Production checkpoint: {checkpoint}")
    print(f"Checkpoint SHA256: {file_sha256(checkpoint)}")
    print(f"Production approved: {bool(pack.get('production', False))}")
    print(f"Equivalence passed: {bool(equivalence.get('passed', False))}")
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
