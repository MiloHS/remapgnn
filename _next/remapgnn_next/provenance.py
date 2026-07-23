from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch


def file_sha256(path: str | Path, chunk_size=2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"state entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def authenticated_load(path: str | Path, expected_sha256: str | None = None):
    path = Path(path)
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"checkpoint hash mismatch: expected {expected_sha256}, got {actual}")
    try:
        pack = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(path, map_location="cpu")
    return pack, actual
