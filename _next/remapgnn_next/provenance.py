from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import sys
import platform


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
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def object_sha256(value) -> str:
    """Deterministic digest for nested checkpoint state, including tensors."""
    digest = hashlib.sha256()
    def visit(item, key="root"):
        digest.update(key.encode("utf-8") + b"\0")
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for name in sorted(item, key=str):
                visit(item[name], f"{key}.{name}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for index, child in enumerate(item):
                visit(child, f"{key}[{index}]")
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")
    visit(value)
    return digest.hexdigest()


def _file_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def build_run_manifest(config, source_checkpoint, *, pair_names, smoke):
    package = Path(__file__).parent
    root = package.parents[1]
    implementation = {
        str(path.resolve()): file_sha256(path)
        for path in sorted(package.glob("*.py"))
    }
    for relative in ("_next/scripts/train.py", "next", "jobs_next_train.pbs",
                     "jobs_next_audit.pbs"):
        path = (root / relative).resolve()
        if path.is_file():
            implementation[str(path)] = file_sha256(path)
    data = {}
    for pair in pair_names:
        real = []
        real_paths = config.paths.real_field_paths(pair)
        available_variables = []
        for path in real_paths:
            record = {"path": str(path), "available": path.is_file()}
            if path.is_file():
                record["sha256"] = file_sha256(path)
                import xarray as xr
                with xr.open_dataset(path) as dataset:
                    record["variables"] = sorted(
                        name for name in config.panel.real_fields if name in dataset
                    )
                    record["sizes"] = {name: int(value) for name, value in dataset.sizes.items()}
                available_variables.append(set(record["variables"]))
            real.append(record)
        included = sorted(set.intersection(*available_variables)) if len(available_variables) == 2 else []
        data[pair] = {
            "edge": _file_record(config.paths.edge_path(pair)),
            "map": _file_record(config.paths.map_path(pair)),
            "real": real,
            "requested_real_fields": list(config.panel.real_fields),
            "included_real_fields": included,
            "skipped_real_fields": sorted(set(config.panel.real_fields) - set(included)),
        }
    return {
        "format": "remapgnn.run_manifest", "schema_version": 1,
        "config": config.to_dict(),
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "config_file": _file_record(config.path) if config.path else None,
        "implementation_sha256": implementation,
        "data": data,
        "source_checkpoint": _file_record(source_checkpoint),
        "source_manifest": _file_record(config.model.source_manifest),
        "fv_checkpoint": _file_record(config.fv_checkpoint),
        "smoke": bool(smoke),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
        },
    }


def build_bilinear_run_manifest(
    config, *, pair_names, stage, normalization,
    source_checkpoint=None, source_audit=None, smoke=False,
    analysis_mode=None, capability_checkpoint=None,
):
    package = Path(__file__).parent
    root = package.parents[1]
    implementation = {
        str(path.resolve()): file_sha256(path)
        for path in sorted(package.glob("*.py"))
    }
    for relative in (
        "_next/scripts/train_bilinear.py", "_next/scripts/audit.py",
        "_next/scripts/audit_bilinear.py",
        "next", "jobs_next_train.pbs", "jobs_next_audit.pbs",
    ):
        path = (root / relative).resolve()
        if path.is_file():
            implementation[str(path)] = file_sha256(path)
    data = {}
    for pair in pair_names:
        real = []
        available_variables = []
        for path in config.paths.real_field_paths(pair):
            record = {"path": str(path), "available": path.is_file()}
            if path.is_file():
                record["sha256"] = file_sha256(path)
                import xarray as xr
                with xr.open_dataset(path) as dataset:
                    record["variables"] = sorted(
                        name for name in config.panel.real_fields if name in dataset
                    )
                available_variables.append(set(record["variables"]))
            real.append(record)
        data[pair] = {
            "edge": _file_record(config.paths.edge_path(pair)),
            "bilinear_map": _file_record(config.paths.bilinear_map_path(pair)),
            "np2_map": (
                _file_record(
                    Path(config.paths.maps)
                    / f"map_{pair}_{config.benchmarks.np2_suffix}.nc"
                )
                if (
                    Path(config.paths.maps)
                    / f"map_{pair}_{config.benchmarks.np2_suffix}.nc"
                ).is_file()
                else None
            ),
            "real": real,
            "included_real_fields": (
                sorted(set.intersection(*available_variables))
                if len(available_variables) == 2 else []
            ),
        }
    return {
        "format": "remapgnn.bilinear_run_manifest",
        "schema_version": 1,
        "config": config.to_dict(),
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "config_file": _file_record(config.path) if config.path else None,
        "implementation_sha256": implementation,
        "data": data,
        "stage": str(stage),
        "edge_normalization": normalization,
        "source_checkpoint": (
            None if source_checkpoint is None else _file_record(source_checkpoint)
        ),
        "source_audit": (
            None if source_audit is None else _file_record(source_audit)
        ),
        "capability_checkpoint": (
            None if capability_checkpoint is None
            else _file_record(capability_checkpoint)
        ),
        "smoke": bool(smoke),
        "analysis_mode": analysis_mode,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
        },
    }


def verify_run_manifest(manifest, *, allow_implementation_mismatch=False):
    if (
        manifest.get("format") == "remapgnn.bilinear_run_manifest"
        and manifest.get("schema_version") == 1
    ):
        records = []
        for name in (
            "config_file", "source_checkpoint", "source_audit",
            "capability_checkpoint",
        ):
            if manifest.get(name):
                records.append(manifest[name])
        for item in manifest["data"].values():
            records.extend((item["edge"], item["bilinear_map"]))
            if item.get("np2_map"):
                records.append(item["np2_map"])
            records.extend(x for x in item["real"] if x.get("available"))
        for record in records:
            if file_sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"authenticated input changed: {record['path']}")
        mismatches = tuple(
            path for path, expected in manifest["implementation_sha256"].items()
            if file_sha256(path) != expected
        )
        if mismatches and not allow_implementation_mismatch:
            raise ValueError(f"implementation changed: {mismatches[0]}")
        return mismatches
    if manifest.get("format") != "remapgnn.run_manifest" or manifest.get("schema_version") != 1:
        raise ValueError("invalid run manifest")
    records = [manifest["source_checkpoint"], manifest["source_manifest"],
               manifest["fv_checkpoint"]]
    if manifest.get("config_file"):
        records.append(manifest["config_file"])
    for item in manifest["data"].values():
        records.extend((item["edge"], item["map"]))
        records.extend(x for x in item["real"] if x.get("available"))
    for record in records:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"authenticated input changed: {record['path']}")
    mismatches = tuple(
        path for path, expected in manifest["implementation_sha256"].items()
        if file_sha256(path) != expected
    )
    if mismatches and not allow_implementation_mismatch:
        raise ValueError(f"implementation changed: {mismatches[0]}")
    return mismatches


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
