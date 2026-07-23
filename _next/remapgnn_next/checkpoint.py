from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .config import ExperimentConfig, StageConfig
from .progressive import ConservativeCorrectionStage, ProgressiveRemapper
from .provenance import file_sha256, tensor_state_sha256


CLEAN_FV_FORMAT = "remapgnn.clean_fv"
CLEAN_PROGRESSIVE_FORMAT = "remapgnn.clean_progressive"
CLEAN_TRAINING_FORMAT = "remapgnn.clean_training"
PROGRESSIVE_SCHEMA_VERSION = 1
TRAINING_SCHEMA_VERSION = 3

STRUCTURAL_STAGE_FIELDS = (
    "edge_dim",
    "hidden",
    "geometry_hidden",
    "router_hidden",
)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validated_progressive_pack(path, *, require_production=False):
    pack = torch_load(path)
    if (
        pack.get("format") != CLEAN_PROGRESSIVE_FORMAT
        or pack.get("schema_version") != PROGRESSIVE_SCHEMA_VERSION
    ):
        raise ValueError("not a supported clean progressive checkpoint")
    if require_production:
        if not pack.get("production", False):
            raise ValueError("checkpoint is not marked as a production conversion")
        equivalence = pack.get("conversion_checks", {}).get("equivalence", {})
        if not equivalence.get("passed", False):
            raise ValueError("production checkpoint has no passing equivalence record")
    for item in pack.get("stages", ()):
        expected = item.get("state_sha256")
        if expected and tensor_state_sha256(item["state"]) != expected:
            raise ValueError(f"stage state hash mismatch for {item.get('config', {}).get('name')!r}")
    return pack


def _stage_from_item(item, *, config=None):
    stage_config = StageConfig.from_dict(item["config"]) if config is None else config
    stage = ConservativeCorrectionStage(stage_config)
    stage.load_state_dict(item["state"], strict=True)
    stage.set_training_phase("frozen")
    return stage


def load_progressive_checkpoint(path, base_operator=None, *, require_production=False):
    pack = _validated_progressive_pack(path, require_production=require_production)
    stages = [_stage_from_item(item) for item in pack["stages"]]
    return ProgressiveRemapper(base_operator, stages), pack


def _exact_prefix_match(expected: StageConfig, actual: StageConfig):
    if expected.to_dict() != actual.to_dict():
        raise ValueError(
            f"configured frozen prefix {expected.name!r} differs from its checkpoint definition"
        )


def _structural_match(expected: StageConfig, actual: StageConfig):
    mismatches = [
        name
        for name in STRUCTURAL_STAGE_FIELDS
        if getattr(expected, name) != getattr(actual, name)
    ]
    if mismatches:
        raise ValueError(
            f"checkpoint initialization for {expected.name!r} has incompatible "
            f"structural fields: {mismatches}"
        )


def build_training_model(config: ExperimentConfig, source=None):
    """Build the configured prefix plus exactly one selected trainable stage."""
    source_path = Path(source or config.model.source_checkpoint)
    pack = _validated_progressive_pack(source_path, require_production=True)
    source_items = list(pack["stages"])
    source_configs = [StageConfig.from_dict(item["config"]) for item in source_items]
    source_by_name = {value.name: (value, item) for value, item in zip(source_configs, source_items)}
    runtime_edge_features = tuple(pack.get("runtime_data", {}).get("edge_features", ()))
    if tuple(config.features.edge) != runtime_edge_features:
        raise ValueError(
            "configured edge features differ from the authenticated source checkpoint; "
            "a new feature layout requires a new runtime-data checkpoint"
        )

    configured = list(config.stages)
    train_index = next(
        index for index, stage in enumerate(configured)
        if stage.name == config.model.train_stage
    )
    prefix_configs = configured[:train_index]
    if config.model.prefix_through is None:
        if prefix_configs:
            raise ValueError("configured prefix exists but model.prefix_through is null")
    else:
        if not prefix_configs or prefix_configs[-1].name != config.model.prefix_through:
            raise ValueError("configured prefix does not end at model.prefix_through")

    stages = []
    for position, expected in enumerate(prefix_configs):
        if position >= len(source_items):
            raise ValueError(f"source checkpoint is missing prefix stage {expected.name!r}")
        actual = source_configs[position]
        if actual.name != expected.name:
            raise ValueError(
                f"source prefix order mismatch: expected {expected.name!r}, got {actual.name!r}"
            )
        _exact_prefix_match(expected, actual)
        stages.append(_stage_from_item(source_items[position]))

    train_config = configured[train_index]
    if train_config.edge_dim != len(runtime_edge_features):
        raise ValueError(
            f"{train_config.name!r} expects {train_config.edge_dim} edge features, "
            f"but the source checkpoint supplies {len(runtime_edge_features)}"
        )
    if config.model.initialization == "fresh":
        train_stage = ConservativeCorrectionStage(train_config)
        initialization_source = None
    else:
        source_name = config.model.checkpoint_stage or train_config.name
        if source_name not in source_by_name:
            raise ValueError(f"source checkpoint has no stage {source_name!r}")
        source_config, source_item = source_by_name[source_name]
        _structural_match(train_config, source_config)
        train_stage = _stage_from_item(source_item, config=train_config)
        initialization_source = source_name
    train_stage.set_training_phase("frozen")
    stages.append(train_stage)
    model = ProgressiveRemapper(None, stages)
    metadata = {
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "prefix_stages": [stage.name for stage in prefix_configs],
        "train_stage": train_config.name,
        "initialization": config.model.initialization,
        "initialization_source": initialization_source,
        "stage_configs": [stage.config.to_dict() for stage in stages],
    }
    return model, pack, metadata


def load_training_checkpoint(path_or_pack, *, require_completed=True):
    pack = (
        torch_load(path_or_pack)
        if isinstance(path_or_pack, (str, Path))
        else path_or_pack
    )
    if (
        pack.get("format") != CLEAN_TRAINING_FORMAT
        or pack.get("schema_version") != TRAINING_SCHEMA_VERSION
    ):
        raise ValueError("not a supported clean training checkpoint")
    if require_completed and not pack.get("completed", False):
        raise ValueError("training checkpoint is incomplete")
    configs = [StageConfig.from_dict(value) for value in pack["model_stage_configs"]]
    model = ProgressiveRemapper(
        None, [ConservativeCorrectionStage(value) for value in configs]
    )
    state = (
        pack["identity_model_state"]
        if pack.get("selected_identity", False)
        else pack["best_model_state"]
    )
    model.load_state_dict(state, strict=True)
    for stage in model.stages:
        stage.set_training_phase("frozen")
    source = pack.get("provenance", {}).get("source_checkpoint") or {}
    source_path = Path(source.get("path", ""))
    if not source_path.is_file():
        raise FileNotFoundError(f"training source checkpoint is unavailable: {source_path}")
    actual = file_sha256(source_path)
    if actual != source.get("sha256"):
        raise ValueError("training source checkpoint hash mismatch")
    progressive_pack = _validated_progressive_pack(source_path, require_production=True)
    return model, progressive_pack, source_path


def validate_fv_reference(progressive_pack: Mapping, fv_path, actual_sha256):
    recorded = progressive_pack.get("fv_checkpoint", {})
    expected = recorded.get("sha256")
    if expected and expected != actual_sha256:
        raise ValueError(
            f"FV checkpoint does not match progressive source: {fv_path}"
        )
