from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping
import math

"""Reads json configs into Python settings: pairs, data paths, stage sizes, training
    schedules, and pass thresholds."""

_GATE_MODES = {"forced_open", "forced_closed", "soft", "hard", "straight_through"}
_INITIALIZATION_MODES = {"fresh", "checkpoint"}
_PAIR_ROLES = {"train", "selection", "protected", "external_resolution"}


def _finite(name, value, *, positive=False, nonnegative=False):
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and float(value) <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and float(value) < 0:
        raise ValueError(f"{name} must be nonnegative")


def _strict_dataclass(cls, raw, label):
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class StageConfig:
    name: str
    band_lower: float
    band_upper: float
    edge_dim: int = 8
    hidden: int = 48
    geometry_hidden: int = 32
    router_hidden: int = 32
    delta_scale: float = 0.25
    reference_floor: float = 1.0e-3
    edge_chunk: int = 50000
    projection_iterations: int = 200
    projection_row_tolerance: float = 1.0e-8
    projection_column_tolerance: float = 1.0e-10
    field_gate_low: float = 0.4
    field_gate_high: float = 0.6
    local_gate_low: float = 0.1
    local_gate_high: float = 0.9
    gate_feature_epsilon: float = 1.0e-4
    epsilon: float = 1.0e-8
    capability_gate_mode: str = "forced_open"
    router_gate_mode: str = "straight_through"
    deployment_gate_mode: str = "hard"

    def __post_init__(self):
        if not self.name or self.band_lower >= self.band_upper:
            raise ValueError("stage needs a name and an increasing band")
        for name in ("edge_dim", "hidden", "geometry_hidden", "router_hidden",
                     "projection_iterations", "edge_chunk"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{self.name}.{name} must be an integer")
        if min(self.edge_dim, self.hidden, self.geometry_hidden, self.router_hidden) <= 0:
            raise ValueError(f"{self.name}: network dimensions must be positive")
        if self.projection_iterations <= 0:
            raise ValueError(f"{self.name}: projection_iterations must be positive")
        for low, high, label in ((self.field_gate_low, self.field_gate_high, "field"),
                                 (self.local_gate_low, self.local_gate_high, "local")):
            if not 0.0 <= low < high <= 1.0:
                raise ValueError(f"{self.name}: invalid {label} router thresholds")
        for mode in (self.capability_gate_mode, self.router_gate_mode, self.deployment_gate_mode):
            if mode not in _GATE_MODES:
                raise ValueError(f"{self.name}: unknown gate mode {mode!r}")
        for name in ("band_lower", "band_upper", "delta_scale"):
            _finite(f"{self.name}.{name}", getattr(self, name), nonnegative=True)
        for name in ("reference_floor", "gate_feature_epsilon", "epsilon"):
            _finite(f"{self.name}.{name}", getattr(self, name), positive=True)
        for name in ("projection_row_tolerance", "projection_column_tolerance"):
            _finite(f"{self.name}.{name}", getattr(self, name), positive=True)
        if self.edge_chunk < 0:
            raise ValueError(f"{self.name}.edge_chunk must be nonnegative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageConfig":
        data = dict(raw)
        band = data.pop("band", None)
        if band is not None:
            data.setdefault("band_lower", band["lower"])
            data.setdefault("band_upper", band["upper"])
        aliases = {"geom_hidden": "geometry_hidden", "gate_hidden": "router_hidden",
                   "q_floor": "reference_floor", "gate_feature_eps": "gate_feature_epsilon",
                   "eps": "epsilon", "projection_iterations_train": "projection_iterations"}
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown stage configuration keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PathsConfig:
    analysis: str
    maps: str
    models: str
    reports: str = "_next/reports"
    real_fields: str = "data/MIRA-Datasets"
    graph_suffix: str = "kdist_a3p0_mink8"
    output_checkpoint: str = "progressive_next.pt"
    history: str = "progressive_next_history.csv"

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"paths.{name} must be nonempty")
        if self.output_checkpoint == self.history:
            raise ValueError("checkpoint and history paths must differ")

    def edge_path(self, pair):
        return Path(self.analysis) / f"edge_dataset_{pair}_{self.graph_suffix}.parquet"

    def map_path(self, pair):
        return Path(self.maps) / f"map_{pair}_conserve.nc"

    def real_field_paths(self, pair):
        source, target = pair.split("_to_", 1)
        def one(grid):
            mesh = grid.split("-")[0]
            return Path(self.real_fields) / "Meshes" / "UniformlyRefined" / mesh / \
                f"sample_NM16_O10_{grid}_TPW_CFR_TPO_A1_A2.nc"
        return one(source), one(target)

    @property
    def checkpoint_path(self):
        return Path(self.models) / self.output_checkpoint

    @property
    def history_path(self):
        return Path(self.models) / self.history


@dataclass(frozen=True)
class FeatureConfig:
    edge: tuple[str, ...]

    def __post_init__(self):
        if not self.edge or any(not value for value in self.edge):
            raise ValueError("features.edge must be nonempty")
        if len(self.edge) != len(set(self.edge)):
            raise ValueError("features.edge contains duplicates")


@dataclass(frozen=True)
class PanelConfig:
    quadrature_resolution: int = 8
    smoother_neighbors: int = 9
    frequency_cells_per_k_squared: float = 6.0
    max_degrees_per_epoch: int = 4
    modes_per_degree: int = 6
    target_mixtures: int = 16
    safety_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.125, 1.25, 1.75)
    safety_modes_per_level: int = 3
    safety_mixtures: int = 16
    audit_max_degrees: int = 5
    audit_modes_per_degree: int = 8
    audit_target_mixtures: int = 24
    audit_safety_modes_per_level: int = 6
    audit_safety_mixtures: int = 24
    real_fields: tuple[str, ...] = (
        "AnalyticalFun1", "AnalyticalFun2", "TotalPrecipWater", "CloudFraction", "Topography"
    )

    def __post_init__(self):
        for name in ("quadrature_resolution", "smoother_neighbors", "max_degrees_per_epoch",
                     "modes_per_degree", "safety_modes_per_level", "audit_max_degrees",
                     "audit_modes_per_degree", "audit_safety_modes_per_level"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"panel.{name} must be positive")
        for name in ("target_mixtures", "safety_mixtures", "audit_target_mixtures",
                     "audit_safety_mixtures"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"panel.{name} must be nonnegative")
        _finite("panel.frequency_cells_per_k_squared", self.frequency_cells_per_k_squared, positive=True)
        if not self.safety_levels or any(not math.isfinite(x) or x < 0 for x in self.safety_levels):
            raise ValueError("panel.safety_levels must be finite and nonnegative")
        if len(self.real_fields) != len(set(self.real_fields)):
            raise ValueError("panel.real_fields contains duplicates")


@dataclass(frozen=True)
class PhaseConfig:
    capability_epochs: int = 60
    capability_learning_rate: float = 2.0e-4
    router_epochs: int = 24
    router_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip: float = 1.0
    target_batch: int = 2
    safety_batch: int = 4
    evaluation_interval: int = 4

    def __post_init__(self):
        for name in ("capability_epochs", "router_epochs", "target_batch", "safety_batch",
                     "evaluation_interval"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"phases.{name} must be positive")
        for name in ("capability_learning_rate", "router_learning_rate", "gradient_clip"):
            _finite(f"phases.{name}", getattr(self, name), positive=True)
        _finite("phases.weight_decay", self.weight_decay, nonnegative=True)


@dataclass(frozen=True)
class LossConfig:
    guard_tolerance: float = 0.005
    fv_guard_tolerance: float = 0.02
    cvar_fraction: float = 0.25
    guard_weight: float = 6.0
    local_weight: float = 0.5
    gate_teacher_weight: float = 0.1
    safety_gate_weight: float = 0.05
    correction_weight: float = 1.0e-5
    router_teacher_temperature: float = 0.1

    def __post_init__(self):
        for name in ("guard_tolerance", "fv_guard_tolerance", "guard_weight", "local_weight",
                     "gate_teacher_weight", "safety_gate_weight", "correction_weight"):
            _finite(f"loss.{name}", getattr(self, name), nonnegative=True)
        _finite("loss.cvar_fraction", self.cvar_fraction, positive=True)
        if self.cvar_fraction > 1:
            raise ValueError("loss.cvar_fraction must be at most 1")
        _finite("loss.router_teacher_temperature", self.router_teacher_temperature, positive=True)


@dataclass(frozen=True)
class SelectionConfig:
    capability_minimum_improvement: float = 0.001
    final_minimum_gain: float = 0.02
    safety_tolerance: float = 0.02
    prior_band_tolerance: float = 0.01

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _finite(f"selection.{name}", getattr(self, name), nonnegative=True)


@dataclass(frozen=True)
class AuditConfig:
    row_tolerance: float = 1.0e-8
    column_tolerance: float = 1.0e-10
    minimum_target_gain: float = 0.03
    maximum_safety_regression: float = 0.02
    maximum_prior_band_regression: float = 0.01
    maximum_fv_regression: float = 0.02
    field_batch: int = 2
    timing_repeats: int = 5
    zero_error_tolerance: float = 1.0e-14

    def __post_init__(self):
        for name in ("row_tolerance", "column_tolerance", "zero_error_tolerance"):
            _finite(f"audit.{name}", getattr(self, name), positive=True)
        for name in ("minimum_target_gain", "maximum_safety_regression",
                     "maximum_prior_band_regression", "maximum_fv_regression"):
            _finite(f"audit.{name}", getattr(self, name), nonnegative=True)
        if self.field_batch <= 0 or self.timing_repeats <= 0:
            raise ValueError("audit batch and timing counts must be positive")


@dataclass(frozen=True)
class ModelConfig:
    """How a trainable model is assembled from an approved clean checkpoint."""

    source_checkpoint: str
    source_manifest: str
    prefix_through: str | None
    train_stage: str
    initialization: str = "fresh"
    checkpoint_stage: str | None = None

    def __post_init__(self):
        if not self.source_checkpoint:
            raise ValueError("model.source_checkpoint is required")
        if not self.source_manifest:
            raise ValueError("model.source_manifest is required")
        if not self.train_stage:
            raise ValueError("model.train_stage is required")
        if self.initialization not in _INITIALIZATION_MODES:
            raise ValueError(
                f"model.initialization must be one of {sorted(_INITIALIZATION_MODES)}"
            )
        if self.initialization == "fresh" and self.checkpoint_stage is not None:
            raise ValueError("model.checkpoint_stage is only valid for checkpoint initialization")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    run_name: str
    pair_roles: Mapping[str, tuple[str, ...]]
    paths: PathsConfig
    features: FeatureConfig
    fv_checkpoint: str
    model: ModelConfig
    stages: tuple[StageConfig, ...]
    panel: PanelConfig = field(default_factory=PanelConfig)
    phases: PhaseConfig = field(default_factory=PhaseConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    seed: int = 2407
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)
    path: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.schema_version != 4:
            raise ValueError(f"unsupported experiment schema {self.schema_version}; expected 4")
        if not self.fv_checkpoint:
            raise ValueError("FV checkpoint path is required")
        if not self.run_name.strip():
            raise ValueError("run_name must be nonempty")
        names = [stage.name for stage in self.stages]
        if not names or len(names) != len(set(names)):
            raise ValueError("at least one uniquely named stage is required")
        if self.model.train_stage not in names:
            raise ValueError(f"train stage {self.model.train_stage!r} is not configured")
        train_index = names.index(self.model.train_stage)
        if train_index != len(names) - 1:
            raise ValueError("the train stage must be the final configured stage")
        if self.model.prefix_through is None:
            if train_index != 0:
                raise ValueError("prefix_through is required when configured prefix stages exist")
        else:
            if self.model.prefix_through not in names:
                raise ValueError(f"prefix stage {self.model.prefix_through!r} is not configured")
            if names.index(self.model.prefix_through) != train_index - 1:
                raise ValueError("prefix_through must immediately precede the train stage")
        roles = {name: set(values) for name, values in self.pair_roles.items()}
        unknown_roles = set(roles) - _PAIR_ROLES
        if unknown_roles:
            raise ValueError(f"unknown pair roles: {sorted(unknown_roles)}")
        if not roles.get("train") or not roles.get("selection"):
            raise ValueError("train and selection pair roles must be nonempty")
        if any(not isinstance(pair, str) or not pair for values in self.pair_roles.values() for pair in values):
            raise ValueError("pair roles must contain nonempty strings")
        ordered = ("train", "selection", "protected", "external_resolution")
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                overlap = roles.get(left, set()) & roles.get(right, set())
                if overlap:
                    raise ValueError(f"pair-role leakage between {left} and {right}: {sorted(overlap)}")
        selected = self.stages[train_index]
        if selected.capability_gate_mode != "forced_open":
            raise ValueError("capability training must force the selected stage open")
        if selected.router_gate_mode != "straight_through":
            raise ValueError("router training must use straight-through routing")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path=None):
        data = dict(raw)
        allowed = {"schema_version", "run_name", "seed", "pair_roles", "paths", "features",
                   "fv_checkpoint", "model", "stages", "panel", "phases", "loss",
                   "selection", "audit"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
        paths = _strict_dataclass(PathsConfig, dict(data["paths"]), "paths")
        feature_raw = dict(data["features"])
        if set(feature_raw) != {"edge"}:
            raise ValueError(f"features supports only 'edge'; got {sorted(feature_raw)}")
        features = FeatureConfig(edge=tuple(feature_raw["edge"]))
        panel_raw = dict(data.get("panel", {}))
        if "safety_levels" in panel_raw:
            panel_raw["safety_levels"] = tuple(panel_raw["safety_levels"])
        if "real_fields" in panel_raw:
            panel_raw["real_fields"] = tuple(panel_raw["real_fields"])
        return cls(
            schema_version=int(data["schema_version"]), run_name=str(data.get("run_name", "progressive_next")),
            pair_roles={key: tuple(value) for key, value in data["pair_roles"].items()},
            paths=paths, features=features, fv_checkpoint=str(data["fv_checkpoint"]),
            model=_strict_dataclass(ModelConfig, dict(data["model"]), "model"),
            stages=tuple(StageConfig.from_dict(value) for value in data["stages"]),
            panel=_strict_dataclass(PanelConfig, panel_raw, "panel"),
            phases=_strict_dataclass(PhaseConfig, dict(data.get("phases", {})), "phases"),
            loss=_strict_dataclass(LossConfig, dict(data.get("loss", {})), "loss"),
            selection=_strict_dataclass(SelectionConfig, dict(data.get("selection", {})), "selection"),
            audit=_strict_dataclass(AuditConfig, dict(data.get("audit", {})), "audit"),
            seed=int(data.get("seed", 2407)), raw=data, path=None if path is None else Path(path),
        )

    def to_dict(self):
        value = asdict(self)
        value.pop("raw", None); value.pop("path", None)
        return value

    def pairs(self, *roles):
        return tuple(dict.fromkeys(pair for role in roles for pair in self.pair_roles.get(role, ())))

    @property
    def progressive_checkpoint(self):
        """Compatibility name for the approved clean source checkpoint."""
        return self.model.source_checkpoint

    def stage(self, name: str) -> StageConfig:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"unknown configured stage {name!r}")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return ExperimentConfig.from_dict(json.load(stream), path=path)
