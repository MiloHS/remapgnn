from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import math

from .config import (
    AuditConfig, FeatureConfig, LossConfig, PathsConfig, PhaseConfig,
    SelectionConfig, _GATE_MODES, _PAIR_ROLES, _finite, _strict_dataclass,
)


@dataclass(frozen=True)
class DegreeBandConfig:
    name: str
    degree_min: int
    degree_max: int
    trainable: bool = True

    def __post_init__(self):
        if not self.name or self.degree_min < 1 or self.degree_max < self.degree_min:
            raise ValueError("degree band needs a name and an increasing positive range")

    def contains(self, degree):
        return self.degree_min <= int(degree) <= self.degree_max


@dataclass(frozen=True)
class BilinearStageConfig:
    name: str
    target_band: str
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
        if not self.name or not self.target_band:
            raise ValueError("bilinear stage needs a name and target_band")
        for name in (
            "edge_dim", "hidden", "geometry_hidden", "router_hidden",
            "projection_iterations", "edge_chunk",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{self.name}.{name} must be an integer")
        if min(self.edge_dim, self.hidden, self.geometry_hidden, self.router_hidden) <= 0:
            raise ValueError(f"{self.name}: network dimensions must be positive")
        if self.projection_iterations <= 0 or self.edge_chunk < 0:
            raise ValueError(f"{self.name}: invalid projection/chunk settings")
        for low, high, label in (
            (self.field_gate_low, self.field_gate_high, "field"),
            (self.local_gate_low, self.local_gate_high, "local"),
        ):
            if not 0 <= low < high <= 1:
                raise ValueError(f"{self.name}: invalid {label} router thresholds")
        for mode in (
            self.capability_gate_mode, self.router_gate_mode,
            self.deployment_gate_mode,
        ):
            if mode not in _GATE_MODES:
                raise ValueError(f"{self.name}: unknown gate mode {mode!r}")
        for name in (
            "delta_scale", "reference_floor", "gate_feature_epsilon", "epsilon",
            "projection_row_tolerance", "projection_column_tolerance",
        ):
            _finite(
                f"{self.name}.{name}", getattr(self, name),
                positive=name != "delta_scale", nonnegative=name == "delta_scale",
            )

    @classmethod
    def from_dict(cls, raw):
        return _strict_dataclass(cls, dict(raw), "bilinear stage")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BilinearPanelConfig:
    quadrature_resolution: int = 8
    smoother_neighbors: int = 9
    max_degrees_per_epoch: int = 4
    modes_per_degree: int = 6
    target_mixtures: int = 16
    guard_mixtures: int = 16
    cross_target_mixtures: int = 12
    cross_guard_mixtures: int = 12
    cross_target_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    boundary_guard_width: int = 4
    audit_max_degrees: int = 5
    audit_modes_per_degree: int = 8
    audit_mixture_multiplier: int = 2
    real_fields: tuple[str, ...] = (
        "AnalyticalFun1", "AnalyticalFun2", "TotalPrecipWater",
        "CloudFraction", "Topography",
    )

    def __post_init__(self):
        for name in (
            "quadrature_resolution", "smoother_neighbors", "max_degrees_per_epoch",
            "modes_per_degree", "audit_max_degrees", "audit_modes_per_degree",
            "audit_mixture_multiplier", "boundary_guard_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"panel.{name} must be positive")
        for name in (
            "target_mixtures", "guard_mixtures", "cross_target_mixtures",
            "cross_guard_mixtures",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"panel.{name} must be nonnegative")
        if not self.cross_target_fractions or any(
            not math.isfinite(x) or not 0 < x < 1
            for x in self.cross_target_fractions
        ):
            raise ValueError("cross-target fractions must lie strictly between 0 and 1")
        if len(self.real_fields) != len(set(self.real_fields)):
            raise ValueError("panel.real_fields contains duplicates")


@dataclass(frozen=True)
class BilinearBaselineConfig:
    kind: str = "conservative_esmf_bilinear"
    conservation: str = "global_constant"

    def __post_init__(self):
        if self.kind != "conservative_esmf_bilinear":
            raise ValueError("unsupported bilinear baseline kind")
        if self.conservation != "global_constant":
            raise ValueError("unsupported bilinear conservation method")


@dataclass(frozen=True)
class BenchmarkConfig:
    fv_checkpoint: str = "_next/checkpoints/fv_relax1.pt"
    fv_progressive_checkpoint: str = "_next/checkpoints/progressive.pt"
    np2_suffix: str = "conserve_np2"

    def __post_init__(self):
        if not self.fv_checkpoint or not self.fv_progressive_checkpoint or not self.np2_suffix:
            raise ValueError("benchmark paths and suffix must be nonempty")


@dataclass(frozen=True)
class BilinearSelectionConfig(SelectionConfig):
    capability_safety_tolerance: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        _finite(
            "selection.capability_safety_tolerance",
            self.capability_safety_tolerance,
            nonnegative=True,
        )


@dataclass(frozen=True)
class BilinearExperimentConfig:
    schema_version: int
    run_name: str
    pair_roles: Mapping[str, tuple[str, ...]]
    paths: PathsConfig
    features: FeatureConfig
    baseline: BilinearBaselineConfig
    benchmarks: BenchmarkConfig
    bands: tuple[DegreeBandConfig, ...]
    stages: tuple[BilinearStageConfig, ...]
    panel: BilinearPanelConfig = field(default_factory=BilinearPanelConfig)
    phases: PhaseConfig = field(default_factory=PhaseConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    selection: BilinearSelectionConfig = field(
        default_factory=BilinearSelectionConfig
    )
    audit: AuditConfig = field(default_factory=AuditConfig)
    seed: int = 2407
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)
    path: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.schema_version != 5:
            raise ValueError("bilinear experiment schema must be 5")
        if not self.run_name.strip():
            raise ValueError("run_name must be nonempty")
        band_names = [band.name for band in self.bands]
        if len(band_names) != len(set(band_names)):
            raise ValueError("degree band names must be unique")
        ordered = sorted(self.bands, key=lambda value: value.degree_min)
        if ordered[0].degree_min != 1:
            raise ValueError("degree bands must begin at degree 1")
        for left, right in zip(ordered, ordered[1:]):
            if left.degree_max + 1 != right.degree_min:
                raise ValueError("degree bands must be contiguous and non-overlapping")
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)) or not stage_names:
            raise ValueError("stage names must be nonempty and unique")
        expected = [band.name for band in self.bands if band.trainable]
        if [stage.target_band for stage in self.stages] != expected:
            raise ValueError("stages must follow the ordered trainable bands")
        roles = {name: set(values) for name, values in self.pair_roles.items()}
        unknown = set(roles) - _PAIR_ROLES
        if unknown:
            raise ValueError(f"unknown pair roles: {sorted(unknown)}")
        if not roles.get("train") or not roles.get("selection"):
            raise ValueError("train and selection pairs are required")
        names = list(roles)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = roles[left] & roles[right]
                if overlap:
                    raise ValueError(f"pair-role leakage: {sorted(overlap)}")
        if any(stage.edge_dim != len(self.features.edge) for stage in self.stages):
            raise ValueError("every stage edge_dim must match configured edge features")

    @classmethod
    def from_dict(cls, raw, *, path=None):
        data = dict(raw)
        allowed = {
            "schema_version", "run_name", "seed", "pair_roles", "paths",
            "features", "baseline", "bands", "stages", "panel", "phases",
            "loss", "selection", "audit", "benchmarks",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown bilinear experiment keys: {sorted(unknown)}")
        feature_raw = dict(data["features"])
        if set(feature_raw) != {"edge"}:
            raise ValueError("bilinear features support only edge")
        panel_raw = dict(data.get("panel", {}))
        for name in ("cross_target_fractions", "real_fields"):
            if name in panel_raw:
                panel_raw[name] = tuple(panel_raw[name])
        return cls(
            schema_version=int(data["schema_version"]),
            run_name=str(data["run_name"]),
            seed=int(data.get("seed", 2407)),
            pair_roles={key: tuple(value) for key, value in data["pair_roles"].items()},
            paths=_strict_dataclass(PathsConfig, dict(data["paths"]), "paths"),
            features=FeatureConfig(tuple(feature_raw["edge"])),
            baseline=_strict_dataclass(
                BilinearBaselineConfig, dict(data.get("baseline", {})), "baseline"
            ),
            benchmarks=_strict_dataclass(
                BenchmarkConfig, dict(data.get("benchmarks", {})), "benchmarks"
            ),
            bands=tuple(
                _strict_dataclass(DegreeBandConfig, dict(value), "degree band")
                for value in data["bands"]
            ),
            stages=tuple(BilinearStageConfig.from_dict(value) for value in data["stages"]),
            panel=_strict_dataclass(BilinearPanelConfig, panel_raw, "panel"),
            phases=_strict_dataclass(PhaseConfig, dict(data.get("phases", {})), "phases"),
            loss=_strict_dataclass(LossConfig, dict(data.get("loss", {})), "loss"),
            selection=_strict_dataclass(
                BilinearSelectionConfig,
                dict(data.get("selection", {})), "selection",
            ),
            audit=_strict_dataclass(AuditConfig, dict(data.get("audit", {})), "audit"),
            raw=data,
            path=None if path is None else Path(path),
        )

    def to_dict(self):
        value = asdict(self)
        value.pop("raw", None)
        value.pop("path", None)
        return value

    def pairs(self, *roles):
        return tuple(dict.fromkeys(
            pair for role in roles for pair in self.pair_roles.get(role, ())
        ))

    def stage(self, name):
        return next(stage for stage in self.stages if stage.name == name)

    def band(self, name):
        return next(band for band in self.bands if band.name == name)
