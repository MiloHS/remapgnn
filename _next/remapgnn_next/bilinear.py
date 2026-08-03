from __future__ import annotations

"""Fixed ESMF bilinear baseline plus an exact global conservation adjustment."""

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .fields import grid_quadrature
from .geometry import (
    build_smoother, enhanced_geometry_features, materialize_geometry_columns,
)
from .sparse import apply_operator, index_sum
from .types import PairData, SparseOperator


BASELINE_KIND = "conservative_esmf_bilinear"


def pair_feature_options(stages):
    """Return the static pair features required by an ordered stage list."""
    configs = [getattr(stage, "config", stage) for stage in stages]
    return {
        "geometry_layout": (
            "bilinear_v2" if any(
                getattr(value, "geometry_layout", "intrinsic_v1")
                == "bilinear_v2"
                for value in configs
            ) else "intrinsic_v1"
        ),
        "gradient_layout": (
            "covariance_v2" if any(
                getattr(value, "gradient_layout", "scalar_v1")
                == "covariance_v2"
                for value in configs
            ) else "scalar_v1"
        ),
    }


def compute_geometry_normalization(
    pairs: Iterable[PairData], feature_indices=None,
):
    """Population statistics for enhanced geometry over training graphs only."""
    count = 0
    total = None
    square = None
    for pair in pairs:
        values = pair.correction_geometry
        if values is None:
            raise ValueError(f"{pair.pair}: enhanced correction geometry is absent")
        work = values.to(torch.float64)
        if feature_indices is not None:
            work = work[:, tuple(feature_indices)]
        if not bool(torch.isfinite(work).all()):
            raise ValueError(f"{pair.pair}: correction geometry is non-finite")
        if total is None:
            total = work.sum(0)
            square = work.square().sum(0)
        else:
            total += work.sum(0)
            square += work.square().sum(0)
        count += work.shape[0]
    if count <= 0:
        raise ValueError("cannot normalize an empty collection of pair geometries")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    std = torch.where(std > 1.0e-12, std, torch.ones_like(std))
    return {"mean": mean.tolist(), "std": std.tolist(), "count": int(count)}


def bilinear_map_path(maps, pair):
    return Path(maps) / f"map_{pair}_esmf_bilinear.nc"


def compute_edge_normalization(
    edge_paths: Iterable[str | Path], feature_names: Sequence[str]
) -> dict[str, list[float]]:
    """Compute deterministic float64 population statistics over training graphs."""
    import pandas as pd

    names = tuple(feature_names)
    count = 0
    total = np.zeros(len(names), dtype=np.float64)
    square = np.zeros(len(names), dtype=np.float64)
    for path in edge_paths:
        frame = materialize_geometry_columns(pd.read_parquet(path))
        values = frame[list(names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite edge feature in {path}")
        count += values.shape[0]
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
    if count <= 0:
        raise ValueError("cannot normalize an empty collection of edge graphs")
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std > 1.0e-12, std, 1.0)
    return {"edge_mean": mean.tolist(), "edge_std": std.tolist()}


def _edge_geometry(edge_path, feature_names, normalization):
    import pandas as pd

    frame = materialize_geometry_columns(pd.read_parquet(edge_path))
    source = frame["source_index"].to_numpy(dtype=np.int64)
    target = frame["target_index"].to_numpy(dtype=np.int64)
    n_src, n_tgt = int(source.max()) + 1, int(target.max()) + 1
    if set(np.unique(source)) != set(range(n_src)):
        raise ValueError(f"{edge_path}: correction graph does not cover every source")
    if set(np.unique(target)) != set(range(n_tgt)):
        raise ValueError(f"{edge_path}: correction graph does not cover every target")

    def unique(index, columns, size):
        result = np.zeros((size, len(columns)), dtype=np.float64)
        result[index] = frame[list(columns)].to_numpy(dtype=np.float64)
        return result

    xyz_src = unique(source, ("src_x", "src_y", "src_z"), n_src)
    xyz_tgt = unique(target, ("tgt_x", "tgt_y", "tgt_z"), n_tgt)
    area_src = unique(source, ("src_area",), n_src).reshape(-1)
    area_tgt = unique(target, ("tgt_area",), n_tgt).reshape(-1)
    if np.any(area_src <= 0) or np.any(area_tgt <= 0):
        raise ValueError(f"{edge_path}: cell areas must be positive")
    if abs(area_src.sum() - area_tgt.sum()) > 1.0e-10:
        raise ValueError(f"{edge_path}: source and target total areas differ")
    values = frame[list(feature_names)].to_numpy(dtype=np.float64)
    mean = np.asarray(normalization["edge_mean"], dtype=np.float64)
    std = np.asarray(normalization["edge_std"], dtype=np.float64)
    if mean.shape != (len(feature_names),) or std.shape != mean.shape:
        raise ValueError("edge normalization does not match configured features")
    values = ((values - mean) / std).astype(np.float32)
    return source, target, values, xyz_src, xyz_tgt, area_src, area_tgt


def load_bilinear_operator(path, area_src, area_tgt, expected_src_xyz, expected_tgt_xyz):
    import xarray as xr

    with xr.open_dataset(path) as data:
        weight = np.asarray(data["S"].values).reshape(-1).astype(np.float64)
        target = np.asarray(data["row"].values).reshape(-1).astype(np.int64) - 1
        source = np.asarray(data["col"].values).reshape(-1).astype(np.int64) - 1
        lon_src = np.deg2rad(np.asarray(data["xc_a"].values, dtype=np.float64))
        lat_src = np.deg2rad(np.asarray(data["yc_a"].values, dtype=np.float64))
        lon_tgt = np.deg2rad(np.asarray(data["xc_b"].values, dtype=np.float64))
        lat_tgt = np.deg2rad(np.asarray(data["yc_b"].values, dtype=np.float64))

    def xyz(lon, lat):
        return np.stack(
            (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
            axis=1,
        )

    if xyz(lon_src, lat_src).shape != expected_src_xyz.shape or \
            xyz(lon_tgt, lat_tgt).shape != expected_tgt_xyz.shape:
        raise ValueError(f"{path}: bilinear map dimensions differ from edge geometry")
    center_error = max(
        float(np.abs(xyz(lon_src, lat_src) - expected_src_xyz).max()),
        float(np.abs(xyz(lon_tgt, lat_tgt) - expected_tgt_xyz).max()),
    )
    if center_error > 1.0e-6:
        raise ValueError(f"{path}: bilinear/edge ordering mismatch ({center_error:.3e})")
    if not np.isfinite(weight).all():
        raise ValueError(f"{path}: bilinear weights are non-finite")
    src = torch.tensor(source, dtype=torch.long)
    tgt = torch.tensor(target, dtype=torch.long)
    values = torch.tensor(weight, dtype=torch.float64)
    rows = index_sum(values, tgt, len(area_tgt))
    if float((rows - 1.0).abs().max()) > 1.0e-12:
        raise ValueError(f"{path}: bilinear target rows do not sum to one")
    return SparseOperator.from_weight(
        src, tgt, values,
        torch.tensor(area_src, dtype=torch.float64),
        torch.tensor(area_tgt, dtype=torch.float64),
        provenance={"kind": "esmf_bilinear", "path": str(path)},
    )


def apply_conservative_bilinear(pair: PairData, source: torch.Tensor):
    """Apply bilinear interpolation, then its rank-one global mass correction."""
    squeeze = source.ndim == 1
    values = source.unsqueeze(0) if squeeze else source
    operator = pair.baseline_operator
    if operator.weight.device != values.device:
        operator = operator.to(values.device)
    raw = apply_operator(operator, values)
    work_source = values.to(torch.float64)
    work_raw = raw.to(torch.float64)
    source_integral = (
        work_source * pair.area_src.to(values.device, torch.float64).view(1, -1)
    ).sum(dim=1)
    target_area = pair.area_tgt.to(values.device, torch.float64)
    target_integral = (work_raw * target_area.view(1, -1)).sum(dim=1)
    shift = (source_integral - target_integral) / target_area.sum()
    adjusted = (work_raw + shift.view(-1, 1)).to(values.dtype)
    # Casting the adjusted field back to the network dtype introduces one
    # additional rounding residual.  Two deterministic uniform refinements
    # keep the runtime field conservative without changing the rank-one map.
    for _ in range(2):
        rounded_integral = (
            adjusted.to(torch.float64) * target_area.view(1, -1)
        ).sum(dim=1)
        refinement = (source_integral - rounded_integral) / target_area.sum()
        adjusted = (
            adjusted.to(torch.float64) + refinement.view(-1, 1)
        ).to(values.dtype)
        shift = shift + refinement
    return (
        adjusted.squeeze(0) if squeeze else adjusted,
        raw.squeeze(0) if squeeze else raw,
        shift.squeeze(0) if squeeze else shift,
    )


def build_bilinear_pair(
    pair_name,
    edge_path,
    map_path,
    *,
    feature_names,
    normalization: Mapping,
    correction_reference_kind="uniform",
    bilinear_reference_fraction=0.0,
    quadrature_resolution=8,
    smoother_neighbors=9,
    geometry_layout="intrinsic_v1",
    gradient_layout="scalar_v1",
    device="cpu",
):
    source, target, edge, xyz_src, xyz_tgt, area_src, area_tgt = _edge_geometry(
        edge_path, feature_names, normalization
    )
    baseline = load_bilinear_operator(
        map_path, area_src, area_tgt, xyz_src, xyz_tgt
    )
    source_index = torch.tensor(source, dtype=torch.long)
    target_index = torch.tensor(target, dtype=torch.long)
    degree = index_sum(
        torch.ones(source_index.numel(), dtype=torch.float64),
        target_index, len(area_tgt),
    ).clamp_min(1.0)
    uniform_reference = 1.0 / degree[target_index]
    if geometry_layout not in {"intrinsic_v1", "bilinear_v2"}:
        raise ValueError(f"unknown geometry layout {geometry_layout!r}")
    if gradient_layout not in {"scalar_v1", "covariance_v2"}:
        raise ValueError(f"unknown gradient layout {gradient_layout!r}")
    needs_bilinear_edges = (
        correction_reference_kind == "blended_bilinear"
        or geometry_layout == "bilinear_v2"
    )
    bilinear_reference = torch.zeros_like(uniform_reference)
    bilinear_membership = torch.zeros_like(uniform_reference, dtype=torch.bool)
    if needs_bilinear_edges:
        if bool((baseline.weight < 0).any()):
            raise ValueError("bilinear-aware features require nonnegative weights")
        correction_key = target_index * baseline.n_src + source_index
        baseline_key = baseline.tgt_index * baseline.n_src + baseline.src_index
        order = torch.argsort(correction_key)
        locations = torch.searchsorted(correction_key[order], baseline_key)
        if bool((locations >= correction_key.numel()).any()) or not torch.equal(
            correction_key[order][locations], baseline_key
        ):
            raise ValueError(
                f"{map_path}: bilinear stencil is not contained in correction graph"
            )
        matched = order[locations]
        bilinear_reference[matched] = baseline.weight
        bilinear_membership[matched] = True
    if correction_reference_kind == "uniform":
        reference = uniform_reference
    elif correction_reference_kind == "blended_bilinear":
        fraction = float(bilinear_reference_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("bilinear reference fraction must be in [0,1]")
        reference = (
            (1.0 - fraction) * uniform_reference
            + fraction * bilinear_reference
        )
    else:
        raise ValueError(
            f"unknown correction reference {correction_reference_kind!r}"
        )
    reference_rows = index_sum(reference, target_index, len(area_tgt))
    if float((reference_rows - 1.0).abs().max()) > 1.0e-12:
        raise ValueError("correction reference rows do not sum to one")
    correction_graph = SparseOperator.from_weight(
        source_index, target_index, reference,
        torch.tensor(area_src, dtype=torch.float64),
        torch.tensor(area_tgt, dtype=torch.float64),
        provenance={
            "kind": "kdist_correction_graph",
            "path": str(edge_path),
            "reference": correction_reference_kind,
            "bilinear_reference_fraction": float(
                bilinear_reference_fraction
            ),
        },
    )
    source_quadrature = grid_quadrature(
        map_path, "a", quadrature_resolution, xyz_src, area_src
    )
    target_quadrature = grid_quadrature(
        map_path, "b", quadrature_resolution, xyz_tgt, area_tgt
    )
    src_neighbors, src_weights = build_smoother(xyz_src, smoother_neighbors)
    tgt_neighbors, tgt_weights = build_smoother(xyz_tgt, smoother_neighbors)
    pair = PairData(
        pair=str(pair_name),
        edge_features=torch.tensor(edge, dtype=torch.float32, device=device),
        src_xyz=torch.tensor(xyz_src, dtype=torch.float32, device=device),
        tgt_xyz=torch.tensor(xyz_tgt, dtype=torch.float32, device=device),
        src_neighbor_index=src_neighbors.to(device),
        src_neighbor_weight=src_weights.to(device),
        tgt_neighbor_index=tgt_neighbors.to(device),
        tgt_neighbor_weight=tgt_weights.to(device),
        fv_operator=correction_graph.to(device),
        base_operator=baseline.to(device),
        correction_reference=reference.to(device),
        metadata={
            "baseline_kind": BASELINE_KIND,
            "edge_path": str(edge_path),
            "map_path": str(map_path),
            "source_key": str(pair_name).split("_to_", 1)[0],
            "source_quadrature": source_quadrature,
            "target_quadrature": target_quadrature,
            "panel_quadrature_resolution": int(quadrature_resolution),
        },
    )
    if geometry_layout == "bilinear_v2" or gradient_layout == "covariance_v2":
        enhanced, gradient = enhanced_geometry_features(
            pair, bilinear_reference, bilinear_membership, reference
        )
        pair = replace(
            pair,
            correction_geometry=(
                enhanced if geometry_layout == "bilinear_v2" else None
            ),
            gradient_coefficient=(
                gradient if gradient_layout == "covariance_v2" else None
            ),
        )
    return pair
