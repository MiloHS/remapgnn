from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .types import FieldBatch

"""Generates individual test/training fields."""


QUADRATIC_TERMS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def longitude_latitude_to_xyz(longitude_degrees, latitude_degrees):
    longitude = np.deg2rad(np.asarray(longitude_degrees, dtype=np.float64))
    latitude = np.deg2rad(np.asarray(latitude_degrees, dtype=np.float64))
    return np.stack(
        (np.cos(latitude) * np.cos(longitude),
         np.cos(latitude) * np.sin(longitude), np.sin(latitude)), axis=-1,
    )


def _normalize_xyz(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1.0e-300)


def _solid_angle(a, b, c):
    triple = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    denominator = (
        1.0 + np.einsum("...i,...i->...", a, b)
        + np.einsum("...i,...i->...", a, c)
        + np.einsum("...i,...i->...", b, c)
    )
    return 2.0 * np.arctan2(triple, denominator)


def _fan_area(vertices, valid_counts):
    count, corners, _ = vertices.shape
    slot = np.arange(corners)[None, :]
    valid = slot < valid_counts[:, None]
    center = _normalize_xyz(
        (vertices * valid[..., None]).sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
    )
    area = np.zeros(count)
    for index in range(corners):
        following = (index + 1) % corners
        endpoint = np.where(
            (following >= valid_counts)[:, None], vertices[:, 0, :], vertices[:, following, :]
        )
        area += np.where(
            index < valid_counts, _solid_angle(center, vertices[:, index, :], endpoint), 0.0
        )
    return area


def load_grid_corners(map_path, side):
    import xarray as xr
    with xr.open_dataset(map_path) as dataset:
        longitude = dataset[f"xv_{side}"].values.astype(np.float64)
        latitude = dataset[f"yv_{side}"].values.astype(np.float64)
        area = dataset[f"area_{side}"].values.astype(np.float64)
        centers = longitude_latitude_to_xyz(
            dataset[f"xc_{side}"].values, dataset[f"yc_{side}"].values
        )
    vertices = longitude_latitude_to_xyz(longitude, latitude)
    is_zero = (np.abs(longitude) <= 1.0e-12) & (np.abs(latitude) <= 1.0e-12)
    trailing = np.cumprod(is_zero[:, ::-1].astype(np.int64), axis=1)
    stripped = np.clip(vertices.shape[1] - trailing.sum(axis=1), 3, vertices.shape[1])
    full = np.full(vertices.shape[0], vertices.shape[1], dtype=np.int64)
    if np.any(stripped < full):
        use_stripped = np.abs(_fan_area(vertices, stripped) - area) <= np.abs(
            _fan_area(vertices, full) - area
        )
        valid_counts = np.where((stripped < full) & use_stripped, stripped, full)
    else:
        valid_counts = full
    return vertices, valid_counts, area, centers


def _reference_triangles(resolution):
    result = []
    for left in range(resolution):
        for right in range(resolution - left):
            triangles = [((left, right), (left + 1, right), (left, right + 1))]
            if right < resolution - left - 1:
                triangles.append(((left + 1, right), (left, right + 1), (left + 1, right + 1)))
            for triangle in triangles:
                result.append(tuple(
                    (a / resolution, b / resolution, 1.0 - a / resolution - b / resolution)
                    for a, b in triangle
                ))
    return result


def build_grid_quadrature(vertices, valid_counts, resolution=8):
    count, corners, _ = vertices.shape
    slots = np.arange(corners)[None, :]
    valid = slots < valid_counts[:, None]
    center = _normalize_xyz(
        (vertices * valid[..., None]).sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
    )
    cells = np.arange(count)
    points, weights, indices = [], [], []
    for corner in range(corners):
        following = (corner + 1) % corners
        endpoint = np.where(
            (following >= valid_counts)[:, None], vertices[:, 0, :], vertices[:, following, :]
        )
        active = corner < valid_counts
        for first, second, third in _reference_triangles(int(resolution)):
            p1 = _normalize_xyz(first[0] * center + first[1] * vertices[:, corner] + first[2] * endpoint)
            p2 = _normalize_xyz(second[0] * center + second[1] * vertices[:, corner] + second[2] * endpoint)
            p3 = _normalize_xyz(third[0] * center + third[1] * vertices[:, corner] + third[2] * endpoint)
            points.append(_normalize_xyz(p1 + p2 + p3))
            weights.append(np.where(active, _solid_angle(p1, p2, p3), 0.0))
            indices.append(cells)
    points, weights, indices = np.concatenate(points), np.concatenate(weights), np.concatenate(indices)
    keep = weights > 1.0e-14
    points, weights, indices = points[keep], weights[keep], indices[keep]
    cell_area = np.bincount(indices, weights=weights, minlength=count)
    return {
        "points": points, "weights": weights, "cell_index": indices, "cell_area": cell_area
    }


def grid_quadrature(map_path, side, resolution=8, expected_centers=None):
    vertices, valid, area, centers = load_grid_corners(map_path, side)
    if expected_centers is not None:
        error = float(np.abs(centers - np.asarray(expected_centers)).max())
        if error > 1.0e-6:
            raise ValueError(f"map/edge cell ordering mismatch: max error {error:.3e}")
    result = build_grid_quadrature(vertices, valid, resolution)
    result.update(area=area, centers=centers)
    return result


def grid_moments(map_path, side, resolution=8, expected_centers=None):
    quadrature = grid_quadrature(map_path, side, resolution, expected_centers)
    coordinate = np.stack([
        cell_average(lambda xyz, d=d: xyz[:, d], quadrature) for d in range(3)
    ], axis=1)
    quadratic = np.stack([
        cell_average(lambda xyz, a=a, b=b: xyz[:, a] * xyz[:, b], quadrature)
        for a, b in QUADRATIC_TERMS
    ], axis=1)
    return {"coordinate": coordinate, "quadratic": quadratic, **quadrature}


def stable_seed(key: str, seed=0) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    return value % (2**31 - 1) + int(seed)


def source_keyed_mode_split(
    source_key: str, degree: int, seed: int, split: str,
    *, train_fraction=0.70, validation_fraction=0.15,
):
    """Deterministic train/validation/audit split keyed only by source mesh."""
    orders = np.arange(-int(degree), int(degree) + 1, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(f"{source_key}:{degree}", seed))
    rng.shuffle(orders)
    count = int(orders.size)
    if count < 3:
        return [int(x) for x in orders] if split == "train" else []
    train_count = min(max(1, int(round(train_fraction * count))), count - 2)
    validation_count = min(
        max(1, int(round(validation_fraction * count))), count - train_count - 1
    )
    partitions = {
        "train": orders[:train_count],
        "validation": orders[train_count:train_count + validation_count],
        "val": orders[train_count:train_count + validation_count],
        "audit": orders[train_count + validation_count:],
        "test": orders[train_count + validation_count:],
    }
    if split not in partitions:
        raise ValueError(f"unknown field split {split!r}")
    return [int(value) for value in partitions[split]]


def real_spherical_harmonic(degree, order, xyz):
    """Stable real spherical harmonic, up to a harmless per-mode constant."""
    xyz = np.asarray(xyz, dtype=np.float64)
    radius = np.sqrt((xyz * xyz).sum(axis=1))
    cosine = np.clip(xyz[:, 2] / np.maximum(radius, 1.0e-30), -1.0, 1.0)
    longitude = np.arctan2(xyz[:, 1], xyz[:, 0])
    absolute_order = abs(int(order))
    sine = np.sqrt(np.clip(1.0 - cosine * cosine, 0.0, None))
    sectoral = np.ones_like(cosine)
    for index in range(1, absolute_order + 1):
        sectoral *= sine * np.sqrt((2.0 * index + 1.0) / (2.0 * index))
    if degree == absolute_order:
        legendre = sectoral
    else:
        next_value = cosine * np.sqrt(2.0 * absolute_order + 3.0) * sectoral
        if degree == absolute_order + 1:
            legendre = next_value
        else:
            previous, current = sectoral, next_value
            for index in range(absolute_order + 2, int(degree) + 1):
                a = np.sqrt(
                    (2.0 * index - 1.0) * (2.0 * index + 1.0)
                    / ((index - absolute_order) * (index + absolute_order))
                )
                b = np.sqrt(
                    (2.0 * index + 1.0) * (index + absolute_order - 1.0)
                    * (index - absolute_order - 1.0)
                    / ((2.0 * index - 3.0) * (index - absolute_order) * (index + absolute_order))
                )
                previous, current = current, cosine * a * current - b * previous
            legendre = current
    if order == 0:
        return legendre
    phase = np.cos(order * longitude) if order > 0 else np.sin(absolute_order * longitude)
    return np.sqrt(2.0) * legendre * phase


def cell_average(function, quadrature):
    values = np.asarray(function(quadrature["points"]), dtype=np.float64)
    numerator = np.bincount(
        quadrature["cell_index"], weights=values * quadrature["weights"],
        minlength=len(quadrature["cell_area"]),
    )
    return numerator / np.maximum(quadrature["cell_area"], 1.0e-300)


def harmonic_batch(
    *, source_key, source_quadrature, target_quadrature, degrees,
    modes_per_degree, split, seed, area_src, role, pair_key=None, sample_seed=None,
):
    source, truth, frequencies, labels, keys = [], [], [], [], []
    effective_k = np.sqrt(len(area_src) / 6.0)
    normalized_area = np.asarray(area_src, dtype=np.float64)
    normalized_area /= np.maximum(normalized_area.sum(), 1.0e-30)
    for degree_index, degree in enumerate(degrees):
        orders = source_keyed_mode_split(source_key, degree, seed, split)
        if sample_seed is not None and len(orders) > 1:
            rng = np.random.default_rng(stable_seed(
                f"{pair_key or source_key}:{degree}:{split}:{degree_index}", sample_seed
            ))
            rng.shuffle(orders)
        for order in orders[:int(modes_per_degree)]:
            function = lambda xyz, d=degree, o=order: real_spherical_harmonic(d, o, xyz)
            source_value = cell_average(function, source_quadrature)
            target_value = cell_average(function, target_quadrature)
            rms = np.sqrt(np.sum(normalized_area * np.square(source_value)))
            if rms <= 1.0e-14:
                raise ValueError(f"zero RMS harmonic Y_{degree}_{order}")
            source.append(torch.tensor(source_value / rms, dtype=torch.float32))
            truth.append(torch.tensor(target_value / rms, dtype=torch.float32))
            frequencies.append(float(degree / effective_k))
            labels.append((int(degree), int(order)))
            keys.append(f"{source_key}:Y:{degree}:{order}")
    return FieldBatch(
        torch.stack(source), torch.stack(truth), torch.tensor(frequencies),
        labels, [role] * len(labels), keys,
    )


def balanced_mixtures(batch: FieldBatch, area_src, count, seed, *, role=None):
    if count <= 0:
        return FieldBatch(
            batch.source[:0], batch.truth[:0], batch.frequency[:0], [], [], []
        )
    rng = np.random.default_rng(seed)
    area = area_src.to(batch.source.dtype)
    area = area / area.sum().clamp_min(1.0e-30)
    levels = sorted(float(value) for value in torch.unique(batch.frequency).tolist())
    base, remainder = divmod(int(count), len(levels))
    allocation = [base] * len(levels)
    for index in range(remainder):
        allocation[(int(seed) + index) % len(levels)] += 1
    source, truth, frequency, labels, keys = [], [], [], [], []
    mixture_index = 0
    for level, level_count in zip(levels, allocation):
        indices = torch.where(torch.isclose(batch.frequency, batch.frequency.new_tensor(level)))[0]
        if indices.numel() < 2:
            continue
        for _ in range(level_count):
            coefficient = torch.tensor(rng.standard_normal(indices.numel()), dtype=batch.source.dtype)
            mixed_source = (coefficient[:, None] * batch.source[indices]).sum(dim=0)
            mixed_truth = (coefficient[:, None] * batch.truth[indices]).sum(dim=0)
            rms = (area * mixed_source.square()).sum().sqrt().clamp_min(1.0e-20)
            source.append(mixed_source / rms)
            truth.append(mixed_truth / rms)
            frequency.append(torch.tensor(level, dtype=batch.frequency.dtype))
            labels.append((-1, -1))
            keys.append(f"mixture:{seed}:{mixture_index}")
            mixture_index += 1
    return FieldBatch(
        torch.stack(source), torch.stack(truth), torch.stack(frequency), labels,
        [role or batch.roles[0]] * len(source), keys,
    )


def analytic_function(name, xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    longitude = np.arctan2(xyz[:, 1], xyz[:, 0])
    latitude = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    if name == "smooth1":
        return 1.0 + 0.25 * xyz[:, 0] - 0.15 * xyz[:, 1] + 0.10 * xyz[:, 2] \
            + 0.20 * np.sin(2.0 * longitude) * np.cos(latitude)
    if name == "smooth2":
        return np.exp(0.5 * xyz[:, 0] - 0.25 * xyz[:, 1]) \
            + 0.10 * np.cos(3.0 * longitude) * np.square(np.cos(latitude))
    raise ValueError(f"unknown analytic field {name!r}")


def analytic_batch(source_quadrature, target_quadrature, area_src):
    area = np.asarray(area_src, dtype=np.float64)
    area /= np.maximum(area.sum(), 1.0e-30)
    source, truth = [], []
    for name in ("smooth1", "smooth2"):
        x = cell_average(lambda xyz, n=name: analytic_function(n, xyz), source_quadrature)
        y = cell_average(lambda xyz, n=name: analytic_function(n, xyz), target_quadrature)
        mean = float(np.sum(area * x))
        x, y = x - mean, y - mean
        rms = float(np.sqrt(np.sum(area * np.square(x))))
        source.append(torch.tensor(x / rms, dtype=torch.float32))
        truth.append(torch.tensor(y / rms, dtype=torch.float32))
    return FieldBatch(
        torch.stack(source), torch.stack(truth), torch.full((2,), float("nan")),
        [(-400, 0), (-400, 1)], ["smooth", "smooth"],
        ["analytic:smooth1", "analytic:smooth2"],
    )


def concatenate_batches(batches):
    batches = [batch for batch in batches if batch.source.shape[0]]
    if not batches:
        raise ValueError("cannot concatenate an empty field panel")
    masks = [x.is_target for x in batches]
    return FieldBatch(
        torch.cat([x.source for x in batches]), torch.cat([x.truth for x in batches]),
        torch.cat([x.frequency for x in batches]),
        [label for x in batches for label in x.labels],
        [role for x in batches for role in x.roles],
        [key for x in batches for key in x.source_keys],
        [family for x in batches for family in (x.families or x.roles)],
        torch.cat(masks),
    )


def load_real_field(path: str | Path, name: str):
    import xarray as xr
    with xr.open_dataset(path) as dataset:
        if name not in dataset:
            raise KeyError(f"{name} not found in {path}")
        return np.asarray(dataset[name].values, dtype=np.float64).reshape(-1)


def real_field_batch(paths, names, n_source, n_target, area_src):
    """Load available paired real fields and normalize like analytic anchors."""
    source_path, target_path = map(Path, paths)
    if not source_path.is_file() or not target_path.is_file():
        return None
    area = np.asarray(area_src, dtype=np.float64)
    area /= np.maximum(area.sum(), 1.0e-30)
    source, truth, labels, keys = [], [], [], []
    for index, name in enumerate(names):
        try:
            x, y = load_real_field(source_path, name), load_real_field(target_path, name)
        except (KeyError, OSError):
            continue
        if x.size != int(n_source) or y.size != int(n_target):
            continue
        mean = float(np.sum(area * x))
        x, y = x - mean, y - mean
        rms = float(np.sqrt(np.sum(area * x * x)))
        if rms <= 1.0e-14:
            continue
        source.append(torch.tensor(x / rms, dtype=torch.float32))
        truth.append(torch.tensor(y / rms, dtype=torch.float32))
        labels.append((-500, index)); keys.append(f"real:{name}")
    if not source:
        return None
    count = len(source)
    return FieldBatch(
        torch.stack(source), torch.stack(truth), torch.full((count,), float("nan")),
        labels, ["safety"] * count, keys, ["real"] * count, torch.zeros(count, dtype=torch.bool),
    )
