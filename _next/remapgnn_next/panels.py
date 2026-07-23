from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .fields import (
    analytic_batch, balanced_mixtures, concatenate_batches, harmonic_batch,
    real_field_batch, stable_seed,
)
from .types import FieldBatch, PairData

"""Groups those fields into training, validation, and audit panels."""

@dataclass(frozen=True)
class PanelSpec:
    name: str
    roles: tuple[str, ...]


def make_panel(parts: Iterable[FieldBatch]) -> FieldBatch:
    panel = concatenate_batches(parts)
    if panel.source_keys and len(set(panel.source_keys)) != len(panel.source_keys):
        seen, duplicates = set(), []
        for key in panel.source_keys:
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)
        raise ValueError(
            f"field panel contains duplicate source keys: {duplicates[:3]}"
        )
    return panel


def assert_split_disjoint(*panels: FieldBatch):
    sets = [set(panel.source_keys) for panel in panels]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"source-key leakage between panels: {sorted(overlap)[:3]}")


def band_degrees(source_k, lower, upper, *, maximum=0, offset=0):
    first = max(1, int(np.floor(float(lower) * source_k)) + 1)
    last = max(first, int(round(float(upper) * source_k)))
    degrees = list(range(first, last + 1))
    if maximum and len(degrees) > int(maximum):
        start = int(offset) % len(degrees)
        indices = (start + np.floor(np.arange(int(maximum)) * len(degrees) / int(maximum)).astype(int)) % len(degrees)
        degrees = sorted({degrees[int(index)] for index in indices})
    return degrees


def safety_degree(source_k, level, band_lower, band_upper):
    """Choose a safety degree that cannot round into the open target band.

    Target degrees use ``band_lower < degree / K``.  On meshes with a
    non-integer effective K, rounding a safety level at ``band_lower`` can
    therefore place it just inside the target band.  Clamp lower-side guards
    below the first target degree and upper-side guards above the last one.
    """
    source_k = float(source_k)
    level = float(level)
    lower = float(band_lower)
    upper = float(band_upper)
    if lower < level <= upper:
        raise ValueError(
            f"safety level {level} lies inside target band ({lower}, {upper}]"
        )
    first_target = max(1, int(np.floor(lower * source_k)) + 1)
    last_target = max(first_target, int(round(upper * source_k)))
    nearest = max(1, int(round(level * source_k)))
    if level <= lower:
        degree = min(nearest, first_target - 1)
        if degree < 1:
            raise ValueError(
                f"safety level {level} has no realizable degree below target band"
            )
        return degree
    return max(nearest, last_target + 1)


def _retag(batch, role, family, *, key_prefix=""):
    keys = [f"{key_prefix}{key}" for key in batch.source_keys]
    count = batch.source.shape[0]
    target = role == "target"
    return FieldBatch(
        batch.source, batch.truth, batch.frequency, list(batch.labels), [role] * count,
        keys, [family] * count, torch.full((count,), target, dtype=torch.bool),
    )


def _harmonics(pair, degrees, modes, split, seed, sample_seed, role):
    source_key = pair.metadata.get("source_key", pair.pair.split("_to_", 1)[0])
    source_k = np.sqrt(pair.n_src / 6.0)
    return harmonic_batch(
        source_key=source_key,
        source_quadrature=pair.metadata["source_quadrature"],
        target_quadrature=pair.metadata["target_quadrature"],
        degrees=degrees, modes_per_degree=modes, split=split, seed=seed,
        area_src=pair.area_src.detach().cpu().numpy(), role=role,
        pair_key=pair.pair, sample_seed=sample_seed,
    ), source_k


def _level_harmonics(
    pair, levels, modes, split, seed, sample_seed, *, band_lower, band_upper
):
    source_k = np.sqrt(pair.n_src / 6.0)
    parts = []
    for level_index, level in enumerate(levels):
        degree = safety_degree(source_k, level, band_lower, band_upper)
        batch, _ = _harmonics(
            pair, [degree], modes, split, seed, sample_seed + 101 * level_index, "safety"
        )
        # Preserve the realizable harmonic frequency degree/K.  It equals the
        # requested level only when level*K is integral; this distinction is
        # observable for sources such as ICO with non-integer effective K.
        batch = FieldBatch(
            batch.source, batch.truth, batch.frequency,
            batch.labels, batch.roles, batch.source_keys, ["guard_mode"] * batch.source.shape[0],
            torch.zeros(batch.source.shape[0], dtype=torch.bool),
        )
        parts.append(batch)
    return concatenate_batches(parts)


def build_panel(
    config, pair: PairData, *, stage_config, split, epoch, smoke=False, audit=False
):
    """Build the deterministic production target/safety panel for one pair."""
    panel = config.panel
    stage = stage_config
    seed = int(config.seed)
    source_k = np.sqrt(pair.n_src / float(panel.frequency_cells_per_k_squared))
    maximum = 1 if smoke else (panel.audit_max_degrees if audit else panel.max_degrees_per_epoch)
    modes = 1 if smoke else (panel.audit_modes_per_degree if audit else panel.modes_per_degree)
    degrees = band_degrees(source_k, stage.band_lower, stage.band_upper,
                           maximum=maximum, offset=epoch)
    target, _ = _harmonics(pair, degrees, modes, split, seed, seed + 1009 * int(epoch), "target")
    target = _retag(target, "target", "target_mode")
    mixture_count = 0 if smoke else (panel.audit_target_mixtures if audit else panel.target_mixtures)
    target_mix = balanced_mixtures(target, pair.area_src.cpu(), mixture_count, seed + 4001 * int(epoch), role="target")
    target_mix = _retag(target_mix, "target", "target_mixture", key_prefix="target:")

    levels = (0.5,) if smoke else panel.safety_levels
    safety_modes = 1 if smoke else (panel.audit_safety_modes_per_level if audit else panel.safety_modes_per_level)
    safety = _level_harmonics(
        pair, levels, safety_modes, split, seed, seed + 2003 * int(epoch),
        band_lower=stage.band_lower, band_upper=stage.band_upper,
    )
    safety_mix_count = 0 if smoke else (panel.audit_safety_mixtures if audit else panel.safety_mixtures)
    safety_mix = balanced_mixtures(safety, pair.area_src.cpu(), safety_mix_count,
                                   seed + 5003 * int(epoch), role="safety")
    safety_mix = _retag(safety_mix, "safety", "guard_mixture", key_prefix="guard:")

    analytic = analytic_batch(
        pair.metadata["source_quadrature"], pair.metadata["target_quadrature"],
        pair.area_src.detach().cpu().numpy(),
    )
    analytic = _retag(analytic, "safety", "smooth")
    pieces = [target, target_mix, safety, safety_mix, analytic]
    if not smoke:
        real = real_field_batch(
            config.paths.real_field_paths(pair.pair), panel.real_fields,
            pair.n_src, pair.n_tgt, pair.area_src.detach().cpu().numpy(),
        )
        if real is not None:
            pieces.append(real)
    return make_panel(pieces)


def smooth_analytic_fields(source_xyz, target_xyz, area_src):
    functions = (
        lambda xyz: np.sin(2.0 * xyz[:, 0]) + 0.25 * xyz[:, 1] * xyz[:, 2],
        lambda xyz: np.exp(0.6 * xyz[:, 0] - 0.2 * xyz[:, 1]) + 0.1 * xyz[:, 2],
    )
    source, truth = [], []
    area = np.asarray(area_src, dtype=np.float64); area /= np.maximum(area.sum(), 1.0e-30)
    for function in functions:
        x, y = function(np.asarray(source_xyz)), function(np.asarray(target_xyz))
        mean = np.sum(area * x); rms = np.sqrt(np.sum(area * np.square(x - mean)))
        source.append(torch.tensor((x - mean) / rms, dtype=torch.float32))
        truth.append(torch.tensor((y - mean) / rms, dtype=torch.float32))
    return FieldBatch(
        torch.stack(source), torch.stack(truth), torch.full((2,), float("nan")),
        [("smooth", 0), ("smooth", 1)], ["safety", "safety"],
        ["analytic:smooth:0", "analytic:smooth:1"], ["smooth", "smooth"],
        torch.zeros(2, dtype=torch.bool),
    )
