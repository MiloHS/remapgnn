from __future__ import annotations

"""Categorical harmonic-band panels for conservative bilinear training."""

import hashlib

import numpy as np
import torch

from .fields import (
    analytic_batch, concatenate_batches, harmonic_batch, real_field_batch,
    stable_seed,
)
from .panels import assert_split_disjoint
from .types import FieldBatch, PairData


def _retag(batch, *, role, family, band, shared=False):
    count = batch.source.shape[0]
    degrees = torch.tensor(
        [int(label[0]) if int(label[0]) >= 0 else -1 for label in batch.labels],
        dtype=torch.long,
    )
    return FieldBatch(
        batch.source, batch.truth,
        torch.full((count,), float("nan"), dtype=torch.float32),
        list(batch.labels), [role] * count, list(batch.source_keys),
        [family] * count, torch.full((count,), role == "target", dtype=torch.bool),
        torch.full((count,), bool(shared), dtype=torch.bool),
        degrees, [band] * count,
    )


def _empty_like(batch):
    return FieldBatch(
        batch.source[:0], batch.truth[:0], batch.frequency[:0], [], [], [],
        [], torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool),
        torch.zeros(0, dtype=torch.long), [],
    )


def _sample_degrees(band, maximum, epoch):
    values = np.arange(band.degree_min, band.degree_max + 1)
    count = min(int(maximum), values.size)
    if count == values.size:
        return values.tolist()
    start = int(epoch) % values.size
    indices = (
        start + np.floor(np.arange(count) * values.size / count).astype(int)
    ) % values.size
    return sorted({int(values[index]) for index in indices})


def _harmonics(
    config, pair, band, split, epoch, modes, maximum, role, family,
    *, degrees=None,
):
    source_key = pair.metadata.get("source_key", pair.pair.split("_to_", 1)[0])
    batch = harmonic_batch(
        source_key=source_key,
        source_quadrature=pair.metadata["source_quadrature"],
        target_quadrature=pair.metadata["target_quadrature"],
        degrees=(
            _sample_degrees(band, maximum, epoch)
            if degrees is None else list(degrees)
        ),
        modes_per_degree=modes,
        split=split,
        seed=config.seed,
        area_src=pair.area_src.detach().cpu().numpy(),
        role=role,
        pair_key=pair.pair,
        sample_seed=config.seed + 1009 * int(epoch),
        frequency_cells_per_k_squared=1.0,
    )
    return _retag(batch, role=role, family=family, band=band.name)


def _normalized_mix(source, truth, area):
    rms = (area * source.square()).sum().sqrt()
    if float(rms) <= 1.0e-12:
        raise ValueError("mixture components cancel to zero source RMS")
    return source / rms, truth / rms


def _cross_target_role(fraction):
    return "target" if float(fraction) > 0.5 else "safety"


def _within_band_mixtures(batch, area_src, count, seed, *, role, family):
    if count <= 0 or batch.source.shape[0] < 2:
        return _empty_like(batch)
    rng = np.random.default_rng(int(seed))
    area = area_src.to(batch.source.dtype)
    area = area / area.sum().clamp_min(1.0e-30)
    source, truth, keys = [], [], []
    for index in range(int(count)):
        size = min(batch.source.shape[0], 2 + index % 3)
        chosen = rng.choice(batch.source.shape[0], size=size, replace=False)
        coefficient = torch.tensor(rng.standard_normal(size), dtype=batch.source.dtype)
        mixed_source = (coefficient[:, None] * batch.source[chosen]).sum(0)
        mixed_truth = (coefficient[:, None] * batch.truth[chosen]).sum(0)
        mixed_source, mixed_truth = _normalized_mix(
            mixed_source, mixed_truth, area
        )
        components = [
            (batch.source_keys[int(item)], float(coefficient[position]))
            for position, item in enumerate(chosen)
        ]
        digest = hashlib.sha256(repr(components).encode()).hexdigest()[:20]
        source.append(mixed_source)
        truth.append(mixed_truth)
        keys.append(f"{family}:{digest}")
    return FieldBatch(
        torch.stack(source), torch.stack(truth),
        torch.full((len(source),), float("nan")),
        [(-1, -1)] * len(source), [role] * len(source), keys,
        [family] * len(source),
        torch.full((len(source),), role == "target", dtype=torch.bool),
        torch.zeros(len(source), dtype=torch.bool),
        torch.full((len(source),), -1, dtype=torch.long),
        [batch.bands[0]] * len(source),
    )


def _cross_mixtures(
    left, rights, area_src, count, seed, *, role, family, fractions=None
):
    if count <= 0 or not rights:
        return _empty_like(left)
    rng = np.random.default_rng(int(seed))
    area = area_src.to(left.source.dtype)
    area = area / area.sum().clamp_min(1.0e-30)
    fractions = tuple(fractions or (0.5,))
    source, truth, keys, bands, families = [], [], [], [], []
    identities = set()
    for index in range(int(count)):
        right = rights[index % len(rights)]
        fraction = float(fractions[index % len(fractions)])
        a = np.sqrt(fraction)
        # Component sampling is random but the panel is a set of scientific
        # fields. Resample deterministic collisions instead of silently
        # inserting the same field twice and tripping the source-key guard.
        for _ in range(1024):
            first = int(rng.integers(left.source.shape[0]))
            second = int(rng.integers(right.source.shape[0]))
            sign = -1.0 if rng.integers(2) else 1.0
            b = sign * np.sqrt(1.0 - fraction)
            mixed_source = a * left.source[first] + b * right.source[second]
            mixed_truth = a * left.truth[first] + b * right.truth[second]
            if float((area * mixed_source.square()).sum().sqrt()) <= 1.0e-12:
                b = abs(b)
                mixed_source = a * left.source[first] + b * right.source[second]
                mixed_truth = a * left.truth[first] + b * right.truth[second]
            identity = (
                left.source_keys[first], right.source_keys[second],
                float(a), float(b), role,
            )
            if identity not in identities:
                identities.add(identity)
                break
        else:
            raise ValueError(
                "could not generate the requested number of unique "
                f"{family} fields"
            )
        mixed_source, mixed_truth = _normalized_mix(
            mixed_source, mixed_truth, area
        )
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:20]
        source.append(mixed_source)
        truth.append(mixed_truth)
        keys.append(f"{family}:{digest}")
        families.append(
            f"{family}_{int(round(100 * fraction)):02d}"
            if family == "cross_target_mixture" else family
        )
        bands.append(
            left.bands[first] if role == "target"
            else f"{left.bands[first]}+{right.bands[second]}"
        )
    return FieldBatch(
        torch.stack(source), torch.stack(truth),
        torch.full((len(source),), float("nan")),
        [(-1, -1)] * len(source), [role] * len(source), keys,
        families,
        torch.full((len(source),), role == "target", dtype=torch.bool),
        torch.zeros(len(source), dtype=torch.bool),
        torch.full((len(source),), -1, dtype=torch.long), bands,
    )


def build_band_panel(
    config, pair: PairData, *, stage_config, split, epoch,
    smoke=False, audit=False,
):
    panel = config.panel
    target_band = config.band(stage_config.target_band)
    guard_bands = [band for band in config.bands if band.name != target_band.name]
    maximum = 1 if smoke else (
        panel.audit_max_degrees if audit else panel.max_degrees_per_epoch
    )
    modes = 1 if smoke else (
        panel.audit_modes_per_degree if audit else panel.modes_per_degree
    )
    multiplier = panel.audit_mixture_multiplier if audit and not smoke else 1
    target = _harmonics(
        config, pair, target_band, split, epoch, modes, maximum,
        "target", "target_mode",
    )
    guards = []
    for index, band in enumerate(guard_bands):
        boundary = index == 0 and not audit
        boundary_degrees = (
            range(
                band.degree_min,
                min(
                    band.degree_max + 1,
                    band.degree_min + panel.boundary_guard_width,
                ),
            )
            if boundary else None
        )
        guards.append(_harmonics(
            config, pair, band, split, epoch + 101 * (index + 1),
            modes, maximum, "safety",
            "boundary_guard" if boundary else "guard_mode",
            degrees=boundary_degrees,
        ))
    pieces = [target, *guards]
    target_mix_count = 1 if smoke else panel.target_mixtures * multiplier
    guard_total = 1 if smoke else panel.guard_mixtures * multiplier
    cross_target_count = 1 if smoke else panel.cross_target_mixtures * multiplier
    cross_guard_total = 1 if smoke else panel.cross_guard_mixtures * multiplier
    pieces.append(_within_band_mixtures(
        target, pair.area_src, target_mix_count,
        stable_seed(f"{pair.pair}:{stage_config.name}:{split}:target", epoch),
        role="target", family="target_mixture",
    ))
    guard_base, guard_remainder = divmod(guard_total, len(guards))
    guard_counts = [
        guard_base + (1 if index < guard_remainder else 0)
        for index in range(len(guards))
    ]
    pieces.extend(
        _within_band_mixtures(
            guard, pair.area_src, count,
            stable_seed(f"{pair.pair}:{guard.bands[0]}:{split}:guard", epoch),
            role="safety", family="guard_mixture",
        )
        for guard, count in zip(guards, guard_counts)
    )
    fraction_base, fraction_remainder = divmod(
        cross_target_count, len(panel.cross_target_fractions)
    )
    for fraction_index, fraction in enumerate(panel.cross_target_fractions):
        count = fraction_base + (
            1 if fraction_index < fraction_remainder else 0
        )
        pieces.append(_cross_mixtures(
            target, guards, pair.area_src, count,
            stable_seed(
                f"{pair.pair}:{stage_config.name}:{split}:cross-target:"
                f"{fraction}", epoch
            ),
            role=_cross_target_role(fraction),
            family="cross_target_mixture",
            fractions=(fraction,),
        ))
    combinations = [
        (guards[left], guards[right])
        for left in range(len(guards))
        for right in range(left + 1, len(guards))
    ]
    cross_base, cross_remainder = divmod(
        cross_guard_total, len(combinations)
    )
    for index, (left, right) in enumerate(combinations):
        pieces.append(_cross_mixtures(
            left, [right], pair.area_src,
            cross_base + (1 if index < cross_remainder else 0),
            stable_seed(
                f"{pair.pair}:{left.bands[0]}:{right.bands[0]}:"
                f"{split}:cross-guard", epoch
            ),
            role="safety", family="cross_guard_mixture",
        ))
    analytic = _retag(
        analytic_batch(
            pair.metadata["source_quadrature"],
            pair.metadata["target_quadrature"],
            pair.area_src.detach().cpu().numpy(),
        ),
        role="safety", family="smooth", band="unbanded", shared=True,
    )
    pieces.append(analytic)
    if not smoke:
        real = real_field_batch(
            config.paths.real_field_paths(pair.pair), panel.real_fields,
            pair.n_src, pair.n_tgt, pair.area_src.detach().cpu().numpy(),
        )
        if real is not None:
            pieces.append(_retag(
                real, role="safety", family="real",
                band="unbanded", shared=True,
            ))
    result = concatenate_batches(pieces)
    if len(result.source_keys) != len(set(result.source_keys)):
        raise ValueError("categorical band panel contains duplicate source keys")
    return result


__all__ = ["build_band_panel", "assert_split_disjoint"]
