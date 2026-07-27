from __future__ import annotations

"""Human-facing method comparison without changing training or audit state."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Sequence

import numpy as np
import torch

from .checkpoint import (
    CLEAN_PROGRESSIVE_FORMAT, CLEAN_TRAINING_FORMAT,
    load_progressive_checkpoint, load_training_checkpoint, validate_fv_reference,
)
from .evaluation import area_relative_l2, load_map_operator
from .fields import (
    analytic_batch, cell_average, load_real_field, real_spherical_harmonic,
    source_keyed_mode_split,
)
from .panels import build_panel
from .progressive import ProgressiveRemapper
from .provenance import authenticated_load, canonical_json_sha256, file_sha256
from .sparse import apply_operator
from .types import FieldBatch


REAL_FIELDS = (
    "AnalyticalFun1", "AnalyticalFun2", "TotalPrecipWater",
    "CloudFraction", "Topography",
)


@dataclass(frozen=True)
class ComparisonFields:
    batch: FieldBatch
    display_source: torch.Tensor
    display_truth: torch.Tensor
    offsets: torch.Tensor
    scales: torch.Tensor
    explicit: torch.Tensor
    display_names: tuple[str, ...]
    units: tuple[str, ...]

    def subset(self, indices) -> "ComparisonFields":
        index = torch.as_tensor(indices, dtype=torch.long)
        host = index.tolist()
        return ComparisonFields(
            self.batch.subset(index),
            self.display_source[index],
            self.display_truth[index],
            self.offsets[index],
            self.scales[index],
            self.explicit[index],
            tuple(self.display_names[i] for i in host),
            tuple(self.units[i] for i in host),
        )


def _one(
    source, truth, *, frequency, label, key, family, display_name,
    offset=0.0, scale=1.0, unit="normalized", explicit=True,
):
    source = torch.as_tensor(source, dtype=torch.float32).reshape(1, -1)
    truth = torch.as_tensor(truth, dtype=torch.float32).reshape(1, -1)
    offset, scale = float(offset), float(scale)
    return ComparisonFields(
        FieldBatch(
            source, truth, torch.tensor([frequency], dtype=torch.float32),
            [label], ["analysis"], [key], [family],
            torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
        ),
        source * scale + offset,
        truth * scale + offset,
        torch.tensor([offset], dtype=torch.float64),
        torch.tensor([scale], dtype=torch.float64),
        torch.tensor([explicit], dtype=torch.bool),
        (display_name,), (unit,),
    )


def _combine(parts: Sequence[ComparisonFields]) -> ComparisonFields:
    if not parts:
        raise ValueError("at least one field or field set is required")
    by_key = {}
    order = []
    for part in parts:
        for index, key in enumerate(part.batch.source_keys):
            selected = part.subset([index])
            if key not in by_key:
                order.append(key)
                by_key[key] = selected
            elif bool(selected.explicit[0]) and not bool(by_key[key].explicit[0]):
                by_key[key] = selected
    values = [by_key[key] for key in order]
    batches = [value.batch for value in values]
    return ComparisonFields(
        FieldBatch(
            torch.cat([x.source for x in batches]),
            torch.cat([x.truth for x in batches]),
            torch.cat([x.frequency for x in batches]),
            [x.labels[0] for x in batches],
            [x.roles[0] for x in batches],
            [x.source_keys[0] for x in batches],
            [x.families[0] for x in batches],
            torch.cat([x.is_target for x in batches]),
            torch.cat([
                x.shared_anchor
                if x.shared_anchor is not None
                else torch.zeros(1, dtype=torch.bool)
                for x in batches
            ]),
            (
                torch.cat([
                    x.degrees
                    if x.degrees is not None
                    else torch.full((1,), -1, dtype=torch.long)
                    for x in batches
                ])
                if any(x.degrees is not None or x.bands for x in batches)
                else None
            ),
            (
                [
                    (x.bands[0] if x.bands else "unbanded")
                    for x in batches
                ]
                if any(x.degrees is not None or x.bands for x in batches)
                else []
            ),
        ),
        torch.cat([x.display_source for x in values]),
        torch.cat([x.display_truth for x in values]),
        torch.cat([x.offsets for x in values]),
        torch.cat([x.scales for x in values]),
        torch.cat([x.explicit for x in values]),
        tuple(x.display_names[0] for x in values),
        tuple(x.units[0] for x in values),
    )


def _harmonic(pair, degree, order, divisor, *, explicit):
    degree, order = int(degree), int(order)
    if degree < 1 or abs(order) > degree:
        raise ValueError(
            f"invalid harmonic degree/order ({degree}, {order}); require "
            "degree >= 1 and |order| <= degree"
        )
    function = lambda xyz: real_spherical_harmonic(degree, order, xyz)
    source = cell_average(function, pair.metadata["source_quadrature"])
    truth = cell_average(function, pair.metadata["target_quadrature"])
    area = pair.area_src.detach().cpu().numpy().astype(np.float64)
    area /= np.maximum(area.sum(), 1.0e-300)
    rms = float(np.sqrt(np.sum(area * source * source)))
    if rms <= 1.0e-14:
        raise ValueError(f"harmonic Y_{degree}_{order} has zero source RMS")
    source_key = pair.metadata.get("source_key", pair.pair.split("_to_", 1)[0])
    effective_k = math.sqrt(pair.n_src / float(divisor))
    return _one(
        source / rms, truth / rms, frequency=degree / effective_k,
        label=(degree, order), key=f"{source_key}:Y:{degree}:{order}",
        family="harmonic", display_name=f"Y({degree},{order})",
        explicit=explicit,
    )


def _analytic(pair, name):
    batch = analytic_batch(
        pair.metadata["source_quadrature"],
        pair.metadata["target_quadrature"],
        pair.area_src.detach().cpu().numpy(),
    )
    keys = list(batch.source_keys)
    key = f"analytic:{name}"
    if key not in keys:
        raise ValueError(f"unknown analytic field {name!r}; available: smooth1, smooth2")
    index = keys.index(key)
    return _one(
        batch.source[index], batch.truth[index], frequency=float("nan"),
        label=batch.labels[index], key=key, family="analytic",
        display_name=name, explicit=True,
    )


def _real(config, pair, name):
    if name not in REAL_FIELDS:
        raise ValueError(f"unknown real field {name!r}; available: {', '.join(REAL_FIELDS)}")
    source_path, target_path = config.paths.real_field_paths(pair.pair)
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(
            f"paired real-field files are unavailable for {pair.pair}: "
            f"{source_path}, {target_path}"
        )
    source = load_real_field(source_path, name)
    truth = load_real_field(target_path, name)
    if source.size != pair.n_src or truth.size != pair.n_tgt:
        raise ValueError(f"real field {name!r} does not match pair dimensions")
    area = pair.area_src.detach().cpu().numpy().astype(np.float64)
    area /= np.maximum(area.sum(), 1.0e-300)
    offset = float(np.sum(area * source))
    scale = float(np.sqrt(np.sum(area * np.square(source - offset))))
    if scale <= 1.0e-14:
        raise ValueError(f"real field {name!r} has zero source RMS")
    unit = ""
    try:
        import xarray as xr
        with xr.open_dataset(source_path) as data:
            unit = str(data[name].attrs.get("units", ""))
    except (OSError, KeyError):
        pass
    source_key = pair.metadata.get("source_key", pair.pair.split("_to_", 1)[0])
    return _one(
        (source - offset) / scale, (truth - offset) / scale,
        frequency=float("nan"), label=(-500, REAL_FIELDS.index(name)),
        key=f"{source_key}:real:{name}", family="real",
        display_name=name, offset=offset, scale=scale,
        unit=unit or "native", explicit=True,
    )


def _from_batch(batch, *, explicit=False):
    count = batch.source.shape[0]
    return ComparisonFields(
        batch,
        batch.source.clone(),
        batch.truth.clone(),
        torch.zeros(count, dtype=torch.float64),
        torch.ones(count, dtype=torch.float64),
        torch.full((count,), bool(explicit), dtype=torch.bool),
        tuple(batch.source_keys),
        tuple("normalized" for _ in range(count)),
    )


def select_fields(
    config, pair, model, selectors=(), *, band=None, profile_degrees=16,
    profile_modes=3, field_set=None,
):
    parts = []
    for value in selectors:
        kind, separator, payload = value.partition(":")
        if not separator:
            raise ValueError(
                f"invalid field selector {value!r}; expected harmonic:D:O, "
                "analytic:NAME, or real:NAME"
            )
        if kind == "harmonic":
            pieces = payload.split(":")
            if len(pieces) != 2:
                raise ValueError(f"invalid harmonic selector {value!r}")
            parts.append(_harmonic(
                pair, int(pieces[0]), int(pieces[1]),
                config.panel.frequency_cells_per_k_squared, explicit=True,
            ))
        elif kind == "analytic":
            parts.append(_analytic(pair, payload))
        elif kind == "real":
            parts.append(_real(config, pair, payload))
        else:
            raise ValueError(f"unknown field selector kind {kind!r}")

    if band is not None:
        lower, upper = map(float, band)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0 or upper <= lower:
            raise ValueError("band must satisfy 0 <= lower < upper")
        source_k = math.sqrt(
            pair.n_src / float(config.panel.frequency_cells_per_k_squared)
        )
        first = max(1, int(math.ceil(lower * source_k)))
        last = int(math.floor(upper * source_k))
        if last < first:
            raise ValueError("selected band has no realizable harmonic degree")
        available = np.arange(first, last + 1)
        count = min(int(profile_degrees), available.size)
        indices = np.linspace(0, available.size - 1, count).round().astype(int)
        source_key = pair.metadata.get("source_key", pair.pair.split("_to_", 1)[0])
        for degree in sorted(set(int(available[index]) for index in indices)):
            orders = source_keyed_mode_split(
                source_key, degree, config.seed, "audit"
            )[:int(profile_modes)]
            for order in orders:
                parts.append(_harmonic(
                    pair, degree, order,
                    config.panel.frequency_cells_per_k_squared, explicit=False,
                ))

    if field_set is not None:
        kind, separator, stage_name = field_set.partition(":")
        if kind != "audit" or not separator:
            raise ValueError("field set must have the form audit:STAGE")
        names = [stage.name for stage in model.stages]
        if stage_name not in names:
            raise ValueError(f"unknown audit stage {stage_name!r}; available: {names}")
        panel = build_panel(
            config, pair, stage_config=model.stages[names.index(stage_name)].config,
            split="audit", epoch=config.seed, audit=True,
        )
        parts.append(_from_batch(panel))
    return _combine(parts)


def load_comparison_model(checkpoint, config=None):
    pack, checkpoint_sha = authenticated_load(checkpoint)
    if pack.get("format") == CLEAN_PROGRESSIVE_FORMAT:
        model, progressive_pack = load_progressive_checkpoint(checkpoint)
        source_path = Path(checkpoint)
    elif pack.get("format") == CLEAN_TRAINING_FORMAT:
        if config is not None and (
            canonical_json_sha256(pack.get("config"))
            != canonical_json_sha256(config.to_dict())
        ):
            raise ValueError(
                "comparison config differs from the candidate's saved "
                "scientific config"
            )
        model, progressive_pack, source_path = load_training_checkpoint(pack)
    else:
        raise ValueError("comparison requires a clean progressive or completed training checkpoint")
    return model.eval(), progressive_pack, source_path, checkpoint_sha


def parse_methods(values, stage_names):
    methods = []
    for value in values:
        if value in {"fv", "np2"}:
            name = value
        elif value.startswith("stage:"):
            stage = value.split(":", 1)[1]
            if stage not in stage_names:
                raise ValueError(f"unknown stage {stage!r}; available: {stage_names}")
            name = f"stage:{stage}"
        else:
            raise ValueError("methods must be fv, np2, or stage:NAME")
        if name not in methods:
            methods.append(name)
    if not methods:
        raise ValueError("at least one comparison method is required")
    return tuple(methods)


def _metrics(prediction, truth, source, area_tgt, area_src, scale, offset):
    pred = prediction.double()
    target = truth.double()
    source = source.double()
    at = area_tgt.double()
    a_s = area_src.double()
    at_norm = at / at.sum().clamp_min(1.0e-300)
    difference = pred - target
    relative = float(area_relative_l2(
        pred.float().unsqueeze(0), target.float().unsqueeze(0), at
    )[0])
    physical_difference = difference * float(scale)
    physical_pred = pred * float(scale) + float(offset)
    physical_source = source * float(scale) + float(offset)
    return {
        "area_relative_l2": relative,
        "area_rmse": float(
            (at_norm * physical_difference.square()).sum().clamp_min(0).sqrt()
        ),
        "area_mean_signed_error": float((at_norm * physical_difference).sum()),
        "max_absolute_error": float(physical_difference.abs().max()),
        "source_integral": float((a_s * physical_source).sum()),
        "target_integral": float((at * physical_pred).sum()),
        "conservation_error": float(
            (at * physical_pred).sum() - (a_s * physical_source).sum()
        ),
    }


@torch.no_grad()
def evaluate_methods(model, pair, fields, methods, np2=None, *, device="cpu", batch_size=2):
    device = torch.device(device)
    host_pair = pair
    pair = pair.to(device)
    stage_names = [stage.name for stage in model.stages]
    methods = parse_methods(methods, stage_names)
    selected_stages = [
        stage_names.index(method.split(":", 1)[1])
        for method in methods if method.startswith("stage:")
    ]
    runner = None
    if selected_stages:
        runner = ProgressiveRemapper(
            model.base_operator,
            list(model.stages[:max(selected_stages) + 1]),
        ).to(device).eval()
    if "np2" in methods and np2 is None:
        raise ValueError("np2 was selected but no np2 operator was supplied")
    np2_device = None if np2 is None else np2.to(device)
    predictions = {method: {} for method in methods}
    rows = []
    for start in range(0, fields.batch.source.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), fields.batch.source.shape[0])
        source = fields.batch.source[start:stop].to(device)
        diagnostic = None
        if runner is not None:
            _, diagnostic = runner(pair, source, return_diagnostics=True)
        for method in methods:
            if method == "fv":
                value = (
                    diagnostic.fv_output if diagnostic is not None
                    else apply_operator(pair.fv_operator, source)
                )
            elif method == "np2":
                value = apply_operator(np2_device, source)
            else:
                stage_name = method.split(":", 1)[1]
                index = [stage.name for stage in diagnostic.stages].index(stage_name)
                value = diagnostic.stage_outputs[index]
            host_value = value.detach().cpu()
            for local, index in enumerate(range(start, stop)):
                if bool(fields.explicit[index]):
                    predictions[method][index] = host_value[local]
                degree, order = fields.batch.labels[index]
                rows.append({
                    "pair": host_pair.pair,
                    "field": fields.display_names[index],
                    "source_key": fields.batch.source_keys[index],
                    "family": fields.batch.families[index],
                    "degree": int(degree),
                    "order": int(order),
                    "nu": float(fields.batch.frequency[index]),
                    "method": method,
                    "unit": fields.units[index],
                    **_metrics(
                        host_value[local], fields.batch.truth[index],
                        fields.batch.source[index], host_pair.area_tgt,
                        host_pair.area_src, fields.scales[index], fields.offsets[index],
                    ),
                })
    by_field = {}
    for row in rows:
        by_field.setdefault(row["source_key"], {})[row["method"]] = row
    for group in by_field.values():
        fv = group.get("fv")
        np2_row = group.get("np2")
        for row in group.values():
            row["error_over_fv"] = (
                float("nan") if fv is None else
                row["area_relative_l2"] / max(fv["area_relative_l2"], 1.0e-30)
            )
            row["error_over_np2"] = (
                float("nan") if np2_row is None else
                row["area_relative_l2"] / max(np2_row["area_relative_l2"], 1.0e-30)
            )
    return rows, predictions


def frequency_summary(rows):
    groups = {}
    for row in rows:
        if math.isfinite(row["nu"]):
            groups.setdefault((row["method"], row["nu"]), []).append(
                row["area_relative_l2"]
            )
    result = []
    for (method, frequency), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        result.append({
            "method": method, "nu": float(frequency), "n_fields": int(array.size),
            "mean_area_relative_l2": float(array.mean()),
            "median_area_relative_l2": float(np.median(array)),
            "worst_area_relative_l2": float(array.max()),
        })
    return result


def xyz_lon_lat(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    longitude = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    latitude = np.rad2deg(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    return longitude, latitude


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "field"


def request_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_state(root="."):
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=normal")),
    }


def write_values_netcdf(path, pair, fields, methods, predictions):
    explicit = torch.where(fields.explicit)[0].tolist()
    if not explicit:
        return None
    import xarray as xr

    method_names = list(methods)
    normalized_prediction = np.stack([
        np.stack([predictions[name][index].numpy() for index in explicit])
        for name in method_names
    ])
    scale = fields.scales[explicit].numpy()
    offset = fields.offsets[explicit].numpy()
    physical_prediction = (
        normalized_prediction * scale[None, :, None]
        + offset[None, :, None]
    )
    physical_truth = fields.display_truth[explicit].numpy()
    dataset = xr.Dataset(
        data_vars={
            "source_normalized": (
                ("field", "source_cell"),
                fields.batch.source[explicit].numpy(),
            ),
            "truth_normalized": (
                ("field", "target_cell"),
                fields.batch.truth[explicit].numpy(),
            ),
            "prediction_normalized": (
                ("method", "field", "target_cell"), normalized_prediction,
            ),
            "source": (
                ("field", "source_cell"),
                fields.display_source[explicit].numpy(),
            ),
            "truth": (("field", "target_cell"), physical_truth),
            "prediction": (
                ("method", "field", "target_cell"), physical_prediction,
            ),
            "error": (
                ("method", "field", "target_cell"),
                physical_prediction - physical_truth[None, :, :],
            ),
            "source_area": (("source_cell",), pair.area_src.cpu().numpy()),
            "target_area": (("target_cell",), pair.area_tgt.cpu().numpy()),
            "source_xyz": (
                ("source_cell", "xyz"), pair.src_xyz.cpu().numpy(),
            ),
            "target_xyz": (
                ("target_cell", "xyz"), pair.tgt_xyz.cpu().numpy(),
            ),
            "normalization_offset": (("field",), offset),
            "normalization_scale": (("field",), scale),
        },
        coords={
            "method": method_names,
            "field": [fields.display_names[index] for index in explicit],
            "source_key": (
                "field", [fields.batch.source_keys[index] for index in explicit]
            ),
            "unit": ("field", [fields.units[index] for index in explicit]),
            "xyz": ["x", "y", "z"],
        },
        attrs={"pair": pair.pair},
    )
    source_lon, source_lat = xyz_lon_lat(pair.src_xyz.cpu().numpy())
    target_lon, target_lat = xyz_lon_lat(pair.tgt_xyz.cpu().numpy())
    dataset["source_longitude"] = ("source_cell", source_lon)
    dataset["source_latitude"] = ("source_cell", source_lat)
    dataset["target_longitude"] = ("target_cell", target_lon)
    dataset["target_latitude"] = ("target_cell", target_lat)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataset.to_netcdf(temporary)
    temporary.replace(path)
    dataset.close()
    return path


def _scatter(ax, longitude, latitude, values, title, *, vmin, vmax, cmap):
    image = ax.scatter(
        longitude, latitude, c=values, s=0.7, linewidths=0,
        rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap,
    )
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude")
    ax.set_title(title)
    return image


def plot_spatial(path, pair, fields, field_index, methods, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_lon, source_lat = xyz_lon_lat(pair.src_xyz.cpu().numpy())
    target_lon, target_lat = xyz_lon_lat(pair.tgt_xyz.cpu().numpy())
    scale = float(fields.scales[field_index])
    offset = float(fields.offsets[field_index])
    source = fields.display_source[field_index].numpy()
    truth = fields.display_truth[field_index].numpy()
    predicted = {
        method: predictions[method][field_index].numpy() * scale + offset
        for method in methods
    }
    all_values = np.concatenate((truth, *predicted.values()))
    low, high = np.nanpercentile(all_values, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    if low == high:
        low, high = low - 1.0, high + 1.0
    errors = {name: value - truth for name, value in predicted.items()}
    error_limit = float(np.nanpercentile(
        np.abs(np.concatenate(tuple(errors.values()))), 99.5
    ))
    if not np.isfinite(error_limit) or error_limit == 0:
        error_limit = 1.0

    columns = len(methods) + 1
    figure, axes = plt.subplots(
        2, columns, figsize=(4.0 * columns, 7.2), constrained_layout=True,
        squeeze=False,
    )
    field_image = _scatter(
        axes[0, 0], target_lon, target_lat, truth, "truth",
        vmin=low, vmax=high, cmap="viridis",
    )
    _scatter(
        axes[1, 0], source_lon, source_lat, source, "source",
        vmin=low, vmax=high, cmap="viridis",
    )
    for column, method in enumerate(methods, start=1):
        _scatter(
            axes[0, column], target_lon, target_lat, predicted[method], method,
            vmin=low, vmax=high, cmap="viridis",
        )
        error_image = _scatter(
            axes[1, column], target_lon, target_lat, errors[method],
            f"{method} − truth", vmin=-error_limit, vmax=error_limit,
            cmap="coolwarm",
        )
    figure.colorbar(field_image, ax=axes[0, :], shrink=0.8, label=fields.units[field_index])
    figure.colorbar(error_image, ax=axes[1, 1:], shrink=0.8, label="signed error")
    figure.suptitle(f"{pair.pair}: {fields.display_names[field_index]}")
    path = Path(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_frequency(path, summary, stages):
    if not summary:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    methods = list(dict.fromkeys(row["method"] for row in summary))
    for method in methods:
        values = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: row["nu"],
        )
        frequency = [row["nu"] for row in values]
        mean = [max(row["mean_area_relative_l2"], 1.0e-16) for row in values]
        worst = [max(row["worst_area_relative_l2"], 1.0e-16) for row in values]
        line, = axis.plot(frequency, mean, marker="o", markersize=3, label=method)
        axis.plot(frequency, worst, linestyle="--", alpha=0.55, color=line.get_color())
    boundaries = sorted({
        float(value)
        for stage in stages
        for value in (stage.config.band_lower, stage.config.band_upper)
    })
    for boundary in boundaries:
        axis.axvline(boundary, color="0.55", linewidth=0.8, linestyle=":")
    axis.set(
        xlabel="normalized frequency ν",
        ylabel="area-weighted relative L2 error",
        yscale="log",
        title="Frequency error profile (solid mean, dashed worst)",
    )
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    path = Path(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
