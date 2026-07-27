from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from remapgnn_next.comparison import (
    _metrics, _real, evaluate_methods, frequency_summary, parse_methods,
    plot_frequency, plot_spatial, request_hash, select_fields,
    write_values_netcdf,
)
from remapgnn_next.config import StageConfig
from remapgnn_next.progressive import ConservativeCorrectionStage, ProgressiveRemapper
from remapgnn_next.types import FieldBatch


def with_quadrature(pair):
    def quadrature(points):
        count = points.shape[0]
        return {
            "points": points.numpy().astype(np.float64),
            "weights": np.ones(count),
            "cell_index": np.arange(count),
            "cell_area": np.ones(count),
            "area": np.ones(count),
            "centers": points.numpy().astype(np.float64),
        }
    return replace(
        pair,
        metadata={
            "source_key": "SRC",
            "source_quadrature": quadrature(pair.src_xyz),
            "target_quadrature": quadrature(pair.tgt_xyz),
        },
    )


def analysis_config(tmp_path=None):
    paths = None
    if tmp_path is not None:
        class Paths:
            @staticmethod
            def real_field_paths(pair):
                return tmp_path / "source.nc", tmp_path / "target.nc"
        paths = Paths()
    return SimpleNamespace(
        seed=2407,
        panel=SimpleNamespace(
            frequency_cells_per_k_squared=6.0,
            audit_max_degrees=2,
            audit_modes_per_degree=2,
            audit_target_mixtures=0,
            safety_levels=(0.5,),
            audit_safety_modes_per_level=1,
            audit_safety_mixtures=0,
            real_fields=(),
        ),
        paths=paths,
    )


def test_method_evaluation_matches_independent_metrics(synthetic_pair):
    pair = with_quadrature(synthetic_pair)
    stage = ConservativeCorrectionStage(
        StageConfig(name="mid", band_lower=1.0, band_upper=1.25)
    )
    model = ProgressiveRemapper(pair.fv_operator, [stage])
    fields = select_fields(
        analysis_config(), pair, model, ["analytic:smooth1"],
    )
    rows, predictions = evaluate_methods(
        model, pair, fields, ["fv", "stage:mid", "np2"],
        pair.fv_operator, device="cpu",
    )
    assert {row["method"] for row in rows} == {"fv", "stage:mid", "np2"}
    assert all(0 in values for values in predictions.values())
    truth = fields.batch.truth[0].double().numpy()
    prediction = predictions["fv"][0].double().numpy()
    area = pair.area_tgt.double().numpy()
    expected = np.sqrt(
        np.sum(area * np.square(prediction - truth))
        / np.sum(area * np.square(truth))
    )
    observed = next(row for row in rows if row["method"] == "fv")
    assert observed["area_relative_l2"] == pytest.approx(expected, rel=1e-6)
    assert observed["error_over_fv"] == 1.0


def test_profile_fields_are_streamed_not_retained(synthetic_pair):
    pair = with_quadrature(synthetic_pair)
    model = ProgressiveRemapper(pair.fv_operator, [
        ConservativeCorrectionStage(
            StageConfig(name="mid", band_lower=1.0, band_upper=1.25)
        )
    ])
    fields = select_fields(
        analysis_config(), pair, model, [],
        band=(1.0, 3.0), profile_degrees=2, profile_modes=1,
    )
    assert not bool(fields.explicit.any())
    rows, predictions = evaluate_methods(
        model, pair, fields, ["fv"], device="cpu",
    )
    assert rows
    assert predictions["fv"] == {}
    summary = frequency_summary(rows)
    assert summary
    assert all(row["n_fields"] >= 1 for row in summary)


def test_real_field_normalization_restores_physical_values(tmp_path, synthetic_pair):
    pair = with_quadrature(synthetic_pair)
    source = np.arange(pair.n_src, dtype=np.float64) + 10.0
    truth = np.arange(pair.n_tgt, dtype=np.float64) * 2.0 + 11.0
    xr.Dataset({"Topography": ("cell", source)}).to_netcdf(tmp_path / "source.nc")
    xr.Dataset({"Topography": ("cell", truth)}).to_netcdf(tmp_path / "target.nc")
    fields = _real(analysis_config(tmp_path), pair, "Topography")
    assert np.allclose(fields.display_source[0].numpy(), source)
    assert np.allclose(fields.display_truth[0].numpy(), truth)
    area = pair.area_src.numpy().astype(np.float64)
    area /= area.sum()
    normalized = fields.batch.source[0].numpy()
    assert abs(np.sum(area * normalized)) < 1e-6
    assert np.sum(area * normalized**2) == pytest.approx(1.0, rel=1e-6)


def test_metric_boundaries():
    area = torch.tensor([0.25, 0.75], dtype=torch.float64)
    result = _metrics(
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        area, area, 1.0e-8, -3.0,
    )
    assert result["area_relative_l2"] == 0.0
    assert result["area_rmse"] == 0.0
    assert result["conservation_error"] == 0.0


def test_netcdf_and_plots_round_trip(tmp_path, synthetic_pair):
    pair = with_quadrature(synthetic_pair)
    model = ProgressiveRemapper(pair.fv_operator, [
        ConservativeCorrectionStage(
            StageConfig(name="mid", band_lower=1.0, band_upper=1.25)
        )
    ])
    fields = select_fields(
        analysis_config(), pair, model, ["harmonic:1:0"],
    )
    rows, predictions = evaluate_methods(
        model, pair, fields, ["fv"], device="cpu",
    )
    values = write_values_netcdf(
        tmp_path / "values.nc", pair, fields, ["fv"], predictions
    )
    with xr.open_dataset(values) as data:
        assert data.sizes["field"] == 1
        assert data.sizes["method"] == 1
        assert np.allclose(data["error"], data["prediction"] - data["truth"])
    spatial = plot_spatial(
        tmp_path / "spatial.png", pair, fields, 0, ["fv"], predictions
    )
    profile = plot_frequency(
        tmp_path / "profile.png", frequency_summary(rows), model.stages
    )
    assert spatial.stat().st_size > 0
    assert profile.stat().st_size > 0


def test_method_and_request_validation():
    assert parse_methods(
        ["fv", "stage:mid", "fv"], ["mid"]
    ) == ("fv", "stage:mid")
    assert request_hash({"b": 2, "a": 1}) == request_hash({"a": 1, "b": 2})
