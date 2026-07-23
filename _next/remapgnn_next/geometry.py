from __future__ import annotations

import torch
from torch import nn

from .sparse import edge_sum_fields, index_sum

"""Geometry-derived features and small network pieces."""

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *, depth=2, final_zero=False):
        super().__init__()
        if depth < 1:
            raise ValueError("MLP depth must be at least one")
        layers: list[nn.Module] = []
        dimension = int(input_dim)
        for _ in range(depth - 1):
            layers.extend((nn.Linear(dimension, hidden_dim), nn.SiLU()))
            dimension = int(hidden_dim)
        final = nn.Linear(dimension, output_dim)
        if final_zero:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)


def area_centered_normalize(field, area):
    """Offset/scale normalization stable for tiny and negative rescalings."""
    squeeze = field.ndim == 1
    if squeeze:
        field = field.unsqueeze(0)
    work, weights = field.to(torch.float64), area.to(torch.float64)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    mean = (work * weights.view(1, -1)).sum(dim=1, keepdim=True)
    variance = ((work - mean).square() * weights.view(1, -1)).sum(dim=1, keepdim=True)
    scale = torch.where(variance > 0.0, variance, torch.ones_like(variance)).sqrt()
    normalized = ((work - mean) / scale).to(field.dtype)
    result = normalized, mean.to(field.dtype), scale.to(field.dtype)
    if squeeze:
        return tuple(value.squeeze(0) for value in result)
    return result


def smooth(field, neighbor_index, neighbor_weight):
    squeeze = field.ndim == 1
    if squeeze:
        field = field.unsqueeze(0)
    output = (field[:, neighbor_index] * neighbor_weight.unsqueeze(0)).sum(dim=2)
    return output.squeeze(0) if squeeze else output


def build_smoother(xyz, neighbors=9):
    """Build the audited Gaussian k-nearest-neighbor row smoother."""
    import numpy as np
    points = np.asarray(xyz, dtype=np.float64)
    count = points.shape[0]
    k = min(max(2, int(neighbors)), count)
    try:
        from scipy.spatial import cKDTree
        distance, index = cKDTree(points).query(points, k=k)
    except ImportError:
        distance_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        index = np.argpartition(distance_matrix, kth=k - 1, axis=1)[:, :k]
        distance = np.take_along_axis(distance_matrix, index, axis=1)
    if k == 1:
        index, distance = index[:, None], distance[:, None]
    bandwidth = np.maximum(distance[:, -1], 1.0e-12)
    weight = np.exp(-np.square(distance / bandwidth[:, None]))
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1.0e-300)
    return torch.as_tensor(index, dtype=torch.long), torch.as_tensor(weight, dtype=torch.float32)


def materialize_geometry_columns(frame):
    """Add the supermesh-free geometry columns consumed by clean checkpoints."""
    import numpy as np
    result = frame.copy()
    source_area = result["src_area"].to_numpy(dtype=np.float64, copy=False)
    target_area = result["tgt_area"].to_numpy(dtype=np.float64, copy=False)
    source_h, target_h = np.sqrt(source_area), np.sqrt(target_area)
    mean_h = np.sqrt(0.5 * (source_area + target_area))
    result["src_h"], result["tgt_h"] = source_h, target_h
    result["log_src_area"], result["log_tgt_area"] = np.log(source_area), np.log(target_area)
    ratio = result.get("area_ratio_tgt_over_src", target_area / source_area)
    result["log_area_ratio_tgt_over_src"] = np.log(np.maximum(ratio, 1.0e-30))
    source = result[["src_x", "src_y", "src_z"]].to_numpy(dtype=np.float64)
    target = result[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=np.float64)

    def basis(points):
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        east = np.cross(z[None], points)
        bad = np.linalg.norm(east, axis=1) < 1.0e-10
        east[bad] = np.cross(x[None], points[bad])
        east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1.0e-30)
        north = np.cross(points, east)
        north /= np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1.0e-30)
        return east, north

    target_east, target_north = basis(target)
    source_east, source_north = basis(source)
    delta = source - target
    target_u = (delta * target_east).sum(axis=1)
    target_v = (delta * target_north).sum(axis=1)
    source_u = ((-delta) * source_east).sum(axis=1)
    source_v = ((-delta) * source_north).sum(axis=1)
    distance = np.sqrt(np.square(target_u) + np.square(target_v))
    values = {
        "tgt_tan_e": target_u, "tgt_tan_n": target_v,
        "src_tan_e": source_u, "src_tan_n": source_v,
        "tgt_tan_e_over_h_tgt": target_u / target_h,
        "tgt_tan_n_over_h_tgt": target_v / target_h,
        "tgt_tan_e_over_h_mean": target_u / mean_h,
        "tgt_tan_n_over_h_mean": target_v / mean_h,
        "src_tan_e_over_h_src": source_u / source_h,
        "src_tan_n_over_h_src": source_v / source_h,
        "tan_dist": distance, "tan_dist_over_h_src": distance / source_h,
        "tan_dist_over_h_tgt": distance / target_h,
        "tan_dist_over_h_mean": distance / mean_h,
        "tgt_tan_e2_over_h2": np.square(target_u) / np.square(mean_h),
        "tgt_tan_en_over_h2": target_u * target_v / np.square(mean_h),
        "tgt_tan_n2_over_h2": np.square(target_v) / np.square(mean_h),
    }
    for name, value in values.items():
        result[name] = value.astype(np.float32)
    target_count = result.groupby("target_index", sort=False)["source_index"].transform("size")
    source_count = result.groupby("source_index", sort=False)["target_index"].transform("size")
    result["target_candidate_count_log"] = np.log1p(target_count)
    result["source_candidate_count_log"] = np.log1p(source_count)
    if "knn_rank" in result:
        result["knn_rank_over_target_count"] = result["knn_rank"] / np.maximum(target_count - 1, 1)
    return result


def normalized_feature_tensors(edge_path, feature_spec, normalization):
    import numpy as np
    import pandas as pd
    frame = materialize_geometry_columns(pd.read_parquet(edge_path))
    source_index = frame["source_index"].to_numpy(dtype=np.int64)
    target_index = frame["target_index"].to_numpy(dtype=np.int64)
    n_source, n_target = int(source_index.max()) + 1, int(target_index.max()) + 1

    def unique(index, names, size):
        value = np.zeros((size, len(names)), dtype=np.float32)
        value[index] = frame[names].to_numpy(dtype=np.float32)
        return value

    edge = frame[feature_spec["edge"]].to_numpy(dtype=np.float32)
    source = unique(source_index, feature_spec["source"], n_source)
    target = unique(target_index, feature_spec["target"], n_target)
    edge = (edge - normalization["edge_mean"]) / normalization["edge_std"]
    source = (source - normalization["src_mean"]) / normalization["src_std"]
    target = (target - normalization["tgt_mean"]) / normalization["tgt_std"]
    source_xyz = unique(source_index, ["src_x", "src_y", "src_z"], n_source)
    target_xyz = unique(target_index, ["tgt_x", "tgt_y", "tgt_z"], n_target)
    source_area = unique(source_index, ["src_area"], n_source).reshape(-1)
    target_area = unique(target_index, ["tgt_area"], n_target).reshape(-1)
    return {
        "frame": frame,
        "src_index": torch.tensor(source_index, dtype=torch.long),
        "tgt_index": torch.tensor(target_index, dtype=torch.long),
        "edge": torch.tensor(edge), "source": torch.tensor(source), "target": torch.tensor(target),
        "src_xyz": torch.tensor(source_xyz), "tgt_xyz": torch.tensor(target_xyz),
        "area_src": torch.tensor(source_area), "area_tgt": torch.tensor(target_area),
    }


def graph_features(field, neighbor_index, neighbor_weight, area, epsilon=1.0e-4):
    first = smooth(field, neighbor_index, neighbor_weight)
    second = smooth(first, neighbor_index, neighbor_weight)
    high_first = (field - first).abs()
    high_second = (field - second).abs()
    curvature_raw = (field - 2.0 * first + second).abs()
    roughness = high_first + high_second + curvature_raw
    graph = torch.stack(
        (
            torch.log1p(high_first),
            torch.log1p(high_second),
            curvature_raw / (roughness + float(epsilon)),
            high_first / (high_first + high_second + float(epsilon)),
        ),
        dim=2,
    )
    area_model = area.to(field.dtype)
    denominator = (area_model.view(1, -1) * field.square()).sum(dim=1).clamp_min(1.0e-30)
    energy_first = (
        area_model.view(1, -1) * (field - first).square()
    ).sum(dim=1) / denominator
    energy_second = (
        area_model.view(1, -1) * (field - second).square()
    ).sum(dim=1) / denominator
    global_features = torch.stack(
        (
            torch.log(energy_first + float(epsilon)),
            torch.log(energy_second + float(epsilon)),
            torch.log(energy_first / (energy_second + float(epsilon)) + float(epsilon)),
            energy_first / (energy_second + float(epsilon)),
        ),
        dim=1,
    )
    return graph, global_features, first, second


def intrinsic_geometry_features(pair, reference, epsilon=1.0e-8):
    """Eight rotation-invariant edge/stencil descriptors used by every stage."""
    source_index, target_index = pair.src_index, pair.tgt_index
    source, target = pair.src_xyz[source_index], pair.tgt_xyz[target_index]
    tangent = source - (source * target).sum(dim=1, keepdim=True) * target
    source_h = pair.area_src[source_index].clamp_min(1.0e-20).sqrt()
    target_h = pair.area_tgt[target_index].clamp_min(1.0e-20).sqrt()
    tangent_scaled = tangent / torch.sqrt(source_h * target_h).view(-1, 1)
    radius = tangent_scaled.norm(dim=1)
    n_tgt = pair.n_tgt
    first = edge_sum_fields(
        reference.view(1, -1, 1) * tangent_scaled.view(1, -1, 3), target_index, n_tgt
    ).squeeze(0)
    outer = tangent_scaled[:, :, None] * tangent_scaled[:, None, :]
    second = tangent_scaled.new_zeros((n_tgt, 3, 3))
    second.index_add_(0, target_index, reference.view(-1, 1, 1) * outer)
    trace = second.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(float(epsilon))
    frobenius = second.square().sum(dim=(1, 2))
    first_norm = first.norm(dim=1)
    first_dot = (tangent_scaled * first[target_index]).sum(dim=1)
    second_edge = torch.einsum("ei,eij,ej->e", tangent_scaled, second[target_index], tangent_scaled)
    radius_squared = radius.square().clamp_min(float(epsilon))
    anisotropy = (2.0 * frobenius / trace.square() - 1.0).clamp(0.0, 1.0)
    return torch.stack(
        (
            radius,
            radius.square(),
            torch.log((pair.area_src[source_index] / pair.area_tgt[target_index]).clamp_min(1.0e-20)),
            pair.fv_operator.weight.to(pair.edge_features.dtype),
            first_dot / trace.sqrt()[target_index],
            first_norm[target_index] / trace.sqrt()[target_index],
            second_edge / (radius_squared * trace[target_index]),
            anisotropy[target_index],
        ),
        dim=1,
    )
