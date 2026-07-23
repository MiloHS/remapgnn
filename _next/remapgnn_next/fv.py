from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as functional

from .constraints import project_with_moment_relaxation
from .sparse import index_sum
from .types import PairData, SparseOperator

"""Builds the frozen finite-volume baseline operator."""

def edge_softmax(logits, target_index, n_target):
    work = logits.float()
    maximum = torch.full(
        (n_target,), -torch.inf, device=logits.device, dtype=torch.float32
    )
    maximum.scatter_reduce_(0, target_index, work, reduce="amax", include_self=True)
    exponential = torch.exp((work - maximum[target_index]).clamp(-50.0, 50.0))
    return exponential / index_sum(exponential, target_index, n_target)[target_index].clamp_min(1.0e-12)


class GeometryNetwork(nn.Module):
    """Frozen residual-gated bipartite geometry network used by the FV base."""

    def __init__(self, source_dim, target_dim, edge_dim, hidden=128, decoder_chunk_size=10000):
        super().__init__()
        self.decoder_chunk_size = int(decoder_chunk_size)
        self.src_encoder = nn.Sequential(
            nn.Linear(source_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.tgt_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.src_to_tgt_msg = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.attn_score = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.attn_value = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.context_gate = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.tgt_update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.tgt_to_src_msg = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.src_update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.edge_logit = nn.Linear(hidden, 1)
        self.raw_weight = nn.Linear(hidden, 1)

    def forward(self, source_node, target_node, edge, source_index, target_index):
        n_source, n_target = source_node.shape[0], target_node.shape[0]
        source_hidden = self.src_encoder(source_node)
        target_hidden = self.tgt_encoder(target_node)
        edge_hidden = self.edge_encoder(edge)
        target_degree = index_sum(
            torch.ones_like(target_index, dtype=torch.float32), target_index, n_target
        ).clamp_min(1.0)
        source_degree = index_sum(
            torch.ones_like(source_index, dtype=torch.float32), source_index, n_source
        ).clamp_min(1.0)
        mean_message = self.src_to_tgt_msg(
            torch.cat((source_hidden[source_index], edge_hidden), dim=1)
        )
        mean_context = index_sum(mean_message, target_index, n_target) / target_degree[:, None]
        attention_logits, attention_values = [], []
        for start in range(0, edge_hidden.shape[0], self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, edge_hidden.shape[0])
            si, ti, eh = source_index[start:end], target_index[start:end], edge_hidden[start:end]
            attention_logits.append(self.attn_score(torch.cat(
                (target_hidden[ti], source_hidden[si], eh), dim=1
            )).squeeze(-1))
            attention_values.append(self.attn_value(torch.cat((source_hidden[si], eh), dim=1)))
        attention_logits = torch.cat(attention_logits)
        attention_values = torch.cat(attention_values)
        attention = edge_softmax(attention_logits, target_index, n_target)
        attention_context = index_sum(
            attention_values * attention[:, None].to(attention_values.dtype), target_index, n_target
        )
        gate = torch.sigmoid(self.context_gate(torch.cat(
            (target_hidden, mean_context, attention_context), dim=1
        )))
        context = gate * attention_context + (1.0 - gate) * mean_context
        target_updated = self.tgt_update(torch.cat((target_hidden, context), dim=1))
        reverse = self.tgt_to_src_msg(torch.cat(
            (target_updated[target_index], edge_hidden), dim=1
        ))
        source_context = index_sum(reverse, source_index, n_source) / source_degree[:, None]
        source_updated = self.src_update(torch.cat((source_hidden, source_context), dim=1))
        logits, raw = [], []
        for start in range(0, edge_hidden.shape[0], self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, edge_hidden.shape[0])
            decoded = self.edge_decoder(torch.cat(
                (
                    source_updated[source_index[start:end]],
                    target_updated[target_index[start:end]], edge_hidden[start:end],
                ), dim=1,
            ))
            logits.append(self.edge_logit(decoded).squeeze(-1))
            raw.append(self.raw_weight(decoded).squeeze(-1))
        logits, raw = torch.cat(logits), torch.cat(raw)
        return logits, functional.softplus(raw) + 1.0e-12


@dataclass(frozen=True)
class FVBuildConfig:
    scale: float = 1.0
    projection_iterations: int = 800
    projection_epsilon_relative: float = 1.0e-12
    linear_ridge: float = 1.0e-4
    linear_relax: float = 1.0
    linear_iterations: int = 1
    quadratic_ridge: float = 1.0e-3
    quadratic_relax: float = 1.0
    quadratic_iterations: int = 3


def build_fv_operator(
    network: GeometryNetwork,
    *,
    source_node_features,
    target_node_features,
    edge_features,
    src_index,
    tgt_index,
    area_src,
    area_tgt,
    source_linear_moments,
    target_linear_moments,
    source_quadratic_moments,
    target_quadratic_moments,
    config=FVBuildConfig(),
    provenance: Mapping | None = None,
):
    """Build the authoritative relax1 FV operator from geometry and moments."""
    network.eval()
    with torch.no_grad():
        signed_score, _ = network(
            source_node_features, target_node_features, edge_features, src_index, tgt_index
        )
        n_target = int(area_tgt.numel())
        degree = index_sum(torch.ones_like(signed_score), tgt_index, n_target).clamp_min(1.0)
        uniform_mass = area_tgt.to(signed_score.dtype)[tgt_index] / degree[tgt_index]
        raw_mass = uniform_mass * (1.0 + float(config.scale) * signed_score.float())
        linear = source_linear_moments[src_index] - target_linear_moments[tgt_index]
        quadratic = source_quadratic_moments[src_index] - target_quadratic_moments[tgt_index]
        mass = project_with_moment_relaxation(
            raw_mass, src_index, tgt_index, area_src, area_tgt,
            linear_coefficient=linear, quadratic_coefficient=quadratic,
            linear_relax=config.linear_relax, quadratic_relax=config.quadratic_relax,
            linear_iterations=config.linear_iterations,
            quadratic_iterations=config.quadratic_iterations,
            linear_ridge=config.linear_ridge, quadratic_ridge=config.quadratic_ridge,
            projection_iterations=config.projection_iterations,
            epsilon_relative=config.projection_epsilon_relative,
        )
    return SparseOperator.from_mass(
        src_index, tgt_index, mass, area_src, area_tgt,
        provenance={} if provenance is None else provenance,
    )


def geometry_network_from_checkpoint(pack: Mapping) -> GeometryNetwork:
    schema = pack.get("schema", {})
    features = pack.get("features", {})
    network = GeometryNetwork(
        len(features.get("source", pack.get("src_node_features", ()))),
        len(features.get("target", pack.get("tgt_node_features", ()))),
        len(features.get("edge", pack.get("edge_features", ()))),
        hidden=int(schema.get("hidden", pack.get("hidden", 128))),
        decoder_chunk_size=int(schema.get("decoder_chunk_size", pack.get("decoder_chunk_size", 10000))),
    )
    state = pack.get("state") or pack.get("model_state_dict")
    network.load_state_dict(state, strict=True)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network.eval()


def build_pair_from_files(
    pair_name,
    edge_path,
    map_path,
    fv_checkpoint,
    progressive_checkpoint,
    *,
    device="cpu",
    quadrature_resolution=8,
    smoother_neighbors=9,
):
    """Load geometry, build the authoritative FV base, and assemble PairData."""
    from .fields import grid_moments, grid_quadrature
    from .geometry import build_smoother, normalized_feature_tensors

    fv_features = fv_checkpoint["features"]
    tensors = normalized_feature_tensors(
        edge_path,
        {"edge": fv_features["edge"], "source": fv_features["source"],
         "target": fv_features["target"]},
        fv_features["normalization"],
    )
    fv_quadrature_resolution = int(fv_checkpoint.get("quadrature_resolution", 8))
    source_moments = grid_moments(
        map_path, "a", fv_quadrature_resolution, tensors["src_xyz"].numpy()
    )
    target_moments = grid_moments(
        map_path, "b", fv_quadrature_resolution, tensors["tgt_xyz"].numpy()
    )
    if int(quadrature_resolution) == fv_quadrature_resolution:
        source_panel_quadrature, target_panel_quadrature = source_moments, target_moments
    else:
        source_panel_quadrature = grid_quadrature(
            map_path, "a", quadrature_resolution, tensors["src_xyz"].numpy()
        )
        target_panel_quadrature = grid_quadrature(
            map_path, "b", quadrature_resolution, tensors["tgt_xyz"].numpy()
        )
    network = geometry_network_from_checkpoint(fv_checkpoint).to(device)
    build = FVBuildConfig(**fv_checkpoint["build"])
    moved = {name: value.to(device) if torch.is_tensor(value) else value for name, value in tensors.items()}
    operator = build_fv_operator(
        network,
        source_node_features=moved["source"], target_node_features=moved["target"],
        edge_features=moved["edge"], src_index=moved["src_index"], tgt_index=moved["tgt_index"],
        area_src=moved["area_src"], area_tgt=moved["area_tgt"],
        source_linear_moments=torch.tensor(source_moments["coordinate"], dtype=torch.float32, device=device),
        target_linear_moments=torch.tensor(target_moments["coordinate"], dtype=torch.float32, device=device),
        source_quadratic_moments=torch.tensor(source_moments["quadratic"], dtype=torch.float32, device=device),
        target_quadratic_moments=torch.tensor(target_moments["quadratic"], dtype=torch.float32, device=device),
        config=build,
        provenance={"fv_checkpoint_sha256": fv_checkpoint.get("source", {}).get("sha256")},
    )
    runtime = progressive_checkpoint["runtime_data"]
    edge_names = runtime["edge_features"]
    statistics = runtime["normalization"]
    edge_values = tensors["frame"][edge_names].to_numpy(dtype="float32")
    edge_values = (edge_values - statistics["edge_mean"]) / statistics["edge_std"]
    source_neighbor_index, source_neighbor_weight = build_smoother(
        tensors["src_xyz"].numpy(), smoother_neighbors
    )
    target_neighbor_index, target_neighbor_weight = build_smoother(
        tensors["tgt_xyz"].numpy(), smoother_neighbors
    )
    return PairData(
        pair=str(pair_name), edge_features=torch.tensor(edge_values, device=device),
        src_xyz=moved["src_xyz"], tgt_xyz=moved["tgt_xyz"],
        src_neighbor_index=source_neighbor_index.to(device),
        src_neighbor_weight=source_neighbor_weight.to(device),
        tgt_neighbor_index=target_neighbor_index.to(device),
        tgt_neighbor_weight=target_neighbor_weight.to(device),
        fv_operator=operator, src_node_features=moved["source"], tgt_node_features=moved["target"],
        fv_coord_src=torch.tensor(source_moments["coordinate"], dtype=torch.float32, device=device),
        fv_coord_tgt=torch.tensor(target_moments["coordinate"], dtype=torch.float32, device=device),
        fv_quad_src=torch.tensor(source_moments["quadratic"], dtype=torch.float32, device=device),
        fv_quad_tgt=torch.tensor(target_moments["quadratic"], dtype=torch.float32, device=device),
        metadata={
            "edge_path": str(edge_path), "map_path": str(map_path),
            "source_quadrature": source_panel_quadrature,
            "target_quadrature": target_panel_quadrature,
            "fv_quadrature_resolution": fv_quadrature_resolution,
            "panel_quadrature_resolution": int(quadrature_resolution),
            "source_key": str(pair_name).split("_to_", 1)[0],
        },
    )
