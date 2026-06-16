from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum_torch(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Minimal torch-only scatter sum over dimension 0."""
    if src.ndim == 1:
        out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out

    out = torch.zeros((dim_size, src.shape[1]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class MeanGNNSinkhorn(nn.Module):
    """v8-style bipartite GNN: mean aggregation, no attention."""

    def __init__(self, src_dim: int, tgt_dim: int, edge_dim: int, hidden: int = 128, decoder_chunk_size: int = 10000):
        super().__init__()
        self.decoder_chunk_size = decoder_chunk_size

        self.src_encoder = nn.Sequential(
            nn.Linear(src_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.tgt_encoder = nn.Sequential(
            nn.Linear(tgt_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.src_to_tgt_msg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.tgt_to_src_msg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.src_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.tgt_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.edge_logit = nn.Linear(hidden, 1)
        self.raw_weight = nn.Linear(hidden, 1)

    def forward(self, src_node_attr, tgt_node_attr, edge_attr, src_index, tgt_index, n_src, n_tgt):
        src_h = self.src_encoder(src_node_attr)
        tgt_h = self.tgt_encoder(tgt_node_attr)
        edge_h = self.edge_encoder(edge_attr)

        msg_st = self.src_to_tgt_msg(torch.cat([src_h[src_index], edge_h], dim=1))
        agg_t = scatter_sum_torch(msg_st, tgt_index, n_tgt)
        deg_t = scatter_sum_torch(torch.ones_like(tgt_index, dtype=torch.float32), tgt_index, n_tgt)
        agg_t = agg_t / torch.clamp(deg_t[:, None], min=1.0)
        tgt_h2 = self.tgt_update(torch.cat([tgt_h, agg_t], dim=1))

        msg_ts = self.tgt_to_src_msg(torch.cat([tgt_h[tgt_index], edge_h], dim=1))
        agg_s = scatter_sum_torch(msg_ts, src_index, n_src)
        deg_s = scatter_sum_torch(torch.ones_like(src_index, dtype=torch.float32), src_index, n_src)
        agg_s = agg_s / torch.clamp(deg_s[:, None], min=1.0)
        src_h2 = self.src_update(torch.cat([src_h, agg_s], dim=1))

        return self._decode_edges(src_h2, tgt_h2, edge_h, src_index, tgt_index)

    def _decode_edges(self, src_h, tgt_h, edge_h, src_index, tgt_index):
        logits = []
        raw_ws = []

        num_edges = edge_h.shape[0]
        for start in range(0, num_edges, self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, num_edges)
            dec_in = torch.cat(
                [
                    src_h[src_index[start:end]],
                    tgt_h[tgt_index[start:end]],
                    edge_h[start:end],
                ],
                dim=1,
            )
            dec_h = self.edge_decoder(dec_in)
            logits.append(self.edge_logit(dec_h).squeeze(-1))
            raw_ws.append(self.raw_weight(dec_h).squeeze(-1))

        logit = torch.cat(logits, dim=0)
        raw_w = torch.cat(raw_ws, dim=0)

        prob = torch.sigmoid(logit)
        positive_weight = F.softplus(raw_w) + 1.0e-12
        q = torch.sqrt(torch.clamp(prob, min=1.0e-12)) * positive_weight

        return logit, positive_weight, q


class HybridAttentionGNNSinkhorn(nn.Module):
    """v10-style bipartite GNN: mean aggregation + real target-wise source attention."""

    def __init__(self, src_dim: int, tgt_dim: int, edge_dim: int, hidden: int = 128, decoder_chunk_size: int = 10000):
        super().__init__()
        self.decoder_chunk_size = decoder_chunk_size

        self.src_encoder = nn.Sequential(
            nn.Linear(src_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.tgt_encoder = nn.Sequential(
            nn.Linear(tgt_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.src_to_tgt_msg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.attn_score = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        self.attn_value = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.tgt_update = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.tgt_to_src_msg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.src_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.edge_logit = nn.Linear(hidden, 1)
        self.raw_weight = nn.Linear(hidden, 1)

    @staticmethod
    def edge_softmax_by_target(logits: torch.Tensor, tgt_index: torch.Tensor, n_tgt: int) -> torch.Tensor:
        logits_f = logits.float()

        max_per_tgt = torch.full(
            (n_tgt,),
            -torch.inf,
            device=logits.device,
            dtype=torch.float32,
        )
        max_per_tgt.scatter_reduce_(
            0,
            tgt_index,
            logits_f,
            reduce="amax",
            include_self=True,
        )

        shifted = logits_f - max_per_tgt[tgt_index]
        shifted = torch.clamp(shifted, min=-50.0, max=50.0)
        exp_logits = torch.exp(shifted)

        denom = scatter_sum_torch(exp_logits, tgt_index, n_tgt)
        attn = exp_logits / torch.clamp(denom[tgt_index], min=1.0e-12)
        return attn

    def forward(self, src_node_attr, tgt_node_attr, edge_attr, src_index, tgt_index, n_src, n_tgt):
        src_h = self.src_encoder(src_node_attr)
        tgt_h = self.tgt_encoder(tgt_node_attr)
        edge_h = self.edge_encoder(edge_attr)

        num_edges = edge_h.shape[0]

        # 1. Original stable mean source -> target context.
        msg_st = self.src_to_tgt_msg(torch.cat([src_h[src_index], edge_h], dim=1))
        mean_context = scatter_sum_torch(msg_st, tgt_index, n_tgt)
        deg_t = scatter_sum_torch(torch.ones_like(tgt_index, dtype=torch.float32), tgt_index, n_tgt)
        mean_context = mean_context / torch.clamp(deg_t[:, None], min=1.0)

        # 2. Real target-wise source attention context.
        attn_logits = []
        attn_values = []

        for start in range(0, num_edges, self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, num_edges)

            s_idx = src_index[start:end]
            t_idx = tgt_index[start:end]
            e_h = edge_h[start:end]

            score_in = torch.cat([tgt_h[t_idx], src_h[s_idx], e_h], dim=1)
            value_in = torch.cat([src_h[s_idx], e_h], dim=1)

            attn_logits.append(self.attn_score(score_in).squeeze(-1))
            attn_values.append(self.attn_value(value_in))

        attn_logits = torch.cat(attn_logits, dim=0)
        attn_values = torch.cat(attn_values, dim=0)

        attn = self.edge_softmax_by_target(attn_logits, tgt_index, n_tgt)
        weighted_values = attn_values * attn.to(attn_values.dtype)[:, None]
        attn_context = scatter_sum_torch(weighted_values, tgt_index, n_tgt)

        # 3. Hybrid update.
        tgt_h2 = self.tgt_update(torch.cat([tgt_h, mean_context, attn_context], dim=1))

        # 4. Reverse target -> source pass.
        msg_ts = self.tgt_to_src_msg(torch.cat([tgt_h2[tgt_index], edge_h], dim=1))
        agg_s = scatter_sum_torch(msg_ts, src_index, n_src)
        deg_s = scatter_sum_torch(torch.ones_like(src_index, dtype=torch.float32), src_index, n_src)
        agg_s = agg_s / torch.clamp(deg_s[:, None], min=1.0)
        src_h2 = self.src_update(torch.cat([src_h, agg_s], dim=1))

        return self._decode_edges(src_h2, tgt_h2, edge_h, src_index, tgt_index)

    def _decode_edges(self, src_h, tgt_h, edge_h, src_index, tgt_index):
        logits = []
        raw_ws = []

        num_edges = edge_h.shape[0]
        for start in range(0, num_edges, self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, num_edges)
            dec_in = torch.cat(
                [
                    src_h[src_index[start:end]],
                    tgt_h[tgt_index[start:end]],
                    edge_h[start:end],
                ],
                dim=1,
            )
            dec_h = self.edge_decoder(dec_in)
            logits.append(self.edge_logit(dec_h).squeeze(-1))
            raw_ws.append(self.raw_weight(dec_h).squeeze(-1))

        logit = torch.cat(logits, dim=0)
        raw_w = torch.cat(raw_ws, dim=0)

        prob = torch.sigmoid(logit)
        positive_weight = F.softplus(raw_w) + 1.0e-12
        q = torch.sqrt(torch.clamp(prob, min=1.0e-12)) * positive_weight

        return logit, positive_weight, q



class GatedHybridAttentionGNNSinkhorn(HybridAttentionGNNSinkhorn):
    """
    v11-style bipartite GNN.

    Starts from v10 hybrid attention, but learns a per-target hidden-channel gate
    between the stable mean context and selective attention context:

        gate_i = sigmoid(MLP([tgt_h_i, mean_context_i, attn_context_i]))
        context_i = gate_i * attn_context_i + (1 - gate_i) * mean_context_i

    Then target update uses [tgt_h_i, context_i].
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        edge_dim: int,
        hidden: int = 128,
        decoder_chunk_size: int = 10000,
    ):
        super().__init__(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

        self.context_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.tgt_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, src_node_attr, tgt_node_attr, edge_attr, src_index, tgt_index, n_src, n_tgt):
        src_h = self.src_encoder(src_node_attr)
        tgt_h = self.tgt_encoder(tgt_node_attr)
        edge_h = self.edge_encoder(edge_attr)

        num_edges = edge_h.shape[0]

        # 1. Stable mean source -> target context.
        msg_st = self.src_to_tgt_msg(torch.cat([src_h[src_index], edge_h], dim=1))
        mean_context = scatter_sum_torch(msg_st, tgt_index, n_tgt)
        deg_t = scatter_sum_torch(torch.ones_like(tgt_index, dtype=torch.float32), tgt_index, n_tgt)
        mean_context = mean_context / torch.clamp(deg_t[:, None], min=1.0)

        # 2. Real target-wise source attention context.
        attn_logits = []
        attn_values = []

        for start in range(0, num_edges, self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, num_edges)

            s_idx = src_index[start:end]
            t_idx = tgt_index[start:end]
            e_h = edge_h[start:end]

            score_in = torch.cat([tgt_h[t_idx], src_h[s_idx], e_h], dim=1)
            value_in = torch.cat([src_h[s_idx], e_h], dim=1)

            attn_logits.append(self.attn_score(score_in).squeeze(-1))
            attn_values.append(self.attn_value(value_in))

        attn_logits = torch.cat(attn_logits, dim=0)
        attn_values = torch.cat(attn_values, dim=0)

        attn = self.edge_softmax_by_target(attn_logits, tgt_index, n_tgt)
        weighted_values = attn_values * attn.to(attn_values.dtype)[:, None]
        attn_context = scatter_sum_torch(weighted_values, tgt_index, n_tgt)

        # 3. Gated hybrid context.
        gate = torch.sigmoid(self.context_gate(torch.cat([tgt_h, mean_context, attn_context], dim=1)))
        gated_context = gate * attn_context + (1.0 - gate) * mean_context

        tgt_h2 = self.tgt_update(torch.cat([tgt_h, gated_context], dim=1))

        # 4. Reverse target -> source pass.
        msg_ts = self.tgt_to_src_msg(torch.cat([tgt_h2[tgt_index], edge_h], dim=1))
        agg_s = scatter_sum_torch(msg_ts, src_index, n_src)
        deg_s = scatter_sum_torch(torch.ones_like(src_index, dtype=torch.float32), src_index, n_src)
        agg_s = agg_s / torch.clamp(deg_s[:, None], min=1.0)
        src_h2 = self.src_update(torch.cat([src_h, agg_s], dim=1))

        return self._decode_edges(src_h2, tgt_h2, edge_h, src_index, tgt_index)


class ResidualGatedHybridAttentionGNNSinkhorn(HybridAttentionGNNSinkhorn):
    """
    v12-style bipartite GNN.

    Safer version of v11:
      - mean_context remains the baseline
      - attention only enters as a bounded residual correction

        gate_i = sigmoid(MLP([tgt_h_i, mean_context_i, attn_context_i]))
        context_i = mean_context_i + residual_scale * gate_i * (attn_context_i - mean_context_i)

    With residual_scale=0.25, even a fully-open gate only moves 25% of the way
    from mean context to attention context.
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        edge_dim: int,
        hidden: int = 128,
        decoder_chunk_size: int = 10000,
        residual_scale: float = 0.25,
    ):
        super().__init__(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

        self.residual_scale = residual_scale

        self.context_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Residual-gated context has shape hidden, so target update uses [tgt_h, context].
        self.tgt_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, src_node_attr, tgt_node_attr, edge_attr, src_index, tgt_index, n_src, n_tgt):
        src_h = self.src_encoder(src_node_attr)
        tgt_h = self.tgt_encoder(tgt_node_attr)
        edge_h = self.edge_encoder(edge_attr)

        num_edges = edge_h.shape[0]

        # 1. Stable mean source -> target context.
        msg_st = self.src_to_tgt_msg(torch.cat([src_h[src_index], edge_h], dim=1))
        mean_context = scatter_sum_torch(msg_st, tgt_index, n_tgt)
        deg_t = scatter_sum_torch(torch.ones_like(tgt_index, dtype=torch.float32), tgt_index, n_tgt)
        mean_context = mean_context / torch.clamp(deg_t[:, None], min=1.0)

        # 2. Real target-wise source attention context.
        attn_logits = []
        attn_values = []

        for start in range(0, num_edges, self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, num_edges)

            s_idx = src_index[start:end]
            t_idx = tgt_index[start:end]
            e_h = edge_h[start:end]

            score_in = torch.cat([tgt_h[t_idx], src_h[s_idx], e_h], dim=1)
            value_in = torch.cat([src_h[s_idx], e_h], dim=1)

            attn_logits.append(self.attn_score(score_in).squeeze(-1))
            attn_values.append(self.attn_value(value_in))

        attn_logits = torch.cat(attn_logits, dim=0)
        attn_values = torch.cat(attn_values, dim=0)

        attn = self.edge_softmax_by_target(attn_logits, tgt_index, n_tgt)
        weighted_values = attn_values * attn.to(attn_values.dtype)[:, None]
        attn_context = scatter_sum_torch(weighted_values, tgt_index, n_tgt)

        # 3. Residual-gated hybrid context.
        gate = torch.sigmoid(self.context_gate(torch.cat([tgt_h, mean_context, attn_context], dim=1)))
        gated_context = mean_context + self.residual_scale * gate * (attn_context - mean_context)

        tgt_h2 = self.tgt_update(torch.cat([tgt_h, gated_context], dim=1))

        # 4. Reverse target -> source pass.
        msg_ts = self.tgt_to_src_msg(torch.cat([tgt_h2[tgt_index], edge_h], dim=1))
        agg_s = scatter_sum_torch(msg_ts, src_index, n_src)
        deg_s = scatter_sum_torch(torch.ones_like(src_index, dtype=torch.float32), src_index, n_src)
        agg_s = agg_s / torch.clamp(deg_s[:, None], min=1.0)
        src_h2 = self.src_update(torch.cat([src_h, agg_s], dim=1))

        return self._decode_edges(src_h2, tgt_h2, edge_h, src_index, tgt_index)


class GateConditionedHybridAttentionGNNSinkhorn(HybridAttentionGNNSinkhorn):
    """
    v14-style pair-conditioned gate.

    The last pair_cond_dim columns of edge_attr are mesh-family conditioning
    features, e.g.

      src_mesh_is_RLL, src_mesh_is_CS, src_mesh_is_ICOD,
      tgt_mesh_is_RLL, tgt_mesh_is_CS, tgt_mesh_is_ICOD

    Unlike v13, these conditioning features do NOT enter the edge encoder,
    message MLPs, attention score, or attention value. They only enter the
    gate that mixes mean_context and attn_context.

    This tests the hypothesis:
      mesh family should control how much attention to trust,
      but should not globally perturb all edge representations.
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        edge_dim: int,
        hidden: int = 128,
        decoder_chunk_size: int = 10000,
        pair_cond_dim: int = 6,
    ):
        if edge_dim <= pair_cond_dim:
            raise ValueError(
                f"edge_dim={edge_dim} must be greater than pair_cond_dim={pair_cond_dim}"
            )

        self.full_edge_dim = int(edge_dim)
        self.pair_cond_dim = int(pair_cond_dim)
        self.base_edge_dim = int(edge_dim - pair_cond_dim)

        # Initialize the parent with only the physical/local edge features.
        super().__init__(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=self.base_edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

        self.context_gate = nn.Sequential(
            nn.Linear(hidden * 3 + self.pair_cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.tgt_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, src_node_attr, tgt_node_attr, edge_attr, src_index, tgt_index, n_src, n_tgt):
        base_edge_attr = edge_attr[:, :self.base_edge_dim]
        pair_cond_edge = edge_attr[:, self.base_edge_dim:]

        src_h = self.src_encoder(src_node_attr)
        tgt_h = self.tgt_encoder(tgt_node_attr)
        edge_h = self.edge_encoder(base_edge_attr)

        num_edges = edge_h.shape[0]

        # 1. Stable mean source -> target context.
        msg_st = self.src_to_tgt_msg(torch.cat([src_h[src_index], edge_h], dim=1))
        mean_context = scatter_sum_torch(msg_st, tgt_index, n_tgt)
        deg_t = scatter_sum_torch(torch.ones_like(tgt_index, dtype=torch.float32), tgt_index, n_tgt)
        mean_context = mean_context / torch.clamp(deg_t[:, None], min=1.0)

        # 2. Real target-wise source attention context, using only base edge features.
        attn_logits = []
        attn_values = []

        for start in range(0, num_edges, self.decoder_chunk_size):
            end = min(start + self.decoder_chunk_size, num_edges)

            s_idx = src_index[start:end]
            t_idx = tgt_index[start:end]
            e_h = edge_h[start:end]

            score_in = torch.cat([tgt_h[t_idx], src_h[s_idx], e_h], dim=1)
            value_in = torch.cat([src_h[s_idx], e_h], dim=1)

            attn_logits.append(self.attn_score(score_in).squeeze(-1))
            attn_values.append(self.attn_value(value_in))

        attn_logits = torch.cat(attn_logits, dim=0)
        attn_values = torch.cat(attn_values, dim=0)

        attn = self.edge_softmax_by_target(attn_logits, tgt_index, n_tgt)
        weighted_values = attn_values * attn.to(attn_values.dtype)[:, None]
        attn_context = scatter_sum_torch(weighted_values, tgt_index, n_tgt)

        # 3. Pair conditioning enters only the gate.
        pair_cond_sum = scatter_sum_torch(pair_cond_edge.to(mean_context.dtype), tgt_index, n_tgt)
        pair_cond_tgt = pair_cond_sum / torch.clamp(
            deg_t[:, None].to(pair_cond_sum.dtype),
            min=1.0,
        )

        gate_in = torch.cat([tgt_h, mean_context, attn_context, pair_cond_tgt], dim=1)
        gate = torch.sigmoid(self.context_gate(gate_in))

        gated_context = gate * attn_context + (1.0 - gate) * mean_context
        tgt_h2 = self.tgt_update(torch.cat([tgt_h, gated_context], dim=1))

        # 4. Reverse target -> source pass.
        msg_ts = self.tgt_to_src_msg(torch.cat([tgt_h2[tgt_index], edge_h], dim=1))
        agg_s = scatter_sum_torch(msg_ts, src_index, n_src)
        deg_s = scatter_sum_torch(torch.ones_like(src_index, dtype=torch.float32), src_index, n_src)
        agg_s = agg_s / torch.clamp(deg_s[:, None], min=1.0)
        src_h2 = self.src_update(torch.cat([src_h, agg_s], dim=1))

        return self._decode_edges(src_h2, tgt_h2, edge_h, src_index, tgt_index)

def build_model(
    architecture: str,
    src_dim: int,
    tgt_dim: int,
    edge_dim: int,
    hidden: int = 128,
    decoder_chunk_size: int = 10000,
) -> nn.Module:
    if architecture == "mean_gnn":
        return MeanGNNSinkhorn(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

    if architecture == "hybrid_attention":
        return HybridAttentionGNNSinkhorn(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

    if architecture == "gated_hybrid_attention":
        return GatedHybridAttentionGNNSinkhorn(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

    if architecture == "pair_conditioned_gated_hybrid_attention":
        return GatedHybridAttentionGNNSinkhorn(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
        )

    if architecture == "residual_gated_hybrid_attention":
        return ResidualGatedHybridAttentionGNNSinkhorn(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
            residual_scale=0.25,
        )

    if architecture == "gate_conditioned_hybrid_attention":
        return GateConditionedHybridAttentionGNNSinkhorn(
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            decoder_chunk_size=decoder_chunk_size,
            pair_cond_dim=6,
        )

    raise ValueError(f"Unknown architecture: {architecture}")
