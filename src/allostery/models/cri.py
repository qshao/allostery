from __future__ import annotations

import torch
from torch import Tensor, nn


class CRILatentInteractionModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, edge_types: int, dropout: float = 0.0) -> None:
        super().__init__()
        if edge_types < 2:
            raise ValueError('edge_types must be at least 2')
        self.edge_types = edge_types
        self.message_dim = 3
        edge_input_dim = (2 * state_dim) + 1
        self.baseline_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, self.message_dim),
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, edge_types),
        )
        self.edge_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * state_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, self.message_dim),
                )
                for _ in range(edge_types)
            ]
        )

    def forward(
        self,
        state_features: Tensor,
        edge_index: Tensor,
        edge_distance: Tensor,
        edge_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if state_features.ndim != 4:
            raise ValueError('state_features must have shape (batch, time, residues, state_dim)')
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0)
            edge_distance = edge_distance.unsqueeze(0)
            if edge_mask is None:
                edge_mask = torch.ones((edge_index.shape[0], edge_index.shape[1]), dtype=torch.bool, device=state_features.device)
            else:
                edge_mask = edge_mask.unsqueeze(0) if edge_mask.ndim == 1 else edge_mask
        elif edge_index.ndim != 3:
            raise ValueError('edge_index must have shape (edges, 2) or (batch, edges, 2)')

        batch_size, num_steps, num_residues, _state_dim = state_features.shape
        if edge_index.shape[0] == 1 and batch_size > 1:
            edge_index = edge_index.expand(batch_size, -1, -1)
            edge_distance = edge_distance.expand(batch_size, -1)
            edge_mask = edge_mask.expand(batch_size, -1) if edge_mask is not None else None
        elif edge_index.shape[0] != batch_size:
            raise ValueError('edge_index batch dimension must match state_features')
        if edge_distance.shape[:2] != edge_index.shape[:2]:
            raise ValueError('edge_distance must match edge_index shape')
        if edge_mask is None:
            edge_mask = edge_index[..., 0] >= 0
        elif edge_mask.ndim == 1:
            edge_mask = edge_mask.unsqueeze(0)
        if edge_mask.shape != edge_index.shape[:2]:
            raise ValueError('edge_mask must match the leading edge dimensions')

        baseline = self.baseline_head(state_features)
        edge_count = edge_index.shape[1]
        if edge_count == 0:
            empty_prob = state_features.new_empty((batch_size, 0, self.edge_types))
            empty_score = state_features.new_empty((batch_size, 0))
            return {
                'acceleration': baseline,
                'edge_type_prob': empty_prob,
                'edge_score': empty_score,
                'edge_mask': edge_mask.to(dtype=state_features.dtype),
            }

        safe_edge_index = edge_index.clamp_min(0)
        senders = safe_edge_index[..., 0]
        receivers = safe_edge_index[..., 1]
        sender_index = senders[:, None, :, None].expand(batch_size, num_steps, edge_count, state_features.shape[-1])
        receiver_index = receivers[:, None, :, None].expand(batch_size, num_steps, edge_count, self.message_dim)
        sender_state = state_features.gather(2, sender_index)
        receiver_state = state_features.gather(2, receivers[:, None, :, None].expand(batch_size, num_steps, edge_count, state_features.shape[-1]))
        pair_state = torch.cat((sender_state, receiver_state), dim=-1)
        pair_summary = pair_state.mean(dim=1)
        distance_feature = edge_distance.to(state_features.device, dtype=state_features.dtype)[..., None]
        classifier_input = torch.cat((pair_summary, distance_feature), dim=-1)
        edge_type_prob = torch.softmax(self.edge_classifier(classifier_input), dim=-1)

        type_messages = torch.stack([decoder(pair_state) for decoder in self.edge_decoders], dim=-2)
        weighted_messages = torch.sum(type_messages * edge_type_prob[:, None, :, :, None], dim=-2)
        edge_mask_float = edge_mask.to(dtype=state_features.dtype)
        weighted_messages = weighted_messages * edge_mask_float[:, None, :, None]
        acceleration = baseline.clone()
        acceleration.scatter_add_(2, receiver_index, weighted_messages)

        edge_score = (1.0 - edge_type_prob[:, :, 0]) * edge_mask_float
        return {
            'acceleration': acceleration,
            'edge_type_prob': edge_type_prob,
            'edge_score': edge_score,
            'edge_mask': edge_mask_float,
        }


__all__ = ['CRILatentInteractionModel']
