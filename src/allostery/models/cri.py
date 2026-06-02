from __future__ import annotations

import torch
from torch import Tensor, nn


class CRILatentInteractionModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, edge_types: int, dropout: float = 0.0) -> None:
        super().__init__()
        if edge_types < 2:
            raise ValueError("edge_types must be at least 2")
        self.edge_types = edge_types
        self.message_dim = 3
        edge_input_dim = (2 * state_dim) + 1
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

    def forward(self, state_features: Tensor, edge_index: Tensor, edge_distance: Tensor) -> dict[str, Tensor]:
        if state_features.ndim != 4:
            raise ValueError("state_features must have shape (batch, time, residues, state_dim)")
        batch_size, num_steps, num_residues, _state_dim = state_features.shape
        if edge_index.numel() == 0:
            empty_prob = state_features.new_empty((batch_size, 0, self.edge_types))
            empty_score = state_features.new_empty((batch_size, 0))
            return {
                "acceleration": state_features.new_zeros((batch_size, num_steps, num_residues, self.message_dim)),
                "edge_type_prob": empty_prob,
                "edge_score": empty_score,
            }

        senders = edge_index[:, 0]
        receivers = edge_index[:, 1]
        sender_state = state_features[:, :, senders, :]
        receiver_state = state_features[:, :, receivers, :]
        pair_state = torch.cat((sender_state, receiver_state), dim=-1)
        pair_summary = pair_state.mean(dim=1)
        distance_feature = edge_distance.to(state_features.device, dtype=state_features.dtype)[None, :, None].expand(
            batch_size,
            -1,
            -1,
        )
        classifier_input = torch.cat((pair_summary, distance_feature), dim=-1)
        edge_type_prob = torch.softmax(self.edge_classifier(classifier_input), dim=-1)

        type_messages = torch.stack([decoder(pair_state) for decoder in self.edge_decoders], dim=-2)
        weighted_messages = torch.sum(type_messages * edge_type_prob[:, None, :, :, None], dim=-2)
        acceleration = state_features.new_zeros((batch_size, num_steps, num_residues, self.message_dim))
        for edge_id, receiver in enumerate(receivers.tolist()):
            acceleration[:, :, receiver, :] = acceleration[:, :, receiver, :] + weighted_messages[:, :, edge_id, :]

        edge_score = 1.0 - edge_type_prob[:, :, 0]
        return {
            "acceleration": acceleration,
            "edge_type_prob": edge_type_prob,
            "edge_score": edge_score,
        }


__all__ = ["CRILatentInteractionModel"]
