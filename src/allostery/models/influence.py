from __future__ import annotations

import torch
from torch import Tensor, nn


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0.0 and i < num_layers - 1:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    return nn.Sequential(*layers)


class AllostericInfluenceModel(nn.Module):
    """
    Learns a directed residue-residue influence matrix by predicting per-residue
    accelerations through attention-weighted message passing over all pairs.

    influence_matrix[j, i] encodes how strongly residue i's motion drives
    residue j's acceleration — the directed allosteric influence of i on j.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_encoder_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_encoder_layers <= 0:
            raise ValueError('num_encoder_layers must be greater than zero')
        self.hidden_dim = hidden_dim
        # Encode time-averaged state to compute stable Q and K
        self.encoder = _build_mlp(state_dim, hidden_dim, num_encoder_layers, dropout)
        # Q[j]: what kind of influence does j receive?
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # K[i]: what kind of influence does i send?
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # V[i,t]: time-varying message content from sender i
        self.value_proj = nn.Linear(state_dim, hidden_dim)
        # Baseline: each residue's self-driven acceleration
        self.baseline_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 3),
        )
        # Decode aggregated influence messages to acceleration delta
        self.decode_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, state_features: Tensor) -> dict[str, Tensor]:
        """
        Args:
            state_features: [batch, time, N, state_dim]  (positions + velocities)

        Returns:
            acceleration:     [batch, time, N, 3]
            influence_matrix: [batch, N, N]  — influence_matrix[j, i] = influence of i on j
        """
        if state_features.ndim != 4:
            raise ValueError('state_features must have shape (batch, time, N, state_dim)')
        batch_size, num_steps, num_residues, _ = state_features.shape

        # Temporal mean state used to compute influence topology
        mean_state = state_features.mean(dim=1)  # [batch, N, state_dim]
        encoded = self.encoder(mean_state)         # [batch, N, hidden_dim]

        Q = self.query_proj(encoded)               # [batch, N, hidden_dim]
        K = self.key_proj(encoded)                 # [batch, N, hidden_dim]

        # Scaled dot-product: attn_logits[j, i] = Q_j · K_i / sqrt(d)
        scale = float(self.hidden_dim) ** -0.5
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * scale  # [batch, N, N]

        # Exclude self-influence so each residue only receives cross-residue signals
        diag_mask = torch.eye(num_residues, dtype=torch.bool, device=state_features.device)
        attn_logits = attn_logits.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

        # influence_matrix[b, j, i] = softmax over i → directed allosteric influence
        influence_matrix = torch.softmax(attn_logits, dim=-1)  # [batch, N, N]

        # Time-varying sender messages
        V = self.value_proj(state_features)  # [batch, time, N, hidden_dim]

        # Aggregate: aggregated[b,t,j,:] = sum_i influence[b,j,i] * V[b,t,i,:]
        aggregated = torch.matmul(influence_matrix.unsqueeze(1), V)  # [batch, time, N, hidden_dim]

        baseline = self.baseline_net(state_features)          # [batch, time, N, 3]
        acceleration = baseline + self.decode_net(aggregated)  # [batch, time, N, 3]

        return {
            'acceleration': acceleration,
            'influence_matrix': influence_matrix,
        }


__all__ = ['AllostericInfluenceModel']
