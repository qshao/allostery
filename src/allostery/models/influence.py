from __future__ import annotations

import torch
from torch import Tensor, nn

from allostery.features.amino_acid import NUM_AMINO_ACID_TYPES


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
    accelerations through multi-head attention over all pairs.

    Key design choices:
    - Temporal Conv1d encoder: Q/K are derived from dynamic correlation patterns
      over the full window (not just the temporal mean), so topology reflects
      actual dynamic coupling rather than mean structural proximity.
    - Multi-head attention: each head independently discovers an allosteric pathway,
      avoiding the single-pathway bottleneck of single-head attention.
    - Sigmoid (not softmax): independent per-pair scores with no zero-sum row budget,
      allowing multiple residues to simultaneously strongly influence a given residue.
    - Residue identity embedding: amino acid type conditions Q/K so the model can
      distinguish PRO (rigid) from GLY (flexible) etc.

    influence_matrix[j, i] = mean over heads of sigmoid(Q_j · K_i / sqrt(d)):
    the directed allosteric influence of i on j.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_encoder_layers: int = 2,
        dropout: float = 0.0,
        residue_chunk_size: int | None = None,
        min_sequence_separation: int = 1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if num_encoder_layers <= 0:
            raise ValueError('num_encoder_layers must be greater than zero')
        if residue_chunk_size is not None and residue_chunk_size <= 0:
            raise ValueError('residue_chunk_size must be greater than zero')
        if min_sequence_separation < 1:
            raise ValueError(
                'min_sequence_separation must be at least 1 (diagonal must always be masked)'
            )
        if num_heads <= 0:
            raise ValueError('num_heads must be greater than zero')
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})'
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.residue_chunk_size = residue_chunk_size
        self.min_sequence_separation = min_sequence_separation

        # Temporal conv encoder: captures dynamic correlation patterns over the window.
        # Conv1d(state_dim → hidden_dim, kernel=3, padding=1) handles any T ≥ 1.
        # Mean-pool over T → fixed-size per-residue dynamic representation.
        conv_layers: list[nn.Module] = [
            nn.Conv1d(state_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_encoder_layers - 1):
            conv_layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            ])
        if dropout > 0.0:
            conv_layers.append(nn.Dropout(dropout))
        self.temporal_encoder = nn.Sequential(*conv_layers)

        # Residue identity embedding: conditions Q/K on amino acid type (20 + 1 unknown).
        self.residue_embedding = nn.Embedding(NUM_AMINO_ACID_TYPES, hidden_dim)

        # Multi-head Q and K for attention topology.
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # V: time-varying message content from senders.
        self.value_proj = nn.Linear(state_dim, hidden_dim)

        # Baseline: each residue's self-driven acceleration.
        self.baseline_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 3),
        )

        # Decode aggregated multi-head influence messages → acceleration delta.
        self.decode_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        state_features: Tensor,
        residue_type_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            state_features:       [batch, time, N, state_dim]
            residue_type_indices: [N] long tensor of AA indices (0–20); optional

        Returns:
            acceleration:     [batch, time, N, 3]
            influence_matrix: [batch, N, N] — mean over heads;
                              influence_matrix[j, i] = influence of i on j
        """
        if state_features.ndim != 4:
            raise ValueError('state_features must have shape (batch, time, N, state_dim)')
        B, T, N, D = state_features.shape
        H, Hd = self.num_heads, self.head_dim

        if N < 2:
            baseline = self.baseline_net(state_features)
            influence_matrix = torch.zeros(B, N, N, device=state_features.device)
            return {'acceleration': baseline, 'influence_matrix': influence_matrix}

        # --- Temporal encoding for Q/K ---
        # [B, T, N, D] → [B*N, D, T] for Conv1d, then mean-pool over T → [B, N, hidden_dim]
        x = state_features.permute(0, 2, 3, 1).reshape(B * N, D, T)
        x = self.temporal_encoder(x)                        # [B*N, hidden_dim, T]
        x = x.mean(dim=-1).reshape(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        # Condition on residue identity (broadcast over batch dimension).
        if residue_type_indices is not None:
            x = x + self.residue_embedding(residue_type_indices).unsqueeze(0)

        # --- Multi-head sigmoid attention ---
        # Q/K: [B, N, hidden_dim] → [B, H, N, Hd]
        Q = self.query_proj(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3)
        K = self.key_proj(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3)

        scale = float(Hd) ** -0.5
        # attn_logits[b, h, j, i] = Q_j · K_i / sqrt(Hd); j=receiver, i=sender
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) * scale  # [B, H, N, N]

        # Mask within min_sequence_separation. sigmoid(−∞) = 0 exactly.
        if self.min_sequence_separation >= N:
            raise ValueError(
                f'min_sequence_separation={self.min_sequence_separation} leaves no valid pairs '
                f'for a protein of {N} residues. '
                f'Use a value less than {N}.'
            )
        indices = torch.arange(N, device=state_features.device)
        sep_mask = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs() < self.min_sequence_separation
        attn_logits = attn_logits.masked_fill(sep_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        head_attentions = torch.sigmoid(attn_logits)  # [B, H, N, N]; independent per pair

        # influence_matrix = mean over heads → [B, N, N]
        influence_matrix = head_attentions.mean(dim=1)

        # --- Time-varying value messages (per head) ---
        V = self.value_proj(state_features).reshape(B, T, N, H, Hd)  # [B, T, N, H, Hd]

        # Aggregate: aggregated[b,t,j,h,d] = Σ_i head_attentions[b,h,j,i] * V[b,t,i,h,d]
        chunk = self.residue_chunk_size
        if chunk is None or chunk >= N:
            aggregated = torch.einsum('bhjm,btmhd->btjhd', head_attentions, V)
        else:
            parts = []
            for start in range(0, N, chunk):
                stop = min(start + chunk, N)
                rows = head_attentions[:, :, start:stop, :]      # [B, H, chunk, N]
                parts.append(torch.einsum('bhjm,btmhd->btjhd', rows, V))
            aggregated = torch.cat(parts, dim=2)

        aggregated = aggregated.reshape(B, T, N, self.hidden_dim)  # [B, T, N, hidden_dim]

        baseline = self.baseline_net(state_features)
        acceleration = baseline + self.decode_net(aggregated)

        return {
            'acceleration': acceleration,
            'influence_matrix': influence_matrix,
        }


__all__ = ['AllostericInfluenceModel']
