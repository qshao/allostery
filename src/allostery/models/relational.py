from __future__ import annotations

import torch
from torch import Tensor, nn

from allostery.models.encoders import PairEncoder, ResidueEncoder


class RelationalScoreModel(nn.Module):
    def __init__(
        self,
        residue_dim: int,
        pair_dim: int,
        hidden_dim: int,
        target_dim: int = 3,
        residue_layers: int = 2,
        pair_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.residue_encoder = ResidueEncoder(
            input_dim=residue_dim,
            hidden_dim=hidden_dim,
            num_layers=residue_layers,
            dropout=dropout,
        )
        self.pair_encoder = PairEncoder(
            input_dim=pair_dim,
            hidden_dim=hidden_dim,
            num_layers=pair_layers,
            dropout=dropout,
        )
        combined_dim = hidden_dim * 4
        self.score_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.target_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(
        self,
        residue_features: Tensor,
        pair_index: Tensor,
        pair_features: Tensor,
    ) -> dict[str, Tensor]:
        residue_embedding = self.residue_encoder(residue_features)
        pair_embedding = self.pair_encoder(pair_features)

        left_embedding = residue_embedding[:, pair_index[:, 0], :]
        right_embedding = residue_embedding[:, pair_index[:, 1], :]
        symmetric_pair_embedding = self._combine_pair_embeddings(
            left_embedding=left_embedding,
            right_embedding=right_embedding,
            pair_embedding=pair_embedding,
        )
        scores = self.score_head(symmetric_pair_embedding).squeeze(-1)
        target_prediction = self.target_head(symmetric_pair_embedding)
        return {"scores": scores, "target_pred": target_prediction}

    @staticmethod
    def _combine_pair_embeddings(
        left_embedding: Tensor,
        right_embedding: Tensor,
        pair_embedding: Tensor,
    ) -> Tensor:
        return torch.cat(
            (
                left_embedding + right_embedding,
                torch.abs(left_embedding - right_embedding),
                left_embedding * right_embedding,
                pair_embedding,
            ),
            dim=-1,
        )


__all__ = ["RelationalScoreModel"]
