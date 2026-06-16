from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn

from allostery.data import build_training_samples
from allostery.io.checkpoint import ModelCheckpoint, load_checkpoint
from allostery.io.pdb import ResidueRecord
from allostery.io.trajectory import load_trajectory
from allostery.models.cri import CRILatentInteractionModel
from allostery.models.influence import AllostericInfluenceModel
from allostery.models.relational import RelationalScoreModel


class ResidueIdentifier(TypedDict):
    index: int
    chain_id: str
    residue_number: int
    name: str


class PairScore(TypedDict):
    residue_i: ResidueIdentifier
    residue_j: ResidueIdentifier
    score: float



def _tensorize_window(
    residue_features: object,
    pair_index: object,
    pair_features: object,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(residue_features, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(pair_index, dtype=torch.int64),
        torch.as_tensor(pair_features, dtype=torch.float32).unsqueeze(0),
    )


def _residue_identifier(residue: ResidueRecord) -> ResidueIdentifier:
    return {
        "index": residue.index,
        "chain_id": residue.chain_id,
        "residue_number": residue.residue_number,
        "name": residue.name,
    }


def build_scoring_model(checkpoint: ModelCheckpoint) -> nn.Module:
    if checkpoint.model_family == 'cri':
        model = CRILatentInteractionModel(
            state_dim=checkpoint.residue_dim,
            hidden_dim=checkpoint.hidden_dim,
            edge_types=checkpoint.pair_layers,
            dropout=checkpoint.dropout,
        )
    elif checkpoint.model_family == 'influence':
        residue_chunk_size = checkpoint.metadata.get('training', {}).get('residue_chunk_size')
        model = AllostericInfluenceModel(
            state_dim=checkpoint.residue_dim,
            hidden_dim=checkpoint.hidden_dim,
            num_encoder_layers=checkpoint.residue_layers,
            dropout=checkpoint.dropout,
            min_sequence_separation=checkpoint.min_sequence_separation,
            residue_chunk_size=residue_chunk_size,
        )
    else:
        model = RelationalScoreModel(
            residue_dim=checkpoint.residue_dim,
            pair_dim=checkpoint.pair_dim,
            hidden_dim=checkpoint.hidden_dim,
            target_dim=checkpoint.target_dim,
            residue_layers=checkpoint.residue_layers,
            pair_layers=checkpoint.pair_layers,
            dropout=checkpoint.dropout,
        )
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    return model


def load_scoring_model(checkpoint_path: str | Path) -> nn.Module:
    return build_scoring_model(load_checkpoint(checkpoint_path))


def score_trajectory(
    model: RelationalScoreModel,
    pdb_path: str | Path,
    window_size: int = 8,
    horizon_size: int = 4,
    stride: int = 2,
    topology_path: str | Path | None = None,
) -> list[PairScore]:
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    samples = build_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        horizon_size=horizon_size,
        stride=stride,
    )
    if not samples:
        raise ValueError(
            "trajectory did not yield any scoring windows "
            f"for window_size={window_size}, horizon_size={horizon_size}, stride={stride}"
        )
    if isinstance(model, CRILatentInteractionModel):
        raise ValueError("score_trajectory requires a relational checkpoint; use score_cri_trajectory for CRI models")

    model.eval()
    with torch.no_grad():
        window_scores = []
        for sample in samples:
            residue_features, pair_index, pair_features = _tensorize_window(
                residue_features=sample.residue_features,
                pair_index=sample.pair_index,
                pair_features=sample.pair_features,
            )
            output = model(residue_features, pair_index, pair_features)
            window_scores.append(output["scores"].squeeze(0))

    averaged_scores = torch.stack(window_scores, dim=0).mean(dim=0)
    ranked_scores = [
        {
            "residue_i": _residue_identifier(trajectory.residues[int(left_index)]),
            "residue_j": _residue_identifier(trajectory.residues[int(right_index)]),
            "score": float(score.item()),
        }
        for (left_index, right_index), score in zip(samples[0].pair_index, averaged_scores, strict=True)
    ]
    ranked_scores.sort(key=lambda item: item["score"], reverse=True)
    return ranked_scores


__all__ = ["PairScore", "ResidueIdentifier", "build_scoring_model", "load_scoring_model", "score_trajectory"]
