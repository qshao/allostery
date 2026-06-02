from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from allostery.cri.data import CRISample, build_cri_training_samples
from allostery.io.checkpoint import save_checkpoint
from allostery.io.pdb import load_multimodel_pdb
from allostery.models.cri import CRILatentInteractionModel
from allostery.training.cri_objectives import cri_loss


@dataclass(frozen=True, slots=True)
class CRITrainResult:
    model: CRILatentInteractionModel
    num_samples: int
    last_loss: float


def _tensorize_sample(sample: CRISample) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(sample.state_features[None, ...], dtype=torch.float32),
        torch.as_tensor(sample.acceleration_targets[None, ...], dtype=torch.float32),
        torch.as_tensor(sample.edge_index, dtype=torch.long),
        torch.as_tensor(sample.edge_distance, dtype=torch.float32),
    )


def train_cri_model(
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
    edge_types: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    entropy_weight: float,
    no_edge_weight: float,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> CRITrainResult:
    trajectory = load_multimodel_pdb(Path(pdb_path))
    samples = build_cri_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        distance_cutoff=distance_cutoff,
        max_neighbors=max_neighbors,
    )
    if not samples:
        raise ValueError("trajectory did not yield any CRI training windows")

    state_dim = int(samples[0].state_features.shape[-1])
    model = CRILatentInteractionModel(state_dim=state_dim, hidden_dim=hidden_dim, edge_types=edge_types, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        for sample in samples:
            state_features, targets, edge_index, edge_distance = _tensorize_sample(sample)
            output = model(state_features, edge_index, edge_distance)
            losses = cri_loss(
                output,
                targets,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
            )
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().item())

    if checkpoint_path is not None:
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            config_snapshot=config_snapshot or {},
            residue_dim=state_dim,
            pair_dim=1,
            hidden_dim=hidden_dim,
            target_dim=3,
            residue_layers=1,
            pair_layers=edge_types,
            dropout=dropout,
            model_family="cri",
        )

    return CRITrainResult(model=model, num_samples=len(samples), last_loss=last_loss)


__all__ = ["CRITrainResult", "train_cri_model"]
