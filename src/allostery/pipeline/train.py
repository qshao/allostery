from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from allostery.data import TrainingSample, build_training_samples
from allostery.io.checkpoint import save_checkpoint
from allostery.io.pdb import load_multimodel_pdb
from allostery.models.relational import RelationalScoreModel
from allostery.training.objectives import TrainingLossBreakdown, consistency_loss, future_summary_loss


@dataclass(frozen=True, slots=True)
class TrainResult:
    model: RelationalScoreModel
    num_samples: int
    last_loss: float


@dataclass(frozen=True, slots=True)
class _TensorSample:
    residue_features: Tensor
    pair_index: Tensor
    pair_features: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class _FeatureDimensions:
    residue_dim: int
    pair_dim: int
    target_dim: int


def _load_training_samples(
    pdb_path: str | Path,
    window_size: int,
    horizon_size: int,
    stride: int,
) -> list[TrainingSample]:
    trajectory = load_multimodel_pdb(Path(pdb_path))
    samples = build_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        horizon_size=horizon_size,
        stride=stride,
    )
    if not samples:
        raise ValueError(
            "trajectory did not yield any training windows "
            f"for window_size={window_size}, horizon_size={horizon_size}, stride={stride}"
        )
    return samples


def _sample_dimensions(samples: list[TrainingSample]) -> _FeatureDimensions:
    first_sample = samples[0]
    return _FeatureDimensions(
        residue_dim=int(first_sample.residue_features.shape[-1]),
        pair_dim=int(first_sample.pair_features.shape[-1]),
        target_dim=int(first_sample.targets.shape[-1]),
    )


def _build_model(
    samples: list[TrainingSample],
    hidden_dim: int,
    residue_layers: int,
    pair_layers: int,
    dropout: float,
) -> RelationalScoreModel:
    dimensions = _sample_dimensions(samples)
    return RelationalScoreModel(
        residue_dim=dimensions.residue_dim,
        pair_dim=dimensions.pair_dim,
        hidden_dim=hidden_dim,
        target_dim=dimensions.target_dim,
        residue_layers=residue_layers,
        pair_layers=pair_layers,
        dropout=dropout,
    )


def _tensorize_sample(sample: TrainingSample) -> _TensorSample:
    return _TensorSample(
        residue_features=torch.as_tensor(sample.residue_features[None, ...], dtype=torch.float32),
        pair_index=torch.as_tensor(sample.pair_index, dtype=torch.int64),
        pair_features=torch.as_tensor(sample.pair_features[None, ...], dtype=torch.float32),
        targets=torch.as_tensor(sample.targets[None, ...], dtype=torch.float32),
    )


def train_relational_model(
    pdb_path: str | Path,
    window_size: int = 8,
    horizon_size: int = 4,
    stride: int = 2,
    hidden_dim: int = 32,
    residue_layers: int = 2,
    pair_layers: int = 2,
    dropout: float = 0.0,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    consistency_weight: float = 0.1,
) -> TrainResult:
    samples = _load_training_samples(
        pdb_path=pdb_path,
        window_size=window_size,
        horizon_size=horizon_size,
        stride=stride,
    )
    model = _build_model(
        samples=samples,
        hidden_dim=hidden_dim,
        residue_layers=residue_layers,
        pair_layers=pair_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    last_loss = 0.0
    for _ in range(epochs):
        previous_scores: Tensor | None = None
        for sample in samples:
            batch = _tensorize_sample(sample)
            output = model(batch.residue_features, batch.pair_index, batch.pair_features)
            summary_term = future_summary_loss(output["target_pred"], batch.targets)
            if previous_scores is None:
                consistency_term = summary_term.new_zeros(())
            else:
                consistency_term = consistency_weight * consistency_loss(
                    previous_scores,
                    output["scores"],
                )
            losses = TrainingLossBreakdown(
                future_summary=summary_term,
                consistency=consistency_term,
            )
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            previous_scores = output["scores"].detach()
            last_loss = float(losses.total.detach().item())

    return TrainResult(model=model, num_samples=len(samples), last_loss=last_loss)


def train_model(
    pdb_path: str | Path,
    window_size: int = 8,
    horizon_size: int = 4,
    stride: int = 2,
    hidden_dim: int = 32,
    residue_layers: int = 2,
    pair_layers: int = 2,
    dropout: float = 0.0,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    consistency_weight: float = 0.1,
    checkpoint_path: str | Path | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> TrainResult:
    result = train_relational_model(
        pdb_path=pdb_path,
        window_size=window_size,
        horizon_size=horizon_size,
        stride=stride,
        hidden_dim=hidden_dim,
        residue_layers=residue_layers,
        pair_layers=pair_layers,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        consistency_weight=consistency_weight,
    )
    if checkpoint_path is not None:
        dimensions = _sample_dimensions(
            _load_training_samples(
                pdb_path=pdb_path,
                window_size=window_size,
                horizon_size=horizon_size,
                stride=stride,
            )
        )
        save_checkpoint(
            path=checkpoint_path,
            model=result.model,
            config_snapshot=config_snapshot or {},
            residue_dim=dimensions.residue_dim,
            pair_dim=dimensions.pair_dim,
            hidden_dim=hidden_dim,
            target_dim=dimensions.target_dim,
            residue_layers=residue_layers,
            pair_layers=pair_layers,
            dropout=dropout,
        )
    return result


__all__ = ["TrainResult", "train_model", "train_relational_model"]
