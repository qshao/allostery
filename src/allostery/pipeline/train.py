from __future__ import annotations

import copy
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
from allostery.training.runtime import (
    BatchedRelationalSample,
    iter_batches,
    resolve_device,
    seed_everything,
    split_samples,
    stack_relational_batch,
)


@dataclass(frozen=True, slots=True)
class TrainResult:
    model: RelationalScoreModel
    num_samples: int
    last_loss: float
    best_epoch: int = 0
    best_validation_loss: float | None = None
    train_samples: int = 0
    validation_samples: int = 0
    batch_size: int = 1


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
            'trajectory did not yield any training windows '
            f'for window_size={window_size}, horizon_size={horizon_size}, stride={stride}'
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


def _training_batch(samples: list[TrainingSample], device: torch.device) -> BatchedRelationalSample:
    return stack_relational_batch(samples, device)


def _train_batch(
    model: RelationalScoreModel,
    batch: BatchedRelationalSample,
    optimizer: torch.optim.Optimizer,
    consistency_weight: float,
    previous_scores: Tensor | None,
) -> tuple[float, Tensor]:
    output = model(batch.residue_features, batch.pair_index, batch.pair_features)
    summary_term = future_summary_loss(output['target_pred'], batch.targets)
    consistency_terms: list[Tensor] = []
    if previous_scores is not None:
        consistency_terms.append(consistency_loss(previous_scores, output['scores'][0]))
    for index in range(1, output['scores'].shape[0]):
        consistency_terms.append(consistency_loss(output['scores'][index - 1], output['scores'][index]))
    if consistency_terms:
        consistency_term = consistency_weight * torch.stack(consistency_terms).mean()
    else:
        consistency_term = summary_term.new_zeros(())
    losses = TrainingLossBreakdown(
        future_summary=summary_term,
        consistency=consistency_term,
    )
    optimizer.zero_grad()
    losses.total.backward()
    optimizer.step()
    return float(losses.total.detach().item()), output['scores'][-1].detach()


def _evaluate_epoch(
    model: RelationalScoreModel,
    samples: list[TrainingSample],
    device: torch.device,
    consistency_weight: float,
    batch_size: int,
) -> float:
    if not samples:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_batches = 0
    previous_scores: Tensor | None = None
    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size):
            batch = _training_batch(batch_samples, device)
            output = model(batch.residue_features, batch.pair_index, batch.pair_features)
            summary_term = future_summary_loss(output['target_pred'], batch.targets)
            consistency_terms: list[Tensor] = []
            if previous_scores is not None:
                consistency_terms.append(consistency_loss(previous_scores, output['scores'][0]))
            for index in range(1, output['scores'].shape[0]):
                consistency_terms.append(consistency_loss(output['scores'][index - 1], output['scores'][index]))
            if consistency_terms:
                consistency_term = consistency_weight * torch.stack(consistency_terms).mean()
            else:
                consistency_term = summary_term.new_zeros(())
            losses = TrainingLossBreakdown(
                future_summary=summary_term,
                consistency=consistency_term,
            )
            total_loss += float(losses.total.detach().item())
            total_batches += 1
            previous_scores = output['scores'][-1].detach()
    model.train()
    return total_loss / float(total_batches)


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
    validation_fraction: float = 0.2,
    patience: int = 5,
    seed: int = 0,
    device: str = 'cpu',
    batch_size: int = 4,
) -> TrainResult:
    seed_everything(seed)
    torch_device = resolve_device(device)
    samples = _load_training_samples(
        pdb_path=pdb_path,
        window_size=window_size,
        horizon_size=horizon_size,
        stride=stride,
    )
    train_samples, validation_samples = split_samples(samples, validation_fraction, seed)
    if not train_samples:
        raise ValueError('training split did not yield any training samples')

    model = _build_model(
        samples=samples,
        hidden_dim=hidden_dim,
        residue_layers=residue_layers,
        pair_layers=pair_layers,
        dropout=dropout,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    last_loss = 0.0

    for epoch in range(epochs):
        model.train()
        previous_scores: Tensor | None = None
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = _training_batch(batch_samples, torch_device)
            last_loss, previous_scores = _train_batch(
                model=model,
                batch=batch,
                optimizer=optimizer,
                consistency_weight=consistency_weight,
                previous_scores=previous_scores,
            )

        if validation_samples:
            validation_loss = _evaluate_epoch(
                model=model,
                samples=validation_samples,
                device=torch_device,
                consistency_weight=consistency_weight,
                batch_size=batch_size,
            )
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    break

    if best_validation_loss is not None:
        model.load_state_dict(best_state)
    model = model.to('cpu')

    return TrainResult(
        model=model,
        num_samples=len(samples),
        last_loss=last_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        batch_size=batch_size,
    )


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
    validation_fraction: float = 0.2,
    patience: int = 5,
    seed: int = 0,
    device: str = 'cpu',
    batch_size: int = 4,
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
        validation_fraction=validation_fraction,
        patience=patience,
        seed=seed,
        device=device,
        batch_size=batch_size,
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
            metadata={
                'training': {
                    'seed': seed,
                    'device': device,
                    'batch_size': batch_size,
                    'validation_fraction': validation_fraction,
                    'patience': patience,
                    'train_samples': result.train_samples,
                    'validation_samples': result.validation_samples,
                    'best_epoch': result.best_epoch,
                    'best_validation_loss': result.best_validation_loss,
                    'last_loss': result.last_loss,
                }
            },
        )
    return result


__all__ = ['TrainResult', 'train_model', 'train_relational_model']
