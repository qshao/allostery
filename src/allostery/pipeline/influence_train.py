from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from allostery.influence.data import InfluenceSample, build_influence_samples
from allostery.io.checkpoint import save_checkpoint
from allostery.io.pdb import load_multimodel_pdb
from allostery.models.influence import AllostericInfluenceModel
from allostery.training.influence_objectives import influence_loss
from allostery.training.runtime import (
    iter_batches,
    resolve_device,
    seed_everything,
    split_samples,
    stack_influence_batch,
)


@dataclass(frozen=True, slots=True)
class InfluenceTrainResult:
    model: AllostericInfluenceModel
    num_samples: int
    last_loss: float
    best_epoch: int = 0
    best_validation_loss: float | None = None
    train_samples: int = 0
    validation_samples: int = 0
    batch_size: int = 1


def _evaluate_epoch(
    model: AllostericInfluenceModel,
    samples: list[InfluenceSample],
    device: torch.device,
    sparsity_weight: float,
    batch_size: int,
) -> float:
    if not samples:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size):
            batch = stack_influence_batch(batch_samples, device)
            output = model(batch.state_features)
            losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
            total_loss += float(losses.total.detach().item())
            total_batches += 1
    model.train()
    return total_loss / float(total_batches)


def train_influence_model(
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float,
    hidden_dim: int,
    num_encoder_layers: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    sparsity_weight: float,
    preprocess: str = 'none',
    validation_fraction: float = 0.2,
    patience: int = 5,
    seed: int = 0,
    device: str = 'cpu',
    batch_size: int = 4,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> InfluenceTrainResult:
    seed_everything(seed)
    torch_device = resolve_device(device)
    trajectory = load_multimodel_pdb(Path(pdb_path))
    samples = build_influence_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        preprocess=preprocess,
    )
    if not samples:
        raise ValueError('trajectory did not yield any influence training windows')

    train_samples, validation_samples = split_samples(samples, validation_fraction, seed)
    if not train_samples:
        raise ValueError('training split did not yield any influence training samples')

    state_dim = int(samples[0].state_features.shape[-1])
    model = AllostericInfluenceModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
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
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = stack_influence_batch(batch_samples, torch_device)
            output = model(batch.state_features)
            losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().item())

        if validation_samples:
            validation_loss = _evaluate_epoch(
                model=model,
                samples=validation_samples,
                device=torch_device,
                sparsity_weight=sparsity_weight,
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

    if checkpoint_path is not None:
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            config_snapshot=config_snapshot or {},
            residue_dim=state_dim,
            pair_dim=1,
            hidden_dim=hidden_dim,
            target_dim=3,
            residue_layers=num_encoder_layers,
            pair_layers=1,
            dropout=dropout,
            model_family='influence',
            metadata={
                'training': {
                    'seed': seed,
                    'device': device,
                    'batch_size': batch_size,
                    'validation_fraction': validation_fraction,
                    'patience': patience,
                    'train_samples': len(train_samples),
                    'validation_samples': len(validation_samples),
                    'best_epoch': best_epoch,
                    'best_validation_loss': best_validation_loss,
                    'last_loss': last_loss,
                }
            },
        )

    return InfluenceTrainResult(
        model=model,
        num_samples=len(samples),
        last_loss=last_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        batch_size=batch_size,
    )


__all__ = ['InfluenceTrainResult', 'train_influence_model']
