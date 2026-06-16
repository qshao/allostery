from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from allostery.cri.data import CRISample, build_cri_training_samples
from allostery.io.checkpoint import save_checkpoint
from allostery.io.trajectory import load_trajectory
from allostery.models.cri import CRILatentInteractionModel
from allostery.training.cri_objectives import cri_loss
from allostery.training.runtime import (
    BatchedCRISample,
    iter_batches,
    resolve_device,
    seed_everything,
    split_samples,
    stack_cri_batch,
)


@dataclass(frozen=True, slots=True)
class CRITrainResult:
    model: CRILatentInteractionModel
    num_samples: int
    last_loss: float
    best_epoch: int = 0
    best_validation_loss: float | None = None
    train_samples: int = 0
    validation_samples: int = 0
    batch_size: int = 1


def _tensorize_sample(sample: CRISample, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(sample.state_features[None, ...], dtype=torch.float32, device=device),
        torch.as_tensor(sample.acceleration_targets[None, ...], dtype=torch.float32, device=device),
        torch.as_tensor(sample.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(sample.edge_distance, dtype=torch.float32, device=device),
    )


def _evaluate_epoch(
    model: CRILatentInteractionModel,
    samples: list[CRISample],
    device: torch.device,
    entropy_weight: float,
    no_edge_weight: float,
    batch_size: int,
) -> float:
    if not samples:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size):
            batch = stack_cri_batch(batch_samples, device)
            output = model(batch.state_features, batch.edge_index, batch.edge_distance, batch.edge_mask)
            losses = cri_loss(
                output,
                batch.acceleration_targets,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
            )
            total_loss += float(losses.total.detach().item())
            total_batches += 1
    model.train()
    return total_loss / float(total_batches)


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
    min_sequence_separation: int = 0,
    preprocess: str = 'none',
    validation_fraction: float = 0.2,
    patience: int = 5,
    seed: int = 0,
    device: str = 'cpu',
    batch_size: int = 4,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    topology_path: str | Path | None = None,
) -> CRITrainResult:
    seed_everything(seed)
    torch_device = resolve_device(device)
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    samples = build_cri_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        distance_cutoff=distance_cutoff,
        max_neighbors=max_neighbors,
        min_sequence_separation=min_sequence_separation,
        preprocess=preprocess,
    )
    if not samples:
        raise ValueError('trajectory did not yield any CRI training windows')

    train_samples, validation_samples = split_samples(samples, validation_fraction, seed)
    if not train_samples:
        raise ValueError('training split did not yield any CRI training samples')

    state_dim = int(samples[0].state_features.shape[-1])
    model = CRILatentInteractionModel(state_dim=state_dim, hidden_dim=hidden_dim, edge_types=edge_types, dropout=dropout).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    last_loss = 0.0

    width = len(str(epochs))
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = stack_cri_batch(batch_samples, torch_device)
            output = model(batch.state_features, batch.edge_index, batch.edge_distance, batch.edge_mask)
            losses = cri_loss(
                output,
                batch.acceleration_targets,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
            )
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().item())
            epoch_loss_sum += last_loss
            epoch_batch_count += 1

        train_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        if validation_samples:
            validation_loss = _evaluate_epoch(
                model=model,
                samples=validation_samples,
                device=torch_device,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
                batch_size=batch_size,
            )
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch + 1}", flush=True)
                    break
            if verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
        elif verbose:
            print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)

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
            residue_layers=1,
            pair_layers=edge_types,
            dropout=dropout,
            model_family='cri',
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

    return CRITrainResult(
        model=model,
        num_samples=len(samples),
        last_loss=last_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        batch_size=batch_size,
    )


__all__ = ['CRITrainResult', 'train_cri_model']
