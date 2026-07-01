from __future__ import annotations

import copy
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from allostery.features.amino_acid import aa_name_to_idx
from allostery.influence.data import InfluenceSample, build_influence_samples
from allostery.io.checkpoint import save_checkpoint
from allostery.io.trajectory import load_trajectory
from allostery.models.influence import AllostericInfluenceModel
from allostery.training.influence_objectives import InfluenceLossBreakdown, influence_loss
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
    val_groups: list[list[InfluenceSample]],
    device: torch.device,
    sparsity_weight: float,
    batch_size: int,
    residue_type_indices: torch.Tensor | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for group in val_groups:
            for batch_samples in iter_batches(group, batch_size):
                batch = stack_influence_batch(batch_samples, device)
                output = model(batch.state_features, residue_type_indices=residue_type_indices)
                losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
                total_loss += float(losses.total.detach().item())
                total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


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
    min_sequence_separation: int = 1,
    preprocess: str = 'none',
    validation_fraction: float = 0.2,
    patience: int = 5,
    seed: int = 0,
    device: str = 'cpu',
    batch_size: int = 4,
    grad_clip_norm: float | None = 1.0,
    mixed_precision: bool = False,
    normalize: bool = True,
    lr_scheduler: str = 'plateau',
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    topology_path: str | Path | None = None,
    residue_chunk_size: int | None = None,
    deterministic: bool = False,
    progress_fn: Callable[[int, float, float | None], None] | None = None,
    num_heads: int = 4,
    window_sizes: list[int] | None = None,
) -> InfluenceTrainResult:
    """Train the AllostericInfluenceModel.

    window_sizes: if provided, trains on all listed window sizes simultaneously
                  (multi-scale). Each size forms its own batch bucket so batches
                  are always homogeneous in the time dimension. Overrides window_size
                  for sample generation; window_size is still used as the canonical
                  single-scale fallback.
    num_heads:    number of attention heads (hidden_dim must be divisible by num_heads).
    """
    min_sequence_separation = max(min_sequence_separation, 1)
    seed_everything(seed, deterministic=deterministic)
    torch_device = resolve_device(device)
    use_amp = mixed_precision and torch_device.type == 'cuda'
    if mixed_precision and not use_amp:
        warnings.warn(
            'mixed_precision requested but device is not CUDA; running in full precision',
            stacklevel=2,
        )
    scaler = torch.amp.GradScaler(torch_device.type, enabled=use_amp)

    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)

    # Residue identity indices — fixed per protein, passed to every forward call.
    residue_type_indices = torch.tensor(
        [aa_name_to_idx(r.name) for r in trajectory.residues],
        dtype=torch.long,
        device=torch_device,
    )

    # Build sample groups — one list per window size.
    effective_window_sizes = window_sizes if window_sizes else [window_size]
    all_groups: list[list[InfluenceSample]] = []
    for ws in effective_window_sizes:
        group = build_influence_samples(
            trajectory.coordinates,
            window_size=ws,
            stride=stride,
            time_step=time_step,
            preprocess=preprocess,
            normalize=normalize,
        )
        if group:
            all_groups.append(group)

    if not all_groups:
        raise ValueError('trajectory did not yield any influence training windows')

    # Split each group independently to avoid leakage across timescales.
    train_groups: list[list[InfluenceSample]] = []
    val_groups: list[list[InfluenceSample]] = []
    for group in all_groups:
        tr, va = split_samples(group, validation_fraction, seed)
        if tr:
            train_groups.append(tr)
        if va:
            val_groups.append(va)

    if not train_groups:
        raise ValueError('training split did not yield any influence training samples')

    total_samples = sum(len(g) for g in all_groups)
    total_train = sum(len(g) for g in train_groups)
    total_val = sum(len(g) for g in val_groups)

    state_dim = int(all_groups[0][0].state_features.shape[-1])
    model = AllostericInfluenceModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        residue_chunk_size=residue_chunk_size,
        min_sequence_separation=min_sequence_separation,
        num_heads=num_heads,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if lr_scheduler not in {'none', 'plateau'}:
        raise ValueError(f"lr_scheduler must be one of none, plateau (got {lr_scheduler!r})")
    scheduler = None
    if lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

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

        # Iterate over each window-size bucket (same T within a batch).
        # Shuffle within each bucket each epoch for diversity.
        for group in train_groups:
            shuffled = list(group)
            random.shuffle(shuffled)
            for batch_samples in iter_batches(shuffled, batch_size):
                batch = stack_influence_batch(batch_samples, torch_device)
                optimizer.zero_grad()
                with torch.autocast(device_type=torch_device.type, enabled=use_amp):
                    output = model(batch.state_features, residue_type_indices=residue_type_indices)
                    losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
                if not torch.isfinite(losses.total):
                    raise ValueError(
                        f'non-finite training loss at epoch {epoch + 1}, batch {epoch_batch_count + 1}'
                    )
                scaler.scale(losses.total).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                last_loss = float(losses.total.detach().item())
                epoch_loss_sum += last_loss
                epoch_batch_count += 1

        train_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        if val_groups:
            validation_loss = _evaluate_epoch(
                model=model,
                val_groups=val_groups,
                device=torch_device,
                sparsity_weight=sparsity_weight,
                batch_size=batch_size,
                residue_type_indices=residue_type_indices,
            )
            if scheduler is not None:
                scheduler.step(validation_loss)
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, validation_loss)
            elif verbose:
                marker = '  [best]' if is_best else ''
                print(
                    f'epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}',
                    flush=True,
                )
            if patience > 0 and epochs_without_improvement >= patience:
                if progress_fn is None and verbose:
                    print(f'early stop at epoch {epoch + 1}', flush=True)
                break
        else:
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, None)
            elif verbose:
                print(f'epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}', flush=True)

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
            min_sequence_separation=min_sequence_separation,
            num_heads=num_heads,
            metadata={
                'training': {
                    'seed': seed,
                    'device': device,
                    'batch_size': batch_size,
                    'validation_fraction': validation_fraction,
                    'patience': patience,
                    'train_samples': total_train,
                    'validation_samples': total_val,
                    'best_epoch': best_epoch,
                    'best_validation_loss': best_validation_loss,
                    'last_loss': last_loss,
                    'normalize': normalize,
                    'residue_chunk_size': residue_chunk_size,
                    'window_sizes': effective_window_sizes,
                }
            },
        )

    return InfluenceTrainResult(
        model=model,
        num_samples=total_samples,
        last_loss=last_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        train_samples=total_train,
        validation_samples=total_val,
        batch_size=batch_size,
    )


__all__ = ['InfluenceLossBreakdown', 'InfluenceTrainResult', 'train_influence_model']
