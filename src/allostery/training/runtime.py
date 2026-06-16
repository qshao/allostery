from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np
import torch
from torch import Tensor

from allostery.cri.data import CRISample
from allostery.data import TrainingSample
from allostery.influence.data import InfluenceSample

T = TypeVar('T')


@dataclass(frozen=True, slots=True)
class BatchedRelationalSample:
    residue_features: Tensor
    pair_index: Tensor
    pair_features: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class BatchedInfluenceSample:
    state_features: Tensor         # [batch, time, N, state_dim]
    acceleration_targets: Tensor   # [batch, time, N, 3]


@dataclass(frozen=True, slots=True)
class BatchedCRISample:
    state_features: Tensor
    acceleration_targets: Tensor
    edge_index: Tensor
    edge_distance: Tensor
    edge_mask: Tensor


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == '':
        return torch.device('cpu')
    return torch.device(device)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def iter_batches(items: Sequence[T], batch_size: int) -> list[list[T]]:
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    return [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


def split_samples(
    samples: Sequence[T],
    validation_fraction: float,
    seed: int,
) -> tuple[list[T], list[T]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError('validation_fraction must be greater than or equal to zero and less than one')
    items = list(samples)
    if not items or validation_fraction <= 0.0 or len(items) == 1:
        return items, []

    validation_count = int(round(len(items) * validation_fraction))
    validation_count = max(1, min(len(items) - 1, validation_count))
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    validation_indices = sorted(indices[:validation_count])
    train_indices = sorted(indices[validation_count:])
    return [items[index] for index in train_indices], [items[index] for index in validation_indices]


def stack_relational_batch(samples: Sequence[TrainingSample], device: torch.device) -> BatchedRelationalSample:
    if not samples:
        raise ValueError('samples must not be empty')
    reference_pair_index = torch.as_tensor(samples[0].pair_index, dtype=torch.int64, device=device)
    pair_index = reference_pair_index
    for sample in samples[1:]:
        current_pair_index = torch.as_tensor(sample.pair_index, dtype=torch.int64, device=device)
        if not torch.equal(pair_index, current_pair_index):
            raise ValueError('pair_index must be identical within a relational batch')
    return BatchedRelationalSample(
        residue_features=torch.stack(
            [torch.as_tensor(sample.residue_features, dtype=torch.float32, device=device) for sample in samples],
            dim=0,
        ),
        pair_index=pair_index,
        pair_features=torch.stack(
            [torch.as_tensor(sample.pair_features, dtype=torch.float32, device=device) for sample in samples],
            dim=0,
        ),
        targets=torch.stack(
            [torch.as_tensor(sample.targets, dtype=torch.float32, device=device) for sample in samples],
            dim=0,
        ),
    )


def stack_influence_batch(samples: Sequence[InfluenceSample], device: torch.device) -> BatchedInfluenceSample:
    if not samples:
        raise ValueError('samples must not be empty')
    return BatchedInfluenceSample(
        state_features=torch.stack(
            [torch.as_tensor(s.state_features, dtype=torch.float32, device=device) for s in samples],
            dim=0,
        ),
        acceleration_targets=torch.stack(
            [torch.as_tensor(s.acceleration_targets, dtype=torch.float32, device=device) for s in samples],
            dim=0,
        ),
    )


def stack_cri_batch(samples: Sequence[CRISample], device: torch.device) -> BatchedCRISample:
    if not samples:
        raise ValueError('samples must not be empty')

    batch_size = len(samples)
    max_edges = max(sample.edge_index.shape[0] for sample in samples)
    edge_index = torch.full((batch_size, max_edges, 2), -1, dtype=torch.long, device=device)
    edge_distance = torch.zeros((batch_size, max_edges), dtype=torch.float32, device=device)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=device)
    for batch_index, sample in enumerate(samples):
        edge_count = sample.edge_index.shape[0]
        if edge_count == 0:
            continue
        edge_index[batch_index, :edge_count] = torch.as_tensor(sample.edge_index, dtype=torch.long, device=device)
        edge_distance[batch_index, :edge_count] = torch.as_tensor(
            sample.edge_distance,
            dtype=torch.float32,
            device=device,
        )
        edge_mask[batch_index, :edge_count] = True

    return BatchedCRISample(
        state_features=torch.stack(
            [torch.as_tensor(sample.state_features, dtype=torch.float32, device=device) for sample in samples],
            dim=0,
        ),
        acceleration_targets=torch.stack(
            [torch.as_tensor(sample.acceleration_targets, dtype=torch.float32, device=device) for sample in samples],
            dim=0,
        ),
        edge_index=edge_index,
        edge_distance=edge_distance,
        edge_mask=edge_mask,
    )
