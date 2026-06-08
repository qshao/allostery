from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.dynamics import build_residue_dynamics
from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class InfluenceSample:
    state_features: np.ndarray         # [time, N, state_dim]
    acceleration_targets: np.ndarray   # [time, N, 3]


def build_influence_samples(
    coordinates: np.ndarray,
    window_size: int,
    stride: int,
    time_step: float = 1.0,
    preprocess: str = 'none',
) -> list[InfluenceSample]:
    trajectory = _validate_coordinate_window(coordinates)
    if window_size < 3:
        raise ValueError('window_size must be at least 3 for central differences')
    if stride <= 0:
        raise ValueError('stride must be greater than zero')
    if trajectory.shape[0] < window_size:
        return []

    samples: list[InfluenceSample] = []
    for start in range(0, trajectory.shape[0] - window_size + 1, stride):
        window = trajectory[start : start + window_size]
        dynamics = build_residue_dynamics(window, time_step=time_step, preprocess=preprocess)
        samples.append(
            InfluenceSample(
                state_features=dynamics.state_features,
                acceleration_targets=dynamics.accelerations,
            )
        )
    return samples


__all__ = ['InfluenceSample', 'build_influence_samples']
