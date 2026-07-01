from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.alignment import align_trajectory_coordinates, center_trajectory_coordinates
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
    normalize: bool = False,
) -> list[InfluenceSample]:
    trajectory = _validate_coordinate_window(coordinates)
    if window_size < 3:
        raise ValueError('window_size must be at least 3 for central differences')
    if stride <= 0:
        raise ValueError('stride must be greater than zero')
    if trajectory.shape[0] < window_size:
        return []

    # Pre-process the full trajectory once — one SVD batch of [T,3,3] instead of
    # one per window (e.g. ~5000 calls for KRAS WT w5 → 1 call).
    if preprocess == 'align':
        trajectory = align_trajectory_coordinates(trajectory)
        window_preprocess = 'none'
    elif preprocess == 'center':
        trajectory = center_trajectory_coordinates(trajectory)
        window_preprocess = 'none'
    else:
        window_preprocess = preprocess

    samples: list[InfluenceSample] = []
    for start in range(0, trajectory.shape[0] - window_size + 1, stride):
        window = trajectory[start : start + window_size]
        dynamics = build_residue_dynamics(window, time_step=time_step, preprocess=window_preprocess, normalize=normalize)
        samples.append(
            InfluenceSample(
                state_features=dynamics.state_features,
                acceleration_targets=dynamics.accelerations,
            )
        )
    return samples


__all__ = ['InfluenceSample', 'build_influence_samples']
