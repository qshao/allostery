from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.pairwise import build_pairwise_features
from allostery.features.residue import build_residue_motion_features
from allostery.windows import generate_window_slices


@dataclass(frozen=True, slots=True)
class TrainingSample:
    residue_features: np.ndarray
    pair_index: np.ndarray
    pair_features: np.ndarray
    targets: np.ndarray


def _validate_trajectory_coordinates(coordinates: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(coordinates, dtype=np.float32)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("coordinates must have shape (num_frames, num_residues, 3)")
    return trajectory


def build_future_pair_targets(future_coordinates: np.ndarray) -> np.ndarray:
    future_pairwise = build_pairwise_features(future_coordinates)
    return np.column_stack(
        (
            future_pairwise.distance_mean,
            future_pairwise.distance_std,
            future_pairwise.distance_range,
        )
    ).astype(np.float32, copy=False)


def build_training_samples(
    coordinates: np.ndarray,
    window_size: int,
    horizon_size: int,
    stride: int,
) -> list[TrainingSample]:
    trajectory = _validate_trajectory_coordinates(coordinates)
    samples: list[TrainingSample] = []
    for past_slice, future_slice in generate_window_slices(
        num_frames=trajectory.shape[0],
        window_size=window_size,
        horizon_size=horizon_size,
        stride=stride,
    ):
        past_coordinates = trajectory[past_slice]
        past_pairwise = build_pairwise_features(past_coordinates)
        samples.append(
            TrainingSample(
                residue_features=build_residue_motion_features(past_coordinates),
                pair_index=past_pairwise.pair_index,
                pair_features=past_pairwise.summary_features,
                targets=build_future_pair_targets(trajectory[future_slice]),
            )
        )
    return samples


__all__ = ["TrainingSample", "build_future_pair_targets", "build_training_samples"]
