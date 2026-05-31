from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.residue import _summary_statistics, _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class PairwiseFeatures:
    pair_index: np.ndarray
    distance_series: np.ndarray
    delta_distance_series: np.ndarray
    distance_mean: np.ndarray
    distance_std: np.ndarray
    distance_min: np.ndarray
    distance_max: np.ndarray
    distance_range: np.ndarray

    @property
    def summary_features(self) -> np.ndarray:
        return np.stack(
            (
                self.distance_mean,
                self.distance_std,
                self.distance_min,
                self.distance_max,
                self.distance_range,
            ),
            axis=1,
        ).astype(np.float32, copy=False)


def unordered_pair_index(num_residues: int) -> np.ndarray:
    if num_residues < 0:
        raise ValueError("num_residues must be non-negative")

    pairs = [(i, j) for i in range(num_residues) for j in range(i + 1, num_residues)]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def build_pairwise_features(window_coordinates: np.ndarray) -> PairwiseFeatures:
    coordinates = _validate_coordinate_window(window_coordinates)
    pair_index = unordered_pair_index(coordinates.shape[1])

    if pair_index.size == 0:
        num_frames = coordinates.shape[0]
        empty_pairs = np.empty(0, dtype=np.float32)
        return PairwiseFeatures(
            pair_index=pair_index,
            distance_series=np.empty((0, num_frames), dtype=np.float32),
            delta_distance_series=np.empty((0, max(num_frames - 1, 0)), dtype=np.float32),
            distance_mean=empty_pairs,
            distance_std=empty_pairs,
            distance_min=empty_pairs,
            distance_max=empty_pairs,
            distance_range=empty_pairs,
        )

    pair_vectors = coordinates[:, pair_index[:, 0], :] - coordinates[:, pair_index[:, 1], :]
    distance_series = np.linalg.norm(pair_vectors, axis=-1).T.astype(np.float32)
    delta_distance_series = np.diff(distance_series, axis=1)
    distance_summary = _summary_statistics(distance_series, axis=1)
    return PairwiseFeatures(
        pair_index=pair_index,
        distance_series=distance_series,
        delta_distance_series=delta_distance_series,
        distance_mean=distance_summary[:, 0],
        distance_std=distance_summary[:, 1],
        distance_min=distance_summary[:, 2],
        distance_max=distance_summary[:, 3],
        distance_range=distance_summary[:, 4],
    )
