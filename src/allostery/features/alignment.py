from __future__ import annotations

import numpy as np

from allostery.features.residue import _validate_coordinate_window


def center_trajectory_coordinates(window_coordinates: np.ndarray) -> np.ndarray:
    coordinates = _validate_coordinate_window(window_coordinates)
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    return centered.astype(np.float32, copy=False)


def align_trajectory_coordinates(
    window_coordinates: np.ndarray,
    reference_frame_index: int = 0,
) -> np.ndarray:
    coordinates = _validate_coordinate_window(window_coordinates)
    if reference_frame_index < 0 or reference_frame_index >= coordinates.shape[0]:
        raise IndexError("reference_frame_index is out of range")

    reference = coordinates[reference_frame_index]
    aligned = np.empty_like(coordinates)
    for frame_index, frame in enumerate(coordinates):
        aligned[frame_index] = _kabsch_align(frame, reference)
    return aligned.astype(np.float32, copy=False)


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_centered = mobile - mobile.mean(axis=0, keepdims=True)
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    covariance = mobile_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    return mobile_centered @ rotation + reference.mean(axis=0, keepdims=True)


__all__ = ["align_trajectory_coordinates", "center_trajectory_coordinates"]
