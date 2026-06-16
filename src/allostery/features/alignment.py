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
    reference_centroid = reference.mean(axis=0, keepdims=True)
    reference_centered = reference - reference_centroid

    frame_centroids = coordinates.mean(axis=1, keepdims=True)        # [T, 1, 3]
    mobile_centered = coordinates - frame_centroids                  # [T, N, 3]

    # Per-frame covariance: [T, 3, 3] = mobileᵀ @ reference
    covariance = np.einsum('tni,nj->tij', mobile_centered, reference_centered)
    left, _, right_t = np.linalg.svd(covariance)                    # [T,3,3] each
    det = np.linalg.det(np.einsum('tij,tjk->tik', right_t.transpose(0, 2, 1), left.transpose(0, 2, 1)))
    sign = np.where(det < 0.0, -1.0, 1.0)                           # [T]
    right_t = right_t.copy()
    right_t[:, -1, :] *= sign[:, None]
    rotation = np.einsum('tij,tjk->tik', right_t.transpose(0, 2, 1), left.transpose(0, 2, 1))

    aligned = np.einsum('tni,tij->tnj', mobile_centered, rotation) + reference_centroid
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
