from __future__ import annotations

import numpy as np
import pytest

from allostery.features.alignment import align_trajectory_coordinates, _kabsch_align, center_trajectory_coordinates


def test_center_trajectory_coordinates_removes_translation() -> None:
    coordinates = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ],
        dtype=np.float32,
    )

    centered = center_trajectory_coordinates(coordinates)

    np.testing.assert_allclose(centered.mean(axis=1), np.zeros((2, 3), dtype=np.float32), atol=1e-6)


def test_align_trajectory_coordinates_preserves_pairwise_distances() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]],
        ],
        dtype=np.float32,
    )

    aligned = align_trajectory_coordinates(coordinates, reference_frame_index=0)

    np.testing.assert_allclose(
        np.linalg.norm(aligned[:, 0] - aligned[:, 1], axis=-1),
        np.array([1.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_align_trajectory_coordinates_rejects_invalid_reference_frame() -> None:
    with pytest.raises(IndexError, match="reference_frame_index"):
        align_trajectory_coordinates(np.zeros((2, 2, 3), dtype=np.float32), reference_frame_index=2)


def test_batched_align_matches_per_frame_loop() -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((5, 7, 3)).astype(np.float32)
    reference = coords[0]
    expected = np.stack([_kabsch_align(frame, reference) for frame in coords], axis=0)
    result = align_trajectory_coordinates(coords, reference_frame_index=0)
    assert result.shape == coords.shape
    np.testing.assert_allclose(result, expected, atol=1e-5)
