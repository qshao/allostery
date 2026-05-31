from __future__ import annotations

import numpy as np
import unittest

from allostery.features.pairwise import build_pairwise_features
from allostery.features.residue import build_residue_motion_features


def test_build_residue_motion_features_returns_one_summary_per_residue() -> None:
    window_coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            [[3.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    features = build_residue_motion_features(window_coordinates)

    assert features.shape == (2, 10)
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [1.5, 0.5, 1.0, 2.0, 1.0, 1.1111112, 0.5665577, 0.33333337, 1.6666666, 1.3333333],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.6666667, 0.47140452, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_build_pairwise_features_returns_distance_dynamics_and_summary() -> None:
    window_coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    features = build_pairwise_features(window_coordinates)

    np.testing.assert_array_equal(
        features.pair_index,
        np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        features.distance_series,
        np.array(
            [
                [1.0, 2.0, 4.0],
                [3.0, 3.0, 3.0],
                [2.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        features.delta_distance_series,
        np.array(
            [
                [1.0, 2.0],
                [0.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        features.summary_features,
        np.array(
            [
                [2.3333333, 1.2472191, 1.0, 4.0, 3.0],
                [3.0, 0.0, 3.0, 3.0, 0.0],
                [1.3333334, 0.47140452, 1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_build_residue_motion_features_rejects_bad_coordinate_shape() -> None:
    with np.testing.assert_raises_regex(ValueError, "window_coordinates"):
        build_residue_motion_features(np.zeros((3, 2), dtype=np.float32))


def test_build_pairwise_features_exposes_named_summary_statistics() -> None:
    window_coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    features = build_pairwise_features(window_coordinates)

    np.testing.assert_allclose(
        features.distance_mean,
        np.array([2.3333333, 3.0, 1.3333334], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features.distance_std,
        np.array([1.2472191, 0.0, 0.47140452], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features.distance_min,
        np.array([1.0, 3.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features.distance_max,
        np.array([4.0, 3.0, 2.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features.distance_range,
        np.array([3.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features.summary_features,
        np.column_stack(
            (
                features.distance_mean,
                features.distance_std,
                features.distance_min,
                features.distance_max,
                features.distance_range,
            )
        ),
        atol=1e-6,
    )


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(
        unittest.FunctionTestCase(test_build_residue_motion_features_returns_one_summary_per_residue)
    )
    suite.addTest(
        unittest.FunctionTestCase(test_build_pairwise_features_returns_distance_dynamics_and_summary)
    )
    suite.addTest(
        unittest.FunctionTestCase(test_build_residue_motion_features_rejects_bad_coordinate_shape)
    )
    suite.addTest(
        unittest.FunctionTestCase(test_build_pairwise_features_exposes_named_summary_statistics)
    )
    return suite
