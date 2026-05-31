from __future__ import annotations

import numpy as np

from allostery.data import TrainingSample, build_future_pair_targets, build_training_samples
from allostery.features.pairwise import build_pairwise_features
from allostery.features.residue import build_residue_motion_features
from allostery.io.pdb import load_multimodel_pdb


def test_build_future_pair_targets_uses_compact_pair_summary(fixture_path) -> None:
    trajectory = load_multimodel_pdb(fixture_path / "tiny_trajectory.pdb")

    targets = build_future_pair_targets(trajectory.coordinates[1:3])
    future_pairwise = build_pairwise_features(trajectory.coordinates[1:3])

    np.testing.assert_allclose(
        targets,
        np.column_stack(
            (
                future_pairwise.distance_mean,
                future_pairwise.distance_std,
                future_pairwise.distance_range,
            )
        ),
        atol=1e-6,
    )


def test_build_training_samples_constructs_windowed_feature_target_examples(fixture_path) -> None:
    trajectory = load_multimodel_pdb(fixture_path / "tiny_trajectory.pdb")

    samples = build_training_samples(
        trajectory.coordinates,
        window_size=1,
        horizon_size=2,
        stride=1,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, TrainingSample)
    np.testing.assert_allclose(
        sample.residue_features,
        build_residue_motion_features(trajectory.coordinates[0:1]),
        atol=1e-6,
    )
    expected_pairwise = build_pairwise_features(trajectory.coordinates[0:1])
    np.testing.assert_array_equal(sample.pair_index, expected_pairwise.pair_index)
    np.testing.assert_allclose(sample.pair_features, expected_pairwise.summary_features, atol=1e-6)
    np.testing.assert_allclose(
        sample.targets,
        build_future_pair_targets(trajectory.coordinates[1:3]),
        atol=1e-6,
    )


def load_tests(loader, tests, pattern):
    import unittest
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures"
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(lambda: test_build_future_pair_targets_uses_compact_pair_summary(fixture_path)))
    suite.addTest(unittest.FunctionTestCase(lambda: test_build_training_samples_constructs_windowed_feature_target_examples(fixture_path)))
    return suite
