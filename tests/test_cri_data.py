from __future__ import annotations

import numpy as np

from allostery.cri.data import CRISample, build_cri_training_samples


def test_build_cri_training_samples_constructs_dynamics_and_graph_windows() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 1.0, 0.0]],
            [[4.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 2.0, 0.0]],
            [[9.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 3.0, 0.0]],
        ],
        dtype=np.float32,
    )

    samples = build_cri_training_samples(
        coordinates,
        window_size=4,
        stride=1,
        time_step=1.0,
        distance_cutoff=4.0,
        max_neighbors=1,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, CRISample)
    assert sample.state_features.shape == (2, 3, 6)
    assert sample.acceleration_targets.shape == (2, 3, 3)
    np.testing.assert_array_equal(sample.edge_index, np.array([[1, 0], [0, 1], [0, 2]], dtype=np.int64))
    assert len(sample.incoming_edges) == 3


def test_build_cri_training_samples_returns_empty_for_short_trajectory() -> None:
    samples = build_cri_training_samples(
        np.zeros((2, 3, 3), dtype=np.float32),
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=4.0,
        max_neighbors=1,
    )

    assert samples == []
