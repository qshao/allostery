from __future__ import annotations

import numpy as np
import unittest

from allostery.features.pairwise import unordered_pair_index
from allostery.windows import generate_window_slices


def test_generate_window_slices_builds_past_future_ranges() -> None:
    windows = generate_window_slices(
        num_frames=10,
        window_size=4,
        horizon_size=2,
        stride=2,
    )

    assert windows == [
        (slice(0, 4), slice(4, 6)),
        (slice(2, 6), slice(6, 8)),
        (slice(4, 8), slice(8, 10)),
    ]
    assert [past.start for past, _ in windows] == [0, 2, 4]
    assert [past.stop for past, _ in windows] == [4, 6, 8]
    assert [future.start for _, future in windows] == [4, 6, 8]
    assert [future.stop for _, future in windows] == [6, 8, 10]


def test_generate_window_slices_rejects_nonpositive_stride() -> None:
    with np.testing.assert_raises_regex(ValueError, "stride"):
        generate_window_slices(num_frames=10, window_size=4, horizon_size=2, stride=0)


def test_generate_window_slices_rejects_nonpositive_window_size() -> None:
    with np.testing.assert_raises_regex(ValueError, "window_size"):
        generate_window_slices(num_frames=10, window_size=0, horizon_size=2, stride=1)


def test_generate_window_slices_rejects_nonpositive_horizon_size() -> None:
    with np.testing.assert_raises_regex(ValueError, "horizon_size"):
        generate_window_slices(num_frames=10, window_size=4, horizon_size=0, stride=1)


def test_generate_window_slices_returns_empty_list_for_too_few_frames() -> None:
    assert generate_window_slices(num_frames=5, window_size=4, horizon_size=2, stride=1) == []


def test_unordered_pair_index_uses_upper_triangle() -> None:
    pairs = unordered_pair_index(4)

    assert np.array_equal(
        pairs,
        np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
    )


def test_unordered_pair_index_rejects_negative_residue_count() -> None:
    with np.testing.assert_raises_regex(ValueError, "num_residues"):
        unordered_pair_index(-1)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(test_generate_window_slices_builds_past_future_ranges))
    suite.addTest(
        unittest.FunctionTestCase(test_generate_window_slices_rejects_nonpositive_stride)
    )
    suite.addTest(
        unittest.FunctionTestCase(test_generate_window_slices_rejects_nonpositive_window_size)
    )
    suite.addTest(
        unittest.FunctionTestCase(test_generate_window_slices_rejects_nonpositive_horizon_size)
    )
    suite.addTest(
        unittest.FunctionTestCase(test_generate_window_slices_returns_empty_list_for_too_few_frames)
    )
    suite.addTest(unittest.FunctionTestCase(test_unordered_pair_index_uses_upper_triangle))
    suite.addTest(
        unittest.FunctionTestCase(test_unordered_pair_index_rejects_negative_residue_count)
    )
    return suite
