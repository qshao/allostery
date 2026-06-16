from __future__ import annotations

import numpy as np
import pytest

from allostery.features.dynamics import build_residue_dynamics


def test_build_residue_dynamics_uses_central_differences() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
            [[9.0, 0.0, 0.0], [0.0, 12.0, 0.0]],
        ],
        dtype=np.float32,
    )

    dynamics = build_residue_dynamics(coordinates, time_step=2.0)

    assert dynamics.positions.shape == (2, 2, 3)
    np.testing.assert_allclose(dynamics.positions, coordinates[1:-1], atol=1e-6)
    np.testing.assert_allclose(
        dynamics.velocities,
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.5, 0.0]],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dynamics.accelerations,
        np.array(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_build_residue_dynamics_rejects_short_windows() -> None:
    with pytest.raises(ValueError, match="at least 3 frames"):
        build_residue_dynamics(np.zeros((2, 3, 3), dtype=np.float32), time_step=1.0)


def test_build_residue_dynamics_rejects_nonfinite_coordinates() -> None:
    coordinates = np.zeros((3, 2, 3), dtype=np.float32)
    coordinates[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_residue_dynamics(coordinates, time_step=1.0)


def test_build_residue_dynamics_rejects_nonpositive_time_step() -> None:
    with pytest.raises(ValueError, match="time_step"):
        build_residue_dynamics(np.zeros((3, 2, 3), dtype=np.float32), time_step=0.0)


def test_normalize_makes_positions_translation_invariant() -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((4, 6, 3)).astype(np.float32)
    shifted = coords + np.array([10.0, -5.0, 3.0], dtype=np.float32)
    base = build_residue_dynamics(coords, normalize=True)
    moved = build_residue_dynamics(shifted, normalize=True)
    np.testing.assert_allclose(base.positions, moved.positions, atol=1e-4)


def test_normalize_false_keeps_absolute_positions() -> None:
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((4, 6, 3)).astype(np.float32)
    shifted = coords + 10.0
    base = build_residue_dynamics(coords, normalize=False)
    moved = build_residue_dynamics(shifted, normalize=False)
    assert not np.allclose(base.positions, moved.positions, atol=1e-4)
