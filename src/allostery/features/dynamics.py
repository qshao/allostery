from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class ResidueDynamics:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    @property
    def state_features(self) -> np.ndarray:
        return np.concatenate((self.positions, self.velocities), axis=-1).astype(np.float32, copy=False)


def build_residue_dynamics(window_coordinates: np.ndarray, time_step: float = 1.0) -> ResidueDynamics:
    if time_step <= 0.0:
        raise ValueError("time_step must be greater than zero")

    coordinates = _validate_coordinate_window(window_coordinates)
    if coordinates.shape[0] < 3:
        raise ValueError("window_coordinates must contain at least 3 frames")
    if not np.isfinite(coordinates).all():
        raise ValueError("window_coordinates must contain only finite values")

    positions = coordinates[1:-1]
    velocities = (coordinates[2:] - coordinates[:-2]) / (2.0 * time_step)
    accelerations = (coordinates[2:] - (2.0 * coordinates[1:-1]) + coordinates[:-2]) / (time_step * time_step)
    return ResidueDynamics(
        positions=positions.astype(np.float32, copy=False),
        velocities=velocities.astype(np.float32, copy=False),
        accelerations=accelerations.astype(np.float32, copy=False),
    )


__all__ = ["ResidueDynamics", "build_residue_dynamics"]
