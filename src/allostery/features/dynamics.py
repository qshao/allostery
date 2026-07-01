from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.alignment import align_trajectory_coordinates, center_trajectory_coordinates
from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class ResidueDynamics:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    @property
    def state_features(self) -> np.ndarray:
        return np.concatenate((self.positions, self.velocities), axis=-1).astype(np.float32, copy=False)


def build_residue_dynamics(
    window_coordinates: np.ndarray,
    time_step: float = 1.0,
    preprocess: str = "none",
    reference_frame_index: int = 0,
    normalize: bool = False,
) -> ResidueDynamics:
    if time_step <= 0.0:
        raise ValueError("time_step must be greater than zero")

    coordinates = _validate_coordinate_window(window_coordinates)
    if coordinates.shape[0] < 3:
        raise ValueError("window_coordinates must contain at least 3 frames")
    if not np.isfinite(coordinates).all():
        raise ValueError("window_coordinates must contain only finite values")

    if preprocess == "center":
        coordinates = center_trajectory_coordinates(coordinates)
    elif preprocess == "align":
        coordinates = align_trajectory_coordinates(coordinates, reference_frame_index=reference_frame_index)
    elif preprocess != "none":
        raise ValueError("preprocess must be one of none, center, or align")

    positions = coordinates[1:-1]
    velocities = (coordinates[2:] - coordinates[:-2]) / (2.0 * time_step)
    accelerations = (coordinates[2:] - (2.0 * coordinates[1:-1]) + coordinates[:-2]) / (time_step * time_step)
    if normalize:
        # Centre positions; normalize each physical quantity by its own window std
        # so all three have O(1) magnitude. Velocities and accelerations are already
        # translation-invariant (finite differences), so they need only scaling.
        # Per-quantity normalization prevents the trivial zero-acceleration solution:
        # without it, acceleration targets are ~10,000× smaller than position features,
        # making MSE(predict-zero) ≈ 4e-7 — a trivially satisfied reconstruction loss
        # that lets the model ignore the dynamics entirely.
        positions = positions - positions.mean(axis=(0, 1), keepdims=True)
        pos_scale = max(float(positions.std()), 1e-8)
        positions = positions / pos_scale
        vel_scale = max(float(velocities.std()), 1e-8)
        velocities = velocities / vel_scale
        accel_scale = max(float(accelerations.std()), 1e-8)
        accelerations = accelerations / accel_scale
    return ResidueDynamics(
        positions=positions.astype(np.float32, copy=False),
        velocities=velocities.astype(np.float32, copy=False),
        accelerations=accelerations.astype(np.float32, copy=False),
    )


__all__ = ["ResidueDynamics", "build_residue_dynamics"]
