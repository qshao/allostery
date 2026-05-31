from __future__ import annotations

import numpy as np


def _validate_coordinate_window(window_coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(window_coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("window_coordinates must have shape (num_frames, num_residues, 3)")
    return coordinates


def _summary_statistics(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] == 0:
        output_shape = list(values.shape)
        del output_shape[axis]
        return np.zeros((*output_shape, 5), dtype=np.float32)

    mean = values.mean(axis=axis)
    std = values.std(axis=axis)
    minimum = values.min(axis=axis)
    maximum = values.max(axis=axis)
    value_range = maximum - minimum
    return np.stack((mean, std, minimum, maximum, value_range), axis=-1).astype(np.float32)


def build_residue_motion_features(window_coordinates: np.ndarray) -> np.ndarray:
    coordinates = _validate_coordinate_window(window_coordinates)

    displacement = np.linalg.norm(np.diff(coordinates, axis=0), axis=-1)
    mean_position = coordinates.mean(axis=0, keepdims=True)
    fluctuation = np.linalg.norm(coordinates - mean_position, axis=-1)

    displacement_summary = _summary_statistics(displacement, axis=0)
    fluctuation_summary = _summary_statistics(fluctuation, axis=0)
    return np.concatenate((displacement_summary, fluctuation_summary), axis=-1)
