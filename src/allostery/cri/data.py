from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.dynamics import build_residue_dynamics
from allostery.features.graph import build_directed_contact_graph, incoming_edge_indices
from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class CRISample:
    state_features: np.ndarray
    acceleration_targets: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray
    incoming_edges: tuple[np.ndarray, ...]


def build_cri_training_samples(
    coordinates: np.ndarray,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
    min_sequence_separation: int = 0,
    preprocess: str = "none",
) -> list[CRISample]:
    trajectory = _validate_coordinate_window(coordinates)
    if window_size < 3:
        raise ValueError("window_size must be at least 3 for central differences")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if min_sequence_separation < 0:
        raise ValueError("min_sequence_separation must be greater than or equal to zero")
    if trajectory.shape[0] < window_size:
        return []

    samples: list[CRISample] = []
    for start in range(0, trajectory.shape[0] - window_size + 1, stride):
        window = trajectory[start : start + window_size]
        dynamics = build_residue_dynamics(window, time_step=time_step, preprocess=preprocess)
        graph = build_directed_contact_graph(
            window,
            distance_cutoff=distance_cutoff,
            max_neighbors=max_neighbors,
            min_sequence_separation=min_sequence_separation,
        )
        samples.append(
            CRISample(
                state_features=dynamics.state_features,
                acceleration_targets=dynamics.accelerations,
                edge_index=graph.edge_index,
                edge_distance=graph.mean_distances,
                incoming_edges=incoming_edge_indices(graph.edge_index, num_residues=window.shape[1]),
            )
        )
    return samples


__all__ = ["CRISample", "build_cri_training_samples"]
