from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class DirectedContactGraph:
    edge_index: np.ndarray
    mean_distances: np.ndarray


def build_directed_contact_graph(
    window_coordinates: np.ndarray,
    distance_cutoff: float,
    max_neighbors: int,
    min_sequence_separation: int = 0,
) -> DirectedContactGraph:
    if distance_cutoff <= 0.0:
        raise ValueError("distance_cutoff must be greater than zero")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be greater than zero")
    if min_sequence_separation < 0:
        raise ValueError("min_sequence_separation must be greater than or equal to zero")

    coordinates = _validate_coordinate_window(window_coordinates)
    if not np.isfinite(coordinates).all():
        raise ValueError("window_coordinates must contain only finite values")

    mean_positions = coordinates.mean(axis=0)
    delta = mean_positions[:, None, :] - mean_positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1).astype(np.float32)
    num_residues = distances.shape[0]

    edges: list[tuple[int, int]] = []
    mean_distances: list[float] = []
    for receiver in range(num_residues):
        candidates = [
            (float(distances[sender, receiver]), sender)
            for sender in range(num_residues)
            if sender != receiver
            and abs(sender - receiver) >= min_sequence_separation
            and distances[sender, receiver] <= distance_cutoff
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        for distance, sender in candidates[:max_neighbors]:
            edges.append((sender, receiver))
            mean_distances.append(distance)

    if not edges:
        return DirectedContactGraph(
            edge_index=np.empty((0, 2), dtype=np.int64),
            mean_distances=np.empty(0, dtype=np.float32),
        )
    return DirectedContactGraph(
        edge_index=np.array(edges, dtype=np.int64),
        mean_distances=np.array(mean_distances, dtype=np.float32),
    )


def incoming_edge_indices(edge_index: np.ndarray, num_residues: int) -> tuple[np.ndarray, ...]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
        raise ValueError("edge_index must have shape (num_edges, 2)")
    return tuple(np.flatnonzero(edge_index[:, 1] == receiver).astype(np.int64) for receiver in range(num_residues))


__all__ = ["DirectedContactGraph", "build_directed_contact_graph", "incoming_edge_indices"]
