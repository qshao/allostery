from __future__ import annotations

import numpy as np
import pytest

from allostery.features.graph import build_directed_contact_graph, incoming_edge_indices


def test_build_directed_contact_graph_uses_cutoff_and_top_k() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    graph = build_directed_contact_graph(coordinates, distance_cutoff=3.0, max_neighbors=1)

    np.testing.assert_array_equal(graph.edge_index, np.array([[1, 0], [0, 1], [1, 2]], dtype=np.int64))
    np.testing.assert_allclose(graph.mean_distances, np.array([1.5, 1.5, 2.5], dtype=np.float32), atol=1e-6)


def test_build_directed_contact_graph_can_enforce_sequence_separation() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0], [5.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    graph = build_directed_contact_graph(
        coordinates,
        distance_cutoff=3.0,
        max_neighbors=2,
        min_sequence_separation=2,
    )

    assert all(abs(int(sender) - int(receiver)) >= 2 for sender, receiver in graph.edge_index)


def test_incoming_edge_indices_groups_edges_by_receiver() -> None:
    edge_index = np.array([[1, 0], [2, 0], [0, 1], [1, 2]], dtype=np.int64)

    incoming = incoming_edge_indices(edge_index, num_residues=3)

    assert [group.tolist() for group in incoming] == [[0, 1], [2], [3]]


def test_build_directed_contact_graph_rejects_invalid_parameters() -> None:
    coordinates = np.zeros((3, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="distance_cutoff"):
        build_directed_contact_graph(coordinates, distance_cutoff=0.0, max_neighbors=1)
    with pytest.raises(ValueError, match="max_neighbors"):
        build_directed_contact_graph(coordinates, distance_cutoff=1.0, max_neighbors=0)
