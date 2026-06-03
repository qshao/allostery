from __future__ import annotations

import torch

from allostery.models.cri import CRILatentInteractionModel


def test_cri_model_predicts_accelerations_and_edge_probabilities() -> None:
    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=3, dropout=0.0)
    state_features = torch.randn(2, 3, 4, 6)
    edge_index = torch.tensor([[1, 0], [2, 0], [0, 1], [1, 2]], dtype=torch.long)
    edge_distance = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float32)

    output = model(state_features, edge_index, edge_distance)

    assert output["acceleration"].shape == (2, 3, 4, 3)
    assert output["edge_type_prob"].shape == (2, 4, 3)
    torch.testing.assert_close(output["edge_type_prob"].sum(dim=-1), torch.ones(2, 4))
    assert output["edge_score"].shape == (2, 4)


def test_cri_model_handles_padded_batched_edges() -> None:
    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=2, dropout=0.0)
    state_features = torch.randn(2, 2, 3, 6)
    edge_index = torch.tensor(
        [
            [[1, 0], [2, 0], [-1, -1]],
            [[1, 0], [-1, -1], [-1, -1]],
        ],
        dtype=torch.long,
    )
    edge_distance = torch.tensor([[1.0, 2.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
    edge_mask = torch.tensor([[True, True, False], [True, False, False]])

    output = model(state_features, edge_index, edge_distance, edge_mask=edge_mask)

    assert output["acceleration"].shape == (2, 2, 3, 3)
    assert output["edge_type_prob"].shape == (2, 3, 2)
    assert output["edge_score"].shape == (2, 3)
    torch.testing.assert_close(output["edge_mask"], edge_mask.to(dtype=torch.float32))
    torch.testing.assert_close(output["edge_score"][1, 1:], torch.zeros(2))


def test_cri_model_handles_empty_edges() -> None:
    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=2, dropout=0.0)
    state_features = torch.randn(1, 2, 3, 6)
    edge_index = torch.empty((0, 2), dtype=torch.long)
    edge_distance = torch.empty(0, dtype=torch.float32)

    output = model(state_features, edge_index, edge_distance)

    assert output["acceleration"].shape == (1, 2, 3, 3)
    assert output["edge_type_prob"].shape == (1, 0, 2)
    assert output["edge_score"].shape == (1, 0)
