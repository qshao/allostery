from __future__ import annotations

from pathlib import Path

import torch

from allostery.pipeline.cri_train import train_cri_model
from allostery.training.cri_objectives import cri_loss


def test_cri_loss_combines_reconstruction_entropy_and_sparsity() -> None:
    prediction = {
        "acceleration": torch.zeros(1, 2, 3, 3),
        "edge_type_prob": torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]], dtype=torch.float32),
        "edge_score": torch.tensor([[0.3, 0.9]], dtype=torch.float32),
    }
    target = torch.ones(1, 2, 3, 3)

    losses = cri_loss(
        prediction,
        target,
        entropy_weight=0.01,
        no_edge_weight=0.02,
    )

    assert losses.reconstruction.item() > 0.0
    assert losses.entropy.item() > 0.0
    assert losses.no_edge.item() > 0.0
    torch.testing.assert_close(losses.total, losses.reconstruction + losses.entropy + losses.no_edge)


def test_train_cri_model_runs_on_tiny_trajectory(fixture_path: Path) -> None:
    result = train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
    )

    assert result.num_samples == 1
    assert result.last_loss >= 0.0
