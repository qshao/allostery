from __future__ import annotations

from pathlib import Path

from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model


def test_score_cri_trajectory_returns_ranked_residue_pairs(fixture_path: Path) -> None:
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

    scores = score_cri_trajectory(
        model=result.model,
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
    )

    assert scores
    assert scores[0]["score"] >= scores[-1]["score"]
    assert "edge_type_probabilities" in scores[0]
