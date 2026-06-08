from __future__ import annotations

from pathlib import Path

from allostery.pipeline.influence_train import train_influence_model
from allostery.pipeline.influence_score import score_influence_trajectory


def test_score_influence_trajectory_returns_ranked_pairs(fixture_path: Path) -> None:
    result = train_influence_model(
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device='cpu',
        batch_size=1,
    )

    scores = score_influence_trajectory(
        model=result.model,
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3,
        stride=1,
        time_step=1.0,
    )

    assert len(scores) > 0
    # Sorted descending by score
    assert scores[0]['score'] >= scores[-1]['score']
    # Directed influence values present
    assert 'influence_i_on_j' in scores[0]
    assert 'influence_j_on_i' in scores[0]
    # Symmetry of score w.r.t. directed values
    for pair in scores:
        expected = (pair['influence_i_on_j'] + pair['influence_j_on_i']) / 2.0
        assert abs(pair['score'] - expected) < 1e-6


def test_score_influence_trajectory_covers_all_pairs(fixture_path: Path) -> None:
    result = train_influence_model(
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device='cpu',
        batch_size=1,
    )

    scores = score_influence_trajectory(
        model=result.model,
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3,
        stride=1,
        time_step=1.0,
    )

    from allostery.io.pdb import load_multimodel_pdb
    trajectory = load_multimodel_pdb(fixture_path / 'tiny_trajectory.pdb')
    num_residues = trajectory.coordinates.shape[1]
    expected_pairs = num_residues * (num_residues - 1) // 2
    assert len(scores) == expected_pairs
