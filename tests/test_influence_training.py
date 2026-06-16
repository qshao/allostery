from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from allostery.training.influence_objectives import InfluenceLossBreakdown, influence_loss
from allostery.pipeline.influence_train import train_influence_model


def test_influence_loss_combines_reconstruction_and_sparsity() -> None:
    prediction = {
        'acceleration': torch.zeros(1, 2, 3, 3),
        'influence_matrix': torch.softmax(torch.randn(1, 3, 3).masked_fill(
            torch.eye(3, dtype=torch.bool).unsqueeze(0), float('-inf')
        ), dim=-1),
    }
    target = torch.ones(1, 2, 3, 3)

    losses = influence_loss(prediction, target, sparsity_weight=0.01)

    assert losses.reconstruction.item() > 0.0
    assert losses.sparsity.item() >= 0.0
    torch.testing.assert_close(losses.total, losses.reconstruction + losses.sparsity)


def test_influence_loss_zero_sparsity_weight_gives_zero_sparsity() -> None:
    prediction = {
        'acceleration': torch.ones(1, 2, 3, 3),
        'influence_matrix': torch.softmax(torch.randn(1, 3, 3).masked_fill(
            torch.eye(3, dtype=torch.bool).unsqueeze(0), float('-inf')
        ), dim=-1),
    }
    target = torch.ones(1, 2, 3, 3)

    losses = influence_loss(prediction, target, sparsity_weight=0.0)

    torch.testing.assert_close(losses.sparsity, torch.tensor(0.0))


def test_train_influence_model_runs_on_tiny_trajectory(fixture_path: Path) -> None:
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

    assert result.num_samples >= 1
    assert result.last_loss >= 0.0
    assert result.train_samples >= 1


def test_non_finite_loss_raises(fixture_path, monkeypatch) -> None:
    import allostery.pipeline.influence_train as mod

    def fake_loss(prediction, target_acceleration, sparsity_weight):
        bad = prediction['acceleration'].sum() * float('inf')
        return mod.InfluenceLossBreakdown(reconstruction=bad, sparsity=bad * 0.0)

    monkeypatch.setattr(mod, 'influence_loss', fake_loss)
    with pytest.raises(ValueError, match='non-finite'):
        train_influence_model(
            pdb_path=fixture_path / 'tiny_trajectory.pdb',
            window_size=3, stride=1, time_step=1.0,
            hidden_dim=8, num_encoder_layers=1, dropout=0.0,
            epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
            validation_fraction=0.0, patience=0, seed=0, device='cpu', batch_size=1,
        )


def test_train_influence_model_saves_checkpoint(fixture_path: Path, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / 'influence.pt'

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
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()

    from allostery.io.checkpoint import load_checkpoint
    ckpt = load_checkpoint(checkpoint_path)
    assert ckpt.model_family == 'influence'
    assert ckpt.residue_layers == 1


def test_mixed_precision_on_cpu_warns_and_runs(fixture_path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = train_influence_model(
            pdb_path=fixture_path / 'tiny_trajectory.pdb',
            window_size=3, stride=1, time_step=1.0,
            hidden_dim=8, num_encoder_layers=1, dropout=0.0,
            epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
            validation_fraction=0.0, patience=0, seed=0, device='cpu',
            batch_size=1, mixed_precision=True,
        )
    assert result.num_samples > 0
    assert any('mixed_precision' in str(w.message) for w in caught)


def test_invalid_lr_scheduler_rejected(fixture_path) -> None:
    with pytest.raises(ValueError, match='lr_scheduler'):
        train_influence_model(
            pdb_path=fixture_path / 'tiny_trajectory.pdb',
            window_size=3, stride=1, time_step=1.0,
            hidden_dim=8, num_encoder_layers=1, dropout=0.0,
            epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
            validation_fraction=0.0, patience=0, seed=0, device='cpu',
            batch_size=1, lr_scheduler='bogus',
        )


def test_plateau_scheduler_runs_with_validation(fixture_path) -> None:
    result = train_influence_model(
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3, stride=1, time_step=1.0,
        hidden_dim=8, num_encoder_layers=1, dropout=0.0,
        epochs=2, learning_rate=1e-3, sparsity_weight=0.0,
        validation_fraction=0.5, patience=0, seed=0, device='cpu',
        batch_size=1, lr_scheduler='plateau',
    )
    assert result.num_samples > 0
