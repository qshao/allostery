from __future__ import annotations

from pathlib import Path

import pytest


def test_influence_training_prints_epoch_lines_when_verbose(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("epoch 1/2")
    assert "train=" in lines[0]
    assert lines[1].startswith("epoch 2/2")


def test_influence_training_is_silent_when_verbose_false(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_influence_training_prints_val_loss_and_best_marker(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    # No validation split here, so no "val=" in output
    assert "val=" not in captured.out


def test_cri_training_prints_epoch_lines_when_verbose(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.cri_train import train_cri_model

    train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("epoch 1/2")
    assert "train=" in lines[0]


def test_cri_training_is_silent_when_verbose_false(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.cri_train import train_cri_model

    train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
