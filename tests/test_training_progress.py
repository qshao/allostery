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


def test_relational_training_prints_epoch_lines_when_verbose(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.train import train_model

    train_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=1,
        horizon_size=1,
        stride=1,
        hidden_dim=8,
        residue_layers=1,
        pair_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        consistency_weight=0.0,
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


def test_influence_progress_fn_called_per_epoch(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    calls: list[tuple[int, float, float | None]] = []

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=3,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
        progress_fn=lambda e, t, v: calls.append((e, t, v)),
    )

    assert len(calls) == 3
    assert calls[0][0] == 1
    assert calls[2][0] == 3
    assert all(isinstance(c[1], float) for c in calls)
    assert all(c[2] is None for c in calls)  # no validation split


def test_cri_progress_fn_called_per_epoch(fixture_path: Path) -> None:
    from allostery.pipeline.cri_train import train_cri_model

    calls: list[int] = []

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
        progress_fn=lambda e, t, v: calls.append(e),
    )

    assert calls == [1, 2]


def test_relational_progress_fn_called_per_epoch(fixture_path: Path) -> None:
    from allostery.pipeline.train import train_model

    calls: list[int] = []

    train_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=1,
        horizon_size=1,
        stride=1,
        hidden_dim=8,
        residue_layers=1,
        pair_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        consistency_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
        progress_fn=lambda e, t, v: calls.append(e),
    )

    assert calls == [1, 2]


def test_progress_fn_takes_priority_over_verbose(fixture_path: Path, capsys) -> None:
    """When progress_fn is set, verbose=True should not print to stdout."""
    from allostery.pipeline.influence_train import train_influence_model

    calls: list[int] = []

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
        progress_fn=lambda e, t, v: calls.append(e),
    )

    captured = capsys.readouterr()
    assert captured.out == ""  # verbose print suppressed by progress_fn
    assert len(calls) == 2
