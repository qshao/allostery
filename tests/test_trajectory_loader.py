from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_load_trajectory_dispatches_pdb_by_extension(fixture_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    result = load_trajectory(fixture_path / "tiny_trajectory.pdb")

    assert result.coordinates.ndim == 3
    assert result.coordinates.shape[2] == 3
    assert len(result.residues) == result.coordinates.shape[1]


def test_load_trajectory_accepts_string_path(fixture_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    result = load_trajectory(str(fixture_path / "tiny_trajectory.pdb"))

    assert result.coordinates.shape[0] > 0


def test_load_trajectory_requires_topology_for_non_pdb() -> None:
    from allostery.io.trajectory import load_trajectory

    with pytest.raises(ValueError, match="topology_path is required"):
        load_trajectory("trajectory.xtc")


def test_load_trajectory_requires_topology_for_dcd() -> None:
    from allostery.io.trajectory import load_trajectory

    with pytest.raises(ValueError, match="topology_path is required"):
        load_trajectory("trajectory.dcd")


def test_load_trajectory_raises_import_error_without_backends(tmp_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    fake_traj = tmp_path / "traj.xtc"
    fake_traj.touch()

    with patch.dict("sys.modules", {"MDAnalysis": None, "mdtraj": None}):
        with pytest.raises(ImportError, match="MDAnalysis"):
            load_trajectory(fake_traj, topology_path=tmp_path / "top.tpr")


def test_load_trajectory_error_message_mentions_both_packages(tmp_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    fake_traj = tmp_path / "traj.xtc"
    fake_traj.touch()

    with patch.dict("sys.modules", {"MDAnalysis": None, "mdtraj": None}):
        with pytest.raises(ImportError) as exc_info:
            load_trajectory(fake_traj, topology_path=tmp_path / "top.tpr")

    message = str(exc_info.value)
    assert "MDAnalysis" in message
    assert "mdtraj" in message


def test_train_influence_accepts_topology_path_kwarg(fixture_path: Path) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    # topology_path=None should work for .pdb (existing behaviour unchanged)
    result = train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
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
        device="cpu",
        batch_size=1,
        topology_path=None,
    )
    assert result.num_samples >= 1
