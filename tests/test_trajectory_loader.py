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


def test_mdtraj_chain_id_none_single_chain_becomes_A(tmp_path: Path) -> None:
    import numpy as np
    from unittest.mock import MagicMock
    from allostery.io.trajectory import _load_via_mdtraj

    mock_chain = MagicMock()
    mock_chain.chain_id = None
    mock_chain.index = 0

    mock_residue = MagicMock()
    mock_residue.chain = mock_chain
    mock_residue.resSeq = 1
    mock_residue.name = "MET"

    mock_atom = MagicMock()
    mock_atom.residue = mock_residue

    mock_topology = MagicMock()
    mock_topology.select.return_value = [0]
    mock_topology.atoms = [mock_atom]

    mock_ca_traj = MagicMock()
    mock_ca_traj.topology = mock_topology
    mock_ca_traj.xyz = np.zeros((1, 1, 3), dtype=float)

    mock_traj = MagicMock()
    mock_traj.topology = mock_topology
    mock_traj.atom_slice.return_value = mock_ca_traj

    import mdtraj as _mdt_mod
    real_load = _mdt_mod.load
    _mdt_mod.load = lambda *a, **kw: mock_traj

    try:
        result = _load_via_mdtraj(tmp_path / "traj.trr", tmp_path / "top.gro")
    finally:
        _mdt_mod.load = real_load

    assert result.residues[0].chain_id == "A"


def test_mdtraj_chain_id_none_second_chain_becomes_B(tmp_path: Path) -> None:
    import numpy as np
    from unittest.mock import MagicMock
    from allostery.io.trajectory import _load_via_mdtraj

    def _make_mock_atom(chain_index: int, resseq: int) -> MagicMock:
        mock_chain = MagicMock()
        mock_chain.chain_id = None
        mock_chain.index = chain_index
        mock_residue = MagicMock()
        mock_residue.chain = mock_chain
        mock_residue.resSeq = resseq
        mock_residue.name = "ALA"
        mock_atom = MagicMock()
        mock_atom.residue = mock_residue
        return mock_atom

    mock_topology = MagicMock()
    mock_topology.select.return_value = [0, 1]
    mock_topology.atoms = [_make_mock_atom(0, 1), _make_mock_atom(1, 1)]

    mock_ca_traj = MagicMock()
    mock_ca_traj.topology = mock_topology
    mock_ca_traj.xyz = np.zeros((1, 2, 3), dtype=float)

    mock_traj = MagicMock()
    mock_traj.topology = mock_topology
    mock_traj.atom_slice.return_value = mock_ca_traj

    import mdtraj as _mdt_mod
    real_load = _mdt_mod.load
    _mdt_mod.load = lambda *a, **kw: mock_traj

    try:
        result = _load_via_mdtraj(tmp_path / "traj.trr", tmp_path / "top.gro")
    finally:
        _mdt_mod.load = real_load

    assert result.residues[0].chain_id == "A"
    assert result.residues[1].chain_id == "B"


def test_mdtraj_chain_id_set_is_preserved(tmp_path: Path) -> None:
    import numpy as np
    from unittest.mock import MagicMock
    from allostery.io.trajectory import _load_via_mdtraj

    mock_chain = MagicMock()
    mock_chain.chain_id = "C"
    mock_chain.index = 2

    mock_residue = MagicMock()
    mock_residue.chain = mock_chain
    mock_residue.resSeq = 10
    mock_residue.name = "GLY"

    mock_atom = MagicMock()
    mock_atom.residue = mock_residue

    mock_topology = MagicMock()
    mock_topology.select.return_value = [0]
    mock_topology.atoms = [mock_atom]

    mock_ca_traj = MagicMock()
    mock_ca_traj.topology = mock_topology
    mock_ca_traj.xyz = np.zeros((1, 1, 3), dtype=float)

    mock_traj = MagicMock()
    mock_traj.topology = mock_topology
    mock_traj.atom_slice.return_value = mock_ca_traj

    import mdtraj as _mdt_mod
    real_load = _mdt_mod.load
    _mdt_mod.load = lambda *a, **kw: mock_traj

    try:
        result = _load_via_mdtraj(tmp_path / "traj.trr", tmp_path / "top.gro")
    finally:
        _mdt_mod.load = real_load

    assert result.residues[0].chain_id == "C"
