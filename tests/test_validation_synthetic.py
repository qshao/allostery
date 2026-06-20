from __future__ import annotations

from pathlib import Path

import numpy as np

from allostery.io.trajectory import load_trajectory
from allostery.validation.synthetic import generate_planted_system


def test_generates_readable_pdb_and_truth_matrix(tmp_path: Path) -> None:
    pdb = tmp_path / "planted.pdb"
    system = generate_planted_system(
        pdb, n_residues=12, n_couplings=4, frames=40, seed=1,
    )
    assert system.pdb_path == pdb
    assert pdb.exists()

    trajectory = load_trajectory(pdb)
    assert trajectory.coordinates.shape == (40, 12, 3)

    matrix = system.coupling_matrix
    assert matrix.shape == (12, 12)
    assert matrix.dtype == bool
    assert np.array_equal(matrix, matrix.T)              # symmetric
    assert not matrix.diagonal().any()                   # zero diagonal
    assert int(np.triu(matrix).sum()) == 4               # exactly n_couplings edges


def test_planted_pairs_respect_min_separation(tmp_path: Path) -> None:
    system = generate_planted_system(
        tmp_path / "p.pdb", n_residues=16, n_couplings=6, frames=20, seed=2,
        min_sequence_separation=2,
    )
    rows, cols = np.where(np.triu(system.coupling_matrix))
    assert np.all((cols - rows) >= 2)


def test_is_deterministic(tmp_path: Path) -> None:
    a = generate_planted_system(tmp_path / "a.pdb", n_residues=10, n_couplings=3, frames=16, seed=7)
    b = generate_planted_system(tmp_path / "b.pdb", n_residues=10, n_couplings=3, frames=16, seed=7)
    assert np.array_equal(a.coupling_matrix, b.coupling_matrix)
    assert (tmp_path / "a.pdb").read_text() == (tmp_path / "b.pdb").read_text()


def test_rejects_too_many_couplings(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError, match="n_couplings"):
        generate_planted_system(tmp_path / "x.pdb", n_residues=5, n_couplings=999, frames=10, seed=0)
