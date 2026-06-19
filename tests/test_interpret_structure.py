from __future__ import annotations

from pathlib import Path

from allostery.io.pdb import load_multimodel_pdb
from allostery.interpret.structure import (
    ResidueStructuralFeatures, StructuralContext, compute_structural_context,
)


def test_structural_context_from_fixture(fixture_path: Path) -> None:
    trajectory = load_multimodel_pdb(fixture_path / "tiny_trajectory.pdb")
    context = compute_structural_context(trajectory)
    assert isinstance(context, StructuralContext)
    n = trajectory.coordinates.shape[1]
    assert len(context.per_residue) == n
    assert all(isinstance(f, ResidueStructuralFeatures) for f in context.per_residue.values())
    assert all(f.rmsf >= 0.0 for f in context.per_residue.values())
    assert all(f.contact_number >= 0 for f in context.per_residue.values())
    first = trajectory.residues[0]
    label = f"{first.chain_id}:{first.residue_number} {first.name}"
    assert context.label_to_index[label] == 0


def test_geometry_returns_radius_of_gyration(fixture_path: Path) -> None:
    trajectory = load_multimodel_pdb(fixture_path / "tiny_trajectory.pdb")
    context = compute_structural_context(trajectory)
    labels = list(context.label_to_index.keys())[:2]
    geom = context.geometry(labels)
    assert geom["n_resolved"] == 2
    assert geom["radius_of_gyration"] >= 0.0
