from __future__ import annotations

from pathlib import Path

import pytest

from allostery.pipeline.pymol_export import write_pymol_script


def test_write_pymol_script_creates_file(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/path/to/protein.pdb"),
        node_labels=["A:1 GLY", "A:2 ALA"],
        centrality={0: 0.8, 1: 0.2},
        top_pairs=[("A:1 GLY", "A:2 ALA", 0.9)],
    )
    assert pml.exists()


def test_write_pymol_script_loads_pdb(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    pdb = Path("/my/structure.pdb")
    write_pymol_script(
        pml_path=pml,
        pdb_path=pdb,
        node_labels=["A:1 GLY"],
        centrality={0: 1.0},
        top_pairs=[],
    )
    content = pml.read_text()
    assert f"load {pdb.resolve()}" in content


def test_write_pymol_script_contains_alter_and_spectrum(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY", "A:2 ALA"],
        centrality={0: 1.0, 1: 0.5},
        top_pairs=[],
    )
    content = pml.read_text()
    assert "alter chain A and resi 1 and name CA" in content
    assert "spectrum b, white_red" in content


def test_write_pymol_script_contains_pair_distance(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY", "A:3 SER"],
        centrality={0: 0.9, 1: 0.1},
        top_pairs=[("A:1 GLY", "A:3 SER", 0.9)],
    )
    content = pml.read_text()
    assert "distance pair_1" in content
    assert "color yellow, pair_*" in content


def test_write_pymol_script_includes_path_edges(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY", "A:2 ALA", "A:3 SER"],
        centrality={0: 1.0, 1: 0.5, 2: 0.0},
        top_pairs=[("A:1 GLY", "A:3 SER", 0.7)],
        path_edges=[("A:1 GLY", "A:2 ALA"), ("A:2 ALA", "A:3 SER")],
    )
    content = pml.read_text()
    assert "distance path_1" in content
    assert "distance path_2" in content
    assert "color cyan, path_*" in content


def test_write_pymol_script_no_path_edges_omits_path_lines(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY"],
        centrality={0: 1.0},
        top_pairs=[],
        path_edges=None,
    )
    content = pml.read_text()
    assert "path_" not in content
    assert "cyan" not in content
