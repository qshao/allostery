from pathlib import Path
import tempfile
import unittest

import numpy as np

from allostery.io.pdb import ResidueRecord, Trajectory, load_multimodel_pdb


def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures"


def test_load_multimodel_pdb_extracts_ca_coordinates(
    fixture_dir: Path | None = None,
) -> None:
    fixture_dir = fixture_dir or fixture_path()
    trajectory = load_multimodel_pdb(fixture_dir / "tiny_trajectory.pdb")

    assert isinstance(trajectory, Trajectory)
    assert trajectory.coordinates.shape == (3, 3, 3)
    assert trajectory.coordinates.dtype == np.float32
    assert trajectory.residues == (
        ResidueRecord(index=0, chain_id="A", residue_number=1, name="GLY"),
        ResidueRecord(index=1, chain_id="A", residue_number=2, name="ALA"),
        ResidueRecord(index=2, chain_id="A", residue_number=3, name="SER"),
    )
    np.testing.assert_allclose(
        trajectory.coordinates[0],
        np.array(
            [
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [13.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        ),
    )


def test_load_multimodel_pdb_rejects_missing_model(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdb"
    path.write_text(
        "ATOM      1  CA  GLY A   1      10.000  10.000  10.000  1.00 20.00           C\n",
        encoding="utf-8",
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "MODEL"):
        load_multimodel_pdb(path)


def test_load_multimodel_pdb_rejects_residue_order_change(tmp_path: Path) -> None:
    path = tmp_path / "bad_order.pdb"
    path.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  CA  GLY A   1      10.000  10.000  10.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2      11.000  10.000  10.000  1.00 20.00           C",
                "ENDMDL",
                "MODEL        2",
                "ATOM      1  CA  ALA A   2      11.100  10.000  10.000  1.00 20.00           C",
                "ATOM      2  CA  GLY A   1      10.100  10.000  10.000  1.00 20.00           C",
                "ENDMDL",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "ordering"):
        load_multimodel_pdb(path)


def test_load_multimodel_pdb_rejects_duplicate_residues(tmp_path: Path) -> None:
    path = tmp_path / "bad_duplicate.pdb"
    path.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  CA  GLY A   1      10.000  10.000  10.000  1.00 20.00           C",
                "ATOM      2  CA  GLY A   1      10.100  10.100  10.100  1.00 20.00           C",
                "ENDMDL",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate residue key"):
        load_multimodel_pdb(path)


def test_load_multimodel_pdb_rejects_insertion_codes(tmp_path: Path) -> None:
    path = tmp_path / "insertion_code.pdb"
    path.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  CA  GLY A   1A     10.000  10.000  10.000  1.00 20.00           C",
                "ENDMDL",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "insertion codes"):
        load_multimodel_pdb(path)


def test_load_multimodel_pdb_ignores_hetatm_ca_records(tmp_path: Path) -> None:
    path = tmp_path / "hetatm_ca.pdb"
    path.write_text(
        "\n".join(
            [
                "MODEL        1",
                "HETATM    1  CA  CA  A   1      10.000  10.000  10.000  1.00 20.00           CA",
                "ENDMDL",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "MODEL block did not contain any CA atoms"
    ):
        load_multimodel_pdb(path)


def _run_with_tmpdir(test_func) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_func(Path(temp_dir))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(
        unittest.FunctionTestCase(
            lambda: test_load_multimodel_pdb_extracts_ca_coordinates(fixture_path())
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            lambda: _run_with_tmpdir(test_load_multimodel_pdb_rejects_missing_model)
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            lambda: _run_with_tmpdir(test_load_multimodel_pdb_rejects_residue_order_change)
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            lambda: _run_with_tmpdir(test_load_multimodel_pdb_rejects_duplicate_residues)
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            lambda: _run_with_tmpdir(test_load_multimodel_pdb_rejects_insertion_codes)
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            lambda: _run_with_tmpdir(test_load_multimodel_pdb_ignores_hetatm_ca_records)
        )
    )
    return suite
