from __future__ import annotations

import csv
from pathlib import Path

import pytest

from allostery.cli import main
from allostery.pipeline.analyze import run_network_analysis


def _write_scores(path: Path) -> None:
    fieldnames = [
        "rank", "score",
        "residue_i_index", "residue_i_chain", "residue_i_number", "residue_i_name",
        "residue_j_index", "residue_j_chain", "residue_j_number", "residue_j_name",
    ]
    rows = [
        {"rank": 1, "score": "0.9", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "GLY",
         "residue_j_index": 1, "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA"},
        {"rank": 2, "score": "0.8", "residue_i_index": 1, "residue_i_chain": "A",
         "residue_i_number": "2", "residue_i_name": "ALA",
         "residue_j_index": 2, "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "SER"},
        {"rank": 3, "score": "0.1", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "GLY",
         "residue_j_index": 2, "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "SER"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_report_includes_threshold_line(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    report = run_network_analysis(scores, top_k=3)
    assert "Suggested threshold" in report


def test_analyze_report_includes_histogram(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    report = run_network_analysis(scores, top_k=3)
    assert "Score Distribution" in report


def test_cli_analyze_out_pml_without_pdb_exits_1(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    pml = tmp_path / "out.pml"
    ret = main(["analyze", str(scores), "--out-pml", str(pml)])
    assert ret == 1


def test_cli_analyze_out_pml_with_pdb_creates_file(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    pml = tmp_path / "out.pml"
    fake_pdb = tmp_path / "protein.pdb"
    fake_pdb.write_text("ATOM record placeholder\n")
    ret = main(["analyze", str(scores), "--top-k", "3", "--out-pml", str(pml), "--pdb", str(fake_pdb)])
    assert ret == 0
    assert pml.exists()
    content = pml.read_text()
    assert "load" in content
    assert "spectrum b, white_red" in content


def test_cli_analyze_out_pml_in_artifacts(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    pml = tmp_path / "out.pml"
    fake_pdb = tmp_path / "protein.pdb"
    fake_pdb.write_text("placeholder\n")
    ret = main(["--quiet", "analyze", str(scores), "--out-pml", str(pml), "--pdb", str(fake_pdb)])
    assert ret == 0
    captured = capsys.readouterr()
    # --quiet mode prints artifact paths to stdout
    assert str(pml) in captured.out


def test_threshold_line_uses_top_k_count_not_total(tmp_path: Path) -> None:
    """Threshold rank is computed on top-k pairs, so the line reports top_k not total."""
    fieldnames = [
        "rank", "score",
        "residue_i_index", "residue_i_chain", "residue_i_number", "residue_i_name",
        "residue_j_index", "residue_j_chain", "residue_j_number", "residue_j_name",
    ]
    # 6 pairs total; top_k=3 — threshold line must say "of 3" not "of 6"
    rows = [
        {"rank": 1, "score": "0.9", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "MET",
         "residue_j_index": 1, "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "GLY"},
        {"rank": 2, "score": "0.8", "residue_i_index": 1, "residue_i_chain": "A",
         "residue_i_number": "2", "residue_i_name": "GLY",
         "residue_j_index": 2, "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "ALA"},
        {"rank": 3, "score": "0.7", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "MET",
         "residue_j_index": 2, "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "ALA"},
        {"rank": 4, "score": "0.001", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "MET",
         "residue_j_index": 3, "residue_j_chain": "A", "residue_j_number": "4", "residue_j_name": "SER"},
        {"rank": 5, "score": "0.001", "residue_i_index": 1, "residue_i_chain": "A",
         "residue_i_number": "2", "residue_i_name": "GLY",
         "residue_j_index": 3, "residue_j_chain": "A", "residue_j_number": "4", "residue_j_name": "SER"},
        {"rank": 6, "score": "0.001", "residue_i_index": 2, "residue_i_chain": "A",
         "residue_i_number": "3", "residue_i_name": "ALA",
         "residue_j_index": 3, "residue_j_chain": "A", "residue_j_number": "4", "residue_j_name": "SER"},
    ]
    scores_path = tmp_path / "scores.csv"
    with open(scores_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = run_network_analysis(scores_path, top_k=3)

    first_line = report.splitlines()[0]
    assert "of 3 scored pairs" in first_line
    assert "of 6" not in first_line
