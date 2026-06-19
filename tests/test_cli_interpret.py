from __future__ import annotations

from pathlib import Path

from allostery.cli import main


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = [
        "1,0.9,0,A,1,GLY,1,A,2,GLY",
        "2,0.8,1,A,2,GLY,2,A,3,GLY",
        "3,0.7,2,A,3,GLY,3,A,4,GLY",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_cli_interpret_writes_outputs(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    code = main([
        "interpret", str(scores),
        "--out-json", str(out_json), "--out-md", str(out_md),
    ])
    assert code == 0
    assert out_json.exists() and out_md.exists()
    assert "interpret" in capsys.readouterr().out
