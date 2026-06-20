from __future__ import annotations

from pathlib import Path

from allostery.pipeline.analyze import run_network_analysis


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = ["1,0.9,0,A,1,GLY,1,A,2,GLY", "2,0.8,1,A,2,GLY,2,A,3,GLY"]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_analyze_writes_report_file(tmp_path: Path) -> None:
    scores = tmp_path / "s.csv"
    _write_scores(scores)
    out = tmp_path / "nested" / "network.txt"
    report = run_network_analysis(scores, top_k=5, out_path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == report
    assert "Allosteric Network" in report
