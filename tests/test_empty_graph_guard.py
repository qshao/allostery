from __future__ import annotations

from pathlib import Path

import pytest

from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.interpret import run_interpretation


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    path.write_text(header + "1,0.9,0,A,1,GLY,1,A,2,GLY\n", encoding="utf-8")


def test_analyze_empty_graph_raises(tmp_path: Path) -> None:
    scores = tmp_path / "s.csv"
    _write_scores(scores)
    with pytest.raises(ValueError, match="top-k"):
        run_network_analysis(scores, top_k=0)


def test_interpret_empty_graph_raises(tmp_path: Path) -> None:
    scores = tmp_path / "s.csv"
    _write_scores(scores)
    with pytest.raises(ValueError, match="top-k"):
        run_interpretation(scores, out_json=tmp_path / "o.json", out_md=tmp_path / "o.md", top_k=0)
