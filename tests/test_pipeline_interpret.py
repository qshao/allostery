from __future__ import annotations

import json
from pathlib import Path

from allostery.pipeline.interpret import run_interpretation


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = [
        "1,0.9,0,A,1,GLY,1,A,2,GLY",
        "2,0.8,1,A,2,GLY,2,A,3,GLY",
        "3,0.7,2,A,3,GLY,3,A,4,GLY",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_run_interpretation_without_llm(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    report = run_interpretation(scores, out_json=out_json, out_md=out_md)
    assert out_json.exists() and out_md.exists()
    assert report["structural_context"] is False
    assert "interpretation" not in report["candidates"]["hubs"][0]


def test_run_interpretation_with_injected_backend(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)

    class _FakeBackend:
        def generate_json(self, system, user, schema):
            return {
                "summary": "s", "mechanism_hypothesis": "m",
                "key_residues": [], "confidence": "low",
                "parametric": False, "caveats": "c",
            }

    report = run_interpretation(
        scores, out_json=tmp_path / "o.json", out_md=tmp_path / "o.md",
        llm="ollama", backend=_FakeBackend(),
    )
    assert report["candidates"]["hubs"][0]["interpretation"]["confidence"] == "low"
    loaded = json.loads((tmp_path / "o.json").read_text())
    assert loaded["candidates"]["hubs"][0]["interpretation"]["confidence"] == "low"
