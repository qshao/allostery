from __future__ import annotations

import json
from pathlib import Path

from allostery.cli import main


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = ["1,0.9,0,A,1,GLY,1,A,2,GLY", "2,0.8,1,A,2,GLY,2,A,3,GLY", "3,0.7,2,A,3,GLY,3,A,4,GLY"]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_analyze_missing_file_is_clean_user_error(tmp_path: Path, capsys) -> None:
    code = main(["analyze", str(tmp_path / "nope.csv")])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert captured.err.strip() != ""


def test_debug_flag_reraises(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(Exception):
        main(["--debug", "analyze", str(tmp_path / "nope.csv")])


def test_interpret_json_mode_emits_object(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    code = main(["--json", "interpret", str(scores),
                 "--out-json", str(tmp_path / "o.json"), "--out-md", str(tmp_path / "o.md")])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["command"] == "interpret"
    assert payload["status"] == "ok"
    assert str(tmp_path / "o.json") in payload["artifacts"]


def test_interpret_quiet_mode_emits_only_artifacts(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    code = main(["--quiet", "interpret", str(scores),
                 "--out-json", str(tmp_path / "o.json"), "--out-md", str(tmp_path / "o.md")])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines() == [str(tmp_path / "o.json"), str(tmp_path / "o.md")]


def test_json_and_quiet_are_mutually_exclusive(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(SystemExit) as exc:
        main(["--json", "--quiet", "analyze", str(tmp_path / "x.csv")])
    assert exc.value.code == 2
