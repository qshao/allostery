from __future__ import annotations

import json
from pathlib import Path

from allostery.cli import main


def test_validate_runs_and_prints_table(capsys) -> None:
    code = main([
        "validate", "--scorers", "dccm,null",
        "--n-residues", "12", "--couplings", "4", "--frames", "48", "--seeds", "1",
    ])
    captured = capsys.readouterr()
    assert code == 0
    assert "best scorer" in captured.out
    assert "dccm" in captured.out


def test_validate_json_mode_is_parseable(tmp_path: Path, capsys) -> None:
    out_json = tmp_path / "report.json"
    code = main([
        "--json", "validate", "--scorers", "dccm,contact,null",
        "--n-residues", "12", "--couplings", "4", "--frames", "48", "--seeds", "1",
        "--out-json", str(out_json),
    ])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["command"] == "validate"
    assert payload["data"]["best_scorer"] in {"dccm", "contact", "null"}
    assert str(out_json) in payload["artifacts"]
    assert out_json.exists()                      # report written to disk
    on_disk = json.loads(out_json.read_text())
    assert len(on_disk["scorers"]) == 3


def test_validate_unknown_scorer_exits_1(capsys) -> None:
    code = main(["validate", "--scorers", "bogus", "--seeds", "1"])
    captured = capsys.readouterr()
    assert code == 1
    assert "unknown scorer" in captured.err
    assert "Traceback" not in captured.err
