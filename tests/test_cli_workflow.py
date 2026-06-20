from __future__ import annotations

import json
from pathlib import Path

from allostery.cli import main


def _cfg(tmp_path: Path, fixture_path: Path, extra: list[str]) -> Path:
    lines = [
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {tmp_path / 'model.pt'}",
        f"  score_csv_path: {tmp_path / 'scores.csv'}",
    ] + extra
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_workflow_command_runs_end_to_end(tmp_path: Path, fixture_path: Path, capsys) -> None:
    config = _cfg(tmp_path, fixture_path, ["analyze:", "  top_k: 3", "interpret:", "  llm: none"])
    code = main(["workflow", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "scores.network.txt").exists()
    assert (tmp_path / "scores.interpret.json").exists()
    assert "workflow complete" in captured.out


def test_workflow_command_json_mode(tmp_path: Path, fixture_path: Path, capsys) -> None:
    config = _cfg(tmp_path, fixture_path, ["interpret:", "  llm: none"])
    code = main(["--json", "workflow", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["command"] == "workflow"
    assert payload["data"]["stages"] == ["train", "score", "interpret"]


def test_workflow_backend_failure_exits_3(tmp_path: Path, fixture_path: Path) -> None:
    from unittest.mock import patch
    config = _cfg(tmp_path, fixture_path, ["interpret:", "  llm: ollama"])
    with patch("allostery.pipeline.workflow.run_interpretation", side_effect=ImportError("backend unavailable")):
        code = main(["workflow", str(config)])
    assert code == 3
