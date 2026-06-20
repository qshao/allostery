from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


def _run_config_text(fixture_path: Path, model_path: Path, scores_path: Path) -> str:
    return "\n".join([
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 1",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 2",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {model_path}",
        f"  score_csv_path: {scores_path}",
    ])


def test_run_command_writes_progress_to_stderr(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    cfg = tmp_path / "config.yaml"
    model = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    cfg.write_text(_run_config_text(fixture_path, model, scores))

    ret = main([str(cfg)])
    assert ret == 0
    captured = capsys.readouterr()
    # Progress goes to stderr, not stdout
    assert "epoch" in captured.err or "Training complete" in captured.err


def test_quiet_flag_suppresses_training_progress(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    cfg = tmp_path / "config.yaml"
    model = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    cfg.write_text(_run_config_text(fixture_path, model, scores))

    ret = main(["--quiet", "run", str(cfg)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "epoch" not in captured.err
    assert "Training complete" not in captured.err


def test_json_flag_suppresses_training_progress(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    cfg = tmp_path / "config.yaml"
    model = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    cfg.write_text(_run_config_text(fixture_path, model, scores))

    ret = main(["--json", "run", str(cfg)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "epoch" not in captured.err
    assert "Training complete" not in captured.err
