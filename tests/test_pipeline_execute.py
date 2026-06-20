from __future__ import annotations

from pathlib import Path

from allostery.config import load_config
from allostery.pipeline.execute import run_scoring, run_training, serialize_config


def _run_config(tmp_path: Path, fixture_path: Path) -> Path:
    checkpoint = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    text = "\n".join([
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
        f"  model_path: {checkpoint}",
        f"  score_csv_path: {scores}",
    ])
    path = tmp_path / "run.yaml"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_run_training_and_scoring(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_run_config(tmp_path, fixture_path))
    result = run_training(config)
    assert result.num_samples >= 1
    assert config.output.model_path.exists()
    count = run_scoring(config)
    assert count == 3
    assert config.output.score_csv_path.exists()


def test_serialize_config_is_json_safe(tmp_path: Path, fixture_path: Path) -> None:
    import json
    config = load_config(_run_config(tmp_path, fixture_path))
    json.dumps(serialize_config(config))  # must not raise
